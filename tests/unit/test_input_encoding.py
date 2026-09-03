import math

import pytest
import torch

from input_encoding import ENCODINGS, apply_encoding


pytestmark = pytest.mark.unit


def _yaw_quaternion(angle: float, count: int) -> torch.Tensor:
    """Rotation about the z axis, as (4, T) in x, y, z, w order."""
    half = angle / 2.0
    q = torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)])
    return q.view(4, 1).expand(4, count).clone()


def _window(count=12, position=None, quaternion=None):
    position = position if position is not None else torch.randn(3, count)
    quaternion = quaternion if quaternion is not None else _yaw_quaternion(0.0, count)
    return torch.cat([quaternion, position]).unsqueeze(0)


def _quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return torch.tensor([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


# --- shape and contract -------------------------------------------------------

@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("channels", [7, 3])
def test_channel_count_and_length_are_preserved(encoding, channels):
    """Extractors declare num_channels; an encoding that changed it would break them."""
    window = torch.randn(4, channels, 16)
    out = apply_encoding(window, encoding)
    assert out.shape == (4, channels, 16)


def test_raw_is_the_identity():
    window = _window()
    assert torch.equal(apply_encoding(window.clone(), "raw"), window)


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="encoding must be one of"):
        apply_encoding(torch.randn(1, 7, 8), "polar")


# --- the invariances that are the point --------------------------------------

def test_br_is_invariant_to_where_the_person_is():
    """
    Body-relative means the same movement scores the same wherever it happened. This
    is the property the anthropometric cue lives in, so it has to actually hold.
    """
    base = _window()
    moved = base.clone()
    moved[:, 4:7, :] += torch.tensor([3.0, -1.5, 0.7]).view(1, 3, 1)

    assert torch.allclose(apply_encoding(base, "br"), apply_encoding(moved, "br"), atol=1e-5)


def test_br_is_invariant_to_which_way_the_person_faces():
    """A constant yaw offset is scene-relative information, and br must remove it."""
    count = 12
    position = torch.randn(3, count)

    facing_forward = _window(count, position=position, quaternion=_yaw_quaternion(0.0, count))
    turned = _window(count, position=position, quaternion=_yaw_quaternion(1.1, count))

    encoded_forward = apply_encoding(facing_forward, "br")
    encoded_turned = apply_encoding(turned, "br")

    # Orientation relative to the first frame is identical either way.
    assert torch.allclose(encoded_forward[:, :4], encoded_turned[:, :4], atol=1e-5)


def test_raw_is_not_invariant_to_translation():
    """Contrast: raw carries the absolute position, which is why it is separable."""
    base = _window()
    moved = base.clone()
    moved[:, 4:7, :] += 2.0
    assert not torch.allclose(apply_encoding(base, "raw"), apply_encoding(moved, "raw"))


@pytest.mark.parametrize("encoding", ["brv", "bra"])
def test_velocity_encodings_are_invariant_to_a_constant_offset(encoding):
    base = _window()
    moved = base.clone()
    moved[:, 4:7, :] += torch.tensor([5.0, 2.0, -3.0]).view(1, 3, 1)
    assert torch.allclose(apply_encoding(base, encoding), apply_encoding(moved, encoding), atol=1e-5)


def test_brv_recovers_a_known_constant_velocity():
    count = 10
    position = torch.arange(count, dtype=torch.float32).view(1, count).expand(3, count) * 0.25
    encoded = apply_encoding(_window(count, position=position.clone()), "brv")
    # Every step advances 0.25 per axis; the first frame is padded from the second.
    assert torch.allclose(encoded[0, 4:7, 1:], torch.full((3, count - 1), 0.25), atol=1e-5)


def test_bra_is_zero_for_constant_velocity():
    """Acceleration of a constant-velocity track is zero - a direct correctness check."""
    count = 10
    position = torch.arange(count, dtype=torch.float32).view(1, count).expand(3, count) * 0.25
    encoded = apply_encoding(_window(count, position=position.clone()), "bra")
    assert torch.allclose(encoded[0, 4:7, 2:], torch.zeros(3, count - 2), atol=1e-5)


# --- quaternion handling ------------------------------------------------------

def test_delta_rotations_stay_unit_quaternions():
    """
    The rotation block of a velocity encoding is a delta ROTATION, not a componentwise
    difference of two quaternions, so it must still have unit norm.
    """
    torch.manual_seed(0)
    quaternion = torch.randn(4, 20)
    quaternion = quaternion / quaternion.norm(dim=0, keepdim=True)
    encoded = apply_encoding(_window(20, quaternion=quaternion), "brv")
    assert torch.allclose(encoded[0, :4].norm(dim=0), torch.ones(20), atol=1e-4)


def test_delta_rotation_sign_is_fixed_for_the_double_cover():
    """
    q and -q are the same rotation. Without normalising the sign the delta flips
    arbitrarily between frames and carries no usable signal - the same bug that made
    angular velocity useless before it was fixed in the extractor.
    """
    count = 16
    quaternion = _yaw_quaternion(0.3, count).clone()
    quaternion[:, ::2] *= -1.0          # same rotations, alternating representation

    encoded = apply_encoding(_window(count, quaternion=quaternion), "brv")
    assert (encoded[0, 3, :] >= 0).all(), "delta rotation scalar part changes sign"


def test_br_composes_the_relative_rotation_correctly():
    count = 4
    start, later = _yaw_quaternion(0.4, 1)[:, 0], _yaw_quaternion(1.0, 1)[:, 0]
    quaternion = torch.stack([start, start, later, later], dim=1)

    encoded = apply_encoding(_window(count, quaternion=quaternion), "br")

    expected = _quaternion_multiply(
        torch.tensor([-start[0], -start[1], -start[2], start[3]]), later
    )
    assert torch.allclose(encoded[0, :4, 2], expected, atol=1e-5)
    # Relative to itself, the first frame is the identity rotation.
    assert torch.allclose(encoded[0, :4, 0], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5)


def test_position_only_channel_set_is_handled():
    window = torch.randn(2, 3, 12)
    for encoding in ENCODINGS:
        out = apply_encoding(window.clone(), encoding)
        assert out.shape == window.shape
    centred = apply_encoding(window.clone(), "br")
    assert torch.allclose(centred[:, :, 0], torch.zeros(2, 3), atol=1e-6)
