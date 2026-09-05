"""
yawc and dyn: the two encodings that exist because capture frames differ per corpus.

yawc removes the content-referenced heading and nothing else; dyn removes every static
cue and is invariant to any rigid transform of the capture frame. Both properties are
the point, so both are pinned here.
"""
import math

import pytest
import torch

from input_encoding import apply_encoding


pytestmark = pytest.mark.unit


def _yaw_about_y(angle: float) -> torch.Tensor:
    half = angle / 2.0
    return torch.tensor([0.0, math.sin(half), 0.0, math.cos(half)])


def _quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return torch.tensor([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _rotate_vectors(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate (3, T) vectors by one quaternion (4,), x,y,z,w."""
    u, w = q[:3].view(3, 1).expand_as(v), q[3]
    c = torch.cross(u, v, dim=0)
    return v + 2.0 * (w * c + torch.cross(u, c, dim=0))


def _random_head_window(count=24, seed=0):
    """A plausible head: small pitch/roll wobble about a heading, standing at 1.6 m."""
    g = torch.Generator().manual_seed(seed)
    base = _yaw_about_y(0.7)
    quaternion = base.view(4, 1) + torch.randn(4, count, generator=g) * 0.05
    quaternion = quaternion / quaternion.norm(dim=0, keepdim=True)
    position = torch.tensor([0.3, 1.6, -0.2]).view(3, 1) + torch.randn(3, count, generator=g) * 0.02
    return torch.cat([quaternion, position]).unsqueeze(0)


def _globally_rotated(window: torch.Tensor, yaw: float, shift=(2.0, 0.0, -1.0)) -> torch.Tensor:
    """The same recording captured in a frame rotated about Y by yaw and shifted."""
    r = _yaw_about_y(yaw)
    quaternion = torch.stack(
        [_quaternion_multiply(r, window[0, :4, t]) for t in range(window.shape[2])], dim=1)
    position = _rotate_vectors(r, window[0, 4:7]) + torch.tensor(shift).view(3, 1)
    return torch.cat([quaternion, position]).unsqueeze(0)


def _mean_facing(window: torch.Tensor) -> torch.Tensor:
    q = window[0, :4]
    forward = torch.zeros(3, 1)
    forward[2] = 1.0
    out = torch.stack([_rotate_vectors(q[:, t], forward)[:, 0] for t in range(q.shape[1])], dim=1)
    return out.mean(dim=1)


# --- yawc ---------------------------------------------------------------------

def test_yawc_turns_the_mean_facing_direction_to_plus_z():
    """The point of yawc: a window that faced 40 degrees left now faces +Z."""
    facing = _mean_facing(apply_encoding(_random_head_window(), "yawc"))
    assert abs(facing[0]) < 1e-4 and facing[2] > 0.9


def test_yawc_is_invariant_to_the_capture_frames_yaw():
    """
    A corpus whose yaw reference is +X and one whose reference is +Z must encode the
    same recording identically - the rotation per-channel standardisation cannot undo.
    """
    base = _random_head_window()
    turned = _globally_rotated(base, yaw=1.3, shift=(0.0, 0.0, 0.0))
    a, b = apply_encoding(base, "yawc"), apply_encoding(turned, "yawc")
    # Orientation and the within-window displacement are heading-relative and match.
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-4)
    residual = lambda x: x[:, 4:7] - x[:, 4:7].mean(dim=2, keepdim=True)  # noqa: E731
    assert torch.allclose(residual(a), residual(b), atol=1e-4)
    # The absolute position is deliberately kept, so where the room's origin sits still
    # shows in x and z. Height never moves under a rotation about the up axis.
    assert torch.allclose(a[:, 5], b[:, 5], atol=1e-4)


def test_yawc_keeps_height_and_the_absolute_position():
    """Only the heading goes; the static cue the model actually uses must survive."""
    base = _random_head_window()
    encoded = apply_encoding(base, "yawc")
    assert torch.allclose(encoded[0, 4:7].mean(dim=1), base[0, 4:7].mean(dim=1), atol=1e-5)
    assert torch.allclose(encoded[0, 5], base[0, 5], atol=1e-5), "height is untouched frame by frame"


def test_yawc_keeps_the_quaternion_unit_norm():
    encoded = apply_encoding(_random_head_window(), "yawc")
    assert torch.allclose(encoded[0, :4].norm(dim=0), torch.ones(encoded.shape[2]), atol=1e-4)


# --- dyn ----------------------------------------------------------------------

def test_dyn_removes_every_static_cue():
    """Mean position zero, mean orientation the identity: nothing anthropometric left."""
    encoded = apply_encoding(_random_head_window(), "dyn")
    assert torch.allclose(encoded[0, 4:7].mean(dim=1), torch.zeros(3), atol=1e-5)
    mean_q = encoded[0, :4].mean(dim=1)
    mean_q = mean_q / mean_q.norm()
    assert torch.allclose(mean_q, torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=2e-2)
    assert torch.allclose(encoded[0, :4].norm(dim=0), torch.ones(encoded.shape[2]), atol=1e-4)


def test_dyn_is_invariant_to_any_rigid_transform_of_the_capture_frame():
    """Rotated AND translated capture frame, same movement, same encoding."""
    base = _random_head_window()
    moved = _globally_rotated(base, yaw=-2.1, shift=(3.0, 0.4, -5.0))
    assert torch.allclose(apply_encoding(base, "dyn"), apply_encoding(moved, "dyn"), atol=1e-4)


def test_dyn_still_carries_the_movement():
    """Invariance must not come from throwing everything away."""
    still = _random_head_window()
    still[0, 4:7] = still[0, 4:7].mean(dim=1, keepdim=True)          # no translation at all
    moving = still.clone()
    moving[0, 6] += torch.linspace(-0.1, 0.1, moving.shape[2])         # sways forward and back
    assert not torch.allclose(apply_encoding(still, "dyn"), apply_encoding(moving, "dyn"), atol=1e-3)


@pytest.mark.parametrize("encoding", ["yawc", "dyn"])
def test_position_only_windows_are_accepted(encoding):
    window = torch.randn(2, 3, 12)
    out = apply_encoding(window.clone(), encoding)
    assert out.shape == window.shape
    if encoding == "dyn":
        assert torch.allclose(out.mean(dim=2), torch.zeros(2, 3), atol=1e-6)
    else:
        assert torch.equal(out, window), "with no orientation there is no heading to remove"


def test_dyn_residual_mean_is_zero_at_far_coordinates():
    """
    Rounding residue scales with the absolute coordinate: at 30 m a float32 centring
    leaves ~1e-5 m in the window mean, a faint copy of the location the encoding removes.
    Centred in float64 the residue is gone even at SLAM-scale coordinates.
    """
    g = torch.Generator().manual_seed(1)
    window = _random_head_window(count=100, seed=3)
    window[0, 4:7] += torch.tensor([27.0, 0.3, -31.0]).view(3, 1)       # Nymeria-scale coordinates
    encoded = apply_encoding(window, "dyn")
    residual_mean = encoded[0, 4:7].mean(dim=1).abs().max()
    assert residual_mean < 1e-6, f"residual window-mean {residual_mean:.2e} m; float32 centring leaves ~1e-5"
