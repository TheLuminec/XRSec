"""
Input encodings, applied in the data layer

NOTE: this module is deliberately NOT called `encodings`. That name belongs to a
stdlib package the interpreter imports during startup, so a local module of that name
is shadowed and never loaded. so every extractor sees the same transform.

Why this is a config axis rather than an extractor detail
--------------------------------------------------------
This repo's strongest negative result is "extractor architecture is worth ~0, spread
under one point across three backbones over ten folds". But those three backbones do
not share an input encoding: `bilstm` and `paper_gnn_bilstm` consume raw channels
while `motion_tdnn` derives kinematics internally. So architecture and encoding varied
together, and the experiment cannot separate them.

The literature runs the same comparison the other way - architecture fixed, encoding
varied - and reports a clear ordering: scene-relative worst, body-relative better,
body-relative velocity better still, body-relative acceleration best. Neither result
answers the other. Promoting encoding to a data-layer transform lets one sweep over
{extractors} x {encodings} answer both.

It also bears directly on the anthropometry finding. `center_position` removes the
window's mean position but leaves absolute orientation, so the "movement only" arm was
measured in roughly the weakest encoding available. A movement-only result at `bra`
is a different claim from a movement-only result at raw-centred.

Head-only approximation
-----------------------
The published body-relative encodings derive a body reference from head AND both
controllers. This corpus is head-only, so:

  raw   scene-relative, unchanged - what the pipeline has always done
  br    pose relative to the window's FIRST FRAME: orientation as q0^-1 . q_t and
        position rotated into q0's frame. Removes where the person is and which way
        they face, without needing to know which axis is vertical - the up-axis
        differs across this corpus, so anything yaw-based would be guessing.
  brv   frame-to-frame delta: the relative rotation q_{t-1}^-1 . q_t and the linear
        difference. Already invariant to absolute pose, so it needs no reference
        frame.
  bra   the same differencing applied twice.

Two more exist because of the cross-dataset audit (docs/GENERALISATION_PROPOSAL.md):

  yawc  gravity-preserving YAW CANONICALISATION. Each window is rotated about the
        world up axis (Y) so that its mean facing direction is +Z. Height, pitch, roll
        and the absolute position all survive; only the content-referenced heading is
        removed. Measured yaw references differ per corpus (+Z for BOXRR, ViewGauss and
        NJIT, +X for Head_and_Gaze, -X for VR_User_Behavior, none for alyx), which is
        a rotation that per-channel standardisation cannot undo. The horizontal
        position is rotated about the window's own mean position, so the static
        position cue is untouched and only the within-window displacement is expressed
        in the head's heading frame. Assumes Y-up, which every dataset here is after
        conversion except NJIT's broken quaternion.
  dyn   DYNAMICS ONLY: pose relative to the window's MEAN pose. Orientation becomes
        q_mean^-1 . q_t and position becomes the residual (p_t - p_mean) rotated into
        the mean-pose body frame. Every static cue - mean position, i.e. height and
        seat, and mean orientation, i.e. posture - is removed by construction, and the
        result is invariant to any rigid transform of the capture frame, so it needs no
        up axis. This is what `center_position` was meant to be: centring left the
        absolute quaternion in, and mean orientation alone scores 0.54-0.81 AUC of
        static posture, so the "behavioural" arm was never only behaviour. `br` is the
        same idea referenced to the noisy first frame instead of the mean.

Channel count is preserved (7 stays 7, 3 stays 3) so every extractor's contract holds:
the rotation block of a velocity encoding is the *delta rotation*, which is still a
unit quaternion, not a componentwise difference of two quaternions.
"""

from __future__ import annotations

import torch

ENCODINGS = ("raw", "br", "brv", "bra", "yawc", "dyn")


def _quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (batch, 4, T) quaternions in x, y, z, w order."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=1)


def _quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], dim=1)


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate (batch, 3, T) vectors by (batch, 4, T) unit quaternions."""
    u = q[:, :3]
    w = q[:, 3:4]
    cross_uv = torch.cross(u, v, dim=1)
    return v + 2.0 * (w * cross_uv + torch.cross(u, cross_uv, dim=1))


def _pad_front(x: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return x
    return torch.cat([x[:, :, :1].expand(-1, -1, width), x], dim=2)


def _normalise(q: torch.Tensor) -> torch.Tensor:
    return q / (q.norm(dim=1, keepdim=True) + 1e-8)


def _split(samples: torch.Tensor):
    """(quaternion or None, position) for either channel set."""
    if samples.shape[1] >= 7:
        return _normalise(samples[:, :4]), samples[:, 4:7]
    return None, samples[:, :3]


def _delta_once(quaternion, position):
    """Frame-to-frame change: relative rotation, and linear difference."""
    linear = position[:, :, 1:] - position[:, :, :-1]
    if quaternion is None:
        return None, _pad_front(linear, 1)

    relative = _quaternion_multiply(
        _quaternion_conjugate(quaternion[:, :, :-1]), quaternion[:, :, 1:]
    )
    # Quaternion double cover: q and -q are the same rotation, so without fixing the
    # sign the delta flips arbitrarily between frames and carries no usable signal.
    sign = torch.where(relative[:, 3:4] < 0, -1.0, 1.0)
    return _pad_front(relative * sign, 1), _pad_front(linear, 1)


def _mean_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Per-window mean rotation, (batch, 4, 1).

    Frames are first put in the hemisphere of the window's first frame - q and -q are
    one rotation, and averaging across the sign flip cancels instead of averaging - then
    the components are averaged and renormalised. Exact for small spreads, which is what
    a five-second head window has.
    """
    reference = quaternion[:, :, :1]
    sign = torch.where((quaternion * reference).sum(dim=1, keepdim=True) < 0, -1.0, 1.0)
    return _normalise((quaternion * sign).mean(dim=2, keepdim=True))


def _heading_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """
    The rotation about world Y that turns the window's mean facing direction to +Z.

    The facing direction is the head's local +Z rotated into the world by each frame and
    averaged; its yaw is atan2(x, z). Rotating about +Y by theta sends (sin a, 0, cos a)
    to (sin(a + theta), 0, cos(a + theta)), so theta = -yaw. Returned as (batch, 4, 1).
    """
    forward = torch.zeros_like(quaternion[:, :3])
    forward[:, 2] = 1.0
    facing = _rotate(quaternion, forward).mean(dim=2)                  # (batch, 3)
    yaw = torch.atan2(facing[:, 0], facing[:, 2])
    half = -yaw / 2.0
    zeros = torch.zeros_like(half)
    return torch.stack([zeros, torch.sin(half), zeros, torch.cos(half)], dim=1).unsqueeze(2)


def _yaw_canonical(quaternion, position):
    """yawc: rotate the window about world up so its mean facing is +Z."""
    centre = position.mean(dim=2, keepdim=True)
    if quaternion is None:
        # Nothing says which way the head faced; the position is left as it is.
        return None, position
    heading = _heading_quaternion(quaternion).expand(-1, -1, position.shape[2])
    return (_quaternion_multiply(heading, quaternion),
            centre + _rotate(heading, position - centre))


def _dynamics_only(quaternion, position):
    """dyn: pose relative to the window's mean pose, every static cue removed."""
    residual = position - position.mean(dim=2, keepdim=True)
    if quaternion is None:
        return None, residual
    inverse = _quaternion_conjugate(_mean_quaternion(quaternion)).expand(-1, -1, position.shape[2])
    relative = _quaternion_multiply(inverse, quaternion)
    sign = torch.where(relative[:, 3:4] < 0, -1.0, 1.0)
    return relative * sign, _rotate(inverse, residual)


def apply_encoding(samples: torch.Tensor, encoding: str) -> torch.Tensor:
    """
    Transform windows in place-compatible fashion, preserving the channel count.

    Args:
        samples: (windows, channels, timesteps)
        encoding: one of ENCODINGS
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}.")
    if encoding == "raw" or samples.numel() == 0:
        return samples

    quaternion, position = _split(samples)

    if encoding in ("yawc", "dyn"):
        quaternion, position = (_yaw_canonical if encoding == "yawc" else _dynamics_only)(
            quaternion, position)
        if quaternion is None:
            return position
        return torch.cat([quaternion, position], dim=1)

    if encoding == "br":
        reference = quaternion[:, :, :1] if quaternion is not None else None
        centred = position - position[:, :, :1]
        if reference is None:
            return centred
        inverse = _quaternion_conjugate(reference).expand(-1, -1, position.shape[2])
        return torch.cat([
            _quaternion_multiply(inverse, quaternion),
            _rotate(inverse, centred),
        ], dim=1)

    quaternion, position = _delta_once(quaternion, position)
    if encoding == "bra":
        quaternion, position = _delta_once(quaternion, position)

    if quaternion is None:
        return position
    return torch.cat([quaternion, position], dim=1)
