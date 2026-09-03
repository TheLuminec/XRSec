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

Channel count is preserved (7 stays 7, 3 stays 3) so every extractor's contract holds:
the rotation block of a velocity encoding is the *delta rotation*, which is still a
unit quaternion, not a componentwise difference of two quaternions.
"""

from __future__ import annotations

import torch

ENCODINGS = ("raw", "br", "brv", "bra")


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
