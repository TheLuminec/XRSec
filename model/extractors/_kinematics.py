"""
Shared kinematic helpers for extractors that derive motion channels from raw pose.

Not an extractor module: the package ``__init__`` skips names starting with ``_``,
so nothing here is imported into the registry. It exists so the quaternion algebra
has one home rather than a copy inside every extractor that needs it.
"""

from __future__ import annotations

import torch


def quaternion_angular_velocity(q: torch.Tensor) -> torch.Tensor:
    """
    Body-frame angular velocity from a (batch, 4, T) quaternion track in (x, y, z, w).

    Uses the small-angle approximation of the relative rotation between consecutive
    frames: omega ~= 2 * vec(q_t^-1 . q_t+1). The sign of the result is normalised for
    the quaternion double cover (q and -q are the same rotation), without which the
    derived velocity flips randomly and carries no usable signal.

    Returns (batch, 3, T-1). Body-frame, so it is invariant to absolute head
    orientation - which is driven by scene content shared across users and is
    therefore a confound rather than an identity cue.
    """
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    previous, current = q[:, :, :-1], q[:, :, 1:]

    # conj(previous) = (-x, -y, -z, w)
    ax, ay, az, aw = -previous[:, 0], -previous[:, 1], -previous[:, 2], previous[:, 3]
    bx, by, bz, bw = current[:, 0], current[:, 1], current[:, 2], current[:, 3]

    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw

    sign = torch.where(rw < 0, -1.0, 1.0)
    return 2.0 * torch.stack([rx * sign, ry * sign, rz * sign], dim=1)


def pad_front(x: torch.Tensor, width: int) -> torch.Tensor:
    """Replicate the first frame so a differenced channel keeps the original length."""
    if width <= 0:
        return x
    return torch.cat([x[:, :, :1].expand(-1, -1, width), x], dim=2)


def split_pose(x: torch.Tensor, num_channels: int, owner: str = "extractor"):
    """
    Split a window into (quaternion, position), or (None, position) with no orientation.

    ``channels=full`` supplies 7 channels (qx, qy, qz, qw, Hx, Hy, Hz);
    ``channels=position`` supplies 3 (Hx, Hy, Hz) and roughly doubles the usable
    corpus, because much of it records head position but no orientation.

    Callers branch on ``quaternion is None`` rather than on a channel count, so a
    layout change lands here rather than in every extractor's forward pass.
    """
    if num_channels == 7:
        return x[:, :4], x[:, 4:]
    if num_channels == 3:
        return None, x
    raise ValueError(
        f"{owner} supports num_channels=7 (channels=full) or 3 (channels=position), "
        f"got num_channels={num_channels}."
    )


def derived_channels(quaternion, position, include_norms: bool = True):
    """
    Closed-form kinematics at the input's own length: velocity, acceleration, and
    window-centred position, plus angular velocity and acceleration when orientation
    is present.

    Differenced channels are front-padded so every entry keeps length T, and the
    orientation-derived ones are simply absent for position-only data rather than
    zero-filled - a constant channel would spend width and tell the model nothing.
    """
    channels = []

    if quaternion is not None:
        omega = quaternion_angular_velocity(quaternion)               # (B, 3, T-1)
        alpha = omega[:, :, 1:] - omega[:, :, :-1]                    # (B, 3, T-2)
        omega, alpha = pad_front(omega, 1), pad_front(alpha, 2)
        channels += [omega, alpha]

    velocity = position[:, :, 1:] - position[:, :, :-1]               # (B, 3, T-1)
    accel = velocity[:, :, 1:] - velocity[:, :, :-1]                  # (B, 3, T-2)
    centred = position - position.mean(dim=2, keepdim=True)
    velocity, accel = pad_front(velocity, 1), pad_front(accel, 2)
    channels += [velocity, accel, centred]

    if include_norms:
        if quaternion is not None:
            channels.append(channels[0].norm(dim=1, keepdim=True))    # |omega|
        channels.append(velocity.norm(dim=1, keepdim=True))

    return channels
