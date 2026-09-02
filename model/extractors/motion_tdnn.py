"""
Motion-TDNN: a derivative channel bank + dilated TDNN + statistics pooling.

Motivation
----------
Measured on the default dataset (43 train / 5 held-out users, 5s windows), the mean
head position of a window separates unseen users on its own at ~0.69-0.71 pair
accuracy, while every dynamics feature tested (angular velocity, spectra, path
length, channel correlations) sits at 0.49-0.53. The previously trained models land
at ~0.72, i.e. barely above what the single strongest raw statistic already gives.

The likely reason is scale, not absence of signal. Position spans ~1 unit and
quaternion components ~1, while the per-step differences that carry motion are
~1e-2. Fed raw into a recurrent stack, the subtle channels are swamped, so the model
converges onto position and stops.

This extractor addresses that directly, borrowing the shape of the x-vector speaker
verification architecture - which solves a structurally identical problem: turn a
variable-length signal into one fixed embedding where same-identity pairs are close.

    1. Channel bank    explicit kinematics (pose, angular/linear velocity and
                       acceleration) computed in closed form rather than left for
                       the network to discover.
    2. BatchNorm       per-channel standardisation over the dataset, so a 1e-2
                       angular velocity gets the same gradient footing as a 1e0
                       position. This is the piece the pipeline never had.
    3. Dilated TDNN    multi-scale temporal context at low cost, no recurrence.
    4. Statistics pool mean and standard deviation over time. Identity lives in the
                       *distribution* of motion within a window, not its order, and
                       pooling statistics is what makes x-vectors work.

Angular velocity is derived from the relative rotation between consecutive frames
(q_t^-1 . q_t+1) in the body frame, so it is invariant to absolute head orientation -
which is driven by scene content shared across users and is therefore a confound.
Absolute position is deliberately kept: it carries height and seated posture, which
the measurements above show is the single most person-specific cue available.

Measured results (43 train / 5 held-out users, seed 67, 20 epochs)
-----------------------------------------------------------------
The idea did not beat the existing baselines. Recorded here so the next attempt does
not repeat it:

    5s @ 20Hz, emb 128, 512 pairs/user      0.667 held-out   (0.930 train)
    + 4x pairs, emb 64, dropout 0.3, 197k params
                                            0.663 held-out   (0.930 train)
    + 195 extra identities from 5 datasets  0.576 held-out   (0.936 train)

for reference: bilstm 0.716, mean-position-only linear probe 0.691, chance 0.500.

Two things this rules out, both with evidence:

1. **Capacity is not the limit.** Cutting the model to a third of its size, halving
   the embedding, tripling dropout and quadrupling the pairs moved the result by
   0.4 points. Training accuracy still reached 0.93. With only 43 training
   identities, the pair task is solvable by memorising who is who at any size.

2. **Naively pooling datasets makes it actively worse.** Coordinate frames do not
   line up (mean head height ranges from 0.00003 in Panonut360 to 2.89 in NJIT), and
   because positives are always same-user and therefore same-dataset, 79% of negative
   pairs become cross-dataset. Raw mean-position distance alone then answers
   "different user?" for 71% of training pairs. The model learns "same dataset?",
   hits 0.86 train accuracy in the first epoch, and transfers nothing to a held-out
   set where every pair is within one dataset.

The obvious next step is per-dataset standardisation of the input channels (or
restricting negative pairs to within-dataset), which would remove the shortcut and
let the extra 195 identities actually count. That is untested.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from feature_extractor import FeatureExtractor, register


def _quaternion_angular_velocity(q: torch.Tensor) -> torch.Tensor:
    """
    Body-frame angular velocity from a (batch, 4, T) quaternion track in (x, y, z, w).

    Uses the small-angle approximation of the relative rotation between consecutive
    frames: omega ~= 2 * vec(q_t^-1 . q_t+1). The sign of the result is normalised for
    the quaternion double cover (q and -q are the same rotation), without which the
    derived velocity flips randomly and carries no usable signal.
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


def _pad_front(x: torch.Tensor, width: int) -> torch.Tensor:
    """Replicate the first frame so a differenced channel keeps the original length."""
    if width <= 0:
        return x
    return torch.cat([x[:, :, :1].expand(-1, -1, width), x], dim=2)


class _TdnnBlock(nn.Module):
    """Dilated 1D convolution -> BatchNorm -> ReLU -> dropout, length preserving."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


@register("motion_tdnn")
class MotionTdnn(FeatureExtractor):
    """
    Args:
        hidden: Width of each TDNN layer.
        layers: Number of dilated TDNN layers (dilations 1, 2, 3, 4, ... ).
        kernel: Temporal kernel size of each TDNN layer.
        dropout: Dropout after each TDNN layer.
        use_derivatives: Include the velocity/acceleration channel bank. False leaves
            only pose, which isolates how much the kinematic channels contribute.
        use_pose: Include absolute pose channels (quaternion and position).
        pool_max: Append max-over-time to the mean/std statistics pooling.
    """

    def __init__(
        self,
        seq_len: int,
        num_channels: int = 7,
        embedding_dim: int = 128,
        hidden: int = 192,
        layers: int = 4,
        kernel: int = 5,
        dropout: float = 0.1,
        use_derivatives: bool = True,
        use_pose: bool = True,
        pool_max: bool = False,
    ):
        super().__init__(
            seq_len=seq_len, num_channels=num_channels, embedding_dim=embedding_dim,
            hidden=hidden, layers=layers, kernel=kernel, dropout=dropout,
            use_derivatives=use_derivatives, use_pose=use_pose, pool_max=pool_max,
        )
        if not (use_pose or use_derivatives):
            raise ValueError("At least one of use_pose or use_derivatives must be True.")

        self.use_derivatives = use_derivatives
        self.use_pose = use_pose
        self.pool_max = pool_max

        bank_channels = (7 if use_pose else 0) + (17 if use_derivatives else 0)
        self.input_norm = nn.BatchNorm1d(bank_channels)

        blocks = []
        in_channels = bank_channels
        for index in range(layers):
            blocks.append(_TdnnBlock(in_channels, hidden, kernel, dilation=index + 1, dropout=dropout))
            in_channels = hidden
        self.tdnn = nn.Sequential(*blocks)

        statistics = 3 if pool_max else 2
        self.embed = nn.Linear(hidden * statistics, embedding_dim)

    def _channel_bank(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (batch, 7, T) -> stacked kinematic channels, all at length T."""
        quaternion, position = x[:, :4], x[:, 4:]
        channels = []

        if self.use_pose:
            normalised_q = quaternion / (quaternion.norm(dim=1, keepdim=True) + 1e-8)
            channels += [normalised_q, position]

        if self.use_derivatives:
            omega = _quaternion_angular_velocity(quaternion)          # (B, 3, T-1)
            alpha = omega[:, :, 1:] - omega[:, :, :-1]                # (B, 3, T-2)
            velocity = position[:, :, 1:] - position[:, :, :-1]       # (B, 3, T-1)
            accel = velocity[:, :, 1:] - velocity[:, :, :-1]          # (B, 3, T-2)
            # Position relative to the window mean: posture/sway with the absolute
            # offset removed, which the pose channels already carry.
            centred = position - position.mean(dim=2, keepdim=True)

            omega, velocity = _pad_front(omega, 1), _pad_front(velocity, 1)
            alpha, accel = _pad_front(alpha, 2), _pad_front(accel, 2)
            channels += [
                omega, alpha, velocity, accel, centred,
                omega.norm(dim=1, keepdim=True),
                velocity.norm(dim=1, keepdim=True),
            ]

        return torch.cat(channels, dim=1)

    def forward(self, x):
        bank = self.input_norm(self._channel_bank(x))
        features = self.tdnn(bank)

        # Statistics pooling: identity is in the distribution of motion over the
        # window, not in the order the frames arrive.
        pooled = [features.mean(dim=2), features.std(dim=2)]
        if self.pool_max:
            pooled.append(features.amax(dim=2))

        return self.embed(torch.cat(pooled, dim=1))

    @classmethod
    def search_space(cls):
        return {
            "hidden": [128, 192, 256],
            "layers": [3, 4, 5],
            "kernel": [3, 5],
            "dropout": [0.0, 0.1, 0.2],
            "use_derivatives": [True, False],
            "pool_max": [True, False],
        }
