"""
Per-dataset input standardisation.

The datasets in this corpus do not share a coordinate frame or a scale. Measured
mean head position, and per-session position range, across the processed data:

    Panonut360          mean y 0.00003     range [1.96, 1.42, 1.98]
    VR_User_Behavior    mean y 1.14        range [0.33, 0.16, 0.30]
    ViewGauss           mean y 1.58        range [0.12, 0.05, 0.06]
    NJIT_6DOF           mean y 1.57        range [5.13, 0.16, 3.54]

That is a ~40x spread in scale and a ~2.9 unit spread in offset. Because a positive
Siamese pair is always the same user and therefore always the same dataset, while
79% of negative pairs are cross-dataset once several are pooled, the offsets alone
answer "different user?" for 71% of training pairs. A model trained on the pooled
data learns to identify the *dataset*, reaches 0.86 training accuracy in one epoch,
and transfers nothing to a held-out split where every pair is within one dataset.

Standardising each dataset's channels to zero mean and unit variance removes that
shortcut while preserving what matters: the *relative* differences between users
within a dataset, which is where the identity signal lives. It also puts position
and quaternion channels on a common scale, which is what the raw pipeline never did.

Statistics are fitted on training windows only and carried in the checkpoint, so
evaluation applies the training-time transform rather than re-deriving it from the
held-out data.
"""

from __future__ import annotations

import torch


MODES = ("none", "per_dataset", "global")
GLOBAL_KEY = "__global__"
_CHUNK = 4096


class ChannelNormalizer:
    """
    Per-channel affine standardisation, keyed by dataset of origin.

    Args:
        mode: "per_dataset" fits one set of statistics per dataset (removes the
            cross-dataset shortcut), "global" fits one set over everything (fixes
            channel scale only), "none" is a no-op.
    """

    def __init__(self, mode: str = "per_dataset"):
        if mode not in MODES:
            raise ValueError(f"normalize must be one of {MODES}, got {mode!r}.")
        self.mode = mode
        self.statistics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @staticmethod
    def _channel_statistics(samples: torch.Tensor, indices: torch.Tensor):
        """Mean/std per channel over (windows, timesteps), accumulated in chunks."""
        channels = samples.shape[1]
        total = torch.zeros(channels, dtype=torch.float64)
        total_sq = torch.zeros(channels, dtype=torch.float64)
        count = 0
        for start in range(0, indices.numel(), _CHUNK):
            block = samples[indices[start:start + _CHUNK]].double()
            total += block.sum(dim=(0, 2))
            total_sq += block.pow(2).sum(dim=(0, 2))
            count += block.shape[0] * block.shape[2]
        if count == 0:
            return torch.zeros(channels), torch.ones(channels)
        mean = total / count
        variance = (total_sq / count - mean.pow(2)).clamp_min(0.0)
        # A dead channel (all one value) must not blow up; leave it at zero.
        std = variance.sqrt().clamp_min(1e-6)
        return mean.float(), std.float()

    def fit(self, sample_index) -> "ChannelNormalizer":
        """Fit statistics from a sample index. Call this on training data only."""
        self.statistics = {}
        if not self.enabled or sample_index.sample_count == 0:
            return self

        if self.mode == "global":
            all_indices = torch.arange(sample_index.sample_count)
            self.statistics[GLOBAL_KEY] = self._channel_statistics(sample_index.samples, all_indices)
            return self

        for dataset_id, name in enumerate(sample_index.dataset_names):
            indices = torch.nonzero(sample_index.window_dataset_ids == dataset_id, as_tuple=False).view(-1)
            if indices.numel():
                self.statistics[name] = self._channel_statistics(sample_index.samples, indices)
        return self

    def transform(self, sample_index) -> "ChannelNormalizer":
        """Standardise a sample index in place using the fitted statistics."""
        if not self.enabled or sample_index.sample_count == 0:
            return self

        if self.mode == "global":
            groups = [(GLOBAL_KEY, torch.arange(sample_index.sample_count))]
        else:
            groups = []
            for dataset_id, name in enumerate(sample_index.dataset_names):
                indices = torch.nonzero(sample_index.window_dataset_ids == dataset_id, as_tuple=False).view(-1)
                if indices.numel():
                    groups.append((name, indices))

        for name, indices in groups:
            statistics = self.statistics.get(name)
            if statistics is None:
                # A dataset never seen during fitting - e.g. evaluating a trained model
                # on a new corpus. Deriving statistics from the target data is
                # unsupervised (no labels are used) and is the standard way to bring a
                # new coordinate frame into the training frame, but it is not the
                # training-time transform, so say so.
                print(f"WARNING: no training statistics for dataset '{name}'; "
                      "fitting them from the evaluation data instead.")
                statistics = self._channel_statistics(sample_index.samples, indices)
                self.statistics[name] = statistics

            mean, std = statistics
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)
            for start in range(0, indices.numel(), _CHUNK):
                block = indices[start:start + _CHUNK]
                sample_index.samples[block] = (sample_index.samples[block] - mean) / std
        return self

    def fit_transform(self, sample_index) -> "ChannelNormalizer":
        return self.fit(sample_index).transform(sample_index)

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "statistics": {
                name: {"mean": mean.tolist(), "std": std.tolist()}
                for name, (mean, std) in self.statistics.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict | None) -> "ChannelNormalizer":
        """Rebuild from a checkpoint. A missing state means an unnormalised model."""
        if not state:
            return cls(mode="none")
        normalizer = cls(mode=state.get("mode", "none"))
        for name, entry in (state.get("statistics") or {}).items():
            normalizer.statistics[name] = (
                torch.tensor(entry["mean"], dtype=torch.float32),
                torch.tensor(entry["std"], dtype=torch.float32),
            )
        return normalizer

    def describe(self) -> str:
        if not self.enabled:
            return "normalization: none"
        return f"normalization: {self.mode} over {len(self.statistics)} dataset(s)"
