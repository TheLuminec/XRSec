"""
PyTorch Dataset for XR user biometric identification.

Wraps the existing data pipeline (UserProfile -> Sampler) into standard PyTorch
Dataset objects for use with DataLoader. Sampled windows are cached to disk per
user directory (see sample_cache.py), so repeat runs skip CSV parsing entirely.

Each sample is a (7, seq_len) tensor representing one window of data:
    - 7 channels: qx, qy, qz, qw, Hx, Hy, Hz
    - seq_len = sample_time * sample_rate time samples
    - Time column is stripped before training
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import sample_cache
from normalization import ChannelNormalizer
from user_profile import UserProfile, channel_count


def _seed_value(seed: int | None, offset: int = 0) -> int:
    base_seed = 0 if seed is None else int(seed)
    return int(np.random.SeedSequence([base_seed, int(offset)]).generate_state(1, dtype=np.uint32)[0])


def _empty_pair_manifest() -> dict[str, torch.Tensor]:
    return {
        "x1_indices": torch.empty(0, dtype=torch.long),
        "x2_indices": torch.empty(0, dtype=torch.long),
        "labels": torch.empty(0, dtype=torch.float32),
        "anchor_user_ids": torch.empty(0, dtype=torch.long),
    }


def make_pair_manifest(x1_indices, x2_indices, labels, anchor_user_ids) -> dict[str, torch.Tensor]:
    """
    Create a normalized manifest for siamese pairs.
    """
    manifest = {
        "x1_indices": torch.as_tensor(x1_indices, dtype=torch.long),
        "x2_indices": torch.as_tensor(x2_indices, dtype=torch.long),
        "labels": torch.as_tensor(labels, dtype=torch.float32),
        "anchor_user_ids": torch.as_tensor(anchor_user_ids, dtype=torch.long),
    }

    lengths = {tensor.shape[0] for tensor in manifest.values()}
    if len(lengths) > 1:
        raise ValueError("Pair manifest fields must all have the same length.")
    return manifest


def concat_pair_manifests(manifests: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Concatenate multiple pair manifests while preserving field order.
    """
    valid_manifests = [manifest for manifest in manifests if manifest["labels"].numel() > 0]
    if not valid_manifests:
        return _empty_pair_manifest()

    return {
        key: torch.cat([manifest[key] for manifest in valid_manifests], dim=0)
        for key in valid_manifests[0]
    }


def _pairs_per_label(pair_count: int, match_ratio: float) -> tuple[int, int]:
    match_ratio = float(min(max(match_ratio, 0.0), 1.0))
    positive_count = int(round(pair_count * match_ratio))
    positive_count = min(max(positive_count, 0), pair_count)
    negative_count = pair_count - positive_count
    return positive_count, negative_count


class SampleDataset:
    """
    Dataset of VR user head movement samples for biometric identification raw.

    Args:
        data_dir: Path to processed_data/users/ directory
        exclude_users: Optional path(s) to exclude
        swap_data: Whether to swap what is included and excluded
    """

    def __init__(
        self,
        data_dir: str | list[str],
        sample_time: int = 1,
        sample_rate: int = 10,
        exclude_users: str | list[str] | None = None,
        swap_data: bool = False,
        channels: str = "full",
        keep_users: set[str] | list[str] | None = None,
    ):
        self.dataset = []
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.num_users = 0
        self.swap_data = swap_data
        self.channels = channels
        self.num_channels = channel_count(channels)

        if exclude_users is None:
            exclude_users = []
        elif isinstance(exclude_users, str):
            exclude_users = [exclude_users]

        # Applied independently of exclude/swap so a subsampled corpus stays
        # subsampled in every split, not just the training one.
        self.keep_users = set(keep_users) if keep_users else None

        data_dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
        # Dataset of origin per user, so normalisation can be fitted per dataset.
        self.dataset_names = [self._dataset_name(directory) for directory in data_dirs]
        self.user_dataset_ids: list[int] = []
        self.session_ids: list[torch.Tensor] = []
        self.skipped_files: dict[str, int] = {}

        # Users are filtered before loading rather than after: excluded users cost
        # nothing to skip, which matters for leave-users-out sweeps.
        self.sample_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        for dataset_id, directory in enumerate(data_dirs):
            for user_dir in self._iter_user_dirs(directory):
                if self.keep_users is not None and user_dir not in self.keep_users:
                    continue
                if self.swap_data:
                    if user_dir not in exclude_users:
                        continue
                else:
                    if user_dir in exclude_users:
                        continue

                self.num_users += 1
                self.user_dataset_ids.append(dataset_id)
                user_samples, user_sessions = self._load_user_samples(user_dir)
                self.sample_count += int(user_samples.shape[0])
                self.dataset.append(user_samples)
                self.session_ids.append(user_sessions)

        cache_note = ""
        if sample_cache.cache_enabled():
            cache_note = f" (cache: {self.cache_hits} hit, {self.cache_misses} built)"
        print(f"Loaded {self.sample_count} samples from {self.num_users} users{cache_note}")
        if self.skipped_files:
            detail = ", ".join(f"{count} {reason}" for reason, count in sorted(self.skipped_files.items()))
            print(f"  skipped unusable files: {detail}")

    @staticmethod
    def _dataset_name(directory: str) -> str:
        """`.../<Dataset_Name>/users` -> `<Dataset_Name>`; the key normalisation uses."""
        path = Path(directory)
        return path.parent.name if path.name == "users" else path.name

    @staticmethod
    def _iter_user_dirs(directory: str):
        """
        Yield each user directory under a dataset root, in sorted order.

        Sorted traversal is what makes flat sample indices stable across machines,
        so the ordering here must not change.
        """
        for name in sorted(os.listdir(directory)):
            user_dir = os.path.join(directory, name)
            if os.path.isdir(user_dir):
                yield user_dir

    def _load_user_samples(self, user_dir: str):
        """
        Return (windows, session_ids) for one user, via cache when possible.

        session_ids is the index of the CSV each window came from, so pair generation
        can tell same-session from cross-session positives.
        """
        cache_path = None
        if sample_cache.cache_enabled():
            cache_path = sample_cache.entry_path(Path(user_dir), self.sample_time,
                                                 self.sample_rate, self.channels)
            cached = sample_cache.load(cache_path)
            if cached is not None:
                self.cache_hits += 1
                return cached

        self.cache_misses += 1
        profile = UserProfile(user_dir, self.sample_time, self.sample_rate, channels=self.channels)
        for reason, count in getattr(profile, "skipped", {}).items():
            self.skipped_files[reason] = self.skipped_files.get(reason, 0) + count

        samples = []
        sessions = []
        for session_index, sampler in enumerate(profile.data_samplers):
            if sampler.sample_count == 0:
                continue
            for sample in sampler.get_all_samples():
                # Drop the SessionTime column, then transpose to (channels, timesteps).
                samples.append(sample[:, 1:].astype(np.float32).T)
                sessions.append(session_index)

        if samples:
            user_samples = torch.tensor(np.array(samples), dtype=torch.float32)
            user_sessions = torch.tensor(sessions, dtype=torch.long)
        else:
            seq_len = self.sample_time * self.sample_rate
            user_samples = torch.empty((0, self.num_channels, seq_len), dtype=torch.float32)
            user_sessions = torch.empty(0, dtype=torch.long)

        if cache_path is not None:
            sample_cache.store(cache_path, user_samples, user_sessions)
        return user_samples, user_sessions

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return self.dataset[idx]


def position_channel_slice(num_channels: int) -> slice:
    """Where the position channels sit, for either channel set."""
    return slice(4, 7) if num_channels >= 7 else slice(0, 3)


class SampleIndex:
    """
    Stable index over flattened per-user samples.
    """

    def __init__(self, sample_dataset: SampleDataset, center_position: bool = False):
        self.sample_time = sample_dataset.sample_time
        self.sample_rate = sample_dataset.sample_rate
        self.seq_len = self.sample_time * self.sample_rate
        self.num_users = sample_dataset.num_users
        self.num_channels = getattr(sample_dataset, "num_channels", 7)
        self.channels = getattr(sample_dataset, "channels", "full")

        self.dataset_names = list(getattr(sample_dataset, "dataset_names", []))
        user_dataset_ids = getattr(sample_dataset, "user_dataset_ids", [])
        self.user_dataset_ids = list(user_dataset_ids)

        self.user_sample_indices: list[torch.Tensor] = []
        dataset_session_ids = getattr(sample_dataset, "session_ids", [])
        flat_samples = []
        flat_sessions = []
        window_dataset_ids = []
        offset = 0
        for user_index, user_samples in enumerate(sample_dataset.dataset):
            sample_count = int(user_samples.shape[0])
            if sample_count > 0:
                flat_samples.append(user_samples)
                indices = torch.arange(offset, offset + sample_count, dtype=torch.long)
                dataset_id = user_dataset_ids[user_index] if user_index < len(user_dataset_ids) else 0
                window_dataset_ids.append(torch.full((sample_count,), dataset_id, dtype=torch.long))
                if user_index < len(dataset_session_ids):
                    flat_sessions.append(dataset_session_ids[user_index])
                else:
                    flat_sessions.append(torch.zeros(sample_count, dtype=torch.long))
            else:
                indices = torch.empty(0, dtype=torch.long)
            self.user_sample_indices.append(indices)
            offset += sample_count

        if flat_samples:
            self.samples = torch.cat(flat_samples, dim=0)
            self.window_dataset_ids = torch.cat(window_dataset_ids, dim=0)
            self.window_session_ids = torch.cat(flat_sessions, dim=0)
        else:
            self.samples = torch.empty((0, self.num_channels, self.seq_len), dtype=torch.float32)
            self.window_dataset_ids = torch.empty(0, dtype=torch.long)
            self.window_session_ids = torch.empty(0, dtype=torch.long)

        self.sample_count = int(self.samples.shape[0])
        self.center_position = center_position

        if center_position and self.sample_count:
            # Remove each window's mean position, leaving only movement within the
            # window. Absolute position carries height and seated posture - the
            # strongest single identity cue measured here (0.768 AUC on unseen users)
            # but an anthropometric one. Centring isolates what is left: how the person
            # moves. If accuracy survives, the task is behavioural; if it collapses to
            # chance, this is body measurement wearing a behavioural label.
            channels = position_channel_slice(self.samples.shape[1])
            position = self.samples[:, channels, :]
            self.samples[:, channels, :] = position - position.mean(dim=2, keepdim=True)

    def __len__(self):
        return self.sample_count


def build_sample_index(
    data_dir,
    sample_time: int = 1,
    sample_rate: int = 10,
    exclude_users=None,
    swap_data: bool = False,
    channels: str = "full",
    center_position: bool = False,
    keep_users=None,
) -> SampleIndex:
    """
    Build a stable sample index using sorted user and file traversal.
    """
    return SampleIndex(
        SampleDataset(
            data_dir,
            sample_time=sample_time,
            sample_rate=sample_rate,
            exclude_users=exclude_users,
            swap_data=swap_data,
            channels=channels,
            keep_users=keep_users,
        ),
        center_position=center_position,
    )


def count_single_session_users(sample_index) -> int:
    """
    Users with fewer than two sessions, which cannot form a cross-session positive.

    `cross_session_positives` falls back to same-session pairs for these, silently as
    far as the reported number is concerned. Recording the count means the
    qualification travels with the result instead of living in a chat log. Measured on
    this corpus: NJIT_6DOF is the only affected dataset (all 18 users, one session
    each); every other dataset has a minimum of 2 sessions per user, so the pooled
    figure is 18/343 = 5.2%.
    """
    sessions = getattr(sample_index, "window_session_ids", None)
    if sessions is None or sessions.numel() == 0:
        return 0

    count = 0
    for window_indices in sample_index.user_sample_indices:
        if window_indices.numel() and torch.unique(sessions[window_indices]).numel() < 2:
            count += 1
    return count


def generate_pair_manifest(
    sample_index: SampleIndex,
    pairs_per_user: int,
    match_ratio: float = 0.5,
    seed: int | None = None,
    within_dataset_negatives: bool = False,
    cross_session_positives: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Deterministically generate siamese pairs from a stable sample index.

    Args:
        within_dataset_negatives: Draw negative partners only from users in the same
            dataset. A positive pair is always the same user and therefore always the
            same dataset, so when several datasets are pooled the negatives become
            overwhelmingly cross-dataset (79% measured on six datasets) and the task
            collapses into "same dataset?" - which is trivially separable and does not
            transfer to a held-out split drawn from one dataset. Restricting negatives
            makes the training objective match the evaluation condition. This is a
            no-op when training on a single dataset.
        cross_session_positives: Build positive pairs from two *different* recording
            sessions of the same user. Same-session pairs share headset mounting,
            seating position and the content being viewed, so a model can score well
            by matching the session rather than the person - and because held-out
            positives are same-session too, that shortcut does not show up as a
            train/test gap. Cross-session verification is the standard requirement in
            biometrics for exactly this reason. Users with only one session fall back
            to same-session pairs and are counted in the manifest metadata.
    """
    if pairs_per_user <= 0 or sample_index.num_users == 0 or sample_index.sample_count == 0:
        return _empty_pair_manifest()

    rng = np.random.default_rng(_seed_value(seed))
    positive_target, negative_target = _pairs_per_label(pairs_per_user, match_ratio)

    x1_indices = []
    x2_indices = []
    labels = []
    anchor_user_ids = []
    cross_session_users = 0
    single_session_users = 0

    user_dataset_ids = getattr(sample_index, "user_dataset_ids", []) or [0] * sample_index.num_users

    def _eligible(user_idx: int, candidate: int) -> bool:
        if candidate == user_idx or len(sample_index.user_sample_indices[candidate]) == 0:
            return False
        if within_dataset_negatives and user_dataset_ids[candidate] != user_dataset_ids[user_idx]:
            return False
        return True

    valid_negative_users = {
        user_idx: [candidate for candidate in range(sample_index.num_users) if _eligible(user_idx, candidate)]
        for user_idx in range(sample_index.num_users)
    }

    for user_idx in range(sample_index.num_users):
        user_samples = sample_index.user_sample_indices[user_idx]
        if len(user_samples) == 0:
            continue

        local_positive_target = positive_target
        local_negative_target = negative_target
        if not valid_negative_users[user_idx]:
            local_positive_target = pairs_per_user
            local_negative_target = 0

        if local_positive_target > 0:
            session_groups = None
            if cross_session_positives:
                sessions = getattr(sample_index, "window_session_ids", None)
                if sessions is not None and sessions.numel():
                    user_sessions = sessions[user_samples]
                    session_groups = [user_samples[user_sessions == s]
                                      for s in torch.unique(user_sessions)]
                    session_groups = [g for g in session_groups if g.numel() > 0]

            if session_groups and len(session_groups) > 1:
                # One window from each of two distinct sessions.
                for _ in range(local_positive_target):
                    first, second = rng.choice(len(session_groups), size=2, replace=False)
                    x1_indices.append(int(rng.choice(session_groups[int(first)].numpy())))
                    x2_indices.append(int(rng.choice(session_groups[int(second)].numpy())))
                cross_session_users += 1
            else:
                # Single-session user (or session ids unavailable): fall back.
                x1_pos = rng.choice(user_samples.numpy(), size=local_positive_target, replace=True)
                x2_pos = rng.choice(user_samples.numpy(), size=local_positive_target, replace=True)
                x1_indices.extend(x1_pos.tolist())
                x2_indices.extend(x2_pos.tolist())
                if cross_session_positives:
                    single_session_users += 1

            labels.extend([1.0] * local_positive_target)
            anchor_user_ids.extend([user_idx] * local_positive_target)

        if local_negative_target > 0:
            x1_neg = rng.choice(user_samples.numpy(), size=local_negative_target, replace=True)
            for x1_idx in x1_neg.tolist():
                negative_user = int(rng.choice(valid_negative_users[user_idx]))
                negative_samples = sample_index.user_sample_indices[negative_user]
                x2_idx = int(rng.choice(negative_samples.numpy()))
                x1_indices.append(x1_idx)
                x2_indices.append(x2_idx)
                labels.append(0.0)
                anchor_user_ids.append(user_idx)

    if cross_session_positives and single_session_users:
        print(f"  cross-session positives: {cross_session_users} users, "
              f"{single_session_users} fell back to same-session (only one session)")

    manifest = make_pair_manifest(x1_indices, x2_indices, labels, anchor_user_ids)
    if manifest["labels"].numel() == 0:
        return manifest

    permutation = torch.as_tensor(
        rng.permutation(manifest["labels"].shape[0]),
        dtype=torch.long,
    )
    return {key: value[permutation] for key, value in manifest.items()}


class PairManifestDataset(Dataset):
    """
    Dataset backed by a flat sample index plus a pair manifest.
    """

    def __init__(self, sample_index: SampleIndex, manifest: dict[str, torch.Tensor]):
        self.sample_index = sample_index
        self.samples = sample_index.samples
        self.manifest = manifest

    def __len__(self):
        return int(self.manifest["labels"].shape[0])

    def __getitem__(self, idx):
        x1_idx = int(self.manifest["x1_indices"][idx])
        x2_idx = int(self.manifest["x2_indices"][idx])
        label = self.manifest["labels"][idx].view(1)
        return (self.samples[x1_idx], self.samples[x2_idx]), label


def create_pair_dataloader(
    sample_index: SampleIndex,
    manifest: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int | None = None,
):
    """
    Create a DataLoader for a manifest-backed siamese dataset.
    """
    pin_memory = device.type == "cuda" if device else False
    generator = torch.Generator().manual_seed(_seed_value(seed))
    dataset = PairManifestDataset(sample_index, manifest)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def select_user_subset(data_dir, max_users: int | None, seed: int | None) -> list[str] | None:
    """
    Deterministically keep at most `max_users` users, stratified across datasets.

    This exists to disambiguate identity count from data diversity. Going from one
    corpus to seven changed three things at once - 48 identities to 343, one dataset
    to seven, and per-dataset normalisation from a no-op to active - so a gain cannot
    be attributed to identity count without holding the others fixed. Subsampling the
    pooled corpus back down to 48 identities, spread across the same seven datasets,
    isolates the variable.

    Allocation is proportional to each dataset's size using largest remainder, so the
    subset mirrors the corpus rather than over-representing whichever dataset happens
    to sort first. Returns None when no subsampling is needed, which keeps the
    filtering cost at zero for ordinary runs.
    """
    if not max_users or max_users <= 0:
        return None

    by_dataset: dict[str, list[str]] = {}
    for directory in ([data_dir] if isinstance(data_dir, str) else list(data_dir)):
        users = [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if os.path.isdir(os.path.join(directory, name))
        ]
        if users:
            by_dataset[directory] = users

    total = sum(len(users) for users in by_dataset.values())
    if max_users >= total:
        return None

    # Largest-remainder apportionment, so the quotas sum to exactly max_users.
    exact = {d: len(u) * max_users / total for d, u in by_dataset.items()}
    quota = {d: int(value) for d, value in exact.items()}
    remaining = max_users - sum(quota.values())
    for directory in sorted(by_dataset, key=lambda d: (-(exact[d] - quota[d]), d))[:remaining]:
        quota[directory] += 1

    rng = np.random.default_rng(_seed_value(seed, 31))
    kept: list[str] = []
    for directory in sorted(by_dataset):
        users = by_dataset[directory]
        take = min(quota[directory], len(users))
        if take:
            order = rng.permutation(len(users))[:take]
            kept.extend(users[int(i)] for i in order)
    return sorted(kept)


def select_validation_users(data_dir, exclude_users, fraction: float, seed: int | None) -> list[str]:
    """
    Deterministically pick users to hold out for epoch selection.

    These are drawn from the *training* users and are disjoint from both the training
    and the reported test group, because the task is generalisation to unseen people:
    a validation split that shares users with training would select on memorisation.

    Returns [] when fraction is 0, which restores the previous single-split behaviour.
    """
    if not fraction or fraction <= 0:
        return []

    excluded = set(exclude_users or [])
    candidates = []
    for directory in ([data_dir] if isinstance(data_dir, str) else list(data_dir)):
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isdir(path) and path not in excluded:
                candidates.append(path)

    if len(candidates) < 3:
        # Too few users to spare any: selecting on one user would be worse than not.
        return []

    count = int(round(len(candidates) * float(fraction)))
    count = min(max(count, 1), len(candidates) - 2)

    rng = np.random.default_rng(_seed_value(seed, 21))
    chosen = rng.choice(len(candidates), size=count, replace=False)
    return sorted(candidates[int(i)] for i in chosen)


def create_dataloader_from_path(
    data_dir,
    batch_size: int,
    device: torch.device,
    is_train: bool = True,
    test_dir=None,
    sample_time: int = 1,
    sample_rate: int = 10,
    samples_per_user: int = 1000,
    val_split: float = 0.2,
    num_workers: int = 0,
    exclude_users=None,
    swap_data: bool = False,
    test_on_excluded: bool = False,
    seed: int | None = None,
    normalize: str = "none",
    within_dataset_negatives: bool = False,
    channels: str = "full",
    cross_session_positives: bool = False,
    center_position: bool = False,
    max_users: int | None = None,
    normalizer: ChannelNormalizer | None = None,
    return_normalizer: bool = False,
    val_user_fraction: float = 0.0,
    return_val: bool = False,
):
    """
    Create DataLoader(s) from dataset paths.

    Args:
        data_dir: Path(s) to data. Training dataset if is_train=True, else evaluation dataset.
        batch_size: Batch size
        device: Device to use (for pin_memory)
        is_train: If True, returns (train_loader, test_loader). If False, returns test_loader.
        test_dir: Optional path to testing data for training. If None and is_train is True, data_dir is split.
        sample_time: Sample time for dataset
        sample_rate: Sample rate for dataset
        samples_per_user: Number of pairs to generate per user for the SiameseDataset
        val_split: Fraction of dataset to use for validation split if test_dir is None
        num_workers: Number of DataLoader workers
        exclude_users: User paths to exclude from data loading
        swap_data: Whether to swap what is included and excluded
        test_on_excluded: If true, uses the excluded paths for the testing dataset instead of doing a random split
        seed: Root seed for deterministic pair generation and splits
        normalize: Input standardisation mode - "none", "per_dataset" or "global".
        within_dataset_negatives: Restrict negative pairs to users from the same
            dataset (see generate_pair_manifest).
        channels: Which input channels to build windows from - "full" (quaternion +
            position) or "position". See user_profile.CHANNEL_SETS.
        cross_session_positives: Draw positive pairs from different sessions of the
            same user (see generate_pair_manifest).
        center_position: Subtract each window's mean position, leaving movement only.
            Separates behaviour from anthropometry (height, seated posture).
        max_users: Keep at most this many users, stratified across datasets, so
            identity count can be varied without changing dataset diversity.
        normalizer: A pre-fitted normalizer to apply instead of fitting a new one.
            Evaluation must pass the training-time normalizer so held-out data is not
            used to derive the transform.
        return_normalizer: Append the fitted normalizer to the returned tuple.
        val_user_fraction: Fraction of training users held out for epoch selection.
            Reporting the best epoch chosen on the same set it reports inflates the
            number - measured at about +0.02, since it is a max over ~20 noisy
            evaluations. Selecting on a disjoint group removes that.
        return_val: Return (train, val, test) instead of (train, test). `val` is None
            when val_user_fraction is 0.
    Returns:
        If is_train is True: tuple of (train_loader, test_loader)
        If is_train is False: test_loader
        With return_normalizer, the ChannelNormalizer is appended to the result.
    """
    pin_memory = device.type == "cuda" if device else False
    keep_users = select_user_subset(data_dir, max_users, seed)
    if keep_users is not None:
        print(f"User subsample: keeping {len(keep_users)} users, stratified across datasets")

    if not is_train:
        eval_swap_data = not swap_data if test_on_excluded else swap_data
        dataset = SiameseDataset(
            data_dir,
            samples_per_user=samples_per_user,
            sample_time=sample_time,
            sample_rate=sample_rate,
            exclude_users=exclude_users,
            swap_data=eval_swap_data,
            seed=_seed_value(seed, 11),
            within_dataset_negatives=within_dataset_negatives,
            channels=channels,
            cross_session_positives=cross_session_positives,
            center_position=center_position,
            keep_users=keep_users,
        )
        # Evaluation never fits its own statistics when a training-time normalizer is
        # available; doing so would let the held-out distribution shape the transform.
        eval_normalizer = normalizer if normalizer is not None else ChannelNormalizer(normalize).fit(dataset.sample_index)
        eval_normalizer.transform(dataset.sample_index)
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(_seed_value(seed, 12)),
        )
        return (test_loader, eval_normalizer) if return_normalizer else test_loader

    # Users reserved for epoch selection are excluded from training too, so the three
    # groups stay user-disjoint.
    validation_users = select_validation_users(data_dir, exclude_users, val_user_fraction, seed)
    if keep_users is not None:
        validation_users = [u for u in validation_users if u in set(keep_users)]
    training_exclusions = list(exclude_users or []) + validation_users

    train_dataset = SiameseDataset(
        data_dir,
        samples_per_user=samples_per_user,
        sample_time=sample_time,
        sample_rate=sample_rate,
        exclude_users=training_exclusions,
        swap_data=swap_data,
        seed=_seed_value(seed, 1),
        within_dataset_negatives=within_dataset_negatives,
        channels=channels,
        cross_session_positives=cross_session_positives,
        center_position=center_position,
        keep_users=keep_users,
    )

    # Fit on the training users only, then apply the same transform everywhere.
    if normalizer is None:
        normalizer = ChannelNormalizer(normalize).fit(train_dataset.sample_index)
    normalizer.transform(train_dataset.sample_index)
    if normalizer.enabled:
        print(normalizer.describe())

    if test_dir is None:
        if test_on_excluded:
            test_dataset = SiameseDataset(
                data_dir,
                samples_per_user=samples_per_user,
                sample_time=sample_time,
                sample_rate=sample_rate,
                exclude_users=exclude_users,
                swap_data=not swap_data,
                seed=_seed_value(seed, 2),
                within_dataset_negatives=within_dataset_negatives,
                channels=channels,
                cross_session_positives=cross_session_positives,
                center_position=center_position,
                keep_users=keep_users,
            )
            normalizer.transform(test_dataset.sample_index)
        else:
            generator = torch.Generator().manual_seed(_seed_value(seed, 3))
            if len(train_dataset) <= 1:
                test_dataset = train_dataset
            else:
                test_size = int(math.floor(len(train_dataset) * val_split))
                test_size = min(max(test_size, 1), len(train_dataset) - 1)
                train_size = len(train_dataset) - test_size
                train_dataset, test_dataset = random_split(
                    train_dataset,
                    [train_size, test_size],
                    generator=generator,
                )
    else:
        test_swap_data = not swap_data if test_on_excluded else swap_data
        test_dataset = SiameseDataset(
            test_dir,
            samples_per_user=samples_per_user,
            sample_time=sample_time,
            sample_rate=sample_rate,
            exclude_users=exclude_users,
            swap_data=test_swap_data,
            seed=_seed_value(seed, 4),
            within_dataset_negatives=within_dataset_negatives,
            channels=channels,
            cross_session_positives=cross_session_positives,
            center_position=center_position,
            keep_users=keep_users,
        )
        normalizer.transform(test_dataset.sample_index)

    val_loader = None
    if validation_users:
        val_dataset = SiameseDataset(
            data_dir,
            samples_per_user=samples_per_user,
            sample_time=sample_time,
            sample_rate=sample_rate,
            exclude_users=validation_users,
            swap_data=True,
            seed=_seed_value(seed, 22),
            within_dataset_negatives=within_dataset_negatives,
            channels=channels,
            cross_session_positives=cross_session_positives,
            center_position=center_position,
            keep_users=keep_users,
        )
        normalizer.transform(val_dataset.sample_index)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(_seed_value(seed, 23)),
        )
        print(f"Validation split: {len(validation_users)} users held out for epoch selection")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(_seed_value(seed, 5)),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(_seed_value(seed, 6)),
    )
    loaders = (train_loader, val_loader, test_loader) if return_val else (train_loader, test_loader)
    return (*loaders, normalizer) if return_normalizer else loaders


class SiameseDataset(Dataset):
    """
    Dataset of VR user head movement samples for biometric identification siamese.
    """

    def __init__(
        self,
        data_dir: str | list[str],
        samples_per_user: int = 1000,
        sample_time: int = 1,
        sample_rate: int = 10,
        exclude_users: str | list[str] | None = None,
        swap_data: bool = False,
        seed: int | None = None,
        match_ratio: float = 0.5,
        within_dataset_negatives: bool = False,
        channels: str = "full",
        cross_session_positives: bool = False,
        center_position: bool = False,
        keep_users=None,
    ):
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.sample_index = build_sample_index(
            data_dir,
            sample_time=sample_time,
            sample_rate=sample_rate,
            exclude_users=exclude_users,
            swap_data=swap_data,
            channels=channels,
            center_position=center_position,
            keep_users=keep_users,
        )
        self.num_users = self.sample_index.num_users
        self.num_samples = self.sample_index.sample_count
        self.samples_per_user = samples_per_user
        self.seed = seed
        self.match_ratio = match_ratio

        self.manifest = generate_pair_manifest(
            self.sample_index,
            pairs_per_user=self.samples_per_user,
            match_ratio=self.match_ratio,
            seed=self.seed,
            within_dataset_negatives=within_dataset_negatives,
            cross_session_positives=cross_session_positives,
        )
        self.siamese_count = int(self.manifest["labels"].shape[0])

        print(f"Created {self.siamese_count} siamese samples")

    def __len__(self):
        return self.siamese_count

    def __getitem__(self, idx):
        x1_idx = int(self.manifest["x1_indices"][idx])
        x2_idx = int(self.manifest["x2_indices"][idx])
        label = self.manifest["labels"][idx].view(1)
        return (self.sample_index.samples[x1_idx], self.sample_index.samples[x2_idx]), label
