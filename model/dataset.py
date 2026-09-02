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
from user_profile import UserProfile


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
    ):
        self.dataset = []
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.num_users = 0
        self.swap_data = swap_data

        if exclude_users is None:
            exclude_users = []
        elif isinstance(exclude_users, str):
            exclude_users = [exclude_users]

        data_dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
        # Dataset of origin per user, so normalisation can be fitted per dataset.
        self.dataset_names = [self._dataset_name(directory) for directory in data_dirs]
        self.user_dataset_ids: list[int] = []
        self.skipped_files: dict[str, int] = {}

        # Users are filtered before loading rather than after: excluded users cost
        # nothing to skip, which matters for leave-users-out sweeps.
        self.sample_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        for dataset_id, directory in enumerate(data_dirs):
            for user_dir in self._iter_user_dirs(directory):
                if self.swap_data:
                    if user_dir not in exclude_users:
                        continue
                else:
                    if user_dir in exclude_users:
                        continue

                self.num_users += 1
                self.user_dataset_ids.append(dataset_id)
                user_samples = self._load_user_samples(user_dir)
                self.sample_count += int(user_samples.shape[0])
                self.dataset.append(user_samples)

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

    def _load_user_samples(self, user_dir: str) -> torch.Tensor:
        """Return one user's windows as (n_samples, 7, seq_len), via cache when possible."""
        cache_path = None
        if sample_cache.cache_enabled():
            cache_path = sample_cache.entry_path(Path(user_dir), self.sample_time, self.sample_rate)
            cached = sample_cache.load(cache_path)
            if cached is not None:
                self.cache_hits += 1
                return cached

        self.cache_misses += 1
        profile = UserProfile(user_dir, self.sample_time, self.sample_rate)
        for reason, count in getattr(profile, "skipped", {}).items():
            self.skipped_files[reason] = self.skipped_files.get(reason, 0) + count

        samples = []
        for sampler in profile.data_samplers:
            if sampler.sample_count == 0:
                continue
            for sample in sampler.get_all_samples():
                # Drop the SessionTime column, then transpose to (channels, timesteps).
                samples.append(sample[:, 1:].astype(np.float32).T)

        if samples:
            user_samples = torch.tensor(np.array(samples), dtype=torch.float32)
        else:
            seq_len = self.sample_time * self.sample_rate
            user_samples = torch.empty((0, 7, seq_len), dtype=torch.float32)

        if cache_path is not None:
            sample_cache.store(cache_path, user_samples)
        return user_samples

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return self.dataset[idx]


class SampleIndex:
    """
    Stable index over flattened per-user samples.
    """

    def __init__(self, sample_dataset: SampleDataset):
        self.sample_time = sample_dataset.sample_time
        self.sample_rate = sample_dataset.sample_rate
        self.seq_len = self.sample_time * self.sample_rate
        self.num_users = sample_dataset.num_users

        self.dataset_names = list(getattr(sample_dataset, "dataset_names", []))
        user_dataset_ids = getattr(sample_dataset, "user_dataset_ids", [])
        self.user_dataset_ids = list(user_dataset_ids)

        self.user_sample_indices: list[torch.Tensor] = []
        flat_samples = []
        window_dataset_ids = []
        offset = 0
        for user_index, user_samples in enumerate(sample_dataset.dataset):
            sample_count = int(user_samples.shape[0])
            if sample_count > 0:
                flat_samples.append(user_samples)
                indices = torch.arange(offset, offset + sample_count, dtype=torch.long)
                dataset_id = user_dataset_ids[user_index] if user_index < len(user_dataset_ids) else 0
                window_dataset_ids.append(torch.full((sample_count,), dataset_id, dtype=torch.long))
            else:
                indices = torch.empty(0, dtype=torch.long)
            self.user_sample_indices.append(indices)
            offset += sample_count

        if flat_samples:
            self.samples = torch.cat(flat_samples, dim=0)
            self.window_dataset_ids = torch.cat(window_dataset_ids, dim=0)
        else:
            self.samples = torch.empty((0, 7, self.seq_len), dtype=torch.float32)
            self.window_dataset_ids = torch.empty(0, dtype=torch.long)

        self.sample_count = int(self.samples.shape[0])

    def __len__(self):
        return self.sample_count


def build_sample_index(
    data_dir,
    sample_time: int = 1,
    sample_rate: int = 10,
    exclude_users=None,
    swap_data: bool = False,
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
        )
    )


def generate_pair_manifest(
    sample_index: SampleIndex,
    pairs_per_user: int,
    match_ratio: float = 0.5,
    seed: int | None = None,
    within_dataset_negatives: bool = False,
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
    """
    if pairs_per_user <= 0 or sample_index.num_users == 0 or sample_index.sample_count == 0:
        return _empty_pair_manifest()

    rng = np.random.default_rng(_seed_value(seed))
    positive_target, negative_target = _pairs_per_label(pairs_per_user, match_ratio)

    x1_indices = []
    x2_indices = []
    labels = []
    anchor_user_ids = []

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
            x1_pos = rng.choice(user_samples.numpy(), size=local_positive_target, replace=True)
            x2_pos = rng.choice(user_samples.numpy(), size=local_positive_target, replace=True)
            x1_indices.extend(x1_pos.tolist())
            x2_indices.extend(x2_pos.tolist())
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
    normalizer: ChannelNormalizer | None = None,
    return_normalizer: bool = False,
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
        normalizer: A pre-fitted normalizer to apply instead of fitting a new one.
            Evaluation must pass the training-time normalizer so held-out data is not
            used to derive the transform.
        return_normalizer: Append the fitted normalizer to the returned tuple.
    Returns:
        If is_train is True: tuple of (train_loader, test_loader)
        If is_train is False: test_loader
        With return_normalizer, the ChannelNormalizer is appended to the result.
    """
    pin_memory = device.type == "cuda" if device else False

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

    train_dataset = SiameseDataset(
        data_dir,
        samples_per_user=samples_per_user,
        sample_time=sample_time,
        sample_rate=sample_rate,
        exclude_users=exclude_users,
        swap_data=swap_data,
        seed=_seed_value(seed, 1),
        within_dataset_negatives=within_dataset_negatives,
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
        )
        normalizer.transform(test_dataset.sample_index)

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
    return (train_loader, test_loader, normalizer) if return_normalizer else (train_loader, test_loader)


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
    ):
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.sample_index = build_sample_index(
            data_dir,
            sample_time=sample_time,
            sample_rate=sample_rate,
            exclude_users=exclude_users,
            swap_data=swap_data,
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
