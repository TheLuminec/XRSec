"""
PyTorch Dataset for XR user biometric identification.

Wraps the existing data pipeline (Users -> UserProfile -> Sampler)
into a standard PyTorch Dataset for use with DataLoader.

Each sample is a (7, 10) tensor representing one second of data:
    - 7 channels: qx, qy, qz, qw, Hx, Hy, Hz
    - 10 time samples at 10Hz
    - Time column (col 0) is stripped
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from users import Users


def create_dataloader_from_path(
    data_dir,
    batch_size: int,
    device: torch.device,
    is_train: bool = True,
    test_dir=None,
    sample_time: int = 1,
    sample_rate: int = 10,
    val_split: float = 0.2,
    num_workers: int = 0,
    exclude_users=None,
    swap_data=False,
    test_on_excluded: bool = False
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
        val_split: Fraction of dataset to use for validation split if test_dir is None
        num_workers: Number of DataLoader workers
        exclude_users: User paths to exclude from data loading
        swap_data: Whether to swap what is included and excluded
        test_on_excluded: If true, uses the excluded paths for the testing dataset instead of doing a random split
    Returns:
        If is_train is True: tuple of (train_loader, test_loader)
        If is_train is False: test_loader
    """
    pin_memory = device.type == 'cuda' if device else False

    if not is_train:
        eval_swap_data = not swap_data if test_on_excluded else swap_data
        dataset = SiameseDataset(data_dir, sample_time=sample_time, sample_rate=sample_rate,
                                 exclude_users=exclude_users, swap_data=eval_swap_data)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)
        return test_loader

    train_dataset = SiameseDataset(data_dir, sample_time=sample_time,
                                   sample_rate=sample_rate, exclude_users=exclude_users, swap_data=swap_data)
    if test_dir is None:
        if test_on_excluded:
            test_dataset = SiameseDataset(
                data_dir, sample_time=sample_time, sample_rate=sample_rate, exclude_users=exclude_users, swap_data=not swap_data)
        else:
            generator = torch.Generator().manual_seed(42)
            test_size = int(len(train_dataset) * val_split)
            train_size = len(train_dataset) - test_size
            train_dataset, test_dataset = random_split(
                train_dataset,
                [train_size, test_size],
                generator=generator
            )
    else:
        test_swap_data = not swap_data if test_on_excluded else swap_data
        test_dataset = SiameseDataset(test_dir, sample_time=sample_time,
                                      sample_rate=sample_rate, exclude_users=exclude_users, swap_data=test_swap_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


class SampleDataset():
    """
    Dataset of VR user head movement samples for biometric identification raw.

    Args:
        data_dir: Path to processed_data/users/ directory
        exclude_users: Optional path(s) to exclude
        swap_data: Whether to swap what is included and excluded
    """

    def __init__(self, data_dir: [str | list], sample_time=1, sample_rate=10, exclude_users: [str | list] = None, swap_data=False):
        self.dataset = []
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.num_users = 0
        self.swap_data = swap_data

        if exclude_users is None:
            exclude_users = []
        elif isinstance(exclude_users, str):
            exclude_users = [exclude_users]

        users = []
        if isinstance(data_dir, str):
            users = [Users(data_dir, sample_time, sample_rate)]
        else:
            for d in data_dir:
                users.append(Users(d, sample_time, sample_rate))

        self.sample_count = 0
        # User directories
        for u in users:
            # User profiles
            for profile in u.user_profiles:
                # Check exclude users

                if self.swap_data:
                    if profile.user_dir not in exclude_users:
                        continue
                else:
                    if profile.user_dir in exclude_users:
                        continue

                self.num_users += 1
                samples = []
                # Data samplers
                for sampler in profile.data_samplers:
                    if sampler.sample_count == 0:
                        continue
                    all_samples = sampler.get_all_samples()  # (num_windows, 10, 8)
                    for sample in all_samples:
                        features = sample[:, 1:].astype(
                            np.float32)      # (10, 7) - strip time col
                        # (7, 10)
                        M = features.T
                        samples.append(M)
                        self.sample_count += 1

                if samples:
                    self.dataset.append(torch.tensor(
                        np.array(samples), dtype=torch.float32))
                else:
                    self.dataset.append(torch.empty(
                        (0, 7, 10), dtype=torch.float32))

        print(
            f"Loaded {self.sample_count} samples from {self.num_users} users")

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return self.dataset[idx]


class SiameseDataset(Dataset):
    """
    Dataset of VR user head movement samples for biometric identification siamese.

    Args:
        data_dir: Path to processed_data/users/ directory
        exclude_users: Optional path(s) to exclude
    """

    def __init__(self, data_dir: [str | list], samples_per_user=10000, sample_time=1, sample_rate=10, exclude_users: [str | list] = None, swap_data=False):
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.sample_dataset = SampleDataset(
            data_dir, sample_time, sample_rate, exclude_users, swap_data)
        self.num_users = self.sample_dataset.num_users
        self.num_samples = self.sample_dataset.sample_count
        self.samples_per_user = samples_per_user

        # Optimal amount of samples, but is too large to fit in memory
        # self.siamese_count = self.num_samples * self.num_samples
        self.siamese_count = self.num_users * self.samples_per_user

        self.dataset = []
        self._generate_dataset()

        print(f"Created {self.siamese_count} siamese samples")
        del self.sample_dataset

    def _generate_dataset(self, seed=67):
        """
        Generate siamese dataset.
        To do this we need to select self.samples_per_user samples from each user.
        Then we need to select matching/non-matching samples.
        If is_match is True, select from the same user, otherwise select from a random user.
        If the random user has no samples, select from the same user. (Giving slight preference to matching samples)
        """
        np.random.seed(seed)

        x1_list = []
        x2_list = []
        y_list = []

        for u in range(self.num_users):
            n_u = len(self.sample_dataset[u])
            if n_u == 0:
                continue

            # Randomly select self.samples_per_user from u
            x_indices = np.random.randint(0, n_u, size=self.samples_per_user)
            x1_list.append(self.sample_dataset[u][x_indices])

            # Pre-select matching/non-matching
            is_match = np.random.rand(self.samples_per_user) < 0.5
            # If is_match is True, select from the same user, otherwise select from a random user (excluding the same user)
            rng = np.random.default_rng(seed)
            choices = np.arange(self.num_users)
            choices = np.delete(choices, u)
            r_users = np.where(is_match, u, rng.choice(choices, size=self.samples_per_user))

            x2_tensors = []
            for i in range(self.samples_per_user):
                r = r_users[i]
                n_r = len(self.sample_dataset[r])
                # If the random user has no samples, select from the same user
                if n_r == 0:
                    r = u
                    n_r = len(self.sample_dataset[r])
                y = np.random.randint(0, n_r)
                x2_tensors.append(self.sample_dataset[r][y])

            x2_list.append(torch.stack(x2_tensors))
            y_list.append(torch.tensor(
                is_match, dtype=torch.float32).unsqueeze(1))

        self.x1 = torch.cat(x1_list, dim=0)
        self.x2 = torch.cat(x2_list, dim=0)
        self.y = torch.cat(y_list, dim=0)
        self.siamese_count = self.x1.shape[0]

    def __len__(self):
        return self.siamese_count

    def __getitem__(self, idx):
        return (self.x1[idx], self.x2[idx]), self.y[idx]
