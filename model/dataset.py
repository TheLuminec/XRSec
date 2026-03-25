"""
PyTorch Dataset for XR user biometric identification.

Wraps the existing data pipeline (Users -> UserProfile -> Sampler)
into a standard PyTorch Dataset for use with DataLoader.

Each sample is a (7, 10) tensor representing one second of data:
    - 7 channels: qx, qy, qz, qw, Hx, Hy, Hz
    - 10 time samples at 10Hz
    - Time column (col 0) is stripped
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from users import Users


class SampleDataset():
    """
    Dataset of VR user head movement samples for biometric identification raw.

    Args:
        data_dir: Path to processed_data/users/ directory
    """
    def __init__(self, data_dir: [str | list], sample_time = 1, sample_rate=10):
        self.dataset = []
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.num_users = 0

        users = []
        if isinstance(data_dir, str):
            users = [Users(data_dir, sample_time, sample_rate)]
        else:
            for d in data_dir:
                users.append(Users(d, sample_time, sample_rate))

        self.sample_count = 0
        for u in users:
            for i in range(len(u)):
                profile = u.user_profiles[i]
                self.num_users += 1
                samples = []
                for sampler in profile.data_samplers:
                    if sampler.sample_count < self.sample_rate * self.sample_time:
                        continue
                    all_samples = sampler.get_all_samples()  # (num_windows, 10, 8)
                    for sample in all_samples:  
                        features = sample[:, 1:].astype(np.float32)      # (10, 7) - strip time col
                        M = features.T                                   # (7, 10)
                        samples.append(M)
                        self.sample_count += 1
                        
                if samples:
                    self.dataset.append(torch.tensor(np.array(samples), dtype=torch.float32))
                else:
                    self.dataset.append(torch.empty((0, 7, 10), dtype=torch.float32))

        print(f"Loaded {self.sample_count} samples from {self.num_users} users")

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return self.dataset[idx]


class SiameseDataset(Dataset):
    """
    Dataset of VR user head movement samples for biometric identification siamese.

    Args:
        data_dir: Path to processed_data/users/ directory
    """
    def __init__(self, data_dir: [str | list], samples_per_user = 10000, sample_time = 1, sample_rate=10):
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.sample_dataset = SampleDataset(data_dir, sample_time, sample_rate)
        self.num_users = self.sample_dataset.num_users
        self.num_samples = self.sample_dataset.sample_count
        self.samples_per_user = samples_per_user

        # Optimal amount of samples, but is too large to fit in memory
        # self.siamese_count = self.num_samples * self.num_samples
        self.siamese_count = self.num_users * self.samples_per_user

        self.dataset = []
        self._generate_dataset()

        print(f"Created {self.siamese_count} siamese samples")

    def _generate_dataset(self, seed = 67):
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
            # If is_match is True, select from the same user, otherwise select from a random user
            r_users = np.where(is_match, u, np.random.randint(0, self.num_users, size=self.samples_per_user))

            x2_tensors = []
            for i in range(self.samples_per_user):
                r = r_users[i]
                n_r = len(self.sample_dataset[r])
                # If the random user has no samples, select from the same user
                y = np.random.randint(0, n_r) if n_r > 0 else 0
                x2_tensors.append(self.sample_dataset[r][y])
            
            x2_list.append(torch.stack(x2_tensors))
            y_list.append(torch.tensor(is_match, dtype=torch.float32).unsqueeze(1))

        self.x1 = torch.cat(x1_list, dim=0)
        self.x2 = torch.cat(x2_list, dim=0)
        self.y = torch.cat(y_list, dim=0)

    def __len__(self):
        return self.siamese_count

    def __getitem__(self, idx):
        return (self.x1[idx], self.x2[idx]), self.y[idx]

