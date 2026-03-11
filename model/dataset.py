"""
PyTorch Dataset for XR user biometric identification.

Wraps the existing data pipeline (Users -> UserProfile -> Sampler)
into a standard PyTorch Dataset for use with DataLoader.

Each sample is a (7, 10) tensor representing one second of data:
    - 7 channels: qx, qy, qz, qw, Hx, Hy, Hz
    - 10 time samples at 10Hz
    - Time column (col 0) is stripped
"""

from torch import long
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
            users = [Users(data_dir)]
        else:
            for d in data_dir:
                users.append(Users(d))

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
                        samples.append(torch.tensor(M, dtype=torch.float32))
                        self.sample_count += 1
                        
                self.dataset.append(samples)

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
    def __init__(self, data_dir: [str | list], sample_time = 1, sample_rate=10):
        self.sample_time = sample_time
        self.sample_rate = sample_rate
        self.sample_dataset = SampleDataset(data_dir, sample_time, sample_rate)
        self.num_users = self.sample_dataset.num_users
        self.num_samples = self.sample_dataset.sample_count
        self.siamese_count = self.num_samples * self.num_samples

        print(f"Created {self.siamese_count} siamese samples")

    def __len__(self):
        return self.siamese_count

    def __getitem__(self, idx):
        x = idx // self.num_samples
        y = idx % self.num_samples
        label = 1 if x == y else 0
        return (self.sample_dataset[x], self.sample_dataset[y], torch.tensor(label, dtype=torch.long))

