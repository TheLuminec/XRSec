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
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from users import Users


class XRSecDataset(Dataset):
    """
    Dataset of VR user head movement samples for biometric identification.

    Args:
        data_dir: Path to processed_data/users/ directory
    """
    def __init__(self, data_dir: str):
        self.samples = []
        self.labels = []

        users = Users(data_dir)
        user_ids = sorted(users.user_profiles.keys())
        self.user_id_to_label = {uid: i for i, uid in enumerate(user_ids)}
        self.label_to_user_id = {i: uid for uid, i in self.user_id_to_label.items()}
        self.num_users = len(user_ids)

        for uid, profile in users.user_profiles.items():
            label = self.user_id_to_label[uid]
            for sampler in profile.data_samplers:
                all_samples = sampler.get_all_samples()  # (num_windows, 10, 8)
                for i in range(sampler.sample_count):
                    sample = all_samples[i]              # (10, 8)
                    features = sample[:, 1:]             # (10, 7) - strip time col
                    M = features.T                       # (7, 10) - channels x time
                    self.samples.append(torch.tensor(M, dtype=torch.float32))
                    self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"Loaded {len(self.samples)} samples from {self.num_users} users")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


if __name__ == "__main__":
    PATH = "datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/"
    dataset = XRSecDataset(PATH)
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")
    print(f"Total samples: {len(dataset)}, Num users: {dataset.num_users}")
