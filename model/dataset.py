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
import random
import collections
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from users import Users


class XRSecDataset(Dataset):
    """
    Dataset of VR user head movement samples for biometric identification.

    Args:
        data_dir: Path to processed_data/users/ directory
        index_load: Tuple slice over each user's samplers, e.g. (0, None)
    """

    def __init__(self, data_dir: str, index_load: tuple = (0, None), siamese: bool = False):
        self.samples = []
        self.labels = []
        self.index_load = index_load
        self.siamese = siamese

        users = Users(data_dir)
        user_ids = sorted(users.user_profiles.keys())
        self.user_id_to_label = {uid: i for i, uid in enumerate(user_ids)}
        self.label_to_user_id = {i: uid for uid, i in self.user_id_to_label.items()}
        self.num_users = len(user_ids)

        for uid, profile in users.user_profiles.items():
            label = self.user_id_to_label[uid]
            used_samplers = profile.data_samplers[self.index_load[0]:self.index_load[1]]
            for sampler in used_samplers:
                all_samples = sampler.get_all_samples()  # (num_windows, 10, 8)
                for i in range(sampler.sample_count):
                    sample = all_samples[i]                          # (10, 8)
                    features = sample[:, 1:].astype(np.float32)      # (10, 7) - strip time col
                    M = features.T                                   # (7, 10)
                    self.samples.append(torch.tensor(M, dtype=torch.float32))
                    self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.norm_mean = None
        self.norm_std = None

        if self.siamese:
            self.label_to_indices = collections.defaultdict(list)
            for idx, label in enumerate(self.labels.tolist()):
                self.label_to_indices[label].append(idx)
            self.unique_labels = list(self.label_to_indices.keys())

        print(f"Loaded {len(self.samples)} samples from {self.num_users} users")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.siamese:
            return self.samples[idx], self.labels[idx]
        
        sample1 = self.samples[idx]
        label1 = self.labels[idx].item()
        
        target_same = random.random() > 0.5
        
        if target_same:
            # Positive pair
            idx2 = random.choice(self.label_to_indices[label1])
            while idx2 == idx and len(self.label_to_indices[label1]) > 1:
                idx2 = random.choice(self.label_to_indices[label1])
            y = 1.0
        else:
            # Negative pair
            label2 = random.choice(self.unique_labels)
            while label2 == label1 and len(self.unique_labels) > 1:
                label2 = random.choice(self.unique_labels)
            idx2 = random.choice(self.label_to_indices[label2])
            y = 0.0
            
        sample2 = self.samples[idx2]
        return (sample1, sample2), torch.tensor([y], dtype=torch.float32)


if __name__ == "__main__":
    PATH = "datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/"
    dataset = XRSecDataset(PATH)
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")
    print(f"Total samples: {len(dataset)}, Num users: {dataset.num_users}")
