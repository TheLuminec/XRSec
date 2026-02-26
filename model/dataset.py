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


class XRSecDataset(Dataset):
    """
    Dataset of VR user head movement samples for biometric identification.

    Args:
        data_dir: Path to processed_data/users/ directory
        index_load: Tuple slice over each user's samplers, e.g. (0, None)
        canonicalize: If True, convert each window to a movement-relative frame
                      (position starts at 0, quaternion starts at identity)
    """

    def __init__(self, data_dir: str, index_load: tuple = (0, None), canonicalize: bool = True):
        self.samples = []
        self.labels = []
        self.index_load = index_load
        self.canonicalize = canonicalize

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
                    if self.canonicalize:
                        features = self._canonicalize_window(features)
                    M = features.T                                   # (7, 10)
                    self.samples.append(torch.tensor(M, dtype=torch.float32))
                    self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.norm_mean = None
        self.norm_std = None
        print(f"Loaded {len(self.samples)} samples from {self.num_users} users")

    @staticmethod
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return (q / norm).astype(np.float32)

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=np.float32)

    def _canonicalize_window(self, features: np.ndarray) -> np.ndarray:
        """
        Converts a (10, 7) [quat, pos] window to movement-relative coordinates.

        - Position is translated so first frame is (0,0,0)
        - Quaternion is made continuous, normalized, and converted to relative
          orientation using q_rel[t] = q0^-1 * q[t]
        """
        out = features.copy()

        quats = out[:, :4]
        for i in range(len(quats)):
            quats[i] = self._quat_normalize(quats[i])
            if i > 0 and np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]

        q0_inv = self._quat_conjugate(quats[0])
        for i in range(len(quats)):
            quats[i] = self._quat_normalize(self._quat_multiply(q0_inv, quats[i]))

        pos0 = out[0, 4:7].copy()
        out[:, 4:7] = out[:, 4:7] - pos0
        out[:, :4] = quats
        return out

    def fit_normalization(self, indices):
        """Fit per-channel z-score stats from a subset of sample indices."""
        if len(indices) == 0:
            raise ValueError("Cannot fit normalization with empty indices")

        stacked = torch.stack([self.samples[i] for i in indices], dim=0)  # (N, 7, 10)
        mean = stacked.mean(dim=(0, 2), keepdim=True)                      # (1, 7, 1)
        std = stacked.std(dim=(0, 2), keepdim=True)
        std = torch.clamp(std, min=1e-6)

        self.norm_mean = mean
        self.norm_std = std
        self.apply_normalization(mean, std)
        return mean, std

    def apply_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Apply provided per-channel z-score stats to all samples."""
        self.norm_mean = mean
        self.norm_std = std
        for i in range(len(self.samples)):
            self.samples[i] = (self.samples[i] - mean.squeeze(0)) / std.squeeze(0)

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
