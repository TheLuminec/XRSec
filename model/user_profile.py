"""
User profile component for XR biometric data processing.

Represents a single user and manages the loading of their
associated experiment data streams into Sampler instances.
"""

from sampler import Sampler
import numpy as np
import os
import pandas as pd


class UserProfile:
    """Holds a collection of data samplers for a specific user ID."""

    def __init__(self, user_dir: str, sample_time: int = 1, sample_rate: int = 10):
        self.user_dir = user_dir
        self.data_samplers = []
        self.sample_time = sample_time
        self.sample_rate = sample_rate

        self._load_data()

    # Column order is the pipeline's contract: (time, qx, qy, qz, qw, Hx, Hy, Hz).
    REQUIRED_COLUMNS = [
        'SessionTime',
        'UnitQuaternion.x', 'UnitQuaternion.y', 'UnitQuaternion.z', 'UnitQuaternion.w',
        'HmdPosition.x', 'HmdPosition.y', 'HmdPosition.z',
    ]

    def _load_data(self):
        """Loads every usable CSV experiment file found in the user's directory."""
        self.skipped = {}
        for file in sorted(os.listdir(self.user_dir)):
            if file.endswith(".csv"):
                self._load_data_sample(os.path.join(self.user_dir, file))

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def _load_data_sample(self, path: str):
        """
        Load a single CSV and instantiate a Sampler, skipping unusable files.

        Unusable files are a real and silent problem in this corpus, so they are
        counted rather than allowed to take down a whole dataset:
          - half of Head_and_Gaze_Behavior_Dataset ships head position and gaze rays
            but no quaternion, which used to raise KeyError and made the entire
            dataset unloadable;
          - PanoSaliency contains single-row sessions whose zero duration divides by
            zero in Sampler.
        """
        df = pd.read_csv(path)

        if any(column not in df.columns for column in self.REQUIRED_COLUMNS):
            self._skip("missing required columns")
            return

        data = np.array(df[self.REQUIRED_COLUMNS], dtype=float)

        if data.shape[0] < 2:
            self._skip("fewer than 2 rows")
            return
        if not np.isfinite(data).all():
            self._skip("non-finite values")
            return
        if data[-1, 0] - data[0, 0] <= 0:
            self._skip("non-positive duration")
            return

        self.data_samplers.append(
            Sampler(data, self.sample_time, self.sample_rate))


if __name__ == "__main__":
    PATH = "datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/1/"
    user_profile = UserProfile(PATH)
    print(user_profile.data_samplers[0].get_sample(0))
