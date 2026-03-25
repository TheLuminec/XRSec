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

    def _load_data(self):
        """Loads all CSV experiment files found in the user's directory."""
        for file in sorted(os.listdir(self.user_dir)):
            if file.endswith(".csv"):
                self._load_data_sample(os.path.join(self.user_dir, file))

    def _load_data_sample(self, path: str):
        """Loads a single CSV file and instantiates a Sampler."""
        df = pd.read_csv(path)

        # Data is in the form of (time, qx, qy, qz, qw, Hx, Hy, Hz)
        data = np.array(df[['SessionTime',
                            'UnitQuaternion.x', 'UnitQuaternion.y', 'UnitQuaternion.z', 'UnitQuaternion.w',
                            'HmdPosition.x', 'HmdPosition.y', 'HmdPosition.z']])

        self.data_samplers.append(Sampler(data, self.sample_time, self.sample_rate))


if __name__ == "__main__":
    PATH = "datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/1/"
    user_profile = UserProfile(PATH)
    print(user_profile.data_samplers[0].get_sample(0))
    

