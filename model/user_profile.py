"""
User profile component for XR biometric data processing.

Represents a single user and manages the loading of their
associated experiment data streams into Sampler instances.
"""

from sampler import Sampler
import numpy as np
import os

class UserProfile:
    """Holds a collection of data samplers for a specific user ID."""
    def __init__(self, user_id: int, user_dir: str):
        self.user_id = user_id
        self.user_dir = user_dir
        self.data_samplers = []

        self._load_data()

    def _load_data(self):
        """Loads all CSV experiment files found in the user's directory."""
        for file in os.listdir(self.user_dir):
            if file.endswith(".csv"):
                self._load_data_sample(os.path.join(self.user_dir, file))

    def _load_data_sample(self, path: str):
        """Loads a single CSV file and instantiates a Sampler with data augmentation."""
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.data_samplers.append(Sampler(data, index_randomness=1, scalar_randomness=2))


if __name__ == "__main__":
    PATH = "datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/1/"
    user_profile = UserProfile(1, PATH)
    print(user_profile.data_samplers[0].get_sample(0))
    

