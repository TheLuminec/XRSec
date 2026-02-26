"""
Entry point for dataset loading.

Traverses the dataset directory structure to discover users
and initialize their associated UserProfile instances.
"""

from user_profile import UserProfile
import numpy as np
import os

class Users:
    """Container managing all loaded user profiles in the dataset."""
    def __init__(self, user_dir: str):
        self.user_dir = user_dir
        self.user_profiles = {}

        self._load_user_profiles()

    def _load_user_profiles(self):
        """Iterates through subdirectories, treating each as a user ID."""
        for file in os.listdir(self.user_dir):
            user_dir = os.path.join(self.user_dir, file)
            if os.path.isdir(user_dir):
                self._load_user_profile(int(file), user_dir)

    def _load_user_profile(self, user_id: int, path: str):
        user_profile = UserProfile(user_id, path)
        self.user_profiles[user_id] = user_profile

if __name__ == "__main__":
    PATH = "datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/"
    users = Users(PATH)
    print(users.user_profiles)