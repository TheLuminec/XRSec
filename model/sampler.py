"""
Data sampler for time-series XR biometric data.

Provides a Sampler class to extract fixed-rate samples from variable-rate
raw data streams, applying optional randomness for data augmentation.
"""

import numpy as np
import math
import random


class Sampler:
    """
    Samples fixed-rate data slices from processed telemetry.

    Handles variable framerate data by finding the closest data point
    to each target temporal step. Supports data augmentation via index
    randomness.
    """

    def __init__(self, data: np.ndarray, sample_rate=10, variance=0.01, index_randomness=1):
        self.data = data
        self.sample_rate = sample_rate
        self.variance = variance
        self.index_randomness = index_randomness
        self.data_len = data.shape[0]
        self.time_start = data[0, 0]
        self.time_end = data[-1, 0]
        self.duration = self.time_end - self.time_start
        self.avg_hertz = data.shape[0] / self.duration
        self.current_index = 0
        self.sample_count = math.floor(self.duration)

        self._preprocess()

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self) -> np.ndarray:
        if self.current_index >= self.sample_count:
            raise StopIteration
        self.current_index += 1
        return self.samples[self.current_index - 1]

    def _get_data_point_closest_to_time(self, time: float, search_range_start: int = 0, search_range_end: int = None) -> int:
        """Finds the index of the raw data point closest to the specified target time."""
        if search_range_end is None:
            search_range_end = self.data_len
        search_range_end = min(search_range_end, self.data_len)
        min_diff = float('inf')
        closest_index = -1
        for i in range(search_range_start, search_range_end):
            diff = abs(self.data[i, 0] - time)
            if diff < min_diff:
                min_diff = diff
                closest_index = i
                if diff <= self.variance:
                    break
        return closest_index

    def _get_sample_slice(self, current_index: int, last_range_end: int = 0) -> tuple[np.ndarray, int]:
        """Extracts a 1-second window of data containing `sample_rate` uniformly spaced samples."""
        result = np.zeros((self.sample_rate, self.data.shape[1]))
        end_range = int(self.avg_hertz * 2)
        for i in range(self.sample_rate):
            current_time = self.time_start + current_index + (i / self.sample_rate)
            data_index = self._get_data_point_closest_to_time(current_time, last_range_end, last_range_end + end_range)
            if self.index_randomness > 0:
                data_index += random.randint(-self.index_randomness, self.index_randomness)
            data_index = max(0, min(self.data_len - 1, data_index))
            result[i] = self.data[data_index]
            last_range_end = data_index
        return result, last_range_end

    def _preprocess(self):
        """
        Pre-computes and caches all fixed-rate data slices from the raw timeline.
        """
        self.samples = np.zeros((self.sample_count, self.sample_rate, self.data.shape[1]))
        last_range_end = 0
        for i in range(self.sample_count):
            self.samples[i], last_range_end = self._get_sample_slice(i, last_range_end)

    def get_sample(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.sample_count:
            raise IndexError("Index out of bounds")
        return self.samples[index]

    def get_all_samples(self) -> np.ndarray:
        return self.samples


if __name__ == "__main__":
    PATH = "datasets/VR_User_Behavior_Dataset_(Spherical_Video_Streaming)/processed_data/users/1/experiment_1_video_0.csv"
    data = np.loadtxt(PATH, delimiter=",", skiprows=1)
    sampler = Sampler(data)
    print(sampler.get_sample(5))
