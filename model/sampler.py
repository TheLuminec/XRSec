import numpy as np
import math

# Class to sample data from processed data files
class Sampler:
    def __init__(self, data: np.ndarray, sample_rate=10, variance=0.01):
        self.data = data
        self.sample_rate = sample_rate
        self.variance = variance
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
        result = np.zeros((self.sample_rate, self.data.shape[1]))
        end_range = int(self.avg_hertz * 2)
        for i in range(self.sample_rate):
            current_time = self.time_start + current_index + (i / self.sample_rate)
            data_index = self._get_data_point_closest_to_time(current_time, last_range_end, last_range_end + end_range)
            result[i] = self.data[data_index]
            last_range_end = data_index
        return result, last_range_end

    def _preprocess(self):
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
