inputs = time_from_start, qx, qy, qz, qw, hz, hy, hz
samples_per = 10
timestep_between_samples = 0.1s

for now output full data resolution (time offset for more points per data sample in training)

data_source:
-processed_data:
    - users
        - data for each user
    - user info, if avaliable
    - tasks meta data
- notes about data collected and methodologies (info.txt works)
- parser.py

siamese model for secure detection
add noise to data to increase robustness
- small random noise
- randomness to which datapoint is selected
