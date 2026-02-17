inputs = time_from_start, qx, qy, qz, qw, hz, hy, hz
samples_per = 10
timestep_between_samples = 0.1s

for now output full data resolution (time offset for more points per data sample in training)

data_source:
- users
- sessions
    - task in session / metadata for each session
    - user info, if avaliable
- notes about data collected and methodologies (info.txt works)


siamese model for secure detection
smaller model to decide if user switch?