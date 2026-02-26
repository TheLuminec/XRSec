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
- scalar increase to set of values
- small random noise
- randomness to which datapoint is selected

## Preprocessing roadmap for Siamese biometric identification

### 1) Window-level canonicalization (recommended)
- **Yes**: normalize each 1-second sample to represent *movement during the window*, not world-frame start pose.
- For position (`HmdPosition.x/y/z`), subtract the first frame position from all 10 frames so each window starts at `(0,0,0)`.
- For orientation (`qx,qy,qz,qw`), left-multiply by the inverse of the first quaternion (`q_rel[t] = q0^-1 * q[t]`) so the first frame is identity.
- This keeps user-specific dynamics (micro-motions, velocity profile, smoothness) while removing static pose bias.

### 2) Add motion-derivative channels
- Keep raw relative channels and add:
  - `Δpos[t] = pos_rel[t] - pos_rel[t-1]`
  - angular velocity from quaternion deltas (`q_rel[t-1]^-1 * q_rel[t]` mapped to axis-angle or 3D rotvec)
- Derivative features are typically more identity-discriminative for gait/posture-style biometrics than absolute pose.

### 3) Re-sample to fixed timestamps + mask jitter
- Current sampling already picks closest points to 10 Hz targets.
- Improve by interpolating to exact timestamps (linear for position, SLERP for quaternion) so each column has consistent timing.
- Keep a quality flag or mask for dropped/repeated frames (useful when packet loss occurs).

### 4) Quaternion hygiene
- Enforce unit norm each frame.
- Resolve sign ambiguity (`q` and `-q` are same rotation) by flipping sign to maintain temporal continuity (e.g., make dot(q[t], q[t-1]) >= 0).
- This prevents artificial jumps that hurt temporal models and Siamese distance learning.

### 5) Dataset-level normalization (after canonicalization)
- Compute train-set means/stds per channel and z-score normalize.
- Do not mix validation/test stats into train normalization.
- Suggested order: canonicalize -> derive motion features -> z-score.

### 6) Pair/triplet construction for future Siamese training
- Build positive pairs from same user, different sessions/tasks/devices when possible.
- Build hard negatives from users with similar coarse movement statistics.
- Balance easy/hard pairs so the embedding learns robust identity boundaries.

### 7) Augmentation changes
- Avoid multiplicative noise that can invert signs globally.
- Prefer small additive Gaussian noise, time warping, mild temporal cropping/shift, and occasional frame dropout.
- Keep augmentations physically plausible for headset motion.

### 8) Evaluation split strategy for biometrics
- Avoid random window split leakage across same recording.
- Split by session/video/date so test windows are temporally disjoint from train windows.
- For deployment realism, include enrollment vs verification/identification style evaluations.
