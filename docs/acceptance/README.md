# Acceptance records for code changes that touch numerics

One directory per change. Each holds the before and after measurements the merge was
accepted on, and the script that produced them, so the statement in the commit message
can be checked rather than believed.

## dyn_float64 (2026-09-04)

`input_encoding._dynamics_only` centres the position residual in float64 before casting
back. Acceptance ruled by the Coordinator: CPU-before against CPU-after, same script,
same checkpoint (`sweeps/314cd507f1/runs/bilstm_dbc29cfc5f/best.pth`, dyn, full corpus),
every held-out corpus within 1e-4 AUC, and the Nymeria residual window-mean no longer
scaling with the absolute coordinate and below 1e-7 m.

| corpus | before (CPU) | after (CPU) | gap |
| --- | --- | --- | --- |
| VR_User_Behavior | 0.521389 | 0.521389 | 2.8e-7 |
| ViewGauss | 0.570754 | 0.570754 | 2.7e-7 |
| Head_and_Gaze | 0.569590 | 0.569590 | 4.6e-8 |
| NJIT | 0.539575 | 0.539576 | 7.2e-7 |
| PanoSaliency | 0.731514 | 0.731634 | **1.2e-4** |
| Panonut360 | 0.543486 | 0.543486 | 2.8e-7 |
| EyeNavGS | 0.528596 | 0.528596 | 3.6e-8 |

**A re-baseline of 1.2e-4 AUC on PanoSaliency**, stated as the rule requires; inert
(below 1e-6) on every other corpus. Mechanism: PanoSaliency's only live channel under
`dyn` is the residual of a unit direction vector (its quaternion is a dead constant), so
it is the one corpus that reads the centring at the 1e-7 level - also a measured clue
that head-direction dynamics are its signal (section 10 step 5).

Residue on Nymeria (SLAM coordinates to 30 m): before, median 1.8e-7 m and max 3.6e-5 m,
scaling with the absolute coordinate; after, median 4.9e-10 m and max 3.8e-8 m, float32
rounding of the centred values only. An earlier expectation of ~1e-14 m assumed float64
storage; windows are stored in float32, so the corrected acceptance is "no longer scales
with the coordinate, below 1e-7 m", met.

The recorded GPU rows differ from CPU scoring by up to 7e-4 with no code change at all
(cuDNN versus CPU float32 in the BiLSTM); score differences below about 1e-3 between
devices are arithmetic, not results.
