# SOTA reproduction: Rack et al. 2023 on who_is_alyx — REGISTERED BEFORE RUNNING

**Written 2026-09-10 on the Miami server, before any training run and before any number
from this code existed on this machine.** Registered per the project rule that a target
chosen after seeing a result is not a target.

## What is being reproduced, and what is NOT

**Reproducing:** Rack, Kobs, Fernando, Hotho, Latoschik, *Versatile User Identification in
Extended Reality Using Pretrained Similarity-Learning* (arXiv:2302.07517, TVCG 2024), on
the `who-is-alyx` dataset, using **their own code** at
`cschell/Versatile-XR-User-Identification` @ `97f054baf04141ccf0943d207f2ff24a8e8bd1aa`
and `cschell/Motion-Learning-Toolbox` @ `b8189e6c6250527b974c0aa5ccae309964eefe5f`.

**NOT reproducing: Schach et al. 2026 (Across-XR, the 18.0% target).** That architecture is
**absent from this code** — verified, not assumed: zero hits for `transformer` across the
whole clone, `src/models/` holds only `cnn_model.py` and `rnn_model.py`, and no config
mentions a 480-d embedding. The two files named in their abstract (`classification_module.py`,
`similarity_module.py`) are training *paradigms*, and `similarity_module.py` takes
`model: nn.Module` — architecture-agnostic. So on this code, "we matched a number Schach
published" is available and "we reproduced Schach" is not. That is a limitation to state,
not a result to abandon.

## The protocol matches exactly, and it was checked rather than assumed

**The config IS the paper's final configuration**, verified cell-by-cell against their
Table IV rather than trusted:

| parameter | their Table IV | `winner_similarity_model.yaml` | |
|---|---|---|---|
| GRU layers | 3 | 3 | match |
| GRU layer size | 450 | 450 | match |
| GRU dropout | 0.28 | 0.28468551172548395 | match |
| learning rate | 2e-5 | 0.00002115735684121537 | match |
| loss | ArcFace | ArcFaceLoss | match |
| embedding size | 192 | 192 | match |

**Subject reconstruction — IDENTITY, not just count.** `01_aggregate.py` selects
`total number of sessions == 2`; their Readme says to delete the folders of the last 8
players to match the paper's database. AVALON's raw copy has 76 players, 71 with two
sessions. Dropping the last 8 and re-selecting gives **63**, matching the config's
`15_fps-63_subjects-metric_learning_movement.hdf5` filename. A count agreement is not an
identity agreement, so the identities are recorded here for a later run to compare against
a list rather than a number:

- **the 63:** 2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63,64,65,66,67,68
- **dropped as "last 8":** 69,70,71,72,73,74,75,76
- **excluded for having != 2 sessions:** 1,6,21,33,54
- arithmetic closes: 76 − 8 − 5 = 63

Other protocol facts read off the config: 15 FPS, window 500 frames (**33.3 s**),
`body_relative_acceleration`, split **train 27 / validation 9 / test 27**, batch 400,
`max_epochs` 500 / `min_epochs` 100, coordinate system forward=x right=z up=y.

## THE GATE — a curve, not a point

Their paper reports results **only in figures**; Tables I–IV are related work and
hyperparameters, with no results table. The values below are the authors' own stated
figures in their prose, for the 27 **test** subjects. Gating a single cell can pass or fail
for reasons unrelated to correctness; a three-cell shape cannot.

| cell (enrollment / use-time) | published | our metric |
|---|---|---|
| all enrollment / 5 min | **99%** | `sequence_top_1_accuracy_5_mins` |
| 10 min / 5 min | **89%** | same metric, enrollment-limited |
| **1 min / 1 min** | **25%** | `sequence_top_1_accuracy_1_mins`, enrollment-limited |

**PASS requires all three:**
1. **Each cell:** the published value falls inside our own **measured seed spread**
   (mean ± spread over **3 seeds**). The tolerance is an observation, not a number invented
   in advance — this literature's own seed spreads run from 1.7 points to ±8, a range too
   wide to pick a tolerance out of.
2. **Ordering holds:** 5-min-full > 5-min-10min > 1-min-1min. A port landing 99/95/60 fails
   on shape even with every point inside a loose tolerance.
3. **Dynamic range reproduced:** full-enrollment minus 1-min/1-min is within the measured
   spread of the published 74-point gap.

**The 1-min/1-min cell is the discriminating one and is treated as primary.** 99% is at
ceiling and would pass almost any working port; 25% sits far from both ceiling and chance
(1/27 = 3.7%), so it is the cell that can actually fail, and it is where seed variance
should be largest.

**Test set, not validation.** Every checkpoint callback in `rnn_similarity.yaml` monitors
`.../validation/mean` — 9 subjects — while the paper's figures are the 27 test subjects, and
gallery size moves rank-1 hard. `test_after_training: False` ships as the default, so the
run as published never touches the set the paper reports. **It will be enabled explicitly
and the certificate will say so.**

## Deviations from the authors' environment, recorded because they specified none

`requirements.txt` is **entirely unpinned** and the code is PL 1.x-era: `configs/trainer/
default.yaml` passes `gpus: 1` and `auto_scale_batch_size`, both **removed in PyTorch
Lightning 2.0**. Their Dockerfile pins Python 3.8 and CUDA 11.4 but installs unpinned
requirements, so **their own Dockerfile no longer builds a working environment today**. Any
reproduction must therefore choose versions the authors did not specify, and every such
choice is recorded here:

| pin | value | why |
|---|---|---|
| python | 3.10.21 (uv standalone) | PL 1.9 supports ≤3.10; their 3.8 is unavailable via uv without a build |
| pytorch-lightning | 1.9.5 | last 1.x — `gpus=` and `auto_scale_batch_size=` still exist |
| torch | 2.0.1+cu118 | **oldest torch that drives Ada/sm_89**, and within PL 1.9's support. torch 1.x cannot run on a 4060 Ti at all |
| torchmetrics | 0.11.4 | contemporary with PL 1.9; later versions break its API |
| pytorch-metric-learning | 2.3.0 | provides `AccuracyCalculator` and `ArcFaceLoss` |
| numpy | <2 | numpy 2 breaks the 2023-era stack |
| **faiss-cpu** | latest | **substituted for `faiss-gpu`**, which has no wheel for this stack. Changes speed, not arithmetic. Coordinator approved the substitution on condition it is recorded — this is that record |
| wandb | `WANDB_MODE=offline` | no account created, no key entered |

**A reproduction on pins the authors did not give is a reproduction of the paper's
*protocol*, not of its *environment*.** If the gate fails, the pins are a candidate cause
and must be ruled out before concluding the port is wrong.

## Falsifier

If any cell's published value sits outside our measured seed spread, OR the ordering
breaks, the reproduction **FAILS** and is reported as failed. A failure is a real result at
this stage and will be reported as prominently as a pass.
