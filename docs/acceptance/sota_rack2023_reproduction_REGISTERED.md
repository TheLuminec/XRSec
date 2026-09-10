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

---

# AMENDMENT 1 — 2026-09-10, before any run

**Amended for a fact about the INSTRUMENT, not a measurement.** The test of legitimacy is
whether the fact could have been known without running the experiment: this one is a grep,
so it could. **The original registration above is left intact rather than edited** — a
registration whose history is not visible is not a registration.

## What changed: their published numbers are not reachable from their published code

Verified by grep on both this machine and AVALON independently:

| block | evidence |
|---|---|
| **no test path at all** | zero hits for `def test_step`, `def test_epoch`, `trainer.test`, `.test(` across `src/` and `run.py`. "test" appears **zero times** in `src/train.py`. `SimilarityModule` defines `training_step`, `validation_step`, `validation_epoch_end`, `predict_step` — nothing else |
| **use-time hardcoded** | `sequence_lengths_minutes=[5, 10, 15]` is a constructor argument at `similarity_module.py:27`, not a config key. There is no 1-minute use-time |
| **one fixed enrolment** | `reference_embeddings = session_1_embeddings[::150]` at `similarity_module.py:99`. No enrolment sweep |

So the code computes: **validation split (9 subjects), one enrolment condition, use-times
5/10/15 min.** The paper's Figure 3 needs the test split, an enrolment axis and a 1-minute
use-time. **None of the three cells registered above is reachable from this code.** Their
*training* code is complete and faithful; their *evaluation* is not published.

"Their code is public" and "their numbers are reachable from their code" are different
claims, and the gap is invisible until someone greps for a test path.

## The amended approach: their model, their protocol, our plumbing

Train with their code, their config and their data — nothing of ours in the training path.
Evaluate with a harness written here implementing the protocol **they state in prose**: test
split, enrolment sweep, 5-minute and 1-minute use-time.

**This is a limitation on REPRODUCTION and an advantage on COMPARISON, in that order.** It
forecloses "we reproduced their published number". But this project's own rule is that a
gate must be one implementation rather than two, because two independently written
comparators can disagree for reasons having nothing to do with the thing compared. A SOTA
table scoring their model with their harness and ours with ours has exactly that defect, and
no reader could separate a model difference from a plumbing difference. **One harness
scoring both arms removes the largest confound in the comparison.**

## The harness is gated before it is trusted

Pointed at the **validation** split with their hardcoded `[::150]` enrolment and 5-minute
use-time, the harness must reproduce `sequence_top_1_accuracy_5_mins/validation/mean` as
*their own module computes it*. That gives the new plumbing a referent in **running code**
rather than in my reading of their prose — the same rule as "an out-of-path harness must
reproduce something from the column it will be compared against".

**No number from the harness on the test split is quotable until that validation gate is
green, and the gap will be reported before any test-split figure.**

## Rejected fallback, and why

Reporting what their code computes on the 9 validation subjects was considered and is
rejected. Not only because it matches no published number: **N=9 against N=27 is a different
task.** Gallery size moves rank-1 hard, so an N=9 figure printed beside published N=17 and
N=27 numbers would be actively misleading rather than merely uninformative. Quote no rank-1
without its N.

## Unchanged by this amendment

Three seeds with measured spread rather than an invented tolerance; the curve-shape
requirement (ordering and dynamic range, not a single point); the 1-min/1-min cell as the
discriminating primary once reachable; the 63 subject identities; and every environment pin
as a recorded deviation.

**And a note to turn on ourselves:** their Dockerfile pins Python 3.8 under *unpinned*
requirements, which is exactly how a repo stops building eighteen months later. Our own pins
should be audited the same way before we ship, rather than having a reviewer find it.

---

# Instrument notes — added 2026-09-10, still before any training run

## Their code runs UNMODIFIED on Python 3.8.20

Their Dockerfile's interpreter. All of `similarity_module`, `classification_module`,
`rnn_model`, `similarity_datamodule`, `window_dataset` and `accuracy_calculator` import
cleanly with no edits to their source. Choosing the interpreter rather than patching
`collections.abc` turned a deviation into a match.

## Their test suite: 17 passed / 3 failed — AND THE INVOCATION IS PART OF THE CLAIM

From the repo root the suite reads **12 failed / 8 passed**; from `tests/` it reads
**3 failed / 17 passed**. Nine failures are `FileNotFoundError` on a relative fixture path —
cwd dependence, not breakage. **"12 failed" would have been a false finding about their
repo, reported with complete confidence.** Record the invocation beside any test count.

The three real failures:

| test | cause |
|---|---|
| `window_dataset_test.py` | constructs `WindowDataHyperparameters` without `original_fps`, which their own dataclass requires. **Test out of date with source** |
| `bin_dataset_test.py` | `dataset_keys have to be of type 'list'` — same shape, bin path, not on our route |
| `brv_data_test.py::test_compute_velocities_simple` | **fixture artefact — investigated below, production path is correct** |

## The velocity failure: right mechanism, wrong fixture, production path clean

Symptom: only row 0 of the slice differs — expected `NaN` at a take's first frame, got real
values — so velocity differenced *across* the take boundary.

Mechanism: `velocities.values[invalid_frames, :] = np.nan` writes through `.values`, which
is a **view on a single-block frame and a copy on a multi-block one**. Their test builds the
frame from `np.random.randint`; the float assignment upcasts int64→float64 and splits the
manager into three blocks, so the boundary NaN is silently discarded.

| input | blocks | `.values` write lands |
|---|---|---|
| int64 (their test fixture) | 3 | **no** |
| float64 (real position data) | 1 | yes |

**My first test of this hypothesis refuted it, and the refutation was wrong** — I used a
float fixture, which exercises the code path perfectly and cannot trigger the behaviour.
Hence the rule: *a fixture must reproduce the conditions that trigger the behaviour, not
merely exercise the code path* — and a hypothesis refuted by such a fixture has not been
refuted.

**TWO sites share the pattern, and the untested one is also on our route** (Coordinator):

| site | function | data | test coverage |
|---|---|---|---|
| `helpers.py:175` | `compute_velocities_simple` | positions | their failing test |
| `helpers.py:214` | `compute_velocities_quats` | rotations | **none** |

Site 214 is the higher risk: it does `velocities[:] = np.nan` then **three separate `.loc`
assignments on column subsets**, one per joint, which is how a frame acquires extra blocks.
**Both verified BEHAVIOURALLY on realistic float64 data** — head plus both controllers,
three takes, `frame_step_size` 1 and 3 — by asserting the property itself (every take's
first `frame_step_size` rows must be all-NaN) rather than a block-count proxy:

```
input dtype float64, blocks 1
fss=1: take-boundary rows that should be NaN but are NOT: none
fss=3: take-boundary rows that should be NaN but are NOT: none
positions fss=1: boundary rows NOT NaN: none
```

**Still to do on the real corpus** rather than on a faithful synthetic: repeat the same
behavioural check inside the actual prep run.

## pandas is a CORRECTNESS pin, not just an install pin

Measured by the Coordinator on AVALON (pandas 3.0.1); this node runs **2.0.3**:

| pandas | frame | `.values` write |
|---|---|---|
| 2.x | float64 single block | lands — **our case, correct** |
| 2.x | multi-block | **silently discarded** — the latent bug |
| 3.x | any | **raises** `ValueError: assignment destination is read-only` (Copy-on-Write) |

So their code is correct on pandas 2 with float data, silently wrong on multi-block, and
**inoperable on pandas 3**. `pandas==2.0.3` joins the deviation list as load-bearing for
correctness rather than convenience.

**The sharper line for the paper:** the repo is runnable today only on a narrow,
now-unsupported interpreter **and** a superseded pandas major — stated as an observation
about reproducibility practice, not as criticism.

## Two things that went our way

Their **datamodule does support a test split** — `setup(stage="test")`, `test_dataloader()`,
and `return_frame_ids = True` set on test only (which is what an enrolment-limited analysis
needs, and nothing else in the shipped code uses). Only the *module* lacks `test_step`, so
the harness reuses more of their code than expected.

Their split uses `np.random.seed` + `np.random.shuffle` — the **legacy RandomState**, whose
stream numpy freezes — so the 27/9/27 partition is reproducible across numpy versions. The
exact mirror of the `Generator` hazard that bit the cross-machine gate, and by luck rather
than design.

## Corroboration that their evaluation code is genuinely absent

`tests/test_dml_mean_std_references.py:4` imports
`analysis.dml_paper.computations.metrics_computation_helper.compute_mean_std_reference_and_query_data`
— and **no `analysis/` package exists anywhere in the release**. A test referencing a
missing module by name, whose name is gallery/probe statistics for the DML paper, is
independent evidence for the missing-evaluation finding rather than an inference from the
absence of `test_step`.
