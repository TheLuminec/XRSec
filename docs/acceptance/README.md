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

Confirmed on `main` after the merge (`06f57e5`, code identity `bc521f7f8e`): identical to the branch run - largest gap 1.2e-4 (PanoSaliency), all others below 1e-6, Nymeria residue 4.9e-10 m median / 3.8e-8 m max.

## amplitude_baseline (2026-09-05)

Two training-free baselines recorded on every run beside the model, both over the
evaluation manifest and both valid under every encoding (GENERALISATION_PROPOSAL 9.14):
`position_lookup_auc`, the mean-position lookup on each window's *recorded* mean position,
and `amplitude_auc`, movement amplitude alone (float64 norm of the per-axis sd of position,
before standardisation, in the corpus's own units). `lookup_auc` keeps its old meaning - the
lookup on the windows as the model sees them - so no existing row changes meaning; on a
`dyn` row it is rounding residue that tracks amplitude and `position_lookup_auc` is the
static baseline it cannot be; on a `raw` row the two coincide to rounding. Pre-approved by
the Coordinator on five criteria (COORDINATION.md at 3a388cf), scripts and artefacts here:

| criterion | script / artefact | result |
| --- | --- | --- |
| 4. no training numerics touched | `amplitude_baseline_acceptance.py`, `amplitude_{raw,dyn}_{before,after}_cpu.json` | `evaluate()` on a raw (`f24f2a6c1e`) and a dyn (`cb0a7dd722`) checkpoint, CPU before and after, same script: `auc`, `eer`, `lookup_auc`, `lookup_eer` and every per-dataset model/lookup figure identical on `repr`. Raw: recorded-position lookup 0.74839091 against the old lookup 0.74839093, gap 1.5e-8. |
| 3. `amplitude_auc` reproduces 9.14's table | `amplitude_baseline_criteria.py corpora`, `amplitude_criteria_dyn10s_4096.json` | On the 4096-identity dyn 10 s checkpoint, seed-67 pairs, every corpus equals the 9.14 harness formula on `repr` and the table's seed-67 column at its precision (PanoSaliency 0.6633, NJIT 0.5913, ViewGauss 0.5644, VR_User_Behavior 0.5161, Head_and_Gaze 0.5373, Panonut360 0.5205, EyeNavGS 0.5190); the encoded-window `lookup_auc` reproduces that table's lookup column too. |
| 2. `position_lookup_auc` equals 9.10's xyz lookup on the same pairs | same script, `amplitude_criteria_dyn5s_419.json` | On the 5 s dyn checkpoint, seed-67 pairs, every corpus equals the 9.10 harness formula (standardised means, Euclidean, xyz) on `repr`, and lands on 9.10's table (three-seed means): Head_and_Gaze 0.867 vs 0.870, VR_User_Behavior 0.719 vs 0.719, ViewGauss 0.933 vs 0.933, NJIT 0.646 vs 0.653. |
| 1. raw row digit-identical | `amplitude_baseline_criteria.py recorded`, `amplitude_criterion1_raw_cpu.json` and the GPU form recorded at the merge | **Passed on the GPU** (`amplitude_criterion1_{raw,dyn4096}_gpu.json`): pooled `lookup_auc` digit-identical to the recorded row on both the raw row (`3ad3e4d5a085`, 0.7288151589359121) and the 4096-identity dyn row (`661054c98a12`, 0.5215709756503836), every per-dataset value at its recorded precision; the raw row's recorded-position lookup equals the old one to 2.6e-9. The re-scored model AUC differs from the recorded one by 4e-10 (raw) and 1.7e-6 (dyn) - cuDNN across runs, inside CLAUDE.md's device band. On CPU the same pooled lookup differed by 1.8e-9 (below). |
| 5. merge window | COORDINATION.md, Model Generalization's heading | merge commit `9277648`; the identity it produced on this machine was `72b8053ec2`, one of three values the same code had before the line-ending fix below. |

**What the gate caught, so the next reader learns it.** The first version standardised the
recorded window means with the checkpoint's `ChannelNormalizer`. On a `dyn` index that
normaliser's target-fit statistics are the *residual* spread, not the position spread, so
the lookup weighted the axes wrongly and read ViewGauss 0.889 against 9.10's 0.933 and NJIT
0.541 against 0.653 - a plausible-looking column that was not the 9.10 lookup. **The
position lookup's standardisation must come from the corpus's own recorded position frames
at index build, independent of any checkpoint or encoding** (`dataset.
standardised_window_mean_positions`), and the normaliser must never touch
`window_mean_positions` again. Pinned by
`test_the_recorded_means_under_dyn_are_standardised_like_the_raw_ones` (a dyn index and a
raw index of one corpus agree on them exactly, before and after normalisation) and
`test_the_recorded_means_are_standardised_on_the_corpus_at_build_and_the_normaliser_leaves_them`.

**Two facts a reader needs to interpret these artefacts.** (1) The pooled `lookup_auc` of
the raw row re-scored on CPU differs from its recorded GPU value by 1.8e-9 while every
per-dataset value reproduces at its recorded precision: one tie flip in a rank-averaged AUC
from cuDNN-versus-CPU float32 window means. CLAUDE.md records device differences of up to
7e-4 on this pipeline; a nine-decimal gap is arithmetic, not a result, and the digit-exact
form of criterion 1 is therefore taken on the device that wrote the row. (2) Training-time
evaluation pairs are drawn with `_seed_value(seed, 4)` and `mode=test` pairs with
`_seed_value(seed, 11)`: two different manifests by construction, so a figure from one never
reproduces a figure from the other, and the 9.10 and 9.14 harnesses (and these scripts) use
the training-time derivation. A pre-float64 `dyn` row cannot serve as a criterion-1
reference either: its encoded lookup moved with the residual fix (9.11), which is what the
`dyn_float64` re-baseline above records.

## code_identity_line_endings (2026-09-06)

`code_identity()` hashed raw file bytes, so identical code hashed differently by line
ending: **`4d243b05d0`** from the stored LF blobs, **`100bd18472`** from a clean CRLF checkout
(a fresh worktree), and **`72b8053ec2`** from the DESKTOP-C checkout, where
`model/extractors/_kinematics.py` alone sits on disk with LF. The three were reconstructed
byte for byte from HEAD's blobs (`git ls-files --eol` shows the odd file), which settled a
dispute in which each side had one of the values right and the mechanism half right: `100bd18472`
was not "different code" and not "the LF twin" - it was the same code, all-CRLF. Fix:
`results_log.digest_tree` normalises CRLF to LF before hashing (merge `20b67bd`);
`.gitattributes` then checks `*.py` out with LF everywhere (`bacb45a`), identity-neutral after
the fix and a third identity step if done before it.

Acceptance ruled by the Coordinator: the digest computed on the working tree must equal the
digest computed from git's stored blobs of the same commit. `code_identity_line_endings.py`
does both: on main after the merge, working tree **`8db420df4c`** = stored blobs
**`8db420df4c`**; the pre-fix commit (`1e3adf3`) maps to its LF value `4d243b05d0`, which is
how pre-fix rows relate to post-fix ones. Numerics-free by inspection (the digest's only
callers are the results logger and the sweep's resume key); 475 tests.

## Nymeria figures: scoring-only rows

No training run has ever had Nymeria in `data_dirs` or `test_dirs`, so until 2026-09-06 no
Nymeria figure in 9.11, 9.12, 9.14 or CLAUDE.md had a shard row - each came from a
scratchpad script scoring dyn checkpoints on CPU with three manifest seeds. `score_nymeria.py`
(repo root) now scores any checkpoint on Nymeria through the pipeline's own `SiameseDataset`
and `evaluate()` and appends a `mode=rescore`, `experiment=nymeria_rescored` row per
checkpoint (test manifest at `_seed_value(seed, 4)` from the checkpoint's own seed, as every
row's is; the earlier three-manifest means sit within about 0.003 of the single-manifest
row). Every quoted dyn checkpoint was re-scored that way on CPU the day the script landed.
