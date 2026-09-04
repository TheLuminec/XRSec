# Generalising to unseen datasets

Status: **proposal, measured where it could be measured without the GPU.** Written
2026-09-03 on DESKTOP-C from the sample cache and existing checkpoints; no training run was
launched. Every number below is reproducible from the scripts named in the appendix.

The question asked was "what can we do to create a model that generalises to unseen
datasets". The short answer is in section 0. Sections 1-3 are the measurements that
produced it; sections 4-7 are the plan.

---

## 0. The answer in ten lines

1. **A third of the corpus is not head pose.** PanoSaliency, Panonut360, Head_and_Gaze's
   V1 files and 360_em store a *unit viewing-direction vector* in `HmdPosition` (norm
   exactly 1.000 on every row). The model has been consuming orientation labelled as
   position. On those datasets it scores at chance, in domain, every fold.
2. **What the pooled model has learned is the static head position.** A three-number
   lookup - each window's mean position, standardised with the checkpoint's own
   training-fitted statistics - matches the trained model on the same held-out users and
   pairs: **0.726 vs 0.723 AUC** over five folds, and per dataset it wins on five of seven.
   On seated, static corpora the network is not adding anything on top.
3. **It therefore transfers exactly as far as that lookup does.** First unseen-dataset
   measurement in this project: the 7-dataset `identity_softmax` model on all 76
   who_is_alyx users scores **0.566 AUC** (five checkpoints, sd 0.006). The static lookup
   on the same pairs scores **0.593**. The same model family *with* alyx identities in
   training scores **0.725** on held-out alyx users, so being unseen costs about 0.16 AUC
   there. Every label-free adaptation tried - embedding centring, CORAL, donor statistics,
   per-session standardisation - is within 0.02 of doing nothing or worse.
4. **The capture frame differs per corpus in ways per-channel standardisation cannot fix**:
   yaw reference (+Z, +X, -X or none), NJIT's orientation rotating about Z while its
   position is Y-up (a parser slip), Nymeria Z-up, native rates 10-250 Hz.
5. So "generalise to unseen datasets" decomposes into two separate problems that need
   separate answers: a **static branch** that is explicit, cohort-normalised and needs no
   learning, and a **dynamics branch** that has to be forced off the static cue by
   construction, trained at BOXRR scale, and beaten against the lookup per dataset.
6. The evaluation has to change before any of that is run: per-dataset metrics with a
   semantics tier, the static probe and the random control recorded on every run, and the
   target-fit normalisation declared rather than falling back silently.

**Measured the following night (section 9).** Trained on BOXRR + alyx and tested on
the seven held-out corpora: the `raw` model transfers below the lookup on every tier-1
corpus and is flat to three decimals in identity count (0.672 / 0.672 / 0.671 at 419 /
1000 / 2096). The dynamics-only branch transfers a small signal (+0.02 to +0.07 over
chance on tier 1) that rises with identity count to 1000 and then flattens, and it
reaches what in-domain training on those corpora reaches. Movement alone identifies
unseen Beat Saber players at about 0.80 and unseen alyx players at 0.53, in domain or
out: the behavioural signal is activity-bound. Fusion and yaw canonicalisation do not
help transfer.

---

## 1. What the position channel actually contains

Audited from the cached 5s@20Hz windows and then confirmed on raw CSVs, 60 users per
dataset (all users where fewer). The Coordinator reproduced the norms independently on
the laptop's copies.

| dataset | `|HmdPosition|` per row | quaternion | up axis | yaw reference (concentration) | mean pos xyz (m) | between-user sd | within-window sd | dup frames | **tier** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BOXRR-23 | 1.64, real | real | Y | +Z (0.98) | 0.00, 1.58, -0.05 | 0.05, 0.13, 0.18 | 0.045, 0.016, 0.021 | 0.3% | 1 |
| who_is_alyx | 1.62, real | real | Y | none (0.33) | -0.07, 1.56, -0.05 | 0.22, 0.09, 0.12 | 0.05, 0.03, 0.05 | 6.6% | 1 |
| VR_User_Behavior | 1.16, real | real | Y | -X (0.56) | 0.03, 1.14, -0.22 | 0.19, 0.04, 0.16 | 0.014, 0.006, 0.016 | 8.3% | 1 |
| ViewGauss | 1.95, real | real | Y | +Z (0.96) | 0.25, 1.58, 0.39 | 0.28, 0.07, 0.29 | 0.020, 0.009, 0.010 | 50.5% | 1 |
| Head_and_Gaze V2 | 1.36, real | real | Y | +X (0.64) | -0.02, 1.27, 0.01 | 0.19, 0.05, 0.19 | 0.021, 0.012, 0.022 | 0.1% | 1 |
| NJIT_6DOF | 3.9, real, room-scale | **rotates about Z** | Y (position) | +Z (0.97) | 2.89, 1.57, 2.25 | 0.57, 0.06, 0.27 | 0.30, 0.015, 0.23 | 0.0% | 1, frame broken |
| EyeNavGS | 1.05 +- 0.26, scene units | real | Y | none (0.14) | -0.28, -0.14, 0.65 | 0.29, 0.26, 0.57 | 0.18, 0.08, 0.18 | 0.4% | 3 |
| PanoSaliency | **1.0000, sd 0** | **constant identity** | - | +Z (1.00, degenerate) | -0.20, 0.08, 0.25 | 0.26, 0.12, 0.20 | 0.19, 0.08, 0.18 | 67% | 2 |
| Panonut360 | **1.000, sd 0** | real, same content | Y | +X (0.37) | 0.36, -0.02, -0.08 | 0.16, 0.07, 0.13 | 0.22, 0.12, 0.24 | 0.4% | 2 |
| Head_and_Gaze V1 | **1.0000, sd 0** | absent | - | - | - | - | - | - | 2 |
| 360_em | **1.0000, sd 0** (Data agent) | absent | - | - | - | - | - | - | 2 |
| Nymeria | real (Data agent) | real | **Z, rotated on conversion** | - | - | - | - | - | 1 |

"Up axis" is from rotating the head's local up vector (0,1,0) by each quaternion and
averaging; "yaw reference" is the same for the forward vector (0,0,1), with the norm of
the mean as concentration (1 = everyone faces the same way all the time). "dup frames" is
the fraction of consecutive 20 Hz frames that are bit-identical, i.e. the native-rate
artefact.

Provenance, from the parsers deleted in `6421567` (`git show 6421567^:datasets/<name>/parser.py`):

- **Head_and_Gaze V1**: `HmdPosition` is computed in the parser as
  `(sin yaw cos pitch, sin pitch, cos yaw cos pitch)` from an equirectangular
  `Pose_Point`. A direction vector by construction.
- **Panonut360**: the parser copies the source's `head_x/y/z`, which are already unit
  vectors; the quaternion is copied alongside, so both channels encode the same thing.
- **NJIT**: the source is `[x, y, z, yaw, pitch, roll]` with y = height (1.5 m, so Y-up),
  but the parser builds the quaternion with `R.from_euler('ZYX', [yaw, pitch, roll],
  degrees=True)`, which applies yaw about **Z**. Orientation and position are in
  different frames. The degrees assumption was never checked either. Parser slip; the
  repair is the axis order, on whichever machine holds the raw `.mat` files.

**Tiers, as decided by the Coordinator with the user's design in mind:**

| tier | meaning | datasets |
| --- | --- | --- |
| 1 | true head pose in metres | BOXRR, alyx, VR_User_Behavior, ViewGauss, Head_and_Gaze V2, NJIT (after repair), Across-XR, Nymeria |
| 2 | direction vector labelled as position | PanoSaliency, Panonut360, Head_and_Gaze V1, 360_em |
| 3 | real position, not a human head in metres | EyeNavGS (virtual camera, scene units) |

The primary transfer claim is measured on tier 1 only. Tier 2 is reported beside it as
"schema our model does not consume", not as failures. Do not synthesise quaternions from
direction vectors: that manufactures a roll that is not in the data.

Two consequences for things already written down:

- "Four of eight datasets have no absolute head height" (CLAUDE.md) understates it: three
  of them have no position at all. `channels=position` on Head_and_Gaze mixes unit vectors
  (V1) with metres (V2) for the same users, and the 360_em "recovery" recovered direction
  vectors.
- The 78% anthropometry figure was measured on a pooled corpus where a third of the
  identities carry no position. On tier-1 data alone the anthropometric share is higher,
  not lower.

---

## 2. What the pooled model has learned: per-dataset held-out AUC against a lookup

Five fold checkpoints of sweep `b732bee5c6` (`bilstm`, `identity_softmax`, 7 datasets,
343 identities, cross-session positives, per-dataset normalisation, validation-selected;
each records its own held-out users). Re-scored on CPU from the cache, pair manifests
regenerated with seed 67, within-dataset negatives, 512 pairs per user, positive fraction
0.500 throughout. The lookup is each window's mean position (3 numbers), standardised
with the **same training-fitted per-dataset statistics the checkpoint carries** (so it is
handed nothing the model was not), compared by Euclidean distance. Nothing is trained for
it. A first pass that fitted the statistics on the held-out cohort instead gave 0.730
pooled; the two agree because the per-channel mean cancels in a difference and only three
per-axis scales change. The random control is the `random` extractor from sweep
`1cba61edba`, five folds, scored the same way.

| held-out dataset | tier | users / fold | **trained model** AUC | **mean-position lookup** AUC | model EER | random control |
| --- | --- | --- | --- | --- | --- | --- |
| ViewGauss | 1 | 7 | **0.938** +-0.019 | 0.932 +-0.041 | 0.127 | 0.506 |
| Head_and_Gaze (V2) | 1 | 20 | **0.898** +-0.023 | 0.868 +-0.021 | 0.177 | 0.499 |
| VR_User_Behavior | 1 | 10 | 0.689 +-0.039 | **0.716** +-0.044 | 0.367 | 0.503 |
| NJIT_6DOF | 1 | 4 | 0.611 +-0.048 | **0.674** +-0.079 | 0.421 | 0.503 |
| PanoSaliency | 2 | 20 | 0.580 +-0.027 | 0.583 +-0.031 | 0.446 | 0.501 |
| Panonut360 | 2 | 4 | 0.504 +-0.013 | 0.508 +-0.013 | 0.498 | 0.505 |
| EyeNavGS | 3 | 4 | 0.490 +-0.011 | 0.493 +-0.010 | 0.508 | 0.486 |
| **pooled** | | 68 | **0.723** +-0.014 | **0.726** +-0.017 | 0.332 | 0.500 |

Reading it:

- **The pooled 0.72-0.74 headline is an average of near-perfect verification on two
  datasets and chance on three.** ViewGauss and Head_and_Gaze V2 have within-window
  position spread of 1-2 cm against between-user spread of 5-30 cm: where a person sits is
  a fingerprint, and the model finds it. On the direction-vector datasets it finds nothing.
- **The model does not beat three numbers.** It wins on the two easiest corpora by
  0.01-0.03 and loses on the other five; over the pooled folds the difference is -0.003
  against a fold sd of 0.014-0.017. This is a stronger statement than the CLAUDE.md
  "trivial descriptor" comparison, which used a 14-number descriptor, accuracy at a
  fitted threshold, and VR_User_Behavior alone. With AUC and the right three numbers,
  the learned model is a static position lookup to within noise.
- Two more static probes on the same folds: mean quaternion alone 0.540 pooled (0.79 on
  ViewGauss), and per-channel **std** alone 0.546 pooled but **0.653 on PanoSaliency** -
  the spread of viewing direction within five seconds separates those users better than
  anything the model extracts from them. That is a dynamics cue sitting in the data that
  the trained model is not using.
- Held-out embeddings are moderately dataset-clustered: between-dataset variance of the
  user means is 0.24 against 0.27 between users within a dataset, and a nearest-centroid
  dataset classifier on held-out windows reaches 0.38 against 0.32 majority chance. The
  AM-softmax denominator spans every training identity across datasets, so the objective
  does reward knowing the corpus; evaluation pairs are within-dataset, so it is not a leak
  at test time, but it is capacity spent on the wrong thing.

### 2b. Where a learned component does exist, in domain

The same probe against the 8-dataset model (sweep `31751868df`: alyx included in
training, margin 0.1, scale 15, 30 epochs, five folds, each recording its held-out users),
so alyx now has an in-domain row:

| held-out dataset | tier | model AUC | lookup AUC | model - lookup |
| --- | --- | --- | --- | --- |
| **who_is_alyx** | 1 | **0.725** +-0.073 | 0.589 +-0.026 | **+0.136** |
| PanoSaliency | 2 | 0.672 +-0.035 | 0.583 +-0.031 | **+0.089** |
| Head_and_Gaze (V2) | 1 | 0.903 +-0.017 | 0.869 +-0.019 | +0.034 |
| Panonut360 | 2 | 0.539 +-0.016 | 0.508 +-0.013 | +0.031 |
| ViewGauss | 1 | 0.896 +-0.050 | 0.938 +-0.036 | -0.042 |
| VR_User_Behavior | 1 | 0.679 +-0.029 | 0.717 +-0.041 | -0.038 |
| NJIT_6DOF | 1 | 0.662 +-0.067 | 0.694 +-0.083 | -0.032 |
| EyeNavGS | 3 | 0.510 +-0.008 | 0.493 +-0.010 | +0.017 |
| pooled | | 0.741 +-0.022 | 0.701 +-0.014 | +0.040 |

The learned component is real, and it appears exactly where people **move**: alyx (walking
an FPS, within-window position sd 5 cm) and PanoSaliency (viewing direction sweeping
within five seconds; its `std_only` lookup was already 0.65). On the seated, still corpora
the model is the lookup or slightly worse. So the model learns movement where movement
exists, and where people sit still it learns where they sit. Section 3 shows that the
movement part is what fails to transfer.

---

## 3. The first unseen-dataset measurement

The same five checkpoints never saw who_is_alyx. Scored on all 76 alyx users, cross-day
positives (both sessions of every user are on different days), within-dataset negatives,
38,912 pairs, positive fraction 0.500. Five label-free ways of bringing an unseen corpus
into the model's frame, plus the lookup:

| how alyx was brought into the training frame | AUC (5 ckpts) | EER |
| --- | --- | --- |
| per-dataset statistics fitted on alyx itself (the pipeline's current fallback) | **0.566** +-0.006 | 0.455 |
| + subtract the alyx mean embedding | 0.568 +-0.007 | 0.454 |
| + CORAL (align embedding covariance to in-domain held-out embeddings) | 0.547 +-0.007 | 0.468 |
| statistics borrowed from VR_User_Behavior | 0.537 +-0.004 | 0.474 |
| statistics borrowed from ViewGauss | 0.517 +-0.002 | 0.489 |
| no normalisation (metres straight in) | 0.531 +-0.003 | 0.478 |
| per-session statistics (corpus-agnostic, removes absolute position) | 0.504 +-0.001 | 0.498 |
| **mean-position lookup, target-fit statistics** (no model) | **0.593** | |
| random extractor, target-fit statistics (5 folds) | 0.496 +-0.006 | |
| **in-domain reference**: 8-dataset model with alyx in training, scored on its ~15 held-out alyx users per fold (sweep `31751868df`, margin 0.1 / scale 15, epochs 30, 5 folds) | **0.725** +-0.073 | |
| lookup on those same held-out alyx users | 0.589 +-0.026 | |

Because gallery composition moves numbers here, the pair was then re-scored on
**identical users**: the 7-dataset checkpoint of fold k on exactly the 15-16 alyx users
the 8-dataset fold k held out, same manifest for all three rows:

| same 15-16 held-out alyx users per fold | AUC (5 folds) |
| --- | --- |
| 7-dataset model, alyx unseen | 0.566 +-0.011 |
| 8-dataset model, alyx in domain | **0.731** +-0.075 |
| mean-position lookup | 0.595 +-0.018 |
| paired, in domain minus unseen | **+0.164**, t(4) = 4.54, 5/5 folds |

The Coordinator caught that those two checkpoints differ in **three** things - dataset
coverage, `identity_margin` (0.1 against 0.35) and epoch budget (30 against 20) - so the
pair was re-run clean: the 8-dataset corpus at margin 0.35 and 20 epochs, i.e.
`b732bee5c6`'s exact configuration plus alyx in training (sweep `5ed2089354`, 5 stratified
folds, which partition the users identically). **One variable now differs, whether alyx
was trained on:**

| same 15-16 held-out alyx users per fold, same margin, same epochs | AUC (5 folds) |
| --- | --- |
| 7-dataset model, alyx unseen | 0.566 +-0.011 |
| 8-dataset model, alyx in domain | **0.749** +-0.050 |
| mean-position lookup | 0.595 +-0.018 |
| paired, in domain minus unseen | **+0.183**, t(4) = 7.52, 5/5 folds |

Three conclusions:

1. **Being unseen costs about 0.18 AUC on alyx**: 0.566 unseen against 0.749 in domain
   on the same users, with the lookup at 0.595 for both. In domain the model adds +0.15
   over the lookup; unseen it adds -0.03. The learned component exists and does not
   travel. The honest one-sentence statement is: *the model generalises exactly as far as
   the static cue does, and the part it actually learns is the part that does not survive
   a change of corpus.*
2. **Label-free adaptation of the embedding does nothing here.** Centring, CORAL, donor
   statistics: all within 0.02 of the plain fallback or worse. Same verdict as AS-Norm.
   Consistent with section 2: there is no dataset-specific *rotation* of a dynamics
   embedding to correct, because the embedding is mostly a position lookup and the only
   thing an unseen corpus changes is what "position" means there.
3. **The lookup transfers better than the model** (0.593 vs 0.566). Whatever the model
   learned beyond position on the seven corpora is slightly *negative* on the eighth.

---

## 4. What this means for the agreed target claim

The design agreed with the user: **train on BOXRR-23 + who_is_alyx, test on the existing
datasets held out entirely, and measure an identity-count curve (419 / 1000 / 2439 / 4439
training identities) against that fixed heterogeneous test set.** Read two ways: held-out
BOXRR users (within-domain, will flatter) and transfer to our datasets (the number that
matters).

Predictions registered before any of it runs, so they cannot be rationalised afterwards:

| held-out dataset | tier | prediction for a BOXRR+alyx model, target-fit statistics |
| --- | --- | --- |
| ViewGauss, Head_and_Gaze V2, VR_User_Behavior | 1 | roughly the lookup's level (0.72-0.92): these are static-position datasets and z-scoring within the cohort makes "relative height and seat" comparable across corpora. The curve over identity count will be **flat** here: the lookup has no training set. |
| NJIT | 1 | below the lookup until the parser is fixed; its quaternion frame is one no training corpus shares. |
| PanoSaliency, Panonut360, HG-V1, 360_em | 2 | chance, at every identity count. Not a generalisation result in either direction. |
| EyeNavGS | 3 | chance. |
| held-out BOXRR users | 1 | high, and rising with identity count, because BOXRR has the strongest static cues in the corpus (mean quaternion alone scores **0.785**, mean pose **0.811**, on 150 BOXRR users with no model). A large within-domain gain will be partly that. |

If those predictions hold, the headline curve says "more identities do not help transfer",
and the reason is not the model: it is that the *transferable* part of the current signal
is a static cue that needs no training data, and the trainable part has never been
isolated. The plan below is built to isolate it.

---

## 5. The plan

### 5.1 First: make the evaluation say what it measures (no GPU, one to two days)

Everything else is uninterpretable without these. In priority order:

1. **Per-dataset metrics on every run.** `evaluate()` has the anchor user and
   `SampleIndex.user_dataset_ids`; group scores by dataset and record
   `test_auc_by_dataset` / `test_eer_by_dataset` in history and the results log. Section
   2's table should be a by-product of every sweep, not a CPU script.
2. **A semantics tier per dataset**, carried in the results row, so a pooled number over
   tiers can be caught by a reader. A one-line `DATASET_TIERS` map in `dataset.py` plus a
   printed warning when tiers are pooled.
3. **The static lookup as a recorded baseline column**: mean-position AUC on the exact
   evaluation manifest, computed inside `evaluate()` from the normalised windows. Costs
   nothing. Any model number is read as "model minus lookup" per dataset.
4. **The random extractor in every grid** (already project practice; keep it).
5. **`eval_normalize` as an explicit config key** (`target_fit` | `none` | `session`)
   replacing the silent WARNING fallback in `ChannelNormalizer.transform`. The fallback is
   the right default for an unseen corpus, but the results row must say it was used.
6. **Subsample folds for the identity-count curve.** With the test set fixed, fold
   variance comes only from which training identities were drawn. At 419 identities,
   five disjoint BOXRR draws fit inside 2020; at 1000, two; at 2439+, seeds only. Record
   `max_users` per dataset so alyx is not subsampled to 15 users at the low end
   (`max_users` is currently one integer apportioned proportionally).
7. Done already: the `max_users` + `test_dirs` filter bug (`a681a6d`), the leakage guard
   (`9204036`), Nymeria's Z-up rotation.

### 5.2 Data layer: one frame for every corpus

| change | what it fixes | cost |
| --- | --- | --- |
| **Gravity-preserving yaw canonicalisation** - rotate each window (or session) about the up axis so its mean forward vector is +Z; new `encoding=yawc` beside `raw`. Keeps pitch, roll and height, removes only the content-driven yaw reference. | +Z / +X / -X / none yaw references across corpora (section 1), which per-channel standardisation cannot undo. | ~60 lines in `input_encoding.py`, 5-fold arm |
| **NJIT parser repair** (`YXZ`, verify degrees) | orientation in a frame no other corpus shares | whoever holds the raw `.mat` files |
| **Frame-hold augmentation** - at train time, resample a fraction of windows as if captured at 8-16 Hz then nearest-upsampled | ViewGauss 50%, PanoSaliency 67% duplicate frames, a per-corpus artefact a BOXRR-trained model never sees | ~30 lines, GPU-side, cheap |
| **Position-dropout augmentation** - zero the position channels (post-normalisation) for a fraction of training windows, with a presence flag | 3-DoF AR glasses exist and report no position; tier-2 datasets have none; the model must not read "zero position" as "average height" | ~20 lines, one arm |
| **Domain-balanced identity sampling** - `balance_identities=cap` already equalises identities; add a per-dataset weight so alyx's 76 are not 3.6% of every epoch beside BOXRR's 2020 | the "few activities" half of the claim is otherwise one activity | ~20 lines |

Not proposed: `br`/`brv`/`bra` as they stand. They remove absolute position, which lost
0.13 AUC because position *is* the signal; the static branch below handles that cue
explicitly instead of throwing it away.

### 5.3 Model: separate what transfers for free from what has to be learned

```
                       window (7 x T), per-dataset z-scored
                                   |
              +--------------------+---------------------+
              |                                          |
      STATIC BRANCH (no learning)                DYNAMICS BRANCH (learned)
      mean position, mean orientation            input = window with mean position
      relative to the target cohort              subtracted AND orientation expressed
      -> Euclidean/cosine score s_static         relative to the window's mean heading,
      reported per tier                           gravity preserved (`encoding=dyn`)
                                                 -> bilstm / tdnn, identity_softmax
                                                 -> cosine score s_dyn
              |                                          |
              +---------------- late fusion -------------+
                     w_static * s_static + w_dyn * s_dyn
                     (two scalars fitted on validation users, in domain)
```

Why this shape and not a bigger network:

- **The static branch already transfers as well as anything measured** (section 3) and
  is fully interpretable: "this person's head sits 1.2 sd above the cohort mean". It has
  no training set, so it cannot inflate with identity count, and it is honest about what it
  is - an anthropometric cue that exists only where the corpus records real position.
- **The dynamics branch is where the generalisation claim lives.** Trained with the static
  cue removed *by construction* rather than by hoping the network ignores it, its
  per-dataset score against the lookup is the number that answers "did the model learn
  something about how this person moves". The 48-vs-343 result says this component is
  identity-count-limited; BOXRR at 2020-4020 identities is the first corpus large enough
  to find out whether it plateaus, and this is the only branch on which the identity-count
  curve is a meaningful experiment.
- **`encoding=dyn` is not `center_position`.** Centring left absolute orientation in, and
  the mean-quaternion probe shows orientation alone is 0.54 AUC pooled and 0.79-0.81 on
  ViewGauss and BOXRR of static posture. CLAUDE.md already says the centred arm "still
  contains absolute quaternion, so 22% is an upper bound on the purely behavioural share";
  section 2 measures how loose that bound is, and it is very loose. So `dyn` is not a
  refinement of `center_position`, it is the first honest version of the experiment the
  22% figure was quoted from. The dynamics input has to remove mean heading as well,
  keeping gravity so pitch/roll *dynamics* survive.
- Fusion weights are two scalars fitted on in-domain validation users, so the reported
  test number never sees the target corpus's labels.

### 5.4 Objective

- **Dataset-masked AM-softmax**: restrict each window's softmax denominator to
  identities from its own dataset (`-inf` on other-dataset logits). The identity-objective
  analogue of `within_dataset_negatives`, which was worth +4.4 on the pair path. Prediction:
  small within-domain change either way; the point is to stop spending capacity on
  "which corpus" (section 2's clustering figures) before adding more corpora.
- Keep `identity_margin` / `identity_scale` at whatever the running sweep selects.

### 5.5 What not to spend runs on

| idea | why not | evidence |
| --- | --- | --- |
| label-free embedding adaptation (centring, CORAL, GOPA-style) | measured at zero on alyx, five checkpoints | section 3 |
| AS-Norm | measured at zero | CLAUDE.md |
| extractor architecture | three backbones within 0.002 | CLAUDE.md |
| DANN / adversarial domain confusion | with two or three source domains the dataset classifier is trivial; and the thing to be invariant to (static frame) is better removed at the input than adversarially | section 5.3 |
| more window length | +0.02 AUC, saturating | CLAUDE.md |

---

## 6. Runs, in order, with what each one decides

Per-run cost on the 5060 Ti at 419 identities is about 5 minutes; at 2020 BOXRR
identities expect 15-25 (window count 315k vs 217k, and the AM-softmax head scales with
classes). All arms carry the `random` control and are read per tier-1 dataset against the
lookup column.

| # | run | arms x folds | decides |
| --- | --- | --- | --- |
| 0 | **clean alyx pair**: the 8-dataset corpus at margin 0.35 / 20 epochs, 5 stratified folds, per-dataset metrics recorded | 1 x 5 | **done** (sweep `5ed2089354`): in domain 0.749 vs unseen 0.566 on the same users, +0.183, t(4) = 7.52 - section 3 |
| 1 | **baseline transfer**: BOXRR+alyx -> the seven, `raw`, target-fit stats | 5 seeds at 419, 2 at 1000, 1 at 2096, + controls | **done** (9.1): 0.672 / 0.672 / 0.671, lookup 0.727, below the lookup on every tier-1 corpus. Tier 2 at chance is a statement about our schema, not about those datasets |
| 2 | **frame**: `yawc` vs `raw`, seed-paired | 2 x 5 | **done** (9.2): +0.025 / +0.015 on the +X / -X corpora as predicted, -0.03 on the +Z ones, net +0.003, not resolved |
| 3 | **dynamics branch alone**: `dyn` encoding, BOXRR+alyx -> the seven | 5 at 419, 1 at 2096 | **done** (9.3): 0.581 +-0.002, tier 1 at 0.52-0.54, all above the floor; censored, and the long-budget rerun shows the transfer figure had plateaued anyway |
| 4 | **identity-count curve on the dynamics branch**: 419 / 1000 / 2096, `epochs=120`, patience 15 | 5 + 2 + 1 | **done** (9.3): 0.582 / 0.600 / 0.598 against the lookup's flat null curve; +0.03 / +0.05 on Head_and_Gaze / ViewGauss from 419 to 2096 |
| 5 | **fusion**: static + dynamics | scoring only | **done** (9.4): retired for transfer; no weight without target labels beats the lookup on tier 1 |
| 8 | **`dyn` in domain**: the 8-dataset corpus, 5 stratified folds, `epochs=60`, patience 15, random control | 2 x 5 | **done** (9.5): alyx 0.530 in domain, seated corpora 0.53-0.55, PanoSaliency 0.73; the activity, not the share |
| 6 | augmentation arms: frame-hold, position-dropout, domain balance | 3 x 5 | not run: run 3's signal is +0.02 to +0.04, too small for a +0.00 to +0.02 arm to be resolvable |
| 7 | masked softmax | 1 x 5 | not run; capacity, not transfer |

Power: paired 5-fold resolves 0.011-0.041. Runs 1-3 are looking for effects of 0.05 or
more or for a floor test, so they are affordable; run 6 is at the edge and should be read
as "not resolved" unless it clears 0.02 at t > 2.8.

---

## 7. Predictions, registered

| claim | prediction | what would falsify it | outcome (section 9) |
| --- | --- | --- | --- |
| BOXRR+alyx `raw` model transfers to tier-1 datasets at about the lookup's level | model - lookup within +-0.03 per dataset | model beats lookup by > 0.05 on two or more tier-1 datasets | **held, harder**: -0.01 to -0.12, below the lookup everywhere |
| identity count does not move `raw` transfer | curve flat within fold sd | monotone rise > 0.03 from 419 to 2020 | **held**: 0.672 / 0.672 / 0.671 |
| `dyn` branch on BOXRR+alyx transfers above chance to tier-1 | 0.53-0.58 AUC at 2020 ids, ~chance at 419 | at chance at 2020 | **held**: 0.52-0.54 at 419 (above the floor, sd < 0.01), 0.51-0.58 at 2096 |
| `dyn` branch rises with identity count | +0.02 to +0.05 from 419 to 2020 | flat | **held**: +0.016 pooled (all of it between 419 and 1000), +0.03 / +0.05 on Head_and_Gaze / ViewGauss, long budget, uncensored for transfer |
| `yawc` helps the +X / -X corpora | +0.01 to +0.04 on Head_and_Gaze V2 and VR_User_Behavior | no change | **held** (+0.025, +0.015) but offset by losses on the +Z corpora; net zero |

---

## 8. Open items

- **Who repairs NJIT** (raw data location) and whether Head_and_Gaze V1 / 360_em are worth
  a `channels=orientation` mode later. Not now; identity count is no longer the constraint.
- **Across-XR (49 users, 5 apps)** is the ideal cross-*activity* test at fixed users and
  fixed rig, once its download clears the WAF block. Nymeria (real glasses, Z-up, one
  sitting) is the ideal cross-*device* test. Both belong in the tier-1 test set, never in
  training.
- Whether to commit the analysis scripts (appendix) under the repo so section 2 and 3 are
  re-runnable by others.

---

## 9. Results of the first night of runs (2026-09-04)

Runs 0-4 of section 6, on DESKTOP-C, all `bilstm`, `identity_softmax`, 5s@20Hz,
cross-session positives, per-dataset normalisation, validation-selected on in-domain
users, `eval_normalize=target_fit` on the held-out corpora, 512 pairs per user, positive
fraction 0.500. Training corpus BOXRR + alyx; held-out corpora the seven. "419" is 343
BOXRR + all 76 alyx, "full" is 2020 + 76. Seeds 1-5 are independent BOXRR subsamples,
so the spread is the subsample-fold spread. Random control at full corpus: 0.498 pooled,
0.494-0.508 per dataset.

### 9.1 Transfer of the `raw` model: at or below the lookup, flat in identity count

| held-out dataset | tier | model, 419 ids (5 seeds) | 1000 ids (2 seeds) | full corpus (2096) | lookup | model - lookup |
| --- | --- | --- | --- | --- | --- | --- |
| ViewGauss | 1 | 0.911 +-0.008 | 0.906 | 0.900 | 0.934 | -0.02 / -0.04 |
| Head_and_Gaze V2 | 1 | 0.750 +-0.008 | 0.753 | 0.749 | 0.869 | **-0.12** |
| VR_User_Behavior | 1 | 0.638 +-0.007 | 0.647 | 0.629 | 0.714 | **-0.08** |
| NJIT | 1 | 0.648 +-0.013 | 0.635 | 0.641 | 0.653 | -0.01 |
| PanoSaliency | 2 | 0.581 +-0.007 | 0.585 | 0.576 | 0.581 | 0.00 |
| Panonut360 | 2 | 0.526 +-0.004 | 0.533 | 0.517 | 0.512 | +0.01 |
| EyeNavGS | 3 | 0.502 +-0.004 | 0.499 | 0.507 | 0.492 | +0.01 |
| **pooled** | | **0.672 +-0.003** | **0.672** | **0.671** | **0.727** | **-0.06** |

Both pre-registered predictions for this run held, one of them harder than predicted:

- **Identity count does not move `raw` transfer.** 419, 1000 and 2096 training
  identities: 0.672, 0.672, 0.671 pooled, and no tier-1 dataset moves by more than
  0.01. The lookup has no training set, and the model behaves like the lookup.
- **The model transfers below the lookup on every tier-1 corpus**, by 0.12 on
  Head_and_Gaze and 0.08 on VR_User_Behavior. In domain (section 2b) the same
  architecture beats the lookup on Head_and_Gaze by +0.03. So what a BOXRR-trained model
  learns about *which position channels matter* is Beat-Saber-specific, and it is worse
  than treating the three axes equally when the corpus changes. Validation on in-domain
  users selects epoch 2-3 of 30 every time.
- The two tier-2 corpora and EyeNavGS sit at the lookup, which is at chance: the schema
  statement of section 4, not a generalisation result.

### 9.2 `yawc`: right where predicted, wrong where not, and unresolved overall

Seed-paired against `raw` at 419 identities:

| dataset | yawc - raw | t(4) | won |
| --- | --- | --- | --- |
| Head_and_Gaze (yaw reference +X) | **+0.025** | 3.01 | 4/5 |
| VR_User_Behavior (yaw reference -X) | **+0.015** | 2.37 | 5/5 |
| ViewGauss (+Z, like BOXRR) | -0.035 | -1.29 | 2/5 |
| NJIT (+Z, broken quaternion frame) | -0.025 | -0.90 | 2/5 |
| pooled | +0.003 | 0.32 | 3/5 |

The prediction was +0.01 to +0.04 on the +X / -X corpora and ~0 on ViewGauss. The first
half held exactly; the second did not - `yawc` costs something on the corpora that
already shared BOXRR's heading, and it triples the seed spread (0.022 against 0.003).
Net zero, and per the power table this is "not resolved". Rotating the *within-window*
displacement into the heading frame is the likely cost: it removes a static horizontal
cue (which way the seat faces relative to the room) that the +Z corpora share with
BOXRR. Not worth an arm on its own; worth keeping as an option for a corpus whose
heading is known to differ.

### 9.3 `dyn`: a small dynamics signal that transfers, and a censored one

Every static cue removed at the input. The lookup on `dyn` windows is 0.506, chance by
construction, so the model's number is the whole signal. The `random` extractor at
419 identities under the same protocol: 0.500 pooled, 0.495-0.507 per dataset.

| held-out dataset | tier | dyn, 419 ids (5 seeds, 30 epochs) | t(4) vs 0.5 |
| --- | --- | --- | --- |
| Head_and_Gaze V2 | 1 | **0.538** +-0.002 | 39.7 |
| NJIT | 1 | 0.528 +-0.007 | 7.9 |
| VR_User_Behavior | 1 | 0.519 +-0.003 | 11.2 |
| ViewGauss | 1 | 0.517 +-0.008 | 4.4 |
| EyeNavGS (virtual camera) | 3 | 0.535 +-0.006 | 11.8 |
| Panonut360 | 2 | 0.516 +-0.004 | 7.6 |
| PanoSaliency | 2 | **0.724** +-0.005 | 84.4 |
| **pooled** | | **0.581 +-0.002** | |

All five of those runs selected epoch 29 or 30 of 30, so the arm was rerun with
`epochs=120` and `early_stopping_patience=15` (run 4), which is also where the
identity-count curve comes from:

| held-out dataset | tier | 419 ids (5 seeds) | 1000 ids (2 seeds) | 2096 ids (1 seed) | 419 -> 2096 |
| --- | --- | --- | --- | --- | --- |
| Head_and_Gaze V2 | 1 | 0.537 +-0.003 | 0.560 +-0.003 | **0.570** | **+0.033** |
| ViewGauss | 1 | 0.523 +-0.007 | 0.555 +-0.007 | **0.571** | **+0.048** |
| NJIT | 1 | 0.522 +-0.009 | 0.527 +-0.003 | 0.540 | +0.018 |
| VR_User_Behavior | 1 | 0.515 +-0.006 | 0.520 +-0.001 | 0.521 | +0.007 |
| Panonut360 | 2 | 0.516 +-0.005 | 0.533 +-0.007 | 0.544 | +0.028 |
| PanoSaliency | 2 | 0.730 +-0.007 | 0.736 +-0.002 | 0.731 | 0.000 |
| EyeNavGS | 3 | 0.535 +-0.003 | 0.529 +-0.001 | 0.529 | -0.006 |
| **pooled** | | **0.582 +-0.001** | **0.600 +-0.001** | **0.598** | **+0.016** |

Selected epochs 60-120 of 120, so validation kept improving for a long time; but the
long budget changed the *transfer* figure at 419 by nothing (0.582 against 0.581), so
the censoring was in domain only and the transfer plateau is real.

Three things, in decreasing order of confidence:

1. **A transferable dynamics component exists on tier 1 and it is small**: +0.02 to +0.04
   over chance at 419 identities, on every tier-1 corpus, from a model that never saw
   seated video viewing. Its seed spread is under 0.01, so it is many sds above the
   floor. The pre-registered prediction was "about chance at 419"; it is slightly above.
2. **It rises with identity count, then stops.** 419 to 1000 identities: +0.018 pooled
   on both seeds, at a seed spread of 0.001; +0.023 on Head_and_Gaze, +0.032 on
   ViewGauss. 1000 to 2096: -0.002 pooled, with Head_and_Gaze and ViewGauss still
   creeping up (+0.010, +0.016) and everything else flat or down. The pre-registered
   band was +0.02 to +0.05 from 419 to 2020; the measured +0.016 pooled and +0.03 to
   +0.05 on the two best-conditioned corpora are inside it. This is the only lever in
   the whole night that moved a transfer number, and it moved it where the corpus has
   a real, stable head pose to move it on.
3. **PanoSaliency at 0.72 is the tier-2 story in reverse.** Its "position" is a viewing
   direction, so the `dyn` residual is *how the viewing direction sweeps within five
   seconds*, and a model trained on real head translation in Beat Saber reads that
   sweep and separates PanoSaliency's users better than anything in this document
   (the `std_only` lookup found 0.65 there in section 2). That is a behavioural signal
   crossing a semantic mismatch, and it says head-direction dynamics carry more
   identity in 360-degree viewing than head translation does - which is a reason to
   build the `channels=orientation` mode the Coordinator deferred, not a result about
   head pose.

### 9.4 Fusion, and what the dynamics branch is worth in domain

Run 5 of section 6, scoring only, on the five 30-epoch `dyn` checkpoints: the lookup
(raw windows, per-dataset z-scored, target-fit on the held-out corpora) and the model's
cosine on the same pair manifests, both z-scored on the checkpoint's own in-domain
validation users (BOXRR/alyx users it never trained on), mixed as
`(1-w) * lookup + w * dyn` with `w` chosen on validation AUC alone.

| | lookup | dyn | fused | w |
| --- | --- | --- | --- | --- |
| **in domain** (BOXRR/alyx validation users, 5 checkpoints) | 0.724-0.764 | **0.727-0.744** | **0.789-0.819** | 0.35-0.45 |
| held out, ViewGauss | 0.935 | 0.511 | 0.835 | |
| held out, Head_and_Gaze V2 | 0.870 | 0.536 | 0.777 | |
| held out, VR_User_Behavior | 0.718 | 0.518 | 0.662 | |
| held out, NJIT | 0.645 | 0.528 | 0.628 | |
| held out, PanoSaliency | 0.579 | 0.723 | **0.723** | |
| held out, pooled | 0.727 | 0.578 | 0.707 | |

Three results, and the third is the one to remember:

- **The fusion as designed does not help on tier 1.** A weight fitted in domain assumes
  the dynamics branch is worth what it is worth in domain, and out of domain it is not,
  so the mixture is pulled 0.04-0.12 below the lookup on every tier-1 corpus. It helps
  only where `dyn` is the stronger of the two (PanoSaliency).
- **Nor does a transfer-aware weight.** Choosing `w` on the *other six* held-out
  corpora (leave-one-corpus-out, never the target's labels) gives 0.15-0.35 and leaves
  every tier-1 corpus at or just below the lookup (ViewGauss 0.91 vs 0.93, Head_and_Gaze
  0.82 vs 0.87, VR_User_Behavior 0.71 vs 0.72, NJIT level), and picks `w = 0` for
  PanoSaliency, forfeiting its gain. A +0.02 to +0.04 dynamics signal is too small
  relative to its own noise to add to a 0.7-0.9 static score at any global weight.
  **Fusion is retired for transfer** unless the dynamics branch gets much stronger.
- **The in-domain dynamics strength is Beat Saber's, not the model's.** Split by
  training corpus, on validation users never trained on:

  | in-domain validation users | lookup | `dyn` |
  | --- | --- | --- |
  | BOXRR, 74-94 users per checkpoint | 0.75-0.80 | **0.78-0.81** |
  | alyx, 14-17 users per checkpoint | 0.58-0.62 | **0.53-0.55** |

  Movement alone identifies unseen Beat Saber players at about 0.80 AUC - better than
  the static lookup there, with every static cue removed by construction, censored at
  30 epochs (those users also chose the epoch, so read it as optimistic by ~0.02). It
  is the first clean behavioural-biometric number in this project. The same branch is
  near chance on alyx users *that were in the training corpus*, and 0.52-0.54 on seated
  video viewing. Two readings, not yet separated: alyx is 3.6% of the training
  identities and may simply not be learned, or Beat Saber's content-locked rhythmic
  movement is far more stereotyped per person than free FPS locomotion with cross-day
  sessions. Section 6 run 8 (`dyn` on the 8-dataset corpus, stratified folds, alyx at
  18% of identities) separates them and gives every corpus an in-domain dynamics
  number of its own.

### 9.5 The dynamics signal per corpus, in domain versus transferred

Run 8: `dyn` on the 8-dataset corpus, 5 stratified folds, `epochs=60` with patience 15
(selected epochs 5-10, so nothing censored), random control on the same folds. Beside
it, the same encoding trained on BOXRR+alyx and transferred:

| dataset | tier | **in domain** (8 corpora, 5 folds) | random | transfer, 419 ids | transfer, 419, long budget | transfer, full corpus |
| --- | --- | --- | --- | --- | --- | --- |
| Head_and_Gaze V2 | 1 | 0.551 +-0.011 | 0.500 | 0.538 | 0.537 | **0.579** |
| ViewGauss | 1 | 0.528 +-0.034 | 0.506 | 0.517 | 0.522 | **0.565** |
| VR_User_Behavior | 1 | 0.531 +-0.015 | 0.498 | 0.519 | 0.521 | 0.512 |
| NJIT | 1 | 0.558 +-0.040 | 0.484 | 0.528 | 0.533 | 0.547 |
| **alyx** | 1 | **0.530 +-0.007** | 0.498 | - | - | - |
| PanoSaliency | 2 | 0.729 +-0.027 | 0.499 | 0.724 | 0.726 | 0.739 |
| Panonut360 | 2 | 0.522 +-0.017 | 0.505 | 0.516 | 0.514 | 0.548 |
| EyeNavGS | 3 | 0.529 +-0.024 | 0.490 | 0.535 | 0.536 | 0.516 |
| pooled | | 0.575 +-0.010 | 0.499 | 0.581 | 0.582 | 0.600 |

What this settles:

- **The alyx question from 9.4 is answered: it is the activity, not the share.** With
  alyx at 18% of training identities and trained in domain, movement alone identifies
  unseen alyx players at 0.530 - the same as the 0.53-0.55 it scored as 3.6% of a
  BOXRR-dominated corpus. Free FPS locomotion across two different days carries little
  per-person structure at a 5-second window; content-locked rhythm-game movement
  carries a great deal (0.78-0.81). The behavioural signal is a property of the activity
  and the session gap, not something more identities of the same activity unlock.
- **On the seated corpora the behavioural signal is small even in domain** (0.53-0.55,
  above the 0.50 floor by 3-5 sd), and **a BOXRR-trained branch already reaches it out
  of domain**: transfer at 419 identities sits 0.01 below in-domain training, and the
  full-corpus transfer *exceeds* in-domain training on Head_and_Gaze (0.579 vs 0.551)
  and ViewGauss (0.565 vs 0.528). Whatever generic head-movement identity exists in
  360-degree viewing, 2000 Beat Saber players teach it about as well as the corpora
  themselves do - which is the first evidence in this project that a learned
  component transfers across activity, and the ceiling it transfers to is low.
- **Identity count moves the dynamics branch where the static cue is strongest**
  (Head_and_Gaze, ViewGauss: +0.03 / +0.05 from 419 to 2096, section 9.3) and not on
  VR_User_Behavior, and the full-corpus transfer now *exceeds* in-domain training on
  both (0.570 vs 0.551, 0.571 vs 0.528).

### 9.6 What the night settles

- **The static branch is the whole transferable signal, and it needs no model.** A
  three-number lookup beats the trained `raw` model on every tier-1 corpus out of
  domain, and the `raw` model's transfer is flat to three decimals from 419 to 2096
  training identities. Any "more identities" claim made on the `raw` pipeline is a
  claim about the lookup, which has no training set.
- **The learned component is real, small, and does transfer once it is forced onto
  dynamics** (`dyn`): +0.02 to +0.07 over chance on tier 1, sd under 0.01, rising with
  identity count from 419 to 1000 and flattening by 2096. Out of domain it reaches or
  exceeds what in-domain training on those corpora achieves, so the ceiling it hits is
  the corpora's, not the model's: seated 360-degree viewing carries little
  per-person movement structure at five seconds.
- **The behavioural signal is activity-bound.** Movement alone identifies unseen Beat
  Saber players at about 0.80 (above the static lookup) and unseen alyx players at
  0.53, whether alyx is 3.6% or 18% of the training identities. Content-locked rhythmic
  movement is a biometric; free FPS locomotion across two days, at this window
  length, barely is.
- **Fusion does not help transfer.** No weight chosen without the target's labels beats
  the lookup on tier 1; the dynamics signal is too small relative to its noise to add
  to a 0.7-0.9 static score. Retired until the dynamics branch is much stronger.
- **`yawc` is not worth an arm**: it helps exactly the corpora predicted and hurts the
  rest by as much.
- What would move the transfer number next, in order: a test set that varies
  *activity* at fixed users and rig (Across-XR, 49 users x 5 apps) so the
  activity-bound finding can be measured directly rather than inferred across
  corpora; a `channels=orientation` mode, because head-direction dynamics carry more
  identity in 360-degree viewing than head translation does (PanoSaliency 0.73 under
  `dyn`); and longer windows for the dynamics branch specifically, since free
  locomotion may need more than five seconds to show a person.

## 10. Next steps, ranked (written 2026-09-04 after section 9)

The night answered the question it was asked: identity count does not move transfer
for the pipeline as it stands, because the transferable signal is static and has no
training set, and the learned signal is bound to the activity it was trained on. What
is left is a short list, each with the question it answers, its cost, and the
prediction registered now.

| # | step | question it answers | cost | prediction |
| --- | --- | --- | --- | --- |
| 1 | **Leave-one-corpus-out over the 8 corpora, `raw` and `dyn`** (train on seven, test on the eighth, every corpus in turn) | Does *diversity* of training corpora buy transfer where 2000 same-activity identities did not? And what is the transfer cost per corpus, not just alyx? | 16 runs, ~2.5 h. **Launched.** | `raw` lands at the lookup +-0.03 on tier 1 (the alyx point was -0.03); `dyn` 0.52-0.58, near its in-domain values. If `raw` beats the lookup on two or more tier-1 corpora, diversity is the lever and the BOXRR-heavy design should change. |
| 2 | **Window length for the dynamics branch**: `dyn`, `sample_time` 10 and 20 with `window_stride=5`, 419 identities, seeds 1-5 | Does free locomotion need more than five seconds to show a person? alyx at 0.53 and the +0.02 window-length result on `raw` both point here. | 10 runs, ~2 h (cache exists at 10 s) | +0.01 to +0.03 on tier 1 and on alyx in domain; if alyx moves above 0.60 the activity-bound reading softens to window-bound. |
| 3 | **A learned static branch** (CPU): mean pose, within-window std and mean orientation as a 10-20 number descriptor, a metric learned across corpora (LDA/PLDA-style or a two-layer scorer), evaluated leave-one-corpus-out against the three-number lookup | Can *any* learned static scorer beat the lookup out of domain, or is the three-number lookup already the ceiling of the static cue? | one afternoon, no GPU | +0.00 to +0.03 on tier 1. A gain would be the first learned thing to beat the lookup across corpora. |
| 4 | **Across-XR** (49 users x 5 applications, converter ready, download WAF-blocked from AVALON; retry from another machine or ask the authors) | The activity-bound finding measured directly: same users, same rig, different application. Cross-app `dyn` AUC is the number. | download 5.4 GB, one conversion, scoring only | cross-app `dyn` well below within-app; the size of that gap is the paper's second claim. |
| 5 | **`channels=orientation`**: quaternion-only windows, plus a converter that puts tier-2 direction vectors into the orientation channel rather than the position channel | Is head-*direction* dynamics the behavioural biometric for 360-degree viewing? PanoSaliency at 0.73 under `dyn` says direction sweeps carry more identity there than translation does, and it would make 240 tier-2 identities usable honestly. | ~1 day of code, then the tier-2 corpora in domain | in-domain `dyn` on the seated corpora rises from 0.53-0.55 toward PanoSaliency's 0.73 if direction is the signal. |
| 6 | **The static cue as an enrolment system**: templates over k windows, cohort normalisation, CMC at N=17 | Places the transferable signal on the field's own axis (rank-1) with an honest enrolment protocol, since this is what would actually ship on glasses. | scoring only | rank-1 at N=17 in the 0.4-0.6 range on tier 1, below published head+controller figures. |

**Retired by section 9, do not re-run:** the identity-count curve on the `raw`
pipeline as a headline (it measures the lookup); fusion of lookup and `dyn`; `yawc` as
an arm; label-free embedding adaptation.

**What the paper can claim now**, in one paragraph: on head pose alone, unseen users
are verified across capture rigs primarily through cohort-relative head position, which
three numbers capture without training and which no trained model in this repository
beats out of domain; a learned movement component exists, is small on seated viewing,
strong on rhythm-game play, rises with training identities up to about a thousand, and
does not carry across activities. Across-XR (step 4) is what turns the last clause from
an inference across corpora into a measurement.

## Appendix: reproduction

All CPU, from `.cache/samples` and the sweep checkpoints on DESKTOP-C.

**Committed:** `audit_frames.py` at the repo root produces the section 1 table (up axis,
yaw reference, position norm and statistics, duplicate frames) for any dataset with
cached windows, and flags a unit-vector position. Run it on every newly converted dataset
before it is trained or tested on.

**Not committed** (one-off confirmations whose job the pipeline is taking over: the
lookup baseline and per-dataset metrics are being added to `evaluate()` and the results
row, after which these are redundant):

| script | produced |
| --- | --- |
| `score_transfer.py` | section 2 model columns, clustering statistics, section 3 alyx variants |
| `static_probe_trainstats.py`, `static_probe_sweep.py` | section 2 and 2b lookup columns over five folds, training-fitted statistics |
| `static_probe.py` | the BOXRR and alyx lookup numbers |
| `alyx_matched.py` | the matched-user table in section 3 |

Checkpoints: `sweeps/b732bee5c6/runs/bilstm_c07b65fe9e_fold{0..4}/best.pth` (7 datasets,
343 identities, records its own held-out users), `sweeps/1cba61edba/runs/random_*` (random
control), `sweeps/31751868df/runs/bilstm_fbbaa6e42b_fold{0..3}` (8 datasets, in-domain
alyx reference).
