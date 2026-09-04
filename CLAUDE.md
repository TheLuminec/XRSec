# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

XR biometric identification research. A Siamese network decides whether two windows of headset motion came from the same person. The research question is whether this generalizes to **users never seen during training**, so nearly every design decision (leave-users-out splits, pair generation, boosting) exists to serve that question.

Current state: the defensible headline is **0.669** on unseen users (chance = 0.50) — `bilstm`, `objective=identity_softmax`, cross-session positives, validation-selected epoch, averaged over 5 leave-users-out folds on VR_User_Behavior. That is the only figure that has survived all three corrections below; earlier numbers in this file and in `results/runs.csv` predate one or more of them. The historical 0.85 is **explained and reproduced**: it was a *seen-user* number. Our lineage never held users out — the MS thesis this repo descends from splits pairs randomly across users ("each video contains sensor data from all users who watched it") and reports **0.8364 on VR_User_Behavior**, the dataset we still use. Verified here directly: the same protocol on our own code (`test_on_excluded=false`, `pair_bce`, `bilstm`, 2s@20Hz) reaches **0.810**, against 0.62–0.67 for the identical configuration with leave-users-out. So it is not a target, not a regression, and not comparable to anything in this file — every number here is leave-users-out.

What moved it, measured with paired folds:

| change | effect |
| --- | --- |
| `objective=identity_softmax` vs pairwise BCE | **+6.5**, 5/5 folds on each of three backbones |
| per-dataset normalization + within-dataset negatives (pooled data) | **+11.1** |
| extractor architecture (3 working ones, 10 folds) | **~0**, spread under 1 point |
| cross-session positives (correction, not a gain) | −1.1 to −1.6 |
| validation-selected epoch (correction) | −2 |

**The extractor is not the constraint.** Two cross-validated sweeps put `paper_gnn_bilstm`, `bilstm` and `motion_tdnn` within 0.002 of each other. The objective, the pairing rules and the data are where the movement is.

**Read every number here against the noise floor.** Which users are held out moves accuracy by ~0.114 (sd 0.037), so single-split differences below ~0.04 are not results. See "The evaluation is noisier than it looks" below; use `sweep.folds`.

## Commands

Run everything with the repo venv from the repo root:

```bash
.venv/Scripts/python -m pytest -q
```

```bash
.venv/Scripts/python model/main.py mode=train
```

```bash
.venv/Scripts/python model/main.py mode=sweep sweep.dry_run=true
```

Single test / single module / by marker:

```bash
.venv/Scripts/python -m pytest tests/unit/test_dataset.py::test_generate_pair_manifest_is_deterministic -q
```

```bash
.venv/Scripts/python -m pytest -m "unit and not slow" -q
```

Hydra overrides work on any config key (no committed run has ever used them — the historical workflow is editing `configs/config.yaml`, which makes each experiment an uncommitted working-tree edit):

```bash
.venv/Scripts/python model/main.py mode=train sample_time=2 sample_rate=20 embedding_dim=64 boosting.enabled=true
```

## Architecture

Data flows through layers that are plain Python objects, not framework abstractions:

`UserProfile` (one user's CSVs) → `Sampler` (one CSV → fixed-rate windows) → `SampleIndex` (all windows flattened into one tensor + per-user index ranges) → **pair manifest** (integer index pairs + labels) → `DataLoader`.

`SampleDataset` walks user directories itself (sorted, filtering *before* loading) rather than going through the `Users` wrapper, so excluded users cost nothing and each user's windows can be cached independently. `model/users.py` is now unused by the training path.

The key idea is the split between `SampleIndex` and the **pair manifest**. Windows are stored exactly once in a single flat tensor; a "dataset" of Siamese pairs is just four parallel integer/float tensors (`x1_indices`, `x2_indices`, `labels`, `anchor_user_ids`) naming positions in it. Pairs are therefore cheap to regenerate, never written to disk, and reproducible from a seed. This is what makes boosting rounds possible without storing pair tensors. Anything that builds or filters a training set should produce a manifest, not a new tensor of windows.

Every window is `(7, sample_time * sample_rate)`: channels are `qx, qy, qz, qw, Hx, Hy, Hz`, with the `SessionTime` column stripped after sampling. `seq_len` is therefore a derived quantity — `sample_time * sample_rate` — and changing either factor changes the model's input dimension and invalidates old checkpoints.

## Feature extractors (slottable)

The only part of the pipeline that decides *how* a window becomes an embedding. Contract: `(batch, num_channels, seq_len) → (batch, embedding_dim)`. Everything downstream — Siamese head, pair generation, boosting, evaluation — is unaffected by swapping one out.

- `model/feature_extractor.py` — the `FeatureExtractor` ABC, the registry (`register` / `create` / `available` / `search_space`), and `check_output_contract()`.
- `model/extractors/` — implementations. **Every module here is auto-imported**, so a new file is picked up with no edits anywhere: define a `FeatureExtractor` subclass, decorate it `@register("name")`, and it is selectable as `extractor=name`.
- `model/list_extractors.py` — prints each extractor, its tunable arguments, defaults, and declared sweep space.

Three are registered: `paper_gnn_bilstm` (the published architecture, the default), `bilstm` (the same minus the GNN branches — the ablation showing what the graph layers contribute), and `random` (ignores its input and emits noise — the chance-level floor any real result must clear).

To add one, write a subclass whose hyperparameters are explicit keyword arguments with defaults, pass them to `super().__init__` so they reach the checkpoint, and declare `search_space()` for the sweep. Set `deterministic = False` on the class if `forward` is stochastic in eval mode; the contract tests verify the flag in **both** directions, so accidental non-determinism (dropout left active at eval, unseeded noise) fails loudly instead of showing up later as unexplained variance between sweep runs. Keys in `search_space()` **must** be real constructor arguments — enforced by the tests, because a bad key would otherwise produce a sweep that silently re-ran the default configuration. `fe.create()` likewise rejects unknown hyperparameters rather than ignoring them.

Selection is `extractor` / `extractor_params` in the config. `extractor_params` defaults to `null` rather than `{}` deliberately — Hydra's struct mode refuses to add keys to an empty dict, which would force `+extractor_params={...}` on every override.

Checkpoints record `extractor`, `extractor_params` and `num_channels`, so `load_checkpoint` rebuilds the exact backbone without the config that produced it (this is what lets boosted rounds warm-start). Checkpoints written before extractors became slottable have an incompatible `state_dict` layout and must be retrained; `load_checkpoint` says so explicitly rather than failing cryptically.

`tests/unit/test_feature_extractors.py` is parametrized over the registry, so every extractor — including one added later — is automatically checked for output shape, varied `seq_len`/`embedding_dim`, sweep-space validity, every declared sweep value building and running, trainability inside `SiameseModel`, declared-vs-actual determinism, and checkpoint round-trip fidelity. Two of those adapt to the extractor rather than assuming one shape: an extractor with no parameters of its own (like `random` at default settings) is checked for the Siamese head training instead, and round-trip fidelity is asserted on `state_dict` for everything, with output equality added only for deterministic extractors.

Artifact stems include the extractor name plus a 6-char digest of any non-default hyperparameters, so sweep runs don't overwrite each other's checkpoints and plots.

`model/model.py` holds only the Siamese head and the model factory — no architecture. The published architecture lives in `model/extractors/paper_gnn_bilstm.py`: a fixed 10-node graph (7 channel nodes + orientation/position/root aggregate nodes) is run through two GNNs — `Ga` (GATConv, attention) and `Gp` (GraphConv, sum) — whose outputs are concatenated with the raw input as `<M, M', b>` (21 features/timestep), then BiLSTM → self-attention → BiLSTM → self-attention → mean-pool → dense. `SiameseModel` wraps it and classifies `|e1 - e2|` through a linear layer, trained with `BCEWithLogitsLoss`. The batched `edge_index` is cached on the module and rebuilt whenever batch size changes — building PyG `Data` objects per forward pass was a major CPU bottleneck, so don't reintroduce that.

`model/train.py` owns the standard path and the shared `run_training`/`prepare_training_round` primitives; `model/boost_train.py` owns the boosted path and receives those primitives as arguments (dependency injection, to avoid a circular import). Boosted rounds: score a deterministic candidate pool with the previous round's best checkpoint, keep the highest-loss pairs per anchor user while preserving label balance, refill the rest with fresh pairs, warm-start from the previous best, evaluate every round against one fixed validation manifest.

Determinism runs through a single root `seed`. `derive_seed(seed, *parts)` in `train.py` hashes string/int parts into a sub-seed, so each manifest, loader, and round gets an independent but reproducible stream. Directory traversal is `sorted()` at every level so sample indices are stable across machines. Preserve both properties in any change to data loading — the reproducibility tests depend on them.

## `data_dirs` defaults to ONE dataset

`configs/config.yaml` ships `data_dirs` with a single entry - VR_User_Behavior - and
the other six commented out. **Every pooled-corpus run must pass `data_dirs`
explicitly.** A command that overrides ten other keys and leaves this one alone trains
on 48 identities while its author believes it is on 343, and nothing in the results row
says otherwise.

This has now cost one pilot outright: balanced identity sampling was motivated by an
87.5x window-count imbalance that exists only in the pooled corpus, and was measured on
VR_User_Behavior, where `max/min` is **1.0x** and balancing is a mathematical no-op. The
only tell was a user count in the loader's stdout.

Two guards were added rather than a resolution to be careful:

- `mode=sweep` prints a **CORPUS banner** naming every dataset before it runs, and says
  explicitly when there is only one that this is the config default and that
  `normalize=per_dataset` / `within_dataset_negatives` are no-ops.
- `num_train_identities` is recorded per run, so the corpus a row was produced on is
  visible in the results table instead of only in a log nobody kept.

It is the same failure shape as the `mode=curve` split fallback and the `sweep_id`
collision: a default silently standing in for the intended experiment and returning a
plausible number. That is the recurring bug in this project, not any particular one of
its instances.

## Sweep mode

`model/sweep.py`, invoked with `mode=sweep`. Enumerates configurations, trains each, ranks them.

- **Axes are namespaced**: `extractor_params.<name>` varies an extractor hyperparameter, anything else varies a top-level config key. `grid: auto` defers to each extractor's `search_space()`.
- **One process, not one per configuration** — the sample cache makes each extra configuration cost about a second of loading rather than a full CSV parse.
- **A failing configuration is recorded and skipped, never fatal.** This is deliberate: generated extractors fail on some combinations, and a sweep that dies on configuration 3 of 54 is useless. Failures are retried on resume; successes are not.
- **Resume is keyed by a digest** of `{extractor, overrides}`, persisted in `{artifact_root}/{sweep_id}/sweep_state.json`.
- `sweep.artifact_root` **is made absolute** in `_normalize_paths` — unlike `boosting.artifact_root`, which is still relative and therefore still cannot resume across runs (see the `auto` path trap above). Don't copy the boosting pattern here.
- Each configuration is appended to `results/runs.csv` with its `sweep_id`; `mode=sweep` itself writes no summary row.

`train_fn` is injected into `run_sweep` so the tests exercise orchestration (ranking, resume, failure isolation) without training anything.

## Splits and the `swap_data` / `test_on_excluded` convention

There is no user-facing "split" abstraction; splits are expressed by a list of user directories plus two booleans, and this is the most error-prone part of the codebase.

- `exclude_users` is a list of **absolute** user directory paths (made absolute by `main.py`).
- `swap_data=False` → keep everything *except* `exclude_users`. `swap_data=True` → keep *only* `exclude_users`.
- `test_on_excluded=True` → the eval set is built with the flag flipped, so train and test see disjoint users.

The default config trains on 43 users and evaluates on 5 held-out ones. **`test_dirs` pointing at a different dataset is incompatible with `test_on_excluded=True`**: the exclude paths belong to the training dataset, nothing matches, the loader silently reports "Loaded 0 samples from 0 users", and evaluation dies with a bare `ZeroDivisionError`. Set `test_on_excluded=false` for cross-dataset evaluation.

### Same-session positives (answered: costs ~1.5 points)

A positive pair is two windows from the same user — and usually, therefore, from the **same recording session**, which shares headset mounting, seating position and the content being viewed. A model can score well by matching the session rather than the person, and because held-out positives are *also* same-session, that shortcut never appears as a train/test gap. This has the same shape as the cross-dataset shortcut, which cost 11 points once fixed.

`cross_session_positives: true` draws positives from two different sessions of the same user. Users with only one session fall back to same-session pairs, and that count is recorded per run as `same_session_fallback_users` so the qualification travels with the number.

Session inventory (users with fewer than 2 sessions):

| dataset | users | 1 session | min | median |
| --- | --- | --- | --- | --- |
| NJIT_6DOF | 18 | **18** | 1 | 1 |
| Head_and_Gaze | 100 | 0 | 34 | 54 |
| PanoSaliency | 99 | 0 | 2 | 22 |
| VR_User_Behavior | 48 | 0 | 18 | 18 |
| ViewGauss | 35 | 0 | 4 | 4 |
| EyeNavGS | 22 | 0 | 12 | 12 |
| Panonut360 | 21 | 0 | 15 | 15 |

**NJIT_6DOF is the only affected dataset.** On the pooled corpus that is 18/343 = 5.2% of users; on VR_User_Behavior alone it is 0%, so the single-dataset cross-session results are fully cross-session.

**Result:** cross-session pairing costs only **1.1–1.6 points** (bilstm 0.685 → 0.669, t(4)=−4.06, lost 5/5; motion_tdnn 0.686 → 0.675, t(4)=−1.40, not distinguishable from zero). The `random` control sits at chance under *both* regimes (0.4947 / 0.4967), so the drop is a real effect on real signal rather than an artifact of the new pair construction. Set against the cross-dataset shortcut — worth 11 points when live — this is the signature of a model that mostly is **not** relying on session matching.

So: same-session pairing was inflating the figure by about a point and a half, and the +6.5 from `identity_softmax` survives intact. Quote the cross-session number.

Predicted beforehand from data alone: between-session position spread is comparable to or *smaller* than within-session spread (0.64–1.26× across three datasets), so position is a user-level property rather than a session fingerprint.

Session provenance lives in `SampleIndex.window_session_ids` and is stored in the sample cache (cache v3).

### Half the corpus has no absolute head height at all

The anthropometric cue - the strongest single thing this model uses - **does not exist in
every dataset**. Mean `HmdPosition` per axis, sampled across users:

| dataset | mean x | mean y | mean z | absolute height? |
| --- | --- | --- | --- | --- |
| ViewGauss | 0.436 | **1.564** | 0.420 | yes |
| NJIT_6DOF | 3.011 | **1.587** | 2.252 | yes (room-scale) |
| VR_User_Behavior | 0.024 | **1.162** | -0.258 | yes (seated) |
| Head_and_Gaze | 0.181 | **0.822** | 0.222 | yes (low origin) |
| PanoSaliency | -0.536 | -0.008 | 0.195 | **no** |
| EyeNavGS | -0.184 | 0.214 | 0.601 | **no** |
| Panonut360 | -0.251 | 0.043 | -0.324 | **no** |
| 360_em | -0.074 | 0.147 | 0.463 | **no** |
| **BOXRR-23** | | **1.602** (sd 0.141) | | **yes, standing** |

Four of eight record position relative to a seated origin, so no axis carries a height.
Those users cannot be separated by anthropometry at all, and whatever the model achieves
on them is posture and behaviour.

**This qualifies the "~78% is absolute head position" finding rather than overturning it.**
That figure was measured on the pooled corpus, where it is an average over datasets that
carry the cue and datasets that cannot. The per-dataset picture is far more uneven than a
single number suggests, and any future anthropometry claim should say which datasets it
rests on.

Height discriminability, between-user sd over within-user sd on the height axis (a
scale-free ratio, so comparable even where the frames differ):

| ViewGauss | BOXRR-23 | NJIT | VR_User_Behavior | the four centred datasets |
| --- | --- | --- | --- | --- |
| 4.03 | **2.36** | 1.97 | 1.61 | 0.21-0.42 (no height axis) |

### A confound to weigh before reading the identity-count curve

**BOXRR is probably an easier corpus per identity, not merely a bigger one.** It has true
standing head height (2.36 discriminability, above all but ViewGauss), one uniform
activity, and absolute coordinates - where half our existing corpus has no height cue at
all and the rest is seated posture. Recorded before the curve is measured: a large gain
at 2439 identities will be **partly BOXRR being easier**, not purely more identities
helping.

The two readings separate it. Held-out BOXRR users measures identity count within a clean
domain and will flatter. Training with BOXRR and testing on our existing datasets asks
whether the acquisition transfers, and is the number that decides whether it was worth
it. If the first is strong and the second flat, the honest conclusion is that we bought
an easier corpus rather than a better model.

### How much is the model actually adding?

Measured under the **corrected** protocol (5 leave-users-out folds, cross-session positives, per-dataset normalization, threshold fitted on validation users and accuracy reported on held-out users) — the same protocol the trained runs use:

| | accuracy |
| --- | --- |
| trivial descriptor: per-channel mean + std of the window, thresholded | 0.562 |
| the same with `center_position` (movement only, no absolute position) | 0.513 |
| **trained model** (`bilstm`, `identity_softmax`) | **0.669** |

So the learned model adds **+0.107 over two lines of numpy**, and absolute position is worth about 0.05 of the trivial descriptor's 0.062 above chance.

This **reverses an earlier reading**. On the old protocol the trivial baseline scored 0.712 against a trained 0.656, which looked like the models adding nothing. That comparison was made on the fixed 5-user split — which is unusually easy — with same-session positives, and against a `best_test_acc` that was itself inflated. Corrected, the ordering flips. Any future "is the network earning its place" claim should be measured this way, not on the fixed split.

### Anthropometry vs behaviour — answered

**About three quarters of what this model does is absolute head position, i.e. height and seated posture.** Measured with `center_position` (5 stratified folds, `identity_softmax`, cross-session positives, validation-selected, paired on matched folds):

| extractor | keeps position | centred (movement only) | difference | t(4) |
| --- | --- | --- | --- | --- |
| bilstm | 0.6691 | 0.5352 | −0.1339 | −9.00 |
| motion_tdnn | 0.6753 | 0.5405 | −0.1348 | −10.27 |
| random (control) | 0.4967 | 0.4967 | 0.0000 | — |

Above the 0.4967 control floor: full headroom 0.1724, movement-only headroom 0.0385 — **22% retained**. Independently by AUC: 0.2237 → 0.0530, **24% retained**. Two architectures agree to within 0.001 on the size of the drop, the control is bit-identical across arms, and the trivial descriptor shows the same thing from the other direction (0.562 → 0.513).

What this does and does not mean:

- **Not spurious.** 0.669 is a real identification result and head height is a genuine biometric. It survived leave-users-out folds, validation-selected epochs and cross-session positives. There is no leakage here.
- **Not primarily behavioural.** This cannot be described as identifying people by how they move when three quarters of it is how tall they are.
- **The behavioural component is real but small**: 0.535 / 0.541 against a 0.497 floor, AUC 0.553 / 0.561, two architectures agreeing. It deserves its own reported number rather than being folded into the headline.

**The behavioural component is identity-count-limited — the diversity confound is resolved.** `max_users=48` subsamples the pooled corpus back to 48 identities stratified across the same 7 datasets, holding dataset diversity fixed and varying only identity count. Balanced pair sets in training and evaluation on both halves (`eval_positive_fraction` 0.5000, control at chance on both metrics):

| | 343 ids acc | AUC | 48 ids acc | AUC |
| --- | --- | --- | --- | --- |
| keeps position | 0.6722 | 0.7264 | 0.6228 | 0.6879 |
| centred (movement only) | 0.5765 | 0.6029 | **0.4906** | **0.4700** |
| random control | 0.4991 | 0.4993 | 0.4988 | 0.4973 |

Headroom above the measured control floor:

| | accuracy | AUC |
| --- | --- | --- |
| keeps position | +0.1731 → +0.1240 (keeps ~72%) | +0.2271 → +0.1906 |
| centred | +0.0774 → **−0.0082** (keeps none) | +0.1036 → **−0.0273** |

**At 48 identities the behavioural arm is at chance.** The anthropometric arm loses about a quarter of its headroom over the same 7× reduction; the behavioural arm loses all of it. So the behavioural signal is not merely helped by more identities — below some threshold between 48 and 343 it is not measurable at all, while head height is nearly as identifiable at 48 as at 343.

Two consequences worth acting on:

1. **The behavioural ceiling is data, not modelling.** Consistent with three architectures tying, `motion_gram` losing, and every gain coming from the objective and identity count. It may not have plateaued at 343 — that is now a live question rather than a rhetorical one.
2. **Every single-dataset behavioural result in this repo is uninterpretable.** VR_User_Behavior alone is 48 identities. The 0.535/0.541 centred figures measured there sit at roughly the level this table shows is indistinguishable from chance, so they should not be quoted as evidence of a behavioural component.

**A caveat on the 48-identity centred figure, and a caveat on the caveat.** The pooled 48-identity subsample draws proportionally from all 7 datasets, so it includes ViewGauss (10Hz native) and PanoSaliency (17Hz), which at `sample_rate=20` under `resample=nearest` contribute 50.5% and 27.0% exact duplicate consecutive frames. Duplicated frames destroy movement and leave absolute position untouched, so they penalise the centred arm specifically. That is worth testing directly - `max_users=48 center_position=true` over `resample: [nearest, bin]`, 10 runs - because it is a real question at fixed identity count.

What it is **not** is a contradiction needing explanation. The single-dataset centred figure (0.5352, +0.0385 over floor) is one this file has already retired as uninterpretable: it is 48 identities, which is exactly where this table shows the behavioural signal is indistinguishable from chance, and +0.0385 on 5 folds of ~10 held-out users sits inside the measured 0.037 fold spread. +0.0385 and -0.0082 are two draws from the same no-signal distribution. Treating the first as a fact the second must account for would invent a sampling effect to explain noise. If `bin` does not move the centred arm, that is one 48-identity measurement agreeing with another - not evidence the collapse is "genuine".

**Both arms were rerun under the fixed code.** The 343 arm came back bit-identical (sweep `bc69fd0d50`, 20/20 configurations), and the 48 arm is sweep `34a943c9a1` on `732a12c` with the balance fix, 20 folds, `eval_positive_fraction` 0.5000, 0 warnings. The label-balance bug never touched the 343 arm - it was always balanced - so both halves of the comparison are on balanced pair sets and effectively on the same code. Nothing here is outstanding.

Superseded, for anyone reading older notes: the earlier version of this comparison put the pooled 343-identity corpus against **VR_User_Behavior alone**, which varies identity count, dataset diversity and `normalize=per_dataset` together. `max_users` exists to separate them and has now done so. The 22.3% -> 44.7% behavioural-share figure came from that confounded pairing; use the table above instead.

One caveat when quoting the 22% (or the 44.7%): the centred arm still contains **absolute quaternion**, and how someone holds their head is itself partly postural. So 22% is an upper bound on the purely behavioural share, not a point estimate. Centring orientation as well, or `channels=position` + `center_position`, would tighten it.

### Pair balance is enforced, not reported

Accuracy is read at the fixed `logit > 0` threshold, so it only means anything on a set whose balance is what was asked for. This went wrong twice:

1. A user with no eligible negative partner had their positives **inflated to the full quota**. On a 48-user stratified subsample the set came out 69% positive and the `random` control scored 0.6886 accuracy at AUC 0.5056 — outscoring both real configurations.
2. Removing the inflation was not enough. Such a user still contributes positives and *no* negatives, so each one shifts the set positive regardless of how many positives they contribute. 2–4 users out of 9–11 per fold left it at 62% positive.

Under `within_dataset_negatives`, any user who is the sole member of their dataset in a fold has no eligible partner, so this is structural rather than a rare edge case — and it is created by `max_users` subsampling and fold stratification, both of which spread small datasets thin.

`enforce_pair_balance` now trims the over-represented label until the realized ratio matches `match_ratio`, and reports how many pairs it dropped. Trimming keeps every user in the evaluation; dropping the affected users instead would have cost 20–40% of a held-out fold. The realized fraction is recorded per run as `eval_positive_fraction` — check it before reading any accuracy figure.

### Selection inflation, and the fix

`best_test_acc` is a **max over ~20 noisy evaluations of the set it reports**, which buys roughly **+0.02 for free**. This was caught by the `random` extractor under cross-validation: it scored 0.5173 as a best-of-20 but **0.4973 at its final epoch** — exactly chance. Every extractor showed the same offset, so **every historical `best_test_acc` in `results/runs.csv` is inflated by about 2 points, and the honest floor for that column is ~0.517, not 0.500.**

`val_user_fraction` fixes it by holding out a group of *training* users — disjoint from both training and the reported test users, since the task is generalisation to unseen people — and choosing the epoch on them. Three columns are now recorded:

| column | meaning |
| --- | --- |
| `selected_test_acc` | test accuracy at the validation-chosen epoch — **report this** |
| `best_test_acc` | max over epochs of the test set — optimistic, kept for continuity |
| `best_val_acc` | the selection signal itself |

Verified with the random extractor at `val_user_fraction=0.25` over 3 seeds: max-over-epochs averaged 0.525 while the validation-selected figure averaged **0.502**, i.e. chance. Default is 0 (historical behaviour) so old comparisons stay like-for-like; set it for anything you intend to quote.

### How many folds a question needs, and which questions are unaffordable

Paired folds are far more sensitive than the raw fold spread suggests - the 0.037 sd is
the spread of *absolute* accuracy across folds, while a paired test only sees the sd of
the *difference*, which is small when an intervention acts consistently. Recovered from
the t-statistics already recorded in this file:

| result | difference | t(4) | paired sd |
| --- | --- | --- | --- |
| cross-session, bilstm | 0.0160 | 4.06 | **0.0088** |
| cross-session, motion_tdnn | 0.0110 | 1.40 | 0.0176 |
| center_position, bilstm | 0.1339 | 9.00 | 0.0333 |
| center_position, motion_tdnn | 0.1348 | 10.27 | 0.0293 |

So paired sd runs 0.009 to 0.033 depending on how systematic the effect is. Minimum
detectable difference, two-sided p<0.05:

| folds | sd=0.009 | sd=0.018 | sd=0.033 |
| --- | --- | --- | --- |
| 3 | 0.022 | 0.045 | 0.082 |
| **5** | **0.011** | **0.022** | **0.041** |
| 10 | 0.006 | 0.013 | 0.024 |
| 15 | 0.005 | 0.010 | 0.018 |

**Read this before designing an experiment, not after.** Three consequences:

1. **A 3-fold pilot resolves almost nothing.** At the sd actually observed in the
   balance pilot (0.039) it could only have detected a difference of ~0.097. The
   hypothesis was +0.005 to +0.02. That pilot was incapable of confirming its own
   premise and could only ever have caught a large effect - which is worth knowing
   *before* spending the runs, and was not.
2. **A 5-fold sweep resolves 0.011 to 0.041.** Fine for the objective (+0.065) and for
   `center_position` (-0.134). Marginal for anything predicted under 0.02.
3. **Some questions are simply unaffordable.** An intervention predicted at +0.005 to
   +0.02 needs 10-15 folds, i.e. 20-30 runs for one comparison. That cost has to be
   weighed against the prediction before running, and for most small-effect ideas the
   honest answer is not to run them at all rather than to run them underpowered and
   read the noise.

The corollary is uncomfortable and worth stating plainly: **most of the remaining ideas
in this project sit at or below the resolution of the evaluation we can afford.** The
things that have moved this project - the objective, identity count, removing shortcuts -
all moved it by 0.05 or more. Prefer interventions with that shape.

### The evaluation is noisier than it looks

**Measured:** holding out a different random 5 users moves a training-free position probe from 0.631 to 0.746 — a **0.114 spread, sd 0.037** — while the binomial error bar on 2560 pairs is only ±0.019. The effective sample size is the number of held-out *users*, not pairs.

Two consequences that should govern how any result here is read:

1. **Differences below ~0.04 on a single split are not real.** An 8-configuration regularization sweep produced a range of 0.682–0.692; that is entirely inside the noise and separates nothing.
2. **The project's fixed split (users 1–5) is unusually easy**: 0.754 on the same probe versus 0.686 for the average random split. Numbers from it are optimistic, and an unreproducible historical high could partly be a lucky split.

Folds are **stratified by dataset**: each dataset's users are assigned round-robin across folds, so every fold's composition is proportional to within one user. This matters once several datasets are pooled — the corpus is 100/99/48/35/22/21/18 users and they differ in difficulty (ViewGauss is 10Hz native and half-duplicated at `sample_rate=20`; NJIT is room-scale walking with one session per user), so a randomly-partitioned fold heavy in one dataset measures something different from its neighbours and inflates the very spread that decides whether a result is real. `run_sweep` prints the per-fold composition so an unbalanced split is visible rather than assumed.

Use `sweep.folds: K` for anything you intend to act on. It ignores `exclude_users`, partitions every user across `data_dirs` into K disjoint held-out groups, runs each configuration on all of them, and ranks by the mean while reporting the spread. It prints an explicit warning when the top two configurations differ by less than the fold standard deviation.

## Data

Model input lives in `processed_datasets/<Dataset_Name>/users/<user_id>/<task>.csv`, gitignored, with required columns `SessionTime`, `UnitQuaternion.{x,y,z,w}`, `HmdPosition.{x,y,z}` at ≥10Hz.

**Which datasets exist is per-machine and must be checked, not assumed** — `processed_datasets/` is ~6.9GB when fully populated and cannot travel through git, so two checkouts of this repo routinely hold different data. Confirm before planning a run:

```bash
for d in processed_datasets/*/users; do echo "$(ls "$d" 2>/dev/null | wc -l) users  $d"; done
```

The table below describes the corpus when fully populated. `normalize=per_dataset` and `within_dataset_negatives` are **no-ops on a single dataset**, so a machine holding only one of these cannot reproduce any multi-dataset result.

**BOXRR-23 format facts**, read from the `xror` library's own source rather than its
documentation, and verified end-to-end against synthetic XROR files:

| | |
| --- | --- |
| device selection | on `type='HMD'` / `joint='HEAD'`. **Never on `name`** - that is an arbitrary hardware string that varies by headset model. |
| quaternion order | axes are `x,y,z` then `i,j,k,1` - **scalar last, already our x,y,z,w**. No reorder, unlike who-is-alyx. Read by declared axis name, never by position. |
| units | 1.0 = 1 metre. No conversion (unlike who-is-alyx's centimetres). |
| time | seconds since recording start. No conversion (unlike who-is-alyx's `delta_time_ms`). |

**Tilt Brush recordings may carry no head track at all.** The library's own `fromTilt()`
adds exactly one device - `BRUSH`, `type='OTHER'` - and no HMD. If BOXRR-23's Tilt Brush
files were produced the same way, that portion of the corpus is brush-tip trajectory
only and is unusable here. `prepare_boxrr.py` skips any recording with no HMD/HEAD
device and reports why rather than guessing. **Confirm against real `--inspect` output
before planning around any recording count**, and prefer Beat Saber sources
(BeatLeader, ScoreSaber) when selecting users from the BSON index.

**What 2419 identities makes measurable, and the design question it raises.** Identity
count is the only data-side lever that has ever moved this project, and at 419 we cannot
tell whether the behavioural component has plateaued - CLAUDE.md has carried that as a
live question since the 48-vs-343 result. `max_users` already subsamples by identity
count, so a corpus of 2419 turns "more identities help" from a slogan into a **curve**:
419 / 1000 / 2419 at fixed dataset diversity.

**But BOXRR is Beat Saber**, a single highly structured activity, where our existing
corpus is 360-degree video viewing and navigation. Adding 2000 Beat Saber identities
changes corpus composition far more than it changes identity count alone, so the curve
has to be read two ways and they answer different questions:

- **held-out BOXRR users** - does identity count help *within* a domain? Clean, and the
  one the curve is really about.
- **train with BOXRR, test on our existing datasets** - do extra identities from a
  different activity improve generalisation to the domains we care about? This is the
  question that decides whether the acquisition was worth it, and it is the harder one.

Expect the second to be weaker than the first. `normalize=per_dataset` and
`within_dataset_negatives` handle the mechanics of pooling, but neither makes Beat Saber
motion resemble seated video viewing.

**Landed.** 2020 BOXRR users converted, taking the corpus to **2439 identities (5.82x)**:
9,025 sessions, 137M rows, 9.0GB downloaded, 21GB on disk, 315,133 windows at 5s@20Hz
(**156 windows/user mean**). One session skipped (single row). All 135 Stage 1 recordings
carried an HMD track; mean |q| 0.99992; native rate 53-120Hz, mean 83.7.

Two things confirmed on real data that the synthetic tests could not have caught:

- **Device names are useless as a key.** 20 users produced 8 distinct HMD name strings -
  `Oculus Quest 2` and `Oculus Quest2` both present, plus `Rift_S` and `Unknown`. Selection
  is on `type='HMD'`/`joint='HEAD'` and this is why.
- **Tilt Brush is absent entirely** from this mirror's replay index: of 4,716,986 records,
  4,661,942 are Beat Saber and the rest carry no app field. The no-HMD risk was real in the
  library but is not present in the data.

**156 windows/user is below our existing median of 295, and that is fine.** The cap was
justified by "land on the median so imbalance does not worsen", but the imbalance *ratio*
is not the quantity that matters - the **absolute effective identity count** is. Going back
for 279 windows/user would raise it perhaps 35% at 1.8x the download, disk and epoch time,
for the same 2020 identities. Identity count is the measured lever; windows per identity is
not.

**Sizing the slice.** At ~53MB per user, and needing to roughly double 419 identities to
clear the resolution floor:

| users | ~size | corpus | vs now |
| --- | --- | --- | --- |
| 500 | 26GB | 919 | 2.2x |
| **2000** | **104GB** | **2419** | **5.8x** |
| 5000 | 259GB | 5419 | 12.9x |

**2000 users is the target**: 5.8x the identity count at ~100GB, comparable to the 7x
jump that turned the behavioural signal from chance into measurable. Select users with
two or more recordings so cross-session positives hold, and take a handful of recordings
each rather than all ~45 - identities per byte is what matters, not recordings per user.

**How big an acquisition has to be to be worth converting.** Identity count is the
binding constraint, but the power analysis sets a floor on what is worth chasing: 48 ->
343 identities (7x) turned the behavioural signal from chance into clearly measurable,
while 419 -> 463 (1.1x) is far below anything a paired 5-fold test can resolve. **An
acquisition needs to roughly double the corpus - ~400+ new identities - to justify the
conversion risk.** In XR biometrics only BOXRR-23 clears that; the rest of the field is
15-100 users per dataset because that is what a lab study yields.

Searched and rejected, recorded so the search is not repeated:

| dataset | verdict |
| --- | --- |
| **GazeBaseVR** | **Disqualified on content, not access.** 407 participants, CC-BY, trivial figshare download - and **no head position or orientation channel at all**. Participants were on a chin rest specifically to suppress head movement, and gaze is expressed as an angle relative to a fixed headset (Lohr et al. 2023, Table 4). The attractive access profile means someone will propose it again; it is the wrong signal, not the wrong licence. |
| Liebers et al. | No confirmed public release; 16 users maximum. |
| OpenNEEDS | 44 users, correct signals (head+hand+gaze), but gated behind a request process to Meta. |

**BOXRR-23** (105,852 users, ~5.35TB) is the largest identity source available and
identity count is our binding constraint, so it is the highest-value acquisition on the
board. The user has agreed to its Data Use Agreement and has confirmed with their advisor that ethics approval is in place, satisfying clause 9. **That agreement carries ongoing
obligations, not just an access checkbox** - it is a HIPAA-style Limited Data Set
agreement from UC Berkeley's Office of Technology Licensing, and these outlive the
download:

| clause | obligation |
| --- | --- |
| 9 | IRB or equivalent ethics approval **in advance of use** - a precondition, not a promise |
| 4 | no further distribution without written consent; requests referred back to Berkeley |
| 5 | **mandatory citation of Nair et al. 2023** in any public disclosure |
| 10-11 | no deanonymization, no contacting subjects, no inferring sensitive attributes |
| 13 | recipient indemnifies UC Berkeley |
| 15 | Berkeley may terminate; all copies must then be destroyed, **including derived ones** |

Consequences to build around rather than remember:

- Clause 15 puts **`.cache/samples/` in scope at EVERY resolution** - cached windows are
  derived copies, entries are keyed per user per `sample_time`/`sample_rate`, and each
  new combination writes its own set. Destruction means all of them on every machine,
  not one named directory.
- Clause 4 makes moving BOXRR-derived data between our three machines an open question.
  Convert wherever the raw data lands; do not centralise then copy.
- Clause 5 means the citation must travel with the data, not live in someone's memory.
- Format is `.XROR`, one tarball per user, with a BSON metadata index that allows
  selecting users by id before downloading - so a slice of N identities is possible
  without chunk boundaries. Official reader: `github.com/MetaGuard/xror`.

Take the **HMD track only**; the controllers are present and head-only is a deliberate
project constraint. An earlier note here claiming "106 chunks of ~1,000 users at ~45GB"
came from an automated page summary and was **wrong** - the repository shows per-user
tarballs.

**who-is-alyx** has its own converter, `prepare_who_is_alyx.py` (`--inspect` first, then convert). Worth knowing about the source: rotation columns are ordered **w,x,y,z** where this pipeline uses x,y,z,w, position is in **centimetres**, and `delta_time_ms` (not `timestamp`) is milliseconds since session start. Columns are read by name and the converter checks the mean quaternion norm, because a silent reordering produces a plausible-looking rotation. 76 players / 146 sessions / 6.74GB raw; most players have two ~45-minute sessions, so nearly all of them can form cross-session positives — unlike NJIT_6DOF.

Getting any *other* new dataset to that layout is still the weakest link:

- `formatter.py` expects `datasets/<name>/parser.py` exposing `parse(dataset_path)` yielding `(user_id, task_id, df)`, and writes to `datasets/<name>/processed_data/users/`.
- That directory does not exist on `main` and neither does any parser — the eight working parsers were removed in commit `6421567` ("Data seperation") and survive only in git history (`git show normalization:datasets/<name>/parser.py`; the `normalization` branch is an ancestor of `main`, not pending work).
- `formatter.py`'s output path (`datasets/<name>/processed_data/users/`) does not match where the model reads from (`processed_datasets/<name>/users/`), so onboarding a dataset ends with a manual move.

### Data condition (audited)

| dataset | users | native Hz | notes |
| --- | --- | --- | --- |
| who_is_alyx | 76 | 22–98 (capped to 60 on conversion) | 70/76 have two ~45-min sessions, so nearly all can form cross-session positives |
| Head_and_Gaze | 100 | 120 | **half the files (2630 `V1_*`) have no quaternion** — gaze rays only |
| PanoSaliency | 99 | 16.5 | 25 single-row sessions (zero duration); below 20Hz |
| VR_User_Behavior | 48 | 89.5 | the default dataset |
| ViewGauss | 35 | 10.1 | well below 20Hz |
| EyeNavGS | 22 | 125 | |
| Panonut360 | 21 | 94 | |
| NJIT_6DOF | 18 | 250 | room-scale walking, position range 5.13m |

Quaternions are unit-norm everywhere and there are no non-finite values. `UserProfile` skips files that are missing required columns, have fewer than two rows, are non-finite, or have non-positive duration, and reports the counts — before this, one bad file raised `KeyError` and took down a whole dataset (which is what made Head_and_Gaze unusable).

### `window_stride`: how often a window starts

Windows were always laid back-to-back: a 180s session at `sample_time=2` gives 90
windows sharing no frames. `window_stride` (seconds, `null` = `sample_time` = the
original behaviour) sets the gap between consecutive window *starts*, so a smaller
value overlaps them. Measured on PanoSaliency at `sample_time=2`: 90,212 windows at
the default, 179,558 at `window_stride=1`.

**Why it matters now.** Our windows are 2s where published results use 10-60s, which
makes window length the most obvious untested lever - but raising `sample_time` to 10
also cuts the window count 5x, so "longer window" and "less training data" move
together and the experiment answers neither. A stride decouples them: `sample_time=10,
window_stride=2` keeps roughly today's example count at five times the context.

**The guard is part of the feature, not a follow-up.** Two windows overlapping by 80%
share most of their frames, so a positive pair drawn from them is close to a
self-match: trivially easy, and *invisible*, because held-out positives would be
inflated identically and no train/test gap would appear. That is the same shape as the
same-session shortcut (worth ~1.5 points) and the cross-dataset shortcut (worth 11).
So `generate_pair_manifest` refuses to pair two windows of one session whose starts
are closer than `sample_time`; windows from different sessions can never share frames
and are unaffected. `test_no_positive_pair_ever_shares_frames` asserts it.

This also removed a defect present at *every* stride: `x1` and `x2` were drawn
independently with replacement, so a positive pair could be a window paired with
itself with probability 1/n. Rare (~1% at typical window counts) but free accuracy,
and it means runs from before this change are not bit-comparable to runs after it.

Per-window start times live in `SampleIndex.window_start_times` (sample cache v4).
**`None` when unavailable, never zeros** - zero is a legitimate start time, and
all-zero times read as "every window begins at t=0", which marks every same-session
pair as a total overlap and silently deletes same-session positives. That bug was
written and caught by the fixture during this change; keep absent distinguishable from
present-and-zero.

Prediction, recorded before measuring: modest and possibly negative on its own.
Overlapping windows are correlated, so 5x the windows is nowhere near 5x the
information, and correlated examples can overfit faster. Honest expectation at fixed
`sample_time`: **-0.01 to +0.02**. Its value is in making the `sample_time` sweep
interpretable, not in the extra windows.

### Window length: measured, real, and far too small to be the gap

5 arms, 30 runs, 5 stratified folds, 419 identities, `bilstm`, `identity_softmax`. The
design separates window length from example count, because raising `sample_time` also cuts
the window count unless a stride compensates:

| arm | `sample_time` | `window_stride` | selected AUC |
| --- | --- | --- | --- |
| A | 2 | 2 | 0.7146 +-0.013 |
| A | 2 | 2 | **0.4980** (random control) |
| B | 5 | 5 | 0.7248 +-0.019 |
| C | 5 | 2 | 0.7222 +-0.009 |
| D | 10 | 10 | **0.7331** +-0.018 |
| D | 10 | 10 | **0.4989** (random control) |
| E | 10 | 2 | 0.7296 +-0.014 |

**The control is flat across `seq_len`** - 0.4980 at 2s against 0.4989 at 10s - so the
floor does not move with window length and cross-arm comparison is valid. That is what the
controls on the two extreme arms were for.

Paired by fold:

| contrast | 2s -> 5s | 5s -> 10s | 2s -> 10s |
| --- | --- | --- | --- |
| non-overlapping (A/B/D) | +0.0102, t=2.76, 5/5 | +0.0082, t=0.94, 4/5 | **+0.0185, t=2.91, 5/5** |
| constant count (A/C/E) | +0.0076, t=2.58, 5/5 | +0.0073, t=1.43, 4/5 | +0.0149, t=2.76, 4/5 |
| redundancy only | 5s stride 5->2: **-0.0026**, t=-0.42 | 10s stride 10->2: **-0.0035**, t=-0.88 | |

**Three conclusions.**

1. **Longer windows help, and it is length rather than example count.** The gain appears
   along *both* tracks - with the window count falling (A/B/D) and with it held constant
   (A/C/E) - which is what the 5-arm design existed to separate.
2. **Overlapping windows buy nothing.** Both redundancy contrasts are flat-to-negative and
   neither is resolved. The stride bought the ability to *ask* the question, not an
   improvement - exactly the prediction recorded before `window_stride` was written.
3. **The effect is marginal and saturating.** +0.019 AUC from 2s to 10s at t=2.91 on 5
   folds sits right at the edge of what the power table says is resolvable, and the
   5s->10s step alone (+0.008, t=0.94) is not resolved at all.

### Window length is retired as the explanation for the identification gap

It was the leading candidate, and it cannot carry the weight. Going from our 2s to the
published 15s is worth roughly **+0.02 AUC**; the shortfall to explain is about **0.2 of
rank-1**. Even granting that identification is more sensitive to window length than
verification, that is an order of magnitude short.

What remains, in the order worth investigating:

1. **The sensor set, which is a scope decision and not a deficiency.** Every published
   comparison uses head **plus both controllers**; we are head-only so the model runs on
   glasses. This is likely the largest single term and we are not going to change it.
2. **Gallery composition.** Their 17 users are one dataset and one activity; our 62 are
   seven pooled corpora. Note this probably cuts *against* us rather than for us - a
   gallery spanning several capture setups may be easier to rank within than one drawn
   from a single session of a single study, so our N=17 figure may be flattered rather
   than penalised.
3. **`gallery_k=8` is our choice, not theirs.** Enrolment dominates probe, and their
   enrolment protocol differs from the default we happened to pick.

Model changes come after those three, not before.

### `resample`: how a window is built from raw frames

`Sampler` originally took the **nearest raw point** to each target time, which fails in
both directions. Below a dataset's native rate it returns the same row repeatedly, so
derived velocity is zero for those steps; above it, it keeps one row in twelve for a
250Hz source and folds the rest in as aliasing. Both matter directly for `brv`/`bra`,
which are computed from consecutive frames.

`resample: bin` averages every raw sample inside each target interval and interpolates
intervals that contain none - an anti-aliasing filter for the first failure, the right
answer for the second. Measured, exact duplicate consecutive frames at `sample_rate=20`:

| dataset | native | `nearest` | `bin` |
| --- | --- | --- | --- |
| ViewGauss | 10Hz | **50.5%** | **0.0%** |
| PanoSaliency | 17Hz | 27.0% | 8.1% |
| VR_User_Behavior | 89Hz | 7.5% | 1.7% |
| NJIT_6DOF | 250Hz | 0.0% | 0.0% |

Quaternions are put in a common hemisphere before averaging (q and -q are the same
rotation, so averaging across a sign flip cancels instead of smoothing) and
renormalized after; measured norm stays 1.0000.

Default is `nearest`, and **measurement says keep it** - `bin` lost on all four
encodings over 40 runs, worst on `raw`. See the screen above. The duplicate frames are
real; removing them costs more than they do, because averaging also removes
high-frequency content that carries identity. `bin` remains available and is the right
choice if a future result depends on honest velocity, but it is not the better default
and the "strictly better-conditioned input" reasoning was wrong.

### Normalization and negative sampling

Datasets share no coordinate frame: mean head height spans 0.00003 (Panonut360) to 2.89 (NJIT), and position range spans 40x. Two settings exist because of this, and they solve **different halves of the same problem**:

- `normalize` (`per_dataset` | `global` | `none`) — `model/normalization.py`. Standardizes each dataset's channels separately, removing cross-dataset offset and scale while preserving the *relative* differences between users within a dataset, which is where identity lives. Statistics are **fitted on training users only** and stored in the checkpoint, so `mode=test` applies the training-time transform instead of re-deriving it from held-out data. An unknown dataset at evaluation time falls back to fitting on the target data and says so.
- `within_dataset_negatives` — negatives are drawn only from users in the same dataset. A positive pair is always the same user and therefore always the same dataset, so pooling six datasets makes **79% of negatives cross-dataset**; raw mean-position distance then answers "different user?" for 71% of training pairs. No-op for a single dataset.

Measured on six datasets (238 identities, evaluated on the same 5 held-out users throughout): 0.576 raw → 0.643 with standardization → **0.687 with both**. Training accuracy falls from 0.936 to 0.830 as the shortcut disappears and the model is forced onto the real task.

## Outputs and the `auto` path trap

`hydra.job.chdir: true`, so every run `cd`s into `runs/YYYY-MM-DD/HH-MM-SS_<mode>/` and all **relative** paths resolve *inside that run directory*. Consequences worth knowing before debugging a "missing file":

- `save_path: auto` → `checkpoints/{experiment}_{datasetTag}_{sample}s_{rate}hz_emb{dim}_{mode}.pth` inside the run dir. There is no top-level `checkpoints/`.
- `model_path: auto` **cannot work in test mode** — it resolves into the freshly created, empty test run dir, and the stem ends in `_test` while training wrote `_train`. Always pass an explicit absolute `model_path` for `mode=test`.
- `boosting.artifact_root: boosting` is likewise relative, so `boosting.resume` never finds prior state across runs. Resume requires an absolute `artifact_root`.
- Stdout is still not captured: metrics are `print`ed, so Hydra's per-run `main.log` files remain empty. Per-run results are now appended to `results/runs.csv` instead (see below); per-epoch history still lives only in checkpoint `history` dicts and PNG plots. To recover a past result: `torch.load(ckpt, map_location='cpu', weights_only=False)['history']`.

## Input encodings

`encoding` (`raw` | `br` | `brv` | `bra`) transforms windows in the **data layer**, so every extractor sees the same input. `model/input_encoding.py`.

**MEASURED, and the literature ordering does not hold here.** 40 runs, `bilstm`, 5
stratified folds, 419 identities, all 8 datasets (sweep `6cc4e6f506`,
`eval_positive_fraction` 0.500 throughout):

| encoding | resample | selected AUC | selected acc |
| --- | --- | --- | --- |
| **raw** | **nearest** | **0.7284 +-0.018** | **0.6727 +-0.018** |
| raw | bin | 0.7086 +-0.011 | 0.6544 +-0.013 |
| bra | nearest | 0.5970 +-0.009 | 0.5697 +-0.005 |
| brv | nearest | 0.5796 +-0.010 | 0.5553 +-0.007 |
| br | nearest | 0.5713 +-0.013 | 0.5472 +-0.008 |

**raw >> bra > brv > br**, against the published **raw < br < brv < bra**. raw beats the
best alternative by 0.13 AUC - roughly 7x the largest fold sd in the table, so this is
not a spread artefact. Our existing default was already the best of the eight.

Why the inversion is plausible rather than suspicious: **~78% of what this model does is
absolute head position**, and `br`/`brv`/`bra` all remove exactly that. The published
ordering comes from setups with controllers and a real body frame, where the
body-relative encodings preserve information ours cannot reconstruct from a head alone.
Removing the dominant cue costs more than the derived kinematics return.

**This retires the confound rather than resolving it in the feared direction.**
Architecture was measured across backbones that did not share an encoding, so
"architecture is worth ~0" could have been an encoding effect in disguise. Encoding
turns out to matter a great deal - and we were already at its maximum, so the
architecture finding stands.

### `resample=bin` loses, and the duplicate-frame story does not survive

Paired by fold, `bin - nearest` on AUC: raw **-0.0198** (t(4)=-2.15), br -0.0117
(-2.33), brv -0.0060 (-1.73), bra -0.0040 (-1.74). **bin lost 1/5 folds on every arm.**

The prediction recorded before the run was that `bin` would help `brv`/`bra`
substantially and `raw`/`br` barely, because only the delta encodings read consecutive
frames. **The observed interaction is the opposite in both sign and rank**: bin hurts
everything, hurts `raw` *most* and `bra` *least*. The consistent reading is that
bin is a low-pass filter that removes high-frequency content carrying identity, and the
delta encodings lose least because they had already discarded most of it.

So the 50.5% duplicate frames are real and are **not** what was holding `brv`/`bra`
back. Keep `resample=nearest` as the default. This does not vindicate nearest-sampling
on principle - it says the encodings that depend on consecutive frames lose here to one
that does not, for a more basic reason than sampling.

**Why it is an axis and not an extractor detail.** The result "extractor architecture is worth ~0, spread under one point across three backbones over ten folds" was measured across backbones that do not share an encoding: `bilstm` and `paper_gnn_bilstm` consume raw channels while `motion_tdnn` derives kinematics internally. Architecture and encoding varied together, so that experiment cannot separate them. The literature runs the comparison the other way — architecture fixed, encoding varied — and reports **raw < br < brv < bra**. Neither result answers the other. One sweep over {extractors} × {encodings} with `sweep.folds` answers both.

It also bears on the anthropometry finding: `center_position` removes the window's mean position but leaves absolute orientation, so the movement-only arm was measured in roughly the weakest encoding available. A movement-only result at `bra` is a different claim from one at raw-centred.

**Head-only approximation.** The published body-relative encodings derive a body frame from head *and* both controllers; this corpus is head-only. So `br` is pose relative to the window's first frame — orientation as `q0⁻¹·q_t`, position rotated into `q0`'s frame. That is up-axis agnostic, which matters because the up axis differs across this corpus and anything yaw-based would be guessing. `brv` and `bra` are frame-to-frame deltas, already invariant to absolute pose.

Channel count is preserved (7 stays 7, 3 stays 3) so every extractor contract holds. The rotation block of a velocity encoding is the **delta rotation** — still a unit quaternion, sign-normalized for the double cover — not a componentwise difference of two quaternions.

The module is called `input_encoding`, not `encodings`: that name belongs to a stdlib package the interpreter loads at startup, and a local module of that name is silently shadowed.

## Channel sets

`channels` selects what a window is built from: `full` (quaternion + position, 7 channels, the original) or `position` (3 channels).

**Why `position` exists.** Requiring quaternion discards 2814 sessions — **48% more data than the pipeline uses** — because much of this corpus records head position but no orientation. Measured: `channels=position` takes Head_and_Gaze from 28,661 to **57,344 windows** (the 2630 `V1_*` files, same 100 users as `V2_*`, so roughly double the windows per identity) and recovers all 13 users of `360_em_dataset`, which is otherwise 100% unusable.

Orientation also measures as a weak identity cue: mean position separates held-out users at **0.768** AUC against **0.529** for quaternion statistics. **Both figures come from the old protocol** (the easy fixed 5-user split, same-session positives) and are optimistic — see the trivial-baseline table below for the corrected numbers. The ordering has held up, but dropping orientation is an experiment, not an assumption, which is why this is a switch and `full` remains the default.

The channel set is part of the sample-cache key and is stored in the checkpoint, so evaluating a position-only model never silently receives 7-channel windows.

**Extractors must honour `self.num_channels`.** `bilstm` and `random` do. `motion_tdnn` and `paper_gnn_bilstm` assume the 7-channel layout — `motion_tdnn` slices `x[:, :4]` as quaternion, `paper_gnn_bilstm` builds a fixed 10-node graph — and cannot run position-only until they read their own channel count. `fe.create()` probes any non-7 channel count with one tiny forward pass and raises a message naming the extractor and what to do, rather than letting an `IndexError` surface from inside the forward pass.

## Training objectives

`objective` selects how the extractor is trained. Both save the same
`forward(x1, x2) -> logit` interface, so evaluation, `mode=test` and the results table
stay comparable.

- **`pair_bce`** (default, original) — BCE over `Linear(|e1 - e2|)`. Every weight is
  tied to an embedding dimension shaped by the training identities, which is a route
  to memorising who is who; train 0.93 against held-out 0.68 is what that looks like.
- **`identity_softmax`** (`model/identity_train.py`) — classify *which user* each
  window belongs to with an additive angular margin (AM-Softmax), then compare
  embeddings by cosine. Uses every **window** as an example rather than every pair
  (~100k windows vs ~100k pairs, against a target with far more structure than one
  bit), and learns no per-dimension weights. This is how speaker and face
  verification are trained, for exactly this generalisation reason.

The AM-Softmax classifier is discarded after training; only the extractor plus the
cosine head is saved. `identity_softmax` forces `head=cosine` — scoring
angular-margin embeddings with a learned linear layer over `|e1 - e2|` would throw
away the structure the objective just created.

**Calibration matters here.** Cosine ranks well but says nothing about where the
accept threshold belongs, and accuracy is read at `logit > 0`. After every epoch the
cosine head's two scalars are refitted on *training* pairs with the extractor frozen.
Skip it and AUC looks fine while accuracy sits at chance for the wrong reason.

`head` is independently selectable (`diff_linear` | `cosine`) for pairwise training.
`diff_linear` keeps the original `classifier.*` parameter names so older checkpoints
still load. Identity training is standard-mode only; boosting stays pairwise.

## The 20-epoch budget is wrong in both directions

Measured over **304 recorded runs** that trained the full 20 epochs, the
validation-selected epoch:

| | |
| --- | --- |
| median | **7** |
| p75 / p90 / p95 | 13 / 18 / 19 |
| selected the final epoch | 4% |

So most runs peak early - `identity_softmax` has a median of 5 - and a long tail is
still improving when training stops. Both naive readings are wrong:

- **"Train longer"** helps only the ~5% censored by the cap.
- **"Truncate to save time"** is worse than it looks. Stopping at 12 epochs saves 40% of
  training but would cost **27% of runs** their selected epoch. At 8 epochs it is 40% of
  runs. The median is not the number that governs this - the tail is.

`early_stopping_patience` handles both: early peakers stop, late ones keep going. 0 by
default, so nothing already recorded changes.

**Do not use it on an axis nobody has characterised yet.** Patience truncates
slow-converging runs, so if a treatment converges slower - a lower margin, a smaller
scale, a longer window - its runs are exactly the ones cut short, and the sweep reports
that setting as worse when it was only stopped earlier. **That is a bias correlated with
the treatment**, which is worse than spending the wall clock, and it would be invisible
in the results table. Use patience on axes whose epoch distribution is already known to
sit well inside the cap; on a first look at a new axis, leave it at 0.

**The recorded distribution is right-censored and this is not a detail.** p90 is epoch 18
against a cap of 20 and 5% select 19 or 20, so for the top decile we do not know what
epoch those runs would have chosen with room to run. Every "best epoch" statistic above
is therefore a lower bound, and some fraction of our existing results may simply be
under-trained. Raising `epochs` to 30 decensors it, and is the right move for a first
look at a new axis - it costs 50% more per run and removes a bias rather than trading
one for another.

## We spend a quarter of our identities choosing an epoch

`val_user_fraction: 0.25` holds out a group of *training* users to select the epoch,
which is what makes `selected_test_acc` honest. It is also expensive in the one currency
this project is short of. At 419 identities over 5 folds:

| `val_user_fraction` | test | validation | **training** |
| --- | --- | --- | --- |
| 0.25 (current) | 83 | 84 | **252** |
| 0.15 | 83 | 50 | 286 |
| 0.10 | 83 | 34 | **302** |
| 0.0 (dishonest) | 83 | 0 | 336 |

**We train on 252 of 336 available identities.** Identity count is the only data-side
lever ever measured to work here - 48 to 343 is the difference between a behavioural
signal and no signal at all - and a quarter of the pool is going to a decision that
picks one number out of about twenty.

Two ways to get them back, neither tried:

1. **Cheap: lower the fraction.** 0.25 -> 0.10 returns 50 identities to training, a 20%
   increase, at the cost of selecting on 34 users instead of 84. Whether that is a good
   trade is empirical and depends on how flat the epoch curve is near its top - if the
   top few epochs are within noise of each other, a noisier choice among them costs
   almost nothing. 3 configs x 5 folds = 15 runs.
2. **Principled: refit on train+val at the chosen epoch.** Standard practice - use the
   validation users to pick the epoch count, then retrain on all 336 identities for that
   many epochs and report on the untouched test users. Selection stays honest, the test
   set is still never seen, and *no identity is spent*. Costs roughly double the
   training time per configuration.

The second is the right answer if identity count really is binding, and it also makes
the first unnecessary. Worth testing (1) first, because it is a config change and its
result tells you how much (2) could possibly be worth.

## Tried and measured at zero: adaptive score normalization

`model/score_norm.py`. Accuracy is read at a fixed `logit > 0` threshold, which assumes
one operating point serves every identity - and it does not, since some sit in a dense
part of the space and score high against everyone. AS-Norm is the standard fix in
speaker verification: rescale each score by how surprising it is for the two sides
involved, using the top-k similarities of each against an impostor cohort. It needs no
retraining and no new data, only embeddings already computed.

**Measured, and it does nothing here.** Spare `pair_bce` checkpoint, 100 unseen
Head_and_Gaze users, cosine scores, cohort built per-identity so window-count imbalance
cannot dominate it:

| cohort | best dAUC |
| --- | --- |
| training users, different dataset | **-0.0014** (negative at every top_k) |
| domain-matched, users disjoint from the trials | **+0.0025** at top_k=200 |

+0.0025 on 13,200 pairs is inside the binomial error alone (~0.008), let alone the
0.037 fold spread. Both readings are zero. The one real signal is that a cohort from a
*different dataset* is actively worse than none, which is consistent with the
cross-dataset normalization problems this corpus has everywhere else.

**Do not spend sweep runs on this.** The one caveat worth keeping is that it was
measured on the checkpoint whose own control scores 0.208 on users it was trained on -
AS-Norm exploits embedding geometry, and that space is barely organised, so there may
be nothing there to exploit rather than nothing to gain. It is post-hoc and costs zero
training runs, so it is worth one line of curiosity next time a properly trained
`identity_softmax` checkpoint is scored, and nothing more than that.

## Two untuned levers on the objective

The objective is the only thing measured to give a large gain (+6.5). Both of these
sit inside it and neither has ever been varied.

### `identity_margin` / `identity_scale` were never swept

AM-Softmax's margin (0.35) and scale (30.0) are the defaults, unchanged across **312
recorded runs** - they were not even columns in the results log until now. These are
the two hyperparameters that decide how hard the objective pushes identities apart, and
0.35/30 are the values the face-recognition literature tuned against corpora with tens
of thousands of identities. We have 343. There is no reason to think the same setting
is right, and it is the cheapest untested thing on the board.

### Window counts per identity span 77x, and that costs ~38% of our identities

`WindowDataset` is flat over windows and the loader shuffles uniformly over them, so an
identity's influence on the gradient is proportional to how much data it happens to
have. **Measured on AVALON, which holds the full corpus** (5s@20Hz, effective =
`sum^2 / sum-of-squares` of the per-identity window counts):

| | pre-BOXRR | with BOXRR |
| --- | --- | --- |
| real identities | 419 | **2439** |
| windows | 216,951 | 532,084 |
| min / median / max | | 0 / 158 / 1260 |
| **effective identities** | **254.0 (60.6%)** | **1138.9 (46.7%)** |

**Effective identities rose 4.48x** - the real gain from BOXRR - while the effective
*fraction* fell from 60.6% to 46.7%, because 2020 fairly uniform users at ~156 windows
sit beside a long right tail reaching 1260. Adding balanced identities next to an
unbalanced corpus does not rebalance it.

**The 419 pre-BOXRR identities are 17.2% of the corpus and hold 40.8% of all windows.**
Under uniform window sampling they therefore supply four times their share of every
epoch's gradient, and the 2020 new identities are correspondingly under-weighted.

*An earlier version of this section said "190 effective of 312". That was measured on
the coordinator's laptop, which holds 8 datasets and **not** `who_is_alyx` - a
343-identity corpus, not 419. The numbers above supersede it, and the discrepancy is
exactly the per-machine hazard this file warns about under "Data".*

The effective count is the inverse participation ratio: the number of *evenly
represented* identities the corpus is worth under uniform window sampling. **We are discarding
about 39% of our identity diversity to sampling imbalance** - on the one axis that has
been measured to bind, and for free, without needing a single new user.

AM-Softmax with imbalanced classes separates frequent identities well and rare ones
poorly, which is the wrong trade when the entire task is generalising to identities
never seen at all.

`balance_identities: true` draws each window with probability inversely proportional to
its identity's count, keeping the epoch the same size. Off by default.

**The first pilot of this was invalid and its result must not be cited.** It ran on the
config's default `data_dirs`, which is **VR_User_Behavior alone** - a corpus where every
identity has 1049 or 1050 windows, `max/min` = **1.0x**, effective identity count
**48/48 = 100%**. There is no imbalance there to correct, so balanced sampling is a
mathematical no-op and the only thing it can contribute is resampling-with-replacement
noise. The -0.030 it produced measured that noise, not the hypothesis.

This is the same failure shape as the `mode=curve` split fallback: a config default
silently standing in for the intended experiment, producing a plausible number with
nothing to flag it. **Pass `data_dirs` explicitly for any pooled-corpus run** - the
default is single-dataset and always has been.

**The premise now holds on the real corpus, and did not when this was piloted.** Post-
BOXRR the imbalance is measured, large and structurally lopsided: 17.2% of identities
hold 40.8% of the windows. Worth an arm in the first full-corpus sweep.

`balance_identities` takes **`off` | `weighted` | `cap`** (`false`/`true` still mean
off/weighted, so existing configs are unchanged):

- **`weighted`** - inverse-frequency, **with replacement**. Equalises identities but
  lowers the number of *distinct* windows seen per epoch, because a 1260-window identity
  gets drawn far fewer times than it has windows. This is the form that was piloted and
  the mechanism that made it suspect.
- **`cap`** - take at most `balance_cap` windows per identity per epoch, without
  replacement, defaulting to the median so identities above it are trimmed and those
  below are untouched. Every window in an epoch is distinct, and the epoch gets
  **cheaper** rather than more expensive. A fresh subset is drawn each epoch, so the
  surplus is trimmed rather than permanently discarded - over many epochs a large
  identity still contributes all of its windows.

**Prefer `cap`.** It raises the effective identity count from the other direction, and
it is the variant that survives the objection to the first one.

**How to read the result, agreed in advance.** The power table says +0.005 to +0.02
needs 10-15 folds; adding this as a cheap arm on a 5-fold sweep makes the arm
affordable, not the comparison resolvable. If it lands inside +-0.02 with t under 2.8
the honest entry is **"not resolved"**, not "small gain" - and since it will be the arm
we hoped would work after acquiring 2020 identities, that is exactly where a small
positive would be over-read.
Until that lands this is **untested**, and the prediction registered beforehand
(+0.005 to +0.02) still stands unmeasured.

**Why it might still cost rather than pay**, predicted before either pilot: inverse-
frequency sampling draws *with replacement*, so a well-recorded identity's 1050 windows
get sampled far fewer times than they exist. Effective identity count rises while the
number of *distinct* windows seen per epoch falls. Fixing diversity by discarding data
may not be a trade worth making. If it does fail on the pooled corpus, the variants that
avoid the mechanism are sqrt-frequency weighting - the standard compromise - or capping
frequent identities without upsampling rare ones.

## Tried and measured at zero: adaptive score normalization

`model/score_norm.py`. Accuracy is read at a fixed `logit > 0` threshold, which assumes
one operating point serves every identity - and it does not, since some sit in a dense
part of the space and score high against everyone. AS-Norm is the standard fix in
speaker verification: rescale each score by how surprising it is for the two sides
involved, using the top-k similarities of each against an impostor cohort. It needs no
retraining and no new data, only embeddings already computed.

**Measured, and it does nothing here.** Spare `pair_bce` checkpoint, 100 unseen
Head_and_Gaze users, cosine scores, cohort built per-identity so window-count imbalance
cannot dominate it:

| cohort | best dAUC |
| --- | --- |
| training users, different dataset | **-0.0014** (negative at every top_k) |
| domain-matched, users disjoint from the trials | **+0.0025** at top_k=200 |

+0.0025 on 13,200 pairs is inside the binomial error alone (~0.008), let alone the
0.037 fold spread. Both readings are zero. The one real signal is that a cohort from a
*different dataset* is actively worse than none, which is consistent with the
cross-dataset normalization problems this corpus has everywhere else.

**Do not spend sweep runs on this.** The one caveat worth keeping is that it was
measured on the checkpoint whose own control scores 0.208 on users it was trained on -
AS-Norm exploits embedding geometry, and that space is barely organised, so there may
be nothing there to exploit rather than nothing to gain. It is post-hoc and costs zero
training runs, so it is worth one line of curiosity next time a properly trained
`identity_softmax` checkpoint is scored, and nothing more than that.

## Two untuned levers on the objective

The objective is the only thing measured to give a large gain (+6.5). Both of these
sit inside it and neither has ever been varied.

### `identity_margin` / `identity_scale` were never swept

AM-Softmax's margin (0.35) and scale (30.0) are the defaults, unchanged across **312
recorded runs** - they were not even columns in the results log until now. These are
the two hyperparameters that decide how hard the objective pushes identities apart, and
0.35/30 are the values the face-recognition literature tuned against corpora with tens
of thousands of identities. We have 343. There is no reason to think the same setting
is right, and it is the cheapest untested thing on the board.

### Window counts per identity span 77x, and that costs ~38% of our identities

`WindowDataset` is flat over windows and the loader shuffles uniformly over them, so an
identity's influence on the gradient is proportional to how much data it happens to
have. Measured on the pooled 7-dataset corpus, at both window lengths because window
count is `floor(duration / sample_time)` and a short session can round down to nothing:

| | `sample_time=2` | `sample_time=5` (what the sweeps run) |
| --- | --- | --- |
| identities with windows | 312 | 312 |
| windows per identity | 34 / 777 / 2639 | 12 / 295 / 1050 |
| max/min | 77.6x | **87.5x** |
| top 10% hold | 23.6% | 23.9% |
| bottom 50% hold | 19.1% | 18.6% |
| **effective identity count** | 193 of 312 | **190 of 312** |

No identity drops out at the longer window, and the imbalance is marginally *worse*
there, so the effect is a property of the corpus rather than of one window length.

The last row is the inverse participation ratio: the number of *evenly represented*
identities this corpus is worth under uniform window sampling. **We are discarding
about 39% of our identity diversity to sampling imbalance** - on the one axis that has
been measured to bind, and for free, without needing a single new user.

AM-Softmax with imbalanced classes separates frequent identities well and rare ones
poorly, which is the wrong trade when the entire task is generalising to identities
never seen at all.

`balance_identities: true` draws each window with probability inversely proportional to
its identity's count, keeping the epoch the same size. Off by default.

**Piloted: no measurable benefit at either window length, and a possible cost at the one we run.** 3 stratified folds on the pooled corpus,
`identity_softmax`, `bilstm`, paired by fold (laptop pilot: `batch_size=256`, so the
absolute numbers do not sit beside the main sweeps - the paired difference does):

| | fold diffs (AUC) | mean | won |
| --- | --- | --- | --- |
| `sample_time=5` (what we run) | +0.008, **-0.070**, -0.027 | **-0.0296** | 1/3 |
| `sample_time=2` | +0.003, +0.042, -0.032 | +0.0042 | 2/3 |

The prediction registered beforehand was **+0.005 to +0.02**. The measured result at the
configuration we actually run is **-0.030**, past the -0.01 threshold agreed for taking
a negative seriously. **The two arms disagree in sign at a fold sd of 0.039**, so the
defensible statement is "no measurable benefit at either window length, and a possible
cost at the one we run" - *not* that balancing hurts. The mechanism below is what makes
the negative plausible; the measurement alone does not establish it. Either way there is
no case for spending 10 more runs hunting a positive.

**Why it plausibly costs rather than pays**, and this was predicted before the pilot ran:
inverse-frequency sampling draws *with replacement*, so a well-recorded identity's 1050
windows get sampled far fewer times than they exist. The corpus's effective identity
count rises, but the number of *distinct windows* the model sees per epoch falls. Fixing
diversity by discarding data is not obviously a trade worth making, and measurement says
it is not.

If anyone returns to this, the variant to try is **sqrt-frequency weighting** rather than
inverse - the standard compromise between uniform and balanced - or capping frequent
identities without upsampling rare ones, which raises diversity without resampling
anything. Neither is a priority.

## Tried and measured at zero: adaptive score normalization

`model/score_norm.py`. Accuracy is read at a fixed `logit > 0` threshold, which assumes
one operating point serves every identity - and it does not, since some sit in a dense
part of the space and score high against everyone. AS-Norm is the standard fix in
speaker verification: rescale each score by how surprising it is for the two sides
involved, using the top-k similarities of each against an impostor cohort. It needs no
retraining and no new data, only embeddings already computed.

**Measured, and it does nothing here.** Spare `pair_bce` checkpoint, 100 unseen
Head_and_Gaze users, cosine scores, cohort built per-identity so window-count imbalance
cannot dominate it:

| cohort | best dAUC |
| --- | --- |
| training users, different dataset | **-0.0014** (negative at every top_k) |
| domain-matched, users disjoint from the trials | **+0.0025** at top_k=200 |

+0.0025 on 13,200 pairs is inside the binomial error alone (~0.008), let alone the
0.037 fold spread. Both readings are zero. The one real signal is that a cohort from a
*different dataset* is actively worse than none, which is consistent with the
cross-dataset normalization problems this corpus has everywhere else.

**Do not spend sweep runs on this.** The one caveat worth keeping is that it was
measured on the checkpoint whose own control scores 0.208 on users it was trained on -
AS-Norm exploits embedding geometry, and that space is barely organised, so there may
be nothing there to exploit rather than nothing to gain. It is post-hoc and costs zero
training runs, so it is worth one line of curiosity next time a properly trained
`identity_softmax` checkpoint is scored, and nothing more than that.

## Two untuned levers on the objective

The objective is the only thing measured to give a large gain (+6.5). Both of these
sit inside it and neither has ever been varied.

### `identity_margin` / `identity_scale` were never swept

AM-Softmax's margin (0.35) and scale (30.0) are the defaults, unchanged across **312
recorded runs** - they were not even columns in the results log until now. These are
the two hyperparameters that decide how hard the objective pushes identities apart, and
0.35/30 are the values the face-recognition literature tuned against corpora with tens
of thousands of identities. We have 343. There is no reason to think the same setting
is right, and it is the cheapest untested thing on the board.

### Window counts per identity span 77x, and that costs ~38% of our identities

`WindowDataset` is flat over windows and the loader shuffles uniformly over them, so an
identity's influence on the gradient is proportional to how much data it happens to
have. Measured on the pooled 7-dataset corpus, at both window lengths because window
count is `floor(duration / sample_time)` and a short session can round down to nothing:

| | `sample_time=2` | `sample_time=5` (what the sweeps run) |
| --- | --- | --- |
| identities with windows | 312 | 312 |
| windows per identity | 34 / 777 / 2639 | 12 / 295 / 1050 |
| max/min | 77.6x | **87.5x** |
| top 10% hold | 23.6% | 23.9% |
| bottom 50% hold | 19.1% | 18.6% |
| **effective identity count** | 193 of 312 | **190 of 312** |

No identity drops out at the longer window, and the imbalance is marginally *worse*
there, so the effect is a property of the corpus rather than of one window length.

The last row is the inverse participation ratio: the number of *evenly represented*
identities this corpus is worth under uniform window sampling. **We are discarding
about 39% of our identity diversity to sampling imbalance** - on the one axis that has
been measured to bind, and for free, without needing a single new user.

AM-Softmax with imbalanced classes separates frequent identities well and rare ones
poorly, which is the wrong trade when the entire task is generalising to identities
never seen at all.

`balance_identities: true` draws each window with probability inversely proportional to
its identity's count, keeping the epoch the same size. Off by default so existing
comparisons stay like-for-like. **Untested - predict before running it.** Honest
expectation: this is the same *kind* of intervention as raising identity count, which
is the only data-side lever that has ever worked here, but 190 -> 312 effective is a
1.6x change where 48 -> 343 was 7x, so **+0.005 to +0.02** rather than anything
dramatic. It is cheap and it is on the right axis.

## Verification metrics

`model/metrics.py`. Accuracy is measured at the fixed `logit > 0` threshold, which conflates ranking quality with operating-point placement — a model can sit at 0.50 accuracy while still ranking pairs usefully. `evaluate(..., return_metrics=True)` adds:

- **ROC-AUC** — threshold-free ranking quality. Ties are rank-averaged, which matters because an untrained model emits a near-constant logit and naive AUC would report 0.0 or 1.0 depending on sort order.
- **EER** and its threshold — the standard biometric verification number, comparable across datasets with different pair balance.

Both are tracked per epoch into `history` (`test_auc`, `test_eer`) and recorded in `results/runs.csv` as `best_test_auc` / `best_test_eer`.

## Head-only is the scope, not a limitation

This project uses head motion alone - quaternion plus HMD position - because the target
is **all of XR, including XR/AR glasses, which have head tracking and no hands at all**.
A model that needs controller channels cannot run on that device class. This buys
generality; it is not a handicap being tolerated.

**It changes how three findings should be read**, and none of them are deficits:

1. **The comparison against published rank-1 figures.** Rack 2023, Schach 2026 and Nair
   2023 all use head **plus both controllers**. Their absolute accuracies bound a
   *different sensor set covering a narrower device class*. The gap between our 0.570 at
   N=17 and their 0.785 is therefore part scope and part performance, and the two cannot
   be separated by matching metric and gallery size alone.
2. **Why the literature's encoding order inverts here.** `br`/`brv`/`bra` derive a body
   frame from head *and* both controllers. A head-only rig cannot build one, so those
   encodings only strip the absolute position that carries most of our signal and return
   nothing in its place. Raw beat all three by 0.13 AUC over 40 runs; their own pipeline
   uses `BRV`. Same fact, both sides.
3. **Dataset selection.** Controller channels are never a reason to prefer a dataset, and
   a dataset that records only head pose is not thereby inferior for our purposes.

**Never** propose hand/controller or eye channels as a way to raise scores, and do not
attribute weak movement-only results to the missing hands.

## Identification vs verification (they are not the same number)

Everything this project reports as a headline is **verification**: given two windows,
same person or not - two classes, chance 0.50, and 0.669 is measured that way. Most of
the XR biometrics literature reports **identification**: given a probe, rank a gallery
of N enrolled users and check whether the right one is first - chance 1/N. A published
rank-1 of 78.5% and our 0.669 are not on the same scale and never were, so the gap
between them is not a gap in performance until both are measured the same way.

`cmc_curve()` in `model/templates.py` computes the second from the same embeddings the
k-curve already needs, so any existing checkpoint can be scored with no retraining.
`mode=curve` prints it beside the verification numbers and records `rank1` /
`gallery_users` per run.

- Gallery and probe come from **different sessions**, as everywhere else here, so a
  correct match cannot be session matching. Single-session users fall back to disjoint
  windows of one session and are counted, exactly as cross-session pairing does.
- Ties are **rank-averaged**, the convention `roc_auc` already uses and for the same
  reason: an untrained model emits a near-constant score, and breaking those ties by
  sort order would report either rank 1 or rank N for no information at all. A
  constant scorer lands at rank (N+1)/2.
- **Never quote rank-1 without N.** Chance moves with the gallery size, so rank-1 at 48
  identities and rank-1 at 419 are different questions. Both are in every row.

**Matching N is half the comparison, and it is the half that is easy to forget.** The
closest published leave-users-out result is **rank-1 closed-set identification over 17
unseen users on a single 15-second window: 83.1% within-application, 78.5% averaged
across applications** (`docs/LITERATURE_BRIEFING.md`, source X). Two mismatches with
ours, not one:

| | theirs | ours |
| --- | --- | --- |
| metric | rank-1 identification, chance 1/17 | pairwise verification, chance 0.50 |
| gallery | 17 users | 343-419 users |
| window | 15s | 2s |

So 0.669 against 0.785 was never a like-for-like gap. `gallery_sizes` reports rank-1
restricted to a random gallery of N users, averaged over draws, from the same scoring
pass - `[17, 48, 100]` by default, with 17 there specifically to sit beside that
result. Ranking against 17 candidates is an easier problem than ranking against 419,
and the difference is not performance.

The window length is the third mismatch and is the one that might be a real deficit;
that is what `window_stride` now makes testable.

### The identification number, measured properly

**rank-1 identification on unseen users, 5 retrained leave-users-out folds**
(`identity_softmax`, 343 identities, sweep `b732bee5c6`, evaluation split recovered
from each checkpoint):

| gallery | rank-1 | sd | chance | x chance |
| --- | --- | --- | --- | --- |
| **N=17** (matched to the published result) | **0.5700** | 0.0201 | 0.0588 | 9.7 |
| N=48 | 0.4172 | 0.0313 | 0.0208 | 20.1 |
| full (61-64) | 0.3852 | 0.0367 | 0.0160 | 24.1 |

Verification on the same embeddings, k=1: AUC 0.7427, EER 0.3112.

**Against the pre-registered band.** Before the number existed, 0.70+ was recorded as
"units were the story", 0.40-0.70 as "units explain part, a real shortfall remains, and
window length is the leading candidate", and below 0.40 as "units are not the story".
**0.570 falls in the middle band.**

So the units correction was worth a great deal and was not the whole story. Comparing
0.669 verification against a published 78.5% rank-1 was meaningless; the honest
comparison is **0.570 against 0.785 at matched metric and matched N**, and about 0.21
of gap survives it. The remaining uncontrolled difference is window length - theirs is
15s, ours 2-5s - which is what the window-length experiment tests.

**0.570 is an upper bound, not a point estimate.** 10-12 users per fold (~17% of each
gallery) have a single session, so their gallery and probe come from one recording and a
correct match there can be session matching rather than identification. `mode=curve` now
reports the cross-session-only figure beside it (`require_cross_session`); the gap
between the two is how much of rank-1 is session matching, and the true cross-session
number is below 0.570 by an unknown amount until that is run.

### The k-curve was not interpretable across k (fixed)

Worth recording because it is the fourth instance of one bug this project keeps
producing. Measured on fold 0 before the fix:

| k | pairs | AUC | users short of windows |
| --- | --- | --- | --- |
| 1 | 30720 | 0.7427 | 5 |
| 4 | 27136 | 0.7424 | 12 |
| 16 | 16384 | **0.6066** | **33** |

AUC appears to collapse with k. But eligibility required a session to hold k windows, so
the population collapsed with it - at k=16 more than half the enrolled users cannot
supply 16 windows and the pair count falls 47%. The k=16 row scores a smaller,
differently-composed set than the k=1 row, so the decline is at least partly the
population changing rather than averaging failing. The same defect made every asymmetric
`[k_ref, k_probe]` pair incomparable to every other.

`window_curve` now fixes the population once, from the widest k anywhere in the sweep,
and every row scores those same users. It costs the users who cannot supply that many
windows - a narrow curve keeps more users than a wide one - which is the price of the
rows meaning anything relative to each other.

**No k>1 number recorded before this fix should be quoted**, including the asymmetric
ones. The variance decomposition is unaffected because it is population-stable, and it
predicts averaging buys little here anyway: fold 0 gives between-user 0.5179,
between-session 0.1943, within-session 0.1761, signal/shift 2.67, plateau k~1. So the
broken curve was probably not hiding a large gain.

### First identification numbers (superseded, kept for the control it established)

Measured on a **weak checkpoint** - `motion_tdnn`, `pair_bce`, `diff_linear`, 5s/20Hz,
trained on 6 pooled datasets - scored on **Head_and_Gaze, which it never saw** (100
unseen users). This is a lower bound and a validation of the code, not a headline.

| enrolment | probe | rank-1 @ N=17 |
| --- | --- | --- |
| 10s | 15s | 0.222 |
| 20s | 15s | 0.261 |
| 40s | 5s | 0.204 |
| 40s | 15s | 0.269 |
| 40s | 30s | 0.315 |
| 80s | 15s | 0.401 |
| 160s | 30s | **0.434** |

Chance is 0.0588. Same checkpoint, same settings, on **seen, in-dataset** users scores
0.208 against 0.204 unseen - so the implementation is sound (monotone in evidence,
well above chance, ordered correctly across N) *and* this checkpoint's embedding space
is barely organised for identification at all, seen users included.

**Three things this establishes.**

1. **Verification and identification are wildly different numbers for the same model.**
   This checkpoint scores AUC 0.752 / acc@EER 0.69 on the same held-out data where its
   rank-1 at the full 100-user gallery is 0.032. Quoting one against the other, which is
   what comparing 0.669 to a published 78.5% was doing, is meaningless.
2. **Enrolment dominates probe, as the literature says.** 8 to 16 gallery windows buys
   +0.13; 1 to 6 probe windows buys +0.11 from a much lower base. `mode=curve` averages
   k on *both* sides, so the symmetric diagonal is the wrong operating point and an
   asymmetric `(gallery_k, probe_k)` is nearly free.
3. **Whether a real gap survives the units correction is UNTESTED.** At matched N=17
   and matched 15s probe this checkpoint reaches 0.269, saturating near 0.434 with far
   more evidence than the published setup used - but the control above says it scores
   0.208 on users it was *trained on*. A model that cannot rank its own training
   identities above 0.21 has an embedding space that is barely organised for this task,
   so its shortfall against 0.785 measures the checkpoint, not the pipeline. This run
   bounds nothing about the remaining gap; the `identity_softmax` run does.

**The objective is the leading hypothesis, and it is directly testable.**
`identity_softmax` is the similarity family the briefing identifies as correct, it is
already worth +6.5 on verification, and rank-1 is precisely what an angular-margin
embedding is trained to serve - so it should gain far more here than it did on
verification. Running `mode=curve` on the 343-identity
`identity_softmax` checkpoint is the single highest-value measurement outstanding.

`gallery_k` / `probe_k` / `probes_per_user` set how much evidence each side gets, and
`curve_k` accepts `[reference, probe]` pairs as well as bare values, so
`curve_k=[[16,1],[16,4],[4,16]]` sweeps the asymmetry directly. The reference side is
worth more, so the symmetric diagonal is not where the good operating points are.
Enrolment size is worth more than probe size in the literature, and the asymmetry is
free to test here.

## Results log

`model/results_log.py` records one run per line. Paths are absolute, anchored to the repo root so `job.chdir` can't misplace them.

**Where it lives (changed):**

| path | what it is |
| --- | --- |
| `results/runs/<machine>.jsonl` | the record. Append-only, one self-describing JSON object per run, one file per machine. |
| `results/runs.csv` | frozen history - every run before the switch. Nothing appends to it. |
| `results/runs_all.csv` | derived view of both, rebuilt after every run. Gitignored. **Read this for analysis.** |

`load_runs()` returns everything from the first two as dicts; `write_combined_csv()` produces the third.

**Why it is not one CSV any more.** The log is committed from three machines and merged with `merge=union`, which unions *lines* - but a CSV's meaning lives in a header those lines share, and this schema migrates by design. The moment two machines held different column counts (57 vs 56), union filed every row from one side under the other's header: 537 rows, 237 duplicated, `seed` 67 reading as 2, `run_dir` holding a git SHA, and 151 rows appearing to have a `template_k` that was pure column shift. Both inputs were individually clean; nothing was wrong until they met. Repaired by rebuilding on column *name*.

JSONL removes the class instead of patching it - a union of self-describing records is correct whatever schema either side used, adding a field is a non-event, and appending never rewrites a line. `run_id` makes every line unique so union can't coalesce two runs that agree on all fields. Tests cover the property, not just the writing: `test_union_merging_two_schemas_keeps_every_field_on_the_right_row` reproduces the exact merge that corrupted the file. **`sweep_id` is only a valid grouping key for rows written at or after `5b61fc0`.** Before that commit the id ignored every top-level config key, so rows from two different experiments can share one — in this file, the 48-identity subsample runs sit under `d6cb92c8a9` alongside the 343-identity pooled runs. They separate on `max_users` (blank vs 48), but grouping on `sweep_id` alone merges them. `sweep_id` also under-partitions for a second reason: runs made before and after a bugfix share it when the config is identical. Those separate on `code_identity`. When analysing rows that straddle that commit, group on the config columns (`max_users`, `objective`, `normalize`, `channels`, `center_position`, `cross_session_positives`, `num_data_dirs`) rather than trusting the id.

It covers all three paths — standard, boosted, and test — and records config (including `extractor` and `extractor_params`), metrics, checkpoint, run dir and git SHA (with a `-dirty` suffix for uncommitted trees). Changing `FIELDS` is safe: shards carry their own keys, so old lines are untouched and the combined view backfills blanks. (`FIELDS` is now the *column order* of the combined view plus the CSV writer that `results_path=...` still selects, not a constraint on what a line may hold.) Logging failures degrade to a warning and never abort a finished run. Add new columns to the end of `FIELDS` so existing files stay readable.

The 95 pre-existing runs under `runs/` are not in this file; they can be backfilled from checkpoint `history` dicts plus each run's `.hydra/config.yaml`.

## Sample cache

`model/sample_cache.py` caches each user directory's sampled windows to `.cache/samples/` (gitignored), keyed by CSV names/sizes/mtimes plus `sample_time`/`sample_rate`. Measured on the default 48-user dataset: **23.4s → 0.6s**, bit-identical output. Because both the train and eval index builds hit the same per-user entries, the double-load is now nearly free, and changing `exclude_users` invalidates nothing.

- Disable with `XRSEC_SAMPLE_CACHE=0`; relocate with `XRSEC_SAMPLE_CACHE_DIR`.
- Deleting `.cache/` is always safe. Entries for superseded signatures are never garbage-collected, so it grows across resolutions (~100MB for two resolutions of one dataset).
- The cache is only valid because sampling is deterministic (`Sampler` is always built with `index_randomness=0`). **If per-epoch index jitter is ever enabled, the cache must be bypassed** or it will freeze one fixed augmentation.

## Retired

- **boosting** — `boosting.enabled=true` refuses with an explanation. Best-round selection reads the set it reports (~+0.02 inflation, no `val_user_fraction` equivalent), `boosting.artifact_root` is relative so resume never worked, and it is pairwise-only so `identity_softmax` (+6.5) cannot apply. No recorded boosted run was ever competitive. Code stays in `model/boost_train.py` for reference.
- **the historical 0.85** — never reproduced, configuration lost, and the corrected protocol cannot account for it even stacking every known inflation. Not a target.

## Known-broken

- `model/validate.py` is dead: it imports `plot_training_history` from `train` (it lives in `utils`), calls `train()` with a dict shape that predates the current config, and assumes the old `datasets/*/processed_data/` layout.

Current baseline: **256 passing, ~10s**.

## GPU throughput

Measured on an RTX 3050 Ti, `bilstm`, seq_len 100, batch 256, one identity-training
step, warm:

| variant | ms/step | windows/s | vs baseline |
| --- | --- | --- | --- |
| baseline (per-batch `.item()`, CPU tensor) | 25.4 | 10,081 | 1.00x |
| **no per-batch sync** | **11.1** | **23,144** | **2.30x** |
| + samples resident on GPU | 10.2 | 25,028 | 2.48x |
| + AMP fp16 | 12.3 | 20,775 | **2.06x - AMP HURTS** |
| + batch 512 | 19.0 | 26,974 | 2.68x |
| + batch 1024 | 36.6 | 28,006 | 2.78x |

**The whole win is not stalling the pipeline.** `total_loss += loss.item()` and
`correct += (...).sum().item()` each force a device sync *every batch*, so the GPU sat
idle waiting for the CPU rather than queueing the next batch. Accumulating as device
tensors and reading once per epoch is **2.30x** for arithmetic that is mathematically
identical - applied to `train_epoch`, `train_identity_epoch` and `evaluate`.

**AMP is measured harmful here and should not be turned on.** fp16 cost 2.48x -> 2.06x
in isolation. These models are small (153k parameters for `bilstm`) and not
compute-bound, so the conversion overhead and `GradScaler` are not repaid, and cuDNN's
LSTM does not use tensor cores usefully at this size. It would also change numerics for
no gain.

**Two further gains exist and are not free**, so they are not applied by default:

- **Samples resident on GPU** is worth another 8%. The window tensor is ~0.4GB at 419
  identities and ~2.2GB at 2419, which fits on larger cards but not on a 4GB laptop, and
  it needs the batch-slicing path rather than per-item `DataLoader` indexing to pay off.
- **Larger batches** raise throughput ~12% from 256 to 1024, but batch size **changes
  the optimisation**, so it is an experiment rather than a speedup. Do not raise it to
  go faster and then compare against runs at 256.

Note the ms/step column rises with batch size while windows/s also rises - throughput is
the figure that matters for epoch time, not per-step latency.

## Performance notes

Keep `num_workers: 0` unless benchmarked: the whole sample tensor lives in memory inside the Dataset, and Windows spawn-based workers pickle it per worker.

Console output must stay ASCII. Windows consoles default to cp1252, so box-drawing characters raise `UnicodeEncodeError` as soon as stdout is piped or redirected — this crashed `mode=test` until it was fixed in `eval.py`.
