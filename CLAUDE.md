# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

XR biometric identification research. A Siamese network decides whether two windows of headset motion came from the same person. The research question is whether this generalizes to **users never seen during training**, so nearly every design decision (leave-users-out splits, pair generation, boosting) exists to serve that question.

## Coordination between sessions

Three sessions work this repo - a coordinator, a trainer, a generalisation session - on
DESKTOP-C, plus a data/storage session on AVALON and ad-hoc ones. As of 2026-09-04 the
coordinator runs on DESKTOP-C too, and **`SendMessage` between the on-machine sessions
works both ways**; use it for anything between them. **`docs/COORDINATION.md` is still
the channel for the AVALON session and for anything that must outlive a session**: read
it after every pull, append under your own heading to reply, delete resolved items. It
also carries the shared-working-tree rules and the GPU queue.

Git is the one channel every session provably shares, and a notice in the repo survives
a session ending, which a chat message does not.

## Current state - read this before quoting any number

**0.669** verification accuracy on unseen users (chance 0.50) is still what the pipeline
measures: `bilstm`, `identity_softmax`, cross-session positives, validation-selected epoch,
5 leave-users-out folds. It survived three protocol corrections and it is not wrong. But
four findings, all from the same day of auditing, change what it *means*:

| finding | consequence |
| --- | --- |
| Per-dataset held-out AUC is **0.93+** where a real head position exists and **~0.49** where the position column holds a unit direction vector | any pooled figure averages near-perfect verification with chance |
| A **training-free three-number lookup** (mean position, Euclidean distance) scores **0.726** where the model scores **0.723**, same folds, same manifests | most of what the pooled model does needs no model |
| The model beats that lookup **in domain by +0.18** on alyx (clean pair, t(4)=7.52) and **loses to it** on every unseen corpus | there IS a learned component, and it is exactly what fails to transfer |
| Every published comparison uses head **plus both controllers**; we are head-only by scope, so the model runs on glasses | part of the gap to published figures is sensor set, not performance |

**The honest one-sentence version.** We identify unseen users well where absolute head
position is recorded - and most of that is *where the headset sat in the tracking space
that day*, which three numbers capture without training; head height is the part that
survives a change of day, and it is smaller. The component the model actually learns is
real, worth about +0.14 in domain, and does not survive a change of corpus.

**The static cue is mostly placement, not height (measured 2026-09-04).** The three-number
lookup, restricted to one axis at a time on the same held-out pairs (three manifest seeds):

| corpus | xyz | y only | xz only | sessions per user |
| --- | --- | --- | --- | --- |
| ViewGauss | 0.933 | 0.886 | 0.871 | 4, one sitting |
| Head_and_Gaze V2 | 0.870 | 0.690 | **0.872** | 54, one sitting |
| VR_User_Behavior | 0.719 | 0.640 | **0.700** | 18, one sitting |
| NJIT | 0.653 | **0.748** | 0.542 | 1 |
| **alyx** (two days) | 0.593 | **0.661** | **0.539** | 2, different days |
| BOXRR held-out | 0.763 | **0.810** | 0.680 | many, across days |
| Nymeria | 0.730 | 0.654 | 0.784 | 2, one sitting, shared map |

On the seated 360-video corpora the lateral coordinates carry the whole lookup and height
adds nothing to them. The audit says why: between-user spread is 0.19-0.29 m in x and z
against 0.04-0.07 m in y, within-window 0.01-0.02 m, and every participant's sessions are
one sitting - so "cross-session" positives there never crossed a sitting, and where the
seat and the tracking origin were placed that day is a perfect per-participant constant.
**alyx is the only corpus in the table with sessions on different days, and there the
lateral cue collapses (0.539) while height stays (0.661).** That is the split: height
survives a day and is a biometric; placement is a same-sitting rig artefact that the
seated corpora, and part of BOXRR (xz-only 0.680, above the pre-registered 0.6 line), are
scoring. Nymeria's lateral figure is the shared SLAM map (see its section).

**The co-location geometry agrees (Trainer, same day).** Per-session mean positions from
the 5s@20Hz cache, P(within-participant separation < between-participant), all / lateral
(x,z) / height (y):

| corpus | participants | all | lateral | height | reading |
| --- | --- | --- | --- | --- | --- |
| ViewGauss | 35 | 0.896 | 0.878 | 0.918 | placement and height |
| Head_and_Gaze V2 | 100 | 0.911 | 0.900 | 0.722 | placement |
| VR_User_Behavior | 48 | 0.715 | 0.711 | 0.661 | placement, not height |
| BOXRR | 4009 | 0.765 | 0.685 | **0.828** | height first, plus a **standing offset** (below) |
| alyx (two days) | 70 | 0.575 | 0.552 | **0.743** | height, not room |
| Nymeria | 50 | 0.846 | 0.844 | 0.659 | shared map |
| EyeNavGS | 22 | 0.499 | 0.507 | 0.482 | nothing (virtual camera) |
| Panonut360, PanoSaliency | | | | | direction vectors, not read |
| NJIT | | | | | single session, untestable |

Predictions registered before the table: both the coordinator and Trainer said the seated
corpora would sit near 0.5 laterally ("one chair for everyone") and **both were wrong** -
within-participant session means are 0.195 m apart against 0.404 m between, so
participants occupy distinguishable positions in a shared rig, and that is what the
lookup reads. **BOXRR resolved to the case neither prediction named.** Median |xz| of a session mean
is **0.125 m** against a between-participant lateral median of 0.200 m (Nymeria, where the
origin really is a room, reads 3.73 / 6.29 m): across 4009 players in different homes every
session mean sits within about 12 cm of a common origin, so the frame *is* re-centred per
session - Trainer's mechanism - and yet within (0.109 m) still beats between at P=0.685 -
the coordinator's consequence. Small offset with P above 0.65 is a **person-specific
standing offset**: it survives re-centring, it is neither the room nor height, and under
the rule it still contaminates `raw` identity counts because it is a per-participant
constant rather than anything learned about how someone moves. Two qualifications travel
with it: BOXRR's height P (0.828) exceeds its lateral (0.685), so most of BOXRR's static cue
is height, the legitimate part, and the offset is the smaller share - "BOXRR is a room
count" is the wrong sentence; and **whether the offset is behavioural or procedural is
open** - a player who draws their guardian boundary the same way each time reproduces the
same observable without standing anywhere characteristic, and geometry alone cannot
separate the two. EyeNavGS, at chance on every axis, is the one translation-frame corpus
with no static cue at all - the corpus to use when the static cue must be absent, which
Nymeria turned out not to be. Head_and_Gaze
was checked for the direction-vector trap: its V1 files are unit vectors with no
quaternion, its V2 files (the only ones the loader takes) are real positions, |pos| 1.30.

Three consequences. (1) The "~78% is absolute head position" finding from
`center_position` stands as a measurement, but "height and seated posture" was the wrong
gloss on the seated corpora: it is mostly placement. (2) **Every `raw` identity-count
result on BOXRR carries a placement caveat** - a person-specific standing offset, not the
room - per the rule registered before the number existed, and the `dyn` results are the
clean ones. (3) Only corpora with sessions
on different days (alyx, BOXRR in part) can say anything about anthropometry at all; a
single-sitting corpus cannot separate a person from where they were put. Trainer's
co-location geometry (per-axis within- vs between-participant separation of session means)
is the companion table and is pending.

**What follows for anyone working here.** Report per dataset with its semantics, never
pooled alone. The mean-position lookup is computed on every run (`lookup_auc`) and is the
number to beat, not the model's previous score. The open problem is not accuracy, it is
building a learned component that transfers - which is what
`docs/GENERALISATION_PROPOSAL.md` is for.

**Identification is a separate scale.** rank-1 **0.570** at a 17-user gallery (chance
0.059), against a published 0.785 at the same gallery size. Never compare a verification
figure to a published rank-1; they are different tasks.

The historical 0.85 is **explained and reproduced**: it was a *seen-user* number. Our
lineage never held users out - the MS thesis this repo descends from splits pairs randomly
across users and reports **0.8364 on VR_User_Behavior**. Verified here directly: the same
protocol on our own code reaches **0.810**, against 0.62-0.67 for the identical
configuration with leave-users-out. Not a target, not a regression, not comparable to
anything in this file - every number here is leave-users-out.

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
  **The flip side is that `torch_geometric` is a hard dependency of every run, not an optional
  one** (Miami, 2026-09-09). `_import_all()` imports every non-underscore module in the
  package at package import, so `paper_gnn_bilstm` loads unconditionally and its module-level
  `from torch_geometric.nn import GATConv, GraphConv` runs even for a `bilstm`-only job.
  Without it, `import feature_extractor` raises `ModuleNotFoundError: No module named
  'torch_geometric'` and the whole test suite fails at collection - **a run that never asked
  for a GNN dies naming a GNN library**, which is why this is written down rather than left
  to be rediscovered. The coordinator told a new machine not to block on it because
  `paper_gnn_bilstm` was on no live arm; that is true of the *arm* and false of the *import*,
  and the mechanism was already documented one line above. When provisioning a machine,
  install it whether or not the GNN is wanted.
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

## Cross-corpus evaluation: what every run now records

Added for the unseen-dataset programme (`docs/GENERALISATION_PROPOSAL.md`), so a
transfer number never travels without the three things that qualify it:

| column / key | meaning |
| --- | --- |
| `test_auc_by_dataset`, `lookup_auc_by_dataset` | AUC per evaluation dataset for the model and for the mean-position lookup, on the same scores, as `name=value;...` |
| `position_lookup_auc`, `amplitude_auc` (each also `_eer` and `_by_dataset`) | the two training-free baselines that are valid under every encoding, on the same pairs: the mean-position lookup on each window's **recorded** position (`SampleIndex.window_mean_positions`, taken before encoding and standardised per dataset on the evaluation corpus's own recorded position frames - the 9.10 definition, independent of the checkpoint), and **movement amplitude alone** (norm of the per-axis sd of position in the window). `lookup_auc` keeps its old meaning - the lookup on the windows as the model sees them - and on a `dyn` row that is rounding residue tracking amplitude, not a baseline (`docs/GENERALISATION_PROPOSAL.md` 9.14) |
| `eval_tiers` | the semantics tiers present in the evaluation set (`dataset.DATASET_TIERS`: 1 head pose in metres, 2 direction vector, 3 other). `evaluate()` announces when a pooled figure mixes tiers |
| `eval_normalize` | how a dataset the normaliser never saw was brought into the training frame: `target_fit` (statistics fitted on the evaluation data, unsupervised, the default and the best label-free option measured), `session` (each session by its own statistics; at chance), `none` (a bound). Replaces what used to be a silent WARNING fallback |
| `unseen_datasets` | which evaluation datasets that policy actually applied to |
| `max_users` | an int as before, or a mapping `{dataset_dir_name: count}` that caps only the named datasets - `max_users={BOXRR-23_Dataset:343}` keeps all 76 alyx users at every point of an identity-count curve. Never applied to `test_dirs` |

Two encodings exist for the frame problem (per-corpus yaw references of +Z / +X / -X /
none, a rotation per-channel standardisation cannot undo): `encoding=yawc` rotates each
window about world up so its mean facing is +Z and keeps everything else; `encoding=dyn`
expresses pose relative to the window's *mean* pose, removing every static cue (height,
seat, posture) and is invariant to any rigid transform of the capture frame. `dyn` is what
`center_position` should have been - centring left the absolute quaternion in, and mean
orientation alone is 0.54-0.81 AUC of static posture.

The command shape for a cross-corpus run: `data_dirs` = training corpora, `test_dirs` =
the held-out corpora, **`test_on_excluded=false`**, **`exclude_users=[]`**, and any path
with parentheses quoted inside the Hydra list.

**`exclude_users=[]` is not optional, and this was found the expensive way.** The config
ships `exclude_users` with VR_User_Behavior users 1-5. With `test_on_excluded=false` those
five are removed from the *evaluation* set as well as the training set, so every
cross-corpus run that left the default in place scored VR_User_Behavior on **43 users,
not 48** - which is every VR_User_Behavior figure in `docs/GENERALISATION_PROPOSAL.md`
section 9 (the transfer table above included: 0.638 / 0.714 are 43-user numbers). The
comparisons inside section 9 are unaffected because every arm made the same omission, and
`sweep.folds` ignores `exclude_users` so no in-domain fold result is touched. It was caught
by a digit-exact reproduction of `lookup_auc` (0.719 on 48 users against the recorded
0.7114), not by reading the tables - and the tell was in every row all along:
`num_excluded_users` = 5 beside `test_on_excluded` = false. Read those two columns
together on any cross-corpus row before quoting it. A loader warning for the combination
is queued (see `docs/COORDINATION.md`); until it exists the override is the guard.

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

### NJIT's orientation is in a different frame from its position - and is repairable

Its quaternion and position disagree. Measured across all 8 datasets by rotating the
device's local +Y axis into world coordinates and averaging - for an upright head this
should point at world up:

| dataset | local +Y -> world | dataset | local +Y -> world |
| --- | --- | --- | --- |
| ViewGauss | (0.05, **0.97**, -0.11) | VR_User_Behavior | (0.01, **0.95**, 0.02) |
| EyeNavGS | (-0.07, **0.93**, 0.13) | Panonut360 | (0.06, **0.95**, 0.05) |
| **NJIT_6DOF** | (0.01, **0.05**, 0.00) | | |

Every dataset puts the headset's up axis at world up around 0.95. **NJIT puts it
nowhere** - and its local +Z lands on world +Z at 0.97, meaning its quaternions are
rotations *about Z* while its position is Y-up. That matches the deleted parser, which
built the quaternion with `R.from_euler('ZYX', [yaw, pitch, roll])` - a Z-up yaw
convention applied to Y-up position data.

**Unlike Nymeria's Z-up frame, this is not a source convention.** Nymeria's was confirmed
by its gravity vector, an actual physical fact about the recording. NJIT is room-scale VR
with no reason for orientation and position to disagree, so this is our own parser bug.

**It is repairable from the processed data - the raw source is not needed.** Tested three
candidates against the corpus-wide invariant:

| candidate | local +Y -> world | up |
| --- | --- | --- |
| as-is | (0.00, 0.06, 0.01) | 0.063 |
| `r (x) q` - world frame only | (0.00, 0.01, -0.06) | 0.006 |
| **`r (x) q (x) r^-1`** - full basis change | (-0.01, **0.97**, 0.00) | **0.975** |

with `r = (-0.70711, 0, 0, 0.70711)`, the same -90-degree rotation about X used for
Nymeria. The full conjugation is what is needed, because both the world *and* device
frames are Z-up in the parser's output. The acceptance criterion is not a guess: 0.975
matches what every other dataset measures.

**Not yet applied.** NJIT is 18 users and 414 windows at 5s, the smallest and least
valuable dataset here, and applying it means rewriting processed CSVs on three machines.
Recorded so whoever re-parses NJIT - or decides to patch it in place - has the verified
transform and the test that confirms it. **Until then NJIT's orientation channel should
not be trusted cross-dataset.**

### Every dataset is Y-up except Nymeria

Verified rather than assumed, using the fact that the vertical axis moves least during
seated and standing tasks - mean and sd per axis across all eight local datasets:

| dataset | quiet axis | dataset | quiet axis |
| --- | --- | --- | --- |
| PanoSaliency | y | Head_and_Gaze | y (1.58 mean) |
| 360_em | y | NJIT_6DOF | y (1.58 mean) |
| EyeNavGS | y | VR_User_Behavior | y (1.16 mean) |
| Panonut360 | y | ViewGauss | y (1.58 mean) |

Add BOXRR-23 (1.602m mean in `HmdPosition.y`), Across XR Applications (its config says
`up: y` outright) and the XR Motion Dataset Catalogue (X right, Y up, Z forward by
construction), and **Nymeria is the sole exception in the entire corpus**: its
`world_device` frame is **Z-up**, confirmed by `gravity_z_world` reading a constant -9.81
with x and y at zero.

**Documenting the exception is not enough, and the reason goes past the obvious one.** The
model consumes seven channels in a fixed order. If height is channel 5 for every dataset
and channel 6 for one, then a model trained on Y-up data and evaluated on Nymeria looks for
the anthropometric cue - the strongest single thing it uses - in the wrong channel.
Cross-dataset transfer would collapse for a reason that has nothing to do with
generalisation, which is exactly the experiment this project is now pointed at.
`normalize=per_dataset` fixes scale, not semantic role.

So Nymeria is rotated at conversion, as an explicit `--up-axis z` step rather than a silent
special case:

| | |
| --- | --- |
| position | `(x, y, z) -> (x, z, -y)` |
| orientation | `q' = r (x) q`, **r on the left**, then renormalise |
| | `r = (-0.70711, 0, 0, 0.70711)` in x,y,z,w |

Left-multiplication is the part to get right: the trajectory quaternion is world-from-device,
so changing the world frame composes on the left. **Swapping the position components without
rotating the quaternion silently decouples position from orientation** - a worse failure
than the exception it would be fixing.

**The check is decisive**, which is what makes the rotation safe to apply: Nymeria ships
`gravity_x/y/z_world`, so the same rotation must take `(0, 0, -9.81)` to `(0, -9.81, 0)`.
If it does not, the rotation is wrong. Secondary checks: mean transformed `HmdPosition.y`
near standing head height (BOXRR measures 1.602m), and mean \|q\| still 1.0000.

### Four datasets store a DIRECTION VECTOR in `HmdPosition`, not a position

Verified independently on raw CSVs - `|HmdPosition|` per dataset, over 25 users x 3 files:

| dataset | mean \|pos\| | sd | what it is |
| --- | --- | --- | --- |
| PanoSaliency | **1.0000** | 1.7e-16 | **unit direction vector** |
| 360_em | **1.0000** | 8.7e-17 | **unit direction vector**, and no quaternion column at all |
| Head_and_Gaze `V1_*` | **1.0000** | 8.0e-17 | **unit direction vector** |
| Panonut360 | **1.0000** | 7.1e-05 | **unit direction vector** |
| VR_User_Behavior | 1.2403 | 9.6e-02 | real position |
| EyeNavGS | 1.6955 | 9.7e-01 | real position (virtual-camera scene units) |
| ViewGauss | 1.7208 | 1.4e-01 | real position |
| NJIT_6DOF | 4.2727 | 1.3e+00 | real position, room-scale |

A norm of exactly 1 to machine epsilon is not a coordinate convention, it is a different
quantity in the same column. **These datasets carry orientation, encoded in the position
slot.**

**PanoSaliency's quaternion column is a constant identity `(0, 0, 0, 1)` on every row**
(verified 2026-09-04 on 25 of its 1583 files: per-file std 0.0, one distinct value per
column). So it has one orientation signal, in the position slot, and four dead channels
where orientation should be; after per-dataset standardisation the dead channels are
zero. Any orientation-derived feature on PanoSaliency (mean quaternion, `dyn`'s heading,
a future `channels=orientation`) is reading nothing there unless the direction vector is
moved into the quaternion slot first. The other tier-2 corpora should be checked the same
way before any orientation claim is made on them.

**This corrects an earlier entry in this file.** A previous version said these datasets
record "position relative to a seated origin" - inferred from their per-axis means sitting
near zero. That inference was wrong: direction vectors average toward zero when the
directions are spread, which produces the same signature. The mechanism matters, because a
seated-origin position still carries posture while a direction vector carries none.

**It also corrects the `channels=position` recovery.** `360_em` going from 0 to 2,360
windows was reported here as recovering a position dataset. Its source columns are
`x_head, y_head, angle_deg_head, GazeRay.*` with no position field anywhere, and its
`HmdPosition` is unit-norm, so what `channels=position` recovered was **13 identities of
direction data**, not position. Still 13 identities; not the thing the entry implied.

**And it sharpens the "~78% is absolute head position" qualification.** That figure comes
from `center_position` on the pooled corpus, which includes PanoSaliency and Panonut360 -
where centring removes the mean of a *direction vector*, not a height. So the measurement
"centring the position channels costs 0.134" stands; the interpretation "that 0.134 is
height and posture" holds only for the datasets that actually carry position. Any
anthropometry claim must name its datasets.

**Provisional per-dataset held-out AUC** from a 7-dataset `identity_softmax` model (fold 0
only, other folds pending) tracks the semantics exactly:

| ViewGauss | Head_and_Gaze | VR_User_Behavior | NJIT | PanoSaliency | Panonut360 | EyeNavGS | pooled |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.942 | 0.929 | 0.711 | 0.682 | 0.568 | 0.492 | 0.495 | 0.738 |

**The pooled headline is an average of near-perfect verification where real head position
exists and chance where it does not.** That is the single most important thing to know
before reading any pooled number in this file.

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

### Nymeria: the lookup was predicted at chance and scores 0.73 - RETRACTED premise

An earlier version of this section said Nymeria was "the one place the static baseline
cannot win", because the SLAM origin was assumed to be set per recording, which would
disable the mean-position lookup between the two halves of a positive pair and leave the
learned component measurable directly. That was a prediction, and Trainer insisted it be
measured before any model was scored on Nymeria. **It failed decisively** (2026-09-04,
50 participants, 20,778 windows at 5s@20Hz, cross-sequence positives, 25,600 balanced
pairs, target-fit standardisation, three manifest seeds):

| | AUC |
| --- | --- |
| three-number mean-position lookup | **0.730** +-0.001 |
| random-score control | 0.499 |
| criterion registered beforehand | 0.50 +-0.02 |

The direct test of the premise, in raw metres on per-sequence mean positions, reproduced
independently on both machines: a participant's two sequences sit **2.13 m** apart
(IQR 1.10-3.18) against **6.44 m** (3.78-9.13) between participants, P(same < different)
= 0.847. **A participant's sequences from one sitting share a map**, so the lookup
identifies *where someone was recorded*, not who they are.

And the position channel carries **no height at all**: Nymeria ships each participant's
measured `height_cm`, and the correlation between mean `HmdPosition.y` and true height
across the 50 participants is **0.057**, while a participant's two sequences agree on
mean y at 0.944. Mean y spreads 1.8 m across participants against 0.11 m of real height
spread - it is an origin offset shared within a sitting, nothing anthropometric.

**Consequences.** Under `raw`, every Nymeria number is a location match and is reported
only as that; a model-vs-lookup contest on Nymeria is not a fair fight. Under `dyn` the
lookup is 0.50 by construction, so Nymeria remains the cross-device (real AR glasses)
and cross-activity instrument - the same instrument as every other corpus, with no
special status. Every Nymeria number also carries the caveat that its positives are
cross-activity within one sitting on one day, so it cannot pay the 1.1-1.6 point
cross-session cost the rest of the corpus pays.

The acquisition stands: **50 participants, 100 sequences, 47.1GB transfer, 20,778 windows
at 5s@20Hz, 17 distinct scripts.** Held at 50 rather than 100 because its value is device
and activity diversity, not identity count.

**Nymeria moves into TRAINING (user decision, 2026-09-08).** It is no longer the held-out
AR-glasses instrument; the seven seated corpora carry testing. The reason is the criterion
this file adopted two days earlier - identity count buys in-domain performance and nothing
across an activity boundary, so activity diversity is the only axis left worth acquiring -
and Nymeria is 17 scripts of daily life in the wild against Beat Saber and Alyx. **This is
the first test of that criterion rather than an argument for it.** Its 50 identities are
noise on the identity-count curve (4096 -> 4146), so whatever it moves is the activity and
not the count, which makes it a clean single-variable experiment and the one that decides
whether any future acquisition is worth making. Registered before the run: pooled transfer
to the seven corpora **+0.005 to +0.03** over the matched-seed 4096 baseline of 0.6156;
**falsifier under +0.005**, which would say one genuinely different activity does no better
than identity count did and would argue against every acquisition on the board. **That band
was registered against a two-seed design that could not have tested it, and Trainer caught it
before launch** - see the power note below; the run is five paired seeds and a second arm. Two
qualifications travel with any Nymeria training figure: one sitting per participant, so its
positives are cross-activity within a sitting and cannot pay the cross-session cost (much
less of a problem under `dyn` than under `raw`); and the cost of the decision is that we no
longer hold an AR-glasses *test* corpus, and the catalogue's search found no other one.

### Activity diversity does NOT transfer: the registered band is excluded (2026-09-09)

The experiment the acquisition criterion was pointed at, and it came back negative. Arm B
holds identity count **exactly** fixed at 419 and swaps 50 BOXRR identities for 50 Nymeria
ones, so the treatment's BOXRR users are a strict subset of the control's and the only
difference is which activity 12% of the identities did. Five paired seeds, 10 s `dyn`,
transfer to the seven held-out corpora:

| arm | pooled transfer | best_epoch |
| --- | --- | --- |
| control (BOXRR 343 + alyx 76) | 0.5997 +-0.0030 | 98.0 |
| treatment (BOXRR 293 + alyx 76 + **Nymeria 50**) | 0.5985 +-0.0038 | 96.4 |

Per-seed delta -0.0048 / +0.0018 / +0.0005 / -0.0024 / -0.0013; mean **-0.0012**, paired sd
0.0026, t(4)=-1.06, won 2/5, **95% CI [-0.0045, +0.0020]**. Convergence matched (capped 1/5
against 2/5), so the delta carries no budget term.

**The registered band was +0.005 to +0.03 with a falsifier under +0.005. The entire band
sits above the entire interval.** One genuinely different activity - daily life on AR
glasses against Beat Saber and Alyx - at 12% of identities and 14.2% of windows, moves
cross-activity transfer by nothing, bounded at 95% below +0.002. **Activity diversity joins
identity count on the list of data-side levers that do not cross a domain boundary**, and
the acquisition criterion adopted on 2026-09-06 - "argue an acquisition on activity or
device diversity" - now has no surviving axis behind it. Argue the next acquisition on
something this file has actually measured to work, or do not make it.

**A registered band is settled by where the INTERVAL falls, not by whether p<0.05 (Trainer,
and it generalises).** The two-sided test here is not significant and the harness would have
printed "not resolved", which is true and badly understates the result: **an interval can
fail to exclude zero while excluding the whole hypothesis it was built to test.** Reporting
only the test would have turned a decisive negative into an inconclusive one - the exact
mirror of the failure the MDD rule above guards against, and just as costly. Check registered
thresholds against the interval; report the CI beside every registered band.

**Arm A settles nothing and its convergence check fired.** Four of five seeds, CI
[-0.0062, +0.0098] - the band edge is inside the interval - and treatment `best_epoch` 106.0
against a control at 117.8 +-1.3 that never stopped early, capped 3/4 against 4/4. So its
delta carries a budget term regardless of resolution, exactly the bias the patience note
above predicts. **The categorical form of that check is what caught it**: against a control
that never stopped early, "the treatment stopped on patience" is a binary fact needing no
mean comparison, which is why the band-from-a-constant version would have missed it.

**The NJIT structure prediction failed, and is scored failed.** Registered: if anything moves
it should be NJIT - room-scale walking, the only held-out corpus resembling daily life - by
at least 0.01 over the mean of the other six. NJIT came top of seven at +0.0060, which is
+0.0081 over that mean. Direction held, threshold missed, and **a line moved afterwards is
not a line**. The noise scale is visible in the same column: Panonut360 came second at
+0.0054 and is tier 2, which should be at chance.

### Across-XR: the first instrument that separates activity from population

Landed 2026-09-09, 5.1GB, 49 files. **All 49 participants appear in all five applications,
no missing cells** - the fully crossed structure BOXRR turned out not to have. Two of the
five are our own training activities: `game_id=3` is **Beat Saber** (BOXRR's activity) and
`game_id=2` is **Half-Life: Alyx** (who_is_alyx's). The rest are Superhot VR, Synth Riders
and a Social VR scenario, 40k-206k rows per participant per game at 90.9 Hz.

**Why this matters more than its identity count.** Every cross-activity number in this file
changes the activity *and* the people *and* the rig together, so "the learned component is
activity-bound" has always been confounded with "it is corpus-bound". Here the person, the
headset, the room and the sitting are fixed and **only the application changes**. 49
identities is far below the acquisition floor and that is not the point - this is a *test*
instrument for a question nothing else in the corpus can ask.

**The placement premise was registered and it FAILED, in the corpus's favour.** All five
games are one sitting, so the prediction (registered at `9664dde`, before measuring) was
that placement would be a per-participant constant across applications - P(within <
between) of **0.85-0.95**, lookup above 0.85, making every `raw` number a placement match.
Falsifier: P below 0.65. Per-game mean head position over all 49x5 cells:

| axis | within med | between med | P(within < between) |
| --- | --- | --- | --- |
| all | 0.356 m | 0.390 m | 0.545 |
| **lateral (x,z)** | 0.350 m | 0.374 m | **0.527** |
| **height (y)** | **0.028 m** | **0.077 m** | **0.754** |

**Lateral placement is at chance across applications** and the falsifier fired by a wide
margin. The mechanism is that **the games move people differently** - Beat Saber is
stationary, Alyx has locomotion, Superhot has dodging - so a 15-minute mean position
records *where the game makes you stand*, not where the rig sits. The applications scramble
the artefact themselves. Height survives at **0.754**, close to alyx's 0.743, and that is
the legitimate cue: within-participant height spread across the five games is 0.068 m
against a between-participant sd of 0.075 m, over a real 1.42-1.73 m range.

**So a cross-application pair here is largely free of the placement artefact, and its
residual static cue is height** - the one this file has consistently called a biometric.
That makes the corpus usable under `raw` with a height caveat and clean under `dyn`, where
before this measurement it looked like a `dyn`-only instrument. Two caveats stay attached:
it is still **one sitting**, so it says nothing about temporal persistence and cannot pay
the cross-session cost; and `take_id` separates a short break, not a day.

**Conversion facts, verified on the files rather than the Readme, which is wrong again.**
Header order is `head_rot_w` **first**; position is in **centimetres** (`head_pos_y`
1.53-1.60 m); |q| = 1.0000; 90.9 Hz native; y-up, matching ours, so no axis remap;
`user_id` matches the filename on every file checked. Their own deterministic test split is
users **32-48**, so a published-comparison arm is available. See
`docs/DATASET_CATALOGUE.md` for the full format table.

### Cross-corpus transfer: the model is BELOW the lookup, and flat in identity count

The experiment the BOXRR acquisition was for. Train on BOXRR+alyx, evaluate on the seven
held-out corpora never trained on. `bilstm`, `identity_softmax`, 30 epochs, target-fit
stats on the held-out corpora, random control 0.498 pooled:

| | pooled | ViewGauss | H&G | VR_UB | NJIT | tier 2 + EyeNavGS |
| --- | --- | --- | --- | --- | --- | --- |
| model, 419 ids | **0.672** +-0.003 | 0.911 | 0.750 | 0.638 | 0.648 | at chance |
| model, 2096 ids | **0.671** | | | | | |
| **lookup** | **0.727** | 0.934 | **0.869** | **0.714** | 0.653 | |

**Two results, both pre-registered, and the second is worse than predicted.**

1. **Transfer is flat in identity count.** 419 to 2096 BOXRR identities moves pooled
   transfer by 0.001. More identities from one activity does not improve generalisation to
   other activities. This is the "we bought an easier corpus rather than a better model"
   outcome recorded before the curve was measured.
2. **The model is worse than three equally weighted numbers** on every tier-1 corpus once
   the corpus changes - by 0.12 on Head_and_Gaze and 0.08 on VR_User_Behavior. Validation
   selects epoch 2-3 of 30 every time, so it overfits the source domain almost immediately.

Identity count was the only data-side lever ever measured to work here. It works
**within** a domain and does not cross one.

### `dyn`: the learned component that does transfer, and the only thing identity count moves

`encoding=dyn` removes every static cue - position centred per window *and* orientation
taken relative to the window's mean heading, gravity kept. The mean-position lookup has
nothing static to read on it: its column on a `dyn` row is rounding residue that tracks
movement amplitude (1e-9 m; 0.50-0.57 per corpus, `docs/GENERALISATION_PROPOSAL.md` 9.14),
so the training-free baseline for `dyn` is **movement amplitude alone** (0.50-0.66 per
corpus; it beats the model on NJIT), and anything the model scores is behaviour, measured
rather than simulated by `center_position` (which leaves absolute orientation in, and the mean quaternion alone
recovers 0.54-0.79 of static posture).

**Two lookup columns, one rule (since the amplitude-baseline merge of 2026-09-05).** On a `dyn` row `lookup_auc` is the lookup on the *encoded* windows -
rounding residue that tracks movement amplitude, not a baseline of anything - and
`position_lookup_auc` (the same lookup on each window's recorded position, standardised per
dataset on the evaluation corpus's own position frames, the 9.10 definition) is the real
static baseline; on a `raw` row the two coincide to
rounding. `amplitude_auc` is movement amplitude alone, the dynamics branch's baseline,
computed before standardisation in the corpus's own units under every encoding. Beside a
`dyn` figure quote `position_lookup_auc` and `amplitude_auc`, never `lookup_auc`.

**Identity-count curve, BOXRR+alyx -> the seven held-out corpora**, `epochs=120`,
`patience=15`:

| identities | pooled | Head_and_Gaze | ViewGauss | NJIT | VR_UB |
| --- | --- | --- | --- | --- | --- |
| 419 (5 seeds) | 0.582 +-0.001 | 0.537 | 0.523 | 0.522 | 0.515 |
| 1000 (2 seeds) | **0.600** +-0.001 | | | | |
| 2096 (1 seed) | 0.598 | **0.570** | **0.571** | 0.540 | 0.521 (flat) |

**This is the only thing in the project that identity count moves across a domain
boundary.** Raw transfer is flat to three decimals over the same range (0.672 / 0.672 /
0.671); `dyn` gains +0.016 pooled and +0.03 to +0.05 on the two best-conditioned corpora -
the pre-registered band, held. All of the gain is between 419 and 1000. It only appears
where the corpus has a stable real head pose: VR_User_Behavior, PanoSaliency and EyeNavGS
are flat.

**Not budget-limited.** The 120-epoch budget changed transfer at 419 by 0.001 against the
30-epoch runs, even though those selected epoch 29-30 of 30. The censoring mattered for the
in-domain figure, not the transfer one.

**At 10 s the window and identity levers add, and identity count then saturates out of
domain while still paying in it (2026-09-05, `docs/GENERALISATION_PROPOSAL.md` 9.14).** `dyn`,
10 s, the seven held-out corpora: 419 identities 0.600 (5 seeds), 2096 (BOXRR capped
at 2020 of the now-4020 users) **0.618**, 4096 **0.618** - the additive prediction to three
decimals, and the second doubling of training identities (1535 -> 3072) moves transfer by
0.001. **Both figures held on a second seed** (2026-09-06): 2096 reads 0.6179 +-0.0005 over
two seeds and 4096 0.6156 +-0.0040, against a falsifier registered beforehand at 0.01 from
the seed-1 value. So the saturation is the result, not one seed's draw - and note the two
seeds separate 4096 by ten times as much as they separate 2096, which is the shape of a
figure that has stopped responding to identities and is reading run-to-run variation. In domain the same three checkpoints read BOXRR 0.845 -> 0.962 -> 0.970 and alyx
0.664 -> 0.799 -> 0.796 on their own validation users, and on the same 914 BOXRR users that
neither the 419 nor the 4096 checkpoint ever saw, **0.844 against 0.970** (the 4096 figure
carries ~0.02 selection optimism, the 419 one none). Identity count is a large lever within
the training activity and a small one across it, and the gain is the model, not easier
users - no recording is shared between any two of the 4020 BOXRR user directories. Movement
amplitude alone reads 0.57 on those users. **The Nymeria trend was seed
noise and is withdrawn**: seed 1 read 0.535 -> 0.544 -> 0.553, monotone and about two
seed-sds a step, but seed 2 sits 0.009 and 0.015 lower at the same two points (0.535 at
2096, 0.538 at 4096), so the two-seed means are 0.540 +-0.007 and 0.545 +-0.011 against
0.535 +-0.004 at 419 - inside the spread at every point. Nymeria is **not resolved** and
stays at 0.53-0.55 for every window length and identity count measured. Every Nymeria figure
is now a shard row rather than a scratchpad number (`experiment=nymeria_rescored`, 29
checkpoints): 419 ids read 0.528 at 5 s, 0.537 at 10 s and 0.538 at 20 s, 2096 reads 0.541 at
5 s and 4096 0.546 at 10 s over two seeds, the `random` control 0.497, and the rows agree with
the scratchpad they replace within 0.003 everywhere - so the old harness had been feeding the
checkpoints the right thing, which is a result about the harness and not a null. The same rows
carry Nymeria's recorded-position lookup at 0.73 (the 9.9 location match) against movement
amplitude at 0.51-0.52. A one-seed
monotone sequence over three points was never enough to call a trend, and calling it one
is the error to learn from here, not the number.

**The ceiling on the seated corpora is theirs, not the model's.** In domain on the
8-dataset corpus (5 folds, uncensored at epochs 5-10, control 0.499) the seated corpora
reach 0.53-0.55 - and the BOXRR-trained branch **matches or exceeds that from outside**:
Head_and_Gaze 0.570 out of domain against 0.551 in, ViewGauss 0.571 against 0.528. Training
on a corpus does not beat training on Beat Saber and transferring in.

**The learned component is activity-bound - less so than first measured, and the
difference is window length and budget.** At 5 s and 30 epochs, unseen Beat Saber players
separated at ~0.80 on movement alone and unseen alyx players at 0.53, and raising alyx from
3.6% to 18% of training identities left it at 0.530 +-0.007. Re-measured on the same
validation-user protocol (never trained on; they chose the epoch, so ~+0.02 optimistic),
five seeds each:

| `dyn`, in domain | BOXRR | alyx |
| --- | --- | --- |
| 5 s, 30 epochs | ~0.80 | 0.530 |
| 5 s, 120 epochs, patience 15 | 0.814 +-0.012 | **0.592** +-0.027 |
| 10 s, stride 5, same budget | 0.845 +-0.004 | **0.664** +-0.019 |

alyx crosses the 0.60 line registered for it by three seed-sds, so the reading softens as
pre-registered: free FPS locomotion across two days *does* show a person at ten seconds.
The activity gap to Beat Saber remains (0.845 vs 0.664), but "barely a biometric" was a
5-second, 30-epoch statement. Two things to carry: the training budget moved alyx in
domain (0.53 -> 0.59) even though it never moved transfer, so in-domain and transfer
figures respond to different levers; and every number in this table carries the +0.02
optimism of the validation-user protocol.

**Fusion is retired for transfer.** Weighting the lookup with `dyn` drags tier 1 below the
lookup under an in-domain weight, and a leave-one-corpus-out weight (0.15-0.35, never the
target's labels) leaves every tier-1 corpus at or below it.

### Yaw canonicalisation: predicted correctly per corpus, worth nothing pooled

`encoding=yawc` (gravity-preserving yaw canonicalisation) seed-paired at 419 identities:
**+0.025** on Head_and_Gaze (t=3.0) and **+0.015** on VR_User_Behavior (t=2.4) - exactly the
+X and -X corpora predicted - but **-0.035** on ViewGauss and **-0.025** on NJIT, the +Z
corpora. Pooled **+0.003, t=0.3**, and it triples seed spread. The per-corpus pattern matched
the prediction and the pooled effect is nil. Not worth an arm.

### A three-number lookup matches the trained model (CONFIRMED)

Measured on all 5 folds of `b732bee5c6`, on the **same held-out users and the same pair
manifests**, with the lookup using **each checkpoint's own training-fitted
`ChannelNormalizer`** - the exact transform the model receives. The probe is a
training-free **mean position**: three numbers per window, Euclidean distance.

| | pooled | ViewGauss | H&G | VR_UB | NJIT | PanoSal | Panonut | EyeNavGS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lookup | **0.726** +-0.017 | 0.932 | 0.868 | **0.716** | **0.674** | 0.583 | 0.508 | 0.493 |
| model | **0.723** +-0.014 | 0.938 | 0.898 | 0.689 | 0.611 | 0.580 | 0.504 | 0.490 |

An earlier pass fitted the lookup's statistics on the held-out cohort, which would have
handed it information the model was denied. Redone properly the result is unchanged, and
the reason is worth keeping: **with Euclidean distance the per-channel mean cancels in
`a - b`**, so the only cohort information the flawed version ever used was three per-axis
scales.

**The model wins on the two best-conditioned corpora and loses on the next two.** Pooled,
it is inside the fold spread of three numbers requiring no training at all.

### But there IS a learned component - it just does not transfer

The decisive contrast, from an 8-dataset model with alyx identities in training
(`31751868df`, ~15 alyx users held out per fold):

| | AUC on alyx |
| --- | --- |
| 8-dataset model, alyx **in domain** | **0.725** +-0.073 |
| static lookup on alyx | ~0.59 |
| 7-dataset model, alyx **never seen** | **0.566** |
| random control | 0.496 |

In domain the model beats the static lookup by roughly **+0.13**. That is the first
evidence anywhere in these numbers of a learned non-static component - and it is **exactly
the component that disappears on transfer**: unseen, the model drops to 0.566, below the
lookup's 0.593 on the same corpus.

So the picture is coherent and uncomfortable:

- the **static cue transfers** (lookup: 0.593 unseen, and it needs no model)
- the **learned cue is real but dataset-specific** (+0.13 in domain, gone when unseen)
- the unseen-dataset cost on alyx is about **0.16 AUC**

That is a sharper statement of the generalisation problem than "the model does not
generalise". It generalises exactly as far as the static cue does, and the part it actually
learns is the part that does not survive a change of corpus.

**A THIRD CAVEAT, VERIFIED FROM THE LOG RATHER THAN INFERRED.** Read off the run records
once DESKTOP-C's shard reached origin:

| sweep | margin | scale | data dirs | epochs |
| --- | --- | --- | --- | --- |
| `31751868df` (in domain) | **0.1** (10 runs), 0.2 (3) | 15 / 30 | 8 | **30** |
| `b732bee5c6` (unseen) | unrecorded, so the 0.35 default | 30 | 7 | **20** |

The in-domain checkpoint comes from the **margin/scale sweep**, whose only 5-fold-complete
cells are at `identity_margin=0.1`. So the pair differs in **three** things - dataset
coverage, margin, and epoch budget - not one.

The confound is bounded rather than fatal: the same sweep puts margin 0.1 about 0.02 above
the default reference, against a measured gap of 0.164. So the direction survives and the
magnitude should not be quoted to three digits. **The clean version is a 7-dataset model
trained at the same margin and epoch budget as the 8-dataset one**, differing only in
whether alyx was in training. Until that exists, treat 0.164 as "large and positive", not
as a measurement.

**Two further caveats on the in-domain figure.** Its spread is +-0.073 against +-0.014 pooled,
because only ~15 alyx users are held out per fold, so the 0.16 gap is about 2.2 sd -
real but not tight. And the two numbers come from different checkpoints scored on
*different* evaluation sets: ~15 held-out alyx users against all 76. Gallery composition
has already been shown to matter here, so the like-for-like version - both models on the
same ~15 users per fold - is the one to quote.

**Standing consequence**: the mean-position lookup is a permanent per-dataset baseline
reported beside every model, not a probe someone runs occasionally. It has now been found
competitive twice, both times only because someone went looking.

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

**The +0.02 is priced for the metric that did the selecting, not for every figure measured on
validation users (2026-09-06).** It comes from `best_test_acc` being a max over ~20
evaluations *of the set it reports*. A checkpoint that chose its epoch on pooled verification
AUC and is then read for **rank-1 identification on one corpus** was not selected on anything
that figure measures, so there is little for selection to inflate - and measuring it settled
the question: validation users 0.858 +-0.009 against 0.862 +-0.019 on 94 users no checkpoint
ever sampled, **+0.004, the wrong sign for optimism**. Residual optimism on a *different*
metric over the same users is bounded by how tightly the two metrics correlate, which is a
thing to measure once rather than a constant to add; do not carry the +0.02 onto a figure
whose metric never chose an epoch.

The method point is the durable half, and it was Trainer's: **a caveat registered before a
number is a test; one carried by habit is a hedge.** This one was restated twice before it
was measured, and it was wrong in the direction that made the result look weaker - the same
lesson as the Nymeria at-chance prediction, pointing the other way.

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

**Compute the design's MDD before registering a band, or the registration is theatre
(2026-09-08).** The coordinator specified the Nymeria activity-diversity run as two seeds
paired against an existing arm, and registered +0.005 to +0.03 with a falsifier under +0.005.
At n=2 the paired multiplier is `t(0.975, 1)/sqrt(2)` = **8.98x**, so against the baseline's
own between-seed sd of 0.0040 the minimum detectable difference is **0.036** - and against a
paired sd of 0.005, **0.045**. **Every number in that registration, the band and the
falsifier alike, sat inside the noise floor of the design meant to test it.** It was not
under-powered, it was unfalsifiable: no outcome could have moved it either way, and a null
would have been read as "activity diversity does nothing" and used to argue against
acquisitions a person was writing to authors about. Five paired seeds bring the MDD to 0.005
and cover both bands; the run is eight.

| paired seeds | multiplier | MDD at sd 0.0040 |
| --- | --- | --- |
| 2 | 8.98x | 0.036 |
| 3 | 2.48x | 0.010 |
| **5** | **1.24x** | **0.005** |
| 8 | 0.84x | 0.003 |

The failure is not that the table above was missing - it is directly above this paragraph and
says "read this before designing an experiment, not after". **A registered prediction is only
a test if the design can resolve it, and checking that is one line of `scipy.stats.t`.** The
drop from n=2 to n=3 is 3.6x on its own, so the cheapest fix to almost any under-powered
paired design here is a third seed.

**And state the treatment's dose before reading a null.** The same run had Nymeria supplying
**2.9% of training windows** (50 identities of 4096; 20,678 windows of 707,017), because
`identity_softmax` samples windows uniformly. A null there cannot distinguish "activity
diversity does not transfer" from "the objective barely saw the second activity", and only
the first licenses the conclusion the run exists to support. `balance_identities` cannot fix
it and moves the wrong way: Nymeria averages **414 windows per identity** against BOXRR's
**151**, so capping at the corpus median *trims Nymeria* and raises BOXRR's share. The fix is
composition, not sampling - hold identity count fixed and swap identities between corpora
(BOXRR 343 + alyx 76 against BOXRR 293 + alyx 76 + Nymeria 50, both 419), which raises the
dose from 2.9% to **14.2%** of windows. **A dose is part of a treatment's definition; a null
without one is a result about the dose.**

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

**Tilt Brush recordings carry no head track at all - CONFIRMED on a real file
(2026-09-08).** The library's own `fromTilt()` adds exactly one device - `BRUSH`,
`type='OTHER'` - and no HMD, and this file predicted that BOXRR-23's Tilt Brush portion
would therefore be brush-tip trajectory only. One user tarball from the unlabelled pool
settles it, verbatim: `hardware.devices` is a single `{"name": "BRUSH", "type": "OTHER"}`
with no HMD anywhere, `software.app.name` reads **"Tilt Brush"** directly rather than by
inference, 153,710 frames over ~4.5 hours. **A prediction registered before the data
existed, and it held.** That portion of the corpus is unusable here. `prepare_boxrr.py` skips any recording with no HMD/HEAD
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

**Our BOXRR mirror is Beat Saber only, and the true release is not** (checked 2026-09-05,
metadata only, nothing fetched). Two facts that must be quoted together:

- **The HuggingFace mirror's index has 0 users with recordings in two or more
  applications.** Of **4,716,986 records in total**, 4,661,942 carry a populated `info`
  block and every one reads `app.name = "Beat Saber"`; 55,044 carry no `info` key at all.
  *(Two numbers in this bullet were wrong until 2026-09-08: 4,716,986 was the total rather
  than the populated count, and the remainder was described as empty stubs. See the
  re-check below - the second error was the material one.)*
- **The official Berkeley release includes Google Poly (Tilt Brush) recordings: 55,178 of
  them, 1.2%**, per the official page's own count, and the datasheet states that one folder
  holds all of a user's recordings and that the source application is identifiable from the
  included metadata. So the cross-application structure exists upstream and **this mirror
  excludes it** - a curation choice its README does not mention.

**This is an index gap, not an absence, and the distinction decides an acquisition.** No
index that shows the Poly recordings is linked from the official page, the datasheet, or
findable by search; `dict.json` is the XROR JSON Schema, not a recording index. So whether
any user has recordings in *both* applications - the thing 9.14 says would actually move
transfer, at scale, under an agreement we already hold - **is unanswered and cannot be
answered from any metadata file we can reach.** Do not write "BOXRR is one activity"; write
"our copy is". Resolving it needs either a complete index from the authors or fetching
per-user folders blind, which is an acquisition decision, not a check.

**RE-CHECKED AND SETTLED (2026-09-08), and the answer kills the acquisition rather than
enabling it.** The mirror's dataset card says TiltBrush users are marked in
`info.software.app`, which raised the possibility that our scan had read the wrong key path.
It had not - there is no path to read. Every one of the 4,716,986 documents was partitioned
by its exact top-level key set (a `Counter` over `sorted(doc.keys())`, exhaustive, not a
sample):

| key set | count | what it is |
| --- | --- | --- |
| `_id, duration, info, num_frames, user_id` | 4,661,942 | **100% `info.software.app.name = "Beat Saber"`**, zero other values, zero partial paths |
| `_id, corrupt_user, duration, num_frames, user_id` | 54,965 | **real recordings with no metadata block** - median 8,977 frames, median duration ~25 min; only 766 at duration 0 |
| `_id, duration, num_frames, user_id` | 79 | genuinely empty (79 duration 0, 78 no frames) |

The partition is exact, with no residue. **The `info` key is absent on those 55,044
documents**, so `info.software.app` is not populated with something else - it does not
exist there, and no scan of any path could have found Tilt Brush in this file.

**And the decisive number needs no resolution of what that bucket is: 92,103 distinct users
hold at least one Beat-Saber-labelled recording, 13,746 hold at least one unlabelled
recording, and the two sets overlap in ZERO users.** Even granting the most generous
reading - that every unlabelled recording is Tilt Brush - **not one person in BOXRR-23 is
recorded in two applications**. And 92,103 + 13,746 = 105,849 against the release's own
105,852 users, so this index covers essentially the entire release: the finding is about
**BOXRR-23**, not about our copy of it.

**So do not write to the authors, and strike the cross-application acquisition.** The thing
the board was pointed at - the same people recorded in two activities, at scale, under an
agreement already held - **does not exist in this corpus**, and no index request or blind
fetch would have produced it. That was one email away from being asked for.

**The unlabelled bucket is Tilt Brush, and it is unusable - CLOSED (2026-09-08).** One user
tarball from the 13,746, fetched with the user's direct approval, inspected and deleted the
same hour so nothing was retained under the DUA's destruction obligations. `software.app.name`
reads **"Tilt Brush"** outright, so the count-proximity coincidence (54,965 against 55,178)
was pointing at the right answer for the wrong reason - and `hardware.devices` holds exactly
one `BRUSH`/`OTHER` entry with **no HMD**, so a head-only pipeline can read nothing from it.
The `corrupt_user` flag marks records whose Beat-Saber-shaped metadata parse failed, which is
what a TILT-format file does to that pipeline.

**Both halves of the cross-application question are now closed, and both closed negative.** No
user in BOXRR-23 is recorded in two applications; and the second application that does exist
there could not have been used by this project even if someone were. **Nothing is owed to
anyone and nothing is left to fetch** - the corpus is Beat Saber for our purposes, by
measurement rather than by assumption, and that sentence can now be written without the
"our copy is" hedge this file has carried since 2026-09-05.

**Method, and it is the recurring bug again from both sides.** The coordinator inferred a
wrong-key-path mechanism from the distributor's prose and was wrong; Data had characterised
55,044 records as empty stubs from **three examples that happened to be substantial**, and
corrected it by dumping every document's key set rather than sampling. Neither the prose nor
the sample was the data. **Partition the whole file by exact key set before describing what
is in it** - it is one `Counter`, it cannot be fooled by a lucky sample, and here it turned
a hypothesis, a mischaracterisation and an arithmetic slip into one exact table.

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

**That criterion is superseded, and identity count is no longer the argument for an
acquisition (2026-09-06).** The identification axis measured what identity count buys, and
the answer is narrow. Across a five-fold increase in training identities it moved **in-domain**
rank-1 a great deal (0.824 -> 0.948 on one clean population), moved **cross-activity transfer
by 0.001**, and did **not touch the per-user split at all** - the near-certain and near-chance
users are the same people at 2096 identities as at 419, shifted together. The project's two
open problems are transfer and what the mean conceals, and more identities of the same
activity address neither. **Argue an acquisition on activity or device diversity, or do not
make it.** The 400-identity floor still applies to anything argued on identity count, but
nothing should be argued on identity count now.

This sharpens rather than weakens the outstanding BOXRR question. What the official release
has and our mirror lacks is **users with recordings in two applications** - cross-activity
structure at scale, under an agreement already held. That is precisely the axis this finding
says is worth acquiring, and it is the one thing on the board that is.

Searched and rejected, recorded so the search is not repeated:

| dataset | verdict |
| --- | --- |
| **GazeBaseVR** | **Disqualified on content, not access.** 407 participants, CC-BY, trivial figshare download - and **no head position or orientation channel at all**. Participants were on a chin rest specifically to suppress head movement, and gaze is expressed as an angle relative to a fixed headset (Lohr et al. 2023, Table 4). The attractive access profile means someone will propose it again; it is the wrong signal, not the wrong licence. |
| Liebers et al. | No confirmed public release; 16 users maximum. |
| OpenNEEDS | 44 users, correct signals (head+hand+gaze), but gated behind a request process to Meta. |

**The rejected list was re-read against the new criterion (2026-09-06), because a rejection
inherits the criterion it was made under.** Three of the four were declined at least partly
on identity count, which is no longer the argument:

- **GazeBaseVR - still dead, reason untouched.** No head position or orientation channel at
  all. Content, not size.
- **Liebers et al. - still dead, reason untouched.** No confirmed public release.
- **Nymeria remainder (186 of 236) - still declined, and now for a better reason.** It is real
  AR glasses and 17 scripts, which is the axis now argued for - but we already hold 50
  participants of exactly that device and that script set, so the remainder adds *identities*
  of a diversity we have rather than new diversity. And every participant is **one sitting**,
  so it can never pay the cross-session cost. Declined on structure, which is what the
  original note meant and now says.
- **OpenNEEDS - status genuinely changes.** 44 identities was the disqualifier and identity
  count is no longer the question. Its task set - reading, drawing, shooting, object
  manipulation - is unlike anything in this corpus, and `docs/DATASET_CATALOGUE.md` already
  records its value as a **test** set rather than a training one, which is precisely the
  activity-diversity role now wanted. Head-only scope applies as everywhere: its hand and gaze
  channels are irrelevant to us, neither a benefit nor a cost. The remaining obstacle is a
  **signed data-use agreement with Meta, which only the user can enter into** - a cost and a
  decision, not a disqualification.

Also worth knowing when device diversity is the argument: the catalogue searched specifically
and found **no AR-glasses motion dataset other than Nymeria**. That axis is close to
exhausted, and we already hold the only entry on it.

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
- Clause 4 governs moving BOXRR-derived data between machines. **Resolved for the Miami
  server (user, 2026-09-09): it is the same institution and sits in its server room, so
  putting BOXRR-derived data there is internal use by the recipient, not further
  distribution, and clause 4 does not bite.** The general rule still stands for anything
  outside that boundary - convert wherever the raw data lands rather than centralising and
  copying - and "same institution" is the test, not "a machine we have an account on".
- **BOXRR-derived data does not go to cloud storage** (user, 2026-09-09, on a Google Drive
  the project now has for code, results and write-ups). That is a policy decision rather
  than an interpretation, and it is the right way round: clause 15's destruction obligation
  is far harder to honour on Drive than on a lab machine, because trash, version history and
  other people's synced clients are all derived copies. Code, results shards and prose are
  fine there; datasets are not.
- A **checkpoint trained on BOXRR is plausibly a derived copy** under clause 15, so shipping
  weights instead of data is not automatically a way around clause 4. Inside the institution
  that is moot; outside it, treat weights as in scope.
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

**Check dataset properties on the files the loader accepts, not on the directory.**
Head_and_Gaze was flagged as a direction-vector corpus from three raw files that happened
to be `V1_*` (unit vectors, no quaternion) - files `UserProfile` skips under
`channels=full`, so nothing in any row came from them; the `V2_*` files the cache holds are
real positions (|pos| 1.30 +-0.06). A property read off a superset of what a run used is
the same failure as the `mode=curve` split fallback and the single-dataset balance pilot.
`audit_frames.py` reads the cache for exactly this reason; a raw-file check has to sample
the same subset, or say which subset it sampled.

**State each column's invariances before running; they partition the bugs.** The step 6
enrolment harness failed its digit-exact gate on its first run with y-only matching exactly
on every seed while xyz and xz were off by 0.002-0.018. y-only is invariant to per-channel
scaling and the other two are not, so the user set, manifests, pair seeds and AUC code
were all ruled out at once - only the normaliser could do that, and the checkpoint-to-seed
mapping had been transcribed in the wrong order. Two registered caveats paid for themselves
as diagnostics that day (this one, and the Nymeria at-chance prediction that was measured
first and failed); a caveat written before the number is a test, one written after is an
excuse. Corollary: a scale-invariant column's standardised and raw-metres values are
identical by construction, so report it once and say so.

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

**Now measured directly on the identification axis rather than inferred from verification
(2026-09-06).** The confound is that k and window length both buy seconds, so the design
holds *total evidence* fixed at 80 distinct seconds: **5 s at k=16 against 10 s at k=8**, the
same 94 clean BOXRR users, paired by seed, both indices built at full stride so neither
inherits its training layout.

| arm | rank-1 @N=17 | AUC |
| --- | --- | --- |
| 5 s, k=16 | **0.862 +-0.019** | 0.9619 |
| 10 s, k=8 | 0.842 +-0.006 | 0.9597 |

**-0.020, paired sd 0.018, t(4)=-2.44, won 1/5** - registered under +0.05 and it came in
*negative*. **Enrolment evidence moves rank-1; how those seconds are packaged into windows
does not.** So window length is retired here by measurement rather than by an
order-of-magnitude argument carried over from verification, and the 10 s in-domain
verification gains elsewhere in this file do not imply an identification gain.

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

**The budget is recoverable from the rows even when the config fields are blank
(2026-09-08).** `epochs` and `early_stopping_patience` are recorded as `None` on the older
transfer rows, so "both arms ran the same budget" would rest on assertion - except that
`best_epoch` and `epochs_run` are recorded, and `epochs_run - best_epoch` is the patience
whenever a run stopped early. Sweep `0840769514`: seeds 2, 4 and 5 read 73+15=88, 88+15=103,
97+15=112, three exact hits, and seeds 1 and 3 sit at the 120 cap with `best_epoch` 114 and
118. **That is patience=15 under a 120 cap, recovered arithmetically from two columns.**
`docs/acceptance/nymeria_activity_analysis.py` derives it per arm and flags anything it
cannot explain, so an arm that silently ran a different budget shows in the output rather
than in a config nobody kept.

**And a censored control makes the convergence check sharper, not weaker.** The coordinator
handed over the band `98.0 +-18.6` as a constant - it is arm B's control, whose seeds stop at
73-118 - and it is wrong for arm A, whose control has **both** seeds at the 120 cap
(`best_epoch` 118 and 116, mean 117.0, no early stops at all). Applied there it would have
called the arms matched or unmatched at random. **The band has to come from each arm's own
control**, and where that control never stopped early the test becomes categorical and
stronger: if the treatment stops on patience where the control never did, that *is* a
convergence difference, with no mean comparison needed. Note why that matters beyond
tidiness - at n=2 a difference of `best_epoch` means carries the same 8.98x penalty as any
other paired comparison, so the constant band would have imported the *same* unresolvable
design into the diagnostic written to catch it. **A check on an under-powered comparison can
be under-powered in the same way; state its resolution too.**

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

### `identity_margin` / `identity_scale`: the default is beaten by +0.016, and kept

AM-Softmax's margin (0.35) and scale (30.0) were the face-recognition defaults, tuned
against corpora with tens of thousands of identities, unchanged across 312 runs. Measured
2026-09-04 on 8 datasets, 419 identities, `bilstm`, `identity_softmax`, epochs 30, 5
stratified folds, all cells under one code identity (`6ac797f158`; the grid's 13 earlier
rows were reproduced bit-identically first, see the results-log section):

| margin / scale | folds | mean AUC | paired vs 0.35/30 | fold sd | t | won |
| --- | --- | --- | --- | --- | --- | --- |
| **0.35 / 30 (default)** | 5 | 0.7248 | - | | | |
| 0.1 / 30 | 5 | 0.7326 | +0.0077 | 0.0067 | 2.56 | 4/5 |
| **0.1 / 15** | 5 | **0.7409** | **+0.0160** | 0.0083 | **4.31** | **5/5** |
| 0.2 / 15 | 3 | 0.7410 | +0.0161 | 0.0103 | 2.71 | 3/3 |

Both levers point the same way and separate: a lower margin helps at fixed scale, and a
lower scale helps on top of it. The premise held - defaults tuned for tens of thousands of
identities push too hard at 419.

**Read it as "beaten by +0.016 on 4 of 8 cells", not as a tuned optimum.** 0.2/30, 0.5/15
and 0.5/30 never ran (the sweep was stopped to free the GPU), so 0.1/15 is the best of
four measured cells and biased upward; the 0.2/15 row is three folds and its missing two
are not missing at random. +0.016 is the size of the cross-session correction and a
quarter of the objective gain.

**The default stays at 0.35/30, deliberately.** Changing `configs/config.yaml` mid-programme
would put every subsequent run on a different footing from the 300+ rows it is compared
against, for a gain below the noise of most of those comparisons. Two consequences to
carry: every run at 0.35/30 is knowingly ~0.016 AUC below what the configuration can do,
so no 0.35/30 figure is a ceiling; and if a result ever lands within ~0.016 of a target,
the first question is whether the margin change closes it, not whether the idea failed.
Switching the default is a single deliberate decision, made once, with a note in every
table that straddles it - and the remaining 22 cells are the price of calling any setting
"best".

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

### A pairwise AUC implies a rank-1, and the implication should be computed first

Under an equal-variance Gaussian score model, d' = sqrt(2) * Phi^-1(AUC), and rank-1 at
gallery N is the probability a genuine score beats N-1 impostor draws. Checked
2026-09-04 on alyx: the per-axis lookup AUCs (0.593 / 0.661 / 0.539) imply rank-1 at N=17
of 0.103 / 0.149 / 0.075, and the enrolment harness measured 0.114 / 0.140 / 0.071. So
before any identification number is run, compute what its verification number already
implies; a rank-1 that lands more than ~0.05 from the implied value is the interesting
result (a score distribution far from Gaussian, i.e. a few very separable users), and one
that lands on it was already known. Two consequences from the same check: P(within <
between), a pairwise property, does not translate into 16-way rank-1 - alyx height has
P=0.743 and identifies at 0.16 among 17; and enrolment averaging cannot lift a static cue
whose limit is between-session shift (alyx k=1 to k=16: flat, whole-session ceiling 0.162).
**Head position alone as an enrolment system on the one cross-day corpus is 2.4x chance
at N=17 and 0.057 at a 70-person gallery. It is not an enrolment system** - but the learned
`dyn` branch is a different matter, and the table below it was retracted and reversed on
2026-09-05: read that paragraph before quoting anything about the model on alyx. The full static
table (Trainer, cross-session gallery vs probe, standardised, rank-1 at N=17, k = the
largest each corpus supports):

| corpus | sessions are | k | xyz | y only | xz only | implied xyz / y / xz |
| --- | --- | --- | --- | --- | --- | --- |
| ViewGauss | one sitting | 3 | 0.814 | 0.540 | 0.627 | 0.63 / 0.48 / 0.45 |
| Head_and_Gaze V2 | one sitting | 8 | 0.609 | 0.142 | 0.618 | 0.44 / 0.17 / 0.45 |
| VR_User_Behavior | one sitting | 16 | 0.790 | 0.114 | 0.832 | 0.20 / 0.13 / 0.18 |
| **alyx** | **different days** | 16 | **0.119** | **0.135** | 0.075 | 0.10 / 0.15 / 0.08 |
| BOXRR held-out (5 ckpts, 73-92 users) | across days | 16 | 0.407 | **0.379** | 0.242 | 0.25 / 0.33 / 0.17 |

BOXRR is the **only corpus where height beats placement** (y 0.379 against xz 0.242), and
every seated corpus is the reverse by a wide margin - which is what a modest standing
offset looks like against what a room looks like, and matches its geometry (height P 0.828
over lateral 0.685). Its population is fixed at k=16 like every other row, which drops 2-3
validation users per checkpoint against the lookup's lists. Two groups and nothing between
them, with BOXRR between them on the distance ratio. Where gallery and probe come from one sitting, xz
(placement) carries everything at 0.6-0.8 - a person not moving between clips, the same
category as BOXRR's standing offset at its easiest. Where they are days apart, xz collapses
to 0.075 and only height survives, at 0.135. **Head height alone never exceeds 0.34 at
N=17 anywhere except ViewGauss (0.540, 35 users, four sessions in one visit).** So section
10's 0.4-0.6 is met on three corpora, all same-sitting, and fails on the only corpus that
measures what a deployment would face. The Gaussian implication was within 0.01 on alyx
and undershot the seated corpora by up to 0.4: their score distributions are far from
Gaussian, because within-sitting placement is nearly constant per person (within-window sd
0.01-0.02 m) so genuine distances are tiny against widely spread impostors. Corpus
property, not harness choice: ViewGauss sessions hold exactly 3 windows at 5 s and
Head_and_Gaze 11, so k=16 is impossible there; population fixed at each corpus's maximum.
k-averaging gained +0.18 on VR_User_Behavior and nothing on alyx.

The compact version is the distance ratio, median genuine / median impostor at k=1:

| corpus | xyz | y | xz |
| --- | --- | --- | --- |
| ViewGauss | 0.144 | 0.176 | 0.132 |
| VR_User_Behavior | 0.301 | 0.531 | 0.183 |
| Head_and_Gaze V2 | 0.330 | 0.579 | 0.183 |
| **alyx** | 0.844 | **0.487** | **0.947** |

On the seated corpora a person's genuine lateral distance is a seventh of a stranger's - a
spike at zero, because they did not move between clips, which no Gaussian score model
represents. **On the only cross-day corpus a person's own head position is 0.95 of the
distance to a stranger's laterally, and 0.49 in height.** That sentence is what the static
cue is worth for a deployment.

**And the learned branch beats head height in every regime - after a retraction (step 6,
2026-09-04, corrected 2026-09-05).** All rows on a 14-user alyx gallery, chance 0.0714,
summed z-scored distances with no learned weight:

| alyx, rank-1 at N=14 | `dyn` alone | height alone | height + `dyn` |
| --- | --- | --- | --- |
| unseen **activity**, 5 s (LODO checkpoint, alyx never in training, 70 users) | 0.181 | 0.166 | **0.239** |
| unseen **users** of a seen activity, 5 s (in-domain folds `ddc9b964e5`) | 0.317 | 0.201 | **0.456** |
| unseen **users**, 10 s at 4096 identities (9.14 checkpoint) | **0.586** | 0.143 | 0.443 |

**The first version of this table was void and its conclusion was the opposite.** The `dyn`
checkpoints record `encoding=dyn`, and their sample indices were built without passing it -
`build_sample_index` defaults to `raw` - so a model trained on pose relative to the window's
own mean was scored on absolute pose. That is a different input distribution, not a degraded
one. It produced dyn 0.096 / 0.147 and fused 0.159 / 0.197, from which this file previously
concluded that "the best alyx number is head height alone and needs no model", that "fusion
adds nothing in either regime", and that the trained model "contributes nothing measurable".
**All three are false.** The model beats height in every regime, four-fold at ten seconds
with 4096 training identities, and fusion is worth +0.058 and +0.139 over the better single
cue. The anti-correlation across folds offered here to explain the wash was real in the
numbers and was explaining an artefact.

Two things survive unchanged: every static column (xyz / y / xz, the geometry, the distance
ratios, the same-sitting-placement versus cross-day-height split), which never ran a model
and whose calibration gate reproduced an independent figure on 15/15 cells; and the
no-fixed-weight problem, now with its sign reversed - at 10 s fusion *hurts* (0.443 against
0.586) because height is weak there and an equal-weight sum drags the stronger cue down.

**The lesson is about gate coverage, not about this table.** The static half had a
digit-exact gate because there was something independent to reproduce; the model half had
none, so nothing checked that the model was fed what it was trained on. **Any scoring of a
checkpoint outside the training path must first reproduce that checkpoint's own recorded
metric on its own recorded users** - if `selected_test_auc` does not come back, the harness
is feeding it something else, and no number from it means anything.

**Gating a checkpoint once buys the right to compare against its recorded row later, without
re-running it (Trainer, 2026-09-08).** The gate discipline was adopted to stop bad scoring,
and its by-product is a growing set of checkpoints certified under the current code identity.
Worked example: the five 419-identity 10 s `dyn` checkpoints of sweep `0840769514` sit at
`71c9783a14`, which crosses the `dyn` float64 re-baseline - normally that comparison has to
be earned. It already was, incidentally, when `score_nymeria.py --gate` reproduced all five
recorded figures under `8db420df4c` at gaps of 2.6e-6 to 5.1e-5
(`docs/acceptance/nymeria_gate.json`). So that arm is reusable as an experimental control and
five training runs are saved. **The cheapest control is one you already gated**, and it is
worth checking `docs/acceptance/` for a certificate before re-running anything for
code-identity reasons alone.

**That gate now has a harness, and it has been run: `score_nymeria.py --gate`.** It scores a
checkpoint through the pipeline's own `SiameseDataset` and `evaluate()` and writes a full
shard row (`mode=rescore`), after first re-scoring the checkpoint on its own recorded users.
Across 29 `dyn` transfer checkpoints, 28 came back within 1e-4 of their recorded figure
(1.5e-9 to 7.5e-5, cuDNN run-to-run) and the only miss was the `random` control, whose score
is noise by construction - which is the sensitivity you want from a gate: it passes what
should reproduce and fails what cannot. Use it for any scoring outside the training path.

**"It was gated" and "there is a committed certificate that it was gated" are different
claims, and only the second survives the session** (Trainer, 2026-09-08). **And a commit that
cannot reach `origin` is not a committed certificate** (2026-09-09): Miami's harness cannot
push to main, so its `boxrr_corpus_avalon_vs_desktopc.json` at `48ef785` exists on exactly
one disk - durable against that session ending, useless to any other session, and therefore
failing the property it was written for. A node that produces certificates needs a route to
origin; where the harness forbids one, the artefact goes to a peer who can push it, and the
commit is not cited until it is *there*. Check `git ls-tree origin/main` before treating a
peer's commit hash as a reference.
 A gate run inline
that prints its gaps and moves on leaves a log line; a gate that writes
`docs/acceptance/*_gate.json` leaves something a later session can cite to skip a re-run.
The five checkpoints reused as an experimental control were certified by the *Nymeria*
rescoring, not by the window-length work that also gated them - because only the first wrote
its artefact out. **Write the gate result to `docs/acceptance/` even when you only need it
inline**; the whole value of the reusable-control pattern above depends on the certificate
existing, not on the check having happened.

**A scripted edit reports that the write succeeded, not that the replacement matched, and
those come apart silently (2026-09-08).** Adding that rule to four harnesses by scripted
string replacement, two of the replacements did not match; the import and the call landed but
the loop that built `gates` did not, leaving `write_gate_certificate(..., gates)` with the
name never bound. **`py_compile` passes that cleanly** - it is a `NameError` at the end of a
two-hour run, not a syntax error. **Assert the match count before writing**
(`assert s.count(old) == 1`) - it turns a silent no-op into a failure at edit time, and after
three instances of a silent non-match in one evening it is the default here, not a nicety.

**What caught this one was reading the diff back**, not a checker: the edit tool printed the
result and lines 67-73 were visibly unchanged. Two claims about the checkers were made before
either was tested, and both were wrong - tested afterwards by reconstructing the buggy file:

| checker | on the buggy file |
| --- | --- |
| `compile()` / `py_compile` | **passes** - an unbound name is a `NameError` at runtime, not a syntax error |
| module-level pooled name pass | **flags `gates`** |
| per-function scoped pass | flags it |

The coordinator wrote that a module-level pass would have missed it because "the bug lived in
one function's scope". It did not - `gates` sits at **module scope**, and the failed
replacement left it assigned *nowhere*, so pooling had nothing to hide it behind. Per-function
scoping is still strictly stronger, for names assigned in one scope and read in another; this
bug simply was not that class. **That error was a plausible inference drawn from a real
general property, in a paragraph about not doing exactly that, when reconstructing the file
and running both checkers takes a minute.**

And Trainer's own claim was the mirror image: the static pass ran *after* the fix, so it
passed a file that was already correct and caught nothing. "The checker caught it" and "the
checker is now verified capable of catching it" are different claims. **A check that reports
success is a claim like any other, and so is one that reports failure** - the coverage scan
below reported five missing certificates that all existed. Same defect, opposite signs: both
sides stated the result of *running* a tool rather than the result of *testing* it. Verify the
property you wanted, not the exit code of the thing meant to produce it.

**Assert on the FIXTURE, not only on the result (2026-09-08).** Testing a memory guard, the
extraction that was supposed to pull the guard out into a file produced an **empty file**
twice - and an empty Python file exits 0, so the test printed "PASSES" twice while validating
nothing at all. That is the deepest form of the same defect: **a test whose subject failed to
load reports the subject's success.** This completes the set - a check can report a failure
that is not real (a regex that missed Windows paths), a success it has not earned (a checker
run after the fix), or a success about nothing (a fixture that never loaded) - **and the third
generalises furthest, because it does not require the checker to be wrong.** That checker was
correct; it was pointed at an empty file.

So the assertion goes on the fixture and it has to be **specific**: not "the file exists" but
"the thing I claim to be testing is in it". **The same rule aimed at a COMPARISON rather
than a fixture** (Miami, 2026-09-09): before diffing two manifests, assert that a known key
resolves on *both* sides after normalisation and that both hold the expected number of
entries. A stated path convention is still a claim about the other machine's output, and if
it is wrong the diff reports *everything* missing and *everything* extra - 17,874 of each -
which reads as catastrophe and is a prefix bug. Two lines convert that into an immediate stop
with sample keys printed. Both this project's false alarms of that shape (a coverage scan
reporting five absent certificates that all existed, a verifier reporting 146 missing and 146
extra on a corpus whose totals matched) would have been caught by it.
 `assert 'GlobalMemoryStatusEx' in body` is what
turned a silent pass into a caught error, and it is one line. A fixture check that only tests
for existence fails in exactly the same way as the guard it is protecting.

**The fixture rule caught a real error the next day, on the coordinator's own work
(2026-09-09).** The Across-XR geometry statistic was computed as
`mean(searchsorted(sorted_between, within)/n)`, which is **P(between < within)**, and printed
under the label `P(within<between)` - so lateral read 0.473 and height 0.246, and the honest
reading of those labels was "height carries nothing", the exact opposite of the truth. The
tell was internal: the printed medians said height's within-pairs were **0.028 m against
0.077 m between**, which cannot produce a low P(within<between). **A statistic that disagrees
with the summary printed beside it is wrong somewhere, and the cheapest resolution is a
fixture whose answer is known by construction** - here, synthetic users separated by 10 m,
which must return 1.000 and did, with an `assert` on it. One inverted comparison would have
put a backwards conclusion about the newest corpus into this file on its first day.

**Two guards that would have failed open, on Windows specifically.** Both were written to
protect the same chain and both were verified only after being challenged:

- **`bc` does not exist in this Git Bash.** The memory guard's fallback was `|| echo 1` -
  "enough memory" - so a guard written to stop a low-memory launch would have silently never
  fired. Do the comparison in Python, which is already a dependency.
- **`kill -0` cannot see a native Windows pid from Git Bash.** It returns non-zero for a pid
  that `OpenProcess` reports alive, so a wait loop would have declared a *running* training
  job missing and **queued a duplicate of it**. That one corrupts an arm rather than wasting
  time. Use a Windows-aware liveness probe and test it against both a real pid and a bogus one.

The pattern in both: **a guard whose failure mode is to pass is worse than no guard**, because
it also removes the caution that would otherwise apply. Verify a guard in both directions -
that it passes when it should and *blocks when it should* - or it is decoration.

**A mid-training number is not a result, however much it looks like one.** The orphaned arm A
seed 1 read test AUC **0.6188 at epoch 36** against a baseline of 0.6156 - which reads as
"+0.003, the registered band is landing" and is nothing of the sort: it is one seed, not
validation-selected, a third of the way through a budget whose control selected epochs 116-118.
Quote `selected_test_auc` from a completed row or quote nothing.

The same trap catches a *check*, and did: a coverage scan of `docs/acceptance/*gate*.json`
reported five sweeps with no certificate when all five had one, because its regex assumed
forward slashes and those entries hold absolute Windows paths. **A check that reports a
failure is a claim like any other**; confirm it the cheap way (substring, or one entry read
by eye) before acting on it.

### The learned branch identifies at 0.858 in domain, and averaging is why

The largest identification figure this project has measured, and it is **entirely
static-free** - `dyn` removes height, seat and placement, so none of it is the rig.
rank-1 at N=17, chance 0.0588, k at each corpus's maximum, the 9.3 five checkpoints,
both gates passed (2026-09-06, `docs/acceptance/step6_*`):

| corpus | k | users | height alone | `dyn` |
| --- | --- | --- | --- | --- |
| BOXRR held-out (**training activity**) | 16 | 73-92 | 0.380 | **0.858 +-0.009** |
| alyx held-out (training activity) | 16 | 12-17 | 0.178 | 0.630 +-0.129 |
| ViewGauss (unseen activity) | 3 | 35 | 0.541 | 0.187 +-0.055 |
| Head_and_Gaze (unseen activity) | 8 | 100 | 0.142 | 0.179 +-0.014 |
| VR_User_Behavior (unseen activity) | 16 | 48 | 0.115 | 0.245 +-0.010 |

**Measured clean: 0.862, and the optimism caveat is withdrawn rather than restated.** 94
BOXRR users no checkpoint ever sampled (2567 available, 100 drawn, 94 surviving the k=16
gate), the same pool for all five seeds - so the spread is the model, which the validation
column could not say because each seed scored different people there:

| clean BOXRR, N=17 | `dyn` | height | height+`dyn` |
| --- | --- | --- | --- |
| k=1 (5 s) | 0.449 +-0.024 | 0.365 | 0.558 |
| k=4 (20 s) | 0.730 +-0.017 | 0.333 | 0.804 |
| **k=16 (80 s)** | **0.862 +-0.019** | 0.386 | **0.902** |

Against 0.858 +-0.009 on validation users: **+0.004, inside one seed's spread and the wrong
sign for optimism.** Height is flat across k with *zero* spread across seeds, since it is read
from recorded positions and no checkpoint touches it - the averaging mechanism again, on
users chosen for it. **0.902 is the largest identification figure in the project**, on users
no checkpoint has seen, and its static half is head height rather than placement.

**Two qualifications survive and both must stay attached.** It is 80 s of enrolment *and*
80 s of probe, so **the k=1 row at 0.449 is the one at evidence comparable to a published
single-15 s-window figure** - and that published figure also has both controllers. And it is
the training activity: the same checkpoints read 0.18-0.25 on an unseen one. alyx's row above
is a 12-17 user gallery, a direction rather than a measurement.

**BOXRR identifies far better than its own verification number implies, and every rank-1
here is a population mean over users who differ enormously (measured 2026-09-06).** The
Gaussian mapping (`d' = sqrt(2)*Phi^-1(AUC)`) is computed on the *same score set and the same
users* as the rank-1 - one distance matrix read two ways - so this is exact, not a
cross-population estimate, and the formula was gated first against alyx's published triple
(0.593/0.661/0.539 -> 0.103/0.150/0.075 against 0.103/0.149/0.075):

| clean 94 BOXRR users | AUC | implied rank-1 @N=17 | measured | offset |
| --- | --- | --- | --- | --- |
| k=1 (5 s) | 0.8188 | 0.340 | **0.449** | **+0.109 +-0.006** |
| k=16 (80 s) | 0.9619 | 0.746 | **0.862** | **+0.116 +-0.018** |

Registered beforehand at +0.06 to survive: it survives, and it is stable across a five-fold
change in enrolment evidence.

**The minority reading was the coordinator's and it is wrong - measured, not argued.** A heavy
right tail and a merely narrower-than-Gaussian distribution both produce this offset, so
per-user rank-1 was compared against a simulated null in which every user is identical by
construction: sd across users **0.283 against a null 0.012 (24x)**, p10/p50/p90
0.067/0.442/0.898 against 0.325/0.340/0.356, **15 of 94 users above 0.80 and 13 below 0.10**
where the null produces none of either. So the departure from the score model is enormous -
but the *concentration* is not: the top decile carries 20.1% of correct identifications
against a null share of 10.2%, the top quartile 45.3% against 25.6%. Twice its share, not ten
times. That is a broad continuum of per-user separability, not a minority carrying the result,
and **the average is real and stays**.

**What changes is the sentence, and it is worth more than the minority story would have
been.** "The model identifies BOXRR players at 0.862" and "a BOXRR player has an 0.862 chance
of being identified" are different claims and only the first is supported. **Every rank-1 in
this project is a population mean over a distribution 24x wider than its score model implies,
so any per-user claim needs the distribution rather than the mean** - cheap to check on any
existing checkpoint, since the distance matrix is already there. For a biometric that is the
substance and not a technicality: the 15 users above 0.90 are exposed at a rate the headline
hides, and the 13 at chance are protected at a rate it hides equally.

**The offset repeats everywhere measured, and 0.948 clean supersedes 0.862.** Five points,
each an exact-population reading of one score set (2026-09-06):

| checkpoint / protocol | users | AUC | implied | measured | offset |
| --- | --- | --- | --- | --- | --- |
| 419 ids, 5 s, k=1 | 94 clean | 0.8188 | 0.340 | 0.449 | +0.109 |
| 419 ids, 10 s, k=8 | 94 clean | 0.9597 | 0.735 | 0.842 | +0.107 |
| 419 ids, 5 s, k=16 | 94 clean | 0.9619 | 0.746 | 0.862 | +0.116 |
| **2096 ids, 10 s, k=8** | **92 clean** | 0.9850 | 0.874 | **0.948** | +0.074 |
| 4096 ids, 10 s, k=8 | 1684 validation | 0.9907 | 0.914 | 0.960 | +0.046 |

**0.948 on 92 users no checkpoint has seen is the largest clean identification figure in the
project** (the 2096-identity arm trained at `max_users=BOXRR-23_Dataset=2020`, leaving 1012
of 4020 users in neither draw). The 4096 row is a validation-user figure and must be labelled
one wherever it appears.

**The offsets fall monotonically, the raw trend is unreadable, and the underlying question is
now settled anyway.** The offset is bounded above by the headroom `1 - implied`, and at AUC
0.99 the implication is already 0.914, so the largest offset arithmetically possible there is
0.086 - below what 419 identities measured. The ceiling forces the raw quantity down whatever
the distribution does, normalising by headroom reverses the direction but moves just as hard
with *evidence alone* (0.165 to 0.457 at fixed identity count on the same users), and the
4096 row changes population as well. So the four-point sequence says nothing on its own.

**Answered by holding everything but identity count fixed, then rescaling.** On the 1012
users clean for *both* arms, at 10 s and k=8, so identity count is the only difference:
419 identities read AUC 0.9583 / implied 0.728 / measured 0.824 / offset **+0.096**, and 2096
read 0.9850 / 0.874 / 0.948 / **+0.074**. The offset still falls by 0.022 with population,
evidence and window length all fixed, so there was a real difference to explain. Rescaling the
419 genuine scores up to the 2096 AUC and recomputing rank-1 explains it entirely:

| monotone map | rescaled rank-1 | measured - rescaled |
| --- | --- | --- |
| **shift** (preserves every gap) | 0.947 | **+0.001** |
| scale (preserves every ratio) | 0.932 | +0.016 |
| stretch (rank order only) | 0.938 | +0.010 |

All three inside the registered +0.02. **The shape did not change; the shrinkage is
arithmetic, and the non-Gaussianity is a stable feature of the task rather than something
that erodes as the model improves.**

**The band passed and the predicted mechanism was wrong - record both.** Trainer registered
the band expecting more identities to help the *hard* users most, reducing the heterogeneity
that produces the offset, which would have shown as measured **below** rescaled. All three
deltas came out slightly positive and the best-fitting map by a wide margin is the **shift**,
at +0.001 - and a shift moves every genuine score by the same constant. So the 419 to 2096
improvement is close to a uniform translation: **everyone improved by about the same amount**,
which is the opposite of the mechanism offered. The entry is "shape unchanged, improvement
uniform", not "heterogeneity reduced". **A registered band can pass for a reason other than
the one predicted, so record the mechanism separately from the band** - otherwise a wrong
model gets confirmed by a right number, which is worse than a failed prediction because
nothing prompts anyone to look again.

**The actionable form of that, which is Trainer's own qualification of it**: the catch was a
property of the design, not of anyone's attention. Running *one* map would have hidden it in
either direction - the shift alone reads +0.001 and "band held", the scale alone reads +0.016
and "weak support", and neither shows that the sign is wrong. So the rule to act on is **run
enough variants that a wrong mechanism has somewhere to show up**; "record the mechanism
separately" is the thing you then notice, not a thing you can do on purpose.

**The consequence reaches past the table.** The per-user split is not a data-quantity
artefact: it survives a five-fold increase in training identities intact - the ~16% of users
almost always identified and the ~14% almost never are still there at 2096 identities, shifted
along with everyone else. **So the split is a property of the people, not a transitional state
more data fixes**, and a deployment claim about "the identification rate" conceals a stable
split rather than a temporary one. Caveats kept: two seeds per arm; the three maps are not
equally flexible, and the shift is the most constrained, which makes its near-exact agreement
more informative rather than less; and this is one corpus and one activity.

**A free diagnostic fell out of it.** *Which* monotone map reproduces an improvement tells you
the shape of that improvement - uniform, proportional, or rank-only - and it costs nothing
wherever two checkpoints are scored on one population. Worth reaching for on any axis where
"who did it help?" matters and only the mean is in hand.

**The 0.785 trap dissolved structurally rather than being avoided.** The registered warning
was that 0.970 verification implies rank-1 0.785, the published figure, and a measured 0.785
would be misread as agreement with the literature. It never arose - because on the population
actually scored the AUC is 0.9907, not 0.970, so the implication is 0.914 and nothing lands
near 0.785. The 0.970 belonged to a different population. **An implication computed on one
population and compared against a measurement on another is a lead, not a check; the fix is
to compute both from one score set, not to annotate the mismatch.** That is the same shape as
the `lookup_auc` rule: derive the comparison from the thing itself rather than from a
neighbouring record.

**The mechanism is the finding, not the number.** Population fixed from k=16, BOXRR:

| k | 1 | 3 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| `dyn` | 0.407 | 0.656 | 0.713 | 0.814 | **0.858** |
| height alone | 0.356 | 0.368 | 0.358 | 0.374 | 0.380 |

**Enrolment averaging lifts the learned cue by +0.45 and the static cue by +0.02.** A static
cue's error is a between-session *bias* - where the headset sat that day - and averaging more
windows from the same session cannot remove a bias. A learned cue's error is per-window
*variance*, which averaging does remove. That extends the registered "enrolment averaging
cannot lift a static cue" from alyx to BOXRR and supplies the contrast case it never had, and
it is why k must be reported with every rank-1: at k=1 the two cues are 0.407 against 0.356
and the whole result would read as marginal.

**A k confound, caught in analysis and worth imitating.** ViewGauss sits at k=3 and
Head_and_Gaze at k=8 because that is all their sessions hold, so part of their low figure is
less evidence rather than an unseen activity. Matched at the same k, BOXRR reads 0.656 and
0.814, so the activity gap survives - but the unmatched table overstates it, and the
comparison is only honest at matched k.

**Fusion has a rule now instead of a prediction, and the rule has been forecast-tested.**
Equal-weight height+`dyn` fusion *adds* below a cue ratio of ~2.3x and *subtracts* above
~2.9x, monotone, loss growing with the ratio - so it helps on BOXRR (0.867 against 0.858) and
hurts on alyx (0.568 against 0.630). Formed on six corpora, it then made three out-of-sample
calls on the clean BOXRR population (ratios 1.2x / 2.2x / 2.2x, all "adds") and got all three,
observed +0.109 / +0.074 / +0.040 - the gain shrinking as the ratio grows, which the rule
asserts rather than merely permits. State it as a forecast: **equal-weight fusion is worth
having when the weaker cue is within about half the stronger, and worth avoiding beyond
roughly a third.** The
prediction it replaces argued from "static is strong here", which was true of *placement* on
the seated corpora and false of height - the harness fuses height, and fusing placement would
mean fusing the artefact this file refuses to report as biometric.

**The second gate is the transferable part.** A checkpoint gate proves the model is fed what
it was trained on; it says nothing about the *enrolment protocol* - population, k, which
session is gallery, the rng, tie handling - because none of that exists on the training path,
and that is what voided the step 6 columns twice. So the harness also recomputed the nine
published static rank-1 figures and had to land on them: all nine within 0.002, BOXRR height
0.380 against the published 0.379. **An out-of-path harness should reproduce something from
the column it will be compared against, not only something from the checkpoint.**

**LODO says the seated ceiling is theirs.** All three leave-one-corpus-out deltas are
negative (-0.034, -0.009, -0.065), so training on six other seated corpora made a *weaker*
seated identifier than Beat Saber plus Alyx did.

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

JSONL removes the class instead of patching it - a union of self-describing records is correct whatever schema either side used, adding a field is a non-event, and appending never rewrites a line. `run_id` makes every line unique so union can't coalesce two runs that agree on all fields. Tests cover the property, not just the writing: `test_union_merging_two_schemas_keeps_every_field_on_the_right_row` reproduces the exact merge that corrupted the file. **`sweep_id` is only a valid grouping key for rows written at or after `5b61fc0`.** Before that commit the id ignored every top-level config key, so rows from two different experiments can share one — in this file, the 48-identity subsample runs sit under `d6cb92c8a9` alongside the 343-identity pooled runs. They separate on `max_users` (blank vs 48), but grouping on `sweep_id` alone merges them. `sweep_id` also under-partitions for a second reason: runs made before and after a bugfix share it when the config is identical. Those separate on `code_identity`. **A third instance, and it is live (2026-09-08): a `mode=rescore` row inherits the
`sweep_id` of the checkpoint it scored.** Sweep `0840769514` holds ten rows - five
`mode=train` transfer runs averaging **0.5997** and five `nymeria_rescored` rows averaging
**0.5374** - and a mean over the `sweep_id` returns **0.5685**, which is neither quantity and
looks entirely plausible. That sweep is the control arm of a queued experiment, so the
analysis must filter `mode=='train'` and `experiment=='transfer'`; grouping on the id alone
hands the treatment a spurious +0.03 before it runs. **Rescoring makes every gated sweep a
mixture, so `sweep_id` is now never sufficient on its own.** When analysing rows that straddle that commit, group on the config columns (`max_users`, `objective`, `normalize`, `channels`, `center_position`, `cross_session_positives`, `num_data_dirs`) rather than trusting the id.

**`code_identity` invalidation was tested once, and the trade held (2026-09-04).** The
margin/scale grid ran at `67c63fa767`; eight `model/*.py` files changed afterwards (the
lookup baseline, per-dataset metrics, EER, checkpoint serialisation) and the tree hashed
to `6ac797f158`. Re-running one grid cell over all five folds under the new identity
reproduced every recorded field on `repr` - `selected_test_auc`, `selected_test_acc`,
`best_epoch`, `best_test_auc`, `final_train_loss` - so the whole optimisation trajectory
was identical and those commits changed no training numerics. The invalidation was
unnecessary in hindsight and cost five runs to prove; the alternative silently reuses
results across a numerics change. **Do not loosen the digest** because of this. Rows at the
two identities are comparable, and any comparison across a code change should be earned
the same way: one cell, every fold, identical on `repr`. **One re-baseline is on record**: the
`dyn` residual moved to float64 at `06f57e5` (code identity `bc521f7f8e`); scored on
CPU before and after on one dyn checkpoint it changed PanoSaliency by 1.2e-4 AUC and no
other corpus by more than 7e-7, so every `dyn` row after that commit is under the new
identity and PanoSaliency's `dyn` figures straddle a 1.2e-4 step. `docs/acceptance/`
holds both sides. **Two more code-identity steps are on record and neither is a re-baseline.** The
amplitude / recorded-position baselines (`position_lookup_auc`, `amplitude_auc`, merge `9277648`,
2026-09-06) added columns and touched no numerics - `evaluate()` on a raw and a dyn checkpoint
reproduced every pre-existing figure digit-exact on CPU before and after, and both recorded
rows digit-exact on the GPU (`docs/acceptance/amplitude_*`). Then `code_identity()` was found
to hash raw bytes and follow line endings: the same merged code read **`4d243b05d0`** from the
stored LF blobs, **`100bd18472`** from a clean CRLF checkout, and **`72b8053ec2`** from this
machine's checkout, where `model/extractors/_kinematics.py` alone sits on disk with LF - so
every identity recorded here (`bc521f7f8e` included) depended on one file's line ending.
`digest_tree` now normalises line endings (merge `20b67bd`; acceptance
`docs/acceptance/code_identity_line_endings.py`: working tree and stored blobs agree), and
`.gitattributes` checks `*.py` out with LF everywhere (`bacb45a`, identity-neutral after the
fix). **The identity on main is now `8db420df4c`**; rows at `bc521f7f8e`, `72b8053ec2` and
`8db420df4c` are comparable and no figure moved. A digest that names no commit names a dirty
tree, and a tree with mixed line endings is one such: check `git ls-files --eol` before
reading an identity off a machine you did not set up.

**One code, three identities, and the third was a mixed-endings tree (settled 2026-09-06).**
`code_identity()` hashed `path.read_bytes()`, so it followed each file's line endings on
disk. Reconstructing `model/` from the stored blobs of `1e3adf3` in each state and hashing
with the old algorithm reproduces every identity this project has argued about, digit-exact:

| tree state | old digest |
| --- | --- |
| all LF (a Linux or `autocrlf=false` checkout) | `4d243b05d0` |
| all CRLF (a clean Windows checkout) | `100bd18472` |
| **this machine's tree**: CRLF except `model/extractors/_kinematics.py`, which sits on disk with LF | **`72b8053ec2`** |

So the identity recorded on every row written here - `bc521f7f8e` and its predecessors
included - depended on one file's line ending, and **no clean checkout of any commit
reproduces it**. Fixed at `20b67bd` by normalising CRLF to LF inside the digest, which
collapses all three to one; `.gitattributes` gained `*.py eol=lf` at `bacb45a` afterwards
and left the identity unchanged, which is what doing the hash fix first bought. Identity on
main is now **`8db420df4c`**, working tree and stored blobs agreeing, and the fixed digest
maps the pre-fix tree to `4d243b05d0` - that is the pair to use when relating old rows to
new. The change touches no numerics: `code_identity()` is called only by the logger and by
`sweep.py`'s resume key.

**Confirmed on real hardware, in both directions (2026-09-09).** Every identity this project
had recorded came from a Windows checkout, and `4d243b05d0` was *reconstructed from stored
blobs* rather than observed - so the fix had never actually been tested off Windows. The
Miami node is a genuine Linux tree, `git ls-files --eol model/` all `w/lf`:

| machine | working tree | `code_identity()` |
| --- | --- | --- |
| Miami (`feng-MS-7B51`) | 29 files LF | **`8db420df4c`** |
| DESKTOP-C | 28 CRLF + 1 LF (the *mixed* tree) | **`8db420df4c`** |

and the harder half: **that reading is not vacuous.** On an all-LF tree the normalisation is
a no-op, so "it matches" is equally consistent with "the fix works" and "the fix was never
exercised". Miami built the CRLF twin - 29 files converted, `assert b'
' in body` before
hashing - and ran both algorithms over both trees: the **old** digest reads `8db420df4c` on
LF against `3c18c64173` on CRLF, the **new** one reads `8db420df4c` on both, with
`assert old_lf != old_crlf` so it cannot pass vacuously. Two real machines whose trees differ
in line endings - one of them the mixed tree that produced `72b8053ec2` under the old
algorithm - now agree, and rows from either are comparable by observation rather than by
argument.

**But the digest does not cover the code that MAKES the corpus (Miami, 2026-09-09).**
`_CODE_ROOT = REPO_ROOT / "model"`, and `prepare_boxrr.py`, `prepare_nymeria.py`,
`prepare_across_xr.py` and `prepare_who_is_alyx.py` all sit at the repo root - **outside the
identity**. So two machines can hold materially different `processed_datasets/` - an `xror`
version bump, a pandas float path, a converter fix - while every row from both reports the
same `code_identity`, and nothing in the log can tell them apart. **`code_identity` certifies
the model code and says nothing about the data**, which is this project's recurring bug
wearing a new hat: a stand-in that looks like the thing being checked.

Two consequences. **Copy an already-gated corpus rather than reconverting it** whenever the
choice exists - it keeps rows comparable by construction. And the gap is worth closing
properly: **record a corpus digest per run** - the sorted processed-file list with sizes, or
a content hash - so a corpus difference is visible in the row instead of being invisible by
design. Until that exists, a cross-machine comparison assumes the corpora match and cannot
check it.

**The digest does not cover the DEPENDENCIES either, and nothing else records them
(2026-09-09).** `code_identity` hashes `model/*.py`; `results_log.py` imports `platform`
solely to name the shard file; and **across all 341 rows of DESKTOP-C's shard and all 82
entries of `FIELDS`, the number of keys naming a Python version, a torch version, a CUDA
version, a device or a host is ZERO**. (Quote it that way rather than as a key count: rows
carry 50-75 keys depending on the run type, modal 69, because the JSONL schema varies by
design - so "a row has N keys" is not a fact about the format.) So two machines can produce numerically different
rows with the same `code_identity` and nothing in the record distinguishing them - which is
not hypothetical, because this file already documents CPU and GPU scoring differing by up to
**7e-4 AUC** and requires same-device acceptance for numerics-touching changes, *while the
row does not say which device was used.* The divergence is live as of today:

| | Python | torch | device |
| --- | --- | --- | --- |
| DESKTOP-C | 3.12.10 | 2.10.0+cu130 | RTX 5060 Ti |
| Miami (primary from today) | 3.13 (3.14 is a hard blocker, below) | rebuilding | RTX 4060 Ti |

**Proposed, not done:** append `python_version`, `torch_version`, `cuda_version` and `device`
to the row. It is additive, which the JSONL design explicitly supports - old lines are
untouched and the combined view backfills blanks - but it touches `model/*.py` and so moves
`code_identity`, which makes it a merge-window decision rather than a quick fix. Do it before
the first cross-machine comparison, not after.

**THE CROSS-MACHINE GATE PASSED, AND THE NUMPY EXPECTATION REGISTERED BELOW WAS WRONG
(2026-09-09).** Two machines, deliberately different on every axis, on the agreed spec -
BOXRR, first 200 users in `sorted()` order, 10 s/20 Hz/stride 5, `raw`, seed 67, 64
pairs/user, metrics on CPU:

| | DESKTOP-C | Miami |
| --- | --- | --- |
| stack | py 3.12.10, numpy **2.4.2**, torch 2.10.0, RTX 5060 Ti | py 3.13.15, numpy **2.5.3**, torch 2.14.0, RTX 4060 Ti |
| users / windows / pairs | 200 / 30,630 / 12,800 | 200 / 30,630 / 12,800 |
| own manifest sha256 | `4b45ea68c92ddd70...` | `4b45ea68c92ddd70...` |
| `position_lookup_auc` | 0.7798685424804688 | 0.7798685424804688 |
| `amplitude_auc` | 0.5664354736328125 | 0.5664354736328125 |

**Both AUCs are bit-identical - |delta| exactly 0.0, not merely inside the 1e-12 band** - and
the counts match. So the two gaps this gate exists for are closed for these machines: the
corpora agree in content and not merely in size, and two different stacks compute the same
training-free arithmetic. That is the measurement that replaces the `explicitly_not_shown`
content caveat in `boxrr_corpus_avalon_vs_desktopc.json`.

**And the registered expectation failed in the useful direction.** Both sides predicted, in
writing beforehand, that numpy 2.4.2 against 2.5.3 would draw *different pairs* from the same
seed - `Generator` streams carry no stability guarantee - and that a manifest-hash mismatch
would therefore not be a finding. **The hashes are identical.** So the hazard is real as a
licence (numpy still guarantees nothing) but did not materialise across these two minors for
the calls `generate_pair_manifest` makes. Keep the caution and drop the expectation: **do not
assume a numpy bump has moved the draw, and do not assume it has not - hash the manifest.**

**A verification that happens to be true is not a verification (Miami, 2026-09-09).** When
the gate refused, the substance was closable by hand - both own-manifests equalled the shared
one, so the pairs provably matched. That hand-closure was *correct here and would have looked
identical if it were wrong*: had the emitter's draw differed from the file it emitted, the
same chain of reasoning would have produced the same confident sentence. **The mechanism
refused and a human argument overrode it, and the argument was right by luck.** Re-run the
tool in the mode that demonstrates the claim instead. This is the same family as the
fixture rule one level up: there, a check passed on nothing; here, a check was bypassed by
reasoning that happened to hold.

**And scope what the pass closes, because it will be over-quoted.** The gate closes two
things and no more: the corpora agree in **content** (identical arithmetic over 30,630
windows drawn from those files is a content measurement, which the size manifest explicitly
was not), and **these two stacks** compute the same **training-free** arithmetic. It says
nothing about the model path, where cuDNN's BiLSTM is documented at up to 7e-4 between
devices, and it does not make the missing env annotation harmless in general - it makes one
pair of stacks measured rather than assumed, on one class of quantity. **"The stacks agree"
is not what was shown.**

**One protocol gap found by the gate refusing.** The emitting side ran with
`--emit-manifest` only, so its report records `scored_manifest_sha256: "own"` and the guard
correctly declined to compare - a side that scores "own" cannot be shown to have scored the
*shared* pairs, even when it did. **The emitter must also pass `--use-manifest` pointing at
its own emitted file.** The substance was verifiable by hand here only because the hash chain
closes (both own-manifests equal the shared one), which is luck rather than design.

**NUMPY'S VERSION CHANGES WHICH PAIRS ARE DRAWN, and the machines already differ (Miami,
2026-09-09).** NumPy freezes the stream of legacy `RandomState`, but **`Generator` method
streams are explicitly not guaranteed stable across feature releases** - and
`generate_pair_manifest` seeds a `default_rng` and then calls `rng.choice` five times per
user plus `rng.permutation`. So two machines on different numpy minor versions draw
**different pairs from the same seed**. Measured: **DESKTOP-C numpy 2.4.2, Miami numpy
2.5.3.**

This is worse than the other environment gaps rather than another instance of them. Those
threaten numerics at 1e-7; this changes the *inputs*. This file already measures what a
different pair draw is worth - the `transfer` against `transfer_rescored` arbitration put it
at **1e-3 to 3e-3 AUC**, larger than the 7e-4 CPU/GPU gap - so a numpy minor bump between two
machines can move a figure by more than the device does, invisibly, under one
`code_identity`. **`numpy` belongs in the env annotation, ahead of `torch`**, and any
cross-machine comparison of a manifest-derived figure is suspect until the versions are
either matched or the manifest is exchanged.

**So exchange the MANIFEST, not just its hash.** A hash tells you the streams diverged and
then stops - and stopping there is the failure, because the layer we actually care about
(does this stack compute the same arithmetic?) never gets tested. Ship the manifest's
`x1_indices`, `x2_indices` and `labels` as an input and both machines score **the same
pairs**, which isolates the environment layer even when numpy differs. It is ~300 KB as
int32 and compresses; it is committable, and once committed the gate is reproducible by a
third machine that has neither version.

**Force CPU explicitly, and not with `CUDA_VISIBLE_DEVICES`.** The lookups run through
`torch` (`static_position_lookup`, `amplitude_lookup` take tensors), so device is a real
variable for them rather than a formality - and on DESKTOP-C `CUDA_VISIBLE_DEVICES=""`
leaves `torch.cuda.is_available()` **True with zero devices**, so the usual mechanism does
not do what it appears to. Set the device in config.

**The training-free baselines are a free cross-machine gate, and they cover BOTH open gaps at
once.** `position_lookup_auc` and `amplitude_auc` need no model, no GPU and no training: given
the same sample index and the same pair manifest they are deterministic, so two machines must
agree to floating-point precision. A disagreement can only come from the corpus content or
the environment - which are exactly the two things nothing currently checks. The size manifest
explicitly did **not** establish content equality, and the row records no stack, so one
training-free run on an agreed user list and seed tests both in minutes, against ~2 h a side
for `sha256` over 42.8 GB that would test only the first. **Run it before any cross-machine
result is compared**, and prefer it to any check that compares bytes, because what matters is
not whether the disks agree but whether the arithmetic does.

**Two preconditions neither party stated, and both were checked before the gate ran**
(Miami): that the measurement is **deterministic on one machine** - run twice locally, same
counts, same hash, same AUCs - because if it is not, a cross-machine comparison measures
nothing; and that the **transport is faithful** - scoring your own pairs back through the
exchanged file reproduces the AUC exactly, so the format is not quietly changing the number
it exists to carry. **A comparison across machines assumes a stability within one that
nobody had tested.**

**Hash a transferred artefact, and reconcile even when the difference is harmless.** The
gate reached origin by paste, and the two copies differed: **one extra blank line**, with
`ast.dump` identical on both parses, so semantically nothing. Reconciled to the committed
copy anyway, because *"it is only whitespace"* is exactly the argument that lets two
implementations drift apart one harmless line at a time - once a file is committed it is the
definition and every other copy is a copy. Note the asymmetry that makes the hash worth
running (Miami): **a mangled paste breaks loudly, but a paste that drops a blank line or a
comment does not** - and the silent case is the one a hash catches and reading does not.

**Establish the layers in order, and never build a shared artefact on an unverified input.**
A manifest emitted from a corpus that has not passed its own file-by-file check would carry a
layer-1 fault into the exchange, where it surfaces as a layer-3 disagreement - **the wrong
answer arriving convincingly**, with the environment blamed for a corpus problem. That
generalises past this gate: a fault in an early layer does not announce itself as one, it
presents as a finding in a later layer that was working correctly.

And the gate must be **one implementation, not two**. Two independently written comparators
can disagree for reasons that have nothing to do with the machines being compared, which is
the failure the gate exists to rule out - so the script is exchanged along with the manifest,
and both sides run the same code. A shared manifest also indexes into the *local* index, so
out-of-range indices must be reported as a named layer-1 corpus failure rather than dying
with an `IndexError` - that is the case where the two user lists differ, and it should read
as a finding rather than a crash.

Miami's first row supplies the encouraging-but-insufficient version: alyx
`position_lookup_auc` **0.6006** against this file's ~0.593 for the alyx xyz lookup. It
reported that as a consistency signal and explicitly not a reproduction - different held-out
users, different manifest, 10 s against 5 s windows, different machine - which is the right
call and the reason the real gate is worth running. **Do not let a near-miss on different
inputs stand in for a match on identical ones.**

**And whatever lands must say what a MISSING env block means** (Miami). The change moves
`code_identity`, so there will be a sharp line with un-annotated rows before it and annotated
rows after - and the rows that most need the annotation are the ones already written. State
in the file that an absent env block means *"written before env annotation existed"*, never
*"unknown stack"*, or a later reader treats the blank as a measurement. Cheap to write now,
impossible to reconstruct later - and this project has already been bitten by a blank read as
a value (`epochs` and `early_stopping_patience` reading `None` on the older transfer rows,
which had to be recovered arithmetically from `best_epoch` and `epochs_run`).

**The sharpest form of it is Miami's**: this is not a gap in coverage, it is **a rule with no
referent**. The same-device requirement is not a note, it is the acceptance standard for any
numerics-touching change - and the record it governs cannot say which device ran, so every
acceptance that ever cited it was resting on someone remembering.

**Python 3.14 is a hard blocker for the pipeline, and a green test suite did not reveal it.**
Hydra 1.3.6 - the newest release, so there is nothing to upgrade to - passes
`LazyCompletionHelp()` to `add_argument`, and 3.14's argparse added a `_check_help` that does
`'%' not in help_string` on an object with no `__contains__`: `ValueError: badly formed help
string`. Every entry point goes through `@hydra.main`, so train, sweep, test and curve all
die. **475 tests passed on that machine while nothing could run**, because the suite never
invokes Hydra's argument parser - it surfaced only when a real training smoke test was
attempted. **A green suite adjacent to the thing you care about is not evidence about the
thing you care about**, and "the tests pass" was reported as "the node is operational". Use a
3.13 interpreter. Miami rejected monkeypatching the installed Hydra for the right reason: a
patched dependency on one machine against an unpatched one elsewhere is another invisible
cross-machine difference, of exactly the kind the paragraph above is about.

**Design the digest over the CSV payload only** (Miami): exclude `PROVENANCE.md` and
`CITATION.txt`, because those legitimately differ per machine - a provenance file records
where and when the conversion ran. Include them and **every machine reports a different
corpus on day one and the digest becomes noise**, which is how a guard gets switched off.

**Measured instance, 2026-09-09.** Data could not confirm DESKTOP-C's BOXRR state from AVALON
and raised the possibility it was still at 2,020 users - which would have meant the
4096-identity arm silently trained on ~2,096 while recording 4,096, making 9.14's saturation
result a comparison of a corpus with itself. **False alarm, and worth the message**: both
machines hold 4,020 users and 17,874 files. But the byte totals differed by **191**, and the
whole delta is `PROVENANCE.md` (AVALON 4,832, DESKTOP-C 4,641) with `CITATION.txt` identical
and the **CSV payload matching to the byte at 42,828,346,579 across 17,872 files**.

**Miami then refused its own answer, and was right to.** A total-against-total comparison
cannot exclude two files differing in compensating directions, and says nothing about content
at equal size - the exact objection it had raised to Data an hour earlier, applied against a
result that had come out the way both of us wanted. So the per-file manifest is committed at
`docs/acceptance/boxrr_manifest_desktop-c.txt.gz` (17,874 lines, sha256
`ebee5cd9...d8a1c5f6` uncompressed, paths relative to the corpus root, sorted by path) to be
diffed entry-by-entry against AVALON's. **"The totals agree" and "all 17,874 files agree
individually" are different claims, and this project has a documented habit of the second
sentence outliving the first.**

**The rule this replaces was mine and was wrong.** I inferred from "no commit hashes to
`100bd18472`" that it came from a dirty tree, and wrote "a digest that names no commit names
a dirty tree". It was a clean checkout - of the same commit, on a machine with different
line endings. **A content digest names a byte-state, and a commit is not one byte-state**;
which of them you get depends on `core.autocrlf`, `.gitattributes`, and whatever wrote each
file last. The check that settles a question like this is reconstruction from the stored
blobs in each candidate state, not an argument from what is absent in the log.

**Two rows for one checkpoint can be two measurements, and `lookup_auc` tells you which
(arbitrated 2026-09-06).** The 9.3 `dyn` checkpoints carry a `mode=train` transfer figure
(0.5811-0.5834) and an `experiment=transfer_rescored` one (0.5781-0.5845) differing by 1e-3
to 3e-3 - larger than the 7e-4 CPU/GPU gap, so not arithmetic. They are **not** a
reproduction and its target: `num_excluded_users` is 5 on the training rows and 0 on the
rescored ones, so VR_User_Behavior is 43 users in one and 48 in the other (the documented
`exclude_users` trap), the rescored rows record no `unseen_datasets` policy, and the pair
draw differs **on every corpus, including the six whose population is unchanged** - because
`generate_pair_manifest` runs one rng over the *pooled* index, so five extra users shift the
stream for every user drawn after them. Gate a checkpoint against its **`mode=train`** row;
that is the only row it can reproduce, and `score_nymeria.py`'s gate says so.

Neither family is wrong and the rescored one is the better population - 48 VR_User_Behavior
users rather than the 43 the `exclude_users` default silently produced. They are simply not
interchangeable: say which family a quoted transfer figure comes from.

The diagnostic generalises and costs one column: `lookup_auc` is training-free, so it cannot
move for any reason involving the model. Here it moves (0.5055 -> 0.5047 pooled) and moves on
corpora whose population is identical between the two rows - EyeNavGS 0.4925 -> 0.5001,
Panonut360 0.4969 -> 0.4887, ViewGauss 0.5022 -> 0.4933. That is decisive before any
checkpoint is loaded: **if the model-free baseline moved, the evaluation set moved**, and no
amount of numerics will explain the difference.

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

Current baseline: **475 passing** - 16.7s on DESKTOP-C, 6.3s on Miami. The suite has grown
rather than broken; the previous "256 passing, ~10s" was stale. Both machines report the
same count, which is a cheap corroboration that the two checkouts are the same code.

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

## CPU and GPU scoring differ by up to 7e-4 AUC

Measured 2026-09-04: scoring a `dyn` checkpoint (`314cd507f1`) on CPU against its recorded
GPU rows gives per-corpus gaps of 4e-6 to 7.1e-4 (PanoSaliency 0.7315 vs 0.7308) with no
code change, because cuDNN's BiLSTM arithmetic differs from CPU float32 at that level. So
a `mode=test` figure produced on the other device from the training run is not a
reproduction check, and any before/after acceptance for a numerics-touching change must
be same-device, same script, same checkpoint. Differences below ~1e-3 between devices
are arithmetic, not results.

## Performance notes

Keep `num_workers: 0` unless benchmarked: the whole sample tensor lives in memory inside the Dataset, and Windows spawn-based workers pickle it per worker.

Console output must stay ASCII. Windows consoles default to cp1252, so box-drawing characters raise `UnicodeEncodeError` as soon as stdout is piped or redirected — this crashed `mode=test` until it was fixed in `eval.py`.
