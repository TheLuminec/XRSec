# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

XR biometric identification research. A Siamese network decides whether two windows of headset motion came from the same person. The research question is whether this generalizes to **users never seen during training**, so nearly every design decision (leave-users-out splits, pair generation, boosting) exists to serve that question.

Current state: on held-out users the model plateaus around **0.55–0.63 accuracy on a balanced binary task** (chance = 0.50), while training accuracy reaches ~0.75. The core result is not yet working — treat infrastructure changes as being in service of diagnosing that, not as ends in themselves.

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

Also note there is no third split: per-epoch best-checkpoint selection and boosted best-round selection both use the same held-out set that gets reported, so reported accuracy is optimistically biased.

### The evaluation is noisier than it looks

**Measured:** holding out a different random 5 users moves a training-free position probe from 0.631 to 0.746 — a **0.114 spread, sd 0.037** — while the binomial error bar on 2560 pairs is only ±0.019. The effective sample size is the number of held-out *users*, not pairs.

Two consequences that should govern how any result here is read:

1. **Differences below ~0.04 on a single split are not real.** An 8-configuration regularization sweep produced a range of 0.682–0.692; that is entirely inside the noise and separates nothing.
2. **The project's fixed split (users 1–5) is unusually easy**: 0.754 on the same probe versus 0.686 for the average random split. Numbers from it are optimistic, and an unreproducible historical high could partly be a lucky split.

Use `sweep.folds: K` for anything you intend to act on. It ignores `exclude_users`, partitions every user across `data_dirs` into K disjoint held-out groups, runs each configuration on all of them, and ranks by the mean while reporting the spread. It prints an explicit warning when the top two configurations differ by less than the fold standard deviation.

## Data

Model input lives in `processed_datasets/<Dataset_Name>/users/<user_id>/<task>.csv`, gitignored, with required columns `SessionTime`, `UnitQuaternion.{x,y,z,w}`, `HmdPosition.{x,y,z}` at ≥10Hz.

**Which datasets exist is per-machine and must be checked, not assumed** — `processed_datasets/` is ~6.9GB when fully populated and cannot travel through git, so two checkouts of this repo routinely hold different data. Confirm before planning a run:

```bash
for d in processed_datasets/*/users; do echo "$(ls "$d" 2>/dev/null | wc -l) users  $d"; done
```

The table below describes the corpus when fully populated. `normalize=per_dataset` and `within_dataset_negatives` are **no-ops on a single dataset**, so a machine holding only one of these cannot reproduce any multi-dataset result.

Getting a *new* dataset to that layout is currently the weakest link:

- `formatter.py` expects `datasets/<name>/parser.py` exposing `parse(dataset_path)` yielding `(user_id, task_id, df)`, and writes to `datasets/<name>/processed_data/users/`.
- That directory does not exist on `main` and neither does any parser — the eight working parsers were removed in commit `6421567` ("Data seperation") and survive only in git history (`git show normalization:datasets/<name>/parser.py`; the `normalization` branch is an ancestor of `main`, not pending work).
- `formatter.py`'s output path (`datasets/<name>/processed_data/users/`) does not match where the model reads from (`processed_datasets/<name>/users/`), so onboarding a dataset ends with a manual move.

### Data condition (audited)

| dataset | users | native Hz | notes |
| --- | --- | --- | --- |
| Head_and_Gaze | 100 | 120 | **half the files (2630 `V1_*`) have no quaternion** — gaze rays only |
| PanoSaliency | 99 | 16.5 | 25 single-row sessions (zero duration); below 20Hz |
| VR_User_Behavior | 48 | 89.5 | the default dataset |
| ViewGauss | 35 | 10.1 | well below 20Hz |
| EyeNavGS | 22 | 125 | |
| Panonut360 | 21 | 94 | |
| NJIT_6DOF | 18 | 250 | room-scale walking, position range 5.13m |

Quaternions are unit-norm everywhere and there are no non-finite values. `UserProfile` skips files that are missing required columns, have fewer than two rows, are non-finite, or have non-positive duration, and reports the counts — before this, one bad file raised `KeyError` and took down a whole dataset (which is what made Head_and_Gaze unusable).

**Requesting a `sample_rate` above a dataset's native rate duplicates frames**, because `Sampler` takes the nearest point to each target time. At 20Hz this is 50.5% duplicate consecutive frames for ViewGauss and 25.9% for PanoSaliency, against 7.1% for VR_User_Behavior — so derived velocity is partly fictitious for the low-rate datasets. Check the table above before raising `sample_rate`.

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

## Verification metrics

`model/metrics.py`. Accuracy is measured at the fixed `logit > 0` threshold, which conflates ranking quality with operating-point placement — a model can sit at 0.50 accuracy while still ranking pairs usefully. `evaluate(..., return_metrics=True)` adds:

- **ROC-AUC** — threshold-free ranking quality. Ties are rank-averaged, which matters because an untrained model emits a near-constant logit and naive AUC would report 0.0 or 1.0 depending on sort order.
- **EER** and its threshold — the standard biometric verification number, comparable across datasets with different pair balance.

Both are tracked per epoch into `history` (`test_auc`, `test_eer`) and recorded in `results/runs.csv` as `best_test_auc` / `best_test_eer`.

## Results log

`model/results_log.py` appends one row per run to `results/runs.csv` (absolute path, anchored to the repo root so `job.chdir` can't misplace it). It covers all three paths — standard, boosted, and test — and records config (including `extractor` and `extractor_params`), metrics, checkpoint, run dir and git SHA (with a `-dirty` suffix for uncommitted trees). Changing `FIELDS` is safe: the file is migrated in place, existing rows are backfilled with blanks, and columns dropped from `FIELDS` are retained rather than deleted. Logging failures degrade to a warning and never abort a finished run. Add new columns to the end of `FIELDS` so existing files stay readable.

The 95 pre-existing runs under `runs/` are not in this file; they can be backfilled from checkpoint `history` dicts plus each run's `.hydra/config.yaml`.

## Sample cache

`model/sample_cache.py` caches each user directory's sampled windows to `.cache/samples/` (gitignored), keyed by CSV names/sizes/mtimes plus `sample_time`/`sample_rate`. Measured on the default 48-user dataset: **23.4s → 0.6s**, bit-identical output. Because both the train and eval index builds hit the same per-user entries, the double-load is now nearly free, and changing `exclude_users` invalidates nothing.

- Disable with `XRSEC_SAMPLE_CACHE=0`; relocate with `XRSEC_SAMPLE_CACHE_DIR`.
- Deleting `.cache/` is always safe. Entries for superseded signatures are never garbage-collected, so it grows across resolutions (~100MB for two resolutions of one dataset).
- The cache is only valid because sampling is deterministic (`Sampler` is always built with `index_randomness=0`). **If per-epoch index jitter is ever enabled, the cache must be bypassed** or it will freeze one fixed augmentation.

## Known-broken

- `model/validate.py` is dead: it imports `plot_training_history` from `train` (it lives in `utils`), calls `train()` with a dict shape that predates the current config, and assumes the old `datasets/*/processed_data/` layout.

Current baseline: **136 passing, ~17s**.

## Performance notes

Keep `num_workers: 0` unless benchmarked: the whole sample tensor lives in memory inside the Dataset, and Windows spawn-based workers pickle it per worker.

Console output must stay ASCII. Windows consoles default to cp1252, so box-drawing characters raise `UnicodeEncodeError` as soon as stdout is piped or redirected — this crashed `mode=test` until it was fixed in `eval.py`.
