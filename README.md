# XRSec

XR biometric identification research codebase built around a Siamese network for headset-motion matching.

## Overview

The current pipeline trains a Siamese model on fixed-size headset motion windows:

- Input channels: `qx, qy, qz, qw, Hx, Hy, Hz`
- Sequence layout: `(channels=7, timesteps=sample_time * sample_rate)`
- Backbone: graph aggregation + BiLSTM + self-attention
- Objective: same-user / different-user prediction via distance logits

The window-to-embedding backbone is a **slottable feature extractor**, so architectures
can be written independently and compared on identical data, splits and seeds.

The training system supports:

- Standard single-run Siamese training
- Deterministic boosted training with regenerated Siamese pair datasets across rounds
- Sweep mode: many extractors and hyperparameter combinations in one command, ranked

## Project Layout

```text
XRSec/
├── configs/
│   └── config.yaml
├── docs/
│   └── ML_PROJECT_STANDARD.md
├── model/
│   ├── main.py              # Hydra entry point (mode=train|test|sweep)
│   ├── train.py             # standard training + shared round primitives
│   ├── boost_train.py       # boosted round orchestration
│   ├── sweep.py             # sweep mode
│   ├── eval.py
│   ├── dataset.py           # sample index + pair manifests
│   ├── sample_cache.py      # on-disk cache of sampled windows
│   ├── results_log.py       # one CSV row per run
│   ├── model.py             # Siamese head + model factory
│   ├── feature_extractor.py # extractor base class + registry
│   ├── extractors/          # extractor implementations (auto-discovered)
│   └── utils.py
├── processed_datasets/
├── results/                 # runs.csv
├── sweeps/                  # sweep state and summaries
├── runs/
└── tests/
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training Modes

### Standard training

Standard mode builds one deterministic Siamese training dataset from the configured users and trains a single model checkpoint.

```powershell
python model/main.py mode=train boosting.enabled=false
```

### Boosted training

Boosted mode trains in rounds. Each round:

- regenerates Siamese training pairs deterministically from the root `seed`
- keeps the hardest pairs from the previous round
- fills the remainder with refreshed pairs
- warm-starts from the previous round's best checkpoint
- evaluates every round against a fixed validation manifest

```powershell
python model/main.py mode=train boosting.enabled=true boosting.rounds=5 boosting.round_epochs=10
```

## Hydra Usage

### Train on one or many datasets

```powershell
python model/main.py mode=train `
  data_dirs=[processed_datasets/A/users,processed_datasets/B/users] `
  batch_size=1024 lr=1e-3 embedding_dim=128
```

### Test on one or many datasets

```powershell
python model/main.py mode=test `
  test_dirs=[processed_datasets/C/users,processed_datasets/D/users] `
  batch_size=1024 model_path=checkpoints/xrsec_multi2_2s_20hz_emb128_train.pth
```

If `test_dirs=[]`, evaluation falls back to `data_dirs`.

## Deterministic Pair Generation

Siamese training pairs are no longer treated as persistent datasets. Instead:

- sample discovery is stable because user folders and CSVs are traversed in sorted order
- pair manifests are regenerated from the configured `seed`
- standard training, validation splits, DataLoader shuffling, and boosted rounds all derive their randomness from that root seed

This keeps training reproducible without storing raw pair tensors on disk.

## Boosting Configuration

Boosting lives under `boosting` in `configs/config.yaml`:

- `enabled`: enable boosted round training
- `rounds`: total number of boosting rounds
- `round_epochs`: epochs per round
- `hard_fraction`: fraction of each round kept from hard examples
- `refresh_fraction`: must equal `1 - hard_fraction`
- `candidate_pairs_per_user`: candidate pool size used when mining hard pairs
- `match_ratio`: target positive/negative pair ratio
- `artifact_root`: where boosted round checkpoints and state are written
- `resume`: resume from existing boosted state instead of starting fresh

## Outputs

With `save_path=auto`, `model_path=auto`, and `graph_path=auto`:

- Checkpoint: `checkpoints/{experiment}_{datasetTag}_{sample}s_{rate}hz_emb{dim}_{mode}.pth`
- Plot: `plots/{experiment}_{datasetTag}_{sample}s_{rate}hz_emb{dim}_{mode}.png`

Hydra writes each run under `runs/YYYY-MM-DD/HH-MM-SS_<mode>/`.

### Standard training outputs

- final best checkpoint at `save_path`
- optional training graph at `graph_path`

### Boosted training outputs

- final best overall checkpoint copied to `save_path`
- round checkpoints under `{boosting.artifact_root}/rounds/`
  - `round_000_best.pth`
  - `round_000_last.pth`
  - etc.
- compact run state at `{boosting.artifact_root}/boost_state.json`
- optional boosted summary plot at `graph_path`
- optional per-round plots in a sibling directory such as `plots/<stem>_rounds/`

## Plotting

When `graph: true`:

- standard runs save a single training-history plot
- boosted runs save:
  - one summary plot across rounds
  - one training-history plot per round

Plotting uses a headless matplotlib backend, so it works in terminal-only environments and tests.

## Feature Extractors

The window-to-embedding backbone is slottable. Each extractor implements
`(batch, 7, seq_len) -> (batch, embedding_dim)` and nothing else in the pipeline
changes when you swap one.

```powershell
python model/main.py mode=train extractor=bilstm
python model/main.py mode=train extractor=bilstm extractor_params={lstm_hidden:128,pooling:last}
```

List what is registered, with defaults and declared sweep spaces:

```powershell
.venv\Scripts\python model/list_extractors.py
```

### Adding one

Drop a module into `model/extractors/`; it is discovered automatically.

```python
from feature_extractor import FeatureExtractor, register

@register("my_extractor")
class MyExtractor(FeatureExtractor):
    def __init__(self, seq_len, num_channels=7, embedding_dim=128, width=64):
        super().__init__(seq_len=seq_len, num_channels=num_channels,
                         embedding_dim=embedding_dim, width=width)
        ...

    def forward(self, x):        # x: (batch, num_channels, seq_len)
        ...                      # return: (batch, embedding_dim)

    @classmethod
    def search_space(cls):
        return {"width": [32, 64, 128]}
```

Hyperparameters must be explicit keyword arguments with defaults, and every key in
`search_space()` must be one of them. If `forward` is stochastic in eval mode, set
`deterministic = False` on the class.

Running `pytest` then validates the new extractor automatically: output contract,
varied sequence lengths and embedding widths, every declared sweep value,
trainability, determinism, and checkpoint round-trip.

### Built-in extractors

| Name | What it is |
| --- | --- |
| `paper_gnn_bilstm` | The published GNN + BiLSTM + attention architecture (default) |
| `bilstm` | The same without the GNN branches - isolates what the graph layers add |
| `random` | Ignores its input and emits noise - the chance-level floor (~50%) |

## Channel Sets

```powershell
python model/main.py mode=train channels=position extractor=bilstm
```

- `full` (default) - quaternion + position, 7 channels
- `position` - position only, 3 channels

`position` unlocks 2814 extra sessions (48% more data): it doubles Head_and_Gaze
(28,661 -> 57,344 windows) and recovers `360_em_dataset`, which records no
orientation at all. Orientation is also a weak identity cue on its own (0.529 AUC on
held-out users vs 0.768 for mean position), so this may cost little - but it is an
experiment, so `full` stays the default.

Not every extractor accepts 3 channels: `bilstm` and `random` do; `motion_tdnn` and
`paper_gnn_bilstm` assume the 7-channel layout and will report that clearly rather
than crashing.

## Training Objectives

```powershell
python model/main.py mode=train objective=identity_softmax extractor=motion_tdnn
```

- `pair_bce` (default) trains BCE over a linear layer on `|e1 - e2|`.
- `identity_softmax` classifies which user each window belongs to with an additive
  angular margin, then compares embeddings by cosine similarity. It uses every window
  as a training example rather than every pair, and learns no per-dimension weights
  tied to the training identities - the standard approach in speaker and face
  verification, for the same generalisation reason.

`identity_softmax` forces `head=cosine`, discards its classifier after training, and
recalibrates the cosine threshold on training pairs each epoch so accuracy stays
meaningful. `head` (`diff_linear` | `cosine`) is also selectable on its own.

## Sweep Mode

Train many configurations in one command and rank them.

```powershell
python model/main.py mode=sweep
```

By default this sweeps the configured extractor over its own declared
`search_space()`. Preview the plan before committing GPU time:

```powershell
python model/main.py mode=sweep sweep.dry_run=true
```

Compare every registered extractor at matched settings:

```powershell
python model/main.py mode=sweep sweep.extractors=all sweep.grid={lr:[0.001,0.0003]}
```

### Axes

`sweep.grid` keys are namespaced. Prefix with `extractor_params.` to vary an
extractor hyperparameter; any other key varies a top-level config value.

```yaml
sweep:
  grid:
    lr: [0.001, 0.0003]
    embedding_dim: [64, 128]
    extractor_params.lstm_hidden: [32, 64, 128]
```

`grid: auto` (the default) uses each extractor's declared `search_space()`.

### Cross-validation

Which users are held out matters more than anything else measured on this data:
swapping the held-out group moves a training-free position probe from 0.631 to 0.746
(sd 0.037), against a +/-0.019 binomial bar on 2560 pairs. A single fixed split
cannot separate configurations that differ by a few points.

```powershell
python model/main.py mode=sweep sweep.folds=5
```

Each configuration is trained on K disjoint held-out user groups and ranked by the
mean, with the spread reported. `sweep.folds` ignores `exclude_users` and partitions
every user across `data_dirs`. The ranking warns when the top two configurations
differ by less than the fold spread.

### Behaviour

- **Failures are isolated.** A configuration that raises is recorded with its error
  and the sweep continues, so one bad generated extractor cannot cost the whole run.
- **Resume is on by default**, keyed by a digest of each configuration. Rerunning the
  same sweep skips completed configurations and retries failed ones.
- **Every configuration is appended to `results/runs.csv`**, tagged with `sweep_id`,
  so sweep and non-sweep results stay comparable in one table.
- `sweep.epochs` shortens each sweep run without touching the top-level `epochs`.
- With `sweep.folds`, resume works per fold, and a configuration still reports a mean
  from whichever folds succeeded if one of them fails.
- `sweep.strategy=random` with `sweep.max_runs=N` takes an unbiased subset;
  capping a `grid` sweep just truncates it in order.

Each sweep writes `{sweep.artifact_root}/{sweep_id}/` containing `sweep_state.json`,
`summary.csv`, and per-configuration checkpoints under `runs/`.

## Input Standardisation

Datasets do not share a coordinate frame - mean head height spans 0.00003 to 2.89
across the corpus and position range spans 40x. Pooling them naively lets a model
identify the *dataset* rather than the *user*, because a positive pair is always the
same user and therefore always the same dataset.

```yaml
normalize: per_dataset          # per_dataset | global | none
within_dataset_negatives: true  # negatives only from users in the same dataset
```

- `normalize` standardises each dataset's channels separately. Statistics are fitted
  on training users only and stored in the checkpoint, so `mode=test` reuses the
  training-time transform rather than deriving one from held-out data.
- `within_dataset_negatives` removes the remaining cross-dataset cue from the pair
  task. Both are no-ops when training on a single dataset.

Measured across six datasets (238 identities, same 5 held-out users):

| configuration | held-out accuracy |
| --- | --- |
| raw | 0.576 |
| + per-dataset standardisation | 0.643 |
| + within-dataset negatives | 0.687 |

## Dataset Health

`UserProfile` skips files that are unusable (missing required columns, under two
rows, non-finite, or zero duration) and reports the counts, instead of letting one
bad file take down a whole dataset.

Native sampling rates vary widely, and requesting a `sample_rate` above a dataset's
native rate duplicates frames rather than adding information:

| dataset | users | native Hz |
| --- | --- | --- |
| Head_and_Gaze | 100 | 120 (half the files carry no quaternion and are skipped) |
| PanoSaliency | 99 | 16.5 |
| VR_User_Behavior | 48 | 89.5 |
| ViewGauss | 35 | 10.1 |
| EyeNavGS | 22 | 125 |
| Panonut360 | 21 | 94 |
| NJIT_6DOF | 18 | 250 |

At `sample_rate=20`, ViewGauss is 50.5% duplicated frames and PanoSaliency 25.9%.

## Cross-Session Verification

```powershell
python model/main.py mode=train cross_session_positives=true
```

A positive pair is normally two windows from the same user, which usually means the
same recording session - sharing headset mounting, seating position and viewed
content. A model can score by matching the session rather than the person, and since
held-out positives are same-session too, that shortcut never shows as a train/test
gap. Cross-session evaluation is the standard requirement in biometrics.

Users with only one session fall back to same-session pairs and are reported.

## Reporting an Honest Number

`best_test_acc` is a max over every epoch of the set it reports, which inflates it by
about +0.02 - a max over ~20 noisy evaluations. A random-output extractor scores
0.4973 at its final epoch but 0.5173 as a best-of.

```powershell
python model/main.py mode=train val_user_fraction=0.25
```

This holds out a group of training users - disjoint from both training and the
reported test users - and chooses the epoch on them. Then report `selected_test_acc`.
Verified with the random extractor over 3 seeds: max-over-epochs averaged 0.525 while
the validation-selected figure averaged 0.502, i.e. chance.

Default is 0, which keeps the historical behaviour so older numbers stay comparable.

## Results Log

Every run appends one row to `results/runs.csv`: config, metrics, checkpoint, run
directory and git SHA, for standard, boosted and test runs alike. Sort that file
instead of reopening checkpoints to compare experiments.

## Sample Cache

Sampled windows are cached per user directory under `.cache/samples/`, keyed by CSV
content signature plus `sample_time`/`sample_rate`. This cuts dataset loading on the
default dataset from ~23s to ~0.6s, and applies to both the train and validation
index builds.

- `XRSEC_SAMPLE_CACHE=0` disables it
- `XRSEC_SAMPLE_CACHE_DIR=...` relocates it
- Deleting `.cache/` is always safe; entries rebuild on demand

## Testing

Run the full test suite with:

```powershell
.venv\Scripts\python -m pytest -q
```

The test suite covers deterministic pair generation, hard-pair selection, boosted round orchestration, plotting, sample caching, and the standard training path.

## More Detail

See `docs/ML_PROJECT_STANDARD.md` for the current project standard, what is already implemented, and the remaining scale-up work.
