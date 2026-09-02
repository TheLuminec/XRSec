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
`search_space()` must be one of them. Running `pytest` then validates the new
extractor automatically: output contract, varied sequence lengths and embedding
widths, every declared sweep value, trainability, and checkpoint round-trip.

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

### Behaviour

- **Failures are isolated.** A configuration that raises is recorded with its error
  and the sweep continues, so one bad generated extractor cannot cost the whole run.
- **Resume is on by default**, keyed by a digest of each configuration. Rerunning the
  same sweep skips completed configurations and retries failed ones.
- **Every configuration is appended to `results/runs.csv`**, tagged with `sweep_id`,
  so sweep and non-sweep results stay comparable in one table.
- `sweep.epochs` shortens each sweep run without touching the top-level `epochs`.
- `sweep.strategy=random` with `sweep.max_runs=N` takes an unbiased subset;
  capping a `grid` sweep just truncates it in order.

Each sweep writes `{sweep.artifact_root}/{sweep_id}/` containing `sweep_state.json`,
`summary.csv`, and per-configuration checkpoints under `runs/`.

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
