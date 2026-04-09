# XRSec

XR biometric identification research codebase built around a Siamese network for headset-motion matching.

## Overview

The current pipeline trains a Siamese model on fixed-size headset motion windows:

- Input channels: `qx, qy, qz, qw, Hx, Hy, Hz`
- Sequence layout: `(channels=7, timesteps=sample_time * sample_rate)`
- Backbone: graph aggregation + BiLSTM + self-attention
- Objective: same-user / different-user prediction via distance logits

The training system now supports both:

- Standard single-run Siamese training
- Deterministic boosted training with regenerated Siamese pair datasets across rounds

## Project Layout

```text
XRSec/
├── configs/
│   └── config.yaml
├── docs/
│   └── ML_PROJECT_STANDARD.md
├── model/
│   ├── main.py
│   ├── train.py
│   ├── eval.py
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
├── processed_datasets/
├── plots/
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

## Testing

Run the full test suite with:

```powershell
.venv\Scripts\python -m pytest -q
```

The test suite covers deterministic pair generation, hard-pair selection, boosted round orchestration, plotting, and the standard training path.

## More Detail

See `docs/ML_PROJECT_STANDARD.md` for the current project standard, what is already implemented, and the remaining scale-up work.
