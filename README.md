# XRSec

XR biometric identification research codebase.

## Milestone A layout (Hydra + clearer structure)

```text
XRSec/
├── configs/
│   ├── config.yaml
│   └── mode/
│       ├── train.yaml
│       └── test.yaml
├── model/
│   ├── main.py
│   ├── train.py
│   ├── test.py
│   └── ...
├── datasets/
├── runs/                    # Hydra run outputs (timestamped)
└── saved_tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Current model

The current pipeline trains a Siamese network on headset motion windows:

- Input channels: `qx, qy, qz, qw, Hx, Hy, Hz`
- Sequence layout: `(channels=7, timesteps=sample_time * sample_rate)`
- Backbone: graph aggregation + BiLSTM + self-attention
- Objective: same-user / different-user prediction via distance logit

## Hydra-based usage

### Train on one or many datasets

```bash
python model/main.py mode=train \
  data_dirs=[datasets/A/processed_data/users,datasets/B/processed_data/users] \
  epochs=20 batch_size=1024 lr=1e-3 embedding_dim=128
```

### Test on one or many datasets

```bash
python model/main.py mode=test \
  test_dirs=[datasets/C/processed_data/users,datasets/D/processed_data/users] \
  batch_size=1024 model_path=checkpoints/xrsec_multi2_1s_10hz_emb128_train.pth
```

If `test_dirs=[]`, testing falls back to `data_dirs`.

## Standardized naming conventions

With `save_path=auto`, `model_path=auto`, and `graph_path=auto`:

- Checkpoint: `checkpoints/{experiment}_{datasetTag}_{sample}s_{rate}hz_emb{dim}_{mode}.pth`
- Plot: `plots/{experiment}_{datasetTag}_{sample}s_{rate}hz_emb{dim}_{mode}.png`

Where:
- `experiment` is from `experiment_name` in config.
- `datasetTag` is either a dataset slug for single dataset or `multiN` for N datasets.

Hydra writes each run to `runs/YYYY-MM-DD/HH-MM-SS_<mode>/`.

## Recommended next steps for scaling

See `docs/ML_PROJECT_STANDARD.md` for the project standard and near-term plan.
