# ML Project Standard for Scaling Experiments (PyTorch)

This repository has a strong modeling core, but scaling up training/data now needs a reproducible experiment workflow.

## 1) Recommended project standard

Use this structure and process as the baseline standard:

- **Single source of truth for config** (YAML + CLI override).
- **Deterministic run metadata** saved per run (seed, git SHA, dataset IDs, model args).
- **Experiment tracker** (MLflow or Weights & Biases).
- **Immutable dataset splits** (versioned train/val/test manifests).
- **Checkpoint policy** (best metric + last + periodic).
- **Evaluation protocol** with fixed metrics and threshold calibration.

A practical stack for this codebase:

- Configs: `OmegaConf`/`Hydra`.
- Tracking: `MLflow` (local first) or `wandb`.
- Metrics: `torchmetrics` for consistent metric computation.

## 2) Immediate gaps found in current codebase

- Training argument wiring had positional mismatches in `model/train.py`.
- Test pipeline expected a device argument while CLI entrypoint did not pass one.
- Reproducibility metadata (seed + run config + code version) is not persisted with checkpoints.
- The repository root `README.md` is currently notes-only and not an operational runbook.

## 3) Minimum operating standard (what to implement next)

1. **Config files first**
   - Add `configs/train.yaml`, `configs/test.yaml`.
   - Keep CLI only for overrides.

2. **Structured outputs**
   - Save each run under `runs/<timestamp>_<short_sha>/`:
     - `config.yaml`
     - `metrics.csv`
     - `best.ckpt`, `last.ckpt`
     - `environment.txt` (PyTorch/CUDA versions)

3. **Versioned splits**
   - Persist split manifests (`train_users.txt`, `val_users.txt`, `test_users.txt`) and reuse across runs.

4. **Evaluation contract**
   - Report at minimum: ROC-AUC, EER, FAR@FRR target, confusion matrix, threshold used.

5. **Hyperparameter sweeps**
   - Start with grid/random search for:
     - `embedding_dim`
     - `lr`
     - `batch_size`
     - siamese pair sampling ratio
   - Record all sweep runs in tracker.

## 4) Scale-up checklist

Before large runs:

- Confirm dataloader throughput (`num_workers`, `pin_memory`, prefetch).
- Validate no train/test leakage by user/session identity.
- Freeze a baseline run and tag it.
- Define stop criteria and early stopping.

## 5) Suggested milestone plan

- **Milestone A (1–2 days):** stable CLI/config + fixed wiring + deterministic seeds + better README.
- **Milestone B (2–4 days):** experiment tracking + split manifests + richer metrics.
- **Milestone C (ongoing):** automated sweeps + ablation table + weekly model card updates.


## Milestone A status (implemented)

- Hydra config scaffolding added under `configs/` with mode-specific config group (`mode/train.yaml`, `mode/test.yaml`).
- Main entrypoint migrated to Hydra (`model/main.py`) and now normalizes paths for Hydra run directories.
- Timestamped run directories enabled in `runs/YYYY-MM-DD/HH-MM-SS_<mode>/`.


## Milestone A next actions (future-proofing)

- Introduce `split_manifests/` and always train/test from manifest files to eliminate user/session leakage risk.
- Save run metadata beside checkpoints (git commit, config snapshot, dataset list hash, seed, package versions).
- Add a strict run-name registry (`runs/index.csv`) to prevent accidental overwrite and simplify comparison tables.
- Add a compatibility layer for old `data_dir`/`test_dir` keys while migrating all scripts to `data_dirs`/`test_dirs`.

