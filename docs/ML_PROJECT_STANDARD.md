# ML Project Standard for Scaling Experiments (PyTorch)

This repository now has a reproducible training core for both standard Siamese training and deterministic boosted round training. The next step is to treat that workflow as the baseline operating standard for future experiments.

## 1) Current implemented standard

The codebase now has these foundations in place:

- **Single source of truth for config** through `configs/config.yaml` with Hydra overrides.
- **Deterministic training seeds** across dataset generation, validation splits, DataLoader shuffling, and boosted rounds.
- **Three supported modes**:
  - standard single-pass Siamese training
  - boosted hard-round training with regenerated pair manifests
  - sweep over extractors and hyperparameters (`mode=sweep`)
- **Checkpoint policy**:
  - standard mode saves the best checkpoint
  - boosted mode saves `best` and `last` checkpoints for every round plus the best overall model
- **Compact boosted state tracking** through `boost_state.json`
- **Training plots** for both standard runs and boosted round summaries
- **Slottable feature extractors** through a registry, so architectures are
  interchangeable and comparable on identical data, splits and seeds
- **Sweep mode** (`mode=sweep`) for ranking extractors and hyperparameter
  combinations in one command, with resume and per-configuration failure isolation
- **A single results table** at `results/runs.csv`, one row per run across standard,
  boosted, test and sweep runs
- **A sample cache** that removes CSV parsing from repeat runs (~23s to ~0.6s on the
  default dataset)
- **Per-dataset input standardisation** with training-only statistics carried in the
  checkpoint, plus within-dataset negative sampling, so several datasets can be
  pooled without the model learning to identify the dataset instead of the user
- **Tolerant data loading** that skips unusable files and reports them, rather than
  failing a whole dataset on one malformed file

## 2) What the boosted workflow standardizes

Boosted training is now the preferred structured workflow when we want iterative retraining on regenerated Siamese pairs without persisting raw training pairs.

The current contract is:

- A root `seed` drives all stochastic decisions.
- User/sample discovery is stable because filesystem traversal is sorted.
- Siamese pair manifests are regenerated on demand instead of stored as tracked datasets.
- Validation pairs are fixed for the full boosted run.
- Each round warm-starts from the previous round's best checkpoint.
- Hard examples are selected from a deterministic candidate pool.
- The rest of the round is refreshed with newly generated pairs.

This gives us reproducibility without paying the storage/debugging cost of saving raw pair tensors for every round.

## 3) Current artifact standard

### Standard training

Expected artifacts:

- final checkpoint at `save_path`
- optional training graph at `graph_path`
- Hydra run directory under `runs/YYYY-MM-DD/HH-MM-SS_<mode>/`

### Boosted training

Expected artifacts:

- final best-overall checkpoint copied to `save_path`
- round checkpoints under `{boosting.artifact_root}/rounds/`
  - `round_000_best.pth`
  - `round_000_last.pth`
  - etc.
- run state under `{boosting.artifact_root}/boost_state.json`
- optional boosted summary graph at `graph_path`
- optional per-round graphs in a sibling directory such as `plots/<stem>_rounds/`

### Sweeps

Expected artifacts under `{sweep.artifact_root}/{sweep_id}/`:

- `sweep_state.json` (resume state, keyed by configuration digest)
- `summary.csv` (ranked results, including failures and their errors)
- `runs/<extractor>_<config_id>/best.pth` per configuration
- one row per configuration in `results/runs.csv`, tagged with `sweep_id`

### Checkpoint metadata

Checkpoints can now carry:

- `checkpoint_kind`
- `round_idx`
- `history`
- `warm_start_from`
- `extractor`, `extractor_params` and `num_channels`, so any checkpoint rebuilds its
  own backbone without the config that produced it
- seed/config metadata added by the training path

## 4) Remaining gaps

Several earlier gaps are now closed: run metadata (git SHA, full config, extractor
identity) is recorded per run in `results/runs.csv`, and sweeps are a first-class mode
rather than manual config editing. What remains:

- **Validation/evaluation metrics are still minimal**
  - current evaluation is loss + accuracy only; no ROC-AUC, EER or confusion matrix
- **No third split**
  - per-epoch checkpoint selection, boosted best-round selection and the reported
    number all use the same held-out users, so reported accuracy is optimistically biased
- ~~No input normalization~~ - done. `normalize: per_dataset` standardises each
  dataset's channels (fitted on training users, stored in the checkpoint), and
  `within_dataset_negatives: true` stops the pair task degrading into "same
  dataset?". Together these took six-dataset training from 0.576 to 0.687 held-out.
- **Package versions are not captured**
  - the git SHA and config are recorded, but not the environment
- **No external experiment tracker**
  - `results/runs.csv` covers comparison; MLflow or W&B would add run management
- **No split-manifest registry yet**
  - boosted pair regeneration is deterministic, but we still do not persist user/session split manifests as first-class experiment assets
- **No formal threshold calibration**
  - the current same/different decision still uses a fixed threshold heuristic

## 5) Updated minimum operating standard

For new experiments in this repository, the minimum acceptable bar should now be:

1. Use Hydra config plus CLI overrides, not ad hoc script edits.
   - For architecture comparisons use `mode=sweep`, not repeated manual runs.
2. Set and record a root `seed`.
3. Choose explicitly between standard and boosted training.
4. Keep `graph: true` for any meaningful training run so loss/accuracy history is preserved.
5. When training on more than one dataset, keep `normalize: per_dataset` and
   `within_dataset_negatives: true`, and check each dataset's native sampling rate
   before raising `sample_rate` - requesting more than the native rate duplicates
   frames instead of adding information.
6. Preserve boosted artifacts when using round training:
   - `boost_state.json`
   - round checkpoints
   - summary and round plots
7. Avoid persisting regenerated pair tensors unless a future debugging need proves that necessary.

## 6) Suggested next milestones

### Milestone B

- Save a config snapshot beside every training run.
- Save environment metadata beside every run.
- Add richer evaluation metrics:
  - ROC-AUC
  - EER
  - confusion matrix
  - threshold used

### Milestone C

- Introduce explicit train/val/test split manifests at the user/session level.
- Add experiment tracking.
- ~~Add sweep support~~ — done; `mode=sweep` covers any top-level config key
  (`embedding_dim`, `lr`, `batch_size`, `samples_per_user`, `boosting.*`) plus
  extractor hyperparameters via `extractor_params.<name>`.

### Milestone D

- Add model cards / experiment summaries.
- Add comparison tooling across standard vs boosted runs.
- Add resume and recovery documentation for long-running boosted jobs.
