# ML Project Standard for Scaling Experiments (PyTorch)

This repository now has a reproducible training core for both standard Siamese training and deterministic boosted round training. The next step is to treat that workflow as the baseline operating standard for future experiments.

## 1) Current implemented standard

The codebase now has these foundations in place:

- **Single source of truth for config** through `configs/config.yaml` with Hydra overrides.
- **Deterministic training seeds** across dataset generation, validation splits, DataLoader shuffling, and boosted rounds.
- **Two supported training modes**:
  - standard single-pass Siamese training
  - boosted hard-round training with regenerated pair manifests
- **Checkpoint policy**:
  - standard mode saves the best checkpoint
  - boosted mode saves `best` and `last` checkpoints for every round plus the best overall model
- **Compact boosted state tracking** through `boost_state.json`
- **Training plots** for both standard runs and boosted round summaries

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

### Checkpoint metadata

Checkpoints can now carry:

- `checkpoint_kind`
- `round_idx`
- `history`
- `warm_start_from`
- seed/config metadata added by the training path

## 4) Remaining gaps

The project is in much better shape than before, but there are still important gaps before this becomes a full experiment platform:

- **Run metadata is still incomplete**
  - we are not yet saving git SHA, package versions, or a frozen config snapshot beside every run
- **Validation/evaluation metrics are still minimal**
  - current evaluation is loss + accuracy only
- **No external experiment tracker yet**
  - MLflow or Weights & Biases would make comparison and sweep management much easier
- **No split-manifest registry yet**
  - boosted pair regeneration is deterministic, but we still do not persist user/session split manifests as first-class experiment assets
- **No formal threshold calibration**
  - the current same/different decision still uses a fixed threshold heuristic

## 5) Updated minimum operating standard

For new experiments in this repository, the minimum acceptable bar should now be:

1. Use Hydra config plus CLI overrides, not ad hoc script edits.
2. Set and record a root `seed`.
3. Choose explicitly between standard and boosted training.
4. Keep `graph: true` for any meaningful training run so loss/accuracy history is preserved.
5. Preserve boosted artifacts when using round training:
   - `boost_state.json`
   - round checkpoints
   - summary and round plots
6. Avoid persisting regenerated pair tensors unless a future debugging need proves that necessary.

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
- Add sweep support for:
  - `embedding_dim`
  - `lr`
  - `batch_size`
  - `samples_per_user`
  - `boosting.hard_fraction`
  - `boosting.candidate_pairs_per_user`
  - `boosting.match_ratio`

### Milestone D

- Add model cards / experiment summaries.
- Add comparison tooling across standard vs boosted runs.
- Add resume and recovery documentation for long-running boosted jobs.
