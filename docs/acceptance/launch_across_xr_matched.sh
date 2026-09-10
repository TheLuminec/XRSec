#!/usr/bin/env bash
# The MATCHED arms of docs/acceptance/across_xr_alignment_REGISTERED.md, one seed each.
#
#   launch_across_xr_matched.sh C1 <seed>    Across-XR users 0-22 alone in training
#   launch_across_xr_matched.sh C2 <seed>    BOXRR + alyx + Across-XR users 0-22
#
# Approved by the Coordinator on 2026-09-10 as a SEPARATELY LABELLED arm: training may use
# Across-XR users 0-22 only, epoch selection uses 23-31 only (explicit validation_users;
# that corpus is then left out of the 25% draw, which still applies to BOXRR and alyx in
# C2), and users 32-48 are never trained on, validated on, or used to choose anything.
# The two arms are never averaged with the zero-shot arm and every figure names its arm.
#
# Mechanics: test_dirs is the corpus with exclude_users = 32-48 and test_on_excluded=true
# (evaluation = exactly those 17); in C1 the SAME corpus is data_dirs, so 32-48 are removed
# from training by the exclude list and 23-31 by validation_users, leaving 0-22. In C2 the
# corpus is also in data_dirs beside BOXRR and alyx, with the same removals.
set -euo pipefail
ARM="${1:?C1|C2}"; SEED="${2:?seed}"
TREE=/run/media/feng/Data/CalebProject/XRSec/.claude/worktrees/across-xr-alignment
MAIN=/run/media/feng/Data/CalebProject/XRSec
PY="$MAIN/.venv313/bin/python"
export XRSEC_SAMPLE_CACHE_DIR="$MAIN/.cache/samples"
XR="$MAIN/processed_datasets/CrossApplicationXR_Dataset/users"
EXCL=""; for u in $(seq 32 48); do EXCL="${EXCL:+$EXCL,}$XR/$u"; done
VAL="";  for u in $(seq 23 31); do VAL="${VAL:+$VAL,}$XR/$u"; done
case "$ARM" in
    C1) DATA="[$XR]"; NAME=across_xr_matched_c1_dyn10s ;;
    C2) DATA="[$MAIN/processed_datasets/BOXRR-23_Dataset/users,$MAIN/processed_datasets/who_is_alyx/users,$XR]"; NAME=across_xr_matched_c2_dyn10s ;;
    *) echo "arm must be C1 or C2" >&2; exit 2 ;;
esac
cd "$TREE"
exec "$PY" model/main.py mode=train \
    "experiment_name=$NAME" \
    "data_dirs=$DATA" \
    "test_dirs=[$XR]" \
    "exclude_users=[$EXCL]" \
    "validation_users=[$VAL]" \
    test_on_excluded=true swap_data=false \
    extractor=bilstm objective=identity_softmax identity_margin=0.35 identity_scale=30.0 \
    encoding=dyn sample_time=10 sample_rate=20 window_stride=5 resample=nearest channels=full \
    normalize=per_dataset eval_normalize=target_fit within_dataset_negatives=true \
    cross_session_positives=true center_position=false \
    epochs=120 early_stopping_patience=15 val_user_fraction=0.25 \
    batch_size=1024 lr=0.001 weight_decay=0.0 samples_per_user=512 embedding_dim=128 \
    max_users=null balance_identities=false \
    "seed=$SEED"
