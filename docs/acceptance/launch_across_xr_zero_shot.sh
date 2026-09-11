#!/usr/bin/env bash
# The zero-shot instrument of docs/acceptance/across_xr_alignment_REGISTERED.md, one seed.
#
#   launch_across_xr_zero_shot.sh <seed>
#
# BOXRR-23 (all users) + who_is_alyx -> dyn, 10 s / 20 Hz / stride 5, bilstm,
# identity_softmax, 120 epochs, patience 15, val 0.25: the 9.14 configuration read field
# by field off row 661054c98a12. Evaluation is Across-XR users 32-48 ONLY: test_dirs is the
# corpus and exclude_users names those 17 with test_on_excluded=true, which keeps only the
# excluded users under test_dirs (eval.py's swap flip). The exclude paths MUST point into
# CrossApplicationXR_Dataset - pointed at the training corpus they match nothing, the loader
# reports 0 users, and the only tell is a stdout line. The harness asserts 17 at gate time.
#
# Self-contained on purpose: sets its own cwd (this worktree) and its own cache dir (the
# main checkout's, so the pre-built BOXRR entries are the ones read), so it does not
# depend on what the queue runner's cwd or environment happen to be.
set -euo pipefail
SEED="${1:?seed}"
# P2 of PAPER_PLAN (raw minus dyn on the same cells) uses the same script with ENCODING=raw;
# the experiment name carries the encoding so the two families can never be pooled by name.
ENCODING="${ENCODING:-dyn}"
# MARGIN/SCALE override the AM-softmax defaults for Amendment 7 (0.1 / 15); the name carries them.
MARGIN="${MARGIN:-0.35}"; SCALE="${SCALE:-30.0}"
SUFFIX=""; [ "$MARGIN" != "0.35" ] && SUFFIX="_m${MARGIN}s${SCALE}"
TREE=/run/media/feng/Data/CalebProject/XRSec/.claude/worktrees/across-xr-alignment
MAIN=/run/media/feng/Data/CalebProject/XRSec
PY="$MAIN/.venv313/bin/python"
export XRSEC_SAMPLE_CACHE_DIR="$MAIN/.cache/samples"
XR="$MAIN/processed_datasets/CrossApplicationXR_Dataset/users"
EXCL=""
for u in 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48; do
    EXCL="${EXCL:+$EXCL,}$XR/$u"
done
cd "$TREE"
exec "$PY" model/main.py mode=train \
    "experiment_name=across_xr_zero_shot_${ENCODING}10s${SUFFIX}" \
    "data_dirs=[$MAIN/processed_datasets/BOXRR-23_Dataset/users,$MAIN/processed_datasets/who_is_alyx/users]" \
    "test_dirs=[$XR]" \
    "exclude_users=[$EXCL]" \
    test_on_excluded=true swap_data=false \
    extractor=bilstm objective=identity_softmax "identity_margin=$MARGIN" "identity_scale=$SCALE" \
    "encoding=$ENCODING" sample_time=10 sample_rate=20 window_stride=5 resample=nearest channels=full \
    normalize=per_dataset eval_normalize=target_fit within_dataset_negatives=true \
    cross_session_positives=true center_position=false \
    epochs=120 early_stopping_patience=15 val_user_fraction=0.25 \
    batch_size=1024 lr=0.001 weight_decay=0.0 samples_per_user=512 embedding_dim=128 \
    max_users=null balance_identities=false \
    "seed=$SEED"
