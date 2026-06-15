#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
log_file="${LOG_FILE:-logs/content_student_token_sync_phase0.log}"

.venv/bin/python -u train_content_token_sync.py \
  --device mps \
  --batch-size 2 \
  --epochs 3 \
  --steps-per-epoch 1000 \
  --learning-rate 2e-4 \
  --scheduler-t-max-epochs 10 \
  --supervised-mel-frames 300 \
  --history-mel-frames 100 \
  --pad-mel-multiple 64 \
  --probe-samples 1024 \
  --full-validation-every 3 \
  --log-every 50 \
  "$@" \
  2>&1 | tee -a "$log_file"
