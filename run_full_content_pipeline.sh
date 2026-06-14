#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

.venv-mio/bin/python -u extract_content_cache.py \
  --output-dir data/mio_vctk_full_compact \
  --device mps \
  --resume \
  --log-every 25 \
  2>&1 | tee -a logs/full_content_cache.log

.venv/bin/python -u train_content_flat_ctc.py \
  --data-dir data/mio_vctk_full_compact \
  --device mps \
  --batch-size 2 \
  --epochs 30 \
  --steps-per-epoch 1000 \
  --probe-samples 1024 \
  --full-validation-every 5 \
  --learning-rate 2e-4 \
  --ctc-weight 0.05 \
  --hidden 512 \
  --layers 10 \
  --heads 8 \
  --attention-context-frames 200 \
  --run-name content_student_flat_ctc_512x10_full \
  --log-every 100 \
  2>&1 | tee -a logs/content_student_flat_ctc_full.log
