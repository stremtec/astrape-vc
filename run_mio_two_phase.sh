#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

.venv/bin/python -u train_content_mio_two_phase.py \
  --data-dir data/mio_vctk_full_compact \
  --audio-root /Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed \
  --transcript-root /Users/asill/asill/research2/datasets/vctk/txt \
  --device mps \
  --batch-size 2 \
  --phase1-epochs 10 \
  --phase2-epochs 20 \
  --steps-per-epoch 1000 \
  --phase1-learning-rate 2e-4 \
  --phase2-learning-rate 5e-6 \
  --teacher-probability 0.5 \
  --teacher-ctc-weight 0.05 \
  --original-ctc-weight 0.05 \
  --delta-weight 0.1 \
  --probe-samples 1024 \
  --full-validation-every 5 \
  --run-name content_student_mio_causal_two_phase \
  --log-every 100 \
  "$@" \
  2>&1 | tee -a logs/content_student_mio_causal_two_phase.log
