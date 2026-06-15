#!/usr/bin/env bash
# Long-running FFL content student training launched in a detached screen session.
#
# Usage:
#   ./run_mio_ffl.sh                 # full 30-epoch training
#   ./run_mio_ffl.sh smoke           # quick (5 step / 1 epoch) smoke
#   ./run_mio_ffl.sh resume          # resume from the output-effect scratch run

set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-full}"

case "${MODE}" in
    smoke)
        STEPS=15
        EPOCHS=2
        PROBE=64
        LOG=1
        FULL_VAL=1
        NAME="content_student_mio_ffl_output_effect_smoke"
        ;;
    resume)
        STEPS=1000
        EPOCHS=30
        PROBE=1024
        LOG=100
        FULL_VAL=5
        NAME="content_student_mio_ffl_output_effect_scratch"
        RUN_RESUME=(--resume "checkpoints/${NAME}.last.pt")
        ;;
    full|*)
        STEPS=1000
        EPOCHS=30
        PROBE=1024
        LOG=100
        FULL_VAL=5
        NAME="content_student_mio_ffl_output_effect_scratch"
        RUN_RESUME=()
        ;;
esac

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${NAME}_$(date -u +%Y%m%d_%H%M%S).log"
SESSION="mio-ffl-output-effect"

# Kill any existing session of the same name to avoid duplication.
screen -S "${SESSION}" -X quit 2>/dev/null || true

# Wrap the launch in nohup so the process survives even if the screen client
# goes away; the captured log is the canonical record.
COMMAND=$(cat <<EOF
exec .venv/bin/python -u train_content_mio_ffl.py --device mps \
    --data-dir data/mio_vctk_full_compact \
    --output-dir checkpoints \
    --run-name ${NAME} \
    --batch-size 2 \
    --epochs ${EPOCHS} \
    --steps-per-epoch ${STEPS} \
    --probe-samples ${PROBE} \
    --full-validation-every ${FULL_VAL} \
    --log-every ${LOG} \
    --learning-rate 2e-4 \
    --ctc-weight 0.05 \
    --delta-weight 0.1 \
    --output-effect-weight 0.5 \
    --output-effect-cosine-weight 0.1 \
    --gate-l2-weight 0 \
    --causal-warmup-epochs 1 \
    --effect-warmup-epochs 1 \
    --effect-warmup-gate 0.25 \
    --pad-mel-multiple 64 \
    --mps-empty-cache-every 100 \
    ${RUN_RESUME[@]:-}
EOF
)

screen -dmS "${SESSION}" bash -c "${COMMAND} 2>&1 | tee -a ${LOG_FILE}"
echo "Screen session '${SESSION}' started, log: ${LOG_FILE}"
echo "Attach with:    screen -r ${SESSION}"
echo "Inspect log:    tail -f ${LOG_FILE}"
