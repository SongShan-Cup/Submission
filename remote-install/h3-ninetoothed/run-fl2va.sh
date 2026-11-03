#!/bin/bash
set -euo pipefail
source "${BASH_SOURCE[0]%/*}/activate.sh"
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1
ulimit -c 0
cd "$H3_PROJECT_ROOT"
mkdir -p outputs/20261103-ninetoothed audit/20261103-ninetoothed
python -u remote-install/h3-ninetoothed/run-instrumented.py fl2va \
 --model-path /public/home/aclsylqidf/shared/MiniMax-H3/diffusers \
 --text 'A dog falls asleep' --first-frame /public/home/aclsylqidf/Golde33443.jpg \
 --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps 50 --seed 42 \
 --output "$H3_PROJECT_ROOT/outputs/20261103-ninetoothed/fl2va_50step.mp4" "$@"
