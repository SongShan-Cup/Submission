set -euo pipefail
source "${BASH_SOURCE[0]%/*}/activate.sh"
cd "$H3_PROJECT_ROOT"
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1
ulimit -c 0
python -u remote-install/h3-fix/capture-first-step.py fl2va --model-path "$H3_MODEL_ROOT" --text 'A dog falls asleep' --first-frame "$H3_INPUT_IMAGE" --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps 50 --seed 42 --output outputs/20261102-fix/fl2va_50step.mp4 > audit/20261102-fix/fl2va-50step.log 2>&1
python remote-install/h3-fix/inspect-video.py
python -u remote-install/h3-fix/replay-first-step.py > audit/20261102-fix/first-step-comparison.log 2>&1
python remote-install/h3-fix/inspect-comparison.py
