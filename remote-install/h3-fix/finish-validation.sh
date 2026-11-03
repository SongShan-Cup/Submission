set -euo pipefail
cd /public/home/aclsylqidf/Projects/minimax-h3-dev
while [ -e /proc/74053/stat ] && [ "$(awk '{print $3}' /proc/74053/stat)" != Z ]; do sleep 5; done
source remote-install/h3-fix/activate.sh
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1
ulimit -c 0
mv outputs/20261102-fix/fl2va_50step.mp4 outputs/20261102-fix/fl2va_before_vision72_fix.mp4
mv audit/20261102-fix/fl2va-50step.log audit/20261102-fix/fl2va-50step-before-vision72-fix.log
python -u remote-install/h3-fix/capture-first-step.py fl2va --model-path /public/home/aclsylqidf/shared/MiniMax-H3/diffusers --text 'A dog falls asleep' --first-frame /public/home/aclsylqidf/Golde33443.jpg --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps 50 --seed 42 --output outputs/20261102-fix/fl2va_50step.mp4 > audit/20261102-fix/fl2va-50step.log 2>&1
python remote-install/h3-fix/inspect-video.py
python -u remote-install/h3-fix/replay-first-step.py > audit/20261102-fix/first-step-comparison.log 2>&1
python remote-install/h3-fix/inspect-comparison.py
