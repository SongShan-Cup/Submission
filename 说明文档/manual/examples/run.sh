#!/bin/bash
set -euo pipefail
python3 -u gpu_quickstart.py --steps 12
python3 -u gpu_quickstart.py --infer --resume checkpoint.pt
