#!/bin/bash
set -euo pipefail
chmod +x build/hello
./build/hello
python3 -u gpu_check.py
