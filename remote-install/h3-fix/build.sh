#!/bin/bash
set -euo pipefail
source /public/home/aclsylqidf/Projects/minimax-h3-dev/remote-install/h3-fix/activate.sh
cd "$H3_PROJECT_ROOT/InfiniCore"
xmake f --hygon-dcu=y --hygon-arch=gfx936 --nv-gpu=n --aten=y --ccl=y --flash-attn=/workspace/h3-deps-20261101/venv/lib/python3.11/site-packages -m release -y
xmake build -j 24 infinicore_cpp_api
xmake install -o "$INFINI_ROOT" infinicore_cpp_api
