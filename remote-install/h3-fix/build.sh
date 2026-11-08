#!/bin/bash
set -euo pipefail
source "${BASH_SOURCE[0]%/*}/activate.sh"
cd "$H3_PROJECT_ROOT/InfiniCore"
xmake f --hygon-dcu=y --hygon-arch=gfx936 --nv-gpu=n --aten=y --ccl=y \
 --flash-attn="$H3_DEPS_ROOT/venv/lib/python3.11/site-packages" -m release -y
xmake build -j 24 infinicore_cpp_api
xmake install -o "$INFINI_ROOT" infinicore_cpp_api
