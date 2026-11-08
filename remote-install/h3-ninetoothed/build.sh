#!/bin/bash
set -euo pipefail
source "${BASH_SOURCE[0]%/*}/activate.sh"
ulimit -c 0
cd "$H3_PROJECT_ROOT"
mkdir -p audit/20261103-ninetoothed "$INFINI_ROOT"
# Copy the already validated ABI-compatible dependencies; never alter that baseline.
cp -a remote-install/h3-fix/native/. "$INFINI_ROOT/"
python InfiniCore/src/infinicore/ops/multi_head_attention/ninetoothed/build_hygon.py \
 --output InfiniCore/build/h3-ninetoothed
install -m 755 InfiniCore/build/h3-ninetoothed/libh3_ninetoothed.so "$INFINI_ROOT/lib/"
cd InfiniCore
xmake f --hygon-dcu=y --hygon-arch=gfx936 --nv-gpu=n --aten=y --ccl=y \
 --ninetoothed=n --hygon-ninetoothed-mha=y \
 --flash-attn="$H3_DEPS_ROOT/venv/lib/python3.11/site-packages" -m release -y
xmake build -j 24 infinicore_cpp_api
xmake install -o "$INFINI_ROOT" infinicore_cpp_api
