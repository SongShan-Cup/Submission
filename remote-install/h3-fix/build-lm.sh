#!/bin/bash
set -euo pipefail
source /public/home/aclsylqidf/Projects/minimax-h3-dev/remote-install/h3-fix/activate.sh
cd "$H3_PROJECT_ROOT/InfiniLM"
xmake f -m release -y
xmake build -j 24 _infinilm
xmake install -o "$H3_PROJECT_ROOT/InfiniLM/python/infinilm" _infinilm
