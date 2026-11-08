#!/bin/bash
set -euo pipefail
source "${BASH_SOURCE[0]%/*}/activate.sh"
cd "$H3_PROJECT_ROOT/InfiniLM"
xmake f -m release -y
xmake build -j 24 _infinilm
xmake install -o "$H3_PROJECT_ROOT/InfiniLM/python/infinilm" _infinilm
