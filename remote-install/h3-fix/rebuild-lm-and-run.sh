set -euo pipefail
source "${BASH_SOURCE[0]%/*}/activate.sh"
cd "$H3_PROJECT_ROOT/InfiniLM"
xmake build -j 24 _infinilm
xmake install -o "$H3_PROJECT_ROOT/InfiniLM/python/infinilm" _infinilm
cd "$H3_PROJECT_ROOT"
mv audit/20261102-fix/fl2va-50step.log audit/20261102-fix/fl2va-50step-before-expanded-scalar.log
bash remote-install/h3-fix/run-fl2va.sh > audit/20261102-fix/fl2va-50step.log 2>&1
