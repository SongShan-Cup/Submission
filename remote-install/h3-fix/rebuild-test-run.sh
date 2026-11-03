set -euo pipefail
cd /public/home/aclsylqidf/Projects/minimax-h3-dev
bash remote-install/h3-fix/rebuild-core.sh
source remote-install/h3-fix/activate.sh
ulimit -c 0
python -u remote-install/h3-fix/check_attention.py > audit/20261102-fix/attention.log 2>&1
mv audit/20261102-fix/fl2va-50step.log audit/20261102-fix/fl2va-50step-before-head96-fix.log
bash remote-install/h3-fix/run-fl2va.sh > audit/20261102-fix/fl2va-50step.log 2>&1
