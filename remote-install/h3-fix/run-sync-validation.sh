set -euo pipefail
cd /public/home/aclsylqidf/Projects/minimax-h3-dev
while [ -e /proc/86764/stat ] && [ "$(awk '{print $3}' /proc/86764/stat)" != Z ]; do sleep 5; done
grep -q 'install ok' audit/20261102-fix/build-lm-sync.log
unset MINIMAX_H3_TRACE_STAGE
bash remote-install/h3-fix/run-native-captured.sh
