# Source this file to select the independently built NineToothed checkpoint.
source "${BASH_SOURCE[0]%/*}/../shared-dev/activate-h3-dev.sh" || return
export INFINI_ROOT="$H3_PROJECT_ROOT/remote-install/h3-ninetoothed/native"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="$H3_PROJECT_ROOT/InfiniCore/build/h3-ninetoothed/triton-cache"
unset MINIMAX_H3_TRACE_STAGE
