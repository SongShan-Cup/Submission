source "${BASH_SOURCE[0]%/*}/../shared-dev/activate-h3-dev.sh" || return
export INFINI_ROOT="$H3_PROJECT_ROOT/remote-install/h3-fix/native"
case ":${LD_LIBRARY_PATH:-}:" in
    *":$INFINI_ROOT/lib:"*) ;;
    *) export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}" ;;
esac
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
