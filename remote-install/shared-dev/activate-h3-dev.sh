# Source inside the MiniMax-H3 worker-0 container.
if [ ! -f /workspace/h3-deps-20261101/activate-h3.sh ] || \
   [ ! -d /public/home/aclsylqidf/Projects/minimax-h3-dev/InfiniLM/python ]; then
    printf '%s\n' 'H3 development environment or shared source directory is unavailable.' >&2
    return 1
fi
source /workspace/h3-deps-20261101/activate-h3.sh || return
export H3_PROJECT_ROOT=/public/home/aclsylqidf/Projects/minimax-h3-dev
# The container runs as root; avoid creating bytecode caches in user-owned NFS source.
export PYTHONDONTWRITEBYTECODE=1

# Prefer the verified FlashAttention build stored with this project.
if [ -f "$H3_PROJECT_ROOT/remote-install/h3-fix/native/lib/libinfinicore_cpp_api.so" ]; then
    export INFINI_ROOT="$H3_PROJECT_ROOT/remote-install/h3-fix/native"
    export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
fi
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
