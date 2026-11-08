# Source inside a MiniMax-H3 worker container after editing remote-install/h3-paths.env.
source "${BASH_SOURCE[0]%/*}/../lib/load-config.sh" || return
if [ ! -x "$H3_DEPS_ROOT/venv/bin/python" ] || \
   [ ! -d "$H3_PROJECT_ROOT/InfiniLM/python" ]; then
    printf '%s\n' 'H3 development environment or shared source directory is unavailable.' >&2
    return 1
fi
source "$H3_DEPS_ROOT/venv/bin/activate"
# The container runs as root; avoid creating bytecode caches in user-owned NFS source.
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="$H3_DTK_ROOT/cuda/cuda"
export DTK_ROOT="$H3_DTK_ROOT"
export ROCM_PATH="$H3_DTK_ROOT"
export HIP_PATH="$H3_DTK_ROOT"
export HYGON_ARCH="${HYGON_ARCH:-gfx936}"
export INFINILM_CXX11_ABI="${INFINILM_CXX11_ABI:-1}"
export INFINI_ROOT="$H3_DEPS_ROOT/native"
export XMAKE_ROOT=y
export XMAKE_GLOBALDIR="$H3_DEPS_ROOT/build-tools"
export XMAKE_PROGRAM_DIR="$H3_DEPS_ROOT/build-tools/.local/share/xmake"
export PATH="$H3_DEPS_ROOT/build-tools/.local/bin:$H3_DTK_ROOT/llvm/bin:$H3_DTK_ROOT/cuda/cuda/bin:$H3_DTK_ROOT/bin:$PATH"
H3_TORCH_LIB="$(python -c 'import os,torch; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$H3_TORCH_LIB:$H3_DTK_ROOT/lib:$H3_DTK_ROOT/cuda/cuda/lib64:$H3_HYHAL_ROOT/lib:$H3_DTK_ROOT/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
for h3_flash_so in "$H3_DEPS_ROOT"/venv/lib/python3.11/site-packages/flash_attn_2_cuda*.so; do
    if [ -f "$h3_flash_so" ]; then
        export FA_SO_PATH="$h3_flash_so"
        export FLASH_ATTN_2_CUDA_SO="$h3_flash_so"
        break
    fi
done
unset h3_flash_so H3_TORCH_LIB

# Prefer the verified FlashAttention build stored with this project.
if [ -f "$H3_PROJECT_ROOT/remote-install/h3-fix/native/lib/libinfinicore_cpp_api.so" ]; then
    export INFINI_ROOT="$H3_PROJECT_ROOT/remote-install/h3-fix/native"
    export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"
fi
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
