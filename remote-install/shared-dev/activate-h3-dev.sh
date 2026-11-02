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
