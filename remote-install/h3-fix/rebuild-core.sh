set -euo pipefail
source /public/home/aclsylqidf/Projects/minimax-h3-dev/remote-install/h3-fix/activate.sh
cd "$H3_PROJECT_ROOT/InfiniCore"
xmake build -j 24 infinicore_cpp_api
cp build/linux/x86_64/release/libinfinicore_cpp_api.so "$INFINI_ROOT/lib/libinfinicore_cpp_api.so"
