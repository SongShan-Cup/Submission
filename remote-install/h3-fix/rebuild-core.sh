set -euo pipefail
source "${BASH_SOURCE[0]%/*}/activate.sh"
cd "$H3_PROJECT_ROOT/InfiniCore"
xmake build -j 24 infinicore_cpp_api
cp build/linux/x86_64/release/libinfinicore_cpp_api.so "$INFINI_ROOT/lib/libinfinicore_cpp_api.so"
