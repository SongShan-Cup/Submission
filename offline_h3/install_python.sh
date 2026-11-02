#!/usr/bin/env bash
# Install only the prepared Python wheels into a PERSONAL venv, retaining DTK torch.
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${IFAB_AFS_HOME:?Set IFAB_AFS_HOME to your platform-provided personal AFS directory}"
h3_venv="${H3_VENV:-$IFAB_AFS_HOME/envs/h3}"
test -f "$bundle_dir/READY.txt"
test ! -f "$bundle_dir/INCOMPLETE.txt"
cd "$bundle_dir"
sha256sum -c SHA256SUMS
python - <<'PY'
import platform, sys, torch
assert sys.version_info[:2] == (3, 11), sys.version
assert platform.machine() == 'x86_64'
assert torch.version.hip, 'Expected the preinstalled vendor HIP PyTorch'
assert torch.__version__.split('+')[0] == '2.9.0', torch.__version__
print('Retaining system torch:', torch.__version__, torch.version.hip, torch.__file__)
PY
if [[ ! -f "$h3_venv/pyvenv.cfg" ]]; then
    python -m venv --system-site-packages "$h3_venv"
fi
source "$h3_venv/bin/activate"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
python -m pip install --no-index --no-deps --no-cache-dir --ignore-installed --require-hashes \
    --find-links "$bundle_dir/wheels" -r "$bundle_dir/install.lock"
python - <<'PY'
import torch
assert torch.version.hip, 'Vendor torch must remain active'
print('torch:', torch.__version__, torch.version.hip, torch.__file__)
PY
echo "Python wheel installation complete: $h3_venv"
echo "Still required: torchvision build, native InfiniCore/InfiniLM, environment paths and GPU smoke tests."
echo "Run python -m pip check; review inherited system-package conflicts separately."
