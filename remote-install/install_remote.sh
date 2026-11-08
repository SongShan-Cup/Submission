#!/usr/bin/env bash
# Install the prepared public+DTK bundle, retaining the system HIP PyTorch.
set -euo pipefail
upload_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$upload_dir/lib/load-config.sh"
: "${IFAB_AFS_HOME:?Personal AFS path is not set}"
h3_python="${H3_SYSTEM_PYTHON:-/usr/local/bin/python}"
h3_venv="${H3_VENV:-$IFAB_AFS_HOME/envs/h3}"
cd "$upload_dir"
sha256sum -c SHA256SUMS
"$h3_python" - <<'PY'
import os, platform, sys, torch
assert sys.version_info[:2] == (3, 11), sys.version
assert platform.machine() == 'x86_64', platform.machine()
assert tuple(map(int, os.confstr('CS_GNU_LIBC_VERSION').split()[-1].split('.'))) >= (2, 28)
assert torch.version.hip and torch.__version__.split('+')[0] == '2.9.0'
print('System torch retained:', torch.__version__, torch.version.hip, torch.__file__)
PY
h3_scratch="$(mktemp -d "${H3_TMP_ROOT:-/tmp}/h3-python-install.XXXXXXXX")"
echo "Temporary files: $h3_scratch"
export TMPDIR="$h3_scratch"
# Stream the parts; do not create another large archive on AFS.
cat h3-python.tar.part-* | tar -xf - --no-same-owner -C "$h3_scratch"
cd "$h3_scratch/h3-python"
sha256sum -c SHA256SUMS
"$h3_python" -m venv --without-pip --system-site-packages "$h3_venv"
export H3_INSTALL_VENV="$h3_venv"
"$h3_venv/bin/python" - <<'PY'
import os, pathlib, runpy, sys
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(os.environ['H3_INSTALL_VENV']).resolve()
assert sys.prefix != sys.base_prefix
wheel, = pathlib.Path('bootstrap').glob('pip-*.whl')
sys.path.insert(0, str(wheel.resolve()))
sys.argv = ['pip', 'install', '--no-index', '--no-deps', '--no-cache-dir',
            '--ignore-installed', '--disable-pip-version-check', str(wheel)]
runpy.run_module('pip', run_name='__main__')
PY
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$h3_venv/bin/python" -m pip install --no-index --no-deps --no-cache-dir \
    --ignore-installed --no-compile --require-hashes --find-links wheels -r install.lock
"$h3_venv/bin/python" - "$h3_venv" <<'PY'
import pathlib, shlex, sys, torch
assert torch.version.hip and torch.__version__.split('+')[0] == '2.9.0'
venv = pathlib.Path(sys.argv[1]).resolve()
assert not pathlib.Path(torch.__file__).resolve().is_relative_to(venv), 'Unexpected private torch'
site = venv / 'lib/python3.11/site-packages'
flash, = site.glob('flash_attn_2_cuda*.so')
with flash.open('rb') as f:
    assert f.read(4) == b'\x7fELF'
torch_lib = pathlib.Path(torch.__file__).resolve().parent / 'lib'
lines = [
    '# Source this file in every new GPU terminal / batch script.',
    'source ' + shlex.quote(str(venv / 'bin/activate')),
    'export FA_SO_PATH=' + shlex.quote(str(flash)),
    'export FLASH_ATTN_2_CUDA_SO="$FA_SO_PATH"',
    'export PATH=' + shlex.quote(os.environ['H3_DTK_ROOT'] + '/llvm/bin:' + os.environ['H3_DTK_ROOT'] + '/cuda/cuda/bin:' + os.environ['H3_DTK_ROOT'] + '/bin:$PATH'),
    'export LD_LIBRARY_PATH=' + shlex.quote(str(torch_lib) + ':' + os.environ['H3_DTK_ROOT'] + '/lib:' + os.environ['H3_DTK_ROOT'] + '/cuda/cuda/lib64:') + '"${LD_LIBRARY_PATH:-}"',
    'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
]
(venv / 'activate-h3.sh').write_text('\n'.join(lines) + '\n')
print('Installed environment:', venv)
print('Private FlashAttention file is readable:', flash)
PY
cp "$upload_dir/verify_remote.py" "$h3_venv/verify_remote.py"
echo "Installation finished. In a GPU terminal run:"
printf 'source %q\n' "$h3_venv/activate-h3.sh"
printf 'python %q --gpu\n' "$h3_venv/verify_remote.py"
echo "Still missing: H3 Diffusers, matching torchvision, native InfiniCore/InfiniLM, NineToothed development environment."
# Only remove the temporary directory created by this script, after success.
cd "$upload_dir"
rm -rf -- "$h3_scratch"
