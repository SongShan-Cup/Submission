"""Verify the uploaded dependency stage; does not load model weights."""
import argparse
import importlib
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
for module in ('torch', 'numpy', 'PIL', 'accelerate', 'transformers', 'av',
               'imageio', 'imageio_ffmpeg', 'soundfile', 'skimage', 'janus',
               'xxhash', 'safetensors', 'einops', 'triton'):
    loaded = importlib.import_module(module)
    print('IMPORT_OK', module, getattr(loaded, '__version__', ''), getattr(loaded, '__file__', ''))
import torch
assert torch.version.hip, 'Expected retained DTK PyTorch'
flash = Path(os.environ['FA_SO_PATH'])
with flash.open('rb') as stream:
    assert stream.read(4) == b'\x7fELF'
print('FLASH_FILE_READ_OK', flash)
if args.gpu:
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    assert torch.cuda.is_available()
    print('DEVICES', torch.cuda.device_count(), torch.cuda.get_device_name(0))
    torch.manual_seed(17)
    q, k, v = [torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    print('FLASH_SDPA_OK', tuple(actual.shape), 'max_abs_error', (actual-expected).abs().max().item())
print('DEPENDENCY_STAGE_OK; this does not certify the full H3 environment.')
