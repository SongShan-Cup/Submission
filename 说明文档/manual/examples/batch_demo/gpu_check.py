"""Small offline DCU check; requires the certified preinstalled PyTorch image."""
import json
from pathlib import Path
import torch

value = float(torch.arange(8, device="cuda").sum().cpu())
assert value == 28.0
result = {"sum": value, "visible_devices": torch.cuda.device_count()}
Path("result.json").write_text(json.dumps(result))
print("GPU_BATCH_RESULT", json.dumps(result), flush=True)
