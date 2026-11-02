"""Explicit reload picks up the saved module version; no hidden hot reload."""
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import model

importlib.invalidate_caches()
importlib.reload(model)
checkpoint = json.loads(Path('checkpoint.json').read_text())
weights = [np.asarray(w) for w in checkpoint['weights']]
_, _, x, y = model.dataset()
predictions = model.forward(x, weights)[1]
result = dict(source_revision=model.SOURCE_REVISION,
              test_accuracy=float(np.mean((predictions >= .5) == y)),
              predictions_sha256=hashlib.sha256(predictions.round(8).tobytes()).hexdigest(),
              checkpoint_sha256=hashlib.sha256(Path('checkpoint.json').read_bytes()).hexdigest())
Path('inference.json').write_text(json.dumps(result))
print('INFER_RESULT', json.dumps(result), flush=True)
assert result['test_accuracy'] >= .9
