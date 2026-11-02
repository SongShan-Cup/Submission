"""Training reads config.json each step; loaded Python code stays unchanged."""
import json
import time
from pathlib import Path
import numpy as np
import model

x, y, test_x, test_y = model.dataset()
weights = model.initialize()
first = [np.zeros_like(w) for w in weights]
second = [np.zeros_like(w) for w in weights]
initial_loss, _ = model.gradients(x, y, weights)
epochs = int(json.loads(Path('config.json').read_text())['epochs'])
for step in range(1, epochs + 1):
    config = json.loads(Path('config.json').read_text())
    lr = float(config['learning_rate'])
    assert 0 < lr <= .1
    loss, grads = model.gradients(x, y, weights)
    for i, gradient in enumerate(grads):
        first[i] = .9 * first[i] + .1 * gradient
        second[i] = .999 * second[i] + .001 * gradient * gradient
        weights[i] -= lr * (first[i] / (1-.9**step)) / (np.sqrt(second[i] / (1-.999**step)) + 1e-8)
    if step == 1 or step % 10 == 0:
        print(json.dumps(dict(epoch=step, loss=round(loss, 6), learning_rate=lr,
                              config_revision=config['revision'], source_revision=model.SOURCE_REVISION)), flush=True)
    time.sleep(min(.1, max(0, float(config.get('step_delay', 0)))))
accuracy = float(np.mean((model.forward(test_x, weights)[1] >= .5) == test_y))
checkpoint = dict(weights=[w.tolist() for w in weights], initial_loss=initial_loss,
                  final_loss=loss, test_accuracy=accuracy, source_revision=model.SOURCE_REVISION,
                  config_revision=config['revision'], learning_rate=lr, epochs=epochs)
Path('checkpoint.json').write_text(json.dumps(checkpoint))
print('TRAIN_RESULT', json.dumps({k: v for k, v in checkpoint.items() if k != 'weights'}), flush=True)
assert loss < initial_loss * .6 and accuracy >= .9
