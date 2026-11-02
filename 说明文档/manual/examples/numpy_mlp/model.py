"""A small, real two-layer classifier. No downloads or third-party services."""
import numpy as np

SOURCE_REVISION = 1


def dataset():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(384, 2)).astype(np.float32)
    y = ((x[:, 0] * x[:, 1]) > 0).astype(np.float32)[:, None]
    return x[:256], y[:256], x[256:], y[256:]


def initialize():
    rng = np.random.default_rng(23)
    return [rng.normal(0, .5, (2, 16)), np.zeros((1, 16)),
            rng.normal(0, .3, (16, 1)), np.zeros((1, 1))]


def forward(x, weights):
    h = np.tanh(x @ weights[0] + weights[1])
    p = 1 / (1 + np.exp(-np.clip(h @ weights[2] + weights[3], -30, 30)))
    return h, p


def gradients(x, y, weights):
    h, p = forward(x, weights)
    delta = (p - y) / len(x)
    hidden = (delta @ weights[2].T) * (1 - h * h)
    loss = float(-np.mean(y * np.log(p + 1e-7) + (1-y) * np.log(1-p + 1e-7)))
    return loss, [x.T @ hidden, hidden.sum(0, keepdims=True),
                  h.T @ delta, delta.sum(0, keepdims=True)]
