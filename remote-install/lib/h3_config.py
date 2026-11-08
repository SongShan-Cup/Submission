"""Load MiniMax-H3 deployment paths from remote-install/h3-paths.env."""
from __future__ import annotations

import os
from pathlib import Path
import shlex


def config_path() -> Path:
    default = Path(__file__).resolve().parents[1] / "h3-paths.env"
    return Path(os.environ.get("H3_CONFIG_FILE", default)).resolve()


def load() -> dict[str, str]:
    path = config_path()
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = shlex.split(value, comments=True, posix=True)[0]
    required = {
        "H3_PROJECT_ROOT", "H3_DEPS_ROOT", "H3_MODEL_ROOT",
        "H3_INPUT_IMAGE", "H3_DTK_ROOT", "H3_HYHAL_ROOT",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"Missing H3 config keys in {path}: {', '.join(missing)}")
    for key, value in values.items():
        os.environ[key] = value
    return values


def require(key: str) -> str:
    if key not in os.environ:
        load()
    value = os.environ[key]
    if not value.startswith("/"):
        raise RuntimeError(f"{key} must be an absolute path: {value}")
    return value
