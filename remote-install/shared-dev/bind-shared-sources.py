"""Compatibility notice for the retired one-site editable-install binding script."""
from pathlib import Path


repairer = Path(__file__).with_name("repair-relocated-env.py")
raise SystemExit(
    "bind-shared-sources.py is retired. Edit remote-install/h3-paths.env, then run:\n"
    f'  "$H3_DEPS_ROOT/venv/bin/python" {repairer}\n'
)
