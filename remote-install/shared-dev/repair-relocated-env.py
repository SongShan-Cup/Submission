"""Repair a moved H3 venv and bind editable packages to configured paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from h3_config import load


def executable_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file())


def repair_python_entry(path: Path, venv: Path) -> bool:
    data = path.read_bytes()
    if b"\0" in data or not data.startswith(b"#!"):
        return False
    lines = data.splitlines(keepends=True)
    first = lines[0].decode("utf-8", "ignore").strip()
    if "/bin/python" not in first:
        return False
    lines[0] = f"#!{venv / 'bin/python'}\n".encode()
    path.write_bytes(b"".join(lines))
    return True


def main() -> None:
    config = load()
    project_root = Path(config["H3_PROJECT_ROOT"]).resolve()
    deps_root = Path(config["H3_DEPS_ROOT"]).resolve()
    venv = deps_root / "venv"
    site = venv / "lib/python3.11/site-packages"
    if not venv.is_dir() or not site.is_dir():
        raise SystemExit(f"Configured venv is incomplete: {venv}")

    # Reconstruct the known venv metadata instead of attempting a textual move.
    # This also repairs a previously corrupted command line or editable mapping.
    base_executable = Path(getattr(sys, "_base_executable", "/usr/local/bin/python")).resolve()
    pyvenv = venv / "pyvenv.cfg"
    pyvenv.write_text(
        "\n".join(
            [
                f"home = {base_executable.parent}",
                "include-system-site-packages = true",
                f"version = {'.'.join(map(str, sys.version_info[:3]))}",
                f"executable = {base_executable.with_name('python3.11')}",
                f"command = {base_executable} -m venv --without-pip "
                f"--system-site-packages {venv}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    finder = site / "__editable___infinicore_0_1_0_finder.py"
    if not finder.is_file():
        raise SystemExit(f"Missing editable installer finder: {finder}")
    mapping = (
        "MAPPING: dict[str, str] = {"
        f"'infinicore': '{project_root / 'InfiniCore/python/infinicore'}', "
        f"'infinicore.test': '{project_root / 'InfiniCore/test/infinicore'}'"
        "}"
    )
    finder_text, replacement_count = re.subn(
        r"(?m)^MAPPING:.*$", mapping, finder.read_text(encoding="utf-8")
    )
    if replacement_count != 1:
        raise SystemExit(f"Could not uniquely rewrite MAPPING in {finder}")
    finder.write_text(finder_text, encoding="utf-8")

    infinilm_pth = next(iter(site.glob("__editable__.infinilm-*.pth")), None)
    if infinilm_pth is None:
        raise SystemExit("Missing editable infinilm .pth file")
    infinilm_pth.write_text(str(project_root / "InfiniLM/python") + "\n", encoding="utf-8")

    changed = [pyvenv, finder, infinilm_pth]
    activate = venv / "bin/activate"
    activate_text = activate.read_text(encoding="utf-8")
    activate_text, activate_count = re.subn(
        r'(?m)^VIRTUAL_ENV=.*$', f'VIRTUAL_ENV="{venv}"', activate_text
    )
    if activate_count != 1:
        raise SystemExit(f"Could not uniquely rewrite VIRTUAL_ENV in {activate}")
    activate.write_text(activate_text, encoding="utf-8")
    changed.append(activate)

    repaired_entrypoints = 0
    for path in executable_files(venv / "bin"):
        if repair_python_entry(path, venv):
            changed.append(path)
            repaired_entrypoints += 1

    for package, repo in (("infinicore", "InfiniCore"), ("infinilm", "InfiniLM")):
        metadata = next(site.glob(f"{package}-*.dist-info/direct_url.json"), None)
        if metadata is None:
            raise SystemExit(f"Missing direct_url.json for {package}")
        data = json.loads(metadata.read_text(encoding="utf-8"))
        data["url"] = (project_root / repo).as_uri()
        metadata.write_text(json.dumps(data) + "\n", encoding="utf-8")
        changed.append(metadata)

    bashrc = Path(os.environ.get("H3_SHELL_RC", Path.home() / ".bashrc"))
    marker = "# MiniMax-H3 configured development environment"
    activation = project_root / "remote-install/shared-dev/activate-h3-dev.sh"
    try:
        content = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
        content = re.sub(
            r"\n# MiniMax-H3 (shared-source d|configured d)evelopment environment\n.*?\nfi\n",
            "\n",
            content,
            flags=re.DOTLOAD,
        )
        if marker not in content:
            content += f"\n{marker}\nif [ -r {activation} ]; then\n    source {activation}\nfi\n"
            bashrc.write_text(content, encoding="utf-8")
        changed.append(bashrc)
    except OSError as error:
        print(f"Skipped shell activation update for {bashrc}: {error}")

    print(f"Configured project: {project_root}")
    print(f"Configured venv: {venv}")
    print(f"Repaired Python entry points: {repaired_entrypoints}")
    print(f"Changed {len(set(changed))} files.")

    dtk_root = Path(config["H3_DTK_ROOT"])
    hyhal_root = Path(config["H3_HYHAL_ROOT"])
    import torch
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            str(deps_root / "native/lib"),
            str(torch_lib),
            str(dtk_root / "lib"),
            str(dtk_root / "cuda/cuda/lib64"),
            str(hyhal_root / "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )
    expected = {
        "infinicore": project_root / "InfiniCore/python/infinicore",
        "infinilm": project_root / "InfiniLM/python/infinilm",
    }
    checks = "\n".join(
        f"import {name}; pathlib.Path({name}.__file__).resolve() == pathlib.Path({str(path)!r}).resolve()"
        for name, path in expected.items()
    )
    subprocess.run(
        [str(venv / "bin/python"), "-c", f"import pathlib\n{checks}"],
        env=env,
        check=True,
    )
    for name, path in expected.items():
        print(f"{name}: {path}")
    print("Relocated environment verification passed.")


if __name__ == "__main__":
    main()
