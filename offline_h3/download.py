#!/usr/bin/env python3
"""Prepare CPython 3.11 Linux x86_64 wheels on an ONLINE Linux/WSL machine.

CPU torch is a real, local-only resolver input. It is never copied to upload/.
This does not certify target DTK ABI, H3 correctness, or native xmake packages.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR_INDEX = "https://pypi.sourcefind.cn/release/dtk/"


def run(args, cwd=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def capture(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_info(path):
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            raise RuntimeError(f"Invalid wheel METADATA: {path}")
        meta = email.message_from_bytes(archive.read(members[0]))
        return canonical(meta["Name"]), meta["Version"], sum(i.file_size for i in archive.infolist())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def clone(repo, ref, destination):
    if destination.exists():
        raise RuntimeError(f"Source already exists: {destination}; use --diffusers-source or a new --out")
    run(["git", "clone", "--depth", "1", "--branch", ref, "--recurse-submodules", repo, destination])
    return capture(["git", "rev-parse", "HEAD"], cwd=destination)


def archive_source(source, destination, root_name):
    def include(info):
        if ".git" in Path(info.name).parts or "__pycache__" in Path(info.name).parts:
            return None
        return info
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname=root_name, filter=include)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("h3-download"))
    parser.add_argument("--stage", choices=["public", "vendor", "complete"], default="public")
    parser.add_argument("--diffusers-source", type=Path, help="Exact organizer-validated checkout, preferred")
    parser.add_argument("--diffusers-ref", default="minimax-h3", help="Candidate branch/tag; commit recorded")
    parser.add_argument("--include-nine", action="store_true", help="Resolve ninetoothed and ntops too; requires complete stage")
    parser.add_argument("--native-sources", action="store_true", help="Also fetch torchvision v0.24.0 and xmake v3.1.2 source")
    parser.add_argument("--glibc-minor", type=int, default=28, help="Maximum manylinux glibc 2.x tag; default 28")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("Use online Linux x86_64 / Windows WSL with Python 3.11 (not native Windows Python).")
    if args.include_nine and args.stage != "complete":
        parser.error("--include-nine requires --stage complete so vendor Triton is constrained")
    if args.glibc_minor < 17:
        parser.error("glibc >= 2.17 required; check target with os.confstr('CS_GNU_LIBC_VERSION')")

    base = args.out.resolve()
    work = base / "local-only"
    upload = base / ("upload-" + args.stage + ("-nine" if args.include_nine else ""))
    if upload.exists():
        parser.error(f"Refusing to mix an existing bundle: {upload}. Use a new --out.")
    work.mkdir(parents=True, exist_ok=True)
    wheels = upload / "wheels"
    wheels.mkdir(parents=True)
    # Partial runs deliberately have no READY file or lock until all steps pass.
    (upload / "INCOMPLETE.txt").write_text("Preparation has not finished. Do not install this directory.\n")
    pip = [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check"]
    tags = ["--python-version", "3.11", "--implementation", "cp", "--abi", "cp311", "--abi", "abi3", "--abi", "none"]
    for minor in range(args.glibc_minor, 16, -1):
        tags += ["--platform", f"manylinux_2_{minor}_x86_64"]
    for tag in ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]:
        tags += ["--platform", tag]
    download = pip + ["download", "--no-cache-dir", "--only-binary=:all:"] + tags
    sources = {}
    vendor = work / "vendor"
    if args.stage in {"vendor", "complete"}:
        vendor.mkdir(exist_ok=True)
        run(download + ["--no-deps", "--index-url", VENDOR_INDEX, "-r", HERE / "requirements-vendor.txt", "-d", vendor])
    if args.stage == "vendor":
        for path in vendor.glob("*.whl"):
            shutil.copy2(path, wheels / path.name)
    else:
        roots = ["-r", str(HERE / "requirements-public.txt")]
        if args.stage == "complete":
            source = args.diffusers_source
            if source is None:
                source = work / "diffusers"
                sources["diffusers_commit"] = clone("https://github.com/huggingface/diffusers.git", args.diffusers_ref, source)
                sources["diffusers_origin"] = "candidate branch/tag; not certified as the organizer's exact build"
            else:
                source = source.resolve()
                sources["diffusers_origin"] = str(source)
                if (source / ".git").exists():
                    sources["diffusers_commit"] = capture(["git", "rev-parse", "HEAD"], cwd=source)
                    sources["diffusers_dirty"] = bool(capture(["git", "status", "--porcelain"], cwd=source))
            if not (source / "src/diffusers/modular_pipelines/minimax_h3").is_dir():
                raise RuntimeError("Selected Diffusers source does not contain modular_pipelines/minimax_h3")
            built = work / "diffusers-wheel"
            built.mkdir(exist_ok=True)
            run(pip + ["wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", built, source])
            candidates = list(built.glob("diffusers-*.whl"))
            if len(candidates) != 1:
                raise RuntimeError("Expected exactly one Diffusers wheel")
            roots += [str(candidates[0])] + [str(p) for p in sorted(vendor.glob("*.whl"))]
            if args.include_nine:
                roots += ["ninetoothed", "ntops"]

        # Resolve the real dependency graph with a real CPU torch wheel locally.
        # This avoids pulling public NVIDIA torch + its multi-GB CUDA dependencies.
        # The CPU wheel is discarded from the deployment set below.
        cpu = work / "cpu-torch"
        cpu.mkdir(exist_ok=True)
        run(download + ["--no-deps", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.9.0+cpu", "-d", cpu])
        cpu_wheels = list(cpu.glob("torch-*.whl"))
        if len(cpu_wheels) != 1:
            raise RuntimeError("Expected exactly one CPU torch resolver wheel")
        resolved = work / ("resolved-" + args.stage + ("-nine" if args.include_nine else ""))
        resolved.mkdir(exist_ok=True)
        run(download + ["--index-url", "https://pypi.org/simple", "-d", resolved, str(cpu_wheels[0])] + roots)
        for path in resolved.glob("*.whl"):
            name, _, _ = wheel_info(path)
            if name == "torch":
                continue
            if name.startswith("nvidia-") or name in {"torchvision", "torchaudio"}:
                raise RuntimeError(f"Unexpected platform wheel: {path.name}; obtain a DTK-compatible build")
            if name == "triton" and "das.opt1.dtk2604.torch290" not in wheel_info(path)[1]:
                raise RuntimeError("Resolver selected non-vendor Triton")
            shutil.copy2(path, wheels / path.name)

    if args.native_sources:
        target = upload / "sources"
        target.mkdir()
        for name, repo, ref in [
            ("torchvision", "https://github.com/pytorch/vision.git", "v0.24.0"),
            ("xmake", "https://github.com/xmake-io/xmake.git", "v3.1.2"),
        ]:
            checkout = work / name
            sources[name + "_commit"] = clone(repo, ref, checkout)
            archive_source(checkout, target / f"{name}-{ref}.tar.gz", f"{name}-{ref}")

    records = []
    seen = set()
    lock = []
    for path in sorted(wheels.glob("*.whl")):
        name, version, expanded = wheel_info(path)
        if name in seen or name == "torch":
            raise RuntimeError(f"Duplicate/protected distribution: {name}")
        seen.add(name)
        digest = sha256(path)
        records.append(dict(file=path.name, name=name, version=version, bytes=path.stat().st_size, expanded_bytes=expanded, sha256=digest))
        lock.append(f"{name}=={version} --hash=sha256:{digest}")
    if args.stage != "vendor":
        (upload / "install.lock").write_text("\n".join(lock) + "\n", encoding="utf-8")
        shutil.copy2(HERE / "install_python.sh", upload / "install_python.sh")
    (upload / "manifest.json").write_text(json.dumps(dict(stage=args.stage, includes_nine=args.include_nine, sources=sources, wheels=records), indent=2) + "\n")
    shutil.copy2(HERE / "下载与安装说明.md", upload / "下载与安装说明.md")
    sums = []
    for path in sorted(upload.rglob("*")):
        if path.is_file() and path.name != "INCOMPLETE.txt":
            sums.append(f"{sha256(path)}  {path.relative_to(upload).as_posix()}")
    (upload / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (upload / "INCOMPLETE.txt").unlink()
    (upload / "READY.txt").write_text("Download stage complete; native builds, target ABI and runtime tests still required.\n")
    print(f"Prepared: {upload}")
    print(f"Wheel bytes: {sum(r['bytes'] for r in records):,}")
    print(f"Expanded wheel bytes (excluding build/cache/output): {sum(r['expanded_bytes'] for r in records):,}")
    print("Upload only upload-*/; local-only contains CPU torch and MUST stay local.")
    if args.stage != "complete":
        print("This stage alone is NOT a complete H3 environment. See 下载与安装说明.md.")


if __name__ == "__main__":
    main()
