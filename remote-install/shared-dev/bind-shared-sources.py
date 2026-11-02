"""Relocate existing editable Python installs without rebuilding native libraries.
Run in the worker-0 container with its existing H3 virtual environment Python.
Back up existing configuration first. This is not a fresh dependency installer.
"""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

old = '/workspace/minimax-h3-upload-20261101'
new = '/public/home/aclsylqidf/Projects/minimax-h3-dev'
site = Path(sys.prefix) / 'lib/python3.11/site-packages'
if sys.prefix != '/workspace/h3-deps-20261101/venv':
    raise SystemExit('Use the existing H3 virtual environment Python.')
for package, repo in [('infinicore', 'InfiniCore'), ('infinilm', 'InfiniLM')]:
    if not (Path(new) / repo / 'python' / package / '__init__.py').is_file():
        raise SystemExit(f'Missing shared package: {package}')

changed = {}
for name in ['__editable__.infinilm-0.1.0.pth', '__editable___infinicore_0_1_0_finder.py']:
    path = site / name
    content = path.read_text()
    if old not in content and new not in content:
        raise SystemExit(f'Unexpected editable install configuration: {path}')
    changed[path] = content.replace(old, new).encode()
for package, repo in [('infinicore', 'InfiniCore'), ('infinilm', 'InfiniLM')]:
    path = site / f'{package}-0.1.0.dist-info/direct_url.json'
    metadata = json.loads(path.read_text())
    if metadata.get('url') not in [Path(old, repo).as_uri(), Path(new, repo).as_uri()]:
        raise SystemExit(f'Unexpected source URL: {path}')
    metadata['url'] = Path(new, repo).as_uri()
    changed[path] = json.dumps(metadata).encode()
for path, data in changed.items():
    path.write_bytes(data)
    print(f'Bound shared source: {path.name}')
# Keep installed-file metadata consistent with the relocated editable configuration.
for package in ['infinicore', 'infinilm']:
    record = site / f'{package}-0.1.0.dist-info/RECORD'
    rows = list(csv.reader(io.StringIO(record.read_text())))
    for row in rows:
        target = site / row[0]
        if target in changed:
            data = changed[target]
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
            row[1:] = [f'sha256={digest}', str(len(data))]
    with record.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)

bashrc = Path('/root/.bashrc')
marker = '# MiniMax-H3 shared-source development environment'
content = bashrc.read_text()
if marker not in content:
    content += '\n' + marker + '\n'
    content += 'if [ -r /public/home/aclsylqidf/Projects/minimax-h3-dev/remote-install/shared-dev/activate-h3-dev.sh ]; then\n'
    content += '    source /public/home/aclsylqidf/Projects/minimax-h3-dev/remote-install/shared-dev/activate-h3-dev.sh\n'
    content += 'fi\n'
    bashrc.write_text(content)
    print('Enabled automatic activation for new container Bash terminals.')
