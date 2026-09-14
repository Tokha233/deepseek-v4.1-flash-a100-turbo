"""Apply reviewed files only to the exact pinned source revision."""
import hashlib
import json
from pathlib import Path

import typer


def main(packages: Path = Path('/usr/local/lib/python3.12/dist-packages')) -> None:
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['files']:
        target=packages/item['target']
        actual=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
        if actual not in {item['before'],item['after']}:
            raise ValueError(f'Unexpected source revision: {target}: {actual}')
        payload=(root/'files'/item['target']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=item['after']:
            raise ValueError(f'Patch payload changed: {target}')
    for item in manifest['files']:
        target=packages/item['target']
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/'files'/item['target']).read_bytes())
    print(f'Applied and verified {len(manifest["files"])} files')


if __name__=='__main__':
    typer.run(main)
