"""Run keplerzip's pinned HTTP benchmark with only model alias and context metadata adapted."""

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import typer

COMMIT = 'bd4fd63d1e01ee25c3ba89eb0f17dfe937461eb7'
BASE = f'https://raw.githubusercontent.com/keplerzip/deepseek-v4.1-flash-a100/{COMMIT}/deploy/tests'
HASHES = {
    'acceptance.py': '17e346c36a2a4689ab82f5424a3accb2e0882e66a705348297089f936e4f5650',
    'dspark_benchmark.py': '19f04b29b0a35034fe38bb93717009768091433c398f7bd7091f73fbd5730be8',
}


def main(output: Path, endpoint: str, mode: str, seed: str,
         model: str = 'deepseek-ai/DeepSeek-V4.1-Flash', context: int = 524288,
         source_dir: Path | None = None) -> None:
    assert mode in ('off', 'high') and context > 0
    output.mkdir(parents=True, exist_ok=False)
    adapted = output / 'runner'
    adapted.mkdir()
    sources = []
    for name, expected in HASHES.items():
        if source_dir:
            raw = (source_dir / name).read_bytes()
        else:
            with urllib.request.urlopen(f'{BASE}/{name}', timeout=60) as response:
                raw = response.read()
        assert hashlib.sha256(raw).hexdigest() == expected, f'Upstream source changed: {name}'
        old, new = ("MODEL = 'DeepSeek-V4.1-Flash'", f'MODEL = {model!r}') if name == 'acceptance.py' else (
            "'context': 262144", f"'context': {context}")
        text = raw.decode()
        assert text.count(old) == 1
        changed = text.replace(old, new)
        (adapted / name).write_text(changed)
        sources.append({'file': name, 'url': f'{BASE}/{name}', 'upstream_sha256': expected,
            'adapted_sha256': hashlib.sha256(changed.encode()).hexdigest(), 'replacement': [old, new]})
    (output / 'provenance.json').write_text(json.dumps({'sources': sources, 'mode': mode,
        'numeric_effort': 75 if mode == 'high' else None, 'seed': seed,
        'server_context_limit': context, 'upstream_server_context_limit': 262144,
        'scope': 'Original request generator, 2 warmups, 5 single samples, 200 requests at C32, T0, sampling seed42, fixed1024 output, ignore_eos; original full-load wall-clock timing. Model alias and reported context are the only source changes. The upstream report did not publish its nonce seed; identical request hashes are not claimed.'}, indent=2))
    subprocess.run([sys.executable, str(adapted / 'dspark_benchmark.py'), '--base-url', endpoint,
        '--output', str(output / 'report.json'), '--seed', seed, '--mode', mode, '--k', '5'], check=True)


if __name__ == '__main__':
    typer.run(main)
