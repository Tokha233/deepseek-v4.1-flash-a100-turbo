"""Measure a single node at C1 through C256, with no overlapping test groups."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent


def main(output: Path, endpoint: str,
         source: Path = ROOT / 'data/short-coding-effort100.json.gz',
         warmup: int = 60, duration: int = 180) -> None:
    output.mkdir(parents=True, exist_ok=False)
    plan = {'concurrencies': [1, 2, 4, 8, 16, 32, 64, 128, 256],
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'warmup_seconds': warmup, 'measurement_seconds': duration,
        'max_tokens': 1024, 'ignore_eos': True, 'endpoint': endpoint}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2))
    for concurrency in plan['concurrencies']:
        directory = output / f'c{concurrency}'
        subprocess.run([sys.executable, str(ROOT / 'replay_sustained.py'), str(directory),
            endpoint, endpoint, '--source', str(source), '--last-turns', '1',
            '--concurrency', str(concurrency), '--max-tokens', '1024', '--ignore-eos',
            '--warmup', str(warmup), '--duration', str(duration)], check=True)
        receipt = json.loads((directory / 'completion.json').read_text())
        assert receipt['measurement_complete'] and receipt['drained'], receipt


if __name__ == '__main__':
    typer.run(main)
