"""Build public short coding prompts; these are not real agent histories."""
import gzip
import hashlib
import json
from pathlib import Path

import typer
from tokenizers import Tokenizer
from encoding_check import load

TOPICS = [
    'an LRU cache with thread-safe get and put operations',
    'a CSV reader supporting quoted delimiters and escaped quotes',
    'a bounded asynchronous queue with cancellation and backpressure',
    'a JSON Lines reader that reports malformed input locations',
    'a time-to-live cache with deterministic clock injection',
    'a retry helper with exponential backoff and a retry deadline',
    'a dependency graph with cycle detection and topological sorting',
    'an atomic file writer that cleans up temporary files on failure',
    'a rate limiter supporting bursts and steady request rates',
    'a merge routine for overlapping half-open time intervals',
    'a streaming SHA256 file verifier with useful error messages',
    'an in-memory event bus supporting unsubscribe during dispatch',
    'a configuration loader with environment overrides and validation',
    'a stable priority queue with FIFO tie breaking',
    'a rolling statistics accumulator supporting bounded memory',
    'a subprocess wrapper supporting timeout and captured stderr',
]


def main(model: Path, output: Path, effort: int = 100, count: int = 256) -> None:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    encoder = load(model / 'encoding/encoding.py', 'checkpoint_encoder')
    rows = []
    for index in range(count):
        tag = hashlib.sha256(f'short-coding-v1/{index}'.encode()).hexdigest()[:24]
        prompt = f'Run {tag}. Implement {TOPICS[index % len(TOPICS)]} in Python. Include code, invariants, edge cases, tests and performance analysis. Continue with concrete examples and discuss concurrent callers.'
        encoded = encoder.encode_messages([{'role': 'user', 'content': prompt}], thinking_mode='thinking', reasoning_effort=effort)
        ids = tokenizer.encode(encoded, add_special_tokens=False).ids
        rows.append({'instance_id': f'short-coding-{index}', 'turns': [{'input_ids': ids}], 'text': prompt})
    lengths = [len(row['turns'][0]['input_ids']) for row in rows]
    data = {'profile': {'kind': 'synthetic-short-coding', 'reasoning_effort': effort,
        'minimum_prompt_tokens': min(lengths), 'maximum_prompt_tokens': max(lengths),
        'distinct_prompts': count, 'distinct_task_templates': len(TOPICS),
        'scope': 'Continuous synthetic short coding requests. Fixed output cap and ignore_eos as recorded in protocol; outputs are not complete agent tasks or quality scores. Each request has a unique cache salt; no prefix sharing. Fixed token window after declared warmup. In-flight requests cancelled only after measurement.',
        'tokenizer_sha256': hashlib.sha256((model / 'tokenizer.json').read_bytes()).hexdigest(),
        'encoder_sha256': hashlib.sha256((model / 'encoding/encoding.py').read_bytes()).hexdigest()}, 'trajectories': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
    print(json.dumps(data['profile'], indent=2))


if __name__ == '__main__':
    typer.run(main)
