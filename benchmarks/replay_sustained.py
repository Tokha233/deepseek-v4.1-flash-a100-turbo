"""Keep real-history agent slots occupied and count streaming tokens in a fixed window."""

import asyncio
import gzip
import hashlib
import itertools
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import typer

ROOT = Path(__file__).resolve().parent
METRICS = ['num_requests_running', 'num_requests_waiting', 'generation_tokens_total',
           'request_success_total', 'num_preemptions_total', 'prefix_cache_queries_total',
           'prefix_cache_hits_total', 'spec_decode_num_drafts_total',
           'spec_decode_num_draft_tokens_total', 'spec_decode_num_accepted_tokens_total']


def parse_metrics(raw: str) -> dict[str, float]:
    values = Counter()
    for line in raw.splitlines():
        if line.startswith('vllm:') and '{' in line:
            key = line.split('{', 1)[0].removeprefix('vllm:')
            if key in METRICS:
                values[key] += float(line.rsplit(' ', 1)[1])
    assert all(key in values for key in ['num_requests_running', 'num_requests_waiting', 'generation_tokens_total'])
    return dict(values)


def main(output: Path, endpoint: str, nodes: list[str], concurrency: int = 128,
         warmup: int = 360, duration: int = 600, dp_ranks: int = 2, last_turns: int = 12,
         source: Path = Path('workloads/agent-histories.json.gz')) -> None:
    assert concurrency > 0 and warmup >= 0 and duration > 0 and duration % 30 == 0
    assert dp_ranks == 0 or len(nodes) == 1
    output.mkdir(parents=True, exist_ok=False)
    raw = source.read_bytes()
    rows = json.loads(gzip.decompress(raw))['trajectories']
    assert len(rows) >= concurrency and len({row['instance_id'] for row in rows}) == len(rows)
    assert all(len(row['turns']) >= last_turns for row in rows)
    prefix = 'sustained-' + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:20]
    protocol = {'endpoint': endpoint, 'nodes': nodes, 'concurrency': concurrency, 'dp_ranks': dp_ranks,
        'warmup_seconds': warmup, 'measurement_seconds': duration, 'last_turns': last_turns,
        'source': str(source.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'distinct_source_trajectories': len(rows), 'effort': 100, 'temperature': 1, 'top_p': .95,
        'top_k': 20, 'max_tokens': 32768, 'ignore_eos': False, 'session_prefix': prefix,
        'scope': 'Continuous replacement of real-history agent sessions, including new-session prefill and natural output. Fixed streaming-token window after fixed warmup; no cherry-picked window. Stored reference tool results are replayed; tools are not executed. In-flight diagnostic requests are cancelled only after measurement.'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    del raw

    async def run() -> None:
        records, samples = [], []
        buckets: dict[int, Counter] = defaultdict(Counter)
        sequence = itertools.count()
        active = 0
        origin = time.perf_counter()
        async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(14400, connect=15),
            limits=httpx.Limits(max_connections=concurrency + 32, keepalive_expiry=2)) as client:
            async def snapshot(label: str | None = None) -> dict:
                begun = time.perf_counter()
                responses = await asyncio.gather(*(client.get(node + '/metrics', timeout=10) for node in nodes))
                values = []
                for i, response in enumerate(responses):
                    response.raise_for_status()
                    if label:
                        (output / f'node{i}-metrics-{label}.txt').write_text(response.text)
                    values.append(parse_metrics(response.text))
                record = {'time': time.time(), 'relative_seconds': time.perf_counter() - origin,
                          'fetch_seconds': time.perf_counter() - begun, 'client_active': active, 'nodes': values}
                if label:
                    (output / f'snapshot-{label}.json').write_text(json.dumps(record, indent=2))
                return record

            before = await snapshot('before')
            assert all(n['num_requests_running'] == n['num_requests_waiting'] == 0 for n in before['nodes'])
            origin = time.perf_counter()
            protocol['started_at'] = time.time()
            (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))

            async def sample() -> None:
                while True:
                    row = await snapshot()
                    samples.append(row)
                    with (output / 'loads.jsonl').open('a') as log:
                        log.write(json.dumps(row) + '\n')
                    await asyncio.sleep(2)

            async def worker(slot: int) -> None:
                nonlocal active
                while True:
                    session = next(sequence)
                    row = rows[session % len(rows)]
                    salt = f'{prefix}-{session}'
                    for turn_index, turn in enumerate(row['turns'][-last_turns:]):
                        request_id = f'{salt}-{turn_index}'
                        headers = {'X-SMG-Routing-Key': salt, 'X-Request-Id': request_id}
                        if dp_ranks:
                            headers['X-data-parallel-rank'] = str(slot % dp_ranks)
                        payload = {'model': 'deepseek-ai/DeepSeek-V4.1-Flash', 'prompt': turn['input_ids'],
                            'temperature': 1, 'top_p': .95, 'top_k': 20, 'seed': 20260911 + 100 * (session % len(rows)) + turn_index,
                            'max_tokens': 32768, 'cache_salt': salt, 'return_token_ids': True,
                            'skip_special_tokens': False, 'stream': True,
                            'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
                        begin = time.perf_counter()
                        first, usage, done, finish, worker_name = None, None, False, None, None
                        token_count = 0
                        active += 1
                        try:
                            async with client.stream('POST', endpoint + '/v1/completions', json=payload, headers=headers) as response:
                                if response.is_error:
                                    (output / f'{request_id}-error.txt').write_bytes(await response.aread())
                                response.raise_for_status()
                                worker_name = response.headers.get('X-DSV4-Worker', f'dp{slot % dp_ranks}' if dp_ranks else endpoint)
                                async for line in response.aiter_lines():
                                    if line == 'data: [DONE]':
                                        done = True
                                    elif line.startswith('data: '):
                                        event = json.loads(line[6:])
                                        count = sum(len(choice.get('token_ids') or []) for choice in event.get('choices', []))
                                        now = time.perf_counter()
                                        if count:
                                            if first is None:
                                                first = now
                                            token_count += count
                                            buckets[math.floor(now - origin)][worker_name] += count
                                        usage = event.get('usage') or usage
                                        for choice in event.get('choices', []):
                                            finish = choice.get('finish_reason') or finish
                            assert done and usage and first is not None
                            assert usage['prompt_tokens'] == len(turn['input_ids'])
                            assert usage['completion_tokens'] == token_count, (request_id, usage, token_count)
                        finally:
                            active -= 1
                            record = {'request_id': request_id, 'session': session, 'instance_id': row['instance_id'],
                                'slot': slot, 'turn': turn_index, 'started_relative': begin - origin,
                                'finished_relative': time.perf_counter() - origin, 'done': done,
                                'ttft': first - begin if first is not None else None, 'usage': usage,
                                'output_token_ids': token_count, 'finish_reason': finish, 'worker': worker_name}
                            records.append(record)
                            with (output / 'requests.jsonl').open('a') as log:
                                log.write(json.dumps(record) + '\n')

            async def measure() -> tuple[dict, dict]:
                await asyncio.sleep(max(0, origin + warmup - time.perf_counter()))
                start = await snapshot('window-start')
                print(json.dumps({'stage': 'measurement_started', 'seconds': time.perf_counter() - origin}), flush=True)
                await asyncio.sleep(max(0, origin + warmup + duration - time.perf_counter()))
                end = await snapshot('window-end')
                return start, end

            sampler = asyncio.create_task(sample())
            workers = [asyncio.create_task(worker(slot)) for slot in range(concurrency)]
            measurement = asyncio.create_task(measure())
            completed = False
            try:
                ready, _ = await asyncio.wait([measurement, sampler, *workers], return_when=asyncio.FIRST_COMPLETED)
                for task in ready:
                    task.result()
                assert measurement in ready, 'A load worker or metrics sampler stopped early'
                start, end = measurement.result()
                window = [buckets.get(second, Counter()) for second in range(warmup, warmup + duration)]
                counts = Counter()
                for row in window:
                    counts.update(row)
                thirty = [sum(sum(row.values()) for row in window[i:i + 30]) / 30 for i in range(0, duration, 30)]
                deltas = {key: sum(b.get(key, 0) - a.get(key, 0) for a, b in zip(start['nodes'], end['nodes'])) for key in METRICS if key.endswith('_total')}
                loads = [s for s in samples if warmup <= s['relative_seconds'] < warmup + duration]
                result = {'output_tokens_per_second': sum(counts.values()) / duration, 'streamed_output_tokens': sum(counts.values()),
                    'measurement_seconds': duration, 'warmup_seconds': warmup, 'concurrency': concurrency,
                    'worker_output_tokens': dict(counts), 'all_30s_output_tps': thirty,
                    'median_30s_output_tps': statistics.median(thirty), 'minimum_30s_output_tps': min(thirty),
                    'mean_client_active': statistics.mean(s['client_active'] for s in loads),
                    'mean_server_running': statistics.mean(sum(n['num_requests_running'] for n in s['nodes']) for s in loads),
                    'mean_server_waiting': statistics.mean(sum(n['num_requests_waiting'] for n in s['nodes']) for s in loads),
                    'server_counter_deltas': deltas,
                    'server_generation_tokens_per_second': deltas['generation_tokens_total'] / (end['relative_seconds'] - start['relative_seconds']),
                    'mean_1_plus_accepted_per_draft': 1 + deltas['spec_decode_num_accepted_tokens_total'] / deltas['spec_decode_num_drafts_total'] if deltas['spec_decode_num_drafts_total'] else None,
                    'cache_hit_fraction': deltas['prefix_cache_hits_total'] / deltas['prefix_cache_queries_total'] if deltas['prefix_cache_queries_total'] else None,
                    'window_start_snapshot_relative': start['relative_seconds'], 'window_end_snapshot_relative': end['relative_seconds'],
                    'scope': protocol['scope']}
                (output / 'measurement-summary.json').write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)
                completed = True
            finally:
                for task in [measurement, sampler, *workers]:
                    task.cancel()
                await asyncio.gather(measurement, sampler, *workers, return_exceptions=True)
                deadline = time.perf_counter() + 120
                while True:
                    after = await snapshot()
                    if all(n['num_requests_running'] == n['num_requests_waiting'] == 0 for n in after['nodes']):
                        break
                    assert time.perf_counter() < deadline, 'Cancelled benchmark requests did not drain'
                    await asyncio.sleep(1)
                await asyncio.sleep(6)
                after = await snapshot('after-drain')
                (output / 'tokens-per-second.json').write_text(json.dumps({str(k): dict(v) for k, v in buckets.items()}))
                receipt = {'measurement_complete': completed, 'drained': True, 'requests_started': len(records),
                    'naturally_completed': sum(r['done'] for r in records), 'cancelled_or_failed': sum(not r['done'] for r in records),
                    'client_streamed_tokens_total': sum(r['output_token_ids'] for r in records),
                    'server_generated_tokens_total': sum(b['generation_tokens_total'] - a['generation_tokens_total'] for a, b in zip(before['nodes'], after['nodes'])),
                    'scope': 'Fixed-window requests may cross boundaries. Cancellation can leave a small amount of generated output unobserved by the client; this is recorded rather than equated with received tokens.'}
                (output / 'completion.json').write_text(json.dumps(receipt, indent=2))
                print(json.dumps(receipt), flush=True)

    asyncio.run(run())


if __name__ == '__main__':
    typer.run(main)
