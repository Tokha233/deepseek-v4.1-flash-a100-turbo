"""Route to fixed DP workers using backend load, including traffic outside this router."""
import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import typer
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from routing_identity import conversation_key

ROOT = Path(__file__).resolve().parent
HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
       'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length'}
METRICS = {'running': 'num_requests_running', 'waiting': 'num_requests_waiting',
           'kv': 'kv_cache_usage_perc', 'tokens': 'generation_tokens_total',
           'preemptions': 'num_preemptions_total'}


def parse_metrics(raw: str, rank: int) -> dict[str, float]:
    values = {}
    for key, metric in METRICS.items():
        rows = [line for line in raw.splitlines() if line.startswith(f'vllm:{metric}{{')
                and f'engine="{rank}"' in line]
        if rows:
            values[key] = float(rows[0].rsplit(' ', 1)[1])
    if values.get('tokens') == 0:
        values = {'running': 0, 'waiting': 0, 'kv': 0, 'preemptions': 0, **values}
    if set(values) != set(METRICS):
        raise ValueError(f'Incomplete per-engine metrics for rank {rank}: {values}')
    return values


@dataclass
class Worker:
    name: str
    endpoint: str
    rank: int
    port: int
    observed: float = 0
    metrics: dict = field(default_factory=dict)
    active: dict[str, float] = field(default_factory=dict)
    capacity: float = 350
    rate_sample: tuple | None = None

    def load(self, now: float) -> float:
        # Backend counters refresh every few seconds. Reserve slots until that lag passes.
        recent = sum(started > now - 6 for started in self.active.values())
        return max(len(self.active), self.metrics.get('running', 0)
                   + 2 * self.metrics.get('waiting', 0) + recent)

    def healthy(self, now: float) -> bool:
        return now - self.observed < 10

    def cost(self, now: float) -> float:
        pressure = 1 + max(0, self.metrics.get('kv', 0) - .85) * 20
        return (self.load(now) + 1) / self.capacity * pressure

    def update(self, values: dict, now: float) -> None:
        if self.rate_sample is None or values['tokens'] < self.rate_sample[1]:
            self.rate_sample = (now, values['tokens'], values['running'])
        elif now - self.rate_sample[0] >= 30:
            elapsed = now - self.rate_sample[0]
            running = (values['running'] + self.rate_sample[2]) / 2
            if running >= 24:
                rate = (values['tokens'] - self.rate_sample[1]) / elapsed
                observed_capacity = max(100, min(4000, rate))
                self.capacity = .85 * self.capacity + .15 * observed_capacity
            self.rate_sample = (now, values['tokens'], values['running'])
        self.metrics, self.observed = values, now


class Router:
    def __init__(self, workers: list[Worker], client: ClientSession, output: Path, affinity_slots: float = 8):
        self.workers, self.client, self.output = workers, client, output
        assert affinity_slots >= 0
        self.affinity_slots = affinity_slots
        self.affinity: dict[str, tuple[str, float]] = {}

    def state(self) -> dict:
        now = time.monotonic()
        return {'time': time.time(), 'policy': 'backend_metrics_with_session_affinity',
                'affinity_slots': self.affinity_slots,
                'workers': [{'name': w.name, 'healthy': w.healthy(now), 'metrics': w.metrics,
                             'active': len(w.active), 'reserved_load': w.load(now),
                             'capacity_estimate': w.capacity, 'cost': w.cost(now),
                             'metrics_age_seconds': now - w.observed} for w in self.workers]}

    async def sample_node(self, endpoint: str) -> None:
        async with self.client.get(endpoint + '/health', timeout=ClientTimeout(total=3)) as response:
            response.raise_for_status()
        async with self.client.get(endpoint + '/metrics', timeout=ClientTimeout(total=3)) as response:
            response.raise_for_status()
            raw = await response.text()
        now = time.monotonic()
        for worker in self.workers:
            if worker.endpoint == endpoint:
                worker.update(parse_metrics(raw, worker.rank), now)

    async def poll(self) -> None:
        endpoints = sorted({w.endpoint for w in self.workers})
        step = 0
        while True:
            results = await asyncio.gather(*(self.sample_node(url) for url in endpoints), return_exceptions=True)
            if step % 5 == 0:
                state = self.state()
                state['poll_errors'] = {url: str(result) for url, result in zip(endpoints, results)
                                        if isinstance(result, BaseException)}
                with (self.output / 'loads.jsonl').open('a') as log:
                    log.write(json.dumps(state) + '\n')
                self.affinity = {key: value for key, value in self.affinity.items()
                                 if value[1] > time.monotonic() - 14400}
            step += 1
            await asyncio.sleep(1)

    def choose(self, key: str) -> Worker:
        now = time.monotonic()
        eligible = [w for w in self.workers if w.healthy(now)]
        if not eligible:
            raise web.HTTPServiceUnavailable(text='No backend has fresh, healthy metrics')
        best = min(eligible, key=lambda w: w.cost(now))
        old = self.affinity.get(key)
        previous = next((w for w in eligible if old and w.name == old[0]), None)
        if previous and previous.metrics.get('kv', 0) < .95 and previous.cost(now) <= best.cost(now) + self.affinity_slots / best.capacity:
            best = previous
        if key:
            self.affinity[key] = (best.name, now)
        return best

    async def forward(self, request: web.Request) -> web.StreamResponse:
        if request.path == '/state':
            return web.json_response(self.state())
        if request.path == '/health':
            return web.json_response(self.state(), status=200 if any(w.healthy(time.monotonic()) for w in self.workers) else 503)
        if request.path not in {'/v1/models', '/v1/chat/completions', '/v1/completions'}:
            raise web.HTTPNotFound()
        body = await request.read()
        data = json.loads(body) if body else {}
        header_salt = request.headers.get('X-SMG-Routing-Key')
        body_salt = data.get('cache_salt')
        if header_salt and body_salt and header_salt != body_salt:
            raise web.HTTPBadRequest(text='Conflicting session key and cache salt')
        key = header_salt or body_salt or conversation_key(data, request.headers.get('Authorization', ''))
        if not isinstance(key, str) or len(key) > 1024:
            raise web.HTTPBadRequest(text='Invalid session key')
        before = self.state()
        worker = self.choose(key)
        request_id = request.headers.get('X-Request-Id') or uuid.uuid4().hex
        reservation = uuid.uuid4().hex
        generation = request.path != '/v1/models'
        started = time.monotonic()
        if generation:
            worker.active[reservation] = started
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in HOP | {'x-data-parallel-rank', 'x-smg-target-worker'}}
        headers['X-Request-Id'] = request_id
        if key:
            headers['X-SMG-Routing-Key'] = key
        status, finished, first, first_token = None, False, None, None
        buffer, usage, done, finish_reason = b'', None, False, None
        usage_updates = 0
        try:
            async with self.client.request(request.method, f'http://127.0.0.1:{worker.port}' + str(request.rel_url),
                                           data=body or None, headers=headers) as upstream:
                status = upstream.status
                outgoing = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP}
                outgoing['X-DSV4-Worker'] = worker.name
                outgoing['X-Request-Id'] = request_id
                response = web.StreamResponse(status=status, headers=outgoing)
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if first is None:
                        first = time.monotonic() - started
                    if generation and data.get('stream'):
                        buffer += chunk
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            line = line.strip()
                            if line == b'data: [DONE]':
                                done = True
                            elif line.startswith(b'data: '):
                                event = json.loads(line[6:])
                                if event.get('usage'):
                                    usage = event['usage']
                                    usage_updates += 1
                                for choice in event.get('choices', []):
                                    delta = choice.get('delta', {})
                                    visible = choice.get('text') or any(delta.get(k) for k in
                                        ['content', 'reasoning', 'reasoning_content', 'tool_calls'])
                                    if visible and first_token is None:
                                        first_token = time.monotonic() - started
                                    finish_reason = choice.get('finish_reason') or finish_reason
                    await response.write(chunk)
                await response.write_eof()
                finished = True
                return response
        finally:
            worker.active.pop(reservation, None)
            if generation:
                audit = {'time': time.time(), 'request_id': request_id, 'session': key, 'worker': worker.name,
                         'effort': data.get('chat_template_kwargs', {}).get('reasoning_effort'),
                         'body_sha256': hashlib.sha256(body).hexdigest(), 'status': status, 'finished': finished,
                         'seconds': time.monotonic() - started, 'first_chunk_seconds': first, 'ttft_seconds': first_token,
                         'usage': usage, 'usage_updates': usage_updates, 'done': done,
                         'finish_reason': finish_reason, 'selection': before}
                with (self.output / 'requests.jsonl').open('a') as log:
                    log.write(json.dumps(audit) + '\n')


async def serve(config: Path, host: str, port: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads(config.read_text())
    workers = [Worker(**value) for value in settings['workers']]
    async with ClientSession(timeout=ClientTimeout(total=14400, connect=15, sock_read=3600),
                             connector=TCPConnector(limit=1024, keepalive_timeout=2),
                             trust_env=False, auto_decompress=False) as client:
        router = Router(workers, client, output, settings.get('affinity_slots', 8))
        poller = asyncio.create_task(router.poll())
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route('*', '/{path:.*}', router.forward)
        runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        print(f'Adaptive backend-metrics router listening on {host}:{port}', flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            poller.cancel()
            await runner.cleanup()


def main(config: Path, host: str = '127.0.0.1', port: int = 30001,
         output: Path = Path('results/adaptive-router')) -> None:
    asyncio.run(serve(config, host, port, output))


if __name__ == '__main__':
    typer.run(main)
