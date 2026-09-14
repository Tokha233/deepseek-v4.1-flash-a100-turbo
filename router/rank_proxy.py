"""Expose fixed vLLM DP ranks as transparent workers for SGLang Model Gateway."""
import asyncio
import hashlib
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


@dataclass
class Worker:
    name: str
    endpoint: str
    rank: int
    port: int


HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length'}
ROOT = Path(__file__).resolve().parent
CONFIG = json.loads(Path(sys.argv[1]).read_text())
MODEL = CONFIG.get('model', 'deepseek-ai/DeepSeek-V4.1-Flash')
TOKENIZER = str(Path(CONFIG['tokenizer']).resolve())
AUDIT = Path(CONFIG['audit']).resolve()
AUDIT.parent.mkdir(parents=True, exist_ok=True)



async def start(worker: Worker, client: ClientSession) -> web.AppRunner:
    async def forward(request: web.Request) -> web.StreamResponse:
        if request.path in {'/server_info', '/get_server_info', '/model_info', '/get_model_info'}:
            async with client.get(worker.endpoint + '/v1/models') as upstream:
                data = await upstream.json()
                if upstream.status != 200:
                    return web.json_response(data, status=upstream.status)
            model = next(x for x in data['data'] if x['id'] == MODEL)
            return web.json_response({'model_id': MODEL, 'model_path': TOKENIZER, 'served_model_name': MODEL,
                                      'tokenizer_path': TOKENIZER,
                                      'dp_size': 1, 'tp_size': 4, 'max_model_len': model.get('max_model_len'),
                                      'is_generation': True, 'architectures': ['DeepseekV41ForCausalLM']})
        body = await request.read()
        wire_cache_salt = None
        if request.path in {'/v1/chat/completions','/v1/completions'} and request.headers.get('X-SMG-Routing-Key'):
            data = json.loads(body)
            wire_cache_salt = data.get('cache_salt')
            salt = request.headers['X-SMG-Routing-Key']
            if len(salt)>1024 or (wire_cache_salt is not None and wire_cache_salt!=salt):
                raise web.HTTPBadRequest(text='Invalid or conflicting cache salt')
            data['cache_salt'] = salt
            body = json.dumps(data).encode()
        if request.path == '/v1/completions':
            data = json.loads(body)
            if isinstance(data.get('prompt'),str) and data['prompt'].startswith('<｜begin▁of▁sentence｜>'):
                data['add_special_tokens'] = False
                body = json.dumps(data).encode()
        headers = {k:v for k,v in request.headers.items() if k.lower() not in HOP_HEADERS | {'x-smg-routing-key'}}
        headers['X-data-parallel-rank'] = str(worker.rank)
        started = time.time()
        audit = None
        if request.path in {'/v1/chat/completions','/v1/completions'}:
            data = json.loads(body)
            audit = {'worker':worker.name, 'rank':worker.rank, 'started':started,
                     'body_sha256':hashlib.sha256(body).hexdigest(), 'keys':sorted(data), 'path':request.path, 'add_special_tokens':data.get('add_special_tokens'),
                     'prompt_sha256':hashlib.sha256(json.dumps(data.get('prompt')).encode()).hexdigest(),
                     'chat_template_kwargs':data.get('chat_template_kwargs'),
                     'max_tokens':data.get('max_tokens',data.get('max_completion_tokens')), 'stream':data.get('stream'),
                     'stream_options':data.get('stream_options'),
                     'top_k':data.get('top_k'), 'top_p':data.get('top_p'), 'seed':data.get('seed'),
                     'cache_salt':data.get('cache_salt'), 'wire_cache_salt':wire_cache_salt, 'temperature':data.get('temperature'),
                     'messages_sha256':hashlib.sha256(json.dumps(data.get('messages'),sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                     'request_id':headers.get('X-Request-Id',headers.get('x-request-id'))}
        status = None
        finished = False
        try:
            async with client.request(request.method, worker.endpoint + str(request.rel_url),
                                      data=body or None, headers=headers) as upstream:
                status = upstream.status
                outgoing = {k:v for k,v in upstream.headers.items() if k.lower() not in HOP_HEADERS}
                outgoing['X-DSV4-Worker'] = worker.name
                response = web.StreamResponse(status=status, headers=outgoing)
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                finished = True
                return response
        finally:
            if audit is not None:
                audit.update(status=status, finished=finished, seconds=time.time()-started)
                with AUDIT.open('a') as log:
                    log.write(json.dumps(audit)+'\n')

    app = web.Application(client_max_size=64*1024*1024)
    app.router.add_route('*', '/{path:.*}', forward)
    runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', worker.port).start()
    return runner


async def main() -> None:
    workers = [Worker(**v) for v in CONFIG['workers']]
    async with ClientSession(timeout=ClientTimeout(total=14400, connect=15, sock_read=3600),
                             connector=TCPConnector(limit=1024, keepalive_timeout=2),
                             trust_env=False, auto_decompress=False) as client:
        runners = [await start(worker,client) for worker in workers]
        print('rank proxies ready', flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            for runner in runners:
                await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
