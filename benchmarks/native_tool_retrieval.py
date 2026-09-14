"""Check native tool retrieval and warm prefix reuse on each selected DP rank."""

import hashlib
import json
import time
from pathlib import Path

import httpx
import typer
from tokenizers import Tokenizer

MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'



def main(output: Path, endpoint: str, dp_ranks: int = 2, tokens: int = 66000, framework: str = "vllm", tokenizer_path: Path = Path("/models/DeepSeek-V4.1-Flash/tokenizer.json")) -> None:
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    noise = '\n'.join(f'Record {i:06d}: ordinary archive entry, status unchanged, no access code here.'
                      for i in range(tokens // 10 + 1000))
    ids = tokenizer.encode(noise, add_special_tokens=False).ids
    assert len(ids) >= tokens
    marker = 'ORCHID-66000-7391'
    text = tokenizer.decode(ids[:tokens // 2]) + f'\nThe unique access code is {marker}.\n'
    text += tokenizer.decode(ids[tokens // 2:tokens]) + '\nCall report with the unique access code.'
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': text}],
               'tools': [{'type': 'function', 'function': {'name': 'report', 'description': 'Report the access code.',
                          'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                                         'required': ['code']}}}],
               'temperature': 0, 'max_tokens': 2048,
               'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 100},
               'stream': True, 'stream_options': {'include_usage': True}}
    (output / 'request.json').write_text(json.dumps(payload))
    results = []
    with httpx.Client(base_url=endpoint, timeout=1800, trust_env=False) as client:
        for rank in range(dp_ranks):
            payload['cache_salt'] = hashlib.sha256(f'{output.resolve()}/{rank}'.encode()).hexdigest()
            for warm in [False, True]:
                label = f'dp{rank}-' + ('warm' if warm else 'cold')
                (output / f'{label}-metrics-before.txt').write_text(client.get('/metrics').raise_for_status().text)
                started, first = time.monotonic(), None
                lines, calls, usage, finish = [], {}, None, None
                with client.stream('POST', '/v1/chat/completions', json=payload,
                                   headers={'X-data-parallel-rank': str(rank)}) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        lines.append(line)
                        if not line.startswith('data: ') or line == 'data: [DONE]':
                            continue
                        event = json.loads(line[6:])
                        usage = event.get('usage') or usage
                        for choice in event.get('choices', []):
                            delta = choice.get('delta', {})
                            if first is None and any(delta.get(k) for k in ['content', 'reasoning', 'reasoning_content', 'tool_calls']):
                                first = time.monotonic()
                            finish = choice.get('finish_reason') or finish
                            for call in delta.get('tool_calls') or []:
                                item = calls.setdefault(call['index'], {'name': '', 'arguments': ''})
                                for key in item:
                                    item[key] += call.get('function', {}).get(key) or ''
                seconds = time.monotonic() - started
                (output / f'{label}.sse').write_text('\n'.join(lines))
                assert usage and first is not None and 'data: [DONE]' in lines and finish != 'length'
                assert usage['prompt_tokens'] >= tokens
                assert len(calls) == 1 and calls[0]['name'] == 'report', calls
                assert json.loads(calls[0]['arguments'])['code'] == marker, calls
                cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
                assert not warm or cached >= tokens - 512, (rank, cached, tokens)
                results.append({'rank': rank, 'warm': warm, 'seconds': seconds, 'ttft': first - started,
                                'usage': usage, 'finish_reason': finish, 'tool_calls': calls})
                (output / 'results.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(results[-1]), flush=True)
    (output / 'validation.json').write_text(json.dumps({'passed': True, 'endpoint': endpoint,
        'dp_ranks': dp_ranks, 'minimum_prompt_tokens': tokens, 'requests': len(results),
        'framework': framework, 'scope': f'{framework} native tool correctness and warm GPU prefix reuse; allow 512 tokens of uncached tail; synthetic retrieval, not agent throughput.'}, indent=2))


if __name__ == '__main__':
    typer.run(main)
