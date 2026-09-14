# Two-node routing

Start one TP4 × DP2 vLLM server per eight-GPU node, then install the router separately:

```bash
python -m venv .router-venv
.router-venv/bin/pip install -r router/requirements.txt
cp router/config.example.json router/config.json
```

Edit the two backend URLs and the existing local tokenizer path. The tokenizer is read from disk; model weights are not downloaded. Start and check the Gateway:

```bash
.router-venv/bin/python router/launch.py router/config.json --dry-run
.router-venv/bin/python router/launch.py router/config.json
curl http://127.0.0.1:30000/workers
```

Use `http://127.0.0.1:30000/v1` as the agent's API base. To accept requests from another host, explicitly set `--host 0.0.0.0`.

The official SGLang Model Gateway owns the cache-aware routing policy. Four logical workers correspond to the two DP ranks on each node. The forwarding adapter inserts the vLLM `X-data-parallel-rank` header. It forwards streaming responses and cancels the upstream request when the client disconnects.

The policy tracks requests passing through the Gateway and estimates prefix reuse from request text. Its load count is not a GPU-utilization metric and does not include clients using a backend directly. Send agent traffic through one Gateway when evaluating this policy. It balances queues when the worker count difference exceeds eight and the relative difference exceeds 1.5; otherwise it prefers a matching prefix.

## Sharing backends with direct clients

When other clients use the inference servers directly, start the additional backend-metrics router after the rank proxies are running:

```bash
.router-venv/bin/python router/adaptive_router.py router/config.json
curl http://127.0.0.1:30001/state
```

Use `http://127.0.0.1:30001/v1` for this policy. It reads each engine's actual running and waiting requests, KV pressure, and generated-token counters. Short-lived reservations account for delayed backend metrics during bursts. Session keys preserve affinity until the load difference warrants migration. Backends without fresh healthy metrics receive no new requests. Capacity estimates follow measured generation rates; this is a scheduling heuristic, not a claim of optimal throughput.

The scheduler forwards requests through the same fixed-rank proxies and records streaming usage, TTFT, elapsed time, worker choice, and selection-time load. It does not retry generation automatically. Eight live cancellations and native two-turn mini-SWE tool loops at numeric efforts 50, 75, and 100 passed on two A800 nodes. With approximately 64 unrelated direct requests on the local node, the six tool-loop requests selected the idle remote node. A subsequent live evaluation used all four DP workers.

Report client usage separately from backend metrics when resources are shared: server counters include other clients. Full benchmark results must include tool execution and unsuccessful trials; short output-capped replays are capacity checks.

## Native request compatibility

Real V4.1 histories passed native prompt-token-count checks, numeric effort 100, top-k 20, top-p 0.95, and streaming usage checks. A real two-turn mini-SWE-agent tool loop also completed. Eight cancelled Gateway streams drained both backend queues and Gateway load counters.

Gateway 0.3.2 drops vLLM's `cache_salt` JSON field during typed reserialization. For explicit cache namespaces, also send the same value as `X-SMG-Routing-Key`; this adapter restores it as `cache_salt` and removes the transport header before calling vLLM. Conflicting values are rejected. Requests without a routing key use ordinary shared prefix caching. This namespace mapping is specific to this adapter.

An arbitrary custom HTTP header is insufficient: this Gateway forwards only a documented subset. The existing `X-SMG-Routing-Key` header passed an end-to-end check. Keep reasoning history in `reasoning_content` when constructing OpenAI-style messages; the raw DeepSeek `reasoning` alias is not part of the Gateway's typed message schema.

The audit JSONL records worker choice, elapsed time and relevant generation settings without recording prompt contents. The completed DeepSWE comparison measured **1,108.67 output tokens/s** over all 339 attempts, using all four DP workers and excluding unrelated output. Client prefix-cache hit ratios were 96.05%, 96.45%, and 97.33% for efforts 50, 75, and 100. See the [full results and limitations](../benchmarks/DEEPSWE_RESULTS.md). No matched single-node DeepSWE run establishes a scaling factor or optimal scheduling.

The packaged proxies close idle upstream connections after two seconds, below vLLM's default five-second HTTP keep-alive. A read-only check against the two real servers reproduced `ServerDisconnectedError` with aiohttp's default fifteen-second setting; all sixteen method-rejection checks with the two-second setting completed. These checks used `POST /health` and generated no model tokens. The completed scored evaluation retained its original fifteen-second proxy configuration: all 70 HTTP 500 responses recovered on the next transport attempt. The shorter setting has not yet been assessed in a complete scored run; it is not a guarantee against other connection failures. Keep the client idle timeout below the server timeout if changing either setting.

## Replaying native histories

The Gateway completion schema accepts prompt strings, not token-ID arrays. For a previously encoded native prompt, decode with special tokens preserved and require that encoding it again with `add_special_tokens=False` reproduces every input ID. The adapter suppresses automatic special-token insertion for completion prompts that already start with the native DeepSeek BOS token. Prompt token counts and prompt hashes are checked end to end in the internal replay.

Use chat requests with explicit tool definitions for real tool calls. A history-only chat replay without tool definitions can generate a DSML call but expose empty chat deltas; one captured 45-token response reproduced this behavior. Native completion replay preserves such raw output. Empty responses and partial streams must be recorded rather than discarded from a benchmark.
