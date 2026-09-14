# Benchmark protocol

The main service benchmark continuously replaces finished real-history sessions. Each slot replays 12 consecutive reference turns, then takes the next history. It uses natural EOS, numeric effort100 encoded into the prompt, temperature1, top_p0.95, top_k20, a32,768 output cap, distinct session cache salts and fixed DP affinity. Generated commands are not executed; full agent scores are separate.

The clock is fixed before execution: 360 s warmup, 600 s measurement. Tokens received inside that window are counted from returned token IDs, including streams crossing either boundary. All 20 thirty-second windows are published. SSE chunk count is never treated as token count. At measurement end, remaining diagnostic streams are cancelled and verified drained.

| Run | GPU nodes | Client C | Output tok/s | Mean server running | Minimum30s tok/s |
|---|---:|---:|---:|---:|---:|
| Round13 remote isolated, current source | 1 | 128 | **1,835.34** | 126.53 | 1,344.97 |
| Round9 local combined | 1 | 128 | 1,923.33 | 126.97 | 1,301.50 |
| Round9 dual combined | 2 | 256 | 3,750.18 | 251.20 | 3,017.07 |
| Round9 remote combined | 1 | 64 | 1,682.01 | 63.62 | 1,066.27 |
| Round9 remote baseline | 1 | 64 | 1,673.63 | 63.61 | 1,293.70 |

These are individual retained runs, not confidence bounds or causal speedup estimates. Full finite-batch replays include cold startup and long tails and produce lower throughput. Historical SGLang C64 results under`results/sglang-*` are finite three-turn replays and must not be compared directly with the600-second sustained table.

The fresh run has exactly 1,101,203 window tokens in client and server counters; no preemptions;92.49% prefix hits. Across startup+measurement,1,517 requests completed and128 in-flight streams were cancelled at the predetermined end. Post-cancellation server output can exceed client output slightly; see`completion.json`. Cancellation is not an evaluation failure and no cancelled request is scored as an agent pass.

## Run with your own authorized history data

```bash
python benchmarks/replay_sustained.py results/run1 http://localhost:8083 http://localhost:8083 \
  --source workloads/agent-histories.json.gz --concurrency 128 --dp-ranks 2 \
  --warmup 360 --duration 600
```

Input is gzipped JSON with`trajectories`: a list of distinct`instance_id` values, each containing at least 12`turns`, each with`input_ids` from the native V4.1 encoder. Supply at least as many distinct trajectories as concurrency. The source corpus is not bundled; [workload-profile.json](results/workload-profile.json) records the public upstream dataset identity, selection rule, exact source SHA256 and token-length distribution. Reproducing the exact number requires access to the same input corpus and tokenization, not just matching synthetic lengths.

Before testing, drain unrelated clients, check each DP's running/waiting gauges, and inspect GPU processes. Keep production traffic on another node. Retain backend counters and startup configuration. There is no automatic process killer in these scripts. A foreground load failure aborts the benchmark; an unfinished window is not a score.

## Native66K tool correctness

```bash
python benchmarks/native_tool_retrieval.py results/66k http://localhost:8083 \
  --tokenizer-path /path/to/existing/DeepSeek-V4.1-Flash/tokenizer.json --dp-ranks 2 --tokens 66000
```

This sends a synthetic access-code retrieval and validates streamed DSML function arguments, usage, EOS and warm-prefix hits. Fresh round13 results: cold TTFT6.286/6.325s; warm0.426/0.438s. Four calls passed. This small correctness check does not certify every long-context agent workload or1M support.

## Evidence and scoring

[Evidence manifest](evidence-manifest.json) binds public exports to original hashes. [SWE500 task ledger](results/swe500-tasks.json) records all 500 task outcomes and patch hashes, without prompt bodies or patches. Original scores and declared environment re-evaluations remain distinct. [Agent evaluation details](../docs/agent-evaluation.md) include old and incomplete new DeepSWE protocols, GPQA output caps and GSM8K extraction differences.

Public hashes support traceability; full private originals are not independently downloadable from this preview. No “official lossless” claim follows from these metrics. Any future release should preserve task/environment digests, sample count, context/output caps, failure counts, tool timing, input/output/reasoning/cache tokens and per-task interaction counts.
