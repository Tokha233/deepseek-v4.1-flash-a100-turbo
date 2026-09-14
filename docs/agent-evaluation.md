# Agent and question-answering evaluations

All output counts include reasoning. Inference runs use the official native checkpoint; kernels dequantize weights for SM80 execution. This is not a proof of arithmetic equivalence to the vendor reference implementation.

| Evaluation | Result | Runtime/protocol |
|---|---:|---|
| SWE-bench Verified 500 | **406/500 (81.2%)** original | Historical capacity baseline; numeric effort100 |
| Same submissions, declared environment re-evaluation | **417/500 (83.4%)** diagnostic | 23 environment cases; report separately |
| Original DeepSWE effort50 | 73/113 (64.60%) | Historical protocol; three-hour task limit |
| Original DeepSWE effort75 | 61/113 (53.98%) | Same historical protocol |
| Original DeepSWE effort100 | 33/113 (29.20%) | Same historical protocol; many timeouts |
| New DeepSWE effort75/100 | Incomplete | Round11; 12-hour task, 30-minute setup; local resume after recorded operator interruption |
| GPQA Diamond | **176/198 (88.89%)** | Round12; one generation, effort100, 65,536 output cap; six length terminations remain counted |
| GSM8K strict | **1013/1319 (76.80%)** | Eight-shot CoT template with chat/thinking adaptation |
| GSM8K flexible extraction | 1260/1319 (95.53%) diagnostic | Same answers; alternate numeric extraction, not a replacement official score |
| Terminal-Bench 2.1 | No final result | 89-task image preparation/preflight incomplete |

SWE500 consumed 21,325,164 output tokens, including 16,382,537 reasoning tokens, over 29,964 model responses and 41,043 tool calls. The maximum observed prompt was 345,995 tokens. Complete controller throughput was 760.72 output tok/s over 28,032.77 seconds; incidents, tools and verification remain included. Original and environment-adjusted scores are both retained.

The official V4.1-Flash model card reports GPQA90.9, DeepSWE mini-SWE74.2, Terminal-Bench2.1 mini-SWE90.3/DSH Minimal90.6. DeepSWE uses eight samples per task, TB2.1 three, 1M context, effort100, temperature1/top_p0.95. These protocols differ from this deployment. GSM8K93.0 refers to the base model. We make no official-equivalence or lossless-quality claim.

Round11 operator interruption at 2026-09-14 12:54:41 UTC retained 82 effort75 and 43 effort100 completed task records. There were 54 and27 passes, respectively. All63 interrupted trial directories (31/32) were archived before incomplete slots restarted; no scored model failures were removed. Subsequent results must declare the interruption and restored slots. Partial completion rates are not final benchmark scores.

Public files under [results](../benchmarks/results) contain exact counts and fixed protocols. The [manifest](../benchmarks/evidence-manifest.json) binds exported metrics to private originals; prompt bodies and full trajectories are not in Git. A source hash is an audit reference, not a substitute for independently accessible raw evidence.
