# DeepSeek-V4.1-Flash · Ampere Turbo

面向长上下文 coding agent 的 SM80 部署与评测方案，基于 [vLLM backport](https://github.com/wtdcode/vllm-backport/tree/master-v013)。**实测硬件：每台 8×A800-SXM4-80GB**；A100 属于移植目标，尚未在 A100 上复测。

[测试方法](benchmarks/README.md) · [C256 与短请求](docs/concurrency-and-short-requests.zh-CN.md) · [Agent 评测](docs/agent-evaluation.md) · [18 个社区项目对照](docs/community-comparison.zh-CN.md)

## 推理速度

所有吞吐为 **output tokens/s，包含 reasoning token**。测试节点隔离生产请求，无额外占卡程序。

<!-- PERFORMANCE_TABLE -->
| 场景 | 硬件 | 客户端并发 | 吞吐 |
|---|---|---:|---:|
| 短 coding，effort100，固定输出 1,024 | 8×A800 | 128 | **2,934.20** |
| 真实长历史持续回放，effort100 | 8×A800 | 128 | **1,835.34** |
| 真实长历史持续回放，effort100 | 8×A800 | 256 | **1,905.89** |
<!-- /PERFORMANCE_TABLE -->

短请求输入 78–90 token，60 秒预热后测完整 180 秒；长历史输入 20,593–73,688 token，360 秒预热后测完整 600 秒、自然 EOS。长历史回放使用真实工具记录，但不执行新生成的命令。单机最多运行 128 条序列，客户端 C256 包含排队。[原始收据与 hash](benchmarks/evidence-manifest.json)

<!-- COMMUNITY_RESULTS -->
社区协议对照（固定 keplerzip commit、C32、200×1,024、temperature0、完整批次计时）：本方案 off **1,905.42 tok/s**（两次均值，1,897.65–1,913.20），high/effort75 **1,480.52 tok/s**（1,447.72–1,513.31）；keplerzip 公布值为 1,379.01 / 1,039.91。相对值约 +38.2% / +42.4%，但对方为 8×A100 TP8、本方案为 8×A800 TP4×DP2，不能归因成纯软件优势。
<!-- /COMMUNITY_RESULTS -->

## 采用的优化

- **SM80 量化路径**：FP4 专家使用 Marlin，dense 层按 batch 选择 BF16 GEMM；避免 A800 落入通用反量化 fallback。
- **Decode**：DSpark5 草稿与 CUDA Graph 验证，FP32 输出头和本地 argmax；减少逐 token kernel launch 与同步。
- **Attention / KV**：稀疏 MLA candidate-only 路径、FP8 KV cache、GPU prefix caching；长 agent 后续轮次可复用前缀。
- **MoE 通信**：TP4×DP2、EP8，使用 AllGather/ReduceScatter，并启用异步 EPLB；草稿和正式请求隔离负载均衡。
- **协议与工程**：原生 DSML、数字 reasoning effort、流式 tool-call 解析、固定 DP 亲和路由，以及 600 秒持续窗口和服务端计数校验。

这些优化来自 vLLM/backport、SGLang 社区和本项目的适配补丁；仓库明确区分上游代码、配置调优和本地工程贡献。

## Agent 与问答评测

| 测试 | 结果 | 说明 |
|---|---:|---|
| SWE-bench Verified | **406/500 · 81.2%** | 实际执行工具与 verifier；早期 runtime |
| SWE500 环境复评 | 417/500 · 83.4% | 同批提交的环境诊断，单列 |
| 最大真实 SWE500 输入 | **345,995 tokens** | 29,964 模型轮、41,043 工具调用 |
| GPQA Diamond | 176/198 · 88.89% | effort100；6 条达到输出上限，仍计分 |
| GSM8K | 76.80% strict / 95.53% flexible | flexible 是答案格式抽取诊断 |
| DeepSWE effort75 / 100 | 续跑中 | 12 小时任务预算；保留中断恢复记录 |
| Terminal-Bench 2.1 | 尚无正式分数 | 89 题评测未完成 |

完整 SWE500 控制器吞吐 **760.72 tok/s**，含工具、验证、故障和尾部耗时。早期 SWE500 分数不代表最新全部补丁已完成质量回归；这些结果也不能证明与官方成绩等效。[逐任务台账](benchmarks/results/swe500-tasks.json)

## 部署

**TP4×DP2 · EP8 · DSpark5 · CUDA Graph · FP4 Marlin · FP8 MLA KV · GPU 前缀缓存**

显存预算 **0.90**，上下文 **524,288**，chunk8,192，每 DP64 条序列；dense BF16 阈值32、FP32 输出头、Engram CPU offload、target EPLB 与 draft 隔离。当前服务为纯文本。

需要八张 80GB NVLink GPU 与充足主存（实测主机约 2TB RAM）。当前为**源码预览版**：运行配置已导出，打包 Dockerfile 尚待完整重建与启动验收；暂不发布已验证镜像。

```bash
git clone https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo.git
cd deepseek-v4.1-flash-a100-turbo
pip install -r requirements.txt
docker build -t deepseek-v41-sm80:local .
python launch.py /path/to/existing/DeepSeek-V4.1-Flash --port 8083 --dry-run
python launch.py /path/to/existing/DeepSeek-V4.1-Flash --port 8083
```

启动器只读挂载已有权重，不下载模型；安装补丁前核验源码 hash。[启动参数](config/serve.json) · [环境变量](config/environment.json) · [补丁清单](patches/manifest.json) · [发布状态](docs/release-plan.zh-CN.md)

```bash
curl http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Explain a[1::2]."}],"max_tokens":4096,"chat_template_kwargs":{"thinking":true,"reasoning_effort":100}}'
```

使用原生 V4.1 DSML 与数字 effort1–100，无额外 Jinja 模板。双机 [负载路由](router/README.md) 根据 running/waiting、KV 压力与会话亲和分流，保留长前缀缓存。LMCache 是可选实验，默认方案使用 GPU 前缀缓存；Engram offload 不等于 CPU KV 缓存。

## 贡献与来源

本仓库整理原生协议/streaming 修复、稀疏 prefill 边界、SM80 shape 调优、EPLB 草稿隔离、会话路由与任务级评测。SM80 模型适配、Marlin 和主要推理底座来自 wtdcode/lazymio 与 vLLM；保留上游归属，不宣称所有算子原创或优于全部社区方案。

[NOTICE](NOTICE) · [Apache-2.0](LICENSE)。不包含模型权重、私有轨迹或密钥；第三方代码与数据沿用各自许可证。
