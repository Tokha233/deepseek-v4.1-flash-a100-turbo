# C256、短请求与社区对照

本页区分三个问题：服务在什么负载下能持续输出多少 token、完整 agent 是否能完成任务、与其他部署方案是否存在可复现的性能差异。吞吐包含 reasoning token。

## 并发含义

实测服务器每节点为 8×A800-SXM4-80GB，TP4×DP2、EP8、DSpark5，GPU 显存预算 0.90。每个 DP 最多运行 64 个序列，所以单节点最多同时运行 128 个序列。

- 单机客户端 C256：256 条待完成 HTTP 请求，最多 128 条同时运行，其余排队。
- 双机客户端 C256：两套完整八卡服务，可同时运行约 256 条请求。
- CUDA Graph 捕获的 token 行数包含草稿验证位置，不能当作请求并发数。

此前双机 C256 的 3,750.18 output tok/s 使用 16 张 A800；不能写成八卡 C256 的吞吐。提高客户端并发可能减少调度空隙，也可能增加冷 prefill、缓存压力和排队时间。必须同时报告实际运行数、等待数、TTFT 与 KV 抢占。

## 三种不同的测试

| 场景 | 输入与输出 | 计时范围 | 能说明什么 |
|---|---|---|---|
| 完整 SWE500 / DeepSWE | 由模型生成动作，实际执行工具与 verifier | 整个控制器运行，包括工具、环境与尾部 | 任务可用性、成绩、任务级成本 |
| 持续长历史回放 | 真实 SWE 历史；20,593–73,688 输入 token；自然 EOS、输出上限 32,768；effort100 | 固定 360 秒预热后完整 600 秒；请求完成立即补充 | 长前缀复用与服务持续输出能力；不执行生成命令 |
| 合成短请求容量 | 16 个 coding 模板，256 个 nonce；78–90 输入 token；effort100；固定输出 128 或 1,024 | 固定 60 秒预热后完整 180 秒 | 指定长度和采样下的容量；不是完整答案的质量评分 |

长历史来自 `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` 的固定 256 条历史，每条 12 轮；“66k”是数据集约 65,994 条记录，不是所有请求的上下文长度。输入中位数为 42,026.5 token。

容量测试的 `ignore_eos=True` 会强制生成到长度预算，可能跨过本应结束的位置。它方便对齐社区容量测试，但不能代替自然输出或 agent 质量测试。短输入降低 prefill 和 KV 读取成本；很短输出又可能提高调度、采样和 HTTP 开销的占比，吞吐未必随长度减少而单调增加。

## 与 keplerzip 对齐的方法

采用 [keplerzip 的原始脚本](https://github.com/keplerzip/deepseek-v4.1-flash-a100/blob/bd4fd63d1e01ee25c3ba89eb0f17dfe937461eb7/deploy/tests/dspark_benchmark.py)，固定 commit `bd4fd63d1e01ee25c3ba89eb0f17dfe937461eb7`。两份下载源码校验 SHA256 后执行；本地只修改模型 API 别名以及报告中的服务器上下文容量，保留请求生成器和计时逻辑。

| 条件 | 对齐方式 |
|---|---|
| 模型 | DeepSeek-V4.1-Flash 原生 checkpoint |
| 请求内容 | 原脚本的 LRU cache 提示，带不同早期 nonce |
| 采样 | temperature0，sampling seed42 |
| 思考 | off 与 high 分开；服务端 tokenize 确认 high 注入数字 effort75 |
| 输出 | 每条 1,024 token，ignore_eos=True |
| 并发与请求数 | C32，每档 200 条负载请求 |
| 预热与单流 | 每档先 2 次预热，再 5 次单流 |
| 负载吞吐分母 | 从提交负载到全部 200 条完成的墙钟时间，包含填充和排空 |
| 负载校验 | 每阶段服务端 generation counter 必须等于请求数×1,024；否则失败 |
| 本地重复 | off/high 固定顺序，完整重复两次；报告两次结果，不选最好一次 |

仍有明确差异：对方报告 8×A100 SXM4 NVSwitch、TP8、256K；本地为 8×A800 SXM4、TP4×DP2、EP8、512K。对方汇总未公布生成 nonce 的 seed，也未公开逐请求 JSONL，所以不能声称请求 hash 完全一致。表格能比较对齐测试方法后的实测值和作者报告值，不能隔离硬件、并行布局、版本与补丁各自的贡献。

单流 decode 估计保留原脚本的 `(completion_tokens-1)/(最后一个含文本事件时间-第一个含文本事件时间)`。一个 DSpark 流式事件可能含多个 token，因此它不是严格逐 token ITL；同时报告端到端 tok/s、TTFT 和负载 P95 延迟。

## 复现

使用现有权重的原生 encoder 生成短请求，不重新下载模型：

```bash
python benchmarks/prepare_short_workload.py /path/to/DeepSeek-V4.1-Flash workloads/short.json.gz
python benchmarks/replay_sustained.py results/short-c128 http://localhost:8083 http://localhost:8083 \
  --source workloads/short.json.gz --last-turns 1 --concurrency 128 \
  --max-tokens 1024 --ignore-eos --warmup 60 --duration 180
```

社区对齐测试实际结果：本方案 off **1,905.42 tok/s**（两次均值，1,897.65–1,913.20），high/effort75 **1,480.52 tok/s**（1,447.72–1,513.31）；keplerzip 报告为 1,379.01 / 1,039.91。相对高约 38.2% / 42.4%。测试过程无失败，所有阶段服务端 generation counter 与 200×1,024 一致。由于硬件、TP/DP 布局和上下文容量不同，这不是严格同机竞赛或软件因果证明。

社区对齐测试会下载上述固定版本的两份小型测试源码；输出目录需不存在。下载内容 hash 不符时停止。

```bash
python benchmarks/run_community_benchmark.py results/community-off http://localhost:8083 off ampere-community-20260914
python benchmarks/run_community_benchmark.py results/community-high http://localhost:8083 high ampere-community-20260914
```

在独占测试节点上顺序运行，先确认所有 DP 的 running/waiting 为零，并核对 GPU 进程。生产 agent 流量留在另一个节点；这些测试没有终止生产任务的逻辑。固定窗口结束后仅取消本轮未结束的诊断请求并等待排空。保存完整窗口与失败记录；不从运行曲线中挑峰值作为主结果。

## 可以如何陈述优势

已完成的 SWE500 原始结果 406/500、81.2%，以及最大真实输入 345,995 token，支持这套部署能处理真实长工具链路。417/500 的环境复评诊断必须另列。早期 SWE500 也不能替代所有后续补丁的整套质量回归。

若对齐的短请求测试超过某仓库报告值，可以明确写该模式、并发、输入/输出长度、硬件与相对差值。若单流更慢而聚合吞吐更高，也应同时公开。

要把“报告值更高”升级为“部署方案更优”，下一步需在同一台八卡 A800 上重跑对方固定版本，保持模型文件、请求集合、采样、缓存条件与预算一致，做交替重复；同时比较 TTFT/P95、错误率、功耗和完整任务成绩。目前不能据此宣称优于所有仓库或与官方质量等效。

更完整的 18 个项目对照见 [社区比较](community-comparison.zh-CN.md)。
