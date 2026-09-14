# 开源定位与社区对照（2026-09-14）

建议主标题：**DeepSeek-V4.1-Flash on Ampere: long-context coding agents, measured end to end**。

中文表达：**让 A800 上的 DeepSeek-V4.1-Flash 真正跑完编码 agent：可复现部署、持续吞吐与任务级评测证据。**

重点应是有边界的工程证据：原生checkpoint的SM80部署 + 原生DSML/thinking协议 + 真实长历史持续回放 + 执行工具的完整任务评分。不要用“比官方快”“全球最快”“与官方无损”作为标题。仓库slug含A100，但实测硬件明确写8×A800-SXM4-80GB；A100是移植目标。

## 检索范围与可复查来源

检索6组GitHub repository查询，读取18个仓库的固定commit README，进一步读取重点项目的benchmark脚本、机器可读结果、评测台账和性能报告。另核对官方SGLang cookbook、dsv4.1分支、vLLM recipe与本地官方模型卡；精确链接与hash见 [official-sources.json](research/official-sources.json)。记录位于 [sources.json](research/sources.json)、[artifacts.json](research/artifacts.json) 与 `research/search-*.json`。这是本次检索范围，不声称穷尽所有开源方案。

下面是**作者报告值**，我们没有在其硬件上独立复现。硬件、模型、上下文、采样、cache和分母不同，所以这张表用于理解各项目的证据与方法，不按tok/s排排行榜。

| 仓库/固定版本 | 模型与硬件 | 实现 | 报告指标 | 解释与边界 |
|---|---|---|---|---|
| [shi3z/deepseekv4.1-A100-custom](https://github.com/shi3z/deepseekv4.1-A100-custom/blob/00f9e878f26746bfc2f0248352f18452aa16e9d8/README.md) (`00f9e878f267`) | V4.1；A100 PCIe，4/7/8 GPU | 自写 Triton/CUDA runtime；FP8/FP4 寄存器解码到 BF16 | 4 GPU：C1 K5 93；C64 plain 930；C256 plain 1,972 tok/s | 离线 decode、先逐条 prefill 再计时；C256 循环使用 64 条提示；不是持续 agent 服务。README 尾部与前部 MTP 状态不一致，不能当完备服务验收。 |
| [keplerzip/deepseek-v4.1-flash-a100](https://github.com/keplerzip/deepseek-v4.1-flash-a100/blob/bd4fd63d1e01ee25c3ba89eb0f17dfe937461eb7/README.md) (`bd4fd63d1e01`) | V4.1；8×A100 SXM4 NVSwitch | vLLM backport；TP8；DSpark5；256K | C32 off 1,379.01 / high 1,039.91；high C1 decode 109.95；另一次 C56 1,334.67 | 短提示、1024 输出、200 请求；操作者回传汇总，逐请求原始 JSONL 未公开。高思考不等于数字 effort100。 |
| [Hakureirm/dsv4-on-ampere](https://github.com/Hakureirm/dsv4-on-ampere/blob/abfd51e8b961cb5e89b73e2933d0703e8a43cfd5/README.md) (`abfd51e8b961`) | V4 Flash；8×A800 | SGLang TP8 + Marlin + SM80 patch | C1 53.1→110.6；聚合 455；GSM8K 96.21% / DSpark 96.13% | 确实有全 1319 题 GSM8K 与 9/9 needle，不能说社区只测速度。但不是 V4.1 SWE500 对照。 |
| [yobo2u/DeepSeek-V4-Flash-0731-A100](https://github.com/yobo2u/DeepSeek-V4-Flash-0731-A100/blob/d4e670184b246a9e40f8aa6b166aa2f7a8dac0c7/README.md) (`d4e670184b24`) | V4-0731；8×A800 | SGLang 0.5.16 + INT8 MoE monkeypatch | C1 decode约217；C16均值约1,232；配置C峰值1,334 | 30组900请求；脚本反复扩展相同上下文、每组同提示预热5次、ignore_eos=True。128K输入组不等于真实冷 agent TTFT，更非1M实测。 |
| [yaleyoou/deepseek-v4-a100-sglang-v0516](https://github.com/yaleyoou/deepseek-v4-a100-sglang-v0516/blob/dd8cdb87ace3b053fcf4f20d898e048a4b33dc8a/README.md) (`dd8cdb87ace3`) | V4-0731；README 4×A100，沿用 benchmark 页为TP8 | BF16 dense + packed FP4到INT8 kernel | benchmark页：C64 1,755.82；C1 71.92 output tok/s | random-ids 1024-in/1024-out，64个请求一批；README与旧benchmark页的硬件/运行配置不同，必须逐页辨认。 |
| [wwwadx/deepseek-v4-flash-a100-deploy](https://github.com/wwwadx/deepseek-v4-flash-a100-deploy/blob/9ce8e194141aac7d24ff487e425e7aa5a0ad356f/README.md) (`9ce8e194141a`) | V4-0731；8×A100 PCIe，无NVLink | vLLM + API与Graph补丁 | 约40单流；C32约1,070；配置1M | 部署与故障记录完整；文中把最大长度称为逐请求KV预留不宜泛化到vLLM动态分页分配。 |
| [yunyinbanfu/deepseek-v4-flash-0731-sm80](https://github.com/yunyinbanfu/deepseek-v4-flash-0731-sm80/blob/1a37df917f43954279e03b4116c3f30562c979cd/README.md) (`1a37df917f43`) | V4-0731；CMP170HX/A800/A100多组 | vLLM SM80；PP4；DSpark | PP4 C64 712.8；三内容单流98.1；1.04M TTFT约544–550s，decode35.6 | 还报告405轮累计1,002,852 token无崩溃；不同表硬件和计时范围需确认，长文可用性并非我们的独占贡献。 |
| [keplerzip/deepseek-v4-flash-a100](https://github.com/keplerzip/deepseek-v4-flash-a100/blob/b9cc033d555f1a0b233194f2ddb1739cbb2ea289/README.md) (`b9cc033d555f`) | V4-0731；8×A100 | TP8；无草稿与DSpark7双方案 | 双启动配置、离线打包和验收材料 | 应与同作者V4.1仓库分开；不能跨模型拿旧数字补当前V4.1表。 |
| [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt/blob/65f2ac21744ba3539f5ace9a8ab9707eff7a235b/README.md) (`65f2ac21744b`) | V4.1；1/2×RTX PRO6000 +4×DGX Spark | 自写异构runtime；本地DSpark；token级前缀复用 | 2 RTX：加权内容decode79.33；C16代码1,181.49；混合309.06；prefill最高8,454 | 同一系统代码、数数、混合负载可差数倍；3次样本/中位数/原始证据归档值得借鉴。硬件完全不同。 |
| [raullenchai/twinspark](https://github.com/raullenchai/twinspark/blob/dfa7f738964e22116771bd742d21a0dd21b97e5b/README.md) (`dfa7f738964e`) | V4-0731；2×DGX Spark | vLLM TP2 RoCE；DSpark7/3 | 单流74.8；多agent配置C4 105.7；prefill1,953 | 报告agent上下文接受率约75%→40%、FP8 KV较FP4 KV稳定；这些是该模型/平台的经验，不是V4.1普遍结论。 |
| [elsung/dgx-spark-deepseek-v4-flash](https://github.com/elsung/dgx-spark-deepseek-v4-flash/blob/99d1889062c11b81e51ce4813cbfc28aed4f1a4d/README.md) (`99d1889062c1`) | V4 Flash；2×DGX Spark | vLLM TP2 | 单流约41；C32约350；prefill约1,785 | 15分钟稳定性及内存泄漏修复复验；硬件与旧模型不同。 |
| [r0b0tlab/deepseek-v4-flash-nvfp4-gb10-benchmark](https://github.com/r0b0tlab/deepseek-v4-flash-nvfp4-gb10-benchmark/blob/cd0b670fc84be39c76065d72ac1f58244e702f32/README.md) (`cd0b670fc84b`) | V4 Flash；2×DGX Spark | 原生Blackwell FP8/MXFP4、MoE padding、RoCE | C1 38.4；65K配置C16 144.6；通信424→22µs | 优化前后还叠加驱动回退问题；通信微测和token/s不能直接相乘。仓库名称含nvfp4，正文实际说明native FP8路径。 |
| [Infatoshi/dsv4-flash-2x-rtxpro6000s](https://github.com/Infatoshi/dsv4-flash-2x-rtxpro6000s/blob/c726dc94538b16e8a72e2a76860050b08f0e602e/README.md) (`c726dc94538b`) | V4-0731 NVFP4重打包；2×RTX PRO6000 | vLLM + Marlin + DSpark3 + indexer | C1 202.7；504K prefill4,339，decode66；演示热cache242 | SM120不是SM100更不是SM80；模型、量化、单流/热cache条件不同；有混合批短请求越界风险说明。 |
| [ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000](https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000/blob/79bbc001db042fc85dc2b15e4175210eaec3e78e/README.md) (`79bbc001db04`) | V4-0731；4×RTX PRO6000 SM120 | SGLang；DP attention；DSpark；调度/DSML修复 | 随机短请求C128 3,560；8192-in/512-out C128仅723 | 公开SWA预算、KV污染、DSpark深度、流式DSML等多项PR；强烈说明短prompt高吞吐不代表agent服务。 |
| [humanrouter/deepseek-v4-flash-gx10-speed-recipe](https://github.com/humanrouter/deepseek-v4-flash-gx10-speed-recipe/blob/43e13fc221dcd1e32851495c5efbe423a0e96090/README.md) (`43e13fc221dc`) | V4 Flash；GX10/GB10 | chunk与B12X调优；草稿6→5 | 8K prefill1,466.8→1,886.49；32K 1,429.52→1,867.14；代码decode55.91不变 | 3次冷请求中位数，对照重跑5次；prefill收益与decode无收益分开记录。 |
| [devteapot/deepseek-v4-flash-tb21-bounty327](https://github.com/devteapot/deepseek-v4-flash-tb21-bounty327/blob/5229e75cc34b00bf67a8ae75ac7242ee8fb1d435/README.md) (`5229e75cc34b`) | V4-0731；Spark runtime；Daytona任务环境 | Harbor0.22 terminus-2；256K；输出8192 | Terminal-Bench2.1：121/178=67.98%，89题×2 | 并发1；macro mean不是best-of-two；保留180行，2个infra-invalid与替补关联；封存495MB release证据。 |
| [Monking-peng/terminal-bench-2-1-reproduction](https://github.com/Monking-peng/terminal-bench-2-1-reproduction/blob/1c1888361c075f77cb11dd83786eeebd1baee913/README.md) (`1c1888361c07`) | TB2.1工程；Codex GPT5.5 demo | Oracle/Nop/agent三对照；Harbor | 单题agent通过；89题Oracle 79pass/5fail/5异常；Nop 0pass/86fail/3异常 | 完整正式89×5 agent评测未完成；可借鉴环境控制与证据结构，不能把Oracle分数算模型成绩。 |
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport/blob/563692c466aa74b457c839aa4860b69692443149/README.md) (`563692c466aa`) | V4.1 SM80底座；master-v013 | 原始模型适配、Marlin、attention和DSpark支持 | 该分支未提供与本项目同协议的V4.1 A800 agent吞吐 | 当前HEAD 563692c466aa；最近提交以README、GLM与offloader为主。不能仅因更新就推断DeepSeek加速。 |

## 和 shi3z 的具体区别

`shi3z` 的主要贡献是自己写SM80数值算子和runtime：FP4专家布局重排、寄存器内反量化、grouped GEMM、copy-engine专家通信、按活跃上下文选择K5/K3/关闭草稿。其专家带宽1.02→1.46TB/s以及dispatch148→20µs均是算子级证据。

其1,972 tok/s值得研究，但 `dsv41/mtp_run.py` 先逐个prefill，然后开始decode计时；超过64个会话时循环使用提示文件。README又注明硬件与别的任务共享，且尾部仍保留“verification未实现”的旧文字，和前文已实现的batched verification矛盾。因此应按具体脚本/commit理解，不能把1,972当作HTTP持续agent吞吐，更不能简单乘两套副本宣传。

我们不把上游SM80能力归为原创。这里的贡献是基于vLLM backport完成可部署的长agent链路：数值reasoning effort、跨工具轮reasoning保留、DSML streaming修复、稀疏prefill边界、专家负载均衡时草稿隔离、SM80特定shape调优、DP亲和路由和任务环境可靠性。SWE500提供了真实工具执行与verifier证据；这比单条问答smoke更接近用户需求。它也不等于模型分布严格无损的数学证明。

## 当前项目的数据应该如何摆放

| 层次 | 已核对的结果 | 能证明什么 |
|---|---|---|
| 完整agent任务 | SWE-bench Verified：406/500=81.2%原始；417/500=83.4%环境复评诊断 | 500个编码任务链路可工作；复评分必须另列 |
| 长任务成本 | SWE500：21,325,164输出，16,382,537 reasoning，29,964模型轮，41,043工具调用；最大输入345,995 token | 不止短prompt生成；存在真实长工具历史 |
| 完整运行吞吐 | SWE500 760.72 output tok/s；旧DeepSWE双机完整运行1,108.67 | 包含工具、排队、评分、故障与排空；不等同GPU满载能力 |
| 持续回放吞吐 | round9单机C128 1,923.33；双机C256 3,750.18；均固定360秒预热+600秒窗口 | 持续供应真实历史下的服务能力；工具结果回放，不执行生成命令 |
| SGLang本地对照 | round9 C64最佳已完成候选408.07；初始138.24；vLLM所选候选同3轮回放612.69 | SGLang已经端到端测过；是具体补丁/配置比较，不是框架普遍优劣 |
| 短问答正确率 | GPQA Diamond176/198=88.89%；GSM8K strict1013/1319=76.80%，flexible1260/1319=95.53% | 有可评分的问答结果；格式抽取与输出上限仍影响分数 |
| DeepSWE新协议 | round11 effort75/100正在续跑 | 不能把部分完成比例写成最终成绩 |
| Terminal-Bench2.1 | 镜像准备/预检阶段，正式89题未完成 | 目前没有可发布的正式Pass@1 |

证据按层次存放在 [benchmarks/results](../benchmarks/results)，含字段白名单导出与源文件hash。历史SWE500、旧DeepSWE、round9吞吐和round11新评测分别保留配置标识。**不能将早期SWE500分数直接标注为当前round9全部新增算子已完成质量回归。**

本次远端独立C128复测新增：**1,835.34 output tok/s**（600秒、1,101,203 token客户端/服务端完全一致），平均运行126.53，prefix hit92.49%，零KV抢占，20个30秒窗口全部保留；66,310-token原生工具检索四次均通过，冷TTFT6.286–6.325秒、热0.426–0.438秒。见 [本次收据](../benchmarks/results/round13/remote-c128-repeat1/measurement-summary.json)。

## 是否可以宣传“基本无损”

暂时不能。官方V4.1-Flash模型卡GPQA是90.9%，本地88.89%，差约−2.01个百分点，198题Wilson95%区间约83.75–92.55%；区间覆盖官方值不等于通过等效性检验。6个回答达到65,536输出上限，仍计入分母。应在相同prompt/输出预算上比较基础runtime、当前补丁runtime与官方API，再对成对任务结果做检验。

官方DeepSWE mini-SWE为74.2%，使用1M上下文、最大推理强度、每题8个sample；本地512K、每题1个计划attempt并经历操作中断恢复，不能直接当严格复现。官方TB2.1 DSH Minimal为90.6%、mini-SWE为90.3%，每题3个sample且无网络；我们的正式TB还没跑完。官方GSM8K93.0是base模型8-shot，不是当前instruct effort100的直接基线。

后续无损声明应提前固定可接受差值，例如任务分数下降不超过2个百分点，再固定同题、同seed、同预算、同scaffold和环境的配对实验；同时报告工具格式错误、重复循环、length终止、基础设施失败。只展示抽样smoke或事后选择最好一次成绩不够。

## 优化借鉴与优先级

1. **官方adaptive verification/按负载调整DSpark。** SGLang高吞吐recipe关闭DSpark；vLLM官方NVIDIA recipe使用概率草稿、block rejection、adaptive verification。当前SM80服务保持K5和统一验证，尚未等价启用官方自适应路径。先做同历史、同采样下K0/K3/K5对照，再检查SM80 attention builder是否支持变长Graph元数据。不能只翻开flag。
2. **通信等待与专家倾斜。** 当前已有本地EP8 BF16 custom AG/RS和target EPLB，草稿专家隔离；未用跨节点EP或DeepEP。两台机器是独立完整服务，通过HTTP分流，所以RoCE 19倍all-reduce微测不能直接套到我们的双机路由。rank等待包含各卡到达collective时间差；降低等待还需要均衡专家/attention工作量。
3. **indexer/稀疏attention。** 本地已有candidate-only MQA、query分片、prefill索引融合、H16 Q192/384 tile16。借鉴社区布局与减少无效candidate扫描，必须保留同样TopK、mask、长度边界和共享KV规则；不能用减少top-k或近似路由换取“无损加速”。
4. **SM80算子布局。** `shi3z` 的专家tile布局和dispatch可作为独立微测对象，但Marlin已有重排和TensorCore路径。只有在相同层shape、batch、专家分布下优于Marlin且通过数值/Graph验证，才值得整合。BF16 dense阈值32已经在当前方案里。
5. **按cache与实际负载路由。** 亲和保留长前缀，同时按running、waiting、KV压力与服务端吞吐估计择机迁移。不要按显存已用百分比判断算力余量。异机工具回合迁移会产生冷prefill，应报告迁移次数、命中率、TTFT P95和每节点实际负载。

round9里通信profile约18.637→15.685ms、目标Graph75.60→72.79ms；稀疏partial kernel15.688→13.299ms是另一个单独候选。持续C64 baseline1,673.63与combined1,682.01几乎持平；本机C128 sparse-only1,827.23与combined1,923.33为一次约5.3%差异。不能将两个kernel收益相加，也不能用自然输出单次差异宣称稳定加速倍数。

## 怎么开源

建议先发源码预览版：固定运行底座digest、23个修改文件的before/after hash、启动配置、路由、native工具smoke、持续回放程序、脱敏分数和对照报告。镜像等到完整Docker重建与重启验收后再打release，不把当前实验机器的挂载rootfs当已发行镜像。

Git保留小体积结果、逐任务元数据、原始receipt hash与复现脚本；大体积profile放Release附件并附SHA256。模型权重、密钥、内网端点、私有完整prompt/trajectory和缓存不入仓库。现在不分发完整回放语料，因此必须直说：公开脚本可在自有语料重跑；复现完全相同数字仍需要有权取得相同数据与固定输入token序列。hash证明关联，不能替代可访问原始证据。

许可证沿用已整理代码的Apache-2.0，保留vLLM/社区/SPDX来源。SM80底座归wtdcode/lazymio；上游PR、改写的SM80算子、运行配置、评测工程分别写贡献，避免包装成从零实现框架。

首屏用“任务分数 + 持续吞吐 + 真实长输入长度”三组证据；图中用不同图表呈现服务持续吞吐与完整agent运行吞吐。不要把不同硬件/不同模型社区数字与本机结果画成横向竞速榜。开源后欢迎A100志愿者按同一脚本提供硬件拓扑、镜像digest、完整receipt再增加A100实测标签。
