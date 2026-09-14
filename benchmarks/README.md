# 单机测试方法

所有主指标来自单机 **8×A800-SXM4-80GB**。测速前确认无其他请求与占卡进程，每组结束后排空再执行下一组。

## 并发曲线

| 参数 | 设置 |
|---|---|
| 客户端并发 | 1、2、4、8、16、32、64、128、256 |
| 输入 | 16 个 coding 模板、256 个 nonce，78–90 token |
| 输出 | 固定 1,024 token，`ignore_eos=True`，包含 reasoning |
| 采样 | effort100，temperature1，top_p0.95，top_k20 |
| 缓存 | 每个请求独立 cache salt，不共享前缀 |
| 计时 | 60 秒预热＋完整 180 秒测量，完成一个请求立即补充 |
| 吞吐 | 测量窗口内收到的实际 output token IDs / 180 秒 |
| 延迟 | 窗口内完成请求的 TTFT、端到端延迟；保留逐请求数据 |

这是固定输出容量测试，不是完整答案的质量评分。客户端 C256 包含服务端排队；同时记录实际 running/waiting、缓存命中及 KV 抢占。所有连续 30 秒窗口均保留，SSE 消息条数不当作 token 数。

完整扫描：

```bash
python benchmarks/run_sweep.py results/sweep http://localhost:8083
```

单档复测：

```bash
python benchmarks/replay_sustained.py results/c128 http://localhost:8083 http://localhost:8083 \
  --source benchmarks/data/short-coding-effort100.json.gz --last-turns 1 \
  --concurrency 128 --max-tokens 1024 --ignore-eos --warmup 60 --duration 180
```

输出目录必须不存在。输出上限、usage 与 token ID 计数不一致会使测试失败；测量结束后仅取消本组未结束的诊断请求并确认排空。

## 消融方法

同机 C128、相同短请求协议，每档重复两次：

1. 基础配置：dense 使用 Marlin，关闭 custom EP8 AG/RS。
2. 加入 dense BF16 分派：至少 32 行时使用 BF16 GEMM。
3. 加入 custom EP8 AllGather/ReduceScatter。

三档都保留 TP4×DP2、EP8、DSpark5、CUDA Graph、FP8 KV、EPLB、稀疏 attention 与正确性补丁。比较的是这两项配置的增量效果，不是整个项目相对原始框架的总加速。报告两次实测值和均值，微测耗时不换算成整模吞吐收益。

## 长 agent 与质量评测

历史长请求测试使用真实 SWE 工具历史，输入 20,593–73,688 token、中位数 42,026.5，effort100、自然 EOS，输出上限 32,768。每条会话连续回放 12 轮，360 秒预热后测完整 600 秒；后续轮次复用前缀。工具结果来自记录，不执行新生成的命令。它与上面的短请求容量曲线分开统计。

完整 SWE500 则实际执行工具与 verifier，成绩、任务环境与配置记录见 [评测详情](../docs/agent-evaluation.md)。[证据清单](evidence-manifest.json) 包含导出结果与原始记录 SHA256。
