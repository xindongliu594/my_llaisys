# Orca Continuous Batching：RTX 5090 复现与性能结果

## 1. 测试目的

本测试验证两件事：

1. 多序列 KV Cache 和 Orca 调度生成的 Token 与原单序列推理一致；
2. 在相同模型、请求数量、并发数和生成长度下，对比原 Round-Robin 调度与 Orca
   Continuous Batching 的 TTFT、端到端延迟和吞吐量。

测试重点是功能完整性与调度正确性。结果不代表针对 RTX 5090 的极限性能。

## 2. 测试环境

- 服务器：`166`（经 `jj` 跳板机连接）；
- GPU：NVIDIA GeForce RTX 5090，测试使用其中一张；
- CUDA：13；
- 模型：DeepSeek-R1-Distill-Qwen-1.5B；
- 后端：LLAISYS C/C++/CUDA；
- 调度器：Round-Robin 基线、Orca iteration-level scheduling；
- 请求数：8；
- 并发数：4；
- 每个请求最大生成长度：32 Token；
- Warmup：1 个请求；
- 采样：确定性 Argmax。

## 3. 正确性验证

运行：

```bash
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
source ~/.xmake/profile
source .venv/bin/activate
export PYTHONPATH="$PWD/python:$PWD"

python test/test_orca_infer.py \
  --model /home/liuxd/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia
```

验证结果：

```text
Single-sequence and Orca token outputs match.
Decode batch sizes: [1, 2, 2, 2, 2, 2, 2, 1]
```

这说明两个并发序列拥有独立 KV Cache，Orca 批量 Decode 与原单序列路径逐 Token
输出一致，并且请求能够在迭代中动态加入和退出。

## 4. 性能复现命令

```bash
python -m llaisys.compare_schedulers \
  --model /home/liuxd/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia \
  --requests 8 \
  --concurrency 4 \
  --max-tokens 32 \
  --warmup 1 \
  --output orca_comparison_5090_final.json
```

Round-Robin 与 Orca 在同一进程中顺序运行，加载同一份模型，并由同一个并发 HTTP
Benchmark 驱动。两轮测试均为 8/8 请求成功、0 请求失败，共生成 256 Token。

## 5. 测试结果

| 指标 | Round-Robin | Orca | 变化 |
|---|---:|---:|---:|
| 总耗时 | 2.622 s | 0.772 s | 缩短 70.6% |
| 请求吞吐 | 3.052 req/s | 10.365 req/s | 3.397× |
| 输出 Token 吞吐 | 97.651 token/s | 331.693 token/s | 3.397× |
| TTFT P50 | 38.09 ms | 24.96 ms | 降低 34.5% |
| TTFT P95 | 57.77 ms | 46.30 ms | 降低 19.8% |
| TTFT P99 | 60.79 ms | 50.56 ms | 降低 16.8% |
| 请求耗时 P50 | 1299.17 ms | 376.87 ms | 降低 71.0% |
| 请求耗时 P95 | 1313.19 ms | 401.53 ms | 降低 69.4% |
| 请求耗时 P99 | 1316.25 ms | 406.10 ms | 降低 69.1% |

Orca 记录到的 Decode Batch 最大为 4，与测试并发数一致。Batch 大小在运行中从 1
逐步增长到 4，并随着序列完成降到 3、2、1，证明这不是固定批处理，而是请求可以在
每次模型迭代边界动态加入或退出的 Continuous Batching。

## 6. 结果解释与实现边界

原 Round-Robin 路径在请求之间切换时会重新计算完整上下文；Orca 为每个序列保存独立
KV Cache，每轮只计算新 Token，并把当前活跃序列组成动态 Decode Batch。因此本组短
请求也获得了明显吞吐提升。

当前实现已经具备 Orca 的核心语义：迭代级调度、Selective Batching、独立多序列
KV Cache、动态接纳/退出和批量 Decode。Embedding、线性层、MLP 与 LM Head 已进行
批处理；由于不同序列的 KV 长度可不同，Self-Attention 目前仍在一次 Forward 内按序列
调用变长 Attention Kernel。它尚未实现 vLLM 式 PagedAttention 或融合的变长 Attention，
因此仍有进一步性能优化空间。

此外，本测试的输入和输出长度较短，样本数为 8，适合作为功能与回归证据，不应当作为
生产容量结论。进行容量规划时，应补充不同 Prompt 长度、输出长度、并发度及长时间稳态
压力测试。
