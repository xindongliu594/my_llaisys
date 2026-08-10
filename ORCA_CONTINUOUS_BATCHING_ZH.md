# LLAISYS Orca Continuous Batching 设计与验证

## 1. 目标

原 `RoundRobinScheduler` 虽然逐 Token 轮转请求，但底层只有一套 KV Cache。每次
切换请求都要用完整上下文重新 Prefill，因此它是调度功能原型，不是真正的
Continuous Batching。

本实现增加 `OrcaScheduler` 和 Qwen2 多序列 C API，使每个请求拥有独立、持久的
KV Cache，并在一次 Decode 调用中处理当前所有活跃序列。

## 2. 迭代级调度

每次 Orca 迭代执行：

1. 从优先级请求池接纳新请求，直到达到活跃请求上限；
2. 对有限数量的新请求执行 Prefill；
3. 将已经完成 Prefill 的所有请求组成动态 Decode Batch；
4. 每个序列生成一个 Token；
5. EOS、Stop、长度到达、取消或超时的序列立即退出并释放 KV Cache；
6. 空出的活跃位置在下一次迭代接纳等待请求。

这对应 Orca 的 iteration-level scheduling 与 selective batching。Prefill 不会要求
所有输入长度一致，Decode Batch 的成员也可以在任意迭代加入或退出。

## 3. 多序列 KV Cache

新增 C API：

- `llaisysQwen2SequenceCreate`；
- `llaisysQwen2SequenceDestroy`；
- `llaisysQwen2SequenceReset`；
- `llaisysQwen2SequenceInfer`；
- `llaisysQwen2SequenceInferSample`；
- `llaisysQwen2BatchInfer`；
- `llaisysQwen2BatchInferSample`。

每个序列状态独立保存：

- `cache_len`；
- 每层 Key Cache 和 Value Cache；
- Token 历史；
- 随机采样状态。

KV Cache 按“实际 Prompt 长度 + 最大输出长度”分配，而不是为每个请求都按照模型的
`max_position_embeddings` 分配，从而避免短请求占用完整长上下文显存。

## 4. Batched Decode

Decode Batch 中的 Token 会一起执行 Embedding、Q/K/V Linear、RoPE、输出投影、MLP
和 LM Head。由于不同序列的 KV 长度可以不同，当前 Self-Attention 在同一个模型
Forward 内按序列调用各自的变长 Attention Kernel，其余主要矩阵计算是批处理的。

这已经消除了旧方案的完整上下文重复 Prefill，并实现了真实的多序列持久 KV Cache
和动态 Decode Batch。后续性能优化可以继续把变长 Attention Kernel 融合为一个
Paged/Variable-Length Attention Kernel。

## 5. 启动

HTTP 服务默认使用 Orca 调度器：

```bash
python -m llaisys.serve \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia \
  --scheduler orca \
  --max-active-requests 32 \
  --max-prefill-per-iteration 1
```

如需与旧基线对比，可设置 `--scheduler round-robin`。

## 6. 验证层次

1. 零层迷你 Qwen2 验证两个独立 Sequence Cache 和 Batched Decode C API；
2. Fake Batched Model 验证动态加入、退出、Batch 大小变化和 Cache 释放；
3. CPU 全套回归验证旧单序列接口不变；
4. NVIDIA 服务器验证 CUDA 编译、真实 1.5B 模型一致性和并发性能；
5. 使用 `python -m llaisys.benchmark` 对比 Round-Robin 与 Orca 的 TTFT、吞吐量和
   P50/P95/P99。
