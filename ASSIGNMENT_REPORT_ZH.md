# LLAISYS 作业报告

本报告记录各项作业里程碑的运行环境、复现流程与验证结果。

## 核心贡献

- 完成作业 #0 至 #4，覆盖 CPU、NVIDIA CUDA 和 MetaX MACA 三种执行后端。
- 使用统一的上层 API、模型执行逻辑和 GPU Kernel 算法适配 NVIDIA RTX 5090 与 MetaX C500，仅将平台差异集中在 Runtime、数据类型兼容层、编译工具链和 BLAS 链接配置中。
- 在两个加速平台上完成 `DeepSeek-R1-Distill-Qwen-1.5B` BF16 端到端推理，贪心解码生成的 Token ID 均与各平台 PyTorch 参考结果一致，直到 EOS。
- 定位 MetaX FP32 Pedantic 计算模式未进入优化 GEMM 路径的问题，将大型 FP32 Linear 延迟由 16.31 ms 降低至 0.643 ms，提升约 25.4 倍，同时保持算子和端到端模型正确性。

## 完成状态

| 阶段 | CPU | NVIDIA RTX 5090 | MetaX C500 |
| --- | --- | --- | --- |
| Runtime API | 通过 | 通过 | 通过 |
| 张量功能 | 通过 | 由统一 Tensor 实现支持 | 由统一 Tensor 实现支持 |
| 八个算子、三种要求数据类型 | 通过 | 通过 | 通过 |
| 1.5B 模型端到端推理 | 通过 | 通过 | 通过 |
| 与 PyTorch 贪心解码结果一致 | 通过 | 通过 | 通过 |

## 已知限制

- 当前 Qwen2 后端按照作业要求实现贪心解码，不支持 Top-k、Top-p 等随机采样。
- 当前模型后端仅支持单设备推理，尚未实现张量并行或流水线并行。
- 公共 GitHub Actions 使用 CPU Runner，因此 GPU Runtime、算子和模型推理由对应的真实算力平台手工验证，验证命令和结果记录在本报告中。
- RTX 5090 使用完整 GPU，而 MetaX C500 使用 50% 计算资源切片，因此跨平台绝对延迟不构成等资源硬件性能比较；平台内相对于 PyTorch 的加速比更具参考意义。

## 基线环境与测试

日期：2026-08-06

### 环境

- 操作系统：Ubuntu Linux 24.04 容器
- GPU：NVIDIA GeForce RTX 5090（32 GB）
- 主机内存：48 GB
- CUDA 工具包：12.8
- Python：3.12.3
- PyTorch：2.6.0a0+ecf3bae40a.nv25.01
- C++ 编译器：GCC 13.3.0
- 构建系统：Xmake 3.0.9+20260806

### 构建流程

```bash
export PATH=/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH
export XMAKE_ROOT=y
xmake f -c
xmake
xmake install
```

### 基线验证

```bash
export PYTHONPATH="$PWD/python"
python test/test_runtime.py --device cpu
```

结果：测试通过。程序成功检测到 CPU Runtime，Runtime API 测试执行成功。

## 作业 #1：张量

日期：2026-08-06

### 实现内容

- 通过当前激活设备的 Runtime API，实现从主机到张量的数据加载。
- 实现连续内存布局检测，并正确处理大小为 1 的维度。
- 实现带参数校验的零拷贝 `view`、`permute` 和 `slice` 操作。
- 张量元数据变换后继续共享底层 Storage，并正确维护字节偏移量。
- 增加形状元素数量、排列维度唯一性、维度范围和切片范围等参数校验。

### 验证方法

```bash
export PYTHONPATH="$PWD/python"
python test/test_tensor.py
```

结果：测试通过。数据加载、视图变换、维度排列、切片、形状与步长元数据以及元素值均与 PyTorch 参考实现一致。

## 作业 #2：CPU 算子

日期：2026-08-06

### 实现内容

- 实现 Argmax，并在出现相同最大值时返回第一个最大值的位置。
- 实现 Embedding 行索引，并增加索引范围校验。
- 实现带可选 Bias 的 Linear，并支持大矩阵并行计算。
- 实现使用 FP32 累加的 RMSNorm。
- 实现基于位置的正弦与余弦旋转位置编码 RoPE。
- 实现包含分组查询头映射和数值稳定 Softmax 的因果自注意力。
- 实现 SwiGLU 激活函数。
- 为所有算子增加 CPU 分发以及形状、数据类型、设备和连续内存布局校验。
- 所有要求的算子均支持 Float32、Float16 和 BFloat16。

### 验证方法

仓库中没有 README 所述的 `test/test_ops.py` 聚合测试脚本，因此逐个运行全部算子测试：

```bash
export PYTHONPATH="$PWD/python"
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/${op}.py" --device cpu
done
```

结果：所有公开 CPU 算子测试均在 Float32、Float16 和 BFloat16 下通过。其中包括输入形状为 `(512, 4096)`、权重形状为 `(4096, 4096)` 的 Linear 大规模测试。初始实现在完整 Linear 测试中的执行时间约为 9 秒；将并行线程数量限制为 64，并按输出元素划分任务后，执行时间约为 8 秒。该优化同时使单 Token 推理中的矩阵向量计算能够并行执行。此外，还在本地补充验证了无 Bias 的 Linear 路径。

修改后重新执行了 CPU Runtime 和作业 #1 张量测试，回归测试均通过。

## 作业 #3：Qwen2 大语言模型推理

日期：2026-08-06

### 实现内容

- 增加模型创建、销毁、权重访问、缓存重置和推理所需的 C ABI 与 ctypes 绑定。
- 使用 C++ 实现完整的 Qwen2 Decoder，包括 Token Embedding、预归一化、Q/K/V 投影、RoPE、分组查询因果注意力、残差连接、SwiGLU MLP、最终归一化、语言模型输出投影和贪心 Argmax 解码。
- 为每一层实现持久化 K/V Cache。Prompt Token 以一个数据块执行 Prefill；后续生成调用每次只处理一个新 Token，并复用已经缓存的 Key 和 Value。
- 实现基于内存映射的 Safetensors 直接加载。原始 BF16 字节被直接复制到 C++ 后端，推理和权重转换过程不依赖 PyTorch。
- 在不同生成请求之间确定性地重置缓存。
- 支持遇到配置中的 EOS Token 时提前终止生成。

### 验证方法

使用指定的 `DeepSeek-R1-Distill-Qwen-1.5B` 模型执行贪心解码测试：

```bash
export PYTHONPATH="$PWD/python"
python test/test_infer.py \
    --model /root/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device cpu \
    --test \
    --max_steps 128
```

结果：测试通过。在生成过程到达 EOS 之前，所有 Token ID 均与 Hugging Face 参考结果完全一致。在记录的 384 vCPU 环境中，Hugging Face 生成阶段耗时 74.24 秒，LLAISYS 生成阶段耗时 51.38 秒。正式测试完整模型之前，还使用一个缩减为两层的模型验证了原始 Safetensors 权重加载、增量缓存更新、缓存重置和重复生成的确定性。

## 作业 #4：NVIDIA CUDA Runtime 里程碑

日期：2026-08-06

### 环境

- GPU：NVIDIA GeForce RTX 5090（32 GiB）
- CUDA 工具包：12.8
- Xmake 选择的 CUDA 架构：`sm_120`

### 实现内容

- 新增可选的 `llaisys-device-nvidia` CUDA 构建目标，通过 `--nv-gpu=y` 启用。
- 实现 NVIDIA 设备发现与选择、设备同步、CUDA Stream 创建/销毁/同步、设备显存分配、Pinned Host Memory 分配，以及四种传输方向的同步和异步内存复制。
- 增加 CUDA 错误检查，并为不同操作提供具体的诊断信息。
- 为 CUDA 静态库启用 Xmake Device Link；生成的位置无关 Device Link 对象可正确链接到 `libllaisys.so`。

### 验证方法

```bash
xmake f -c --nv-gpu=y --cuda=/usr/local/cuda -m release
xmake
xmake install
export PYTHONPATH="$PWD/python"
python test/test_runtime.py --device nvidia
python test/test_runtime.py --device cpu
```

结果：CUDA 构建成功，程序检测到一个 NVIDIA 设备，NVIDIA Runtime 测试与 CPU 回归测试均通过。CUDA 算子和 NVIDIA 端到端模型推理在下一里程碑中完成。

### NVIDIA CUDA 算子与模型推理

- 为 Add、Argmax、Embedding、RMSNorm、RoPE、Self-Attention 和 SwiGLU 实现支持 Float32、Float16 与 BFloat16 的 CUDA Kernel。
- 为 Argmax、RMSNorm 和数值稳定的分组查询因果自注意力实现 Block 级归约。
- 使用 `cublasGemmEx` 实现 Linear，支持 FP32 累加、可选 Bias，以及可使用 Tensor Core 的 Float16/BFloat16 计算。
- 在不改变默认纯 CPU 构建方式的前提下增加 NVIDIA 分发。
- 修复 Self-Attention 参考测试，确保因果 Mask 与 CUDA Tensor 创建在相同设备上。

八个算子测试在 NVIDIA 平台的三种要求数据类型下全部通过，其中包括 `(512, 4096) x (4096, 4096)` 的 Linear 大规模测试。下表列出测试中的代表性大规模用例耗时；每项数据均为同步条件下执行 100 次的平均值。

| 算子/用例 | PyTorch（ms） | LLAISYS（ms） |
| --- | ---: | ---: |
| Add，`512 x 4096`，FP32 | 0.00476 | 0.00467 |
| Argmax，`4096`，FP32 | 0.00483 | 0.00374 |
| Embedding，`50 x 4096`，FP32 | 0.01156 | 0.00277 |
| RMSNorm，`512 x 4096`，FP32 | 0.02436 | 0.00732 |
| RoPE，`512 x 4 x 4096`，FP32 | 0.11198 | 0.02121 |
| SwiGLU，`512 x 4096`，FP32 | 0.02193 | 0.00472 |
| Self-Attention，`q=5, kv=11, h=4`，FP32 | 0.11240 | 0.00392 |
| Linear，`512 x 4096 x 4096`，FP32 | 0.30145 | 0.30945 |
| Linear，`512 x 4096 x 4096`，FP16 | 0.09454 | 0.09875 |
| Linear，`512 x 4096 x 4096`，BF16 | 0.09349 | 0.09771 |

端到端验证使用完整的 BF16 `DeepSeek-R1-Distill-Qwen-1.5B` 模型：

```bash
python test/test_infer.py \
    --model /root/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device nvidia \
    --test \
    --max_steps 128
```

结果：测试通过。生成的 Token 序列在到达 EOS 之前与 Hugging Face 完全一致。在同一张 RTX 5090 上，Hugging Face 生成阶段耗时 1.72 秒，LLAISYS 耗时 0.44 秒，实现 3.91 倍加速。完成 CUDA 修改后，CPU Runtime、张量测试和全部 CPU 算子测试再次通过；使用 `--nv-gpu=n` 的全新纯 CPU 构建也成功完成。

### 平台状态

- NVIDIA CUDA：Runtime、全部要求的算子以及端到端模型推理均已完成，并在 RTX 5090 上验证通过。
- MetaX MACA：Runtime 和全部要求的算子已在 C500 上完成并验证；端到端模型推理结果记录在下一里程碑中。

## 作业 #4：MetaX MACA Runtime 与算子里程碑

日期：2026-08-07

### 环境

- 加速器：MetaX C500，64 GiB 物理显存
- 容器配额：一个切分后的 GPU，50% 计算资源和 32,000 MiB 显存配额
- 驱动：3.8.30
- MACA：3.3.0.15
- 编译器：mxcc 1.0.0，原生 `XCORE1000` 目标
- 参考框架：支持 MetaX 3.3.0.2 的 PyTorch 2.8.0

### 实现内容

- 新增与 NVIDIA 构建互斥的 `--metax-gpu=y` Xmake 配置，同时保留纯 CPU 和 `--nv-gpu=y` NVIDIA 构建方式。
- 增加使用 mxcc 的工具链，通过 `-offload-arch native` 编译原生 MACA Kernel。
- 增加轻量源码兼容层，将 LLAISYS CUDA Runtime API 映射到 `mcRuntime`、将 Float16/BFloat16 类型映射到 MACA 类型，并将 cuBLAS 调用映射到 `mcBLAS`。
- 通过 MetaX 专用编译入口复用已经验证的 CUDA Kernel 算法，避免为每个算子维护相互分叉的实现副本。
- 将 `libllaisys.so` 与 `libmcruntime`、`libmcblas` 链接，并写入 MACA 库的 RPath。

### 构建与验证

```bash
export XMAKE_ROOT=y
xmake f -c -m release --nv-gpu=n --metax-gpu=y
xmake build -j 16
xmake install
export PYTHONPATH="$PWD/python:$PWD"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH"
python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
```

结果：MACA 构建成功，程序检测到一个加速器；Runtime 和八个算子均在 Float32、Float16 与 BFloat16 下通过测试，其中包括大规模 `(512, 4096) x (4096, 4096)` Linear 用例。移植加速器后端之前，还在相同容器中执行了全新的纯 CPU 构建，Runtime、张量和所有 CPU 算子测试均通过，证明作业 #0 至 #2 可以在新平台上正确复现。

### 性能验证

下表中每项数据均为预热 10 次后，同步启动 100 次所得的平均值。大型 Linear 测试包含一个 MetaX 专用兼容映射，将 CUDA 的 FP32 Pedantic 模式映射为 mcBLAS 标准 FP32 累加模式。该模式仍可通过数值测试，并将 LLAISYS FP32 Linear 延迟从 16.31 ms 降低到 0.643 ms。

| 算子/用例 | MetaX PyTorch（ms） | LLAISYS（ms） |
| --- | ---: | ---: |
| Add，`512 x 4096`，FP32 | 0.02288 | 0.03455 |
| Argmax，`4096`，FP32 | 0.01301 | 0.01456 |
| Embedding，`50 x 4096`，FP32 | 0.05417 | 0.01907 |
| RMSNorm，`512 x 4096`，FP32 | 0.08539 | 0.03575 |
| RoPE，`512 x 4 x 4096`，FP32 | 0.35895 | 0.17130 |
| SwiGLU，`512 x 4096`，FP32 | 0.09300 | 0.03453 |
| Self-Attention，`q=5, kv=11, h=4`，FP32 | 0.14996 | 0.01465 |
| Linear，`512 x 4096 x 4096`，FP32 | 0.60225 | 0.64319 |
| Linear，`512 x 4096 x 4096`，FP16 | 0.14869 | 0.17653 |
| Linear，`512 x 4096 x 4096`，BF16 | 0.14974 | 0.17651 |

### MetaX 端到端模型推理

从原始 Safetensors 权重加载完整 BF16 `DeepSeek-R1-Distill-Qwen-1.5B` 模型，并使用贪心解码进行测试：

```bash
export MACA_PATH=/opt/maca
python test/test_infer.py \
    --model /root/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device nvidia \
    --test \
    --max_steps 128
```

结果：测试通过。LLAISYS 生成的 Token ID 在到达 EOS 之前与 MetaX PyTorch 参考结果完全一致。在 50% C500 切分资源上，PyTorch 生成阶段耗时 2.75 秒，LLAISYS 耗时 1.00 秒，实现 2.75 倍加速。因此，作业 #4 已在 NVIDIA RTX 5090 和 MetaX C500 两种 CUDA/类 CUDA 平台上全部完成。
