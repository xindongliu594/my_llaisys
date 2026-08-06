# LLAISYS Assignment Report

This report records the environment, reproduction procedure, and verification
results for each completed assignment milestone.

## Baseline environment and tests

Date: 2026-08-06

### Environment

- Operating system: Ubuntu Linux 24.04 container
- GPU: NVIDIA GeForce RTX 5090 (32 GB)
- Host memory: 48 GB
- CUDA toolkit: 12.8
- Python: 3.12.3
- PyTorch: 2.6.0a0+ecf3bae40a.nv25.01
- C++ compiler: GCC 13.3.0
- Build system: Xmake 3.0.9+20260806

### Build procedure

```bash
export PATH=/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH
export XMAKE_ROOT=y
xmake f -c
xmake
xmake install
```

### Baseline verification

```bash
export PYTHONPATH="$PWD/python"
python test/test_runtime.py --device cpu
```

Result: passed. The CPU runtime was detected and its runtime API test completed
successfully.

## Assignment #1: Tensor

Date: 2026-08-06

### Implementation

- Implemented host-to-tensor data loading through the active runtime API.
- Implemented contiguous-layout detection with singleton-dimension handling.
- Implemented validated, zero-copy `view`, `permute`, and `slice` operations.
- Preserved shared storage and byte offsets for tensor metadata transformations.
- Added validation for shape size, permutation uniqueness, dimensions, and ranges.

### Verification

```bash
export PYTHONPATH="$PWD/python"
python test/test_tensor.py
```

Result: passed. Loading, viewing, permuting, slicing, shape/stride metadata, and
element values all matched the PyTorch reference implementation.

## Assignment #2: CPU operators

Date: 2026-08-06

### Implementation

- Implemented Argmax with first-maximum tie behavior.
- Implemented Embedding row lookup with index validation.
- Implemented Linear with optional bias and parallel large-matrix execution.
- Implemented RMSNorm with FP32 accumulation.
- Implemented RoPE using position-dependent sine and cosine rotations.
- Implemented causal self-attention with grouped-query head mapping and a
  numerically stable softmax.
- Implemented SwiGLU activation.
- Added CPU dispatch, shape/dtype/device validation, and contiguous-layout
  checks for all operators.
- Added Float32, Float16, and BFloat16 support for every required operator.

### Verification

The repository does not contain the `test/test_ops.py` aggregate script named
in the README, so all operator test files were run individually:

```bash
export PYTHONPATH="$PWD/python"
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/${op}.py" --device cpu
done
```

Result: all public CPU operator cases passed for Float32, Float16, and
BFloat16. This includes the Linear case with input shape `(512, 4096)` and
weight shape `(4096, 4096)`. The complete Linear test finished in about nine
seconds on the initial implementation and about eight seconds after capping
parallel execution at 64 threads and partitioning work by output element. This
also enables parallel matrix-vector execution for single-token inference. An
additional local verification covered
the optional no-bias Linear path.

The CPU runtime and Assignment #1 Tensor tests were also rerun successfully as
regression checks.

## Assignment #3: Qwen2 large-language-model inference

Date: 2026-08-06

### Implementation

- Added a C ABI and ctypes bindings for model creation, destruction, weight
  access, cache reset, and inference.
- Implemented the complete Qwen2 decoder in C++, including token embedding,
  pre-normalization, Q/K/V projections, RoPE, grouped-query causal attention,
  residual connections, SwiGLU MLP blocks, final normalization, language-model
  projection, and greedy Argmax decoding.
- Implemented persistent per-layer K/V caches. Prompt tokens are processed as a
  chunk and later generation calls process one new token while reusing cached
  keys and values.
- Implemented direct memory-mapped safetensors loading. Raw BF16 bytes are
  copied into the C++ backend without using PyTorch for inference or weight
  transformation.
- Added deterministic cache reset between independent generation requests.
- Added support for optional early termination at the configured EOS token.

### Verification

The specified `DeepSeek-R1-Distill-Qwen-1.5B` model was tested with greedy
decoding:

```bash
export PYTHONPATH="$PWD/python"
python test/test_infer.py \
    --model /root/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device cpu \
    --test \
    --max_steps 128
```

Result: passed. All generated token IDs matched the Hugging Face reference
exactly through EOS. On the recorded 384-vCPU environment, Hugging Face took
74.24 seconds and LLAISYS took 51.38 seconds for the measured generation
section. A reduced two-layer model was also used to verify raw safetensors
loading, incremental cache updates, cache reset, and deterministic repeated
generation before the full-model test.

## Assignment #4: NVIDIA CUDA runtime milestone

Date: 2026-08-06

### Environment

- GPU: NVIDIA GeForce RTX 5090 (32 GiB)
- CUDA toolkit: 12.8
- CUDA architecture selected by Xmake: `sm_120`

### Implementation

- Added the optional `llaisys-device-nvidia` CUDA build target, enabled with
  `--nv-gpu=y`.
- Implemented NVIDIA device discovery and selection, device synchronization,
  CUDA stream creation/destruction/synchronization, device allocation, pinned
  host allocation, and synchronous/asynchronous copies for all four transfer
  directions.
- Added CUDA error checking with operation-specific diagnostic messages.
- Enabled Xmake device linking for the CUDA static library and emitted the
  device-link object as position-independent code so it can be linked into
  `libllaisys.so`.

### Verification

```bash
xmake f -c --nv-gpu=y --cuda=/usr/local/cuda -m release
xmake
xmake install
export PYTHONPATH="$PWD/python"
python test/test_runtime.py --device nvidia
python test/test_runtime.py --device cpu
```

Result: the CUDA build completed successfully, one NVIDIA device was detected,
and both the NVIDIA runtime test and the CPU regression test passed. CUDA
operator kernels and end-to-end NVIDIA model inference are the next Assignment
#4 milestones and are not claimed by this result.
