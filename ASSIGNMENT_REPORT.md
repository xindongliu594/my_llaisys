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
