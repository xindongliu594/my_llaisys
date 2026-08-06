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
