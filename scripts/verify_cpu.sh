#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export XMAKE_ROOT=y
xmake f -c -m release --nv-gpu=n --metax-gpu=n
xmake build
xmake install
python -m pip install ./python

python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/test_ops.py --device cpu

model_args=()
if [[ $# -gt 0 ]]; then
    model_args=(--model "$1")
fi
python test/test_infer.py "${model_args[@]}" --device cpu --test
