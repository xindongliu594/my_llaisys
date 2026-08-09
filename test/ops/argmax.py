import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import (
    random_tensor,
    check_equal,
    benchmark,
    zero_tensor,
    llaisys_device,
)


def torch_argmax(max_idx, max_val, vals):
    torch.max(vals, keepdim=True, dim=-1, out=(max_val, max_idx))


def test_op_argmax(
    shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    vals, vals_ = random_tensor(shape, dtype_name, device_name)
    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)
    max_val, max_val_ = zero_tensor((1,), dtype_name, device_name)

    torch_argmax(max_idx, max_val, vals)
    llaisys.Ops.argmax(max_idx_, max_val_, vals_)

    assert check_equal(max_val_, max_val, strict=True)
    assert check_equal(max_idx_, max_idx, strict=True)

    if profile:
        benchmark(
            lambda: torch_argmax(max_idx, max_val, vals),
            lambda: llaisys.Ops.argmax(max_idx_, max_val_, vals_),
            device_name,
        )


def test_first_maximum_tie(device_name="cpu"):
    print("   deterministic first-maximum tie")
    vals, vals_ = random_tensor((4,), "f32", device_name)
    vals.copy_(torch.tensor([1.0, 3.0, 3.0, 2.0], device=vals.device))
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        vals_.data_ptr(),
        vals.data_ptr(),
        vals.numel() * vals.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)
    max_val, max_val_ = zero_tensor((1,), "f32", device_name)

    torch_argmax(max_idx, max_val, vals)
    llaisys.Ops.argmax(max_idx_, max_val_, vals_)

    assert int(max_idx.item()) == 1
    assert check_equal(max_val_, max_val, strict=True)
    assert check_equal(max_idx_, max_idx, strict=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [(4,), (4096,)]
    testDtype = ["f32", "f16", "bf16"]
    print(f"Testing Ops.argmax on {args.device}")
    for shape in testShapes:
        for dtype_name in testDtype:
            test_op_argmax(shape, dtype_name, args.device, args.profile)
    test_first_maximum_tie(args.device)

    print("\033[92mTest passed!\033[0m\n")
