import argparse
import subprocess
import sys
from pathlib import Path


OPERATORS = (
    "add",
    "argmax",
    "embedding",
    "linear",
    "rms_norm",
    "rope",
    "self_attention",
    "swiglu",
)


def main():
    parser = argparse.ArgumentParser(description="Run all LLAISYS operator tests")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    ops_dir = Path(__file__).resolve().parent / "ops"
    for operator in OPERATORS:
        command = [
            sys.executable,
            str(ops_dir / f"{operator}.py"),
            "--device",
            args.device,
        ]
        if args.profile:
            command.append("--profile")
        print(f"\n=== Running {operator} on {args.device} ===", flush=True)
        subprocess.run(command, check=True)

    print("\n\033[92mAll operator tests passed!\033[0m\n")


if __name__ == "__main__":
    main()
