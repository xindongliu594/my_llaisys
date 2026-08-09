"""Command-line entry point for the LLAISYS HTTP service."""

from .server import main


if __name__ == "__main__":
    raise SystemExit(main())
