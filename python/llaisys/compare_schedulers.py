"""Compare Round-Robin recomputation with Orca continuous batching."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from transformers import AutoTokenizer

from .benchmark import run_benchmark
from .libllaisys import DeviceType
from .models import Qwen2
from .server import OpenAIAPIServer
from .serving import ChatService, OrcaScheduler, RequestPool, RoundRobinScheduler


def _compact(summary: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in summary.items() if key != "results"}


def compare_schedulers(
    model_path: str,
    device: DeviceType,
    prompt: str,
    requests: int,
    concurrency: int,
    max_tokens: int,
    warmup: int,
) -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = Qwen2(model_path, device)
    model_id = "llaisys-orca-comparison"
    results: Dict[str, object] = {}

    for name in ("round_robin", "orca"):
        pool = RequestPool(max_pending_requests=max(requests, concurrency) + 1)
        if name == "orca":
            scheduler = OrcaScheduler(
                model,
                request_pool=pool,
                max_active_requests=concurrency,
                max_prefill_per_iteration=1,
            )
        else:
            scheduler = RoundRobinScheduler(
                model,
                request_pool=pool,
                max_active_requests=concurrency,
            )
        chat = ChatService(scheduler, tokenizer)
        server = OpenAIAPIServer(
            chat, model_id=model_id, host="127.0.0.1", port=0
        )
        server.start()
        host, port = server.address
        try:
            summary = run_benchmark(
                endpoint=f"http://{host}:{port}",
                model=model_id,
                prompt=prompt,
                requests=requests,
                concurrency=concurrency,
                max_tokens=max_tokens,
                warmup=warmup,
                timeout=600.0,
            )
            compact = _compact(summary)
            compact["server_metrics"] = server.metrics.snapshot()
            if isinstance(scheduler, OrcaScheduler):
                compact["decode_batch_sizes"] = list(
                    scheduler.decode_batch_sizes
                )
            results[name] = compact
        finally:
            server.stop(graceful=True, timeout_seconds=600.0)
        del server
        del chat
        del scheduler
        gc.collect()

    baseline = results["round_robin"]
    orca = results["orca"]
    results["speedup"] = {
        "request_throughput": (
            orca["requests_per_second"] / baseline["requests_per_second"]
        ),
        "output_token_throughput": (
            orca["output_tokens_per_second"]
            / baseline["output_tokens_per_second"]
        ),
        "ttft_p50_reduction": (
            baseline["ttft_seconds"]["p50"] / orca["ttft_seconds"]["p50"]
        ),
    }
    del model
    gc.collect()
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "nvidia"), default="nvidia")
    parser.add_argument("--prompt", default="Explain KV Cache briefly.")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    device = (
        DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU
    )
    results = compare_schedulers(
        model_path=args.model,
        device=device,
        prompt=args.prompt,
        requests=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
    )
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
