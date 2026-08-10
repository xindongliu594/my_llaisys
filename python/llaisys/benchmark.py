"""Concurrent SSE benchmark client for an OpenAI-compatible LLAISYS server."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _one_request(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> Dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: Optional[float] = None
    token_events = 0
    request_id = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                request_id = request_id or chunk.get("id")
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                token_events += 1
        finished = time.perf_counter()
        return {
            "ok": True,
            "request_id": request_id,
            "ttft_seconds": (
                first_token_at - started if first_token_at is not None else None
            ),
            "duration_seconds": finished - started,
            "token_events": token_events,
            "error": None,
        }
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        finished = time.perf_counter()
        return {
            "ok": False,
            "request_id": request_id,
            "ttft_seconds": None,
            "duration_seconds": finished - started,
            "token_events": token_events,
            "error": str(error),
        }


def run_benchmark(
    endpoint: str,
    model: str,
    prompt: str,
    requests: int = 16,
    concurrency: int = 4,
    max_tokens: int = 128,
    warmup: int = 1,
    timeout: float = 300.0,
) -> Dict[str, object]:
    if requests <= 0 or concurrency <= 0 or max_tokens <= 0 or warmup < 0:
        raise ValueError("requests, concurrency and max_tokens must be positive")

    for index in range(warmup):
        _one_request(
            endpoint, model, f"{prompt}\nWarmup {index}", max_tokens, timeout
        )

    wall_started = time.perf_counter()
    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _one_request,
                endpoint,
                model,
                f"{prompt}\nRequest {index}",
                max_tokens,
                timeout,
            )
            for index in range(requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - wall_started

    successful = [result for result in results if result["ok"]]
    ttfts = [
        float(result["ttft_seconds"])
        for result in successful
        if result["ttft_seconds"] is not None
    ]
    durations = [float(result["duration_seconds"]) for result in successful]
    output_tokens = sum(int(result["token_events"]) for result in successful)
    summary: Dict[str, object] = {
        "endpoint": endpoint,
        "model": model,
        "requests": requests,
        "concurrency": concurrency,
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "wall_seconds": wall_seconds,
        "requests_per_second": len(successful) / max(wall_seconds, 1e-9),
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / max(wall_seconds, 1e-9),
        "ttft_seconds": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "request_duration_seconds": {
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
        },
        "errors": [result["error"] for result in results if not result["ok"]],
        "results": results,
    }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark LLAISYS HTTP serving")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain KV Cache briefly.")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    summary = run_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        prompt=args.prompt,
        requests=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        timeout=args.timeout,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
