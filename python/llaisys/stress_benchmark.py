"""Mixed-workload, cancellation, overload, and leak benchmark for LLAISYS."""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .benchmark import percentile


def _prometheus_metrics(endpoint: str, timeout: float) -> Dict[str, float]:
    with urllib.request.urlopen(
        endpoint.rstrip("/") + "/metrics", timeout=timeout
    ) as response:
        lines = response.read().decode("utf-8").splitlines()
    metrics = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name, value = line.split(None, 1)
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def _cancel(endpoint: str, request_id: str, timeout: float) -> bool:
    request = urllib.request.Request(
        endpoint.rstrip("/") + f"/requests/{request_id}/cancel",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bool(json.loads(response.read())["cancelled"])


def _one_stress_request(
    endpoint: str,
    model: str,
    index: int,
    prompt_words: Sequence[int],
    output_lengths: Sequence[int],
    cancellation_ratio: float,
    cancel_after_tokens: int,
    seed: int,
    timeout: float,
) -> Dict[str, object]:
    rng = random.Random(seed + index)
    prompt_size = int(prompt_words[index % len(prompt_words)])
    max_tokens = int(output_lengths[index % len(output_lengths)])
    should_cancel = rng.random() < cancellation_ratio
    prompt = (f"request-{index} " + "token " * prompt_size).strip()
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
    first_token_at = None
    token_events = 0
    request_id = None
    finish_reason = None
    cancel_sent = False
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
                choice = chunk["choices"][0]
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                if choice.get("finish_reason") is None:
                    token_events += 1
                else:
                    finish_reason = choice["finish_reason"]
                if (
                    should_cancel
                    and not cancel_sent
                    and request_id
                    and token_events >= cancel_after_tokens
                ):
                    cancel_sent = _cancel(endpoint, request_id, timeout)
        finished = time.perf_counter()
        outcome = "cancelled" if finish_reason == "cancelled" else "completed"
        return {
            "index": index,
            "outcome": outcome,
            "request_id": request_id,
            "prompt_words": prompt_size,
            "max_tokens": max_tokens,
            "token_events": token_events,
            "cancel_requested": should_cancel,
            "cancel_sent": cancel_sent,
            "finish_reason": finish_reason,
            "ttft_seconds": (
                first_token_at - started if first_token_at is not None else None
            ),
            "duration_seconds": finished - started,
            "error": None,
        }
    except urllib.error.HTTPError as error:
        finished = time.perf_counter()
        outcome = "overloaded" if error.code in (429, 503) else "failed"
        return {
            "index": index,
            "outcome": outcome,
            "request_id": request_id,
            "prompt_words": prompt_size,
            "max_tokens": max_tokens,
            "token_events": token_events,
            "cancel_requested": should_cancel,
            "cancel_sent": cancel_sent,
            "finish_reason": None,
            "ttft_seconds": None,
            "duration_seconds": finished - started,
            "error": f"HTTP {error.code}: {error.reason}",
        }
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        finished = time.perf_counter()
        return {
            "index": index,
            "outcome": "failed",
            "request_id": request_id,
            "prompt_words": prompt_size,
            "max_tokens": max_tokens,
            "token_events": token_events,
            "cancel_requested": should_cancel,
            "cancel_sent": cancel_sent,
            "finish_reason": None,
            "ttft_seconds": None,
            "duration_seconds": finished - started,
            "error": str(error),
        }


def run_stress_benchmark(
    endpoint: str,
    model: str,
    requests: int = 100,
    concurrency: int = 8,
    prompt_words: Sequence[int] = (16, 128, 512),
    output_lengths: Sequence[int] = (8, 32, 128),
    cancellation_ratio: float = 0.1,
    cancel_after_tokens: int = 2,
    seed: int = 0,
    timeout: float = 300.0,
    duration_seconds: float = 0.0,
) -> Dict[str, object]:
    if requests <= 0 and duration_seconds <= 0:
        raise ValueError("requests must be positive unless duration is set")
    if concurrency <= 0 or duration_seconds < 0:
        raise ValueError("concurrency must be positive and duration non-negative")
    if not prompt_words or any(value <= 0 for value in prompt_words):
        raise ValueError("prompt_words must contain positive values")
    if not output_lengths or any(value <= 0 for value in output_lengths):
        raise ValueError("output_lengths must contain positive values")
    if not 0.0 <= cancellation_ratio <= 1.0:
        raise ValueError("cancellation_ratio must be in [0, 1]")
    if cancel_after_tokens <= 0:
        raise ValueError("cancel_after_tokens must be positive")

    before = _prometheus_metrics(endpoint, timeout)
    wall_started = time.perf_counter()
    results: List[Dict[str, object]] = []
    if duration_seconds > 0:
        deadline = wall_started + duration_seconds
        next_index = 0
        index_lock = threading.Lock()

        def duration_worker() -> List[Dict[str, object]]:
            nonlocal next_index
            worker_results = []
            while time.perf_counter() < deadline:
                with index_lock:
                    index = next_index
                    next_index += 1
                worker_results.append(
                    _one_stress_request(
                        endpoint,
                        model,
                        index,
                        prompt_words,
                        output_lengths,
                        cancellation_ratio,
                        cancel_after_tokens,
                        seed,
                        timeout,
                    )
                )
            return worker_results

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(duration_worker) for _ in range(concurrency)]
            for future in as_completed(futures):
                results.extend(future.result())
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _one_stress_request,
                    endpoint,
                    model,
                    index,
                    prompt_words,
                    output_lengths,
                    cancellation_ratio,
                    cancel_after_tokens,
                    seed,
                    timeout,
                )
                for index in range(requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())
    wall_seconds = time.perf_counter() - wall_started
    after = _prometheus_metrics(endpoint, timeout)

    counts = {
        outcome: sum(result["outcome"] == outcome for result in results)
        for outcome in ("completed", "cancelled", "overloaded", "failed")
    }
    successful = [
        result
        for result in results
        if result["outcome"] in ("completed", "cancelled")
    ]
    ttfts = [
        float(result["ttft_seconds"])
        for result in successful
        if result["ttft_seconds"] is not None
    ]
    durations = [float(result["duration_seconds"]) for result in successful]
    reserved_after = after.get("llaisys_kv_cache_reserved_bytes", 0.0)
    active_after = after.get("llaisys_request_active_sequences", 0.0)
    return {
        "endpoint": endpoint,
        "model": model,
        "requests": len(results),
        "concurrency": concurrency,
        "target_duration_seconds": duration_seconds,
        "prompt_words": list(prompt_words),
        "output_lengths": list(output_lengths),
        "cancellation_ratio": cancellation_ratio,
        "wall_seconds": wall_seconds,
        "requests_per_second": len(results) / max(wall_seconds, 1e-9),
        **counts,
        "ttft_seconds": {
            name: percentile(ttfts, quantile)
            for name, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        },
        "request_duration_seconds": {
            name: percentile(durations, quantile)
            for name, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        },
        "resource_leak_detected": reserved_after != 0.0 or active_after != 0.0,
        "metrics_before": before,
        "metrics_after": after,
        "results": sorted(results, key=lambda result: int(result["index"])),
    }


def _csv_ints(value: str) -> List[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="run continuously for this duration instead of a fixed request count",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-words", type=_csv_ints, default=[16, 128, 512])
    parser.add_argument("--output-lengths", type=_csv_ints, default=[8, 32, 128])
    parser.add_argument("--cancellation-ratio", type=float, default=0.1)
    parser.add_argument("--cancel-after-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    summary = run_stress_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        requests=args.requests,
        concurrency=args.concurrency,
        prompt_words=args.prompt_words,
        output_lengths=args.output_lengths,
        cancellation_ratio=args.cancellation_ratio,
        cancel_after_tokens=args.cancel_after_tokens,
        seed=args.seed,
        timeout=args.timeout,
        duration_seconds=args.duration_seconds,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 1 if summary["failed"] or summary["resource_leak_detected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
