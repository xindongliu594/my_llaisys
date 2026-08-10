import argparse
import gc
import json

from transformers import AutoTokenizer

import llaisys
from llaisys.stress_benchmark import run_stress_benchmark


def run_real_model_stress(
    model_path, device, requests, concurrency, cancellation_ratio
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = llaisys.models.Qwen2(model_path, device)
    scheduler = llaisys.OrcaScheduler(
        model,
        max_active_requests=concurrency,
        max_prefill_per_iteration=1,
    )
    chat = llaisys.ChatService(scheduler, tokenizer)
    server = llaisys.OpenAIAPIServer(
        chat,
        model_id="llaisys-stress-test",
        host="127.0.0.1",
        port=0,
    )
    server.start()
    host, port = server.address
    try:
        summary = run_stress_benchmark(
            endpoint=f"http://{host}:{port}",
            model="llaisys-stress-test",
            requests=requests,
            concurrency=concurrency,
            prompt_words=(4, 16, 64),
            output_lengths=(4, 8, 16),
            cancellation_ratio=cancellation_ratio,
            cancel_after_tokens=2,
            timeout=600.0,
        )
    finally:
        server.stop(graceful=True, timeout_seconds=600.0)

    compact = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if summary["failed"]:
        raise AssertionError(f"Stress test had {summary['failed']} failures")
    if summary["resource_leak_detected"]:
        raise AssertionError("KV cache or active sequence did not return to zero")
    if cancellation_ratio > 0 and not summary["cancelled"]:
        raise AssertionError("Stress test did not exercise cancellation")
    del server
    del chat
    del scheduler
    del model
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "nvidia"), default="nvidia")
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--cancellation-ratio", type=float, default=0.25)
    args = parser.parse_args()
    selected_device = (
        llaisys.DeviceType.NVIDIA
        if args.device == "nvidia"
        else llaisys.DeviceType.CPU
    )
    run_real_model_stress(
        args.model,
        selected_device,
        args.requests,
        args.concurrency,
        args.cancellation_ratio,
    )
