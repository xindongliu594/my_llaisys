import argparse
import gc
import math

from transformers import AutoTokenizer

import llaisys


def render_prompt(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return [int(token) for token in tokenizer.encode(text)]


def compare_orca_with_single_sequence(model_path, device, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = llaisys.models.Qwen2(model_path, device)
    prompts = [
        "Explain KV Cache in one sentence.",
        "What is grouped-query attention?",
    ]
    inputs = [render_prompt(tokenizer, prompt) for prompt in prompts]
    expected = [
        model.generate(tokens, max_new_tokens=max_new_tokens, top_k=1)[
            len(tokens) :
        ]
        for tokens in inputs
    ]

    scheduler = llaisys.OrcaScheduler(
        model,
        max_active_requests=len(inputs),
        max_prefill_per_iteration=1,
    )
    requests = []
    for index, tokens in enumerate(inputs):
        session_id = f"orca-{index}"
        scheduler.sessions.create(session_id, initial_tokens=tokens)
        requests.append(
            scheduler.submit(
                session_id,
                [],
                max_new_tokens=max_new_tokens,
                top_k=1,
                logprobs=3,
            )
        )
    list(scheduler.run_until_idle_stream())
    actual = [list(request.generated_tokens) for request in requests]

    if actual != expected:
        raise AssertionError(
            f"Orca output mismatch\nexpected={expected}\nactual={actual}"
        )
    for request in requests:
        if len(request.generated_logprobs) != len(request.generated_tokens):
            raise AssertionError("Every generated token must have logprobs")
        for token, token_logprobs in zip(
            request.generated_tokens, request.generated_logprobs
        ):
            if int(token_logprobs["token_id"]) != token:
                raise AssertionError("Logprob token does not match generation")
            if not math.isfinite(float(token_logprobs["logprob"])):
                raise AssertionError("Selected-token logprob must be finite")
            if len(token_logprobs["top_logprobs"]) != 3:
                raise AssertionError("Expected three top logprobs")
    print("Single-sequence and Orca token outputs match.")
    print("Backend token logprobs are finite and aligned.")
    print("Decode batch sizes:", list(scheduler.decode_batch_sizes))
    print("Generated text:")
    for tokens in actual:
        print(tokenizer.decode(tokens, skip_special_tokens=True))
    del model
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "nvidia"), default="nvidia")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    device = (
        llaisys.DeviceType.NVIDIA
        if args.device == "nvidia"
        else llaisys.DeviceType.CPU
    )
    compare_orca_with_single_sequence(
        args.model, device, args.max_new_tokens
    )
