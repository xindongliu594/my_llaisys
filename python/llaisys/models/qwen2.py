import ctypes
import json
import mmap
import re
import struct
import threading
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..libllaisys import DataType, DeviceType, LIB_LLAISYS
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysSamplingConfig


class Qwen2:
    _LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.(.+)")
    _LAYER_WEIGHTS = {
        "input_layernorm.weight": "attn_norm_w",
        "self_attn.q_proj.weight": "attn_q_w",
        "self_attn.q_proj.bias": "attn_q_b",
        "self_attn.k_proj.weight": "attn_k_w",
        "self_attn.k_proj.bias": "attn_k_b",
        "self_attn.v_proj.weight": "attn_v_w",
        "self_attn.v_proj.bias": "attn_v_b",
        "self_attn.o_proj.weight": "attn_o_w",
        "post_attention_layernorm.weight": "mlp_norm_w",
        "mlp.gate_proj.weight": "mlp_gate_w",
        "mlp.up_proj.weight": "mlp_up_w",
        "mlp.down_proj.weight": "mlp_down_w",
    }

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._model = None
        self._sequence_ids = {}
        self._sequence_capacities = {}
        self._sequence_lock = threading.RLock()
        self._next_sequence_id = 1
        self._device = DeviceType(device)
        model_path = Path(model_path)
        with (model_path / "config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)

        dtype_name = str(config.get("torch_dtype", "bfloat16")).lower()
        dtype = {
            "float32": DataType.F32,
            "float16": DataType.F16,
            "bfloat16": DataType.BF16,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported Qwen2 dtype: {dtype_name}")

        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        self._end_token = int(config["eos_token_id"])
        self._meta = LlaisysQwen2Meta(
            dtype=int(dtype),
            nlayer=int(config["num_hidden_layers"]),
            hs=hidden_size,
            nh=num_heads,
            nkvh=int(config["num_key_value_heads"]),
            dh=int(config.get("head_dim", hidden_size // num_heads)),
            di=int(config["intermediate_size"]),
            maxseq=int(config["max_position_embeddings"]),
            voc=int(config["vocab_size"]),
            epsilon=float(config["rms_norm_eps"]),
            theta=float(config["rope_theta"]),
            end_token=self._end_token,
        )
        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta), int(self._device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        try:
            self._load_weights(model_path)
        except Exception:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
            raise

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    @property
    def eos_token_id(self) -> int:
        return self._end_token

    @property
    def max_sequence_length(self) -> int:
        return int(self._meta.maxseq)

    @property
    def kv_cache_bytes_per_token(self) -> int:
        element_size = {
            int(DataType.F16): 2,
            int(DataType.BF16): 2,
            int(DataType.F32): 4,
        }[int(self._meta.dtype)]
        return (
            2
            * int(self._meta.nlayer)
            * int(self._meta.nkvh)
            * int(self._meta.dh)
            * element_size
        )

    def estimate_kv_cache_bytes(self, capacity: int) -> int:
        if not 0 <= capacity <= self.max_sequence_length:
            raise ValueError("KV cache capacity is outside the model limit")
        return int(capacity) * self.kv_cache_bytes_per_token

    def kv_cache_snapshot(self) -> dict[str, int]:
        with self._sequence_lock:
            allocated_tokens = sum(self._sequence_capacities.values())
            sequence_count = len(self._sequence_capacities)
        return {
            "kv_cache_allocated_bytes": (
                allocated_tokens * self.kv_cache_bytes_per_token
            ),
            "kv_cache_allocated_tokens": allocated_tokens,
            "kv_cache_model_sequences": sequence_count,
            "kv_cache_bytes_per_token": self.kv_cache_bytes_per_token,
        }

    def _weight_handle(self, weights, name: str):
        if name == "model.embed_tokens.weight":
            return weights.in_embed
        if name == "lm_head.weight":
            return weights.out_embed
        if name == "model.norm.weight":
            return weights.out_norm_w

        match = self._LAYER_PATTERN.fullmatch(name)
        if match is None:
            return None
        layer = int(match.group(1))
        suffix = match.group(2)
        field = self._LAYER_WEIGHTS.get(suffix)
        if field is None or layer >= self._meta.nlayer:
            return None
        return getattr(weights, field)[layer]

    def _load_weights(self, model_path: Path):
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        loaded = set()
        for file in sorted(model_path.glob("*.safetensors")):
            with file.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    data_base = 8 + header_size
                    for name, info in header.items():
                        if name == "__metadata__":
                            continue
                        start, end = info["data_offsets"]
                        raw = np.frombuffer(
                            mapped,
                            dtype=np.uint8,
                            count=end - start,
                            offset=data_base + start,
                        )
                        handle = self._weight_handle(weights, name)
                        if handle is None:
                            del raw
                            continue
                        LIB_LLAISYS.tensorLoad(
                            handle, ctypes.c_void_p(raw.ctypes.data)
                        )
                        loaded.add(name)
                        del raw

        expected = 3 + int(self._meta.nlayer) * len(self._LAYER_WEIGHTS)
        if len(loaded) != expected:
            raise RuntimeError(
                f"Incomplete Qwen2 weights: loaded {len(loaded)} of {expected} tensors"
            )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        seed: int = 0,
    ):
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if max_new_tokens is None:
            max_new_tokens = 128
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        output = [int(token) for token in inputs]
        if not output:
            raise ValueError("Qwen2 generation requires at least one input token")
        if len(output) + max_new_tokens > self.max_sequence_length:
            raise ValueError(
                f"Input ({len(output)}) plus requested output "
                f"({max_new_tokens}) exceeds max_sequence_length "
                f"({self.max_sequence_length})"
            )

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        prompt = (ctypes.c_int64 * len(output))(*output)
        sampling = LlaisysSamplingConfig(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            seed=int(seed),
        )
        use_greedy = top_k == 1 and repetition_penalty == 1.0
        infer = (
            LIB_LLAISYS.llaisysQwen2ModelInfer
            if use_greedy
            else LIB_LLAISYS.llaisysQwen2ModelInferSample
        )
        next_token = int(
            infer(self._model, prompt, len(output))
            if use_greedy
            else infer(self._model, prompt, len(output), ctypes.byref(sampling))
        )
        if next_token < 0:
            raise RuntimeError("Qwen2 backend inference failed during prefill")
        for step in range(max_new_tokens):
            output.append(next_token)
            if next_token == self._end_token or step + 1 == max_new_tokens:
                break
            token = ctypes.c_int64(next_token)
            next_token = int(
                infer(self._model, ctypes.byref(token), 1)
                if use_greedy
                else infer(
                    self._model,
                    ctypes.byref(token),
                    1,
                    ctypes.byref(sampling),
                )
            )
            if next_token < 0:
                raise RuntimeError("Qwen2 backend inference failed during decode")
        return output

    @property
    def supports_sequence_batching(self) -> bool:
        return True

    def create_sequence(self, sequence_id: str, capacity: int) -> None:
        key = str(sequence_id)
        with self._sequence_lock:
            if key in self._sequence_ids:
                raise ValueError(f"Sequence already exists: {key}")
            if not 0 < capacity <= self.max_sequence_length:
                raise ValueError("Sequence capacity is outside the model limit")
            numeric_id = self._next_sequence_id
            self._next_sequence_id += 1
            created = LIB_LLAISYS.llaisysQwen2SequenceCreate(
                self._model, numeric_id, capacity
            )
            if not created:
                raise RuntimeError("Failed to allocate sequence KV cache")
            self._sequence_ids[key] = numeric_id
            self._sequence_capacities[key] = int(capacity)

    def destroy_sequence(self, sequence_id: str) -> None:
        key = str(sequence_id)
        with self._sequence_lock:
            numeric_id = self._sequence_ids.pop(key, None)
            self._sequence_capacities.pop(key, None)
            if numeric_id is not None:
                LIB_LLAISYS.llaisysQwen2SequenceDestroy(self._model, numeric_id)

    def configure_sequence_logprobs(
        self, sequence_id: str, top_n: int
    ) -> None:
        if not 1 <= top_n <= 20:
            raise ValueError("top_n must be between 1 and 20")
        with self._sequence_lock:
            numeric_id = self._sequence_ids[str(sequence_id)]
        succeeded = LIB_LLAISYS.llaisysQwen2SequenceConfigureLogprobs(
            self._model, numeric_id, top_n
        )
        if not succeeded:
            raise RuntimeError("Failed to configure Qwen2 sequence logprobs")

    def get_sequence_logprobs(
        self, sequence_id: str, top_n: int
    ) -> dict[str, object]:
        if not 1 <= top_n <= 20:
            raise ValueError("top_n must be between 1 and 20")
        with self._sequence_lock:
            numeric_id = self._sequence_ids[str(sequence_id)]
        selected_token = ctypes.c_int64()
        selected_logprob = ctypes.c_float()
        token_ids = (ctypes.c_int64 * top_n)()
        logprobs = (ctypes.c_float * top_n)()
        count = int(
            LIB_LLAISYS.llaisysQwen2SequenceGetLogprobs(
                self._model,
                numeric_id,
                ctypes.byref(selected_token),
                ctypes.byref(selected_logprob),
                token_ids,
                logprobs,
                top_n,
            )
        )
        if count <= 0:
            raise RuntimeError("Qwen2 sequence logprobs are unavailable")
        return {
            "token_id": int(selected_token.value),
            "logprob": float(selected_logprob.value),
            "top_logprobs": [
                {
                    "token_id": int(token_ids[index]),
                    "logprob": float(logprobs[index]),
                }
                for index in range(count)
            ],
        }

    @staticmethod
    def _sampling_config(
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        seed: int,
    ) -> LlaisysSamplingConfig:
        return LlaisysSamplingConfig(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            seed=int(seed),
        )

    def prefill_sequence(
        self,
        sequence_id: str,
        input_tokens: Sequence[int],
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        seed: int = 0,
    ) -> int:
        with self._sequence_lock:
            numeric_id = self._sequence_ids[str(sequence_id)]
        tokens = [int(token) for token in input_tokens]
        if not tokens:
            raise ValueError("Prefill requires at least one token")
        token_array = (ctypes.c_int64 * len(tokens))(*tokens)
        config = self._sampling_config(
            top_k, top_p, temperature, repetition_penalty, seed
        )
        greedy = top_k == 1 and repetition_penalty == 1.0
        result = int(
            LIB_LLAISYS.llaisysQwen2SequenceInfer(
                self._model, numeric_id, token_array, len(tokens)
            )
            if greedy
            else LIB_LLAISYS.llaisysQwen2SequenceInferSample(
                self._model,
                numeric_id,
                token_array,
                len(tokens),
                ctypes.byref(config),
            )
        )
        if result < 0:
            raise RuntimeError("Qwen2 sequence prefill failed")
        return result

    def decode_batch(
        self,
        sequence_ids: Sequence[str],
        token_ids: Sequence[int],
        sampling_configs: Sequence[Mapping[str, object]],
    ) -> list[int]:
        if not sequence_ids or not (
            len(sequence_ids) == len(token_ids) == len(sampling_configs)
        ):
            raise ValueError("Batch sequence, token, and config lengths must match")
        with self._sequence_lock:
            numeric_ids = [
                self._sequence_ids[str(key)] for key in sequence_ids
            ]
        batch_size = len(numeric_ids)
        id_array = (ctypes.c_uint64 * batch_size)(*numeric_ids)
        token_array = (ctypes.c_int64 * batch_size)(
            *(int(token) for token in token_ids)
        )
        output_array = (ctypes.c_int64 * batch_size)()
        configs = [
            self._sampling_config(
                int(config["top_k"]),
                float(config["top_p"]),
                float(config["temperature"]),
                float(config["repetition_penalty"]),
                int(config["seed"]),
            )
            for config in sampling_configs
        ]
        greedy = all(
            config.top_k == 1 and config.repetition_penalty == 1.0
            for config in configs
        )
        if greedy:
            succeeded = LIB_LLAISYS.llaisysQwen2BatchInfer(
                self._model,
                id_array,
                token_array,
                batch_size,
                output_array,
            )
        else:
            config_array = (LlaisysSamplingConfig * batch_size)(*configs)
            succeeded = LIB_LLAISYS.llaisysQwen2BatchInferSample(
                self._model,
                id_array,
                token_array,
                batch_size,
                config_array,
                output_array,
            )
        if not succeeded:
            raise RuntimeError("Qwen2 batched decode failed")
        return [int(token) for token in output_array]
