import ctypes
import json
import mmap
import re
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

from ..libllaisys import DataType, DeviceType, LIB_LLAISYS
from ..libllaisys.models import LlaisysQwen2Meta


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
    ):
        del top_p, temperature
        if top_k != 1:
            raise NotImplementedError("The assignment backend currently supports greedy decoding only")
        if max_new_tokens is None:
            max_new_tokens = 128
        output = [int(token) for token in inputs]
        if not output:
            raise ValueError("Qwen2 generation requires at least one input token")

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        prompt = (ctypes.c_int64 * len(output))(*output)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, prompt, len(output))
        )
        for _ in range(max_new_tokens):
            output.append(next_token)
            if next_token == self._end_token:
                break
            token = ctypes.c_int64(next_token)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, ctypes.byref(token), 1
                )
            )
        return output
