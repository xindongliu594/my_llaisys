import ctypes

from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]


class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]


LlaisysQwen2Model = ctypes.c_void_p


class LlaisysSamplingConfig(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_size_t),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("seed", ctypes.c_uint64),
    ]


def load_models(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelReset.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelReset.restype = None

    lib.llaisysQwen2ModelInfer.argtypes = [
        LlaisysQwen2Model,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64

    lib.llaisysQwen2ModelInferSample.argtypes = [
        LlaisysQwen2Model,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(LlaisysSamplingConfig),
    ]
    lib.llaisysQwen2ModelInferSample.restype = ctypes.c_int64

    lib.llaisysQwen2SequenceCreate.argtypes = [
        LlaisysQwen2Model,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2SequenceCreate.restype = ctypes.c_int

    lib.llaisysQwen2SequenceDestroy.argtypes = [
        LlaisysQwen2Model,
        ctypes.c_uint64,
    ]
    lib.llaisysQwen2SequenceDestroy.restype = None

    lib.llaisysQwen2SequenceReset.argtypes = [
        LlaisysQwen2Model,
        ctypes.c_uint64,
    ]
    lib.llaisysQwen2SequenceReset.restype = ctypes.c_int

    lib.llaisysQwen2SequenceConfigureLogprobs.argtypes = [
        LlaisysQwen2Model,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2SequenceConfigureLogprobs.restype = ctypes.c_int

    lib.llaisysQwen2SequenceGetLogprobs.argtypes = [
        LlaisysQwen2Model,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2SequenceGetLogprobs.restype = ctypes.c_size_t

    lib.llaisysQwen2SequenceInfer.argtypes = [
        LlaisysQwen2Model,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
    ]
    lib.llaisysQwen2SequenceInfer.restype = ctypes.c_int64

    lib.llaisysQwen2SequenceInferSample.argtypes = [
        LlaisysQwen2Model,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(LlaisysSamplingConfig),
    ]
    lib.llaisysQwen2SequenceInferSample.restype = ctypes.c_int64

    lib.llaisysQwen2BatchInfer.argtypes = [
        LlaisysQwen2Model,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.llaisysQwen2BatchInfer.restype = ctypes.c_int

    lib.llaisysQwen2BatchInferSample.argtypes = [
        LlaisysQwen2Model,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(LlaisysSamplingConfig),
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.llaisysQwen2BatchInferSample.restype = ctypes.c_int
