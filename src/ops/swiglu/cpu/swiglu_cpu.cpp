#include "swiglu_cpu.hpp"

#include "../../cpu_common.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        const float gate_value = llaisys::ops::cpu::toFloat(gate[i]);
        const float up_value = llaisys::ops::cpu::toFloat(up[i]);
        const float silu = gate_value / (1.0f + std::exp(-gate_value));
        out[i] = llaisys::ops::cpu::fromFloat<T>(up_value * silu);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
#define RUN_SWIGLU(TYPE) return swiglu_(reinterpret_cast<TYPE *>(out), reinterpret_cast<const TYPE *>(gate), \
                                        reinterpret_cast<const TYPE *>(up), numel)
    switch (dtype) {
    case LLAISYS_DTYPE_F32: RUN_SWIGLU(float);
    case LLAISYS_DTYPE_F16: RUN_SWIGLU(fp16_t);
    case LLAISYS_DTYPE_BF16: RUN_SWIGLU(bf16_t);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
#undef RUN_SWIGLU
}
} // namespace llaisys::ops::cpu
