#include "argmax_cpu.hpp"

#include "../../cpu_common.hpp"

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t best = 0;
    float best_value = llaisys::ops::cpu::toFloat(vals[0]);
    for (size_t i = 1; i < numel; ++i) {
        const float value = llaisys::ops::cpu::toFloat(vals[i]);
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    *max_idx = static_cast<int64_t>(best);
    *max_val = vals[best];
}

namespace llaisys::ops::cpu {
void argmax(int64_t *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_(max_idx, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(max_idx, reinterpret_cast<fp16_t *>(max_val), reinterpret_cast<const fp16_t *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(max_idx, reinterpret_cast<bf16_t *>(max_val), reinterpret_cast<const bf16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
