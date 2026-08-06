#include "rms_norm_cpu.hpp"

#include "../../cpu_common.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t width, float eps) {
    for (size_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0f;
        for (size_t col = 0; col < width; ++col) {
            const float value = llaisys::ops::cpu::toFloat(in[row * width + col]);
            sum_squares += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(width) + eps);
        for (size_t col = 0; col < width; ++col) {
            const float value = llaisys::ops::cpu::toFloat(in[row * width + col]);
            const float scale = llaisys::ops::cpu::toFloat(weight[col]);
            out[row * width + col] = llaisys::ops::cpu::fromFloat<T>(value * inv_rms * scale);
        }
    }
}

namespace llaisys::ops::cpu {
void rmsNorm(std::byte *out, const std::byte *in, const std::byte *weight,
             llaisysDataType_t dtype, size_t rows, size_t width, float eps) {
#define RUN_RMS(TYPE) return rms_norm_(reinterpret_cast<TYPE *>(out), reinterpret_cast<const TYPE *>(in), \
                                      reinterpret_cast<const TYPE *>(weight), rows, width, eps)
    switch (dtype) {
    case LLAISYS_DTYPE_F32: RUN_RMS(float);
    case LLAISYS_DTYPE_F16: RUN_RMS(fp16_t);
    case LLAISYS_DTYPE_BF16: RUN_RMS(bf16_t);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
#undef RUN_RMS
}
} // namespace llaisys::ops::cpu
