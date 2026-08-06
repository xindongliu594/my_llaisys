#include "rope_cpu.hpp"

#include "../../cpu_common.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len,
           size_t num_heads, size_t head_dim, float theta) {
    const size_t half = head_dim / 2;
    for (size_t seq = 0; seq < seq_len; ++seq) {
        for (size_t head = 0; head < num_heads; ++head) {
            const size_t base = (seq * num_heads + head) * head_dim;
            for (size_t j = 0; j < half; ++j) {
                const float angle = static_cast<float>(pos_ids[seq])
                    / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                const float sin_value = std::sin(angle);
                const float cos_value = std::cos(angle);
                const float a = llaisys::ops::cpu::toFloat(in[base + j]);
                const float b = llaisys::ops::cpu::toFloat(in[base + half + j]);
                out[base + j] = llaisys::ops::cpu::fromFloat<T>(a * cos_value - b * sin_value);
                out[base + half + j] = llaisys::ops::cpu::fromFloat<T>(b * cos_value + a * sin_value);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t num_heads,
          size_t head_dim, float theta) {
#define RUN_ROPE(TYPE) return rope_(reinterpret_cast<TYPE *>(out), reinterpret_cast<const TYPE *>(in), \
                                    pos_ids, seq_len, num_heads, head_dim, theta)
    switch (dtype) {
    case LLAISYS_DTYPE_F32: RUN_ROPE(float);
    case LLAISYS_DTYPE_F16: RUN_ROPE(fp16_t);
    case LLAISYS_DTYPE_BF16: RUN_ROPE(bf16_t);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
#undef RUN_ROPE
}
} // namespace llaisys::ops::cpu
