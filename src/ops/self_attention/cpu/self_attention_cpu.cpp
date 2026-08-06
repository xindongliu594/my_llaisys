#include "self_attention_cpu.hpp"

#include "../../cpu_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t query_len, size_t kv_len, size_t num_heads,
                     size_t num_kv_heads, size_t head_dim, size_t value_dim,
                     float scale) {
    const size_t group_size = num_heads / num_kv_heads;
    std::vector<float> scores(kv_len);

    for (size_t query_pos = 0; query_pos < query_len; ++query_pos) {
        const size_t visible = kv_len - query_len + query_pos + 1;
        for (size_t head = 0; head < num_heads; ++head) {
            const size_t kv_head = head / group_size;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t key_pos = 0; key_pos < visible; ++key_pos) {
                float dot = 0.0f;
                const size_t q_base = (query_pos * num_heads + head) * head_dim;
                const size_t k_base = (key_pos * num_kv_heads + kv_head) * head_dim;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    dot += llaisys::ops::cpu::toFloat(q[q_base + dim])
                        * llaisys::ops::cpu::toFloat(k[k_base + dim]);
                }
                scores[key_pos] = dot * scale;
                max_score = std::max(max_score, scores[key_pos]);
            }

            float denominator = 0.0f;
            for (size_t key_pos = 0; key_pos < visible; ++key_pos) {
                scores[key_pos] = std::exp(scores[key_pos] - max_score);
                denominator += scores[key_pos];
            }

            const size_t out_base = (query_pos * num_heads + head) * value_dim;
            for (size_t dim = 0; dim < value_dim; ++dim) {
                float value = 0.0f;
                for (size_t key_pos = 0; key_pos < visible; ++key_pos) {
                    const size_t v_base = (key_pos * num_kv_heads + kv_head) * value_dim;
                    value += (scores[key_pos] / denominator)
                        * llaisys::ops::cpu::toFloat(v[v_base + dim]);
                }
                out[out_base + dim] = llaisys::ops::cpu::fromFloat<T>(value);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void selfAttention(std::byte *out, const std::byte *q, const std::byte *k,
                   const std::byte *v, llaisysDataType_t dtype, size_t query_len,
                   size_t kv_len, size_t num_heads, size_t num_kv_heads,
                   size_t head_dim, size_t value_dim, float scale) {
#define RUN_ATTN(TYPE) return self_attention_(reinterpret_cast<TYPE *>(out), reinterpret_cast<const TYPE *>(q), \
                                              reinterpret_cast<const TYPE *>(k), reinterpret_cast<const TYPE *>(v), \
                                              query_len, kv_len, num_heads, num_kv_heads, head_dim, value_dim, scale)
    switch (dtype) {
    case LLAISYS_DTYPE_F32: RUN_ATTN(float);
    case LLAISYS_DTYPE_F16: RUN_ATTN(fp16_t);
    case LLAISYS_DTYPE_BF16: RUN_ATTN(bf16_t);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
#undef RUN_ATTN
}
} // namespace llaisys::ops::cpu
