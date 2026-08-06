#include "../../nvidia_common.cuh"
#include "../../nvidia_kernel_utils.cuh"

#include <cfloat>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void selfAttentionKernel(T *out, const T *query, const T *key,
                                    const T *value, size_t query_length,
                                    size_t kv_length, size_t num_heads,
                                    size_t num_kv_heads, size_t head_dim,
                                    size_t value_dim, float scale) {
    extern __shared__ float scores[];
    __shared__ float reduction[detail::BLOCK_SIZE];
    const size_t query_position = blockIdx.x / num_heads;
    const size_t head = blockIdx.x % num_heads;
    const size_t group_size = num_heads / num_kv_heads;
    const size_t kv_head = head / group_size;
    const size_t visible = kv_length - query_length + query_position + 1;
    const size_t query_base = (query_position * num_heads + head) * head_dim;

    float local_max = -FLT_MAX;
    for (size_t key_position = threadIdx.x; key_position < visible;
         key_position += blockDim.x) {
        const size_t key_base = (key_position * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (size_t dimension = 0; dimension < head_dim; ++dimension) {
            dot += detail::toFloat(query[query_base + dimension])
                * detail::toFloat(key[key_base + dimension]);
        }
        scores[key_position] = dot * scale;
        local_max = fmaxf(local_max, scores[key_position]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x],
                                            reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduction[0];

    float local_sum = 0.0f;
    for (size_t key_position = threadIdx.x; key_position < visible;
         key_position += blockDim.x) {
        scores[key_position] = expf(scores[key_position] - max_score);
        local_sum += scores[key_position];
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = reduction[0];

    const size_t output_base = (query_position * num_heads + head) * value_dim;
    for (size_t dimension = threadIdx.x; dimension < value_dim;
         dimension += blockDim.x) {
        float result = 0.0f;
        for (size_t key_position = 0; key_position < visible; ++key_position) {
            const size_t value_base = (key_position * num_kv_heads + kv_head) * value_dim;
            result += (scores[key_position] / denominator)
                * detail::toFloat(value[value_base + dimension]);
        }
        out[output_base + dimension] = detail::fromFloat<T>(result);
    }
}

template <typename T>
void launchSelfAttention(std::byte *out, const std::byte *query,
                         const std::byte *key, const std::byte *value,
                         size_t query_length, size_t kv_length,
                         size_t num_heads, size_t num_kv_heads,
                         size_t head_dim, size_t value_dim, float scale) {
    const unsigned int blocks = static_cast<unsigned int>(query_length * num_heads);
    const size_t shared_bytes = kv_length * sizeof(float);
    selfAttentionKernel<<<blocks, detail::BLOCK_SIZE, shared_bytes>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(query),
        reinterpret_cast<const T *>(key), reinterpret_cast<const T *>(value),
        query_length, kv_length, num_heads, num_kv_heads, head_dim, value_dim,
        scale);
    detail::checkLaunch("self-attention kernel");
}

} // namespace

void selfAttention(std::byte *out, const std::byte *query, const std::byte *key,
                   const std::byte *value, llaisysDataType_t dtype,
                   size_t query_length, size_t kv_length, size_t num_heads,
                   size_t num_kv_heads, size_t head_dim, size_t value_dim,
                   float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launchSelfAttention<float>(out, query, key, value, query_length, kv_length, num_heads, num_kv_heads, head_dim, value_dim, scale);
    case LLAISYS_DTYPE_F16: return launchSelfAttention<__half>(out, query, key, value, query_length, kv_length, num_heads, num_kv_heads, head_dim, value_dim, scale);
    case LLAISYS_DTYPE_BF16: return launchSelfAttention<__nv_bfloat16>(out, query, key, value, query_length, kv_length, num_heads, num_kv_heads, head_dim, value_dim, scale);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
