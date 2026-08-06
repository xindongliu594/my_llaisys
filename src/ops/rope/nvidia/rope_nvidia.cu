#include "../../nvidia_common.cuh"
#include "../../nvidia_kernel_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void ropeKernel(T *out, const T *in, const int64_t *position_ids,
                           size_t num_heads, size_t head_dim, size_t pairs,
                           float theta) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairs) {
        return;
    }
    const size_t half = head_dim / 2;
    const size_t pair = index % half;
    const size_t token_head = index / half;
    const size_t token = token_head / num_heads;
    const size_t base = token_head * head_dim;
    const float angle = static_cast<float>(position_ids[token])
        / powf(theta, 2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    const float first = detail::toFloat(in[base + pair]);
    const float second = detail::toFloat(in[base + half + pair]);
    out[base + pair] = detail::fromFloat<T>(first * cosine - second * sine);
    out[base + half + pair] = detail::fromFloat<T>(second * cosine + first * sine);
}

template <typename T>
void launchRope(std::byte *out, const std::byte *in, const int64_t *position_ids,
                size_t sequence_length, size_t num_heads, size_t head_dim,
                float theta) {
    const size_t pairs = sequence_length * num_heads * (head_dim / 2);
    const int blocks = static_cast<int>((pairs + detail::BLOCK_SIZE - 1) / detail::BLOCK_SIZE);
    ropeKernel<<<blocks, detail::BLOCK_SIZE>>>(reinterpret_cast<T *>(out),
                                               reinterpret_cast<const T *>(in),
                                               position_ids, num_heads, head_dim,
                                               pairs, theta);
    detail::checkLaunch("RoPE kernel");
}

} // namespace

void rope(std::byte *out, const std::byte *in, const int64_t *position_ids,
          llaisysDataType_t dtype, size_t sequence_length, size_t num_heads,
          size_t head_dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launchRope<float>(out, in, position_ids, sequence_length, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16: return launchRope<__half>(out, in, position_ids, sequence_length, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16: return launchRope<__nv_bfloat16>(out, in, position_ids, sequence_length, num_heads, head_dim, theta);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
