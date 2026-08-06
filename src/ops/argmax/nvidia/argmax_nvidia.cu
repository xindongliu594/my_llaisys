#include "../../nvidia_common.cuh"
#include "../../nvidia_kernel_utils.cuh"

#include <cfloat>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void argmaxKernel(int64_t *max_index, T *max_value, const T *values,
                             size_t numel) {
    __shared__ float best_values[detail::BLOCK_SIZE];
    __shared__ int64_t best_indices[detail::BLOCK_SIZE];

    float local_value = -FLT_MAX;
    int64_t local_index = INT64_MAX;
    for (size_t index = threadIdx.x; index < numel; index += blockDim.x) {
        const float value = detail::toFloat(values[index]);
        if (value > local_value || (value == local_value && static_cast<int64_t>(index) < local_index)) {
            local_value = value;
            local_index = static_cast<int64_t>(index);
        }
    }
    best_values[threadIdx.x] = local_value;
    best_indices[threadIdx.x] = local_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float candidate = best_values[threadIdx.x + stride];
            const int64_t candidate_index = best_indices[threadIdx.x + stride];
            if (candidate > best_values[threadIdx.x]
                || (candidate == best_values[threadIdx.x]
                    && candidate_index < best_indices[threadIdx.x])) {
                best_values[threadIdx.x] = candidate;
                best_indices[threadIdx.x] = candidate_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *max_index = best_indices[0];
        *max_value = values[best_indices[0]];
    }
}

template <typename T>
void launchArgmax(int64_t *max_index, std::byte *max_value,
                  const std::byte *values, size_t numel) {
    argmaxKernel<<<1, detail::BLOCK_SIZE>>>(max_index, reinterpret_cast<T *>(max_value),
                                            reinterpret_cast<const T *>(values), numel);
    detail::checkLaunch("argmax kernel");
}

} // namespace

void argmax(int64_t *max_index, std::byte *max_value, const std::byte *values,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launchArgmax<float>(max_index, max_value, values, numel);
    case LLAISYS_DTYPE_F16: return launchArgmax<__half>(max_index, max_value, values, numel);
    case LLAISYS_DTYPE_BF16: return launchArgmax<__nv_bfloat16>(max_index, max_value, values, numel);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
