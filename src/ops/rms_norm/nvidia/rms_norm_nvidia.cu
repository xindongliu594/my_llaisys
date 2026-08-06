#include "../../nvidia_common.cuh"
#include "../../nvidia_kernel_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight,
                              size_t width, float eps) {
    __shared__ float partial[detail::BLOCK_SIZE];
    const size_t row = blockIdx.x;
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = detail::toFloat(in[row * width + column]);
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[0] = rsqrtf(partial[0] / static_cast<float>(width) + eps);
    }
    __syncthreads();

    const float inverse_rms = partial[0];
    for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = detail::toFloat(in[row * width + column]);
        const float scale = detail::toFloat(weight[column]);
        out[row * width + column] = detail::fromFloat<T>(value * inverse_rms * scale);
    }
}

template <typename T>
void launchRmsNorm(std::byte *out, const std::byte *in, const std::byte *weight,
                   size_t rows, size_t width, float eps) {
    rmsNormKernel<<<static_cast<unsigned int>(rows), detail::BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight), width, eps);
    detail::checkLaunch("RMSNorm kernel");
}

} // namespace

void rmsNorm(std::byte *out, const std::byte *in, const std::byte *weight,
             llaisysDataType_t dtype, size_t rows, size_t width, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launchRmsNorm<float>(out, in, weight, rows, width, eps);
    case LLAISYS_DTYPE_F16: return launchRmsNorm<__half>(out, in, weight, rows, width, eps);
    case LLAISYS_DTYPE_BF16: return launchRmsNorm<__nv_bfloat16>(out, in, weight, rows, width, eps);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
