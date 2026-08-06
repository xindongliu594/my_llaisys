#include "../../nvidia_common.cuh"
#include "../../nvidia_kernel_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void addKernel(T *out, const T *a, const T *b, size_t numel) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < numel) {
        out[index] = detail::fromFloat<T>(detail::toFloat(a[index]) + detail::toFloat(b[index]));
    }
}

template <typename T>
void launchAdd(std::byte *out, const std::byte *a, const std::byte *b, size_t numel) {
    const int blocks = static_cast<int>((numel + detail::BLOCK_SIZE - 1) / detail::BLOCK_SIZE);
    addKernel<<<blocks, detail::BLOCK_SIZE>>>(reinterpret_cast<T *>(out),
                                              reinterpret_cast<const T *>(a),
                                              reinterpret_cast<const T *>(b), numel);
    detail::checkLaunch("add kernel");
}

} // namespace

void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launchAdd<float>(out, a, b, numel);
    case LLAISYS_DTYPE_F16: return launchAdd<__half>(out, a, b, numel);
    case LLAISYS_DTYPE_BF16: return launchAdd<__nv_bfloat16>(out, a, b, numel);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
