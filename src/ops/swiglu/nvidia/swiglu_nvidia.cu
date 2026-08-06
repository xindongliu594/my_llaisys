#include "../../nvidia_common.cuh"
#include "../../nvidia_kernel_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < numel) {
        const float gate_value = detail::toFloat(gate[index]);
        const float up_value = detail::toFloat(up[index]);
        const float silu = gate_value / (1.0f + expf(-gate_value));
        out[index] = detail::fromFloat<T>(up_value * silu);
    }
}

template <typename T>
void launchSwiglu(std::byte *out, const std::byte *gate, const std::byte *up,
                  size_t numel) {
    const int blocks = static_cast<int>((numel + detail::BLOCK_SIZE - 1) / detail::BLOCK_SIZE);
    swigluKernel<<<blocks, detail::BLOCK_SIZE>>>(reinterpret_cast<T *>(out),
                                                 reinterpret_cast<const T *>(gate),
                                                 reinterpret_cast<const T *>(up), numel);
    detail::checkLaunch("SwiGLU kernel");
}

} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launchSwiglu<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16: return launchSwiglu<__half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16: return launchSwiglu<__nv_bfloat16>(out, gate, up, numel);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
