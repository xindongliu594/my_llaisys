#include "../../nvidia_common.cuh"
#include "../../nvidia_kernel_utils.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {

void checkCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cublasGetStatusString(status));
    }
}

cublasHandle_t cublasHandle() {
    thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        checkCublas(cublasCreate(&handle), "cublasCreate");
    }
    return handle;
}

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t rows, size_t columns) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t numel = rows * columns;
    if (index < numel) {
        out[index] = detail::fromFloat<T>(detail::toFloat(out[index])
                                          + detail::toFloat(bias[index % columns]));
    }
}

template <typename T>
void launchBias(std::byte *out, const std::byte *bias, size_t rows, size_t columns) {
    if (bias == nullptr) {
        return;
    }
    const size_t numel = rows * columns;
    const int blocks = static_cast<int>((numel + detail::BLOCK_SIZE - 1) / detail::BLOCK_SIZE);
    addBiasKernel<<<blocks, detail::BLOCK_SIZE>>>(reinterpret_cast<T *>(out),
                                                  reinterpret_cast<const T *>(bias),
                                                  rows, columns);
    detail::checkLaunch("linear bias kernel");
}

cudaDataType_t cudaDataType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return CUDA_R_32F;
    case LLAISYS_DTYPE_F16: return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16: return CUDA_R_16BF;
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype, size_t rows,
            size_t out_features, size_t in_features) {
    const int m = static_cast<int>(out_features);
    const int n = static_cast<int>(rows);
    const int k = static_cast<int>(in_features);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudaDataType_t data_type = cudaDataType(dtype);
    const cublasComputeType_t compute_type = dtype == LLAISYS_DTYPE_F32
        ? CUBLAS_COMPUTE_32F_PEDANTIC
        : CUBLAS_COMPUTE_32F;

    checkCublas(cublasGemmEx(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
                             m, n, k, &alpha, weight, data_type, k,
                             in, data_type, k, &beta, out, data_type, m,
                             compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx");

    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launchBias<float>(out, bias, rows, out_features);
    case LLAISYS_DTYPE_F16: return launchBias<__half>(out, bias, rows, out_features);
    case LLAISYS_DTYPE_BF16: return launchBias<__nv_bfloat16>(out, bias, rows, out_features);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
