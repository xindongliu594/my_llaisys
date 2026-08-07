#pragma once

#include <mcblas.h>

using cublasStatus_t = mcblasStatus_t;
using cublasHandle_t = mcblasHandle_t;
using cublasOperation_t = mcblasOperation_t;
using cublasComputeType_t = mcblasComputeType_t;
using cublasGemmAlgo_t = mcblasGemmAlgo_t;
using cudaDataType_t = macaDataType;

constexpr auto CUBLAS_STATUS_SUCCESS = MCBLAS_STATUS_SUCCESS;
constexpr auto CUBLAS_OP_N = MCBLAS_OP_N;
constexpr auto CUBLAS_OP_T = MCBLAS_OP_T;
constexpr auto CUBLAS_COMPUTE_32F = MCBLAS_COMPUTE_32F;
constexpr auto CUBLAS_COMPUTE_32F_PEDANTIC = MCBLAS_COMPUTE_32F_PEDANTIC;
constexpr auto CUBLAS_GEMM_DEFAULT_TENSOR_OP = MCBLAS_GEMM_DEFAULT_TENSOR_OP;
constexpr auto CUDA_R_32F = MACA_R_32F;
constexpr auto CUDA_R_16F = MACA_R_16F;
constexpr auto CUDA_R_16BF = MACA_R_16BF;

inline const char *cublasGetStatusString(cublasStatus_t status) {
    return mcblasGetStatusString(status);
}

inline cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    return mcblasCreate(handle);
}

inline cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return mcblasDestroy(handle);
}

inline cublasStatus_t cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const void *alpha,
    const void *a, cudaDataType_t a_type, int lda, const void *b,
    cudaDataType_t b_type, int ldb, const void *beta, void *c,
    cudaDataType_t c_type, int ldc, cublasComputeType_t compute_type,
    cublasGemmAlgo_t algorithm) {
    return mcblasGemmEx(handle, transa, transb, m, n, k, alpha, a, a_type,
                        lda, b, b_type, ldb, beta, c, c_type, ldc,
                        compute_type, algorithm);
}
