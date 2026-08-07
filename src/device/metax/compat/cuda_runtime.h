#pragma once

#include <mc_runtime.h>

using cudaError_t = mcError_t;
using cudaMemcpyKind = mcMemcpyKind;
using cudaStream_t = mcStream_t;

constexpr auto cudaSuccess = mcSuccess;
constexpr auto cudaMemcpyHostToHost = mcMemcpyHostToHost;
constexpr auto cudaMemcpyHostToDevice = mcMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = mcMemcpyDeviceToHost;
constexpr auto cudaMemcpyDeviceToDevice = mcMemcpyDeviceToDevice;

inline const char *cudaGetErrorString(cudaError_t status) {
    return mcGetErrorString(status);
}

inline cudaError_t cudaGetDeviceCount(int *count) {
    return mcGetDeviceCount(count);
}

inline cudaError_t cudaSetDevice(int device) {
    return mcSetDevice(device);
}

inline cudaError_t cudaDeviceSynchronize() {
    return mcDeviceSynchronize();
}

inline cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    return mcStreamCreate(stream);
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return mcStreamDestroy(stream);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return mcStreamSynchronize(stream);
}

inline cudaError_t cudaMalloc(void **pointer, size_t size) {
    return mcMalloc(pointer, size);
}

inline cudaError_t cudaFree(void *pointer) {
    return mcFree(pointer);
}

inline cudaError_t cudaMallocHost(void **pointer, size_t size) {
    return mcMallocHost(pointer, size, mcMallocHostDefault);
}

inline cudaError_t cudaFreeHost(void *pointer) {
    return mcFreeHost(pointer);
}

inline cudaError_t cudaMemcpy(void *destination, const void *source, size_t size,
                              cudaMemcpyKind kind) {
    return mcMemcpy(destination, source, size, kind);
}

inline cudaError_t cudaMemcpyAsync(void *destination, const void *source,
                                   size_t size, cudaMemcpyKind kind,
                                   cudaStream_t stream) {
    return mcMemcpyAsync(destination, source, size, kind, stream);
}

inline cudaError_t cudaGetLastError() {
    return mcGetLastError();
}
