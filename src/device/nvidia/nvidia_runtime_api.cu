#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

namespace {

void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

cudaMemcpyKind memcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Invalid CUDA memcpy kind");
    }
}

} // namespace

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

void setDevice(int device) {
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
}

void deviceSynchronize() {
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        checkCuda(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
    }
}

void streamSynchronize(llaisysStream_t stream) {
    checkCuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *pointer = nullptr;
    checkCuda(cudaMalloc(&pointer, size), "cudaMalloc");
    return pointer;
}

void freeDevice(void *pointer) {
    if (pointer != nullptr) {
        checkCuda(cudaFree(pointer), "cudaFree");
    }
}

void *mallocHost(size_t size) {
    void *pointer = nullptr;
    checkCuda(cudaMallocHost(&pointer, size), "cudaMallocHost");
    return pointer;
}

void freeHost(void *pointer) {
    if (pointer != nullptr) {
        checkCuda(cudaFreeHost(pointer), "cudaFreeHost");
    }
}

void memcpySync(void *destination, const void *source, size_t size, llaisysMemcpyKind_t kind) {
    checkCuda(cudaMemcpy(destination, source, size, memcpyKind(kind)), "cudaMemcpy");
}

void memcpyAsync(void *destination, const void *source, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    checkCuda(cudaMemcpyAsync(destination, source, size, memcpyKind(kind),
                              reinterpret_cast<cudaStream_t>(stream)),
              "cudaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
