#include "../../nvidia_common.cuh"
#include "../../nvidia_kernel_utils.cuh"

namespace llaisys::ops::nvidia {
namespace {

__global__ void embeddingKernel(unsigned char *out, const int64_t *indices,
                                const unsigned char *weight, size_t row_bytes,
                                size_t total_bytes) {
    const size_t offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < total_bytes) {
        const size_t output_row = offset / row_bytes;
        const size_t column_byte = offset % row_bytes;
        out[offset] = weight[static_cast<size_t>(indices[output_row]) * row_bytes + column_byte];
    }
}

} // namespace

void embedding(std::byte *out, const int64_t *indices, const std::byte *weight,
               size_t index_count, size_t embedding_dim, size_t element_size) {
    const size_t row_bytes = embedding_dim * element_size;
    const size_t total_bytes = index_count * row_bytes;
    const int blocks = static_cast<int>((total_bytes + detail::BLOCK_SIZE - 1) / detail::BLOCK_SIZE);
    embeddingKernel<<<blocks, detail::BLOCK_SIZE>>>(
        reinterpret_cast<unsigned char *>(out), indices,
        reinterpret_cast<const unsigned char *>(weight), row_bytes, total_bytes);
    detail::checkLaunch("embedding kernel");
}

} // namespace llaisys::ops::nvidia
