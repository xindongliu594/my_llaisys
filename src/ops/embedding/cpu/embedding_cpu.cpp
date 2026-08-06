#include "embedding_cpu.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               size_t index_count, size_t embedding_dim, size_t element_size) {
    const size_t row_bytes = embedding_dim * element_size;
    for (size_t i = 0; i < index_count; ++i) {
        std::memcpy(out + i * row_bytes, weight + static_cast<size_t>(index[i]) * row_bytes, row_bytes);
    }
}
} // namespace llaisys::ops::cpu
