#pragma once

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               size_t index_count, size_t embedding_dim, size_t element_size);
}
