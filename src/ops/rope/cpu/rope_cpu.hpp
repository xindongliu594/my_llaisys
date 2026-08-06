#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t num_heads,
          size_t head_dim, float theta);
}
