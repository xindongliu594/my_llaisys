#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {

void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t dtype, size_t numel);

void argmax(int64_t *max_idx, std::byte *max_val, const std::byte *values,
            llaisysDataType_t dtype, size_t numel);

void embedding(std::byte *out, const int64_t *indices, const std::byte *weight,
               size_t index_count, size_t embedding_dim, size_t element_size);

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype, size_t rows,
            size_t out_features, size_t in_features);

void rmsNorm(std::byte *out, const std::byte *in, const std::byte *weight,
             llaisysDataType_t dtype, size_t rows, size_t width, float eps);

void rope(std::byte *out, const std::byte *in, const int64_t *position_ids,
          llaisysDataType_t dtype, size_t sequence_length, size_t num_heads,
          size_t head_dim, float theta);

void selfAttention(std::byte *out, const std::byte *query, const std::byte *key,
                   const std::byte *value, llaisysDataType_t dtype,
                   size_t query_length, size_t kv_length, size_t num_heads,
                   size_t num_kv_heads, size_t head_dim, size_t value_dim,
                   float scale);

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel);

} // namespace llaisys::ops::nvidia
