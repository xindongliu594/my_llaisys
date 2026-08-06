#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype, size_t rows,
            size_t out_features, size_t in_features);
}
