#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rmsNorm(std::byte *out, const std::byte *in, const std::byte *weight,
             llaisysDataType_t dtype, size_t rows, size_t width, float eps);
}
