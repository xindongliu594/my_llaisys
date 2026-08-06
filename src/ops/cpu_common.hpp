#pragma once

#include "../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
inline float toFloat(T value) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<float>(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
inline T fromFloat(float value) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<T>(value);
    } else {
        return static_cast<T>(value);
    }
}

} // namespace llaisys::ops::cpu
