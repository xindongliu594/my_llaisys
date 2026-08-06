#include "linear_cpu.hpp"

#include "../../cpu_common.hpp"

#include <algorithm>
#include <thread>
#include <vector>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t rows, size_t out_features, size_t in_features) {
    const size_t work = rows * out_features * in_features;
    const size_t output_elements = rows * out_features;
    const size_t available_threads = std::min<size_t>(64, std::max(1u, std::thread::hardware_concurrency()));
    const size_t thread_count = work < 1'000'000 ? 1 : std::min(output_elements, available_threads);

    auto worker = [&](size_t worker_id) {
        for (size_t output_idx = worker_id; output_idx < output_elements; output_idx += thread_count) {
            const size_t row = output_idx / out_features;
            const size_t col = output_idx % out_features;
            float sum = bias == nullptr ? 0.0f : llaisys::ops::cpu::toFloat(bias[col]);
            for (size_t k = 0; k < in_features; ++k) {
                sum += llaisys::ops::cpu::toFloat(in[row * in_features + k])
                    * llaisys::ops::cpu::toFloat(weight[col * in_features + k]);
            }
            out[output_idx] = llaisys::ops::cpu::fromFloat<T>(sum);
        }
    };

    if (thread_count == 1) {
        worker(0);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype, size_t rows,
            size_t out_features, size_t in_features) {
#define RUN_LINEAR(TYPE)                                                                 \
    return linear_(reinterpret_cast<TYPE *>(out), reinterpret_cast<const TYPE *>(in),   \
                   reinterpret_cast<const TYPE *>(weight),                               \
                   bias == nullptr ? nullptr : reinterpret_cast<const TYPE *>(bias),     \
                   rows, out_features, in_features)
    switch (dtype) {
    case LLAISYS_DTYPE_F32: RUN_LINEAR(float);
    case LLAISYS_DTYPE_F16: RUN_LINEAR(fp16_t);
    case LLAISYS_DTYPE_BF16: RUN_LINEAR(bf16_t);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
#undef RUN_LINEAR
}
} // namespace llaisys::ops::cpu
