#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia_common.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1 && vals->numel() > 0, "Argmax expects a non-empty 1D input");
    CHECK_ARGUMENT(max_idx->numel() == 1 && max_idx->dtype() == LLAISYS_DTYPE_I64,
                   "Argmax index output must contain one int64 value");
    CHECK_ARGUMENT(max_val->numel() == 1 && max_val->dtype() == vals->dtype(),
                   "Argmax value output must match the input dtype");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax tensors must be contiguous");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(reinterpret_cast<int64_t *>(max_idx->data()), max_val->data(),
                           vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(reinterpret_cast<int64_t *>(max_idx->data()), max_val->data(),
                             vals->data(), vals->dtype(), vals->numel());
#endif
    default: EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
