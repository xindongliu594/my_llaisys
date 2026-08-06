#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(in->ndim() == 2 && weight->ndim() == 1
                       && weight->shape()[0] == in->shape()[1],
                   "RMSNorm tensor shapes are invalid");
    CHECK_ARGUMENT(eps >= 0.0f, "RMSNorm epsilon must be non-negative");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rmsNorm(out->data(), in->data(), weight->data(), out->dtype(),
                            in->shape()[0], in->shape()[1], eps);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
