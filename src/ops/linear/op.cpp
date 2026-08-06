#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
                   "Linear input, weight, and output must be 2D");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1]
                       && out->shape()[0] == in->shape()[0]
                       && out->shape()[1] == weight->shape()[0],
                   "Linear tensor shapes are invalid");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear tensors must be contiguous");
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0],
                       "Linear bias shape is invalid");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear bias must be contiguous");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias == nullptr ? nullptr : bias->data(), out->dtype(),
                           in->shape()[0], weight->shape()[0], weight->shape()[1]);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
