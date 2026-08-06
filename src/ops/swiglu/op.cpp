#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_ARGUMENT(out->ndim() == 2, "SwiGLU tensors must be 2D");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
