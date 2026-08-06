#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(in->ndim() == 3 && in->shape()[2] % 2 == 0,
                   "RoPE input must be 3D with an even head dimension");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->dtype() == LLAISYS_DTYPE_I64
                       && pos_ids->shape()[0] == in->shape()[0],
                   "RoPE position ids must be a matching 1D int64 tensor");
    CHECK_ARGUMENT(theta > 0.0f, "RoPE theta must be positive");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
