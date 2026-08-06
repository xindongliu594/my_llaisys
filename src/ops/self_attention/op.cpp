#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia_common.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "Self-attention tensors must be 3D");
    CHECK_ARGUMENT(q->shape()[0] <= k->shape()[0] && k->shape()[0] == v->shape()[0],
                   "Self-attention sequence lengths are invalid");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0 && k->shape()[1] == v->shape()[1],
                   "Self-attention head counts are invalid");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "Query and key head dimensions must match");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0]
                       && attn_val->shape()[1] == q->shape()[1]
                       && attn_val->shape()[2] == v->shape()[2],
                   "Self-attention output shape is invalid");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self-attention tensors must be contiguous");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::selfAttention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), q->shape()[0], k->shape()[0],
                                  q->shape()[1], k->shape()[1], q->shape()[2],
                                  v->shape()[2], scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::selfAttention(attn_val->data(), q->data(), k->data(), v->data(),
                                     attn_val->dtype(), q->shape()[0], k->shape()[0],
                                     q->shape()[1], k->shape()[1], q->shape()[2],
                                     v->shape()[2], scale);
#endif
    default: EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
