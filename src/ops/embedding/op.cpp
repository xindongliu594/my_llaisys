#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1 && index->dtype() == LLAISYS_DTYPE_I64,
                   "Embedding indices must be a 1D int64 tensor");
    CHECK_ARGUMENT(weight->ndim() == 2 && out->ndim() == 2, "Embedding weight and output must be 2D");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0] && out->shape()[1] == weight->shape()[1],
                   "Embedding output shape is invalid");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const auto *indices = reinterpret_cast<const int64_t *>(index->data());
        for (size_t i = 0; i < index->numel(); ++i) {
            CHECK_ARGUMENT(indices[i] >= 0 && static_cast<size_t>(indices[i]) < weight->shape()[0],
                           "Embedding index is out of range");
        }
        return cpu::embedding(out->data(), indices, weight->data(), index->numel(),
                              weight->shape()[1], weight->elementSize());
    }
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
