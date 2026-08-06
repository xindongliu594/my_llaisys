#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace {

using llaisys::tensor_t;

llaisysTensor_t wrap(tensor_t tensor) {
    return new LlaisysTensor{std::move(tensor)};
}

tensor_t tensor(llaisysTensor_t handle) {
    return handle->tensor;
}

} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;
    size_t cache_len;
    LlaisysQwen2Weights weights{};
    std::vector<tensor_t> key_cache;
    std::vector<tensor_t> value_cache;

    LlaisysQwen2Model(const LlaisysQwen2Meta &meta_, llaisysDeviceType_t device_, int device_id_)
        : meta(meta_), device(device_), device_id(device_id_), cache_len(0) {
        allocateWeights();
        key_cache.reserve(meta.nlayer);
        value_cache.reserve(meta.nlayer);
        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            key_cache.push_back(create({meta.maxseq, meta.nkvh, meta.dh}));
            value_cache.push_back(create({meta.maxseq, meta.nkvh, meta.dh}));
        }
    }

    ~LlaisysQwen2Model() {
        destroyWeights();
    }

    tensor_t create(const std::vector<size_t> &shape, llaisysDataType_t dtype) const {
        return llaisys::Tensor::create(shape, dtype, device, device_id);
    }

    tensor_t create(const std::vector<size_t> &shape) const {
        return create(shape, meta.dtype);
    }

    llaisysTensor_t createWeight(const std::vector<size_t> &shape) const {
        return wrap(create(shape));
    }

    void allocateWeights() {
        weights.in_embed = createWeight({meta.voc, meta.hs});
        weights.out_embed = createWeight({meta.voc, meta.hs});
        weights.out_norm_w = createWeight({meta.hs});

#define ALLOCATE_LAYER_ARRAY(NAME) weights.NAME = new llaisysTensor_t[meta.nlayer]
        ALLOCATE_LAYER_ARRAY(attn_norm_w);
        ALLOCATE_LAYER_ARRAY(attn_q_w);
        ALLOCATE_LAYER_ARRAY(attn_q_b);
        ALLOCATE_LAYER_ARRAY(attn_k_w);
        ALLOCATE_LAYER_ARRAY(attn_k_b);
        ALLOCATE_LAYER_ARRAY(attn_v_w);
        ALLOCATE_LAYER_ARRAY(attn_v_b);
        ALLOCATE_LAYER_ARRAY(attn_o_w);
        ALLOCATE_LAYER_ARRAY(mlp_norm_w);
        ALLOCATE_LAYER_ARRAY(mlp_gate_w);
        ALLOCATE_LAYER_ARRAY(mlp_up_w);
        ALLOCATE_LAYER_ARRAY(mlp_down_w);
#undef ALLOCATE_LAYER_ARRAY

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            weights.attn_norm_w[layer] = createWeight({meta.hs});
            weights.attn_q_w[layer] = createWeight({meta.nh * meta.dh, meta.hs});
            weights.attn_q_b[layer] = createWeight({meta.nh * meta.dh});
            weights.attn_k_w[layer] = createWeight({meta.nkvh * meta.dh, meta.hs});
            weights.attn_k_b[layer] = createWeight({meta.nkvh * meta.dh});
            weights.attn_v_w[layer] = createWeight({meta.nkvh * meta.dh, meta.hs});
            weights.attn_v_b[layer] = createWeight({meta.nkvh * meta.dh});
            weights.attn_o_w[layer] = createWeight({meta.hs, meta.nh * meta.dh});
            weights.mlp_norm_w[layer] = createWeight({meta.hs});
            weights.mlp_gate_w[layer] = createWeight({meta.di, meta.hs});
            weights.mlp_up_w[layer] = createWeight({meta.di, meta.hs});
            weights.mlp_down_w[layer] = createWeight({meta.hs, meta.di});
        }
    }

    void destroyWeights() {
        delete weights.in_embed;
        delete weights.out_embed;
        delete weights.out_norm_w;

#define DESTROY_LAYER_ARRAY(NAME)               \
        do {                                    \
            for (size_t i = 0; i < meta.nlayer; ++i) { \
                delete weights.NAME[i];         \
            }                                   \
            delete[] weights.NAME;              \
            weights.NAME = nullptr;             \
        } while (0)
        DESTROY_LAYER_ARRAY(attn_norm_w);
        DESTROY_LAYER_ARRAY(attn_q_w);
        DESTROY_LAYER_ARRAY(attn_q_b);
        DESTROY_LAYER_ARRAY(attn_k_w);
        DESTROY_LAYER_ARRAY(attn_k_b);
        DESTROY_LAYER_ARRAY(attn_v_w);
        DESTROY_LAYER_ARRAY(attn_v_b);
        DESTROY_LAYER_ARRAY(attn_o_w);
        DESTROY_LAYER_ARRAY(mlp_norm_w);
        DESTROY_LAYER_ARRAY(mlp_gate_w);
        DESTROY_LAYER_ARRAY(mlp_up_w);
        DESTROY_LAYER_ARRAY(mlp_down_w);
#undef DESTROY_LAYER_ARRAY
    }

    void copy(tensor_t destination, tensor_t source) const {
        CHECK_SAME_DTYPE(destination->dtype(), source->dtype());
        CHECK_ARGUMENT(destination->numel() == source->numel(), "Tensor copy size mismatch");
        llaisys::core::context().setDevice(device, device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(
            destination->data(), source->data(), source->numel() * source->elementSize(),
            LLAISYS_MEMCPY_D2D);
    }

    int64_t infer(const int64_t *token_ids, size_t token_count) {
        CHECK_ARGUMENT(token_ids != nullptr && token_count > 0, "Qwen2 inference requires input tokens");
        CHECK_ARGUMENT(cache_len + token_count <= meta.maxseq, "Qwen2 KV cache capacity exceeded");

        auto token_tensor = create({token_count}, LLAISYS_DTYPE_I64);
        token_tensor->load(token_ids);
        auto hidden = create({token_count, meta.hs});
        llaisys::ops::embedding(hidden, token_tensor, tensor(weights.in_embed));

        std::vector<int64_t> positions(token_count);
        for (size_t i = 0; i < token_count; ++i) {
            positions[i] = static_cast<int64_t>(cache_len + i);
        }
        auto position_tensor = create({token_count}, LLAISYS_DTYPE_I64);
        position_tensor->load(positions.data());

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            auto residual = hidden;
            auto normalized = create({token_count, meta.hs});
            llaisys::ops::rms_norm(normalized, hidden, tensor(weights.attn_norm_w[layer]), meta.epsilon);

            auto query_2d = create({token_count, meta.nh * meta.dh});
            auto key_2d = create({token_count, meta.nkvh * meta.dh});
            auto value_2d = create({token_count, meta.nkvh * meta.dh});
            llaisys::ops::linear(query_2d, normalized, tensor(weights.attn_q_w[layer]), tensor(weights.attn_q_b[layer]));
            llaisys::ops::linear(key_2d, normalized, tensor(weights.attn_k_w[layer]), tensor(weights.attn_k_b[layer]));
            llaisys::ops::linear(value_2d, normalized, tensor(weights.attn_v_w[layer]), tensor(weights.attn_v_b[layer]));

            auto query = query_2d->view({token_count, meta.nh, meta.dh});
            auto key = key_2d->view({token_count, meta.nkvh, meta.dh});
            auto value = value_2d->view({token_count, meta.nkvh, meta.dh});
            auto rotated_query = create({token_count, meta.nh, meta.dh});
            auto rotated_key = create({token_count, meta.nkvh, meta.dh});
            llaisys::ops::rope(rotated_query, query, position_tensor, meta.theta);
            llaisys::ops::rope(rotated_key, key, position_tensor, meta.theta);

            copy(key_cache[layer]->slice(0, cache_len, cache_len + token_count), rotated_key);
            copy(value_cache[layer]->slice(0, cache_len, cache_len + token_count), value);
            auto all_keys = key_cache[layer]->slice(0, 0, cache_len + token_count);
            auto all_values = value_cache[layer]->slice(0, 0, cache_len + token_count);

            auto attention = create({token_count, meta.nh, meta.dh});
            llaisys::ops::self_attention(attention, rotated_query, all_keys, all_values,
                                         1.0f / std::sqrt(static_cast<float>(meta.dh)));
            auto attention_2d = attention->view({token_count, meta.nh * meta.dh});
            auto attention_output = create({token_count, meta.hs});
            llaisys::ops::linear(attention_output, attention_2d, tensor(weights.attn_o_w[layer]), nullptr);
            hidden = create({token_count, meta.hs});
            llaisys::ops::add(hidden, residual, attention_output);

            residual = hidden;
            normalized = create({token_count, meta.hs});
            llaisys::ops::rms_norm(normalized, hidden, tensor(weights.mlp_norm_w[layer]), meta.epsilon);
            auto gate = create({token_count, meta.di});
            auto up = create({token_count, meta.di});
            llaisys::ops::linear(gate, normalized, tensor(weights.mlp_gate_w[layer]), nullptr);
            llaisys::ops::linear(up, normalized, tensor(weights.mlp_up_w[layer]), nullptr);
            auto activated = create({token_count, meta.di});
            llaisys::ops::swiglu(activated, gate, up);
            auto mlp_output = create({token_count, meta.hs});
            llaisys::ops::linear(mlp_output, activated, tensor(weights.mlp_down_w[layer]), nullptr);
            hidden = create({token_count, meta.hs});
            llaisys::ops::add(hidden, residual, mlp_output);
        }

        cache_len += token_count;
        auto last_hidden = hidden->slice(0, token_count - 1, token_count);
        auto final_hidden = create({1, meta.hs});
        llaisys::ops::rms_norm(final_hidden, last_hidden, tensor(weights.out_norm_w), meta.epsilon);
        auto logits = create({1, meta.voc});
        llaisys::ops::linear(logits, final_hidden, tensor(weights.out_embed), nullptr);
        auto max_index = create({1}, LLAISYS_DTYPE_I64);
        auto max_value = create({1});
        llaisys::ops::argmax(max_index, max_value, logits->view({meta.voc}));

        int64_t result = 0;
        llaisys::core::context().setDevice(device, device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(
            &result, max_index->data(), sizeof(result),
            device == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2H);
        return result;
    }
};

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                llaisysDeviceType_t device,
                                                int *device_ids,
                                                int ndevice) {
        CHECK_ARGUMENT(meta != nullptr, "Qwen2 metadata must not be null");
        CHECK_ARGUMENT(ndevice == 1 && device_ids != nullptr, "Qwen2 currently supports exactly one device");
        return new LlaisysQwen2Model(*meta, device, device_ids[0]);
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return &model->weights;
    }

    void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
        model->cache_len = 0;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
        return model->infer(token_ids, ntoken);
    }
}
