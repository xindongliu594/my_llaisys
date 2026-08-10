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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
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

struct Qwen2SequenceState {
    size_t capacity = 0;
    size_t cache_len = 0;
    std::vector<tensor_t> key_cache;
    std::vector<tensor_t> value_cache;
    std::vector<int64_t> token_history;
    std::mt19937_64 sampling_rng;
    uint64_t sampling_seed = 0;
    bool sampling_rng_initialized = false;
    size_t requested_logprobs = 0;
    int64_t selected_token = -1;
    float selected_logprob = 0.0f;
    std::vector<int64_t> top_logprob_tokens;
    std::vector<float> top_logprobs;
};

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;
    LlaisysQwen2Weights weights{};
    std::unique_ptr<Qwen2SequenceState> legacy_sequence;
    std::unordered_map<uint64_t, std::unique_ptr<Qwen2SequenceState>> sequences;

    LlaisysQwen2Model(const LlaisysQwen2Meta &meta_, llaisysDeviceType_t device_, int device_id_)
        : meta(meta_), device(device_), device_id(device_id_) {
        allocateWeights();
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

    std::unique_ptr<Qwen2SequenceState> createSequenceState(size_t capacity) const {
        CHECK_ARGUMENT(capacity > 0 && capacity <= meta.maxseq,
                       "Qwen2 sequence capacity is invalid");
        auto state = std::make_unique<Qwen2SequenceState>();
        state->capacity = capacity;
        state->key_cache.reserve(meta.nlayer);
        state->value_cache.reserve(meta.nlayer);
        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            state->key_cache.push_back(create({capacity, meta.nkvh, meta.dh}));
            state->value_cache.push_back(create({capacity, meta.nkvh, meta.dh}));
        }
        return state;
    }

    Qwen2SequenceState &legacyState() {
        if (!legacy_sequence) {
            legacy_sequence = createSequenceState(meta.maxseq);
        }
        return *legacy_sequence;
    }

    Qwen2SequenceState &sequence(uint64_t sequence_id) {
        auto found = sequences.find(sequence_id);
        CHECK_ARGUMENT(found != sequences.end(), "Unknown Qwen2 sequence");
        return *found->second;
    }

    void createSequence(uint64_t sequence_id, size_t capacity) {
        CHECK_ARGUMENT(sequences.find(sequence_id) == sequences.end(),
                       "Qwen2 sequence already exists");
        sequences.emplace(sequence_id, createSequenceState(capacity));
    }

    void destroySequence(uint64_t sequence_id) {
        sequences.erase(sequence_id);
    }

    void resetSequence(uint64_t sequence_id) {
        auto &state = sequence(sequence_id);
        state.cache_len = 0;
        state.token_history.clear();
        state.sampling_rng_initialized = false;
        state.selected_token = -1;
        state.top_logprob_tokens.clear();
        state.top_logprobs.clear();
    }

    void configureSequenceLogprobs(uint64_t sequence_id, size_t top_n) {
        CHECK_ARGUMENT(top_n <= 20, "At most 20 top logprobs are supported");
        auto &state = sequence(sequence_id);
        state.requested_logprobs = top_n;
        state.selected_token = -1;
        state.top_logprob_tokens.clear();
        state.top_logprobs.clear();
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

    tensor_t forward(Qwen2SequenceState &state, const int64_t *token_ids,
                     size_t token_count) {
        CHECK_ARGUMENT(token_ids != nullptr && token_count > 0, "Qwen2 inference requires input tokens");
        CHECK_ARGUMENT(state.cache_len + token_count <= state.capacity,
                       "Qwen2 KV cache capacity exceeded");

        auto token_tensor = create({token_count}, LLAISYS_DTYPE_I64);
        token_tensor->load(token_ids);
        auto hidden = create({token_count, meta.hs});
        llaisys::ops::embedding(hidden, token_tensor, tensor(weights.in_embed));

        std::vector<int64_t> positions(token_count);
        for (size_t i = 0; i < token_count; ++i) {
            positions[i] = static_cast<int64_t>(state.cache_len + i);
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

            copy(state.key_cache[layer]->slice(0, state.cache_len,
                                               state.cache_len + token_count),
                 rotated_key);
            copy(state.value_cache[layer]->slice(0, state.cache_len,
                                                 state.cache_len + token_count),
                 value);
            auto all_keys = state.key_cache[layer]->slice(
                0, 0, state.cache_len + token_count);
            auto all_values = state.value_cache[layer]->slice(
                0, 0, state.cache_len + token_count);

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

        state.cache_len += token_count;
        state.token_history.insert(state.token_history.end(), token_ids,
                                   token_ids + token_count);
        auto last_hidden = hidden->slice(0, token_count - 1, token_count);
        auto final_hidden = create({1, meta.hs});
        llaisys::ops::rms_norm(final_hidden, last_hidden, tensor(weights.out_norm_w), meta.epsilon);
        auto logits = create({1, meta.voc});
        llaisys::ops::linear(logits, final_hidden, tensor(weights.out_embed), nullptr);
        return logits;
    }

    int64_t inferGreedy(Qwen2SequenceState &state, const int64_t *token_ids,
                        size_t token_count) {
        auto logits = forward(state, token_ids, token_count);
        auto max_index = create({1}, LLAISYS_DTYPE_I64);
        auto max_value = create({1});
        llaisys::ops::argmax(max_index, max_value, logits->view({meta.voc}));

        int64_t result = 0;
        llaisys::core::context().setDevice(device, device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(
            &result, max_index->data(), sizeof(result),
            device == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2H);
        if (state.requested_logprobs > 0) {
            captureLogprobs(state, copyLogitsToHost(logits), result);
        }
        return result;
    }

    tensor_t forwardBatch(const std::vector<Qwen2SequenceState *> &states,
                          const int64_t *token_ids) {
        const size_t batch_size = states.size();
        CHECK_ARGUMENT(batch_size > 0 && token_ids != nullptr,
                       "Qwen2 batch inference requires inputs");
        for (const auto *state : states) {
            CHECK_ARGUMENT(state != nullptr && state->cache_len < state->capacity,
                           "Qwen2 batch sequence capacity exceeded");
        }

        auto token_tensor = create({batch_size}, LLAISYS_DTYPE_I64);
        token_tensor->load(token_ids);
        auto hidden = create({batch_size, meta.hs});
        llaisys::ops::embedding(hidden, token_tensor, tensor(weights.in_embed));

        std::vector<int64_t> positions(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            positions[i] = static_cast<int64_t>(states[i]->cache_len);
        }
        auto position_tensor = create({batch_size}, LLAISYS_DTYPE_I64);
        position_tensor->load(positions.data());

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            auto residual = hidden;
            auto normalized = create({batch_size, meta.hs});
            llaisys::ops::rms_norm(normalized, hidden,
                                   tensor(weights.attn_norm_w[layer]),
                                   meta.epsilon);

            auto query_2d = create({batch_size, meta.nh * meta.dh});
            auto key_2d = create({batch_size, meta.nkvh * meta.dh});
            auto value_2d = create({batch_size, meta.nkvh * meta.dh});
            llaisys::ops::linear(query_2d, normalized,
                                 tensor(weights.attn_q_w[layer]),
                                 tensor(weights.attn_q_b[layer]));
            llaisys::ops::linear(key_2d, normalized,
                                 tensor(weights.attn_k_w[layer]),
                                 tensor(weights.attn_k_b[layer]));
            llaisys::ops::linear(value_2d, normalized,
                                 tensor(weights.attn_v_w[layer]),
                                 tensor(weights.attn_v_b[layer]));

            auto query = query_2d->view({batch_size, meta.nh, meta.dh});
            auto key = key_2d->view({batch_size, meta.nkvh, meta.dh});
            auto value = value_2d->view({batch_size, meta.nkvh, meta.dh});
            auto rotated_query = create({batch_size, meta.nh, meta.dh});
            auto rotated_key = create({batch_size, meta.nkvh, meta.dh});
            llaisys::ops::rope(rotated_query, query, position_tensor, meta.theta);
            llaisys::ops::rope(rotated_key, key, position_tensor, meta.theta);

            auto attention = create({batch_size, meta.nh, meta.dh});
            for (size_t i = 0; i < batch_size; ++i) {
                auto &state = *states[i];
                copy(state.key_cache[layer]->slice(0, state.cache_len,
                                                   state.cache_len + 1),
                     rotated_key->slice(0, i, i + 1));
                copy(state.value_cache[layer]->slice(0, state.cache_len,
                                                     state.cache_len + 1),
                     value->slice(0, i, i + 1));
                auto all_keys = state.key_cache[layer]->slice(
                    0, 0, state.cache_len + 1);
                auto all_values = state.value_cache[layer]->slice(
                    0, 0, state.cache_len + 1);
                llaisys::ops::self_attention(
                    attention->slice(0, i, i + 1),
                    rotated_query->slice(0, i, i + 1), all_keys, all_values,
                    1.0f / std::sqrt(static_cast<float>(meta.dh)));
            }

            auto attention_2d = attention->view(
                {batch_size, meta.nh * meta.dh});
            auto attention_output = create({batch_size, meta.hs});
            llaisys::ops::linear(attention_output, attention_2d,
                                 tensor(weights.attn_o_w[layer]), nullptr);
            hidden = create({batch_size, meta.hs});
            llaisys::ops::add(hidden, residual, attention_output);

            residual = hidden;
            normalized = create({batch_size, meta.hs});
            llaisys::ops::rms_norm(normalized, hidden,
                                   tensor(weights.mlp_norm_w[layer]),
                                   meta.epsilon);
            auto gate = create({batch_size, meta.di});
            auto up = create({batch_size, meta.di});
            llaisys::ops::linear(gate, normalized,
                                 tensor(weights.mlp_gate_w[layer]), nullptr);
            llaisys::ops::linear(up, normalized,
                                 tensor(weights.mlp_up_w[layer]), nullptr);
            auto activated = create({batch_size, meta.di});
            llaisys::ops::swiglu(activated, gate, up);
            auto mlp_output = create({batch_size, meta.hs});
            llaisys::ops::linear(mlp_output, activated,
                                 tensor(weights.mlp_down_w[layer]), nullptr);
            hidden = create({batch_size, meta.hs});
            llaisys::ops::add(hidden, residual, mlp_output);
        }

        for (size_t i = 0; i < batch_size; ++i) {
            ++states[i]->cache_len;
            states[i]->token_history.push_back(token_ids[i]);
        }
        auto final_hidden = create({batch_size, meta.hs});
        llaisys::ops::rms_norm(final_hidden, hidden, tensor(weights.out_norm_w),
                               meta.epsilon);
        auto logits = create({batch_size, meta.voc});
        llaisys::ops::linear(logits, final_hidden, tensor(weights.out_embed),
                             nullptr);
        return logits;
    }

    void inferBatchGreedy(const std::vector<Qwen2SequenceState *> &states,
                          const int64_t *token_ids, int64_t *output_ids) {
        CHECK_ARGUMENT(output_ids != nullptr, "Qwen2 batch output must not be null");
        auto logits = forwardBatch(states, token_ids);
        for (size_t i = 0; i < states.size(); ++i) {
            auto max_index = create({1}, LLAISYS_DTYPE_I64);
            auto max_value = create({1});
            llaisys::ops::argmax(max_index, max_value,
                                 logits->slice(0, i, i + 1)->view({meta.voc}));
            llaisys::core::context().setDevice(device, device_id);
            llaisys::core::context().runtime().api()->memcpy_sync(
                output_ids + i, max_index->data(), sizeof(int64_t),
                device == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H
                                             : LLAISYS_MEMCPY_D2H);
            if (states[i]->requested_logprobs > 0) {
                captureLogprobs(
                    *states[i],
                    copyLogitsToHost(
                        logits->slice(0, i, i + 1)->view({meta.voc})),
                    output_ids[i]);
            }
        }
    }

    std::vector<float> copyLogitsToHost(tensor_t logits) const {
        const size_t count = logits->numel();
        std::vector<std::byte> raw(count * logits->elementSize());
        llaisys::core::context().setDevice(device, device_id);
        llaisys::core::context().runtime().api()->memcpy_sync(
            raw.data(), logits->data(), raw.size(),
            device == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2H);

        std::vector<float> scores(count);
        switch (meta.dtype) {
        case LLAISYS_DTYPE_F32: {
            const auto *source = reinterpret_cast<const float *>(raw.data());
            std::copy(source, source + count, scores.begin());
            break;
        }
        case LLAISYS_DTYPE_F16: {
            const auto *source = reinterpret_cast<const llaisys::fp16_t *>(raw.data());
            for (size_t i = 0; i < count; ++i) {
                scores[i] = llaisys::utils::cast<float>(source[i]);
            }
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            const auto *source = reinterpret_cast<const llaisys::bf16_t *>(raw.data());
            for (size_t i = 0; i < count; ++i) {
                scores[i] = llaisys::utils::cast<float>(source[i]);
            }
            break;
        }
        default: EXCEPTION_UNSUPPORTED_DATATYPE(meta.dtype);
        }
        return scores;
    }

    void captureLogprobs(Qwen2SequenceState &state,
                         const std::vector<float> &scores,
                         int64_t selected) const {
        if (state.requested_logprobs == 0 || scores.empty()) {
            return;
        }
        const float maximum = *std::max_element(scores.begin(), scores.end());
        double exponential_sum = 0.0;
        for (float score : scores) {
            exponential_sum += std::exp(static_cast<double>(score - maximum));
        }
        const double log_denominator =
            static_cast<double>(maximum) + std::log(exponential_sum);
        state.selected_token = selected;
        state.selected_logprob = static_cast<float>(
            static_cast<double>(scores.at(static_cast<size_t>(selected)))
            - log_denominator);

        std::vector<size_t> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        const size_t count = std::min(state.requested_logprobs, scores.size());
        std::partial_sort(
            indices.begin(), indices.begin() + count, indices.end(),
            [&scores](size_t left, size_t right) {
                return scores[left] > scores[right];
            });
        state.top_logprob_tokens.resize(count);
        state.top_logprobs.resize(count);
        for (size_t i = 0; i < count; ++i) {
            state.top_logprob_tokens[i] = static_cast<int64_t>(indices[i]);
            state.top_logprobs[i] = static_cast<float>(
                static_cast<double>(scores[indices[i]]) - log_denominator);
        }
    }

    int64_t sample(Qwen2SequenceState &state, tensor_t logits,
                   const LlaisysSamplingConfig &config) {
        CHECK_ARGUMENT(config.temperature > 0.0f, "Temperature must be positive");
        CHECK_ARGUMENT(config.top_p > 0.0f && config.top_p <= 1.0f,
                       "Top-p must be in (0, 1]");
        CHECK_ARGUMENT(config.repetition_penalty > 0.0f,
                       "Repetition penalty must be positive");

        auto scores = copyLogitsToHost(logits);
        const float inverse_temperature = 1.0f / config.temperature;
        for (float &score : scores) {
            score *= inverse_temperature;
        }

        if (config.repetition_penalty != 1.0f) {
            std::unordered_set<int64_t> seen(state.token_history.begin(),
                                             state.token_history.end());
            for (int64_t token : seen) {
                if (token < 0 || static_cast<size_t>(token) >= scores.size()) {
                    continue;
                }
                float &score = scores[static_cast<size_t>(token)];
                score = score < 0.0f ? score * config.repetition_penalty
                                     : score / config.repetition_penalty;
            }
        }

        std::vector<size_t> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        const size_t candidate_count = config.top_k == 0
            ? indices.size()
            : std::min(config.top_k, indices.size());
        std::partial_sort(indices.begin(), indices.begin() + candidate_count, indices.end(),
                          [&scores](size_t left, size_t right) {
                              return scores[left] > scores[right];
                          });
        indices.resize(candidate_count);

        const float maximum = scores[indices.front()];
        std::vector<double> weights(candidate_count);
        double total = 0.0;
        for (size_t i = 0; i < candidate_count; ++i) {
            weights[i] = std::exp(static_cast<double>(scores[indices[i]] - maximum));
            total += weights[i];
        }

        size_t kept = candidate_count;
        if (config.top_p < 1.0f) {
            double cumulative = 0.0;
            for (size_t i = 0; i < candidate_count; ++i) {
                cumulative += weights[i] / total;
                if (cumulative >= config.top_p) {
                    kept = i + 1;
                    break;
                }
            }
        }

        if (!state.sampling_rng_initialized || state.sampling_seed != config.seed) {
            state.sampling_rng.seed(config.seed);
            state.sampling_seed = config.seed;
            state.sampling_rng_initialized = true;
        }
        std::discrete_distribution<size_t> distribution(weights.begin(), weights.begin() + kept);
        const int64_t selected = static_cast<int64_t>(
            indices[distribution(state.sampling_rng)]);
        if (state.requested_logprobs > 0) {
            captureLogprobs(state, scores, selected);
        }
        return selected;
    }

    void inferBatchSample(const std::vector<Qwen2SequenceState *> &states,
                          const int64_t *token_ids,
                          const LlaisysSamplingConfig *configs,
                          int64_t *output_ids) {
        CHECK_ARGUMENT(configs != nullptr && output_ids != nullptr,
                       "Qwen2 batch sampling arguments must not be null");
        auto logits = forwardBatch(states, token_ids);
        for (size_t i = 0; i < states.size(); ++i) {
            output_ids[i] = sample(
                *states[i], logits->slice(0, i, i + 1)->view({meta.voc}),
                configs[i]);
        }
    }

    int64_t inferSample(Qwen2SequenceState &state, const int64_t *token_ids,
                        size_t token_count,
                        const LlaisysSamplingConfig &config) {
        if (config.top_k == 1 && config.repetition_penalty == 1.0f) {
            return inferGreedy(state, token_ids, token_count);
        }
        return sample(state, forward(state, token_ids, token_count), config);
    }
};

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                llaisysDeviceType_t device,
                                                int *device_ids,
                                                int ndevice) {
        try {
            CHECK_ARGUMENT(meta != nullptr, "Qwen2 metadata must not be null");
            CHECK_ARGUMENT(ndevice == 1 && device_ids != nullptr,
                           "Qwen2 currently supports exactly one device");
            return new LlaisysQwen2Model(*meta, device, device_ids[0]);
        } catch (...) {
            // C++ exceptions must never cross the C ABI boundary.
            return nullptr;
        }
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return model == nullptr ? nullptr : &model->weights;
    }

    void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
        if (model != nullptr && model->legacy_sequence) {
            model->legacy_sequence->cache_len = 0;
            model->legacy_sequence->token_history.clear();
            model->legacy_sequence->sampling_rng_initialized = false;
        }
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            return model->inferGreedy(model->legacyState(), token_ids, ntoken);
        } catch (...) {
            // Token ids are non-negative, so -1 is an unambiguous failure value.
            return -1;
        }
    }

    int64_t llaisysQwen2ModelInferSample(LlaisysQwen2Model *model, int64_t *token_ids,
                                         size_t ntoken, const LlaisysSamplingConfig *config) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            CHECK_ARGUMENT(config != nullptr, "Sampling config must not be null");
            return model->inferSample(model->legacyState(), token_ids, ntoken,
                                      *config);
        } catch (...) {
            return -1;
        }
    }

    int llaisysQwen2SequenceCreate(LlaisysQwen2Model *model,
                                   uint64_t sequence_id, size_t capacity) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            model->createSequence(sequence_id, capacity);
            return 1;
        } catch (...) {
            return 0;
        }
    }

    void llaisysQwen2SequenceDestroy(LlaisysQwen2Model *model,
                                     uint64_t sequence_id) {
        if (model != nullptr) {
            model->destroySequence(sequence_id);
        }
    }

    int llaisysQwen2SequenceReset(LlaisysQwen2Model *model,
                                  uint64_t sequence_id) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            model->resetSequence(sequence_id);
            return 1;
        } catch (...) {
            return 0;
        }
    }

    int llaisysQwen2SequenceConfigureLogprobs(
        LlaisysQwen2Model *model, uint64_t sequence_id, size_t top_n) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            model->configureSequenceLogprobs(sequence_id, top_n);
            return 1;
        } catch (...) {
            return 0;
        }
    }

    size_t llaisysQwen2SequenceGetLogprobs(
        LlaisysQwen2Model *model, uint64_t sequence_id,
        int64_t *selected_token, float *selected_logprob,
        int64_t *top_token_ids, float *top_logprobs, size_t capacity) {
        try {
            CHECK_ARGUMENT(model != nullptr && selected_token != nullptr
                           && selected_logprob != nullptr,
                           "Qwen2 logprob arguments must not be null");
            auto &state = model->sequence(sequence_id);
            CHECK_ARGUMENT(state.selected_token >= 0,
                           "Qwen2 logprobs are not available");
            *selected_token = state.selected_token;
            *selected_logprob = state.selected_logprob;
            const size_t count = std::min(capacity, state.top_logprobs.size());
            CHECK_ARGUMENT(count == 0 || (top_token_ids != nullptr
                           && top_logprobs != nullptr),
                           "Qwen2 top-logprob outputs must not be null");
            for (size_t i = 0; i < count; ++i) {
                top_token_ids[i] = state.top_logprob_tokens[i];
                top_logprobs[i] = state.top_logprobs[i];
            }
            return count;
        } catch (...) {
            return 0;
        }
    }

    int64_t llaisysQwen2SequenceInfer(LlaisysQwen2Model *model,
                                      uint64_t sequence_id,
                                      int64_t *token_ids, size_t ntoken) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            return model->inferGreedy(model->sequence(sequence_id), token_ids,
                                      ntoken);
        } catch (...) {
            return -1;
        }
    }

    int64_t llaisysQwen2SequenceInferSample(
        LlaisysQwen2Model *model, uint64_t sequence_id, int64_t *token_ids,
        size_t ntoken, const LlaisysSamplingConfig *config) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            CHECK_ARGUMENT(config != nullptr, "Sampling config must not be null");
            return model->inferSample(model->sequence(sequence_id), token_ids,
                                      ntoken, *config);
        } catch (...) {
            return -1;
        }
    }

    int llaisysQwen2BatchInfer(LlaisysQwen2Model *model,
                               const uint64_t *sequence_ids,
                               const int64_t *token_ids, size_t batch_size,
                               int64_t *output_ids) {
        try {
            CHECK_ARGUMENT(model != nullptr && sequence_ids != nullptr,
                           "Qwen2 batch arguments must not be null");
            std::vector<Qwen2SequenceState *> states;
            states.reserve(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                states.push_back(&model->sequence(sequence_ids[i]));
            }
            model->inferBatchGreedy(states, token_ids, output_ids);
            return 1;
        } catch (...) {
            return 0;
        }
    }

    int llaisysQwen2BatchInferSample(
        LlaisysQwen2Model *model, const uint64_t *sequence_ids,
        const int64_t *token_ids, size_t batch_size,
        const LlaisysSamplingConfig *configs, int64_t *output_ids) {
        try {
            CHECK_ARGUMENT(model != nullptr && sequence_ids != nullptr,
                           "Qwen2 batch arguments must not be null");
            std::vector<Qwen2SequenceState *> states;
            states.reserve(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                states.push_back(&model->sequence(sequence_ids[i]));
            }
            model->inferBatchSample(states, token_ids, configs, output_ids);
            return 1;
        } catch (...) {
            return 0;
        }
    }
}
