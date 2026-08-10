#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    struct LlaisysSamplingConfig {
        float temperature;
        size_t top_k;
        float top_p;
        float repetition_penalty;
        uint64_t seed;
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export void llaisysQwen2ModelReset(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    __export int64_t llaisysQwen2ModelInferSample(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, const LlaisysSamplingConfig * config);

    __export int llaisysQwen2SequenceCreate(struct LlaisysQwen2Model *model, uint64_t sequence_id, size_t capacity);

    __export void llaisysQwen2SequenceDestroy(struct LlaisysQwen2Model *model, uint64_t sequence_id);

    __export int llaisysQwen2SequenceReset(struct LlaisysQwen2Model *model, uint64_t sequence_id);

    __export int llaisysQwen2SequenceConfigureLogprobs(struct LlaisysQwen2Model *model, uint64_t sequence_id, size_t top_n);

    __export size_t llaisysQwen2SequenceGetLogprobs(struct LlaisysQwen2Model *model, uint64_t sequence_id, int64_t *selected_token, float *selected_logprob, int64_t *top_token_ids, float *top_logprobs, size_t capacity);

    __export int64_t llaisysQwen2SequenceInfer(struct LlaisysQwen2Model *model, uint64_t sequence_id, int64_t *token_ids, size_t ntoken);

    __export int64_t llaisysQwen2SequenceInferSample(struct LlaisysQwen2Model *model, uint64_t sequence_id, int64_t *token_ids, size_t ntoken, const LlaisysSamplingConfig *config);

    __export int llaisysQwen2BatchInfer(struct LlaisysQwen2Model *model, const uint64_t *sequence_ids, const int64_t *token_ids, size_t batch_size, int64_t *output_ids);

    __export int llaisysQwen2BatchInferSample(struct LlaisysQwen2Model *model, const uint64_t *sequence_ids, const int64_t *token_ids, size_t batch_size, const LlaisysSamplingConfig *configs, int64_t *output_ids);
}
#endif // LLAISYS_MODELS_QWEN2_H
