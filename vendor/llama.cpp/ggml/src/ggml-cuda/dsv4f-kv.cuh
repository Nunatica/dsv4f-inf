#pragma once
#include "common.cuh"
#include "../ggml-dsv4f-kv.h"

struct dsv4f_kv_state {
    int device, rows, slots, streams;
    ggml_type type;
    size_t row_bytes;
    char * host; // mapped, pinned authoritative backing, owned by llama_kv_cache
    char * host_cpu; // CPU address for DMA staging
    char * cache = nullptr;
    int * map = nullptr;
    int * owner = nullptr;
    unsigned * stamp = nullptr;
    int * ref = nullptr;
    unsigned * ctl = nullptr; // epoch, CLOCK hand
};
// Reserve staging in the graph allocation before the expert-cache budget is
// computed. Grouping bounds staging independently of total context length.
struct dsv4f_kv_scratch {
    int group, capacity;
    bool prefill, indexed;
    size_t size, ids = 0, misses = 0, keys, masks = 0, compressed = 0;
    size_t take(size_t bytes) {
        const size_t result = size;
        size += GGML_PAD(bytes, size_t(256));
        return result;
    }
    explicit dsv4f_kv_scratch(const ggml_tensor * dst) {
        const auto * state = (const dsv4f_kv_state *) ggml_dsv4f_kv_state(dst);
        const auto * q = dst->src[0];
        const auto * raw = dst->src[1];
        const int selected = dst->op_params[11] - raw->ne[1];
        GGML_ASSERT(selected > 0 && (!state || selected <= state->slots));
        prefill = q->ne[1] > 8;
        indexed = dst->src[5] != nullptr;
        size = GGML_PAD(ggml_nbytes(dst), size_t(256));
        if (indexed) {
            group = prefill ? std::min<int64_t>(32, q->ne[1]) : 1;
            capacity = GGML_PAD(std::min<int64_t>(raw->ne[1],dst->op_params[15]) + dst->src[5]->ne[0], int64_t(256));
            ids = take(size_t(group)*capacity*sizeof(int));
            misses = take(size_t(capacity)*sizeof(int));
            keys = take(size_t(group)*capacity*512*sizeof(half));
            masks = take(size_t(group)*capacity*sizeof(half));
            if (prefill && state) compressed = take(size_t(dst->src[6]->ne[0])*state->row_bytes);
        } else if (prefill) {
            // Strata stages the current layer for prefill. The full history is
            // temporary workspace shared across layers, not persistent VRAM KV.
            group = q->ne[1];
            capacity = dst->src[2]->ne[0];
            keys = take(size_t(capacity)*512*sizeof(half));
            GGML_ASSERT(state);
            compressed = take(size_t(capacity - raw->ne[1])*state->row_bytes);
        } else {
            GGML_ASSERT(state);
            group = std::min<int64_t>(q->ne[1], std::min(8, state->slots/selected));
            capacity = GGML_PAD(std::min<int64_t>(dst->src[2]->ne[0], raw->ne[1] + int64_t(group)*selected), int64_t(256));
            ids = take(size_t(capacity)*sizeof(int));
            misses = take(size_t(capacity)*sizeof(int));
            keys = take(size_t(capacity)*512*sizeof(half));
            masks = take(size_t(capacity)*group*sizeof(half));
        }
    }
};
void * ggml_cuda_dsv4f_kv_create(int device, ggml_tensor * host, uint32_t slots);
void ggml_cuda_dsv4f_kv_destroy(void * state);
void ggml_cuda_dsv4f_kv_reset(void * state);
size_t ggml_cuda_dsv4f_kv_bytes(void * state);
void ggml_cuda_dsv4f_kv_write(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_dsv4f_kv_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
