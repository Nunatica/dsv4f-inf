#pragma once
#include "ggml-backend.h"
#include <cstring>

// Strata-style write-through host backing and a bounded GPU residency cache.
// Host storage is deliberately not a graph operand: the scheduler must never
// upload the entire context. The owning llama_kv_cache outlives these graphs.
constexpr int GGML_DSV4F_KV_WRITE = 0x4b565701;
constexpr int GGML_DSV4F_KV_ATTN  = 0x4b564101;
struct ggml_dsv4f_kv_api {
    void * (*create)(ggml_backend_dev_t, ggml_tensor *, uint32_t);
    void (*destroy)(void *);
    void (*reset)(void *);
    size_t (*bytes)(void *);
};
using ggml_dsv4f_kv_api_fn = const ggml_dsv4f_kv_api * (*)();

inline bool ggml_dsv4f_kv_op(const ggml_tensor * t) {
    return t->op == GGML_OP_CUSTOM &&
        (t->op_params[9] == GGML_DSV4F_KV_WRITE || t->op_params[9] == GGML_DSV4F_KV_ATTN);
}
inline void * ggml_dsv4f_kv_state(const ggml_tensor * t) {
    void * p;
    std::memcpy(&p, t->op_params + 12, sizeof(p));
    return p;
}
inline void ggml_dsv4f_kv_cpu(ggml_tensor *, int, int, void *) {
    GGML_ABORT("KV streaming requires the CUDA/HIP backend and flash attention");
}
inline ggml_tensor * ggml_dsv4f_kv_write(ggml_context * ctx, void * state, ggml_tensor * rows, ggml_tensor * ids) {
    ggml_tensor * args[] = {rows, ids};
    auto * out = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 2, ggml_dsv4f_kv_cpu, 1, nullptr);
    out->op_params[9] = GGML_DSV4F_KV_WRITE;
    std::memcpy(out->op_params + 12, &state, sizeof(state));
    return out;
}
// q [dim, queries, heads, streams], raw [dim, rows, 1, streams].
inline ggml_tensor * ggml_dsv4f_kv_attn(ggml_context * ctx, void * state, int stream0,
        ggml_tensor * q, ggml_tensor * raw, ggml_tensor * mask, ggml_tensor * sinks, int bound, float scale, ggml_tensor * commit = nullptr) {
    ggml_tensor * args[] = {q, raw, mask, sinks, commit};
    auto * out = ggml_custom_4d(ctx, GGML_TYPE_F32, q->ne[0], q->ne[2], q->ne[1], q->ne[3],
                              args, commit ? 5 : sinks ? 4 : 3, ggml_dsv4f_kv_cpu, 1, nullptr);
    out->op_params[9] = GGML_DSV4F_KV_ATTN;
    out->op_params[10] = stream0;
    out->op_params[11] = bound;
    std::memcpy(out->op_params + 12, &state, sizeof(state));
    std::memcpy(out->op_params + 14, &scale, sizeof(scale));
    return out;
}
