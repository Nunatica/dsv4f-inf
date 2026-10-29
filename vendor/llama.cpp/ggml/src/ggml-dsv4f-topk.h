#pragma once
#include "ggml.h"

// TOP_K's normal CPU/CUDA implementations ignore this indexer-only hint.
static constexpr int GGML_DSV4F_INDEXER_TOPK = 0x49544b01;
static inline ggml_tensor * ggml_dsv4f_indexer_top_k(ggml_context * ctx, ggml_tensor * scores, int k) {
    ggml_tensor * result = ggml_top_k(ctx, scores, k);
    result->op_params[1] = GGML_DSV4F_INDEXER_TOPK;
    return result;
}
