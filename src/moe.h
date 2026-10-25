#pragma once
#include "ggml.h"
ggml_tensor * dsv4f_moe(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids,
                       ggml_tensor * weights, ggml_tensor * gate, ggml_tensor * up,
                       ggml_tensor * down, float limit, ggml_tensor ** prefill_scratch = nullptr);
ggml_tensor * dsv4f_moe_join(ggml_context * ctx, ggml_tensor * routed, ggml_tensor * shared);

// V4 sqrt-softplus router; preserves the F16-minimum normalization clamp.
void dsv4f_router(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * bias,
                  int nk, bool normalize, float scale, ggml_tensor ** ids, ggml_tensor ** probabilities);
