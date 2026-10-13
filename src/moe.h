#pragma once
#include "ggml.h"
ggml_tensor * dsv4f_moe(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids,
                       ggml_tensor * weights, ggml_tensor * gate, ggml_tensor * up,
                       ggml_tensor * down, float limit, ggml_tensor ** prefill_scratch = nullptr);
ggml_tensor * dsv4f_moe_join(ggml_context * ctx, ggml_tensor * routed, ggml_tensor * shared);
