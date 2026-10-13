#pragma once
#include "common.cuh"
bool ggml_cuda_dsv4f_prefill_supported(const ggml_tensor * node);
void ggml_cuda_dsv4f_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * node);
