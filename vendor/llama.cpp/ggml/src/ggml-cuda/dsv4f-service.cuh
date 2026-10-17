#pragma once
#include "common.cuh"
#include <vector>

// Layer-scoped resident experts and replacement slots. End every successful
// lease before leaving the custom op, including all GPU reads and uploads.
struct ggml_dsv4f_prefill_cache_plan {
    std::vector<const char *> experts;
    std::vector<bool> resident;
    size_t up_offset = 0, down_offset = 0;
    int hits = 0, borrowed = 0;
};
bool ggml_dsv4f_prefill_cache_begin(const ggml_tensor * g, const ggml_tensor * u,
    const ggml_tensor * d, int nk, int group, std::vector<int> & active,
    ggml_dsv4f_prefill_cache_plan & plan);
void ggml_dsv4f_prefill_cache_end(cudaStream_t stream);
bool ggml_dsv4f_service_supported(const ggml_tensor * node);
void ggml_dsv4f_service_compute(void * ctx, ggml_tensor * node);
void ggml_dsv4f_service_finalize(void * ctx, ggml_tensor * node);
void ggml_dsv4f_service_invalidate(const void * base, size_t bytes);
void ggml_dsv4f_service_statistics(uint64_t *, uint64_t *, uint64_t *);
bool ggml_dsv4f_service_resize(bool decode);
// Call before growing physical GPU allocations, outside graph capture when
// reclamation is needed. False means even surrendering the cache cannot fit.
bool ggml_dsv4f_reserve_vram(int device, size_t bytes, bool allow_reclaim = true);
void ggml_dsv4f_trim_workspace(ggml_backend_t backend);
