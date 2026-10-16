#pragma once
#include "common.cuh"
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
