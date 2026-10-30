#pragma once

#include "ggml-backend.h"
#include <stdint.h>
#include <stddef.h>

// MoE expert cache — dynamic VRAM cache for MoE expert weights on CPU-resident
// MUL_MAT_ID. Integration point is the CPU mul_mat_id kernel itself: thread 0
// dispatches cached expert rows to the GPU while the remaining threads compute
// the uncached rows, then results are collected into dst before the node ends.
//
// This table is the bridge between ggml-cpu (which cannot link CUDA) and the
// CUDA backend (which registers the implementations at backend-reg time).
// begin/plan/dispatch/collect for one node are called from a single thread
// (ith == 0); the implementation may assume no concurrent node processing.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_moe_cache_api {
    // Decide whether the cache engages for this MUL_MAT_ID node.
    // Returns the device id to use (>= 0) or -1 to stay on the pure-CPU path.
    // Performs lazy per-device initialization on first use and selects the
    // internal slot pool matching (expert_size, wtype).
    //   tensor_name: src0->name (stable cache key source, e.g. "blk.7.ffn_up_exps.weight")
    //   host_base:   src0->data (source for async inserts)
    //   expert_size: src0->nb[2] (bytes per expert)
    //   n_in/n_out:  src0->ne[0] / src0->ne[1]
    //   wtype:       src0->type
    //   n_expert:    src0->ne[2]
    //   n_tokens:    ids->ne[1] (cache engages only when == 1)
    int (*begin)(const char * tensor_name, const void * host_base, size_t expert_size,
                 int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens);

    // For each of the n_ids expert ids: slot_idx[k] = cache slot index (hit) or
    // -1 (miss; CPU computes the row, an async insert may be enqueued).
    // Returns the number of hits.
    int (*plan)(int dev, const int32_t * ids, int n_ids, int32_t * slot_idx);

    // One batched GPU launch computing all n_hits rows:
    //   out_row[i] (n_out floats) = W[slot_idx_compact[i]] . act_rows[i]
    // act_rows are host fp32 pointers (they may all be the same row for
    // gate/up-style nodes; distinct rows for down-style nodes).
    // Asynchronous; results are pulled into dst by collect().
    void (*dispatch)(int dev, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                     const int32_t * slot_idx_compact, const float * const * act_rows);

    // Synchronize the device's compute stream and copy the n_hits result rows
    // (in dispatch order) into dst_rows[0..n_hits-1] (n_out floats each).
    void (*collect)(int dev, int n_hits, float * const * dst_rows, int64_t n_out);

    // Periodic stats logging (rate-limited internally).
    void (*stats)(void);

    // ---- GPU-resident dst handoff (down-projection round-trip elimination) ----
    // The scheduler offers the GPU-side copy tensor of a CPU MUL_MAT_ID dst
    // BEFORE the CPU split runs. If the cache then computes that node, it
    // scatters its GPU rows directly into gpu_copy_data (async) instead of
    // pulling them to the host.
    void (*redirect_offer)(const void * host_dst_data, size_t nb1, int64_t n_rows,
                           void * gpu_copy_data, void * consumer_backend);
    // Called by the scheduler at the consumer split's input-copy site, after
    // the CPU split completed. Returns 1 if the cache fully populated the GPU
    // copy (it uploads the CPU-computed miss rows here and installs a
    // stream-order dependency on the consumer backend) — the scheduler must
    // then SKIP its own copy. Returns 0 for the normal copy path.
    int (*redirect_finalize)(const void * host_dst_data, void * consumer_backend);

    // ---- fused gate+up+GLU path ----
    // Called by the CPU GLU kernel (swiglu split variant) from EVERY thread:
    // returns the bitmask of dst rows the cache computed on the GPU (fused
    // silu(gate)*up) — those rows must be SKIPPED by the CPU loop. Thread 0
    // (ith == 0) additionally synchronizes the fused chain and scatters the
    // GPU rows into dst before returning. Returns 0 when the cache has no
    // pending fused work for this (src0, src1) pair.
    unsigned long long (*glu_hits)(const void * src0_data, const void * src1_data,
                                   void * dst_data, size_t dst_nb1, int ith);

    // Host weight buffer teardown notification (model unload): drops queued
    // insert jobs sourced from the range and resets per-block tensor-base
    // learning. Must be called before the memory is unmapped/freed.
    void (*invalidate)(const void * base, size_t size);

    // Node wall-time sample for the bail-out judge. code is begin()'s return
    // value: -3 = pure-CPU baseline sample, >= 0 = cache-engaged sample.
    void (*node_time)(int code, int64_t wall_us);
};

// Zero-initialized in ggml-backend.cpp; populated by the CUDA backend in
// ggml_backend_cuda_reg() when the cache is enabled (GGML_CUDA_MOE_CACHE=1).
extern struct ggml_moe_cache_api ggml_moe_cache;
struct ggml_dsv4f_expert_api {
    uint64_t (*dispatch)(const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *,
                         const float *, const int32_t *, const float *, int, float);
    void (*collect)(float *, int);
    void (*finish)(void);
    void (*invalidate)(const void *, size_t);
    // The scheduler offers a unique GPU consumer before the CPU custom op runs.
    // collect then leaves GPU hits on-device; finalize uploads only CPU misses
    // and combines both branches on the consumer's stream.
    void (*offer)(const void * host, void * device, ggml_backend_t consumer);
    bool (*finalize)(const void * host, ggml_backend_t consumer);
    // Host weight operands are descriptors, never scheduler upload inputs.
    bool (*gpu_supported)(const struct ggml_tensor *);
    void (*gpu_compute)(void * cuda_context, struct ggml_tensor *);
    void (*gpu_finalize)(void * cuda_context, struct ggml_tensor *);
    void (*gpu_statistics)(uint64_t * routes, uint64_t * hits, uint64_t * admissions);
    // Quiescent phase boundaries only: return/grow cache blocks according to
    // available headroom, and discard backend graphs/temporary allocations.
    bool (*gpu_resize)(bool decode);
    void (*gpu_trim_workspace)(ggml_backend_t backend);
    void (*gpu_drain)();
    size_t (*gpu_warm_start)();
};
typedef void (*ggml_dsv4f_cpu_fn)(const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *,
                                 const float *, const int32_t *, const float *, int, uint64_t, float, float *);
GGML_BACKEND_API ggml_dsv4f_cpu_fn ggml_dsv4f_cpu_experts;
#define GGML_DSV4F_MOE_TAG 0x44535634
#define GGML_DSV4F_JOIN_TAG 0x44535635
#define GGML_DSV4F_ROUTER_TAG 0x44535636
GGML_BACKEND_API struct ggml_dsv4f_expert_api ggml_dsv4f_experts;

// Optional process-local routing history, owned by the engine. Handles remain
// valid until model unload. A row identifies a layer + expert format; projection
// tensors share the same row. Observe actual routes once, never weight copies.
struct ggml_dsv4f_routing_api {
    void * (*bind)(const struct ggml_tensor *);
    void (*observe)(void *, const int32_t *, const float *, int);
    intptr_t (*count)(const void *, int);
};
GGML_BACKEND_API struct ggml_dsv4f_routing_api ggml_dsv4f_routing;


#ifdef __cplusplus
}
#endif

// Direct SSD/RAM expert source. Leases keep host bytes alive through CPU work or DMA.
struct ggml_dsv4f_host_weights_stats {
    uint64_t resident_bytes, transient_bytes, reads, read_bytes, evictions;
    uint64_t transient_reads, transient_read_bytes;
};
struct ggml_dsv4f_host_weights_api {
    const void * (*acquire)(const void *, size_t);
    void (*release)(const void *);
    // Non-blocking hint. The direct-weight backend may populate its RAM cache
    // in the background; resident backends leave this null.
    void (*prefetch)(const void *, size_t);
    // All-or-nothing batch of complete expert-plane leases. Cold planes can
    // read concurrently into their final RAM-cache allocations (no bounce).
    void (*acquire_many)(const void * const *, const size_t *, const void **, size_t);
    // Prefill scan: preserve resident weights, bypass admission on overflow.
    // Transient allocations remain leased through H2D, then return to the OS.
    void (*acquire_prefill)(const void * const *, const size_t *, const void **, size_t);
    void (*statistics)(struct ggml_dsv4f_host_weights_stats *);
    // Snapshot complete-plane residency without touching LRU or reading SSD.
    void (*resident_many)(const void * const *, const size_t *, bool *, size_t);
    // Hand a plane to the GPU cache. Live CPU/DMA leases remain valid, but
    // its RAM copy is removed at the last release (or immediately if idle).
    void (*discard)(const void *, size_t);
};
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API struct ggml_dsv4f_host_weights_api ggml_dsv4f_host_weights;
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
struct ggml_dsv4f_host_lease {
    const void * data;
    explicit ggml_dsv4f_host_lease(const void * p, size_t n) :
        data(ggml_dsv4f_host_weights.acquire ? ggml_dsv4f_host_weights.acquire(p,n) : p) {}
    ~ggml_dsv4f_host_lease(){if(ggml_dsv4f_host_weights.release)ggml_dsv4f_host_weights.release(data);}
    ggml_dsv4f_host_lease(const ggml_dsv4f_host_lease &) = delete;
    ggml_dsv4f_host_lease & operator=(const ggml_dsv4f_host_lease &) = delete;
};
#endif
