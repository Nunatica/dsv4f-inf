#include "dsv4f-prefill.cuh"
#include "dsv4f-service.cuh"
#include "../ggml-dsv4f-prefill.h"
#include "../ggml-backend-moe-cache.h"
#include "mmq.cuh"
#include "quantize.cuh"
#include "unary.cuh"

#include <chrono>
#include <cstring>
#include <numeric>
#include <vector>

namespace {
struct prefill_state {
    int device;
    cudaStream_t copy = nullptr;
    cudaEvent_t ready[2] = {}, used[2] = {};
    cudaEvent_t done = nullptr;
    bool pending = false;
    int32_t * host = nullptr;
    size_t capacity = 0;
    bool announced = false;
    explicit prefill_state(int dev) : device(dev) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&used[i], cudaEventDisableTiming));
        }
    }
    ~prefill_state() {
        ggml_cuda_set_device(device);
        if (pending) CUDA_CHECK(cudaEventSynchronize(done));
        CUDA_CHECK(cudaStreamSynchronize(copy));
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaEventDestroy(ready[i]));
            CUDA_CHECK(cudaEventDestroy(used[i]));
        }
        CUDA_CHECK(cudaStreamDestroy(copy));
        CUDA_CHECK(cudaEventDestroy(done));
        if (host) CUDA_CHECK(cudaFreeHost(host));
    }
    void reserve(size_t count) {
        if (count <= capacity) return;
        if (host) CUDA_CHECK(cudaFreeHost(host));
        CUDA_CHECK(cudaMallocHost((void **) &host, count * sizeof(int32_t)));
        capacity = count;
    }
};

__global__ void activate(float * gate, const float * up, int64_t begin, int64_t count, float limit) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) gate[begin+i] = ggml_cuda_op_swiglu_clamp_single(gate[begin+i], up[begin+i], limit);
}

__global__ void combine(const float * sorted, const int32_t * inverse, const float * weights,
                        float * out, int ne, int nk, int nt, int ws1, int ws2) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(ne) * nt) return;
    const int t = i / ne, e = i % ne;
    float sum = 0;
    // Keep the original selected-expert order, independent of staging order.
    for (int j = 0; j < nk; ++j) sum += sorted[int64_t(inverse[t*nk+j])*ne+e] * weights[t*ws2+j*ws1];
    out[i] = sum;
}

void product(ggml_backend_cuda_context & ctx, const void * w, const void * q, float * out,
             const int32_t * identity, const int32_t * bounds, int experts, int max_rows,
             int total_rows, int cols, int rows, size_t stride,
             const char * const * expert_ptrs = nullptr, size_t expert_offset = 0) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    int best = 0, tiles = INT_MAX;
    for (int j = 8; j <= 128 && tiles > 1; j += 8) {
        const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_MXFP4, j, false, cc, GGML_PREC_Q8);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > ggml_cuda_info().devices[ctx.device].smpbo) continue;
        const int n = (max_rows + j - 1) / j;
        if (n < tiles) { tiles = n; best = j; }
    }
    GGML_ASSERT(best > 0);
    const mmq_args args = {
        (const char *) w, GGML_TYPE_MXFP4, (const int *) q, identity, bounds, out, nullptr,
        cols, rows, total_rows, cols / QK_MXFP4, total_rows, rows,
        experts, experts, int64_t(stride / sizeof(block_mxfp4)), 0, 0,
        1, 1, 0, 0, 0, max_rows, best, expert_ptrs, expert_offset
    };
    mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q8>(ctx, args, ctx.stream());
}
}

bool ggml_cuda_dsv4f_prefill_supported(const ggml_tensor * n) {
    if (n->op != GGML_OP_CUSTOM || n->op_params[9] != GGML_DSV4F_MOE_TAG || n->op_params[11] != 1) return false;
    const auto * x = n->src[0], * ids = n->src[1], * weights = n->src[2];
    const auto * g = n->src[3], * u = n->src[4], * d = n->src[5];
    if (!x || !ids || !weights || !g || !u || !d || !n->src[6]) return false;
    const int cc = ggml_cuda_info().devices[0].cc;
    // The initial path keeps the target's native MXFP4 + Q8_1 arithmetic.
    return x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && weights->type == GGML_TYPE_F32 &&
        x->ne[1] >= 32 && x->ne[2] == 1 && x->ne[3] == 1 && ids->ne[0] > 0 && ids->ne[0] <= 64 &&
        ids->ne[1] == x->ne[1] && weights->ne[0] == 1 && weights->ne[1] == ids->ne[0] && weights->ne[2] == x->ne[1] &&
        ggml_is_contiguous(x) && ggml_is_contiguous(ids) && ggml_is_contiguous(weights) &&
        g->type == GGML_TYPE_MXFP4 && u->type == g->type && d->type == g->type &&
        !g->extra && !u->extra && !d->extra && ggml_is_contiguous(g) && ggml_is_contiguous(u) && ggml_is_contiguous(d) &&
        ggml_are_same_shape(g, u) && g->ne[0] == x->ne[0] && d->ne[0] == g->ne[1] && d->ne[1] == g->ne[0] &&
        d->ne[2] == g->ne[2] && g->ne[2] > 0 && g->ne[2] <= 256 && g->ne[3] == 1 && d->ne[3] == 1 &&
        g->ne[0] % 512 == 0 && g->ne[1] % 512 == 0 &&
        g->buffer && u->buffer && d->buffer && ggml_backend_buffer_is_host(g->buffer) &&
        ggml_backend_buffer_is_host(u->buffer) && ggml_backend_buffer_is_host(d->buffer) &&
        !blackwell_mma_available(cc) && ggml_cuda_should_use_mmq(g->type, cc, x->ne[1], g->ne[2]);
}

void ggml_cuda_dsv4f_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
    GGML_ASSERT(ggml_cuda_dsv4f_prefill_supported(node));
    const auto * x = node->src[0], * ids = node->src[1], * weights = node->src[2];
    const auto * g = node->src[3], * u = node->src[4], * d = node->src[5];
    const int nt = x->ne[1], ne = x->ne[0], nf = g->ne[1], nk = ids->ne[0], nex = g->ne[2], nr = nt*nk;
    const int group = node->op_params[12];
    const ggml_dsv4f_prefill_layout layout(g, u, d, nr, group);
    GGML_ASSERT(ggml_nbytes(node->src[6]) >= layout.size);
    char * scratch = (char *) node->src[6]->data;
    auto * gate = (float *) (scratch+layout.gate), * up = (float *) (scratch+layout.up);
    auto * down = (float *) (scratch+layout.down);
    void * xq = scratch+layout.xq, * hq = scratch+layout.hq;
    auto * inverse = (int32_t *) (scratch+layout.inverse), * identity = (int32_t *) (scratch+layout.identity);
    auto * bounds = (int32_t *) (scratch+layout.bounds);
    if (!ctx.dsv4f_prefill) ctx.dsv4f_prefill = std::make_shared<prefill_state>(ctx.device);
    auto & state = *static_cast<prefill_state *>(ctx.dsv4f_prefill.get());
    if (!state.announced) {
        std::fprintf(stderr, "[dsv4f-prefill] streamed MXFP4 group=%d ring=2 scratch=%.2f MiB\n", group, layout.size/1048576.);
        state.announced = true;
    }
    const auto start = std::chrono::steady_clock::now();
    // This also retires the previous layer's metadata reads and both staging
    // slots before the pinned plan and events are reused.
    if (state.pending) CUDA_CHECK(cudaEventSynchronize(state.done));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    state.reserve(3*size_t(nr) + 128 + nex + 1);
    int32_t * hids = state.host, * hinv = hids+nr, * hidentity = hinv+nr, * hbounds = hidentity+nr+128;
    CUDA_CHECK(cudaMemcpyAsync(hids, ids->data, nr*sizeof(int32_t), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    std::vector<int> counts(nex, 0), active;
    for (int i = 0; i < nr; ++i) { GGML_ASSERT(hids[i] >= 0 && hids[i] < nex); ++counts[hids[i]]; }
    for (int e = 0; e < nex; ++e) if (counts[e]) active.push_back(e);
    ggml_dsv4f_prefill_cache_plan cache_plan;
    const char * cache_env = std::getenv("DSV4F_PREFILL_CACHE");
    const char * min_env = std::getenv("DSV4F_PREFILL_CACHE_MIN_TOKENS");
    const int min_tokens = min_env ? std::max(1, std::atoi(min_env)) : 4096;
    const bool cached = cache_env && std::atoi(cache_env) != 0 && nt >= min_tokens && ctx.device == 0 &&
        ggml_dsv4f_prefill_cache_begin(g, u, d, nk, group, active, cache_plan);
    auto ** expert_ptrs = (const char **) (scratch+layout.expert_ptrs);
    if (cached) {
        // Synchronous metadata copy: vector storage may be destroyed on return.
        CUDA_CHECK(cudaMemcpy(expert_ptrs, cache_plan.experts.data(), active.size()*sizeof(char *), cudaMemcpyHostToDevice));
        static bool announced_cache = false;
        if (!announced_cache) {
            std::fprintf(stderr, "[dsv4f-prefill-cache] enabled min_tokens=%d hits=%d borrowed=%d direct MMQ pointers\n",
                         min_tokens, cache_plan.hits, cache_plan.borrowed);
            announced_cache = true;
        }
    }
    hbounds[0] = 0;
    for (size_t i = 0; i < active.size(); ++i) hbounds[i+1] = hbounds[i]+counts[active[i]];
    std::vector<int> cursor(nex);
    for (size_t e = 0; e < active.size(); ++e) cursor[active[e]] = hbounds[e];
    for (int i = 0; i < nr; ++i) hinv[i] = cursor[hids[i]]++;
    std::iota(hidentity, hidentity+nr+128, 0);
    CUDA_CHECK(cudaMemcpyAsync(inverse, hinv, nr*sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(identity, hidentity, (nr+128)*sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(bounds, hbounds, (active.size()+1)*sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream()));
    quantize_scatter_mmq_q8_1_cuda((const float *) x->data, inverse, xq, g->type, ne, ne,
                                 GGML_PAD(ne, 512), nt, nr, nk, ctx.stream());
    float limit;
    std::memcpy(&limit, &node->op_params[8], sizeof(limit));
    const int groups = (active.size()+group-1)/group;
    uint64_t copied = 0;
    auto upload = [&](int gi) {
        const int slot = gi%2, first = gi*group, count = std::min(group, int(active.size())-first);
        char * stage = scratch+layout.stage+slot*layout.stage_stride;
        if (gi >= 2) CUDA_CHECK(cudaStreamWaitEvent(state.copy, state.used[slot], 0));
        if (cached) {
            for (int j = 0; j < count; ++j) {
                if (cache_plan.resident[first+j]) continue;
                char * target = const_cast<char *>(cache_plan.experts[first+j]);
                const int expert = active[first+j];
                CUDA_CHECK(cudaMemcpyAsync(target, (const char *) g->data+expert*layout.gbytes,
                    layout.gbytes, cudaMemcpyHostToDevice, state.copy));
                CUDA_CHECK(cudaMemcpyAsync(target+cache_plan.up_offset, (const char *) u->data+expert*layout.ubytes,
                    layout.ubytes, cudaMemcpyHostToDevice, state.copy));
                CUDA_CHECK(cudaMemcpyAsync(target+cache_plan.down_offset, (const char *) d->data+expert*layout.dbytes,
                    layout.dbytes, cudaMemcpyHostToDevice, state.copy));
                copied += layout.gbytes+layout.ubytes+layout.dbytes;
            }
            CUDA_CHECK(cudaEventRecord(state.ready[slot], state.copy));
            return;
        }
        const ggml_tensor * tensors[] = {g, u, d};
        size_t offset = 0;
        for (const auto * tensor : tensors) {
            // Coalesce adjacent selected experts within this group's plane.
            for (int j = 0; j < count;) {
                int end = j+1;
                while (end < count && active[first+end] == active[first+end-1]+1) ++end;
                const size_t bytes = (end-j)*tensor->nb[2];
                CUDA_CHECK(cudaMemcpyAsync(stage+offset+j*tensor->nb[2],
                    (const char *) tensor->data+active[first+j]*tensor->nb[2], bytes, cudaMemcpyHostToDevice, state.copy));
                copied += bytes;
                j = end;
            }
            offset += group*tensor->nb[2];
        }
        CUDA_CHECK(cudaEventRecord(state.ready[slot], state.copy));
    };
    // Queue a complete ring before launching consumers. A slot is only reused
    // after its previous consumer has recorded used, avoiding a future-event wait.
    for (int gi = 0; gi < std::min(2, groups); ++gi) upload(gi);
    for (int gi = 0; gi < groups; ++gi) {
        const int slot = gi%2, first = gi*group, count = std::min(group, int(active.size())-first);
        const int r0 = hbounds[first], rows = hbounds[first+count]-r0;
        int max_rows = 0;
        for (int j = 0; j < count; ++j) max_rows = std::max(max_rows, counts[active[first+j]]);
        const char * stage = scratch+layout.stage+slot*layout.stage_stride;
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), state.ready[slot], 0));
        product(ctx, stage, xq, gate, identity, bounds+first, count, max_rows, nr, ne, nf, layout.gbytes, cached ? expert_ptrs+first : nullptr);
        product(ctx, stage+group*layout.gbytes, xq, up, identity, bounds+first, count, max_rows, nr, ne, nf, layout.ubytes, cached ? expert_ptrs+first : nullptr, cache_plan.up_offset);
        activate<<<(int64_t(rows)*nf+255)/256, 256, 0, ctx.stream()>>>(gate, up, int64_t(r0)*nf, int64_t(rows)*nf, limit);
        // MMQ stores [K block][sorted row], not independent row-major byte
        // strings. Scatter this group's hidden rows using the full-layer row
        // stride so downstream MMQ can use the global expert bounds.
        quantize_scatter_mmq_q8_1_cuda(gate+int64_t(r0)*nf, identity+r0, hq, d->type,
                                     nf, nf, GGML_PAD(nf, 512), rows, nr, 1, ctx.stream());
        product(ctx, stage+group*(layout.gbytes+layout.ubytes), hq, down, identity, bounds+first,
                count, max_rows, nr, nf, ne, layout.dbytes, cached ? expert_ptrs+first : nullptr, cache_plan.down_offset);
        CUDA_CHECK(cudaEventRecord(state.used[slot], ctx.stream()));
        if (gi+2 < groups) upload(gi+2);
    }
    combine<<<(int64_t(ne)*nt+255)/256, 256, 0, ctx.stream()>>>(down, inverse, (const float *) weights->data,
        (float *) node->data, ne, nk, nt, weights->nb[1]/sizeof(float), weights->nb[2]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(state.done, ctx.stream()));
    state.pending = true;
    if (cached) ggml_dsv4f_prefill_cache_end(ctx.stream());
    if (std::getenv("DSV4F_PREFILL_PROFILE")) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        std::fprintf(stderr, "[dsv4f-prefill] layer=%s tokens=%d experts=%zu groups=%d h2d_mib=%.2f wall_ms=%.3f cache=%d hits=%d borrowed=%d\n",
                     g->name, nt, active.size(), groups, copied/1048576., ms, cached, cache_plan.hits, cache_plan.borrowed);
    }
}
