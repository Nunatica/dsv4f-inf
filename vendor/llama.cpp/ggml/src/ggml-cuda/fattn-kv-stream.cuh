#pragma once
#include "dsv4f-kv.cuh"

// Selection, CLOCK resolution, miss uploads and attention all run on the same
// stream at fixed addresses. There is no CPU readback inside captured graphs.
static __global__ void kv_stream_indices(const half * mask, int64_t stride, int rows, int queries,
        int capacity, int * ids) {
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        bool visible = false;
        for (int q = 0; q < queries; ++q) visible |= isfinite(__half2float(mask[q*stride + row]));
        if (visible) {
            const int slot = atomicAdd(&count, 1);
            assert(slot < capacity);
            if (slot < capacity) ids[slot] = row;
        }
    }
    __syncthreads();
    for (int i = count + threadIdx.x; i < capacity; i += blockDim.x) ids[i] = -1;
}
static __global__ void kv_stream_resolve(dsv4f_kv_state s, const int * ids, int * misses,
        int capacity, int raw_rows, int stream) {
    if (threadIdx.x != 0) return;
    unsigned epoch = ++s.ctl[0];
    if (!epoch) { // avoid stale epoch equality after unsigned wraparound
        for (int i = 0; i < s.slots; ++i) s.stamp[i] = 0;
        epoch = ++s.ctl[0];
    }
    for (int i = 0; i < capacity; ++i) {
        misses[i] = -1;
        if (ids[i] < raw_rows) continue;
        const int row = stream*s.rows + ids[i] - raw_rows;
        assert(row >= 0 && row < s.rows*s.streams);
        const int slot = s.map[row];
        if (slot >= 0) { s.stamp[slot] = epoch; s.ref[slot] = 1; }
    }
    unsigned hand = s.ctl[1];
    for (int i = 0; i < capacity; ++i) {
        if (ids[i] < raw_rows) continue;
        const int row = stream*s.rows + ids[i] - raw_rows;
        if (s.map[row] >= 0) continue;
        int slot = -1;
        // Every selected hit was pinned above; only unused slots may be evicted.
        // A bounded sweep also makes a capacity bug fail instead of hanging.
        for (int step = 0; step < 2*s.slots + 1; ++step) {
            const int candidate = hand;
            hand = (hand + 1) % s.slots;
            if (s.stamp[candidate] == epoch) continue;
            if (s.ref[candidate]) { s.ref[candidate] = 0; continue; }
            slot = candidate; break;
        }
        assert(slot >= 0);
        if (slot < 0) return;
        if (s.owner[slot] >= 0) s.map[s.owner[slot]] = -1;
        s.owner[slot] = row; s.map[row] = slot;
        s.stamp[slot] = epoch; s.ref[slot] = 1;
        misses[i] = slot;
    }
    s.ctl[1] = hand;
}
static __global__ void kv_stream_load(dsv4f_kv_state s, const int * ids, const int * misses,
        int raw_rows, int stream) {
    const int slot = misses[blockIdx.x];
    if (slot < 0) return;
    const int row = stream*s.rows + ids[blockIdx.x] - raw_rows;
    for (int i = threadIdx.x; i < s.row_bytes; i += blockDim.x) {
        s.cache[int64_t(slot)*s.row_bytes + i] = s.host[int64_t(row)*s.row_bytes + i];
    }
}
static __global__ void kv_stream_gather(dsv4f_kv_state s, const char * raw, int64_t raw_stride,
        int raw_type, int raw_rows, int stream, const int * ids, int capacity,
        const half * mask, int64_t mask_stride, int queries, half * keys, half * masks) {
    const int i = blockIdx.x;
    const int row = ids[i];
    const char * source = nullptr;
    int type = raw_type;
    if (row >= 0 && row < raw_rows) source = raw + int64_t(row)*raw_stride;
    if (row >= raw_rows) {
        source = s.cache + int64_t(s.map[stream*s.rows + row - raw_rows])*s.row_bytes;
        type = s.type;
    }
    for (int col = threadIdx.x; col < 512; col += blockDim.x) {
        half value = half(0.0f);
        if (source) {
            if (type == GGML_TYPE_Q8_0) {
                const auto & b = ((const block_q8_0 *) source)[col/32];
                value = half(__half2float(b.d)*float(b.qs[col%32]));
            } else value = ((const half *) source)[col];
        }
        keys[int64_t(i)*512 + col] = value;
    }
    for (int q = threadIdx.x; q < queries; q += blockDim.x) {
        masks[int64_t(q)*capacity + i] = row >= 0 ? mask[int64_t(q)*mask_stride + row] : half(-INFINITY);
    }
}
static __global__ void kv_stream_prefill_keys(const char * raw, int64_t raw_stride, int raw_type,
        const char * compressed, int row_bytes, int comp_type, int raw_rows, half * keys) {
    const int row = blockIdx.x;
    const char * source = row < raw_rows ? raw + int64_t(row)*raw_stride : compressed + int64_t(row - raw_rows)*row_bytes;
    const int type = row < raw_rows ? raw_type : comp_type;
    for (int col = threadIdx.x; col < 512; col += blockDim.x) {
        if (type == GGML_TYPE_Q8_0) {
            const auto & b = ((const block_q8_0 *) source)[col/32];
            keys[int64_t(row)*512 + col] = half(__half2float(b.d)*float(b.qs[col%32]));
        } else keys[int64_t(row)*512 + col] = ((const half *) source)[col];
    }
}
void ggml_cuda_dsv4f_kv_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto & s = *(dsv4f_kv_state *) ggml_dsv4f_kv_state(dst);
    GGML_ASSERT(ctx.device == s.device);
    const auto * q = dst->src[0];
    const auto * raw = dst->src[1];
    const auto * mask = dst->src[2];
    const int bound = dst->op_params[11];
    GGML_ASSERT(q->type == GGML_TYPE_F32 && q->ne[0] == 512 && raw->ne[2] == 1);
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && bound > raw->ne[1]);
    GGML_ASSERT(raw->type == GGML_TYPE_F16 || raw->type == GGML_TYPE_Q8_0);
    // The compressed union of a group fits the resident pool even when every
    // query chooses disjoint rows. Raw SWA rows never consume resident slots.
    const dsv4f_kv_scratch layout(dst);
    const int group = layout.group, capacity = layout.capacity;
    auto * base = (char *) dst->data;
    auto * ids = (int *) (base + layout.ids);
    auto * misses = (int *) (base + layout.misses);
    auto * keys = (half *) (base + layout.keys);
    auto * masks = (half *) (base + layout.masks);
    float scale;
    std::memcpy(&scale, dst->op_params + 14, sizeof(scale));
    for (int stream = 0; stream < q->ne[3]; ++stream) {
        const int backing_stream = dst->op_params[10] + stream;
        GGML_ASSERT(backing_stream < s.streams);
        for (int first = 0; first < q->ne[1]; first += group) {
            const int count = std::min<int64_t>(group, q->ne[1] - first);
            const auto * m = (const half *) ((const char *) mask->data + stream*mask->nb[3] + first*mask->nb[1]);
            if (layout.prefill) {
                const int comp_rows = capacity - raw->ne[1];
                GGML_ASSERT(comp_rows <= s.rows && capacity % FATTN_KQ_STRIDE == 0);
                auto * staged = base + layout.compressed;
                CUDA_CHECK(cudaMemcpyAsync(staged, s.host_cpu + int64_t(backing_stream)*s.rows*s.row_bytes,
                    comp_rows*s.row_bytes, cudaMemcpyHostToDevice, ctx.stream()));
                kv_stream_prefill_keys<<<capacity, 256, 0, ctx.stream()>>>(
                    (const char *) raw->data + stream*raw->nb[3], raw->nb[1], raw->type,
                    staged, s.row_bytes, s.type, raw->ne[1], keys);
            } else {
                kv_stream_indices<<<1, 256, 0, ctx.stream()>>>(m, mask->nb[1]/sizeof(half), mask->ne[0], count, capacity, ids);
                kv_stream_resolve<<<1, 1, 0, ctx.stream()>>>(s, ids, misses, capacity, raw->ne[1], backing_stream);
                kv_stream_load<<<capacity, 256, 0, ctx.stream()>>>(s, ids, misses, raw->ne[1], backing_stream);
                kv_stream_gather<<<capacity, 256, 0, ctx.stream()>>>(s,
                    (const char *) raw->data + stream*raw->nb[3], raw->nb[1], raw->type,
                    raw->ne[1], backing_stream, ids, capacity, m, mask->nb[1]/sizeof(half), count, keys, masks);
            }
            CUDA_CHECK(cudaGetLastError());
            ggml_tensor qg = *q;
            qg.data = (char *) q->data + stream*q->nb[3] + first*q->nb[1];
            qg.ne[1] = count; qg.ne[3] = 1;
            ggml_tensor kg = *raw;
            kg.type = GGML_TYPE_F16; kg.data = keys;
            kg.ne[1] = capacity; kg.ne[2] = kg.ne[3] = 1;
            kg.nb[0] = sizeof(half); kg.nb[1] = 512*sizeof(half);
            kg.nb[2] = kg.nb[3] = capacity*kg.nb[1];
            ggml_tensor mg = *mask;
            mg.data = layout.prefill ? (void *) m : (void *) masks;
            mg.ne[0] = capacity; mg.ne[1] = count;
            mg.ne[2] = mg.ne[3] = 1;
            mg.nb[0] = sizeof(half); mg.nb[1] = capacity*sizeof(half);
            mg.nb[2] = mg.nb[3] = mg.ne[1]*mg.nb[1];
            ggml_tensor out = *dst;
            out.op = GGML_OP_FLASH_ATTN_EXT;
            std::memset(out.op_params, 0, sizeof(out.op_params));
            std::memcpy(out.op_params, &scale, sizeof(scale));
            out.data = (char *) dst->data + stream*dst->nb[3] + first*dst->nb[2];
            out.ne[2] = count; out.ne[3] = 1;
            out.src[0] = &qg; out.src[1] = &kg; out.src[2] = &kg;
            out.src[3] = &mg; out.src[4] = dst->src[3];
            ggml_prec_set_acc(&out, GGML_PREC_F32);
            // Preserve the normal optimized prefill implementation and its
            // attention sinks; decode uses the compact FP16 tile kernel.
            if (layout.prefill) ggml_cuda_flash_attn_ext(ctx, &out);
            else ggml_cuda_flash_attn_ext_tile(ctx, &out);
        }
    }
}
