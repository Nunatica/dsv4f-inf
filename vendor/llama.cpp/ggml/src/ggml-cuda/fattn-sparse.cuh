#pragma once

#include <cstring>
#include <atomic>
#include <cstdio>

// HIP sparse decode: gather a query group's visible KV rows, dequantizing only
// those rows, then reuse the ordinary FP16 tile kernel. No host synchronization
// or data-dependent allocation is needed, including during graph capture.
#if defined(GGML_USE_HIP)

static bool ggml_hip_fattn_sparse_applicable(const ggml_tensor * dst, int64_t & capacity) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HIP_SPARSE_FA");
        return !value || std::strcmp(value, "0") != 0;
    }();
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    const int32_t bound = ggml_get_op_params_i32(dst, 4);
    if (!enabled || bound <= 0 || !m || q->ne[1] > 8 || q->ne[3] != 1 ||
        k->ne[0] != 512 || v->ne[0] != 512 || k->ne[2] != v->ne[2] ||
        k->ne[3] != 1 || v->ne[3] != 1 || v->ne[1] != k->ne[1] ||
        m->type != GGML_TYPE_F16 || m->ne[0] != k->ne[1] || m->ne[1] < q->ne[1] ||
        m->ne[2] != 1 || m->ne[3] != 1 || m->nb[0] != sizeof(half)) {
        return false;
    }
    for (const ggml_tensor * t : { k, v }) {
        if ((t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_Q8_0) ||
            t->nb[0] != ggml_type_size(t->type)) {
            return false;
        }
    }
    float max_bias, softcap;
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    // A union can contain at most queries * bound entries. Keep dense attention
    // for prefill and whenever compaction cannot remove at least half the rows.
    capacity = GGML_PAD(int64_t(bound)*q->ne[1], int64_t(FATTN_KQ_STRIDE));
    return k->ne[1] >= 2048 && 2*capacity <= k->ne[1];
}

__launch_bounds__(256)
static __global__ void hip_fattn_sparse_indices(
        const half * mask, int32_t * indices, const int n_kv, const int n_queries,
        const int64_t mask_stride, const int capacity) {
    constexpr int values_per_lane = 8;
    const int wave_size = warpSize;
    const int wave = threadIdx.x / wave_size;
    const int lane = threadIdx.x % wave_size;
    __shared__ int offsets[8];
    __shared__ int count;
    __shared__ int chunk_count;
    if (threadIdx.x == 0) {
        count = 0;
    }
    __syncthreads();
    for (int base = 0; base < n_kv; base += 256*values_per_lane) {
        uint64_t selected[values_per_lane];
        int wave_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int row = base + (wave*values_per_lane + item)*wave_size + lane;
            bool visible = false;
            if (row < n_kv) {
                for (int q = 0; q < n_queries; ++q) {
                    visible |= isfinite(__half2float(mask[q*mask_stride + row]));
                }
            }
            selected[item] = __ballot(visible);
            wave_count += __popcll(selected[item]);
        }
        if (lane == 0) {
            offsets[wave] = wave_count;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int prefix = 0;
            for (int w = 0; w < 256/wave_size; ++w) {
                const int n = offsets[w];
                offsets[w] = prefix;
                prefix += n;
            }
            chunk_count = prefix;
        }
        __syncthreads();
        const uint64_t before_lane = (uint64_t(1) << lane) - 1;
        int item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int slot = count + offsets[wave] + item_offset + __popcll(selected[item] & before_lane);
            if ((selected[item] & (uint64_t(1) << lane)) && slot < capacity) {
                indices[slot] = base + (wave*values_per_lane + item)*wave_size + lane;
            }
            item_offset += __popcll(selected[item]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            count += chunk_count;
        }
        __syncthreads();
    }
    // This is the public n_kv_max contract; never silently truncate a bad hint.
    if (count > capacity) {
        __builtin_trap();
    }
    for (int slot = count + threadIdx.x; slot < capacity; slot += blockDim.x) {
        indices[slot] = -1;
    }
}

template <ggml_type type>
__launch_bounds__(256)
static __global__ void hip_fattn_sparse_gather(
        const char * src, half * dst, const int32_t * indices,
        const int64_t stride_row, const int64_t stride_head, const int capacity,
        const half * mask, half * gathered_mask, const int64_t mask_stride, const int n_queries) {
    const int slot = blockIdx.x;
    const int head = blockIdx.y;
    const int row = indices[slot];
    for (int col = threadIdx.x; col < 512; col += blockDim.x) {
        half value = half(0.0f);
        if (row >= 0) {
            const char * src_row = src + int64_t(row)*stride_row + int64_t(head)*stride_head;
            if constexpr (type == GGML_TYPE_Q8_0) {
                const block_q8_0 & block = ((const block_q8_0 *) src_row)[col/QK8_0];
                value = __float2half(__half2float(block.d)*float(block.qs[col%QK8_0]));
            } else {
                value = ((const half *) src_row)[col];
            }
        }
        dst[(int64_t(head)*capacity + slot)*512 + col] = value;
    }
    // Fuse mask gathering into K gathering; V can share the gathered K buffer.
    if (gathered_mask && head == 0 && threadIdx.x == 0) {
        for (int q = 0; q < n_queries; ++q) {
            gathered_mask[int64_t(q)*capacity + slot] = row >= 0 ? mask[q*mask_stride + row] : half(-INFINITY);
        }
    }
}

static bool ggml_hip_flash_attn_ext_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int64_t capacity;
    if (!ggml_hip_fattn_sparse_applicable(dst, capacity)) {
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    const bool shared_kv = k->data == v->data && k->type == v->type &&
        k->nb[1] == v->nb[1] && k->nb[2] == v->nb[2];
    static std::atomic<bool> reported{false};
    const char * debug = std::getenv("GGML_HIP_SPARSE_FA_DEBUG");
    if (debug && std::strcmp(debug, "0") != 0 && !reported.exchange(true)) {
        // The server filters GGML info logs at its default verbosity. An
        // explicitly requested diagnostic should remain visible there too.
        std::fprintf(stderr, "[hip-sparse-fa] queries=%lld kv=%lld gathered=%lld types=%s/%s shared_kv=%d\n",
            (long long) q->ne[1], (long long) k->ne[1], (long long) capacity,
            ggml_type_name(k->type), ggml_type_name(v->type), int(shared_kv));
    }
    const size_t elements = size_t(capacity)*512*k->ne[2];
    ggml_cuda_pool_alloc<int32_t> indices(ctx.pool(), capacity);
    ggml_cuda_pool_alloc<half> gathered_k(ctx.pool(), elements);
    ggml_cuda_pool_alloc<half> gathered_v(ctx.pool());
    if (!shared_kv) {
        gathered_v.alloc(elements);
    }
    ggml_cuda_pool_alloc<half> gathered_mask(ctx.pool(), capacity*q->ne[1]);
    const auto stream = ctx.stream();
    hipLaunchKernelGGL(hip_fattn_sparse_indices, dim3(1), dim3(256), 0, stream,
        (const half *) m->data, indices.ptr, int(k->ne[1]), int(q->ne[1]),
        int64_t(m->nb[1]/sizeof(half)), int(capacity));
    auto gather = [&](const ggml_tensor * src, half * out, half * mask_out) {
        const auto kernel = src->type == GGML_TYPE_Q8_0 ?
            hip_fattn_sparse_gather<GGML_TYPE_Q8_0> : hip_fattn_sparse_gather<GGML_TYPE_F16>;
        const ggml_cuda_kernel_launch_params params(dim3(capacity, src->ne[2]), dim3(256), 0, stream);
        ggml_cuda_kernel_launch(kernel, params, (const char *) src->data, out, indices.ptr,
            int64_t(src->nb[1]), int64_t(src->nb[2]), int(capacity),
            (const half *) m->data, mask_out, int64_t(m->nb[1]/sizeof(half)), int(q->ne[1]));
    };
    gather(k, gathered_k.ptr, gathered_mask.ptr);
    if (!shared_kv) {
        gather(v, gathered_v.ptr, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());

    auto kv_tensor = [&](const ggml_tensor * src, half * data) {
        ggml_tensor result = *src;
        result.type = GGML_TYPE_F16;
        result.data = data;
        result.ne[1] = capacity;
        result.nb[0] = sizeof(half);
        result.nb[1] = 512*sizeof(half);
        result.nb[2] = capacity*result.nb[1];
        result.nb[3] = result.ne[2]*result.nb[2];
        result.view_src = nullptr;
        result.view_offs = 0;
        return result;
    };
    ggml_tensor kg = kv_tensor(k, gathered_k.ptr);
    ggml_tensor vg = kv_tensor(v, shared_kv ? gathered_k.ptr : gathered_v.ptr);
    ggml_tensor mg = *m;
    mg.data = gathered_mask.ptr;
    mg.ne[0] = capacity;
    mg.ne[1] = q->ne[1];
    mg.nb[1] = capacity*sizeof(half);
    mg.nb[2] = mg.ne[1]*mg.nb[1];
    mg.nb[3] = mg.nb[2];
    mg.view_src = nullptr;
    mg.view_offs = 0;
    ggml_tensor out = *dst;
    out.src[1] = &kg;
    out.src[2] = &vg;
    out.src[3] = &mg;
    ggml_flash_attn_ext_set_n_kv_max(&out, 0);
    ggml_cuda_flash_attn_ext_tile(ctx, &out);
    return true;
}
#endif // GGML_USE_HIP
