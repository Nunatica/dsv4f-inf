#pragma once
#include "../ggml-dsv4f-topk.h"
#include <cstring>
#include <cstdio>
#include <atomic>

#if defined(GGML_USE_HIP)
// Preserve the existing radix path's float-bit total order, including NaNs and
// signed zeros. Equal keys select lower IDs; output order is immaterial to CSA.
static __device__ uint32_t indexer_top512_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    return bits ^ ((uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U);
}

// One CTA per query replaces device-wide histogram launches/scratch. Each
// access is coalesced. Boundary ties use two extra ID radix digits only when
// a partial equal-key group must be selected.
static __global__ void indexer_top512_radix(const float * src, int * dst, int ncols) {
    const int tid = threadIdx.x;
    const float * row = src + (size_t) blockIdx.x * ncols;
    __shared__ int histogram[256];
    __shared__ uint32_t prefix, mask;
    __shared__ int rank;
    __shared__ uint32_t tie_prefix, tie_mask;
    __shared__ int tie_count, written;
    if (!tid) { prefix = 0; mask = 0; rank = 512; }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[tid] = 0;
        __syncthreads();
        // Aggregate consecutive identical bins (causal -inf tails in particular).
        int previous = -1, count = 0;
        for (int col = tid; col < ncols; col += 256) {
            const uint32_t key = indexer_top512_key(row[col]);
            if ((key & mask) != prefix) continue;
            const int bin = (key >> shift) & 255;
            if (bin == previous) { ++count; }
            else {
                if (count) atomicAdd(histogram + previous, count);
                previous = bin; count = 1;
            }
        }
        if (count) atomicAdd(histogram + previous, count);
        __syncthreads();
        if (!tid) {
            int bin = 255;
            while (bin > 0 && histogram[bin] < rank) rank -= histogram[bin--];
            prefix |= (uint32_t) bin << shift;
            mask |= 255U << shift;
            if (!shift) { tie_count = histogram[bin]; tie_prefix = 0; tie_mask = 0; written = 0; }
        }
        __syncthreads();
    }
    if (tie_count > rank) {
        // Width is bounded to 65536, hence the inverted 16-bit ID suffices.
        for (int shift = 8; shift >= 0; shift -= 8) {
            histogram[tid] = 0;
            __syncthreads();
            for (int col = tid; col < ncols; col += 256) {
                const uint32_t id_key = 65535U - (uint32_t) col;
                if (indexer_top512_key(row[col]) == prefix && (id_key & tie_mask) == tie_prefix)
                    atomicAdd(histogram + ((id_key >> shift) & 255), 1);
            }
            __syncthreads();
            if (!tid) {
                int bin = 255;
                while (bin > 0 && histogram[bin] < rank) rank -= histogram[bin--];
                tie_prefix |= (uint32_t) bin << shift;
                tie_mask |= 255U << shift;
            }
            __syncthreads();
        }
    }
    const int tie_last_id = 65535U - tie_prefix;
    int * out = dst + (size_t) blockIdx.x * 512;
    const int lane = tid % warpSize;
    for (int base = 0; base < ncols; base += 256) {
        const int col = base + tid;
        const uint32_t key = col < ncols ? indexer_top512_key(row[col]) : 0;
        const bool selected = col < ncols && (key > prefix || (key == prefix && col <= tie_last_id));
        const uint64_t ballot = __ballot(selected);
        int offset = 0;
        if (!lane) offset = atomicAdd(&written, __popcll(ballot));
        offset = __shfl(offset, 0, warpSize);
        if (selected) out[offset + __popcll(ballot & ((1ULL << lane) - 1))] = col;
    }
}

// ds4-style 4096-key chunk sort followed by a tree of top-512 merges. Packed
// keys retain the same bit order as radix, with a lower-ID tie break. Zero is
// an invalid padding key and is below every genuine key (including negative NaN).
static __device__ uint64_t indexer_top512_pack(const float * row, int id) {
    return ((uint64_t) indexer_top512_key(row[id]) << 32) | (0xffffffffU - (uint32_t) id);
}

template<int N>
static __device__ void indexer_top512_sort(uint64_t * keys) {
    for (int k = 2; k <= N; k *= 2) {
        for (int j = k/2; j; j /= 2) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int other = i ^ j;
                if (other > i) {
                    const uint64_t a = keys[i], b = keys[other];
                    if ((i & k) ? a > b : a < b) { keys[i] = b; keys[other] = a; }
                }
            }
            __syncthreads();
        }
    }
}

static __global__ void indexer_top512_chunks(const float * src, uint64_t * candidates, int * dst,
                                             int ncols, int chunks) {
    __shared__ uint64_t keys[4096];
    const float * row = src + (size_t) blockIdx.x * ncols;
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        const int id = blockIdx.y * 4096 + i;
        keys[i] = id < ncols ? indexer_top512_pack(row, id) : 0;
    }
    __syncthreads();
    indexer_top512_sort<4096>(keys);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        if (chunks == 1) dst[(size_t) blockIdx.x * 512 + i] = 0xffffffffU - (uint32_t) keys[i];
        else candidates[((size_t) blockIdx.x * chunks + blockIdx.y) * 512 + i] = keys[i];
    }
}

static __global__ void indexer_top512_merge(const uint64_t * src, uint64_t * tmp, int * dst, int sets) {
    __shared__ uint64_t keys[1024];
    const int base = 2 * blockIdx.y;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        const int set = base + i/512;
        keys[i] = set < sets ? src[((size_t) blockIdx.x * sets + set) * 512 + i%512] : 0;
    }
    __syncthreads();
    indexer_top512_sort<1024>(keys);
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        if (sets <= 2) dst[(size_t) blockIdx.x * 512+i] = 0xffffffffU - (uint32_t) keys[i];
        else tmp[((size_t) blockIdx.x * ((sets+1)/2) + blockIdx.y) * 512+i] = keys[i];
    }
}

static bool ggml_cuda_indexer_top512(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const char * mode = std::getenv("GGML_HIP_INDEXER_TOP512");
    // Use CTA radix by default; explicit 0 retains the original implementation.
    const int method = !mode || !std::strcmp(mode,"1") ? 1 : !std::strcmp(mode,"2") ? 2 : 0;
    const ggml_tensor * src = dst->src[0];
    if (!method || dst->op_params[1] != GGML_DSV4F_INDEXER_TOPK || dst->ne[0] != 512 ||
        src->ne[0] <= 1024 || src->ne[0] > 65536) return false;
    const int ncols = src->ne[0], nrows = ggml_nrows(src);
    // Single-query performance regresses at larger widths. Keep the tested
    // 1025..8192 range; large contexts retain the existing parallel radix path.
    if (method == 1 && ncols > 8192) return false;
    static std::atomic<bool> reported[3] = {};
    if (!reported[method].exchange(true)) {
        std::fprintf(stderr,"[hip-indexer-top512] method=%d width=%d rows=%d\n", method,ncols,nrows);
    }
    if (method == 1) {
        indexer_top512_radix<<<nrows,256,0,ctx.stream()>>>((const float *) src->data,(int *) dst->data,ncols);
    } else {
        int sets = (ncols+4095)/4096;
        ggml_cuda_pool_alloc<uint64_t> a(ctx.pool(),(size_t) nrows*sets*512);
        ggml_cuda_pool_alloc<uint64_t> b(ctx.pool(),(size_t) nrows*((sets+1)/2)*512);
        uint64_t * in=a.get(), *out=b.get();
        indexer_top512_chunks<<<dim3(nrows,sets),256,0,ctx.stream()>>>(
            (const float *)src->data,in,(int *)dst->data,ncols,sets);
        while (sets > 1) {
            indexer_top512_merge<<<dim3(nrows,(sets+1)/2),256,0,ctx.stream()>>>(in,out,(int *)dst->data,sets);
            sets=(sets+1)/2;std::swap(in,out);
        }
    }
    return true;
}
#endif
