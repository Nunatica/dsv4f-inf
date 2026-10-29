#pragma once

#if defined(GGML_USE_HIP)
#include "mma.cuh"

// Adapted from ds4's 16-query / 128-key indexer tile. Keep ggml strides,
// quantized keys and additive masks; only the QK products use FP16 inputs.
template<ggml_type TYPE_K>
static __global__ void lightning_indexer_hip_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int n_batch, int n_kv, size_t nb1, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3, size_t nbm1, size_t nbm3, int nem3) {
#if defined(RDNA3) && defined(AMD_WMMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    using input_tile = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using output_tile = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int first_key = blockIdx.x*128, first_query = blockIdx.y*16;
    const int stream = blockIdx.z;
    const auto * mask = (const char *) M + (stream%nem3)*nbm3;
    __shared__ int live;
    if (tid == 0) live = 0;
    __syncthreads();
    bool lane_live = false;
    for (int i = tid; i < 16*128; i += 256) {
        const int query = first_query + i/128, key = first_key + i%128;
        if (query < n_batch && key < n_kv &&
            __half2float(((const half *)(mask + query*nbm1))[key]) != -INFINITY) {
            lane_live = true;
        }
    }
    const bool warp_live = __any_sync(0xffffffff, lane_live);
    if (warp_live && threadIdx.x == 0) atomicExch(&live, 1);
    __syncthreads();
    if (!live) {
        for (int i = tid; i < 16*128; i += 256) {
            const int query = first_query + i/128, key = first_key + i%128;
            if (query < n_batch && key < n_kv) {
                ((float *)((char *)dst + stream*nb3 + query*nb1))[key] = -INFINITY;
            }
        }
        return;
    }
    // Key tile survives all 64 heads; each query tile survives eight key warps.
    // Padding avoids mapping every matrix row onto the same LDS banks.
    constexpr int stride = 136;
    __shared__ half keys[128][stride];
    __shared__ half queries[16][stride];
    constexpr auto dequantize = get_dequantize_V<TYPE_K, float, 4>();
    for (int i = tid; i < 128*32; i += 256) {
        const int key = first_key + i/32, d = 4*(i%32);
        float4 v = make_float4(0, 0, 0, 0);
        if (key < n_kv) dequantize(K + stream*nbk3 + key*nbk2, &v, d);
        *(half2 *)&keys[i/32][d] = __float22half2_rn(make_float2(v.x, v.y));
        *(half2 *)&keys[i/32][d+2] = __float22half2_rn(make_float2(v.z, v.w));
    }
    float scores[output_tile::ne] = {};
    for (int head = 0; head < 64; ++head) {
        for (int i = tid; i < 16*128; i += 256) {
            const int query = first_query + i/128;
            const auto * qr = (const float *)((const char *)Q + stream*nbq3 + query*nbq2 + head*nbq1);
            queries[i/128][i%128] = query < n_batch ? half(qr[i%128]) : half(0.f);
        }
        __syncthreads();
        output_tile dot;
#pragma unroll
        for (int d = 0; d < 128; d += 16) {
            input_tile a, b;
            load_ldmatrix(a, (const half2 *)&queries[0][d], stride/2);
            load_ldmatrix(b, (const half2 *)&keys[threadIdx.y*16][d], stride/2);
            mma(dot, a, b);
        }
#pragma unroll
        for (int l = 0; l < output_tile::ne; ++l) {
            // WMMA's accumulator tile is column-major: i=key, j=query.
            const int query = first_query + output_tile::get_j(l);
            const float weight = query < n_batch ?
                ((const float *)((const char *)W + stream*nbw3 + query*nbw1))[head] : 0.f;
            // Match the vector path's ReLU (including its NaN comparison).
            scores[l] += (dot.x[l] > 0.f ? dot.x[l] : 0.f)*weight;
        }
        __syncthreads(); // Every warp has finished reading this head's queries.
    }
#pragma unroll
    for (int l = 0; l < output_tile::ne; ++l) {
        const int query = first_query + output_tile::get_j(l);
        const int key = first_key + threadIdx.y*16 + output_tile::get_i(l);
        if (query < n_batch && key < n_kv) {
            ((float *)((char *)dst + stream*nb3 + query*nb1))[key] =
                scores[l] + __half2float(((const half *)(mask + query*nbm1))[key]);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, W, M, dst, n_batch, n_kv, nb1, nb3,
        nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
    NO_DEVICE_CODE;
#endif
}
#endif
