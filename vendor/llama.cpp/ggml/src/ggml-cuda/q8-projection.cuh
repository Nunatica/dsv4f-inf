#pragma once
#include "vecdotq.cuh"

// Keep the RDNA3 reference's eight virtual warp accumulators and their ordered
// reduction. Fewer physical warps can execute them; independent rows share a CTA.
// Unlike changing the reduction width, this does not reassociate the dot product.
template<int WARPS, int ROWS>
static __global__ void dsv4f_q8_projection(const void * weights, const block_q8_1 * activation,
        float * output, int ni, int no, int row_stride, int wcs, int qcs, int ocs,
        int wss, int qss, int oss, uint3 channel_ratio, uint3 sample_ratio) {
    static_assert(8 % WARPS == 0 && WARPS * ROWS <= 32, "invalid projection tile");
    constexpr int VIRTUAL = 8 / WARPS;
    const int lane = threadIdx.x, part = threadIdx.y % WARPS, group = threadIdx.y / WARPS;
    const int row = blockIdx.x * ROWS + group;
    const int wc = fastdiv(blockIdx.y, channel_ratio), ws = fastdiv(blockIdx.z, sample_ratio);
    const int base = ws*wss + wc*wcs + row*row_stride;
    const auto * q = activation + blockIdx.y*qcs + blockIdx.z*qss;
    float sums[VIRTUAL] = {};
#pragma unroll
    for (int v = 0; v < VIRTUAL; ++v) {
        const int virtual_part = v*WARPS + part;
        if (row < no) for (int b = (virtual_part*32 + lane)/4; b < ni/32; b += 64)
            sums[v] += vec_dot_q8_0_q8_1(weights, q+b, base+b, 2*(lane%4));
    }
    __shared__ float partial[ROWS][7][32];
#pragma unroll
    for (int v = 0; v < VIRTUAL; ++v) {
        const int virtual_part = v*WARPS + part;
        if (virtual_part) partial[group][virtual_part-1][lane] = sums[v];
    }
    __syncthreads();
    if (part) return;
    float sum = sums[0];
#pragma unroll
    for (int p = 0; p < 7; ++p) sum += partial[group][p][lane];
    sum = warp_reduce_sum<32>(sum);
    if (!lane && row < no) output[blockIdx.z*oss + blockIdx.y*ocs + row] = sum;
}

template<int WARPS, int ROWS>
static void dsv4f_q8_projection_launch(const void * w, const void * q, float * out,
        int ni, int no, int row_stride, int channels, int samples,
        int wcs, int qcs, int ocs, int wss, int qss, int oss,
        uint3 channel_ratio, uint3 sample_ratio, cudaStream_t stream) {
    dsv4f_q8_projection<WARPS, ROWS><<<dim3((no+ROWS-1)/ROWS,channels,samples),dim3(32,WARPS*ROWS),0,stream>>>(
        w, (const block_q8_1 *)q, out, ni, no, row_stride, wcs, qcs, ocs,
        wss, qss, oss, channel_ratio, sample_ratio);
    CUDA_CHECK(cudaGetLastError());
}
