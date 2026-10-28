#pragma once
#include "vecdotq.cuh"
#include "unary.cuh"

// Single-column Q8 matvec candidate. Several independent rows share one block;
// WARPS controls each row's reduction width. No weights or activations change.
template<int WARPS,int ROWS,bool FUSED=false>
static __global__ void dsv4f_q8_tile(const void * weights,const block_q8_1 * activation,float * output,
        int ni,int no,int row_stride,int weight_channel_stride,int activation_channel_stride,
        int output_channel_stride,int weight_sample_stride,int activation_sample_stride,
        int output_sample_stride,uint3 channel_ratio,uint3 sample_ratio,const void * gate,float limit) {
    const int lane=threadIdx.x,part=threadIdx.y%WARPS,row_group=threadIdx.y/WARPS;
    const int row=blockIdx.x*ROWS+row_group;
    const int wc=fastdiv(blockIdx.y,channel_ratio),ws=fastdiv(blockIdx.z,sample_ratio);
    const int base=ws*weight_sample_stride+wc*weight_channel_stride+row*row_stride;
    const auto * q=activation+blockIdx.y*activation_channel_stride+blockIdx.z*activation_sample_stride;
    float sum=0,gsum=0;
    if(row<no)for(int b=(part*32+lane)/4;b<ni/32;b+=WARPS*8) {
        sum+=vec_dot_q8_0_q8_1(weights,q+b,base+b,2*(lane%4));
        if constexpr(FUSED)gsum+=vec_dot_q8_0_q8_1(gate,q+b,base+b,2*(lane%4));
    }
    __shared__ float partial[ROWS][WARPS>1?WARPS-1:1][32];
    __shared__ float gate_partial[FUSED?ROWS:1][FUSED && WARPS>1?WARPS-1:1][32];
    if constexpr(WARPS>1) {
        if(part)partial[row_group][part-1][lane]=sum;
        if constexpr(FUSED)if(part)gate_partial[row_group][part-1][lane]=gsum;
        __syncthreads();
        if(part)return;
#pragma unroll
        for(int p=0;p<WARPS-1;++p) {
            sum+=partial[row_group][p][lane];
            if constexpr(FUSED)gsum+=gate_partial[row_group][p][lane];
        }
    }
    sum=warp_reduce_sum<32>(sum);
    if constexpr(FUSED) {
        gsum=warp_reduce_sum<32>(gsum);
        sum=ggml_cuda_op_swiglu_clamp_single(gsum,sum,limit);
    }
    if(!lane && row<no)output[blockIdx.z*output_sample_stride+blockIdx.y*output_channel_stride+row]=sum;
}

template<int WARPS,int ROWS,bool FUSED=false>
static void dsv4f_q8_tile_launch(const void * w,const void * q,float * out,int ni,int no,
        int row_stride,int channels,int samples,int wcs,int qcs,int ocs,int wss,int qss,int oss,
        uint3 channel_ratio,uint3 sample_ratio,cudaStream_t stream,const void * gate=nullptr,float limit=0) {
    dsv4f_q8_tile<WARPS,ROWS,FUSED><<<dim3((no+ROWS-1)/ROWS,channels,samples),dim3(32,WARPS*ROWS),0,stream>>>(
        w,(const block_q8_1*)q,out,ni,no,row_stride,wcs,qcs,ocs,wss,qss,oss,channel_ratio,sample_ratio,gate,limit);
    CUDA_CHECK(cudaGetLastError());
}
