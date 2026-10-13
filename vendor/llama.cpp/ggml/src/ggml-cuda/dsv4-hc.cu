#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter,
        bool packed) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    if (packed) {
        for (int i = 0; i < 2*DSV4_HC; ++i) {
            const float scaled = mixes[i*sm0 + it*sm1] * scale[(i/DSV4_HC)*ss0];
            const float v = 1.0f/(1.0f + expf(-(scaled + base[i*sb0])));
            dst[it*24 + i] = i < DSV4_HC ? v + eps : v*2.0f;
        }
    }

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[packed ? it*24 + 8 + idx : idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

// One warp per token. Keep the reference's summation order for each 4-value
// normalization; shuffle distributes the 16 Sinkhorn values over the lanes.
static __global__ void dsv4_hc_comb_warp(
    const float * mixes,const float * scale,const float * base,float * dst,
    int64_t nt,int64_t sm0,int64_t sm1,int64_t ss0,int64_t sb0,
    int64_t sd0,int64_t sd1,int64_t sd2,float eps,int32_t iterations,bool packed) {
    ggml_cuda_pdl_lc();
    const int lane=threadIdx.x&31;
    const int64_t token=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(token>=nt)return;
    ggml_cuda_pdl_sync();
    const int index=lane&15,d=index&3,src=index/4;
    const float v=mixes[(8+index)*sm0+token*sm1]*scale[2*ss0]+base[(8+index)*sb0];
    float maximum=-INFINITY;
    for(int j=0;j<4;++j)maximum=fmaxf(maximum,__shfl_sync(0xffffffffu,v,src*4+j,32));
    float c=expf(v-maximum),sum=0;
    for(int j=0;j<4;++j)sum+=__shfl_sync(0xffffffffu,c,src*4+j,32);
    c=c*(1.0f/sum)+eps;
    sum=eps;for(int j=0;j<4;++j)sum+=__shfl_sync(0xffffffffu,c,d+4*j,32);c*=1.0f/sum;
    for(int it=1;it<iterations;++it){
        sum=eps;for(int j=0;j<4;++j)sum+=__shfl_sync(0xffffffffu,c,src*4+j,32);c*=1.0f/sum;
        sum=eps;for(int j=0;j<4;++j)sum+=__shfl_sync(0xffffffffu,c,d+4*j,32);c*=1.0f/sum;
    }
    if(lane<16)dst[packed?token*24+8+index:d*sd0+src*sd1+token*sd2]=c;
    if(packed && lane<8){const float x=mixes[lane*sm0+token*sm1]*scale[(lane/4)*ss0];const float q=1.0f/(1.0f+expf(-(x+base[lane*sb0])));dst[token*24+lane]=lane<4?q+eps:2.0f*q;}
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

static __device__ float hc_warp_sum(float v) {
    for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffffffffffffULL,v,d,32);return __shfl_sync(0xffffffffu,v,0,32);
}
static __device__ float hc_block_sum(float v,float * workspace) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;v=hc_warp_sum(v);if(!lane)workspace[warp]=v;__syncthreads();
    v=hc_warp_sum(threadIdx.x<8?workspace[threadIdx.x]:0.f);if(!threadIdx.x)workspace[0]=v;__syncthreads();return workspace[0];
}
// Preserve the GGUF F32 projection weights. The projection is linear, so its
// common input RMS scale is applied after the partial dot reduction.
template <bool fuse_post=false, bool has_comb=false>
static __global__ void hc_full_partial(const float * r,const float * fn,int ne,float * part,
                                      const float * branch=nullptr,const float * post=nullptr,const float * comb=nullptr,
                                      float * residual_out=nullptr,int64_t sx=1,int64_t sr=1,int64_t sh=0,
                                      int64_t sp=1,int64_t sc0=1,int64_t sc1=4) {
    const int b=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x/32,nb=gridDim.x;
    __shared__ float x[512],sums[8];float ss=0;
    for(int i=threadIdx.x;i<512;i+=256){
        const int index=b*512+i;
        if constexpr(fuse_post) {
            const int e=index%ne,h=index/ne;
            float v=branch[e*sx]*post[h*sp];
            if constexpr(has_comb)for(int s=0;s<4;++s)v+=r[e*sr+s*sh]*comb[h*sc0+s*sc1];
            else v+=r[e*sr+h*sh];
            // Preserve the residual as an output: later layers still consume it.
            residual_out[index]=v;x[i]=v;
        }else x[i]=r[index];
        ss+=x[i]*x[i];
    }__syncthreads();
    for(int m=warp;m<24;m+=8){float dot=0;for(int i=lane;i<512;i+=32)dot+=x[i]*fn[m*4*ne+b*512+i];dot=hc_warp_sum(dot);if(!lane)part[m*nb+b]=dot;}
    const float total=hc_block_sum(ss,sums);if(!threadIdx.x)part[24*nb+b]=total;
}
static __global__ void hc_full_finish(const float * r,const float * part,const float * scales,
                                     const float * base,const float * norm,int ne,float norm_eps,float eps,int iterations,float * out) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32,nb=4*ne/512;
    __shared__ float mixes[25],pre[4],mixed[4096],sums[8];
    for(int m=warp;m<25;m+=8){const float v=hc_warp_sum(lane<nb?part[m*nb+lane]:0.f);if(!lane)mixes[m]=v;}__syncthreads();
    if(!warp){
        const float inv=rsqrtf(mixes[24]/float(4*ne)+norm_eps);const int index=lane&15,d=index&3,src=index/4;
        const float v=(mixes[8+index]*inv)*scales[2]+base[8+index];float maximum=-INFINITY;
        for(int j=0;j<4;++j)maximum=fmaxf(maximum,__shfl_sync(0xffffffffu,v,src*4+j,32));
        float c=expf(v-maximum),sum=0;for(int j=0;j<4;++j)sum+=__shfl_sync(0xffffffffu,c,src*4+j,32);c=c*(1.f/sum)+eps;
        sum=eps;for(int j=0;j<4;++j)sum+=__shfl_sync(0xffffffffu,c,d+4*j,32);c*=1.f/sum;
        for(int it=1;it<iterations;++it){sum=eps;for(int j=0;j<4;++j)sum+=__shfl_sync(0xffffffffu,c,src*4+j,32);c*=1.f/sum;sum=eps;for(int j=0;j<4;++j)sum+=__shfl_sync(0xffffffffu,c,d+4*j,32);c*=1.f/sum;}
        if(lane<16)out[ne+8+index]=c;
        if(lane<8){const float x=(mixes[lane]*inv)*scales[lane/4];const float v=1.f/(1.f+expf(-(x+base[lane])));out[ne+lane]=lane<4?v+eps:v*2;if(lane<4)pre[lane]=v+eps;}
    }__syncthreads();
    float ss=0;for(int e=threadIdx.x;e<ne;e+=256){float v=0;for(int h=0;h<4;++h)v+=r[h*ne+e]*pre[h];mixed[e]=v;ss+=v*v;}
    const float inv2=rsqrtf(hc_block_sum(ss,sums)/ne+norm_eps);
    for(int e=threadIdx.x;e<ne;e+=256)out[e]=mixed[e]*inv2*norm[e];
}
static void ggml_cuda_hc_full(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    const int ne=dst->src[0]->ne[0],nb=4*ne/512;
    ggml_cuda_pool_alloc<float> partial(ctx.pool(),25*nb);
    hc_full_partial<><<<nb,256,0,ctx.stream()>>>((const float*)dst->src[0]->data,(const float*)dst->src[1]->data,ne,partial.get());
    hc_full_finish<<<1,256,0,ctx.stream()>>>((const float*)dst->src[0]->data,partial.get(),(const float*)dst->src[2]->data,(const float*)dst->src[3]->data,(const float*)dst->src[4]->data,ne,ggml_get_op_params_f32(dst,0),ggml_get_op_params_f32(dst,1),ggml_get_op_params_i32(dst,3),(float*)dst->data);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_can_fuse_hc_post_full(const ggml_tensor * post,const ggml_tensor * full) {
    static const bool enabled=[] {const char * e=getenv("DSV4F_HC_POST_FULL");return !e || atoi(e)!=0;}();
    if(!enabled || post->op!=GGML_OP_DSV4_HC_POST || full->op!=GGML_OP_DSV4_HC_COMB ||
       ggml_get_op_params_i32(full,2)!=2 || full->src[0]!=post || post->type!=GGML_TYPE_F32 ||
       full->type!=GGML_TYPE_F32 || post->ne[1]!=4 || post->ne[2]!=1 || post->ne[3]!=1 ||
       !ggml_is_contiguous(post))return false;
    for(const auto * t:{post->src[0],post->src[1],post->src[2],full->src[1],full->src[2],full->src[3],full->src[4]})
        if(!t || t->type!=GGML_TYPE_F32)return false;
    return !post->src[3] || post->src[3]->type==GGML_TYPE_F32;
}

void ggml_cuda_op_hc_post_full(ggml_backend_cuda_context & ctx,ggml_tensor * post,ggml_tensor * full) {
    const auto * x=post->src[0],*r=post->src[1],*p=post->src[2],*c=post->src[3];
    const int ne=post->ne[0],nb=4*ne/512;
    ggml_cuda_pool_alloc<float> partial(ctx.pool(),25*nb);
    auto kernel=c?hc_full_partial<true,true>:hc_full_partial<true,false>;
    // The finish kernel consumes every partial and residual output. Keep normal
    // stream ordering here, including on CUDA devices that support PDL.
    kernel<<<nb,256,0,ctx.stream()>>>(
        (const float*)r->data,(const float*)full->src[1]->data,ne,partial.get(),
        (const float*)x->data,(const float*)p->data,c?(const float*)c->data:nullptr,(float*)post->data,
        x->nb[0]/sizeof(float),r->nb[0]/sizeof(float),r->nb[1]/sizeof(float),p->nb[0]/sizeof(float),
        c?c->nb[0]/sizeof(float):0,c?c->nb[1]/sizeof(float):0);
    hc_full_finish<<<1,256,0,ctx.stream()>>>((const float*)post->data,partial.get(),(const float*)full->src[2]->data,
        (const float*)full->src[3]->data,(const float*)full->src[4]->data,ne,ggml_get_op_params_f32(full,0),
        ggml_get_op_params_f32(full,1),ggml_get_op_params_i32(full,3),(float*)full->data);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if(ggml_get_op_params_i32(dst,2)==2){ggml_cuda_hc_full(ctx,dst);return;}
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;
    const bool packed = ggml_get_op_params_i32(dst, 2) != 0;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(packed ? (dst->ne[0] == hc_mix_dim && dst->ne[1] == mixes->ne[1] && ggml_is_contiguous(dst)) :
                        (dst->ne[0] == DSV4_HC && dst->ne[1] == DSV4_HC && dst->ne[2] == mixes->ne[1]));
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    static const bool warp=[] {const char * e=std::getenv("DSV4F_HC_WARP");return !e||atoi(e)!=0;}();
    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const int threads_per_token=warp?32:1;
    const dim3 grid_dims((n_tokens*threads_per_token + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    if(warp) {
        ggml_cuda_kernel_launch(dsv4_hc_comb_warp, launch_params,
            (const float*)mixes->data,(const float*)scale->data,(const float*)base->data,(float*)dst->data,
            n_tokens,nbm0/sizeof(float),nbm1/sizeof(float),nbs0/sizeof(float),nbb0/sizeof(float),
            nbd0/sizeof(float),nbd1/sizeof(float),nbd2/sizeof(float),eps,n_iter,packed);
        return;
    }
    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter, packed);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}
