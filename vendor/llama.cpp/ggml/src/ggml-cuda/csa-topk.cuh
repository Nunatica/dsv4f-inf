#pragma once
#include <atomic>

void ggml_cuda_csa_topk_attention(ggml_backend_cuda_context &, ggml_tensor *, bool);

// Stable compaction preserves physical KV order, independent of top-k's score
// order. There is one list per query, so unrelated queries never inflate a union.
static __device__ void csa_append(bool visible, int row, half bias, int * ids, half * masks,
        int capacity, int * count, int * offsets) {
    const int tid=threadIdx.x, lane=tid%warpSize, warp=tid/warpSize;
    const uint64_t ballot=__ballot(visible);
    if (!lane) offsets[warp]=__popcll(ballot);
    __syncthreads();
    if (!tid) {
        int sum=*count;
        for(int w=0;w<256/warpSize;++w){const int n=offsets[w];offsets[w]=sum;sum+=n;}
        *count=sum;
        assert(sum<=capacity);
    }
    __syncthreads();
    if (visible) {
        const int i=offsets[warp]+__popcll(ballot & ((uint64_t(1)<<lane)-1));
        if(i<capacity){ids[i]=row;masks[i]=bias;}
    }
    __syncthreads();
}
static __global__ void csa_topk_indices(const half * raw_mask,size_t raw_stride,
        const half * comp_mask,size_t comp_stride,const int32_t * topk,size_t topk_stride,
        int first,int queries,int raw_rows,int comp_rows,int nk,int raw_bound,int capacity,
        int * ids,half * masks) {
    const int q=blockIdx.x,t=first+q,tid=threadIdx.x;
    if(q>=queries)return;
    const auto * rm=(const half *)((const char *)raw_mask+t*raw_stride);
    const auto * cm=(const half *)((const char *)comp_mask+t*comp_stride);
    const auto * selected=(const int32_t *)((const char *)topk+t*topk_stride);
    ids+=q*capacity;masks+=q*capacity;
    __shared__ int count,offsets[8],sorted[512];
    if(!tid)count=0;
    __syncthreads();
    for(int base=0;base<raw_rows;base+=256) {
        const int r=base+tid;const half bias=r<raw_rows?rm[r]:half(-INFINITY);
        csa_append(r<raw_rows && isfinite(__half2float(bias)),r,bias,ids,masks,capacity,&count,offsets);
    }
    assert(count<=raw_bound);
    for(int i=tid;i<512;i+=256) {
        const int r=i<nk?selected[i]:INT_MAX;
        sorted[i]=r>=0 && r<comp_rows?r:INT_MAX;
    }
    __syncthreads();
    for(int k=2;k<=512;k*=2)for(int j=k/2;j;j/=2) {
        for(int i=tid;i<512;i+=256) {
            const int other=i^j;
            if(other>i && ((i&k)?sorted[i]<sorted[other]:sorted[i]>sorted[other])) {
                const int tmp=sorted[i];sorted[i]=sorted[other];sorted[other]=tmp;
            }
        }
        __syncthreads();
    }
    for(int base=0;base<512;base+=256) {
        const int i=base+tid,r=sorted[i];
        const half bias=r<comp_rows?cm[r]:half(-INFINITY);
        const bool live=r<comp_rows && (!i || r!=sorted[i-1]) && isfinite(__half2float(bias));
        csa_append(live,r<comp_rows?raw_rows+r:-1,bias,ids,masks,capacity,&count,offsets);
    }
    for(int i=count+tid;i<capacity;i+=256){ids[i]=-1;masks[i]=half(-INFINITY);}
}
static __global__ void csa_topk_gather(dsv4f_kv_state s,const char * raw,size_t raw_stride,
        int raw_type,int raw_rows,int stream,const char * staged,bool prefill,
        const int * ids,int capacity,half * keys) {
    const int i=blockIdx.x,q=blockIdx.y,row=ids[q*capacity+i];
    const char * source=nullptr;int type=raw_type;
    if(row>=0 && row<raw_rows)source=raw+row*raw_stride;
    if(row>=raw_rows) {
        const int r=row-raw_rows;type=s.type;
        source=prefill?staged+size_t(r)*s.row_bytes:
            s.cache+size_t(s.map[stream*s.rows+r])*s.row_bytes;
    }
    for(int col=threadIdx.x;col<512;col+=blockDim.x) {
        half value=half(0.f);
        if(source) {
            if(type==GGML_TYPE_Q8_0) {
                const auto & b=((const block_q8_0 *)source)[col/32];
                value=half(__half2float(b.d)*float(b.qs[col%32]));
            } else value=((const half *)source)[col];
        }
        keys[(size_t(q)*capacity+i)*512+col]=value;
    }
}
static void ggml_cuda_csa_topk(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    const bool resident=dst->op_params[9]==GGML_DSV4F_CSA_ATTN;
    dsv4f_kv_state storage{};
    if(resident) {
        const auto * comp=dst->src[4];
        storage.device=ctx.device;storage.rows=comp->ne[1];storage.streams=comp->ne[3];
        storage.type=comp->type;storage.row_bytes=comp->nb[1];
    }
    auto & s=resident?storage:*(dsv4f_kv_state *)ggml_dsv4f_kv_state(dst);
    const auto * q=dst->src[0],*raw=dst->src[1],*rm=dst->src[2],*topk=dst->src[5],*cm=dst->src[6];
    const dsv4f_kv_scratch l(dst);
    GGML_ASSERT(topk->type==GGML_TYPE_I32 && topk->ne[0]<=512 && rm->type==GGML_TYPE_F16 && cm->type==GGML_TYPE_F16);
    GGML_ASSERT(rm->ne[0]==raw->ne[1] && cm->ne[0]<=s.rows && topk->ne[1]>=q->ne[1]);
    static std::atomic<bool> announced{false};
    if(!announced.exchange(true))std::fprintf(stderr,"[hip-csa-topk] query-local selected KV; group=%d capacity=%d prefill=%d resident=%d\n",l.group,l.capacity,int(l.prefill),int(resident));
    auto * base=(char *)dst->data;
    auto * ids=(int *)(base+l.ids);auto * misses=(int *)(base+l.misses);
    auto * keys=(half *)(base+l.keys);auto * masks=(half *)(base+l.masks);
    auto * staged=base+l.compressed;
    float scale;std::memcpy(&scale,dst->op_params+14,sizeof(scale));
    for(int stream=0;stream<q->ne[3];++stream) {
        const int backing=dst->op_params[10]+stream;
        GGML_ASSERT(backing<s.streams);
        const char * source=resident?(const char *)dst->src[4]->data+stream*dst->src[4]->nb[3]:staged;
        if(l.prefill && !resident)CUDA_CHECK(cudaMemcpyAsync(staged,s.host_cpu+size_t(backing)*s.rows*s.row_bytes,
            cm->ne[0]*s.row_bytes,cudaMemcpyHostToDevice,ctx.stream()));
        for(int first=0;first<q->ne[1];first+=l.group) {
            const int count=std::min<int64_t>(l.group,q->ne[1]-first);
            csa_topk_indices<<<count,256,0,ctx.stream()>>>(
                (const half *)((const char *)rm->data+(stream%rm->ne[3])*rm->nb[3]),rm->nb[1],
                (const half *)((const char *)cm->data+(stream%cm->ne[3])*cm->nb[3]),cm->nb[1],
                (const int32_t *)((const char *)topk->data+stream*topk->nb[3]),topk->nb[1],
                first,count,raw->ne[1],cm->ne[0],topk->ne[0],dst->op_params[15],l.capacity,ids,masks);
            if(!l.prefill && !resident) {
                kv_stream_resolve<<<1,1,0,ctx.stream()>>>(s,ids,misses,l.capacity,raw->ne[1],backing);
                kv_stream_load<<<l.capacity,256,0,ctx.stream()>>>(s,ids,misses,raw->ne[1],backing);
            }
            csa_topk_gather<<<dim3(l.capacity,count),256,0,ctx.stream()>>>(s,
                (const char *)raw->data+stream*raw->nb[3],raw->nb[1],raw->type,raw->ne[1],backing,
                source,l.prefill || resident,ids,l.capacity,keys);
            CUDA_CHECK(cudaGetLastError());
            ggml_tensor qg=*q; qg.data=(char *)q->data+stream*q->nb[3]+first*q->nb[1];
            qg.ne[1]=1;qg.ne[3]=count;qg.nb[3]=q->nb[1];
            ggml_tensor kg=*raw;kg.type=GGML_TYPE_F16;kg.data=keys;
            kg.ne[1]=l.capacity;kg.ne[2]=1;kg.ne[3]=count;
            kg.nb[0]=sizeof(half);kg.nb[1]=512*sizeof(half);kg.nb[2]=kg.nb[3]=l.capacity*kg.nb[1];
            ggml_tensor mg=*rm;mg.data=masks;
            mg.ne[0]=l.capacity;mg.ne[1]=mg.ne[2]=1;mg.ne[3]=count;
            mg.nb[0]=sizeof(half);mg.nb[1]=mg.nb[2]=mg.nb[3]=l.capacity*sizeof(half);
            ggml_tensor out=*dst;out.op=GGML_OP_FLASH_ATTN_EXT;
            std::memset(out.op_params,0,sizeof(out.op_params));std::memcpy(out.op_params,&scale,sizeof(scale));
            out.data=(char *)dst->data+stream*dst->nb[3]+first*dst->nb[2];
            out.ne[2]=1;out.ne[3]=count;out.nb[3]=dst->nb[2];
            out.src[0]=&qg;out.src[1]=out.src[2]=&kg;out.src[3]=&mg;out.src[4]=dst->src[3];
            ggml_prec_set_acc(&out,GGML_PREC_F32);
            ggml_cuda_csa_topk_attention(ctx,&out,l.prefill);
        }
    }
}
