#include "dsv4f-service.cuh"
#include "dsv4f-prefill.cuh"
#include "dsv4f-memory.cuh"
#include "ggml-backend-moe-cache.h"
#include "../ggml-dsv4f-cache-policy.h"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
constexpr int RING=32, MAX_LAYERS=256, MAX_EXPERTS=256, MAX_USED=64, MAX_EMBD=4096;
struct alignas(64) request {
    volatile unsigned long long seq,done,consumed;
    int layer,nk;
    uint64_t hits;
    float limit;
    int32_t ids[MAX_USED];
    float weights[MAX_USED],x[MAX_EMBD],answer[MAX_EMBD];
};
// A slot's state is its reader count, or -1 while a loader owns it. The loader
// must acquire it before invalidating/replacing a weight. No global GPU drain.
__device__ bool acquire(int * state,int slot) {
    int old=atomicAdd(state+slot,0);
    while(old>=0) { const int got=atomicCAS(state+slot,old,old+1);if(got==old)return true;old=got; }
    return false;
}
__global__ void route(request * ring,unsigned long long * sequence,int * table,int * readers,
                      const float * x,const int32_t * ids,const float * weights,int weight_stride,
                      int layer,int ne,int nk,float limit,int32_t * plan,float * plan_w,unsigned long long * issued) {
    __shared__ unsigned long long seq;
    __shared__ uint64_t mask;
    if(threadIdx.x==0) {
        seq=atomicAdd(sequence,1ull)+1ull; *issued=seq; mask=0;
        request * r=ring+(seq%RING);
        const unsigned long long previous=seq>RING?seq-RING:0;
        while(r->consumed!=previous) { /* GPU queue cannot overwrite an unconsumed request. */ }
        r->layer=layer;r->nk=nk;r->limit=limit;
        for(int i=0;i<nk;++i) {
            const int eid=ids[i];
            const int key=layer*MAX_EXPERTS+eid;
            int slot=(eid>=0 && eid<MAX_EXPERTS)?atomicAdd(table+key,0):-1;
            if(slot>=0) {
                if(!acquire(readers,slot))slot=-1;
                else if(atomicAdd(table+key,0)!=slot){atomicSub(readers+slot,1);slot=-1;}
            }
            plan[i]=slot;plan_w[i]=weights[i*weight_stride];
            r->ids[i]=eid;r->weights[i]=plan_w[i];
            if(slot>=0)mask|=uint64_t(1)<<i;
        }
        r->hits=mask;
    }
    __syncthreads();
    request * r=ring+(seq%RING);
    if(mask!=((uint64_t(1)<<nk)-1))for(int i=threadIdx.x;i<ne;i+=blockDim.x)r->x[i]=x[i];
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0){r->seq=seq;__threadfence_system();}
}
__global__ void release_reduce(const float * down,const float * weights,const int32_t * plan,
                               int * readers,int ne,int nk,float * output) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<ne){float v=0;for(int i=0;i<nk;++i)if(plan[i]>=0)v+=weights[i]*down[i*ne+r];output[r]=v;}
    // Reader protection must outlive ALL down reads: release in a subsequent kernel.
}
__global__ void release_readers(int * readers,const int32_t * plan,int nk) {
    const int i=threadIdx.x;if(i<nk && plan[i]>=0)atomicSub(readers+plan[i],1);
}
__global__ void cpu_join(request * ring,const unsigned long long * sequence,float * out,int ne,bool poll) {
    const unsigned long long seq=*sequence;const request * r=ring+(seq%RING);
    __shared__ int timeout;
    if(threadIdx.x==0){timeout=0;if(poll){const long long start=clock64();while(r->done!=seq)if(clock64()-start>(1ll<<36)){timeout=1;break;}}}
    __syncthreads();__threadfence_system();
    for(int i=threadIdx.x;i<ne;i+=blockDim.x)out[i]=timeout?NAN:out[i]+((volatile const float*)r->answer)[i];
}
__global__ void reserve_slot(int * readers,int * table,int slot,int old_key,int * claimed) {
    if(threadIdx.x)return;
    *claimed=(old_key<0 || atomicCAS(readers+slot,0,-1)==0)?1:0;
    if(*claimed && old_key>=0)atomicCAS(table+old_key,slot,-1);
    __threadfence_system();
}
__global__ void publish_slot(int * readers,int * table,int slot,int key) {
    if(threadIdx.x)return;__threadfence();atomicExch(readers+slot,0);atomicExch(table+key,slot);
}
// MXFP4 tile epilogues: no F32 hidden spill, no per-expert down output spill.
__global__ void mxfp4_gu(const char * const * cache,size_t up_offset,const int32_t * plan,
                        const block_q8_1 * xq,block_q8_1 * hq,int ne,int nf,float limit) {
    const int expert=blockIdx.y,slot=plan[expert];if(slot<0)return;
    const int lane=threadIdx.x&31,warp=threadIdx.x/32,row0=blockIdx.x*32+warp*4;
    const char * g=cache[slot];
    const char * u=g+up_offset;const int nb=ne/QK_MXFP4;
    __shared__ float hidden[32];
    float a[4]={},b[4]={};
    for(int k=lane/(QI_MXFP4/VDR_MXFP4_Q8_1_MMVQ);k<nb;k+=VDR_MXFP4_Q8_1_MMVQ*32/QI_MXFP4){
        const int q=VDR_MXFP4_Q8_1_MMVQ*(lane%(QI_MXFP4/VDR_MXFP4_Q8_1_MMVQ));
#pragma unroll
        for(int j=0;j<4;++j){const int index=(row0+j)*nb+k;a[j]+=vec_dot_mxfp4_q8_1(g,xq+k,index,q);b[j]+=vec_dot_mxfp4_q8_1(u,xq+k,index,q);}
    }
#pragma unroll
    for(int j=0;j<4;++j){a[j]=warp_reduce_sum<32>(a[j]);b[j]=warp_reduce_sum<32>(b[j]);if(!lane){const float gv=fminf(a[j],limit),uv=fminf(fmaxf(b[j],-limit),limit);hidden[warp*4+j]=(gv/(1.f+expf(-gv)))*uv;}}
    __syncthreads();
    if(!warp){const float v=hidden[lane],maximum=warp_reduce_max<32>(fabsf(v)),sum=warp_reduce_sum<32>(v);const float scale=maximum/127.f;
        auto * dest=hq+expert*(nf/32)+blockIdx.x;dest->qs[lane]=maximum==0?0:roundf(v/scale);if(!lane)dest->ds=make_half2(scale,sum);}
}
__global__ void mxfp4_down_sum(const char * const * cache,size_t down_offset,const int32_t * plan,
                              const float * weights,const block_q8_1 * hq,int nf,int ne,int nk,float * out) {
    const int lane=threadIdx.x&31,expert=threadIdx.x/32,row0=blockIdx.x*4,nb=nf/QK_MXFP4;
    __shared__ float parts[8][4];float a[4]={};
    if(expert<nk && plan[expert]>=0){const int slot=plan[expert];
        const char * d=cache[slot]+down_offset;const auto * x=hq+expert*nb;
        for(int k=lane/(QI_MXFP4/VDR_MXFP4_Q8_1_MMVQ);k<nb;k+=VDR_MXFP4_Q8_1_MMVQ*32/QI_MXFP4){const int q=VDR_MXFP4_Q8_1_MMVQ*(lane%(QI_MXFP4/VDR_MXFP4_Q8_1_MMVQ));
#pragma unroll
            for(int j=0;j<4;++j)a[j]+=vec_dot_mxfp4_q8_1(d,x+k,(row0+j)*nb+k,q);}}
#pragma unroll
    for(int j=0;j<4;++j){const float v=warp_reduce_sum<32>(a[j]);if(!lane)parts[expert][j]=v;}
    __syncthreads();if(threadIdx.x<4){float v=0;for(int e=0;e<nk;++e)if(plan[e]>=0)v+=weights[e]*parts[e][threadIdx.x];out[row0+threadIdx.x]=v;}
}

struct descriptor {ggml_tensor g,u,d;};
struct job {int slot,key,old_key;};
class service {
    std::mutex mutex,answer_mutex;
    std::condition_variable copies,answered;
    std::thread worker,loader;
    std::atomic<bool> stop{false};
    std::deque<descriptor> layers;
    std::unordered_map<const void*,int> layer_ids;
    ggml_dsv4f_cache_policy policy;
    std::unordered_set<int> pending;
    std::deque<job> queue;
    std::vector<ggml_dsv4f_cache_policy::slot> & slots=policy.slots;
    FILE * trace=nullptr;
    request * host=nullptr,*device=nullptr;
    unsigned long long * seq=nullptr,*issued=nullptr;
    int * table=nullptr,*readers=nullptr,*claim_h=nullptr,*claim_d=nullptr;
    char * slab=nullptr;
    char ** extra=nullptr;
    struct cache_block { char * data; size_t count; };
    std::vector<cache_block> cache_blocks;
    // Fused kernels always read through this stable GPU pointer table, even
    // for the initial cache. Graph replay survives shrinking and regrowing it.
    std::vector<char *> cache_ptrs;
    int base_slots=0,max_slots=0;
    size_t reclaim_free=0,reclaim_hip_free=0;
    float * gate=nullptr,*down=nullptr,*weights=nullptr;
    int32_t * plan=nullptr;
    void * aq=nullptr,*hq=nullptr;
    cudaStream_t copy=nullptr;
    int ne=0,nf=0;
    size_t gs=0,us=0,ds=0,uo=0,doff=0,stride=0;
    ggml_type gt{},ut{},dt{};
    const void * selected_source=nullptr;
    bool configured=false;
    bool poll=true;
    bool fused_mxfp4=false;
    uint64_t routes=0,hits=0,admissions=0,admit_bytes=0,requests=0;
    double cpu_ms=0;
    void admit(const int * misses,int n) {
        if(queue.size()>=8)return;
        const auto chosen=policy.select(misses,n,pending);if(chosen.index<0)return;
        const int key=chosen.key,victim=chosen.index;
        pending.insert(key);slots[victim].ready=false;queue.push_back({victim,key,slots[victim].key});
        if(trace)std::fprintf(trace,"Q %d %d %d\n",key,slots[victim].key,victim);
        copies.notify_one();
    }
    void serve() {
        unsigned long long next=1;int idle=0;
        while(!stop.load(std::memory_order_acquire)) {
            request * r=host+(next%RING);
            if(r->seq!=next){
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
                if(++idle>200000){std::this_thread::sleep_for(std::chrono::microseconds(100));idle=200000;}continue;
            }
            idle=0;std::atomic_thread_fence(std::memory_order_acquire);
            descriptor layer;
            {std::lock_guard<std::mutex> lock(mutex);layer=layers[r->layer];}
            const auto start=std::chrono::steady_clock::now();
            const uint64_t all=(uint64_t(1)<<r->nk)-1;
            if(r->hits!=all)ggml_dsv4f_cpu_experts(&layer.g,&layer.u,&layer.d,r->x,r->ids,r->weights,r->nk,r->hits,r->limit,r->answer);
            else std::fill_n(r->answer,ne,0.f);
            cpu_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            {std::lock_guard<std::mutex> lock(answer_mutex);std::atomic_thread_fence(std::memory_order_release);r->done=next;}answered.notify_all();
            {std::lock_guard<std::mutex> lock(mutex);
                ++requests;routes+=r->nk;
                int keys[MAX_USED],misses[MAX_USED],nmiss=0;
                for(int i=0;i<r->nk;++i){keys[i]=r->layer*MAX_EXPERTS+r->ids[i];hits+=(r->hits>>i)&1;if(!((r->hits>>i)&1))misses[nmiss++]=keys[i];}
                policy.observe(keys,r->nk);
                if(trace){std::fprintf(trace,"R %llu",(unsigned long long)r->hits);for(int i=0;i<r->nk;++i)std::fprintf(trace," %d",keys[i]);std::fprintf(trace,"\n");}
                admit(misses,nmiss);
                if(requests%4300==0)std::fprintf(stderr,"[dsv4f-service] routes=%llu hit=%.2f%% cpu_ms/token=%.3f admissions=%llu copy_mib=%.2f queued=%zu\n",(unsigned long long)routes,100.*hits/std::max<uint64_t>(routes,1),43.*cpu_ms/requests,(unsigned long long)admissions,double(admit_bytes)/(1<<20),queue.size());
            }
            std::atomic_thread_fence(std::memory_order_release);r->consumed=next;++next;
        }
    }
    void load() {
        ggml_cuda_set_device(0);
        std::unique_lock<std::mutex> lock(mutex);
        while(true) {
            copies.wait(lock,[&]{return stop.load()||!queue.empty();});if(stop.load())return;
            job j=queue.front();queue.pop_front();descriptor l=layers[j.key/MAX_EXPERTS];lock.unlock();
            reserve_slot<<<1,32,0,copy>>>(readers,table,j.slot,j.old_key,claim_d);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(copy));
            const bool success=*claim_h!=0;
            if(success){const int eid=j.key%MAX_EXPERTS;
                char * dest=fused_mxfp4?cache_ptrs[j.slot]:slab+j.slot*stride;
                CUDA_CHECK(cudaMemcpyAsync(dest,(const char*)l.g.data+eid*gs,gs,cudaMemcpyHostToDevice,copy));
                CUDA_CHECK(cudaMemcpyAsync(dest+uo,(const char*)l.u.data+eid*us,us,cudaMemcpyHostToDevice,copy));
                CUDA_CHECK(cudaMemcpyAsync(dest+doff,(const char*)l.d.data+eid*ds,ds,cudaMemcpyHostToDevice,copy));
                publish_slot<<<1,32,0,copy>>>(readers,table,j.slot,j.key);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(copy));
            }
            lock.lock();if(success){policy.publish(j.slot,j.key);++admissions;admit_bytes+=gs+us+ds;}slots[j.slot].ready=true;pending.erase(j.key);copies.notify_all();
            if(trace)std::fprintf(trace,"A %d %d\n",j.key,int(success));
        }
    }
    static void callback(void * arg) {
        auto & self=*static_cast<service*>(arg);
        // All preceding device work has completed, so the latest request is
        // visible. No CUDA/HIP API may be called from this runtime callback.
        unsigned long long latest=0;
        for(int i=0;i<RING;++i)latest=std::max(latest,(unsigned long long)(self.host[i].seq));
        std::unique_lock<std::mutex> lock(self.answer_mutex);
        self.answered.wait(lock,[&]{return self.host[latest%RING].done==latest || self.stop.load();});
    }
    void init(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d,int nk) {
        if(host){GGML_ASSERT(g->type==gt && u->type==ut && d->type==dt && g->ne[0]==ne && g->ne[1]==nf);return;}
        const char * cache_mode=std::getenv("DSV4F_CACHE_POLICY");if(!cache_mode)cache_mode="legacy";
        ggml_dsv4f_cache_policy::mode selected;
        if(!std::strcmp(cache_mode,"legacy"))selected=ggml_dsv4f_cache_policy::mode::legacy;
        else if(!std::strcmp(cache_mode,"scored"))selected=ggml_dsv4f_cache_policy::mode::scored;
        else if(!std::strcmp(cache_mode,"aged"))selected=ggml_dsv4f_cache_policy::mode::aged;
        else throw std::runtime_error("DSV4F_CACHE_POLICY must be legacy, scored or aged");
        auto setting=[](const char * name,unsigned fallback,unsigned maximum) {
            const char * s=std::getenv(name);if(!s)return fallback;
            char * end=nullptr;const unsigned long v=std::strtoul(s,&end,10);
            if(end==s || *end || *s=='-' || v<1 || v>maximum)throw std::runtime_error(std::string("invalid cache setting: ")+name);
            return unsigned(v);
        };
        const unsigned decay=setting("DSV4F_CACHE_DECAY_TOKENS",256,65536);
        double ratio=1.25;
        if(const char * value=std::getenv("DSV4F_CACHE_REPLACE_RATIO")) {
            char * end=nullptr;ratio=std::strtod(value,&end);
            if(end==value || *end || !std::isfinite(ratio) || ratio<1 || ratio>16)throw std::runtime_error("invalid DSV4F_CACHE_REPLACE_RATIO");
        }
        policy.configure(selected,43ULL*decay,ratio);
        std::fprintf(stderr,"[dsv4f-service] cache_policy=%s decay_tokens=%u replacement_ratio=%.3f\n",cache_mode,decay,ratio);
        if(const char * path=std::getenv("DSV4F_CACHE_TRACE")) {
            trace=std::fopen(path,"w");if(!trace)throw std::runtime_error("cannot open DSV4F_CACHE_TRACE");
            std::setvbuf(trace,nullptr,_IOFBF,1<<20);
        }
        ne=g->ne[0];nf=g->ne[1];gt=g->type;ut=u->type;dt=d->type;gs=g->nb[2];us=u->nb[2];ds=d->nb[2];
        const size_t align=std::lcm(std::lcm(ggml_type_size(gt),ggml_type_size(ut)),ggml_type_size(dt));
        uo=(gs+ggml_type_size(ut)-1)/ggml_type_size(ut)*ggml_type_size(ut);
        doff=(uo+us+ggml_type_size(dt)-1)/ggml_type_size(dt)*ggml_type_size(dt);stride=(doff+ds+align-1)/align*align;
        const char * mode=std::getenv("DSV4F_GPU_WAIT");poll=!mode || std::strcmp(mode,"callback")!=0;
        const char * fusion=std::getenv("DSV4F_MXFP4_FUSED");fused_mxfp4=nk<=8 && gt==GGML_TYPE_MXFP4 && dt==GGML_TYPE_MXFP4 && ne%128==0 && nf%32==0 && (!fusion || atoi(fusion)!=0);
        const size_t scratch=5*sizeof(unsigned long long)+MAX_LAYERS*MAX_EXPERTS*sizeof(int)+
            MAX_USED*(size_t(nf+ne+1)*sizeof(float)+sizeof(int32_t))+
            (GGML_PAD(ne,MATRIX_ROW_PADDING)+MAX_USED*GGML_PAD(nf,MATRIX_ROW_PADDING))/QK8_1*sizeof(block_q8_1)+
            43*g->ne[2]*(sizeof(char *)+sizeof(int));
        if(!ggml_dsv4f_reserve_vram(0,scratch))throw std::runtime_error("DSV4F VRAM headroom cannot be preserved for service scratch");
        CUDA_CHECK(cudaHostAlloc((void**)&host,RING*sizeof(request),cudaHostAllocMapped));std::memset(host,0,RING*sizeof(request));CUDA_CHECK(cudaHostGetDevicePointer((void**)&device,host,0));
        CUDA_CHECK(cudaMalloc((void**)&issued,4*sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&seq,sizeof(unsigned long long)));CUDA_CHECK(cudaMemset(seq,0,sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&table,MAX_LAYERS*MAX_EXPERTS*sizeof(int)));CUDA_CHECK(cudaMemset(table,0xff,MAX_LAYERS*MAX_EXPERTS*sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&gate,MAX_USED*nf*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&down,MAX_USED*ne*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&weights,MAX_USED*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&plan,MAX_USED*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&aq,GGML_PAD(ne,MATRIX_ROW_PADDING)/QK8_1*sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&hq,MAX_USED*GGML_PAD(nf,MATRIX_ROW_PADDING)/QK8_1*sizeof(block_q8_1)));
        CUDA_CHECK(cudaHostAlloc((void**)&claim_h,sizeof(int),cudaHostAllocMapped));CUDA_CHECK(cudaHostGetDevicePointer((void**)&claim_d,claim_h,0));
        max_slots=43*g->ne[2];
        CUDA_CHECK(cudaMalloc((void**)&extra,max_slots*sizeof(char *)));
        CUDA_CHECK(cudaMalloc((void**)&readers,max_slots*sizeof(int)));CUDA_CHECK(cudaMemset(readers,0xff,max_slots*sizeof(int)));
        int count;
        if(fused_mxfp4) { grow_cache();count=cache_ptrs.size(); }
        else count=ggml_dsv4f_allocate_cache(&slab,stride,max_slots,0,"dsv4f-service");
        base_slots=count;
        slots.resize(count);
        if(trace)std::fprintf(trace,"S %d %d\n",base_slots,count);
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
        stop=false;worker=std::thread([this]{serve();});loader=std::thread([this]{load();});
        std::fprintf(stderr,"[dsv4f-service] GPU routing + CPU service: %d slots %.2f GiB, wait=%s\n",count,double(count*stride)/(1ULL<<30),poll?"poll":"callback");
    }
public:
    ~service(){clear();}
    bool compatible(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d,int nk) {
        std::lock_guard<std::mutex> lock(mutex);
        const char * phase=std::getenv("DSV4F_PHASE_WORKSPACE");
        // The pointer-table epilogue has eight expert warps. Unsupported routing
        // geometries retain the CPU/legacy path before any extension is used.
        if(nk>8 && (!cache_blocks.empty() || (phase && std::atoi(phase)!=0)))return false;
        if(configured)return gt==g->type && ut==u->type && dt==d->type && ne==g->ne[0] && nf==g->ne[1] && gs==g->nb[2] && us==u->nb[2] && ds==d->nb[2];
        selected_source=g->data;configured=true;gt=g->type;ut=u->type;dt=d->type;ne=g->ne[0];nf=g->ne[1];gs=g->nb[2];us=u->nb[2];ds=d->nb[2];return true;
    }
    void statistics(uint64_t * r,uint64_t * h,uint64_t * a){std::lock_guard<std::mutex> lock(mutex);*r=routes;*h=hits;*a=admissions;}
    void grow_cache() {
        // Cap each block at 64 MiB (or one complete expert) so small temporary
        // buffers do not force the entire initial cache to be surrendered.
        const size_t block_slots=std::min<size_t>(64,std::max<size_t>(1,(64ULL<<20)/stride));
        while(cache_ptrs.size()<size_t(max_slots)) {
            char * block=nullptr;
            const size_t existing=cache_ptrs.size();
            const size_t count=ggml_dsv4f_allocate_cache(&block,stride,
                std::min(block_slots,size_t(max_slots)-existing),0,"dsv4f-service",existing*stride);
            if(!count)break;
            cache_blocks.push_back({block,count});
            for(size_t i=0;i<count;++i)cache_ptrs.push_back(block+i*stride);
        }
        if(!cache_ptrs.empty())CUDA_CHECK(cudaMemcpy(extra,cache_ptrs.data(),cache_ptrs.size()*sizeof(char *),cudaMemcpyHostToDevice));
        slots.resize(cache_ptrs.size());
    }
    void drain() {
        CUDA_CHECK(cudaDeviceSynchronize());
        for(int i=0;i<RING;++i)while(host[i].consumed!=host[i].seq)std::this_thread::yield();
    }
    // mutex held, service/loader drained; keep is a block boundary.
    void shrink_cache(size_t keep) {
        if(keep>=slots.size())return;
        if(fused_mxfp4 && keep) {
            std::vector<int> base(keep),tail(slots.size()-keep);
            std::iota(base.begin(),base.end(),0);std::iota(tail.begin(),tail.end(),keep);
            auto score=[&](int i){return policy.score(slots[i].key);};
            std::sort(base.begin(),base.end(),[&](int a,int b){return score(a)<score(b);});
            std::sort(tail.begin(),tail.end(),[&](int a,int b){return score(a)>score(b);});
            for(size_t i=0;i<std::min(base.size(),tail.size());++i) {
                const int dst=base[i],src=tail[i];
                if(slots[src].key<0 || (slots[dst].key>=0 && score(src)<=score(dst)))break;
                CUDA_CHECK(cudaMemcpyAsync(cache_ptrs[dst],cache_ptrs[src],stride,cudaMemcpyDeviceToDevice,copy));
                slots[dst].key=slots[src].key;
            }
            CUDA_CHECK(cudaStreamSynchronize(copy));
        }
        // Invalidate every removed lookup before returning its storage. Kernels
        // captured with the old capacity may still run, but cannot hit it.
        std::vector<int> lookup(MAX_LAYERS*MAX_EXPERTS,-1),states(slots.size(),-1);
        for(size_t i=0;i<keep;++i)if(slots[i].key>=0){lookup[slots[i].key]=i;states[i]=0;}
        CUDA_CHECK(cudaMemcpy(table,lookup.data(),lookup.size()*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(readers,states.data(),states.size()*sizeof(int),cudaMemcpyHostToDevice));
        if(fused_mxfp4) {
            size_t count=cache_ptrs.size();
            while(count>keep) {
                const auto block=cache_blocks.back();
                GGML_ASSERT(count-block.count>=keep);
                CUDA_CHECK(cudaFree(block.data));count-=block.count;cache_blocks.pop_back();
            }
            cache_ptrs.resize(keep);
        } else {
            GGML_ASSERT(keep==0);CUDA_CHECK(cudaFree(slab));slab=nullptr;
        }
        base_slots=std::min<size_t>(base_slots,keep);
        slots.resize(keep);policy.rebuild();
        if(trace){std::fprintf(trace,"S %d %zu\n",base_slots,slots.size());std::fflush(trace);}
    }
    size_t reclaim(size_t bytes) {
        if(!host || slots.empty() || !bytes)return 0;
        ggml_cuda_set_device(0);drain();
        std::unique_lock<std::mutex> lock(mutex);
        copies.wait(lock,[&]{return pending.empty();});
        CUDA_CHECK(cudaStreamSynchronize(copy));
        size_t keep=slots.size(),released=0;
        if(fused_mxfp4) {
            for(auto it=cache_blocks.rbegin();it!=cache_blocks.rend() && released<bytes;++it) {
                keep-=it->count;released+=it->count*stride;
            }
        } else { released=keep*stride;keep=0; }
        shrink_cache(keep);
        std::fprintf(stderr,"[dsv4f-service] reclaimed=%.2f MiB remaining_slots=%zu initial_slots=%d\n",released/1048576.,keep,base_slots);
        return released;
    }
    bool resize(bool decode) {
        if(!host)return false;
        ggml_cuda_set_device(0);drain();
        {
            std::unique_lock<std::mutex> lock(mutex);
            copies.wait(lock,[&]{return pending.empty();});
            CUDA_CHECK(cudaStreamSynchronize(copy));
            if(!decode) {
                // Capture both counters BEFORE freeing cache blocks. HIP sees
                // cudaFree immediately, while AMDGPU's physical counter lags.
                // Sampling after shrink would omit those bytes from the next
                // settle target and stop decode cache growth several GiB early,
                // especially when target and MTP workspaces switch in sequence.
                reclaim_free=ggml_dsv4f_free_vram();
                size_t total;CUDA_CHECK(cudaMemGetInfo(&reclaim_hip_free,&total));
                shrink_cache(base_slots);
            } else if(fused_mxfp4) {
#if defined(GGML_USE_HIP) && defined(__linux__)
                size_t hip_free,total;CUDA_CHECK(cudaMemGetInfo(&hip_free,&total));
                constexpr size_t tolerance=64ULL<<20;
                if(reclaim_hip_free && hip_free>reclaim_hip_free+tolerance) {
                    const size_t expected=reclaim_free+(hip_free-reclaim_hip_free)-tolerance;
                    const auto start=std::chrono::steady_clock::now();
                    const auto deadline=start+std::chrono::milliseconds(200);
                    while(ggml_dsv4f_free_vram()<expected && std::chrono::steady_clock::now()<deadline)
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    std::fprintf(stderr,"[dsv4f-service] VRAM accounting: released=%.0f MiB expected_free=%.0f MiB physical_free=%.0f MiB wait_ms=%.2f\n",
                        (hip_free-reclaim_hip_free)/1048576.,expected/1048576.,ggml_dsv4f_free_vram()/1048576.,
                        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
                }
#endif
                grow_cache();
            }
        }
        // Include other contexts' workspaces and driver allocations. A cache
        // smaller than its original prefill size is valid, including zero.
        if(!ggml_dsv4f_reserve_vram(0,0))throw std::runtime_error("DSV4F VRAM headroom cannot be preserved after phase switch");
        std::fprintf(stderr,"[dsv4f-service] phase=%s base_slots=%d extra_slots=%zu total_slots=%zu cache=%.2f MiB free=%.0f MiB\n",
            decode?"decode":"prefill",base_slots,slots.size()-base_slots,slots.size(),slots.size()*stride/1048576.,ggml_dsv4f_free_vram()/1048576.);
        if(trace){std::fprintf(trace,"S %d %zu\n",base_slots,slots.size());std::fflush(trace);}
        return true;
    }
    void clear() {
        if(!host){configured=false;selected_source=nullptr;return;}ggml_cuda_set_device(0);CUDA_CHECK(cudaDeviceSynchronize());stop.store(true,std::memory_order_release);copies.notify_all();answered.notify_all();worker.join();loader.join();
        for(auto block:cache_blocks)CUDA_CHECK(cudaFree(block.data));cache_blocks.clear();cache_ptrs.clear();
        cudaFree(extra);cudaFree(slab);cudaFree(seq);cudaFree(issued);cudaFree(table);cudaFree(readers);cudaFree(gate);cudaFree(down);cudaFree(weights);cudaFree(plan);cudaFree(aq);cudaFree(hq);cudaFreeHost(host);cudaFreeHost(claim_h);cudaStreamDestroy(copy);
        if(trace){std::fclose(trace);trace=nullptr;}
        configured=false;selected_source=nullptr;host=device=nullptr;slab=nullptr;extra=nullptr;base_slots=max_slots=0;layers.clear();layer_ids.clear();policy.clear();pending.clear();queue.clear();routes=hits=admissions=admit_bytes=requests=0;cpu_ms=0;
    }
    void invalidate(const void * base,size_t bytes) {
        const uintptr_t lo=(uintptr_t)base,hi=lo+bytes;bool found=selected_source && (uintptr_t)selected_source>=lo && (uintptr_t)selected_source<hi;
        {std::lock_guard<std::mutex> lock(mutex);for(const auto & l:layers)for(const void * p:{l.g.data,l.u.data,l.d.data})if((uintptr_t)p>=lo && (uintptr_t)p<hi)found=true;}
        if(found)clear();
    }
    void enqueue_one(ggml_backend_cuda_context & ctx,ggml_tensor * dst,int token_index) {
        const auto * g=dst->src[3],*u=dst->src[4],*d=dst->src[5];init(g,u,d,dst->src[1]->ne[0]);
        int layer;
        {std::lock_guard<std::mutex> lock(mutex);auto it=layer_ids.find(g->data);if(it==layer_ids.end()){layer=layers.size();GGML_ASSERT(layer<MAX_LAYERS);layers.push_back({*g,*u,*d});layer_ids[g->data]=layer;}else layer=it->second;}
        float limit;std::memcpy(&limit,&dst->op_params[8],sizeof(limit));const int nk=dst->src[1]->ne[0];auto stream=ctx.stream();
        route<<<1,256,0,stream>>>(device,seq,table,readers,(const float*)dst->src[0]->data,(const int32_t*)dst->src[1]->data,(const float*)dst->src[2]->data,dst->src[2]->nb[1]/sizeof(float),layer,ne,nk,limit,plan,weights,issued+token_index);
        if(slots.empty() && !fused_mxfp4){
            // No space for a complete expert: keep the GPU service/routing and
            // compute all routed experts on the CPU, without a fatal assertion.
            CUDA_CHECK(cudaMemsetAsync(dst->data,0,ne*sizeof(float),stream));
            return;
        }
        quantize_row_q8_1_cuda((const float*)dst->src[0]->data,nullptr,aq,ut,ne,ne,1,1,ne,1,1,1,stream);
        GGML_ASSERT(gt==ut);
        if(fused_mxfp4 && nk<=8){
            mxfp4_gu<<<dim3(nf/32,nk),256,0,stream>>>(extra,uo,plan,(const block_q8_1*)aq,(block_q8_1*)hq,ne,nf,limit);
            mxfp4_down_sum<<<ne/4,256,0,stream>>>(extra,doff,plan,weights,(const block_q8_1*)hq,nf,ne,nk,(float*)dst->data);
            release_readers<<<1,64,0,stream>>>(readers,plan,nk);CUDA_CHECK(cudaGetLastError());
            return;
        }
        ggml_cuda_moe_cache_mmv(slab+uo,ut,(const char*)aq,plan,gate,ne,nf,slots.size(),stride,nk,1,stream,slab,GGML_GLU_OP_SWIGLU_CLAMP,limit);
        quantize_row_q8_1_cuda(gate,nullptr,hq,dt,nf,nf,nk*nf,1,nf,nk,1,1,stream);
        ggml_cuda_moe_cache_mmv(slab+doff,dt,(const char*)hq,plan,down,nf,ne,slots.size(),stride,nk,nk,stream,nullptr,0);
        release_reduce<<<(ne+255)/256,256,0,stream>>>(down,weights,plan,readers,ne,nk,(float*)dst->data);
        release_readers<<<1,64,0,stream>>>(readers,plan,nk);CUDA_CHECK(cudaGetLastError());
    }
    void enqueue(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
        // GPU expert scratch is reused in stream order, while each token has a
        // separate ring request/answer. Submit all CPU work before the join so
        // both verification tokens can overlap the GPU and shared expert work.
        for (int64_t t=0;t<dst->ne[1];++t) {
            ggml_tensor token=*dst, x=*dst->src[0], ids=*dst->src[1], w=*dst->src[2];
            x.data=(char*)x.data+t*x.nb[1]; x.ne[1]=1;
            ids.data=(char*)ids.data+t*ids.nb[1]; ids.ne[1]=1;
            w.data=(char*)w.data+t*w.nb[2]; w.ne[2]=1;
            token.data=(char*)dst->data+t*dst->nb[1]; token.ne[1]=1;
            token.src[0]=&x; token.src[1]=&ids; token.src[2]=&w;
            enqueue_one(ctx,&token,t);
        }
        if(!dst->op_params[10])finish(ctx,dst);
    }
    void finish(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
        if(!poll)CUDA_CHECK(cudaLaunchHostFunc(ctx.stream(),callback,this));
        for(int64_t t=0;t<dst->ne[1];++t)
            cpu_join<<<1,256,0,ctx.stream()>>>(device,issued+t,(float*)((char*)dst->data+t*dst->nb[1]),ne,poll);
        CUDA_CHECK(cudaGetLastError());
    }
};
service & instance(){static service value;return value;}
}
bool ggml_dsv4f_service_supported(const ggml_tensor * n) {
    if (n->op == GGML_OP_CUSTOM && n->op_params[11] == 1) return ggml_cuda_dsv4f_prefill_supported(n);
    const char * enabled=std::getenv("DSV4F_GPU_SERVICE");
    if(!enabled||atoi(enabled)==0||!ggml_dsv4f_cpu_experts||n->op!=GGML_OP_CUSTOM)return false;
    if(n->op_params[9]==GGML_DSV4F_JOIN_TAG)return true;
    if(n->op_params[9]!=GGML_DSV4F_MOE_TAG)return false;
    const auto * g=n->src[3],*u=n->src[4],*d=n->src[5];
    return g&&u&&d&&g->type==u->type&&(g->type==GGML_TYPE_MXFP4 || g->type==GGML_TYPE_Q4_K || g->type==GGML_TYPE_Q6_K)&&(d->type==GGML_TYPE_MXFP4 || d->type==GGML_TYPE_Q4_K || d->type==GGML_TYPE_Q6_K)&&!g->extra&&!u->extra&&!d->extra&&
        g->ne[0]<=MAX_EMBD && n->src[0]->ne[1]>=1 && n->src[0]->ne[1]<=4 && n->src[1]->ne[0]<MAX_USED && g->ne[2]<=MAX_EXPERTS &&
        g->buffer&&u->buffer&&d->buffer&&ggml_backend_buffer_is_host(g->buffer)&&ggml_backend_buffer_is_host(u->buffer)&&ggml_backend_buffer_is_host(d->buffer) && instance().compatible(g,u,d,n->src[1]->ne[0]);
}
void ggml_dsv4f_service_compute(void * ctx,ggml_tensor * node){
    auto & cuda_ctx = *static_cast<ggml_backend_cuda_context*>(ctx);
    if (node->op_params[11] == 1) ggml_cuda_dsv4f_prefill(cuda_ctx, node);
    else instance().enqueue(cuda_ctx,node);
}
void ggml_dsv4f_service_finalize(void * ctx,ggml_tensor * node){instance().finish(*static_cast<ggml_backend_cuda_context*>(ctx),node);}
void ggml_dsv4f_service_invalidate(const void * base,size_t bytes){instance().invalidate(base,bytes);}

void ggml_dsv4f_service_statistics(uint64_t * r,uint64_t * h,uint64_t * a){instance().statistics(r,h,a);}
bool ggml_dsv4f_service_resize(bool decode){return instance().resize(decode);}

bool ggml_dsv4f_reserve_vram(int device,size_t bytes,bool allow_reclaim) {
    const char * enabled=std::getenv("DSV4F_GPU_SERVICE");
    if(!enabled || !std::atoi(enabled))return true;
    ggml_cuda_set_device(device);
    const size_t configured=ggml_dsv4f_headroom();
    const size_t library_guard=ggml_dsv4f_guard();
    // Temporary allocations must not consume the allowance for lazy library
    // code/workspaces. Replenish it at every allocation and graph boundary,
    // including when target and MTP contexts alternate their workspaces.
    const size_t headroom=configured>SIZE_MAX-library_guard?SIZE_MAX:configured+library_guard;
    size_t free=ggml_dsv4f_free_vram();
    size_t deficit=ggml_dsv4f_vram_deficit(free,headroom,bytes);
    if(!deficit)return true;
    // Service cache lives on logical device zero. Never reclaim from another
    // device or synchronize/free while a stream is being captured.
    if(device==0 && allow_reclaim) {
#if defined(GGML_USE_HIP) && defined(__linux__)
        // Cache/workspace frees are visible to HIP before AMDGPU sysfs. Give
        // the physical counter time to settle before evicting useful experts.
        // A real desktop allocation remains counted after the bounded wait.
        size_t hip_free,total;CUDA_CHECK(cudaMemGetInfo(&hip_free,&total));
        if(!ggml_dsv4f_vram_deficit(hip_free,headroom,bytes)) {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);
            do {
                free=ggml_dsv4f_free_vram();deficit=ggml_dsv4f_vram_deficit(free,headroom,bytes);
                if(!deficit)return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } while(std::chrono::steady_clock::now()<deadline);
        }
#endif
        // Avoid repeated tiny trims as driver accounting settles. The larger
        // library allowance is already included in headroom above.
        const size_t guard=64ULL<<20;
        const size_t request=deficit>SIZE_MAX-guard?SIZE_MAX:deficit+guard;
        if(instance().reclaim(request)) {
            // AMDGPU sysfs may lag frees. Remeasure physical VRAM rather than
            // assuming all released bytes have already become available.
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);
            do {
                free=ggml_dsv4f_free_vram();deficit=ggml_dsv4f_vram_deficit(free,headroom,bytes);
                if(!deficit)return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } while(std::chrono::steady_clock::now()<deadline);
        }
    }
    std::fprintf(stderr,"[dsv4f-service] insufficient VRAM: allocation=%.2f MiB free=%.2f MiB headroom=%.2f MiB guard=%.2f MiB reclaim=%s\n",
        bytes/1048576.,free/1048576.,configured/1048576.,library_guard/1048576.,allow_reclaim?"allowed":"capture");
    return false;
}
