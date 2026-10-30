#include "common.cuh"
#include "dsv4f-memory.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "../ggml-backend-moe-cache.h"
#include "ggml-backend-impl.h"
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "dsv4f-service.cuh"
#include "../ggml-dsv4f-routing.h"

namespace {
__global__ void activate(float * g, const float * u, int n, float limit) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { float v = fminf(g[i], limit); g[i] = v/(1.0f + expf(-v))*fmaxf(-limit, fminf(u[i], limit)); }
}
__global__ void reduce(const float * rows, const float * weights, float * out, int ne, int nh) {
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r < ne) { float v = 0; for (int k = 0; k < nh; ++k) v += rows[k*ne + r]*weights[k]; out[r] = v; }
}
__global__ void join(float * cpu, const float * gpu, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) cpu[i] += gpu[i];
}
struct key {
    const void * base; int expert;
    bool operator==(const key & b) const { return base == b.base && expert == b.expert; }
};
struct hash { size_t operator()(const key & k) const { return (size_t(k.base)>>6) ^ (size_t(k.expert)*0x9e3779b9); } };
struct slot { key id{}; bool occupied=false, ready=false, pinned=false; uint64_t age=0; void * routing=nullptr; };
struct job { int si; const char * g; const char * u; const char * d; };
class cache {
    std::mutex mu;
    std::condition_variable cv;
    std::thread loader;
    std::deque<job> queue;
    std::vector<slot> slots;
    std::unordered_map<key,int,hash> lookup;
    std::unordered_map<key,uint64_t,hash> frequency;
    const ggml_tensor * cg=nullptr, *cu=nullptr, *cd=nullptr;
    std::vector<const void *> sources;
    ggml_type gt{},ut{},dt{};
    size_t gs=0,us=0,ds=0,stride=0,uo=0,doff=0;
    int ne=0,nf=0,nh=0,cursor=0;
    uint64_t epoch=0,hits=0,routes=0;
    char * slab=nullptr;
    float * act=nullptr,*gate=nullptr,*up=nullptr,*down=nullptr,*result=nullptr,*dw=nullptr;
    void * aq=nullptr,*hq=nullptr;
    int32_t * di=nullptr;
    float * host=nullptr;
    cudaStream_t stream=nullptr,copy=nullptr;
    cudaEvent_t compute_done=nullptr,consumer_done=nullptr;
    const void * offered_host=nullptr;
    void * offered_device=nullptr;
    ggml_backend_t offered_consumer=nullptr;
#ifdef USE_CUDA_GRAPH
    std::unordered_map<uint64_t,cudaGraphExec_t> expert_graphs;
#endif
    int32_t hi[64]{};
    float hw[64]{};
    std::vector<int> pinned;
    std::vector<int> missed;
    bool stop=false,initialized=false,failed=false;
    void load() {
        ggml_cuda_set_device(0);
        std::unique_lock<std::mutex> lock(mu);
        for (;;) {
            cv.wait(lock,[&]{return stop || !queue.empty();});
            if (stop) return;
            const job j=queue.front(); queue.pop_front();
            lock.unlock();
            // finish can release a hit slot before its GPU branch completes.
            // Weight replacement must wait for all previously dispatched reads.
            CUDA_CHECK(cudaStreamWaitEvent(copy,compute_done,0));
            ggml_dsv4f_host_lease g(j.g,gs),u(j.u,us),d(j.d,ds);
            CUDA_CHECK(cudaMemcpyAsync(slab + j.si*stride, g.data, gs, cudaMemcpyHostToDevice,copy));
            CUDA_CHECK(cudaMemcpyAsync(slab + j.si*stride + uo, u.data, us, cudaMemcpyHostToDevice,copy));
            CUDA_CHECK(cudaMemcpyAsync(slab + j.si*stride + doff, d.data, ds, cudaMemcpyHostToDevice,copy));
            CUDA_CHECK(cudaStreamSynchronize(copy));
            lock.lock(); slots[j.si].ready=true;
        }
    }
    bool init(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d) {
        if (failed || !ggml_is_quantized(g->type) || !ggml_is_quantized(u->type) || !ggml_is_quantized(d->type)) return false;
        if (initialized) return g->type==gt && u->type==ut && d->type==dt && g->nb[2]==gs && u->nb[2]==us && d->nb[2]==ds && g->ne[0]==ne && g->ne[1]==nf;
        ne=g->ne[0]; nf=g->ne[1]; gt=g->type; ut=u->type; dt=d->type;
        gs=g->nb[2]; us=u->nb[2]; ds=d->nb[2];
        // One complete expert per slot. Align every projection's stride to its block byte size.
        const size_t align=std::lcm(std::lcm(ggml_type_size(gt),ggml_type_size(ut)),ggml_type_size(dt));
        uo=((gs+ggml_type_size(ut)-1)/ggml_type_size(ut))*ggml_type_size(ut);
        doff=((uo+us+ggml_type_size(dt)-1)/ggml_type_size(dt))*ggml_type_size(dt);
        stride=((doff+ds+align-1)/align)*align;
        const int ns=ggml_dsv4f_allocate_cache(&slab,stride,g->ne[2]*43,0,"dsv4f-experts");
        if (!ns) { failed=true; return false; }
        slots.resize(ns);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_done,cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&consumer_done,cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(compute_done,stream));
        CUDA_CHECK(cudaEventRecord(consumer_done,stream));
        CUDA_CHECK(cudaMalloc((void **)&act,ne*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&gate,64*nf*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&up,64*nf*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&down,64*ne*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&result,ne*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&dw,64*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&di,64*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&aq,GGML_PAD(ne,MATRIX_ROW_PADDING)/QK8_1*sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&hq,64*GGML_PAD(nf,MATRIX_ROW_PADDING)/QK8_1*sizeof(block_q8_1)));
        CUDA_CHECK(cudaMallocHost((void **)&host,ne*sizeof(float)));
        initialized=true;
        loader=std::thread([this]{load();});
        fprintf(stderr,"[dsv4f-experts] complete-expert cache: %d slots, %.2f GiB, CPU/GPU overlap\n",ns,double(ns*stride)/(1ULL<<30));
        return true;
    }
public:
    ~cache() { clear(); }
    void clear() {
        { std::lock_guard<std::mutex> lock(mu); stop=true; }
        cv.notify_all(); if(loader.joinable())loader.join();
        if (!initialized) return;
        ggml_cuda_set_device(0);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventSynchronize(consumer_done));
#ifdef USE_CUDA_GRAPH
        for (const auto & entry : expert_graphs) CUDA_CHECK(cudaGraphExecDestroy(entry.second));
        expert_graphs.clear();
#endif
        (void)cudaFree(slab);(void)cudaFree(act);(void)cudaFree(gate);(void)cudaFree(up);(void)cudaFree(down);(void)cudaFree(result);
        (void)cudaFree(dw);(void)cudaFree(di);(void)cudaFree(aq);(void)cudaFree(hq);(void)cudaFreeHost(host);
        (void)cudaStreamDestroy(stream);(void)cudaStreamDestroy(copy);
        (void)cudaEventDestroy(compute_done);(void)cudaEventDestroy(consumer_done);
        slots.clear();lookup.clear();frequency.clear();sources.clear();queue.clear();pinned.clear();missed.clear();
        slab=nullptr; initialized=false;stop=false;failed=false;cg=cu=cd=nullptr;
        offered_host=nullptr;offered_device=nullptr;offered_consumer=nullptr;
    }
    uint64_t dispatch(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d,
                      const float * x,const int32_t * ids,const float * weights,int nk,float limit) {
        ggml_cuda_set_device(0);
        nh=0;
        if (!init(g,u,d)) { cg=nullptr; return 0; }
        cg=g;cu=u;cd=d;nh=0;pinned.clear();missed.clear();uint64_t mask=0;
        {
            std::lock_guard<std::mutex> lock(mu); ++epoch;
            for (const void * p : {g->data,u->data,d->data}) {
                if (std::find(sources.begin(),sources.end(),p)==sources.end()) sources.push_back(p);
            }
            for(int i=0;i<nk;++i) {
                key k{g->data,ids[i]};++frequency[k];++routes;
                auto it=lookup.find(k);
                if(it!=lookup.end() && slots[it->second].ready) {
                    int s=it->second;slots[s].pinned=true;slots[s].age=epoch;
                    pinned.push_back(s);hi[nh]=s;hw[nh++]=weights[i];mask|=uint64_t(1)<<i;++hits;
                } else missed.push_back(ids[i]);
            }
        }
        if(!nh)return 0;
        // The previous consumer may still read our result buffer.
        CUDA_CHECK(cudaStreamWaitEvent(stream,consumer_done,0));
        CUDA_CHECK(cudaMemcpyAsync(di,hi,nh*sizeof(int32_t),cudaMemcpyHostToDevice,stream));
        CUDA_CHECK(cudaMemcpyAsync(dw,hw,nh*sizeof(float),cudaMemcpyHostToDevice,stream));
        CUDA_CHECK(cudaMemcpyAsync(act,x,ne*sizeof(float),cudaMemcpyHostToDevice,stream));
        auto launch_experts = [&] {
        quantize_row_q8_1_cuda(act,nullptr,aq,ut,ne,ne,1,1,ne,1,1,1,stream);
        if(gt==ut) {
            ggml_cuda_moe_cache_mmv(slab+uo,ut,(const char *)aq,di,gate,ne,nf,slots.size(),stride,nh,1,stream,slab,GGML_GLU_OP_SWIGLU_CLAMP,limit);
        } else {
            ggml_cuda_moe_cache_mmv(slab+uo,ut,(const char *)aq,di,up,ne,nf,slots.size(),stride,nh,1,stream,nullptr,0);
            quantize_row_q8_1_cuda(act,nullptr,aq,gt,ne,ne,1,1,ne,1,1,1,stream);
            ggml_cuda_moe_cache_mmv(slab,gt,(const char *)aq,di,gate,ne,nf,slots.size(),stride,nh,1,stream,nullptr,0);
            activate<<<(nh*nf+255)/256,256,0,stream>>>(gate,up,nh*nf,limit);
        }
        quantize_row_q8_1_cuda(gate,nullptr,hq,dt,nf,nf,nh*nf,1,nf,nh,1,1,stream);
        ggml_cuda_moe_cache_mmv(slab+doff,dt,(const char *)hq,di,down,nf,ne,slots.size(),stride,nh,nh,stream,nullptr,0);
        reduce<<<(ne+255)/256,256,0,stream>>>(down,dw,result,ne,nh);
        CUDA_CHECK(cudaGetLastError());
        };
#ifdef USE_CUDA_GRAPH
        static const bool use_graphs=[] {
            const char * e=std::getenv("DSV4F_EXPERT_GRAPHS");
            return (!e || atoi(e)!=0) && !std::getenv("GGML_CUDA_DISABLE_GRAPHS");
        }();
        if (use_graphs) {
            // IDs, routing weights, and activation are mutable device inputs.
            // Only hit count and the scalar clamp affect captured launch args.
            uint32_t limit_bits;std::memcpy(&limit_bits,&limit,sizeof(limit_bits));
            const uint64_t graph_key=(uint64_t(limit_bits)<<32)|uint32_t(nh);
            auto it=expert_graphs.find(graph_key);
            if (it==expert_graphs.end()) {
                cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));
                launch_experts();
                CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
                CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
                CUDA_CHECK(cudaGraphDestroy(graph));
                it=expert_graphs.emplace(graph_key,executable).first;
            }
            CUDA_CHECK(cudaGraphLaunch(it->second,stream));
        } else
#endif
        {
            launch_experts();
        }
        if (!offered_host) CUDA_CHECK(cudaMemcpyAsync(host,result,ne*sizeof(float),cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaEventRecord(compute_done,stream));
        return mask;
    }
    void collect(float * out,int n) {
        if (out == offered_host) return;
        CUDA_CHECK(cudaStreamSynchronize(stream));std::memcpy(out,host,n*sizeof(float));
    }
    void offer(const void * h,void * d,ggml_backend_t consumer) {
        static const bool enabled=[] {
            const char * e=std::getenv("DSV4F_GPU_JOIN");return e && atoi(e)!=0;
        }();
        if (!enabled || !ggml_backend_is_cuda(consumer)) return;
        auto * ctx=(ggml_backend_cuda_context *)consumer->context;
        if (ctx->device != 0) return;
        offered_host=h;offered_device=d;offered_consumer=consumer;
    }
    bool finalize(const void * h,ggml_backend_t consumer) {
        if (!offered_host || h!=offered_host || consumer!=offered_consumer) return false;
        const bool populated=initialized && nh>0;
        if (populated) {
            auto * ctx=(ggml_backend_cuda_context *)consumer->context;
            ggml_cuda_set_device(ctx->device);
            // Snapshot CPU output in pinned memory; never expose allocator-owned
            // host memory to an outstanding DMA after this callback returns.
#if defined(GGML_USE_HIP)
            const cudaError_t state=hipEventQuery(consumer_done);
            const cudaError_t not_ready=hipErrorNotReady;
#else
            const cudaError_t state=cudaEventQuery(consumer_done);
            const cudaError_t not_ready=cudaErrorNotReady;
#endif
            if (state==not_ready) CUDA_CHECK(cudaEventSynchronize(consumer_done));
            else CUDA_CHECK(state);
            std::memcpy(host,h,ne*sizeof(float));
            CUDA_CHECK(cudaStreamWaitEvent(ctx->stream(),compute_done,0));
            CUDA_CHECK(cudaMemcpyAsync(offered_device,host,ne*sizeof(float),cudaMemcpyHostToDevice,ctx->stream()));
            join<<<(ne+255)/256,256,0,ctx->stream()>>>((float *)offered_device,result,ne);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(consumer_done,ctx->stream()));
        }
        offered_host=nullptr;offered_device=nullptr;offered_consumer=nullptr;
        return populated;
    }
    void finish() {
        if(!initialized || !cg)return;
        std::lock_guard<std::mutex> lock(mu);
        for(int si:pinned)slots[si].pinned=false;
        // Admit at most one complete expert per layer visit to cap DRAM/PCIe contention.
        for(int eid:missed) {
            if(queue.size() >= 8) break;
            key k{cg->data,eid};if(lookup.count(k))continue;
            int victim=-1;uint64_t score=UINT64_MAX,age=UINT64_MAX;
            for(int i=0;i<std::min<int>(slots.size(),128);++i) {
                int si=cursor++%slots.size();auto & s=slots[si];
                if(!s.occupied){victim=si;break;}
                if(!s.ready || s.pinned)continue;
                uint64_t f=frequency[s.id]+ggml_dsv4f_routing_bonus(ggml_dsv4f_routing_count(s.routing,s.id.expert));
                if(f<score || (f==score && s.age<age)){score=f;age=s.age;victim=si;}
            }
            void * routing=ggml_dsv4f_routing_bind(cg);
            const uint64_t candidate=frequency[k]+ggml_dsv4f_routing_bonus(ggml_dsv4f_routing_count(routing,eid));
            if(victim<0 || (slots[victim].occupied && candidate<=score))continue;
            auto & s=slots[victim];if(s.occupied)lookup.erase(s.id);
            s={k,true,false,false,epoch,routing};lookup[k]=victim;
            queue.push_back({victim,(const char *)cg->data+eid*gs,(const char *)cu->data+eid*us,(const char *)cd->data+eid*ds});
            cv.notify_one();break;
        }
        if(epoch%4300==0)fprintf(stderr,"[dsv4f-experts] routes=%llu hit=%.2f%% queued=%zu\n",(unsigned long long)routes,100.0*hits/std::max<uint64_t>(1,routes),queue.size());
    }
    void invalidate(const void * base,size_t bytes) {
        const uintptr_t lo=(uintptr_t)base,hi=lo+bytes;
        bool found=false;
        {std::lock_guard<std::mutex> lock(mu);for(const void * p:sources)if((uintptr_t)p>=lo && (uintptr_t)p<hi){found=true;break;}}
        if(found)clear();
    }
};
cache & instance(){static cache c;return c;}
uint64_t dispatch(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d,const float * x,const int32_t * ids,const float * w,int n,float limit){return instance().dispatch(g,u,d,x,ids,w,n,limit);}
void collect(float * out,int n){instance().collect(out,n);}
void finish(){instance().finish();}
void invalidate(const void * p,size_t n){instance().invalidate(p,n);ggml_dsv4f_service_invalidate(p,n);}
void offer(const void * h,void * d,ggml_backend_t b){instance().offer(h,d,b);}
bool finalize(const void * h,ggml_backend_t b){return instance().finalize(h,b);}
}
extern "C" void ggml_dsv4f_experts_register() {
    const char * e=std::getenv("DSV4F_WHOLE_EXPERT");
    if(e && atoi(e)==0)return;
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || count<1) { (void)cudaGetLastError(); return; }
    ggml_dsv4f_experts={dispatch,collect,finish,invalidate,offer,finalize,
        ggml_dsv4f_service_supported,ggml_dsv4f_service_compute,ggml_dsv4f_service_finalize,ggml_dsv4f_service_statistics,
        ggml_dsv4f_service_resize,ggml_dsv4f_trim_workspace,ggml_dsv4f_service_drain,ggml_dsv4f_service_warm_start};
}
