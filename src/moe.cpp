#include "moe.h"
#include "ggml-dsv4f-prefill.h"
#include "mxfp4.h"
#include "ggml-cpu.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-dsv4f-routing.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {
struct cpu_options {
    int chunk=32,rows=1;
    bool contiguous=false,physical=false;
    cpu_options() {
        auto number=[](const char * name,int fallback,int maximum) {
            const char * v=std::getenv(name);if(!v)return fallback;char * end=nullptr;const auto n=std::strtol(v,&end,10);
            if(end==v || *end || n<1 || n>maximum)throw std::runtime_error(std::string("invalid CPU setting: ")+name);
            return int(n);
        };
        chunk=number("DSV4F_CPU_ROW_CHUNK",32,4096);rows=number("DSV4F_CPU_MXFP4_ROWS",1,4);
        if(rows!=1 && rows!=2 && rows!=4)throw std::runtime_error("DSV4F_CPU_MXFP4_ROWS must be 1, 2 or 4");
        if(const char * v=std::getenv("DSV4F_CPU_SCHEDULE")) {
            if(!std::strcmp(v,"static"))contiguous=true;
            else if(std::strcmp(v,"dynamic"))throw std::runtime_error("DSV4F_CPU_SCHEDULE must be dynamic or static");
        }
        if(const char * v=std::getenv("DSV4F_CPU_AFFINITY")) {
            if(!std::strcmp(v,"physical"))physical=true;
            else if(std::strcmp(v,"none"))throw std::runtime_error("DSV4F_CPU_AFFINITY must be none or physical");
        }
        if(rows>1 && !dsv4f_mxfp4_rows_supported()) {
            std::fprintf(stderr,"[dsv4f-cpu] multi-row AVX2 unavailable; using reference dot product\n");rows=1;
        }
    }
};
const cpu_options & options(){static const cpu_options value;return value;}

const std::vector<int> & physical_cpus() {
    static const std::vector<int> cpus=[] {
        std::vector<int> result;
#if defined(__linux__)
        cpu_set_t allowed;CPU_ZERO(&allowed);
        if(sched_getaffinity(0,sizeof(allowed),&allowed))return result;
        std::set<std::pair<int,int>> seen;
        for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&allowed)) {
            const auto base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
            int core,package;
            if(!(std::ifstream(base+"core_id")>>core) || !(std::ifstream(base+"physical_package_id")>>package))continue;
            if(seen.insert({package,core}).second)result.push_back(cpu);
        }
#endif
        return result;
    }();return cpus;
}
// One persistent lane, shared by serial graph execution. No thread creation per layer.
class workers {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable work, done;
    std::function<void(int)> fn;
    int count = 0;
    // Scheduling writes must not invalidate the epoch polled by idle workers
    // or the batch descriptor read by every active worker.
    alignas(64) std::atomic<int> next{0};
    alignas(64) std::atomic<int> pending{0};
    alignas(64) std::atomic<unsigned> epoch{0};
    int spin_us = 0;
    bool fast_sync = true;
    bool stop = false;
    int width=1;
    std::vector<int> cpus;
    void bind(int id) {
#if defined(__linux__)
        char name[16];std::snprintf(name,sizeof(name),"dsv4f-cpu-%02d",id);pthread_setname_np(pthread_self(),name);
        if(!cpus.empty()) {
            cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpus[id],&mask);
            const int error=pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask);
            if(error)std::fprintf(stderr,"[dsv4f-cpu] affinity failed worker=%d cpu=%d error=%d\n",id,cpus[id],error);
        }
#else
        (void)id;
#endif
    }
    void consume(int id) {
        if(options().contiguous){for(int i=int(int64_t(count)*id/width),end=int(int64_t(count)*(id+1)/width);i<end;++i)fn(i);}
        else for(int i;(i=next.fetch_add(1,std::memory_order_relaxed))<count;)fn(i);
    }
public:
    workers() {
        const char * e = std::getenv("DSV4F_CPU_THREADS");
        int n = e ? std::atoi(e) : 11;
        n = std::max(1, std::min(n, 128));
        const char * spin = std::getenv("DSV4F_CPU_SPIN_US");
        spin_us = spin ? std::max(0, std::min(20000, std::atoi(spin))) : 300;
        const char * sync = std::getenv("DSV4F_CPU_FAST_SYNC");
        fast_sync = !sync || std::atoi(sync) != 0;
        width=n;
        if(options().physical) {
            const auto & topology=physical_cpus();
            if(int(topology.size())>=n)cpus.assign(topology.begin(),topology.begin()+n);
            else std::fprintf(stderr,"[dsv4f-cpu] physical affinity unavailable for %d workers (%zu eligible cores); using scheduler\n",n,topology.size());
        }
        std::fprintf(stderr,"[dsv4f-cpu] workers=%d affinity=%s schedule=%s chunk=%d dot_rows=%d dot_isa=%s cpus=",n,cpus.empty()?"none":"physical",options().contiguous?"static":"dynamic",options().chunk,options().rows,dsv4f_mxfp4_rows_isa());
        for(int cpu:cpus)std::fprintf(stderr,"%d,",cpu);
        std::fprintf(stderr,"\n");
        for (int i = 1; i < n; ++i) threads.emplace_back([this,i] {
            bind(i);
            unsigned seen = 0;
            for (;;) {
                const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us);
                while (spin_us && epoch.load(std::memory_order_acquire) == seen && std::chrono::steady_clock::now() < until) {
#if defined(__x86_64__) || defined(_M_X64)
                    __builtin_ia32_pause();
#else
                    std::this_thread::yield();
#endif
                }
                // A spinning worker already observed the release publication.
                // Only sleeping workers need the condition-variable mutex.
                if (!fast_sync || seen == epoch.load(std::memory_order_acquire)) {
                    std::unique_lock<std::mutex> lock(mutex);
                    work.wait(lock, [&] { return stop || seen != epoch.load(std::memory_order_acquire); });
                    if (stop) return;
                }
                seen = epoch.load(std::memory_order_acquire);
                consume(i);
                if (fast_sync) {
                    // Release all output writes before publishing completion.
                    // Lock before notification to avoid racing the caller's wait.
                    if (pending.fetch_sub(1,std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lock(mutex);
                        done.notify_one();
                    }
                } else {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (--pending == 0) done.notify_one();
                }
            }
        });
    }
    ~workers() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        work.notify_all();
        for (auto & t : threads) t.join();
    }
    void run(int n, std::function<void(int)> f) {
        if (!n) return;
        static thread_local bool bound=false;if(!bound){bind(0);bound=true;}
        { std::lock_guard<std::mutex> lock(mutex); fn = std::move(f); count = n; next = 0; pending = threads.size(); ++epoch; }
        work.notify_all(); consume(0);
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [&] { return pending.load(std::memory_order_acquire) == 0; });
    }
};

void gate_up_rows_data(const ggml_tensor * gate,const ggml_tensor * up,const void * gate_data,const void * up_data,
                       const void * ag,const void * au,int start,int end,int ne,float limit,float * output) {
    const auto * tg=ggml_get_type_traits_cpu(gate->type),*tu=ggml_get_type_traits_cpu(up->type);
    const int tile=options().rows;
    const bool tiled=tile>1 && gate->type==GGML_TYPE_MXFP4 && up->type==GGML_TYPE_MXFP4 &&
        tg->vec_dot_type==GGML_TYPE_Q8_0 && tu->vec_dot_type==GGML_TYPE_Q8_0 && ne%32==0;
    int r=start;
    if(tiled) {
      const auto full=dsv4f_mxfp4_kernel_for(tile),tail=dsv4f_mxfp4_kernel_for(2);
      const char * gb=(const char*)gate_data,*ub=(const char*)up_data;
      for(;r<end;) {
        const int count=std::min(tile/2,end-r);const void * streams[4];float values[4];
        for(int i=0;i<count;++i){streams[2*i]=gb+(r+i)*gate->nb[1];streams[2*i+1]=ub+(r+i)*up->nb[1];}
        (count==tile/2?full:tail)(ne,streams,ag,values);
        for(int i=0;i<count;++i){const float g=std::fmin(values[2*i],limit),u=std::fmax(-limit,std::fmin(values[2*i+1],limit));output[r+i]=(g/(1.f+std::exp(-g)))*u;}
        r+=count;
      }
    }
    for(;r<end;++r){float g,u;tg->vec_dot(ne,&g,0,(const char*)gate_data+r*gate->nb[1],0,ag,0,1);tu->vec_dot(ne,&u,0,(const char*)up_data+r*up->nb[1],0,au,0,1);g=std::fmin(g,limit);u=std::fmax(-limit,std::fmin(u,limit));output[r]=(g/(1.f+std::exp(-g)))*u;}
}
void gate_up_rows(const ggml_tensor * gate,const ggml_tensor * up,const void * ag,const void * au,
                  int eid,int start,int end,int ne,float limit,float * output) {
    ggml_dsv4f_host_lease gate_lease((const char*)gate->data+eid*gate->nb[2],gate->nb[2]);
    ggml_dsv4f_host_lease up_lease((const char*)up->data+eid*up->nb[2],up->nb[2]);
    gate_up_rows_data(gate,up,gate_lease.data,up_lease.data,ag,au,start,end,ne,limit,output);
}
void down_rows_data(const ggml_tensor * down,const void * down_data,const void * hq,int start,int end,int nf,float * output) {
    const auto * td=ggml_get_type_traits_cpu(down->type);int r=start;
    const int tile=options().rows;
    if(tile>1 && down->type==GGML_TYPE_MXFP4 && td->vec_dot_type==GGML_TYPE_Q8_0 && nf%32==0) {
        const auto full=dsv4f_mxfp4_kernel_for(tile);const char * base=(const char*)down_data;
        for(;r<end;) {
            const int count=std::min(tile,end-r);const void * streams[4];
            for(int i=0;i<count;++i)streams[i]=base+(r+i)*down->nb[1];
            (count==tile?full:dsv4f_mxfp4_kernel_for(count))(nf,streams,hq,output+r);r+=count;
        }
    }
    for(;r<end;++r)td->vec_dot(nf,output+r,0,(const char*)down_data+r*down->nb[1],0,hq,0,1);
}
void down_rows(const ggml_tensor * down,const void * hq,int eid,int start,int end,int nf,float * output) {
    ggml_dsv4f_host_lease down_lease((const char*)down->data+eid*down->nb[2],down->nb[2]);
    down_rows_data(down,down_lease.data,hq,start,end,nf,output);
}

class host_lease_batch {
    const void * held[128];
    size_t count=0;
public:
    host_lease_batch() = default;
    host_lease_batch(const host_lease_batch &) = delete;
    host_lease_batch & operator=(const host_lease_batch &) = delete;
    ~host_lease_batch() {
        if (ggml_dsv4f_host_weights.release)
            while(count) ggml_dsv4f_host_weights.release(held[--count]);
    }
    const void * acquire(const void * data,size_t size) {
        GGML_ASSERT(count<128);
        const void * actual=ggml_dsv4f_host_weights.acquire ? ggml_dsv4f_host_weights.acquire(data,size) : data;
        held[count++]=actual;
        return actual;
    }
};

void gate_up_experts(workers & lane,const ggml_tensor * gate,const ggml_tensor * up,
                     const void * ag,const void * au,const int32_t * ids,const int * misses,int nm,
                     int ne,int nf,int chunk,int gc,float limit,float * h) {
    host_lease_batch leases;
    const void * gate_data[64],* up_data[64];
    for(int m=0;m<nm;++m) {
        const int eid=ids[misses[m]];
        gate_data[m]=leases.acquire((const char*)gate->data+eid*gate->nb[2],gate->nb[2]);
        up_data[m]=leases.acquire((const char*)up->data+eid*up->nb[2],up->nb[2]);
    }
    lane.run(nm*gc,[&](int task){const int m=task/gc,r0=task%gc*chunk;
        gate_up_rows_data(gate,up,gate_data[m],up_data[m],ag,au,r0,std::min(nf,r0+chunk),ne,limit,h+size_t(m)*nf);});
}

void down_experts(workers & lane,const ggml_tensor * down,const void * hq,const int32_t * ids,
                  const int * misses,int nm,int ne,int nf,size_t hs,int chunk,int dc,float * out) {
    host_lease_batch leases;
    const void * down_data[64];
    for(int m=0;m<nm;++m) {
        const int eid=ids[misses[m]];
        down_data[m]=leases.acquire((const char*)down->data+eid*down->nb[2],down->nb[2]);
    }
    lane.run(nm*dc,[&](int task){const int m=task/dc,r0=task%dc*chunk;
        down_rows_data(down,down_data[m],(const char*)hq+m*hs,r0,std::min(ne,r0+chunk),nf,out+size_t(m)*ne);});
}

// Called only by the device service thread. Scratch stays resident across layers.
void cpu_experts(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
                 const float * x, const int32_t * ids, const float * prob, int nk, uint64_t hit_mask,
                 float limit, float * result) {
    static workers lane;
    static const bool profiling = std::getenv("DSV4F_CPU_PROFILE") != nullptr;
    using clock = std::chrono::steady_clock;
    auto stamp = [&] { return profiling ? clock::now() : clock::time_point{}; };
    const auto t0 = stamp();
    const int ne=gate->ne[0], nf=up->ne[1];
    const auto * tg=ggml_get_type_traits_cpu(gate->type), *tu=ggml_get_type_traits_cpu(up->type), *td=ggml_get_type_traits_cpu(down->type);
    const auto * qg=ggml_get_type_traits_cpu(tg->vec_dot_type), *qu=ggml_get_type_traits_cpu(tu->vec_dot_type), *qd=ggml_get_type_traits_cpu(td->vec_dot_type);
    struct scratch { std::vector<uint8_t> ag,au,hq; std::vector<float> h,out; };
    static thread_local scratch storage;
    auto & a=storage;
    a.ag.resize(ggml_row_size(tg->vec_dot_type,ne)); a.au.resize(ggml_row_size(tu->vec_dot_type,ne));
    qg->from_float(x,a.ag.data(),ne);
    if(tg->vec_dot_type==tu->vec_dot_type) std::copy(a.ag.begin(),a.ag.end(),a.au.begin()); else qu->from_float(x,a.au.data(),ne);
    int misses[64],nm=0;
    for(int i=0;i<nk;++i)if(!(hit_mask&(uint64_t(1)<<i)))misses[nm++]=i;
    a.h.resize(size_t(nm)*nf);a.out.resize(size_t(nm)*ne);
    const size_t hs=ggml_row_size(td->vec_dot_type,nf);a.hq.resize(size_t(nm)*hs);
    const int chunk=options().chunk,gc=(nf+chunk-1)/chunk,dc=(ne+chunk-1)/chunk;
    const auto t1 = stamp();
    gate_up_experts(lane,gate,up,a.ag.data(),a.au.data(),ids,misses,nm,ne,nf,chunk,gc,limit,a.h.data());
    const auto t2 = stamp();
    for(int m=0;m<nm;++m)qd->from_float(a.h.data()+size_t(m)*nf,a.hq.data()+m*hs,nf);
    const auto t3 = stamp();
    down_experts(lane,down,a.hq.data(),ids,misses,nm,ne,nf,hs,chunk,dc,a.out.data());
    const auto t4 = stamp();
    std::fill_n(result,ne,0.0f);
    for(int m=0;m<nm;++m)for(int r=0;r<ne;++r)result[r]+=a.out[size_t(m)*ne+r]*prob[misses[m]];
    if (profiling) {
        const clock::time_point points[] = {t0,t1,t2,t3,t4,stamp()};
        static thread_local double sums[5] = {};
        static thread_local uint64_t calls = 0;
        for (int i=0;i<5;++i) sums[i] += std::chrono::duration<double,std::micro>(points[i+1]-points[i]).count();
        if (++calls % 1000 == 0) {
            std::fprintf(stderr,"[dsv4f-cpu] calls=1000 prepare_us=%.3f gu_us=%.3f quant_us=%.3f down_us=%.3f reduce_us=%.3f\n",sums[0]/1000,sums[1]/1000,sums[2]/1000,sums[3]/1000,sums[4]/1000);
            std::fill_n(sums,5,0.0);
        }
    }
}

void compute(ggml_tensor * dst, int ith, int, void *) {
    if(ith!=0)return;
    // Small verification batches retain the whole-expert CPU fallback too.
    if (dst->ne[1] > 1) {
        float limit; std::memcpy(&limit, &dst->op_params[8], sizeof(limit));
        const int nk=dst->src[1]->ne[0];
        int32_t ids[64]; float prob[64];
        for (int64_t t=0;t<dst->ne[1];++t) {
            for (int i=0;i<nk;++i) {
                ids[i]=*(const int32_t*)((const char*)dst->src[1]->data+t*dst->src[1]->nb[1]+i*dst->src[1]->nb[0]);
                prob[i]=*(const float*)((const char*)dst->src[2]->data+t*dst->src[2]->nb[2]+i*dst->src[2]->nb[1]);
                GGML_ASSERT(ids[i]>=0 && ids[i]<dst->src[3]->ne[2]);
            }
            if(dst->ne[1]<=4)ggml_dsv4f_routing_observe(ggml_dsv4f_routing_bind(dst->src[3]),ids,prob,nk);
            cpu_experts(dst->src[3],dst->src[4],dst->src[5],
                        (const float*)((const char*)dst->src[0]->data+t*dst->src[0]->nb[1]),
                        ids,prob,nk,0,limit,(float*)((char*)dst->data+t*dst->nb[1]));
        }
        return;
    }
    static workers lane;
    static const bool profiling = std::getenv("DSV4F_PROFILE") != nullptr;
    using clock = std::chrono::steady_clock;
    auto stamp = [&] { return profiling ? clock::now() : clock::time_point{}; };
    const auto t0 = stamp();
    const auto * x = dst->src[0];
    const auto * ids = dst->src[1];
    const auto * weights = dst->src[2];
    const auto * gate = dst->src[3];
    const auto * up = dst->src[4];
    const auto * down = dst->src[5];
    float limit;
    std::memcpy(&limit, &dst->op_params[8], sizeof(limit));
    const int ne = x->ne[0], nf = up->ne[1], nk = ids->ne[0];
    const auto * tg = ggml_get_type_traits_cpu(gate->type);
    const auto * tu = ggml_get_type_traits_cpu(up->type);
    const auto * td = ggml_get_type_traits_cpu(down->type);
    const auto * qg = ggml_get_type_traits_cpu(tg->vec_dot_type);
    const auto * qu = ggml_get_type_traits_cpu(tu->vec_dot_type);
    const auto * qd = ggml_get_type_traits_cpu(td->vec_dot_type);
    GGML_ASSERT(tg->vec_dot && tu->vec_dot && td->vec_dot && qg->from_float && qu->from_float && qd->from_float);
    std::vector<uint8_t> ag(ggml_row_size(tg->vec_dot_type, ne));
    std::vector<uint8_t> au(ggml_row_size(tu->vec_dot_type, ne));
    qg->from_float((const float *)x->data, ag.data(), ne);
    if (tg->vec_dot_type == tu->vec_dot_type) au = ag;
    else qu->from_float((const float *)x->data, au.data(), ne);
    std::vector<int32_t> selected(nk);
    std::vector<float> prob(nk);
    for (int i = 0; i < nk; ++i) {
        selected[i] = *(const int32_t *)((const char *)ids->data + i*ids->nb[0]);
        prob[i] = *(const float *)((const char *)weights->data + i*weights->nb[1]);
        GGML_ASSERT(selected[i] >= 0 && selected[i] < gate->ne[2]);
    }
    ggml_dsv4f_routing_observe(ggml_dsv4f_routing_bind(gate),selected.data(),prob.data(),nk);
    const auto t1 = stamp();
    uint64_t hit_mask = 0;
    if (ggml_dsv4f_experts.dispatch) {
        hit_mask = ggml_dsv4f_experts.dispatch(gate, up, down, (const float *)x->data,
                                             selected.data(), prob.data(), nk, limit);
    }
    const auto t2 = stamp();
    std::vector<int> misses;
    for (int i = 0; i < nk; ++i) if (!(hit_mask & (uint64_t(1) << i))) misses.push_back(i);
    const int nm = misses.size();
    std::vector<float> h(size_t(nm)*nf), out(size_t(nm)*ne);
    const size_t hs = ggml_row_size(td->vec_dot_type, nf);
    std::vector<uint8_t> hq(size_t(nm)*hs);
    const int chunk = options().chunk;
    const int gc = (nf + chunk - 1)/chunk, dc = (ne + chunk - 1)/chunk;
    const auto t3 = stamp();
    gate_up_experts(lane,gate,up,ag.data(),au.data(),selected.data(),misses.data(),nm,ne,nf,chunk,gc,limit,h.data());
    const auto t4 = stamp();
    for (int m = 0; m < nm; ++m) qd->from_float(h.data() + size_t(m)*nf, hq.data() + m*hs, nf);
    const auto t5 = stamp();
    down_experts(lane,down,hq.data(),selected.data(),misses.data(),nm,ne,nf,hs,chunk,dc,out.data());
    const auto t6 = stamp();
    auto * result = (float *)dst->data;
    std::fill(result, result + ne, 0.0f);
    if (hit_mask) ggml_dsv4f_experts.collect(result, ne);
    const auto t7 = stamp();
    for (int m = 0; m < nm; ++m) {
        const float w = prob[misses[m]];
        for (int r = 0; r < ne; ++r) result[r] += out[size_t(m)*ne + r]*w;
    }
    const auto t8 = stamp();
    if (ggml_dsv4f_experts.finish) ggml_dsv4f_experts.finish();
    if (profiling) {
        const auto t9 = stamp();
        static double totals[9] = {};
        static uint64_t calls = 0, routes = 0, misses_n = 0, bytes = 0;
        const clock::time_point points[] = {t0,t1,t2,t3,t4,t5,t6,t7,t8,t9};
        for (int i = 0; i < 9; ++i) totals[i] += std::chrono::duration<double,std::milli>(points[i+1]-points[i]).count();
        ++calls; routes += nk; misses_n += nm;
        bytes += uint64_t(nm)*(gate->nb[2]+up->nb[2]+down->nb[2]);
        if (calls % 43 == 0) {
            double total = 0; for (double v : totals) total += v;
            // A sample spans 43 custom-op calls (the target model's layer count).
            // collect is the remaining GPU wait after CPU work; admit only schedules copies.
            std::fprintf(stderr, "[dsv4f-profile] sample=%llu moe_calls=43 moe_ms=%.3f prepare=%.3f dispatch=%.3f setup=%.3f gu=%.3f quant_h=%.3f down=%.3f collect=%.3f reduce=%.3f admit=%.3f hit=%.2f%% cpu_weight_mib=%.2f\n",
                         (unsigned long long)(calls/43), total, totals[0],totals[1],totals[2],totals[3],totals[4],totals[5],totals[6],totals[7],totals[8],
                         100.0*(routes-misses_n)/routes, double(bytes)/(1ULL<<20));
            std::fill(std::begin(totals),std::end(totals),0.0);routes=misses_n=bytes=0;
        }
    }
}
}

ggml_tensor * dsv4f_moe(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids,
                       ggml_tensor * weights, ggml_tensor * gate, ggml_tensor * up,
                       ggml_tensor * down, float limit, ggml_tensor ** prefill_scratch) {
    if (!ggml_dsv4f_cpu_experts) ggml_dsv4f_cpu_experts = cpu_experts;
    GGML_ASSERT(ids->ne[0] <= 64 && x->ne[1] >= 1);
    const bool prefill = x->ne[1] >= 32;
    // TOP_K is a view into the full expert sort and retains its larger token
    // stride. The streamed host plan reads a compact [used experts, tokens].
    if (prefill) {
        ids = ggml_cont(ctx, ids);
        weights = ggml_cont(ctx, weights);
    }
    const int group = ggml_dsv4f_prefill_group();
    ggml_tensor * scratch = nullptr;
    if (prefill) {
        const ggml_dsv4f_prefill_layout layout(gate, up, down, x->ne[1]*ids->ne[0], group);
        if (prefill_scratch && *prefill_scratch && ggml_nbytes(*prefill_scratch) >= layout.size) {
            scratch = *prefill_scratch;
        } else {
            scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, layout.size);
            ggml_set_name(scratch, "dsv4f_prefill_scratch");
            if (prefill_scratch) *prefill_scratch = scratch;
        }
    }
    ggml_tensor * args[] = {x, ids, weights, gate, up, down, scratch};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], 1, 1, args, prefill ? 7 : 6, compute, 1, nullptr);
    static_assert(GGML_MAX_OP_PARAMS >= 13*sizeof(int32_t), "custom op parameters too small");
    // custom callback metadata occupies the first 24 bytes on 64-bit hosts.
    std::memcpy(&result->op_params[8], &limit, sizeof(limit));
    result->op_params[9] = GGML_DSV4F_MOE_TAG;
    result->op_params[11] = prefill ? 1 : 0;
    result->op_params[12] = group;
    return result;
}

static void cpu_join(ggml_tensor * dst,int,int,void *) {
    for(int64_t i=0;i<ggml_nelements(dst);++i)((float*)dst->data)[i]=((float*)dst->src[0]->data)[i]+((float*)dst->src[1]->data)[i];
}
ggml_tensor * dsv4f_moe_join(ggml_context * ctx,ggml_tensor * routed,ggml_tensor * shared) {
    routed->op_params[10]=1;
    ggml_tensor * args[]={routed,shared};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_F32,routed->ne[0],routed->ne[1],1,1,args,2,cpu_join,1,nullptr);
    result->op_params[9]=GGML_DSV4F_JOIN_TAG;return result;
}
