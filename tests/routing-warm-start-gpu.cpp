#include "moe.h"
#include "ggml-backend.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static std::array<intptr_t,512> history{};
static std::atomic<unsigned> observations{0};
static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}

// Tiny synthetic weights test transfers and numerical parity, without timing
// requests or loading a real model.
static void check(ggml_backend_t gpu,ggml_backend_t cpu,ggml_type type,bool cached) {
    constexpr int ne=512,nf=512,experts=8,nk=2;
    history.fill(0);observations=0;
    require(ggml_dsv4f_experts.gpu_warm_start()==0,"uninitialized service skips preload");
    auto * wc=ggml_init({1<<20,nullptr,true});
    auto * g=ggml_new_tensor_3d(wc,type,ne,nf,experts),*u=ggml_dup_tensor(wc,g),*d=ggml_new_tensor_3d(wc,type,nf,ne,experts);
    auto * wb=ggml_backend_alloc_ctx_tensors(wc,cpu);
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(4101);std::normal_distribution<float> normal(0,.03f);
    for(auto * t:{g,u,d}) {
        {
            std::vector<float> f(ggml_nelements(t));for(auto & v:f)v=normal(rng);
            ggml_quantize_chunk(type,f.data(),t->data,0,t->ne[1]*t->ne[2],t->ne[0],nullptr);
        }
    }
    auto * ctx=ggml_init({4<<20,nullptr,true});
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ne,1),*ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,nk,1),*w=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,nk,1);
    auto * out=dsv4f_moe(ctx,x,ids,w,g,u,d,1.25f);
    auto * graph=ggml_new_graph_custom(ctx,1024,false);ggml_build_forward_expand(graph,out);
    for(auto * t:{x,ids,w})ggml_set_input(t);ggml_set_output(out);
    ggml_backend_t backends[]={gpu,cpu};auto * sched=ggml_backend_sched_new(backends,nullptr,2,1024,false,true);
    require(ggml_backend_sched_alloc_graph(sched,graph) && ggml_backend_sched_get_tensor_backend(sched,out)==gpu,"GPU service placement");
    std::vector<float> in(ne),actual(ne),expected(ne);for(auto & v:in)v=normal(rng)*8;
    int32_t selected[nk]={0,1};const float prob[nk]={.4f,.6f};
    auto compute=[&]{
        ggml_backend_tensor_set(x,in.data(),0,in.size()*sizeof(float));
        ggml_backend_tensor_set(ids,selected,0,sizeof(selected));ggml_backend_tensor_set(w,prob,0,sizeof(prob));
        require(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS,"GPU computation");
        ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));ggml_dsv4f_experts.gpu_drain();
    };
    compute();require(ggml_dsv4f_experts.gpu_warm_start()==0,"zero history skips transfers and drains admissions");
    for(int i=0;i<experts;++i)history[i]=100+i;

    uint64_t routes,hits,admissions;ggml_dsv4f_experts.gpu_statistics(&routes,&hits,&admissions);
    const auto observed=observations.load();
    const auto hooks=ggml_dsv4f_routing;ggml_dsv4f_routing={};
    require(ggml_dsv4f_experts.gpu_warm_start()==0,"disabled routing skips transfers");ggml_dsv4f_routing=hooks;
    const auto copied=ggml_dsv4f_experts.gpu_warm_start();
    require(cached?copied>=2 && copied<experts:copied==0,"preload bounded by existing per-format capacity");

    require(observations.load()==observed && history[7]==107,"preload does not observe routes");
    uint64_t r,h,a;ggml_dsv4f_experts.gpu_statistics(&r,&h,&a);
    require(r==routes && h==hits && a==admissions,"preload leaves demand statistics unchanged");
    require(ggml_dsv4f_experts.gpu_warm_start()==0,"hot residents are retained without repeated copies");
    selected[0]=7;selected[1]=6;compute();ggml_dsv4f_experts.gpu_statistics(&r,&h,&a);
    require(r-routes==nk && h-hits==(cached?nk:0),"previously unrequested hot experts hit immediately after preload");
    ggml_dsv4f_cpu_experts(g,u,d,in.data(),selected,prob,nk,0,1.25f,expected.data());
    double error=0,norm=0;
    for(int i=0;i<ne;++i){require(std::isfinite(actual[i]),"finite GPU output");error+=std::pow(actual[i]-expected[i],2);norm+=std::pow(expected[i],2);}
    require(std::sqrt(error/std::max(norm,1e-30))<.04,"preloaded payloads match CPU reference");
    std::printf("Routing VRAM preload %s cache=%d copies=%zu parity PASS\n",ggml_type_name(type),int(cached),copied);
    ggml_dsv4f_experts.invalidate(g->data,ggml_nbytes(g));ggml_backend_sched_free(sched);
    ggml_free(ctx);ggml_backend_buffer_free(wb);ggml_free(wc);
}

int main(int argc,char **) {
    const bool cached=argc==1;
    setenv("DSV4F_GPU_SERVICE","1",1);setenv("DSV4F_GPU_WAIT","callback",1);
    setenv("DSV4F_EXPERT_CACHE_MIB",cached?"3":"0",1);setenv("GGML_CUDA_MOE_CACHE","0",1);
    setenv("DSV4F_EXPERT_LOOKAHEAD","0",1);setenv("DSV4F_MTP","0",1);
    ggml_dsv4f_routing={[](const ggml_tensor *)->void *{return history.data();},
        [](void *,const int32_t *,const float * w,int n){for(int i=0;i<n;++i)if(w[i]!=0)++observations;},
        [](const void * row,int id)->intptr_t{return ((const intptr_t *)row)[id];}};
    ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 77;
    auto * cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,4);
    require(ggml_dsv4f_experts.gpu_warm_start,"GPU preload hook registered");
    for(auto type:{GGML_TYPE_MXFP4,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K})check(gpu,cpu,type,cached);
    ggml_dsv4f_routing={};ggml_backend_free(gpu);ggml_backend_free(cpu);
}
