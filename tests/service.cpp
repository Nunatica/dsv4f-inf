#include "moe.h"
#include "ggml-backend.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
static ggml_dsv4f_cpu_fn original_cpu;
static std::atomic<int> delayed{0};
static bool cached = true;
static void delayed_cpu(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d,const float * x,const int32_t * ids,const float * w,int nk,uint64_t hits,float limit,float * out) {
    if(delayed.fetch_sub(1)>0)std::this_thread::sleep_for(std::chrono::milliseconds(2));
    original_cpu(g,u,d,x,ids,w,nk,hits,limit,out);
}
static bool check(ggml_backend_t gpu,ggml_backend_t cpu,ggml_type type,ggml_type dtype,int nt=1) {
    constexpr int ne=256,nf=512,experts=8,nk=3;
    auto * weights=ggml_init({2*1024*1024,nullptr,true});
    auto * g=ggml_new_tensor_3d(weights,type,ne,nf,experts),*u=ggml_new_tensor_3d(weights,type,ne,nf,experts),*d=ggml_new_tensor_3d(weights,dtype,nf,ne,experts);
    auto * wb=ggml_backend_alloc_ctx_tensors(weights,cpu);ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(42);std::normal_distribution<float> normal(0,.08);
    for(auto * w:{g,u,d}) {std::vector<float> f(ggml_nelements(w));for(auto & v:f)v=normal(rng);ggml_quantize_chunk(w->type,f.data(),w->data,0,w->ne[1]*w->ne[2],w->ne[0],nullptr);}
    ggml_context * ctx[2];ggml_tensor * x[2],*ids[2],*prob[2],*out[2];ggml_cgraph * graph[2];ggml_backend_buffer_t buffer[2]={nullptr,nullptr};
    ggml_backend_t backends[]={gpu,cpu};auto * scheduler=ggml_backend_sched_new(backends,nullptr,2,4096,false,true);
    for(int k=0;k<2;++k) {
        ctx[k]=ggml_init({4*1024*1024,nullptr,true});x[k]=ggml_new_tensor_2d(ctx[k],GGML_TYPE_F32,ne,nt);ids[k]=ggml_new_tensor_2d(ctx[k],GGML_TYPE_I32,nk,nt);prob[k]=ggml_new_tensor_3d(ctx[k],GGML_TYPE_F32,1,nk,nt);
        if(k){out[k]=dsv4f_moe_join(ctx[k],dsv4f_moe(ctx[k],x[k],ids[k],prob[k],g,u,d,.3f),ggml_scale(ctx[k],x[k],.001f));}
        else{auto * x3=ggml_reshape_3d(ctx[k],x[k],ne,1,nt);auto * h=ggml_swiglu_clamp(ctx[k],ggml_mul_mat_id(ctx[k],g,x3,ids[k]),ggml_mul_mat_id(ctx[k],u,x3,ids[k]),.3f);auto * weighted=ggml_mul(ctx[k],ggml_mul_mat_id(ctx[k],d,h,ids[k]),prob[k]);auto * sum=ggml_view_2d(ctx[k],weighted,ne,nt,weighted->nb[2],0);for(int j=1;j<nk;++j)sum=ggml_add(ctx[k],sum,ggml_view_2d(ctx[k],weighted,ne,nt,weighted->nb[2],j*weighted->nb[1]));out[k]=ggml_add(ctx[k],sum,ggml_scale(ctx[k],x[k],.001f));}
        graph[k]=ggml_new_graph(ctx[k]);ggml_build_forward_expand(graph[k],out[k]);if(!k)buffer[k]=ggml_backend_alloc_ctx_tensors(ctx[k],cpu);
        else {ggml_set_input(x[k]);ggml_set_input(ids[k]);ggml_set_input(prob[k]);
            if(!ggml_backend_sched_alloc_graph(scheduler,graph[k]))std::abort();
            if(ggml_backend_sched_get_tensor_backend(scheduler,out[k])!=gpu || out[k]->src[0]->src[3]!=g)std::abort();
        }
    }
    original_cpu=ggml_dsv4f_cpu_experts;delayed=8;ggml_dsv4f_cpu_experts=delayed_cpu;
    bool ok=true;double worst=0;
    const bool resize_test=cached && type==GGML_TYPE_MXFP4 && dtype==GGML_TYPE_MXFP4;
    uint64_t full_routes=0,full_hits=0,unused=0;
    for(int iteration=0;iteration<96;++iteration){
        if(resize_test && iteration==24) {
            ggml_backend_sched_synchronize(scheduler);
            ggml_dsv4f_experts.gpu_trim_workspace(gpu);
            // More than 64 small experts exercises multiple extension blocks.
            setenv("DSV4F_EXPERT_CACHE_MIB","16",1);
            ok &= ggml_dsv4f_experts.gpu_resize(true);
        }
        if(resize_test && iteration==48) {
            ok &= ggml_dsv4f_experts.gpu_resize(true); // drain counters, retain allocation
            ggml_dsv4f_experts.gpu_statistics(&full_routes,&full_hits,&unused);
        }
        if(resize_test && iteration==64) {
            ok &= ggml_dsv4f_experts.gpu_resize(true);
            uint64_t r,h;ggml_dsv4f_experts.gpu_statistics(&r,&h,&unused);
            // The base fits five experts; all eight can hit only if the new
            // extension is read by GPU kernels, including graph replays.
            ok &= r-full_routes==uint64_t(16*nk*nt) && h-full_hits==r-full_routes;
            full_routes=r;full_hits=h;
            ggml_backend_sched_synchronize(scheduler);
            ggml_dsv4f_experts.gpu_trim_workspace(gpu);
            ok &= ggml_dsv4f_experts.gpu_resize(false);
            setenv("DSV4F_EXPERT_CACHE_MIB","1",1);
        }
        if(resize_test && iteration==80) {
            ok &= ggml_dsv4f_experts.gpu_resize(false);
            uint64_t r,h;ggml_dsv4f_experts.gpu_statistics(&r,&h,&unused);
            // The shifted hot set includes three experts that first entered
            // through the extension. They must survive its removal via D2D.
            ok &= r-full_routes==uint64_t(16*nk*nt) && h-full_hits==r-full_routes;
        }
        std::vector<float> input(ne*nt),a(ne*nt),b(ne*nt);for(auto & v:input)v=normal(rng)*8;std::vector<int> selected(nk*nt);std::vector<float> w(nk*nt);for(int t=0;t<nt;++t)for(int j=0;j<nk;++j){
            selected[t*nk+j]=resize_test && iteration<80?(iteration+t*3+j)%5+(iteration<24?0:3):(iteration+t*3+j*2)%experts;
            w[t*nk+j]=float(j+1)/6;}
        for(int k=0;k<2;++k){ggml_backend_tensor_set(x[k],input.data(),0,ne*nt*sizeof(float));ggml_backend_tensor_set(ids[k],selected.data(),0,selected.size()*sizeof(int));ggml_backend_tensor_set(prob[k],w.data(),0,w.size()*sizeof(float));if((k?ggml_backend_sched_graph_compute(scheduler,graph[k]):ggml_backend_graph_compute(cpu,graph[k]))!=GGML_STATUS_SUCCESS)ok=false;}
        ggml_backend_tensor_get(out[0],a.data(),0,ne*nt*sizeof(float));ggml_backend_tensor_get(out[1],b.data(),0,ne*nt*sizeof(float));double error=0,norm=0;for(int i=0;i<ne*nt;++i){error+=(a[i]-b[i])*(a[i]-b[i]);norm+=a[i]*a[i];ok&=std::isfinite(b[i]);}double rel=std::sqrt(error/std::max(norm,1e-20));worst=std::max(worst,rel);ok&=rel<.04;
    }
    uint64_t routes=0,hits=0,admissions=0;ggml_dsv4f_experts.gpu_statistics(&routes,&hits,&admissions);
    const char * policy=std::getenv("DSV4F_CACHE_POLICY");
    const bool conservative=policy && !std::strcmp(policy,"aged");
    // Uniform Q4_K/Q6_K routes need not churn all eight experts with a conservative
    // admission threshold. Require cache population here; the policy test and
    // MXFP4 hot-set shift independently check adaptation and replacement.
    ok &= routes>0 && (cached ? hits>0 && hits<routes && admissions>=(conservative?4:6) : hits==0 && admissions==0);
    printf("Device service + deferred shared join nt=%d %s/%s worst relative RMSE=%g routes=%llu hits=%llu admissions=%llu %s\n",nt,ggml_type_name(type),ggml_type_name(dtype),worst,(unsigned long long)routes,(unsigned long long)hits,(unsigned long long)admissions,ok?"PASS":"FAIL");
    // Both the service and loader must retire before the source RAM is freed.
    ggml_dsv4f_experts.invalidate(g->data,ggml_nbytes(g));ggml_dsv4f_cpu_experts=original_cpu;
    ggml_backend_sched_free(scheduler);
    for(int k=0;k<2;++k){if(buffer[k])ggml_backend_buffer_free(buffer[k]);ggml_free(ctx[k]);}ggml_backend_buffer_free(wb);ggml_free(weights);return ok;
}
int main(int argc,char ** argv) {
    cached = argc < 3;
    setenv("DSV4F_GPU_SERVICE","1",1);setenv("DSV4F_GPU_WAIT",argc>1?argv[1]:"poll",1);setenv("DSV4F_EXPERT_CACHE_MIB",cached?"1":"0",1);setenv("GGML_CUDA_MOE_CACHE","0",1);
    ggml_backend_load_all();ggml_cpu_init();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
    if(!gpu||!cpu)return 2;
    bool ok=check(gpu,cpu,GGML_TYPE_MXFP4,GGML_TYPE_MXFP4);ok&=check(gpu,cpu,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K);for(int nt:{2,4}){ok&=check(gpu,cpu,GGML_TYPE_MXFP4,GGML_TYPE_MXFP4,nt);ok&=check(gpu,cpu,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K,nt);}
    ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
}
