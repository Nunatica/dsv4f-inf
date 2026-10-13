#include "moe.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

static bool check(ggml_backend_t gpu, ggml_type gt, ggml_type ut, ggml_type dt, float limit) {
    auto * ctx = ggml_init({64*1024*1024, nullptr, false});
    constexpr int ne=256, nf=512, experts=8, used=3;
    auto * g=ggml_new_tensor_3d(ctx,gt,ne,nf,experts);
    auto * u=ggml_new_tensor_3d(ctx,ut,ne,nf,experts);
    auto * d=ggml_new_tensor_3d(ctx,dt,nf,ne,experts);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ne,1);
    auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,1);
    auto * w=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,used,1);
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0,0.08);
    for(auto * t:{g,u,d}) {
        std::vector<float> f(ggml_nelements(t));
        for(float & v:f)v=normal(rng);
        ggml_quantize_chunk(t->type,f.data(),t->data,0,t->ne[1]*t->ne[2],t->ne[0],nullptr);
    }
    for(int i=0;i<ne;++i)((float *)x->data)[i]=normal(rng)*8;
    for(int i=0;i<used;++i){((int32_t *)ids->data)[i]=i*2+1;((float *)w->data)[i]=0.2f*(i+1);}
    auto * x3=ggml_reshape_3d(ctx,x,ne,1,1);
    auto * gate=ggml_mul_mat_id(ctx,g,x3,ids);
    auto * up=ggml_mul_mat_id(ctx,u,x3,ids);
    auto * h=std::isfinite(limit) ? ggml_swiglu_clamp(ctx,gate,up,limit) : ggml_swiglu_split(ctx,gate,up);
    auto * out=ggml_mul(ctx,ggml_mul_mat_id(ctx,d,h,ids),w);
    auto * ref=ggml_view_2d(ctx,out,ne,1,out->nb[2],0);
    for(int i=1;i<used;++i)ref=ggml_add(ctx,ref,ggml_view_2d(ctx,out,ne,1,out->nb[2],i*out->nb[1]));
    auto * gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,ref);
    auto saved=ggml_dsv4f_experts;
    ggml_dsv4f_experts={};
    auto ref_plan=ggml_graph_plan(gf,4,nullptr);
    std::vector<uint8_t> ref_work(ref_plan.work_size);
    ref_plan.work_data=ref_work.data();
    auto reference=[&]{if(ggml_graph_compute(gf,&ref_plan)!=GGML_STATUS_SUCCESS)std::abort();};
    reference();
    std::vector<float> gold((float *)ref->data,(float *)ref->data+ne);
    auto * fused=dsv4f_moe(ctx,x,ids,w,g,u,d,limit);
    auto * test=ggml_new_graph(ctx);ggml_build_forward_expand(test,fused);
    if(ggml_graph_compute_with_ctx(ctx,test,4)!=GGML_STATUS_SUCCESS)std::abort();
    float cpu_error=0;
    for(int i=0;i<ne;++i)cpu_error=std::max(cpu_error,std::abs(gold[i]-((float *)fused->data)[i]));
    bool ok=cpu_error<1e-4;
    printf("%s/%s/%s limit=%g CPU max_error=%g\n",ggml_type_name(gt),ggml_type_name(ut),ggml_type_name(dt),limit,cpu_error);
    // The GPU service calls a separate CPU entry point. Exercise worker
    // synchronization and reused scratch with changing inputs and hit masks.
    float service_error=0;
    const float full_w[used]={.2f,.4f,.6f};
    std::vector<float> service_result(ne);
    for(int repeat=0;repeat<8;++repeat) {
        for(int i=0;i<ne;++i)((float*)x->data)[i]=normal(rng)*8;
        for(uint64_t mask:{0ULL,2ULL,5ULL,7ULL}) {
            for(int i=0;i<used;++i)((float*)w->data)[i]=(mask&(1ULL<<i))?0:full_w[i];
            reference();
            ggml_dsv4f_cpu_experts(g,u,d,(float*)x->data,(int32_t*)ids->data,full_w,used,mask,limit,service_result.data());
            for(int i=0;i<ne;++i){ok=ok && std::isfinite(service_result[i]);service_error=std::max(service_error,std::abs(((float*)ref->data)[i]-service_result[i]));}
        }
    }
    for(int i=0;i<used;++i)((float*)w->data)[i]=full_w[i];
    reference();
    gold.assign((float*)ref->data,(float*)ref->data+ne);
    ok=ok && std::isfinite(service_error) && service_error<1e-4;
    printf("  service CPU mixed/zero-miss max_error=%g\n",service_error);
    ggml_dsv4f_experts=saved;
    if(saved.dispatch) {
        // Fill the complete-expert cache, then exercise GPU hits and CPU/GPU mixed routes.
        uint64_t hits=0;
        for(int iteration=0;iteration<100 && hits!=7;++iteration) {
            hits=saved.dispatch(g,u,d,(float *)x->data,(int32_t *)ids->data,(float *)w->data,used,limit);
            float ignored[ne];if(hits)saved.collect(ignored,ne);saved.finish();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if(hits!=7){printf("GPU cache failed to warm: mask=%llu\n",(unsigned long long)hits);ok=false;}
        for(int mixed=0;mixed<2;++mixed) {
            if(mixed) {
                ((int32_t *)ids->data)[1]=6;
                ggml_dsv4f_experts={};reference();ggml_dsv4f_experts=saved;
                gold.assign((float *)ref->data,(float *)ref->data+ne);
            }
            ggml_graph_compute_with_ctx(ctx,test,4);
            double err=0,norm=0;
            for(int i=0;i<ne;++i){double v=((float *)fused->data)[i]-gold[i];err+=v*v;norm+=gold[i]*gold[i];}
            double rel=std::sqrt(err/std::max(norm,1e-20));
            printf("  %s GPU relative_RMSE=%g\n",mixed?"mixed":"all-hit",rel);
            ok=ok && std::isfinite(rel) && rel<0.04;
            // Exercise the scheduler's on-device join contract repeatedly. CPU
            // output is only its miss contribution until the consumer joins it.
            auto * device_ctx=ggml_init({1024*1024,nullptr,true});
            auto * device_out=ggml_new_tensor_1d(device_ctx,GGML_TYPE_F32,ne);
            auto * device_buffer=ggml_backend_alloc_ctx_tensors(device_ctx,gpu);
            for (int repeat=0;repeat<8;++repeat) {
                saved.offer(fused->data,device_out->data,gpu);
                ggml_graph_compute_with_ctx(ctx,test,4);
                if (!saved.finalize(fused->data,gpu)) { ok=false; break; }
                ggml_backend_synchronize(gpu);
                std::vector<float> joined(ne);
                ggml_backend_tensor_get(device_out,joined.data(),0,ne*sizeof(float));
                double je=0,jn=0;
                for(int i=0;i<ne;++i){double v=joined[i]-gold[i];je+=v*v;jn+=gold[i]*gold[i];}
                const double jr=std::sqrt(je/std::max(jn,1e-20));
                ok=ok && std::isfinite(jr) && jr<0.04;
                if(!repeat)printf("  %s GPU join relative_RMSE=%g\n",mixed?"mixed":"all-hit",jr);
            }
            ggml_backend_buffer_free(device_buffer);ggml_free(device_ctx);
        }
        saved.invalidate(g->data,ggml_nbytes(g));
    }
    ggml_free(ctx);
    return ok;
}
int main() {
    setenv("GGML_CUDA_MOE_CACHE","0",1);
    setenv("DSV4F_EXPERT_CACHE_MIB","64",1);
    setenv("DSV4F_GPU_JOIN","1",1);
    ggml_backend_load_all();ggml_cpu_init();
    if(!ggml_dsv4f_experts.dispatch){fprintf(stderr,"GPU backend is required for complete expert parity tests\n");return 2;}
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 2;
    bool ok=true;
    ok &= check(gpu,GGML_TYPE_MXFP4,GGML_TYPE_MXFP4,GGML_TYPE_MXFP4,0.3f);
    ok &= check(gpu,GGML_TYPE_Q4_K,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K,10.f);
    ok &= check(gpu,GGML_TYPE_Q6_K,GGML_TYPE_Q4_K,GGML_TYPE_Q4_K,INFINITY);
    ggml_backend_free(gpu);
    printf("Complete expert parity: %s\n",ok?"PASS":"FAIL");
    return ok?0:1;
}
