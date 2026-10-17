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

static bool check(ggml_backend_t gpu, ggml_backend_t cpu, int nt, float limit, int ne=512, int nf=512, int nex=37) {
    const int nk=std::min(6,nex);
    const int depth=nt==257?3:1;
    ggml_context * wc=ggml_init({2*1024*1024,nullptr,true});
    auto * g=ggml_new_tensor_3d(wc,GGML_TYPE_MXFP4,ne,nf,nex);
    auto * u=ggml_dup_tensor(wc,g);
    auto * d=ggml_new_tensor_3d(wc,GGML_TYPE_MXFP4,nf,ne,nex);
    auto * wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu)));
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(731); std::normal_distribution<float> normal(0,.08f);
    for(auto * w:{g,u,d}) {
        std::vector<float> values(ggml_nelements(w)); for(auto & v:values)v=normal(rng);
        ggml_quantize_chunk(w->type,values.data(),w->data,0,w->ne[1]*w->ne[2],w->ne[0],nullptr);
    }
    ggml_backend_t backends[]={gpu,cpu};
    auto * sched=ggml_backend_sched_new(backends,nullptr,2,1024,false,true);
    ggml_context * contexts[2]; ggml_backend_buffer_t buffers[2]={};
    ggml_tensor * x[2], * ids[2], * prob[2], * out[2]; ggml_cgraph * graphs[2];
    for(int k=0;k<2;++k) {
        contexts[k]=ggml_init({4*1024*1024,nullptr,true});
        x[k]=ggml_new_tensor_2d(contexts[k],GGML_TYPE_F32,ne,nt);
        auto * full_ids=ggml_new_tensor_2d(contexts[k],GGML_TYPE_I32,nex,nt);
        ids[k]=ggml_view_2d(contexts[k],full_ids,nk,nt,full_ids->nb[1],0);
        prob[k]=ggml_new_tensor_3d(contexts[k],GGML_TYPE_F32,1,nk,nt);
        if(!k) {
            auto * dg=ggml_dup_tensor(contexts[k],g), * du=ggml_dup_tensor(contexts[k],u), * dd=ggml_dup_tensor(contexts[k],d);
            out[k]=x[k];
            for(int layer=0;layer<depth;++layer) {
                auto * input=ggml_reshape_3d(contexts[k],ggml_cont(contexts[k],out[k]),ne,1,nt);
                auto * gate=ggml_mul_mat_id(contexts[k],dg,input,ids[k]);
                auto * up=ggml_mul_mat_id(contexts[k],du,input,ids[k]);
                auto * hidden=ggml_swiglu_clamp(contexts[k],gate,up,limit);
                auto * weighted=ggml_mul(contexts[k],ggml_mul_mat_id(contexts[k],dd,hidden,ids[k]),prob[k]);
                out[k]=nullptr;
                for(int j=0;j<nk;++j) {
                    auto * v=ggml_view_2d(contexts[k],weighted,ne,nt,weighted->nb[2],j*weighted->nb[1]);
                    out[k]=out[k]?ggml_add(contexts[k],out[k],v):v;
                }
            }
            buffers[k]=ggml_backend_alloc_ctx_tensors(contexts[k],gpu);
            for(auto pair:{std::pair<ggml_tensor*,ggml_tensor*>{g,dg},{u,du},{d,dd}})
                ggml_backend_tensor_set(pair.second,pair.first->data,0,ggml_nbytes(pair.first));
        } else {
            ggml_tensor * scratch=nullptr;
            out[k]=x[k];
            for(int layer=0;layer<depth;++layer) {
                out[k]=dsv4f_moe(contexts[k],ggml_cont(contexts[k],out[k]),ids[k],prob[k],g,u,d,limit,&scratch);
                if(out[k]->src[6]!=scratch)std::abort();
            }
            ggml_set_input(x[k]);ggml_set_input(full_ids);ggml_set_input(prob[k]);ggml_set_output(out[k]);
        }
        graphs[k]=ggml_new_graph_custom(contexts[k],1024,false);ggml_build_forward_expand(graphs[k],out[k]);
        if(k) {
            if(!ggml_dsv4f_experts.gpu_supported(out[k]) || !ggml_backend_sched_alloc_graph(sched,graphs[k]))std::abort();
            if(ggml_backend_sched_get_tensor_backend(sched,out[k])!=gpu || out[k]->src[3]!=g ||
               ggml_backend_sched_get_tensor_backend(sched,out[k]->src[6])!=gpu)std::abort();
        }
    }
    bool ok=true; double worst=0;
    std::vector<float> input(ne*nt), weights(nk*nt), a(ne*nt), b(ne*nt);std::vector<int32_t> selected(nk*nt);
    const int sparse[]={0,3,7,12,18,21,29};
    for(int step=0;step<8;++step) {
        for(auto & v:input)v=normal(rng)*20;
        for(int t=0;t<nt;++t)for(int j=0;j<nk;++j) {
            selected[t*nk+j]=step%2?(nex>=30?sparse[(t+j+step)%7]:(t+j+step)%nex):(t*7+j*5+step)%nex;
            weights[t*nk+j]=float(j+1)/21;
        }
        for(int k=0;k<2;++k) {
            ggml_backend_tensor_set(x[k],input.data(),0,input.size()*4);
            std::vector<int32_t> padded(nex*nt,-1);
            for(int t=0;t<nt;++t)std::copy_n(selected.data()+t*nk,nk,padded.data()+t*nex);
            ggml_backend_tensor_set(ids[k]->view_src,padded.data(),0,padded.size()*4);
            ggml_backend_tensor_set(prob[k],weights.data(),0,weights.size()*4);
            ok&=(k?ggml_backend_sched_graph_compute(sched,graphs[k]):ggml_backend_graph_compute(gpu,graphs[k]))==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(out[k],k?b.data():a.data(),0,a.size()*4);
        }
        double error=0,norm=0;
        for(size_t i=0;i<a.size();++i){error+=double(a[i]-b[i])*(a[i]-b[i]);norm+=double(a[i])*a[i];ok&=std::isfinite(b[i]);}
        const double rel=std::sqrt(error/std::max(norm,1e-20));worst=std::max(worst,rel);ok&=rel<.003;
    }
    if(depth==1 && std::getenv("DSV4F_GPU_SERVICE")) {
        // Return leased slots to the ordinary decode kernels, including GPU
        // table lookups and admission. Compare with token zero of the reference.
        auto * dc=ggml_init({2*1024*1024,nullptr,true});
        auto * dx=ggml_new_tensor_2d(dc,GGML_TYPE_F32,ne,1);
        auto * di=ggml_new_tensor_2d(dc,GGML_TYPE_I32,nk,1);
        auto * dw=ggml_new_tensor_3d(dc,GGML_TYPE_F32,1,nk,1);
        auto * result=dsv4f_moe(dc,dx,di,dw,g,u,d,limit);
        auto * graph=ggml_new_graph(dc);ggml_build_forward_expand(graph,result);
        auto * ds=ggml_backend_sched_new(backends,nullptr,2,1024,false,true);
        ggml_set_input(dx);ggml_set_input(di);ggml_set_input(dw);ggml_set_output(result);
        ok&=ggml_backend_sched_alloc_graph(ds,graph);
        ggml_backend_tensor_set(dx,input.data(),0,ne*sizeof(float));
        ggml_backend_tensor_set(di,selected.data(),0,nk*sizeof(int32_t));
        ggml_backend_tensor_set(dw,weights.data(),0,nk*sizeof(float));
        ok&=ggml_backend_sched_graph_compute(ds,graph)==GGML_STATUS_SUCCESS;
        std::vector<float> decoded(ne);ggml_backend_tensor_get(result,decoded.data(),0,ne*sizeof(float));
        double error=0,norm=0;
        for(int i=0;i<ne;++i){error+=double(a[i]-decoded[i])*(a[i]-decoded[i]);norm+=double(a[i])*a[i];ok&=std::isfinite(decoded[i]);}
        const double rel=std::sqrt(error/std::max(norm,1e-20));ok&=rel<.04;
        std::printf("Prefill to decode nt=%d relative RMSE=%g %s\n",nt,rel,rel<.04?"PASS":"FAIL");
        ggml_backend_sched_free(ds);ggml_free(dc);
    }
    std::printf("Streamed prefill vs GPU MUL_MAT_ID ne=%d nf=%d nt=%d strided_ids clamp=%g worst_rel_rmse=%g %s\n",ne,nf,nt,limit,worst,ok?"PASS":"FAIL");
    ggml_dsv4f_experts.invalidate(g->data,ggml_nbytes(g));
    ggml_backend_sched_free(sched);
    for(int k=0;k<2;++k){if(buffers[k])ggml_backend_buffer_free(buffers[k]);ggml_free(contexts[k]);}
    ggml_backend_buffer_free(wb);ggml_free(wc);return ok;
}

int main(int argc, char ** argv) {
    const bool cached=argc>1;
    const bool boundary=cached && !std::strcmp(argv[1],"threshold");
    if(cached) {
        setenv("DSV4F_GPU_SERVICE","1",1);
        setenv("DSV4F_EXPERT_CACHE_MIB",!std::strcmp(argv[1],"zero-cache")?"0":"8",1);
        setenv("DSV4F_PREFILL_CACHE","1",1);
        setenv("DSV4F_PREFILL_CACHE_MIN_TOKENS",boundary?"4096":"32",1);
        setenv("DSV4F_PREFILL_GROUP","4",1);
    }
    setenv("GGML_CUDA_MOE_CACHE","0",1);
    ggml_backend_load_all();ggml_cpu_init();
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
    if(!gpu||!cpu)return 2;
    bool ok=true;
    if(boundary) {
        for(int nt:{4095,4096})ok&=check(gpu,cpu,nt,1.25f,512,512,9);
        ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
    }
    for(int nt:{32,127,257,2051})for(float limit:{1.25f,INFINITY})ok&=check(gpu,cpu,nt,limit);
    ok&=check(gpu,cpu,129,1.25f,4096,2048,8); // Cache too small: fallback.
    if(cached) {
        ok&=check(gpu,cpu,127,1.25f,512,512,8); // Entire active set can hit.
        if(std::strcmp(argv[1],"zero-cache")) {
            setenv("DSV4F_EXPERT_CACHE_MIB","128",1);
            ok&=check(gpu,cpu,129,1.25f,4096,2048,8); // Actual model geometry, direct cache MMQ.
        }
    }
    ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
}
