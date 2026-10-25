#include "moe.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool check_router(ggml_backend_t gpu,ggml_backend_t cpu,int ne,int nk,int nt,bool norm,bool strided) {
    struct state {ggml_context * ctx;ggml_backend_buffer_t buffer;ggml_cgraph * graph;
                  ggml_tensor * storage,*x,*bias,*ids,*weights;};
    state states[3]{};const int pitch=ne+(strided?7:0);constexpr float scale=1.7f;
    for(int mode=0;mode<3;++mode) {
        auto & a=states[mode];a.ctx=ggml_init({2<<20,nullptr,true});
        a.storage=ggml_new_tensor_2d(a.ctx,GGML_TYPE_F32,pitch,nt);
        a.x=ggml_view_2d(a.ctx,a.storage,ne,nt,pitch*4,0);
        a.bias=ggml_new_tensor_1d(a.ctx,GGML_TYPE_F32,ne);
        if(mode)dsv4f_router(a.ctx,a.x,a.bias,nk,norm,scale,&a.ids,&a.weights);
        else {
            auto * scores=ggml_sqrt(a.ctx,ggml_softplus(a.ctx,a.x));
            a.ids=ggml_cont(a.ctx,ggml_top_k(a.ctx,ggml_add(a.ctx,scores,a.bias),nk));
            a.weights=ggml_reshape_3d(a.ctx,ggml_get_rows(a.ctx,ggml_reshape_3d(a.ctx,scores,1,ne,nt),a.ids),1,nk,nt);
            if(norm) {auto * w=ggml_reshape_2d(a.ctx,a.weights,nk,nt);a.weights=ggml_reshape_3d(a.ctx,ggml_div(a.ctx,w,ggml_clamp(a.ctx,ggml_sum_rows(a.ctx,w),6.103515625e-5f,INFINITY)),1,nk,nt);}
            a.weights=ggml_scale(a.ctx,a.weights,scale);
        }
        ggml_set_input(a.storage);ggml_set_input(a.bias);ggml_set_output(a.ids);ggml_set_output(a.weights);
        a.graph=ggml_new_graph_custom(a.ctx,1024,false);ggml_build_forward_expand(a.graph,a.weights);ggml_build_forward_expand(a.graph,a.ids);
        a.buffer=ggml_backend_alloc_ctx_tensors(a.ctx,mode==2?cpu:gpu);if(!a.buffer)return false;
    }
    std::mt19937 rng(173+ne+nk+nt);std::uniform_real_distribution<float> dist(-25.f,40.f);
    std::vector<float> x(pitch*nt),bias(ne),ref(nk*nt),actual(nk*nt);std::vector<int32_t> ri(nk*nt),ai(nk*nt);
    bool ok=true;double max_error=0;
    for(int rep=0;rep<16;++rep) {
        for(auto & v:x)v=rep==0?0.f:rep==1?-100.f:rep==2?1000.f:rep==3?-30.f:rep==4?-20.f:dist(rng);
        for(auto & v:bias)v=rep<5?0.f:dist(rng)*.1f;
        for(int mode=0;mode<3;++mode) {
            auto & a=states[mode];ggml_backend_tensor_set(a.storage,x.data(),0,x.size()*4);ggml_backend_tensor_set(a.bias,bias.data(),0,bias.size()*4);
            ok&=ggml_backend_graph_compute(mode==2?cpu:gpu,a.graph)==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(a.ids,ai.data(),0,ai.size()*4);ggml_backend_tensor_get(a.weights,actual.data(),0,actual.size()*4);
            if(!mode){ri=ai;ref=actual;}
            else {
                ok&=ri==ai;
                for(size_t k=0;k<ref.size();++k){max_error=std::max(max_error,double(std::abs(ref[k]-actual[k])));ok&=std::isfinite(actual[k]) && std::abs(ref[k]-actual[k])<=2e-6f*std::max(1.f,std::abs(ref[k]));}
                if(mode==1 && std::memcmp(ref.data(),actual.data(),ref.size()*4)!=0) {
                    std::fprintf(stderr,"router GPU bitwise mismatch ne=%d nk=%d nt=%d norm=%d strided=%d rep=%d max_error=%g\n",ne,nk,nt,int(norm),int(strided),rep,max_error);ok=false;
                }
            }
        }
    }
    double times[2]{};
    if(ne==256 && nk==6 && nt==1 && norm && !strided)for(int mode=0;mode<2;++mode) {
        std::vector<double> samples;
        for(int rep=-5;rep<51;++rep){auto start=std::chrono::steady_clock::now();ggml_backend_graph_compute(gpu,states[mode].graph);ggml_backend_synchronize(gpu);if(rep>=0)samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());}
        std::sort(samples.begin(),samples.end());times[mode]=samples[samples.size()/2];
    }
    std::printf("router ne=%d nk=%d nt=%d norm=%d strided=%d GPU-bitwise CPU-tolerance %s max_error=%g old_us=%.3f fused_us=%.3f\n",ne,nk,nt,int(norm),int(strided),ok?"PASS":"FAIL",max_error,times[0],times[1]);
    for(auto & a:states){ggml_backend_buffer_free(a.buffer);ggml_free(a.ctx);}return ok;
}
int main() {
    setenv("GGML_CUDA_MOE_CACHE","0",1);ggml_backend_load_all();
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);if(!gpu)return 77;
    if(std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu))),"ROCm")) {
        ggml_backend_free(gpu);return 77; // The fused GPU path is HIP-only.
    }
    auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);if(!cpu)return 1;
    bool ok=true;
    for(int ne:{6,32,255,256,512})for(int nk:{1,6,32})if(nk<=ne)for(int nt:{1,2,4})for(bool norm:{false,true})for(bool strided:{false,true})
        ok&=check_router(gpu,cpu,ne,nk,nt,norm,strided);
    ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
}
