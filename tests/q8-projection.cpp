#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Compare the Q-B dispatch and graph replay against the existing GPU
// operation bit-for-bit, including views, broadcast samples, and fallbacks.
static bool use_default = false;
static void select_mode(size_t mode) {
    if (mode && use_default) unsetenv("GGML_HIP_Q8_PROJECTION");
    else setenv("GGML_HIP_Q8_PROJECTION", mode ? "1" : "0", 1);
}
static bool check(ggml_backend_t gpu,int ni,int no,int channels,int samples,int nt,bool shared,bool strided=false,bool biased=false) {
    struct state {ggml_context * ctx;ggml_backend_buffer_t buf;ggml_cgraph * graph;
                  ggml_tensor * ws,*gs,*xs,*w,*g,*x,*out,*ub,*gb;};
    std::vector<state> modes(2);
    for(size_t mode=0;mode<modes.size();++mode) {
        auto & s=modes[mode];s.ctx=ggml_init({4<<20,nullptr,true});
        s.ws=ggml_new_tensor_4d(s.ctx,GGML_TYPE_Q8_0,ni,no+(strided?3:0),channels+(strided?1:0),1);
        s.gs=shared?ggml_dup_tensor(s.ctx,s.ws):nullptr;
        s.w=strided?ggml_view_4d(s.ctx,s.ws,ni,no,channels,1,s.ws->nb[1],s.ws->nb[2],s.ws->nb[3],s.ws->nb[1]):s.ws;
        s.g=shared?(strided?ggml_view_4d(s.ctx,s.gs,ni,no,channels,1,s.gs->nb[1],s.gs->nb[2],s.gs->nb[3],s.gs->nb[1]):s.gs):nullptr;
        s.xs=ggml_new_tensor_4d(s.ctx,GGML_TYPE_F32,ni+(strided?32:0),nt,channels+(strided?1:0),samples);
        s.x=strided?ggml_view_4d(s.ctx,s.xs,ni,nt,channels,samples,s.xs->nb[1],s.xs->nb[2],s.xs->nb[3],32*4):s.xs;
        auto * up=ggml_mul_mat(s.ctx,s.w,s.x);
        if(biased){s.ub=ggml_new_tensor_4d(s.ctx,GGML_TYPE_F32,no,1,channels,1);up=ggml_add(s.ctx,up,s.ub);}
        if(shared) {
            auto * gate=ggml_mul_mat(s.ctx,s.g,s.x);
            if(biased){s.gb=ggml_new_tensor_4d(s.ctx,GGML_TYPE_F32,no,1,channels,1);gate=ggml_add(s.ctx,gate,s.gb);}
            s.out=mode>=2?ggml_swiglu_clamp(s.ctx,gate,up,1.25f):
                ggml_swiglu_split(s.ctx,ggml_clamp(s.ctx,gate,-INFINITY,1.25f),ggml_clamp(s.ctx,up,-1.25f,1.25f));
        } else s.out=up;
        s.graph=ggml_new_graph_custom(s.ctx,1024,false);ggml_build_forward_expand(s.graph,s.out);
        ggml_set_input(s.xs);ggml_set_output(s.out);
        s.buf=ggml_backend_alloc_ctx_tensors(s.ctx,gpu);if(!s.buf)return false;
        ggml_backend_buffer_set_usage(s.buf,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    std::mt19937 rng(85431);std::normal_distribution<float> normal(0,.03f);
    for(int gate=0;gate<(shared?2:1);++gate) {
        auto * w=gate?modes[0].gs:modes[0].ws;
        std::vector<float> values(ggml_nelements(w));for(auto & v:values)v=normal(rng);
        std::vector<uint8_t> bytes(ggml_nbytes(w));
        ggml_quantize_chunk(GGML_TYPE_Q8_0,values.data(),bytes.data(),0,ggml_nrows(w),ni,nullptr);
        for(auto & s:modes)ggml_backend_tensor_set(gate?s.gs:s.ws,bytes.data(),0,bytes.size());
    }
    if(biased)for(int gate=0;gate<(shared?2:1);++gate){std::vector<float> bias(ggml_nelements(modes[0].ub));for(auto & v:bias)v=normal(rng);for(auto & s:modes)ggml_backend_tensor_set(gate?s.gb:s.ub,bias.data(),0,bias.size()*4);}
    std::vector<float> input(ggml_nelements(modes[0].xs)),expected(ggml_nelements(modes[0].out)),actual(expected.size());
    bool ok=true;double worst=0;
    for(float scale:{.01f,1.f,20.f}) {
        for(auto & v:input)v=normal(rng)*scale;
        for(size_t mode=0;mode<modes.size();++mode) {
            auto & s=modes[mode];select_mode(mode);
            ggml_backend_tensor_set(s.xs,input.data(),0,input.size()*4);
            for(int i=0;i<3;++i)ok&=ggml_backend_graph_compute(gpu,s.graph)==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(s.out,actual.data(),0,actual.size()*4);
            if(!mode)expected=actual;
            double e=0,n=0;for(size_t i=0;i<actual.size();++i){const double d=actual[i]-expected[i];e+=d*d;n+=double(expected[i])*expected[i];ok&=std::isfinite(actual[i]);}
            const double rel=std::sqrt(e/std::max(n,1e-30));worst=std::max(worst,rel);ok&=rel<2e-6;ok&=std::memcmp(expected.data(),actual.data(),actual.size()*4)==0;
            if((nt>1 || (ni==128 && no==7)) && mode==1)ok&=std::memcmp(expected.data(),actual.data(),actual.size()*4)==0;
        }
    }
    for(size_t mode=0;mode<modes.size();++mode) {
        auto & s=modes[mode];select_mode(mode);
        std::vector<double> times;
        for(int i=-3;i<20;++i) {
            auto start=std::chrono::steady_clock::now();
            ok&=ggml_backend_graph_compute(gpu,s.graph)==GGML_STATUS_SUCCESS;ggml_backend_synchronize(gpu);
            if(i>=0)times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
        }
        std::sort(times.begin(),times.end());
        std::printf("Q8 projection graph K=%d M=%d channels=%d samples=%d tokens=%d shared=%d strided=%d biased=%d mode=%zu median_us=%.3f worst_relative=%.9g %s\n",
            ni,no,channels,samples,nt,int(shared),int(strided),int(biased),mode,times[10],worst,ok?"PASS":"FAIL");
    }
    for(auto & s:modes){ggml_backend_buffer_free(s.buf);ggml_free(s.ctx);}
    return ok;
}
int main(int argc, char ** argv) {
    use_default = argc == 2 && std::strcmp(argv[1], "--default") == 0;
    setenv("GGML_CUDA_MOE_CACHE","0",1);ggml_backend_load_all();
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);if(!gpu)return 77;
    if(std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu))),"ROCm")) {ggml_backend_free(gpu);return 77;}
    bool ok=check(gpu,1024,32768,1,1,1,false);
    ok&=check(gpu,1024,32768,2,2,1,false);
    ok&=check(gpu,1024,32768,2,2,1,false,true);
    ok&=check(gpu,1024,32768,1,1,1,false,false,true); // bias fusion falls back
    ok&=check(gpu,1024,32768,1,1,1,true); // gate/GLU epilogue falls back
    ok&=check(gpu,1024,32768,1,1,2,false); // multi-token falls back
    ok&=check(gpu,1024,32768,1,1,4,false);
    ok&=check(gpu,1024,32736,1,1,1,false); // other row count falls back
    ok&=check(gpu,1056,32768,1,1,1,false); // other inner dimension falls back
    ok&=check(gpu,4096,1024,8,1,1,false); // grouped O-A remains unchanged
    ggml_backend_free(gpu);return ok?0:1;
}
