#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/q8-tile.cuh"
#include "ggml-cuda/quantize.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static bool check(ggml_backend_t gpu,int ni,int no) {
    auto & ctx=*static_cast<ggml_backend_cuda_context*>(gpu->context);const auto stream=ctx.stream();
    auto * gc=ggml_init({1<<20,nullptr,true});
    auto * wt=ggml_new_tensor_2d(gc,GGML_TYPE_Q8_0,ni,no);
    auto * xt=ggml_new_tensor_2d(gc,GGML_TYPE_F32,ni,1);
    auto * yt=ggml_new_tensor_2d(gc,GGML_TYPE_F32,no,1);
    std::mt19937 rng(84431+ni+no);
    const float scales[]={0.f,.03f,-.125f,.0001f,.75f};
    std::vector<block_q8_0> host(size_t(ni/32)*no);
    for(size_t b=0;b<host.size();++b) {
        const auto h=ggml_fp32_to_fp16(scales[b%5]);std::memcpy(&host[b].d,&h,2);
        for(auto & x:host[b].qs)x=int(rng()%255)-127;
    }
    std::vector<block_q8_1> q(ni/32);
    for(size_t b=0;b<q.size();++b) {
        ggml_fp16_t h[]={ggml_fp32_to_fp16(.0001f*(1+b%17)),0};std::memcpy(&q[b].ds,h,4);
        for(auto & x:q[b].qs)x=int(rng()%255)-127;
    }
    const size_t bytes=host.size()*sizeof(block_q8_0);
    const int copies=std::min<size_t>(64,std::max<size_t>(1,((192ULL<<20)+bytes-1)/bytes));
    std::vector<char*> weights(copies);
    for(auto & p:weights){CUDA_CHECK(cudaMalloc((void**)&p,bytes));CUDA_CHECK(cudaMemcpy(p,host.data(),bytes,cudaMemcpyHostToDevice));}
    block_q8_1 * dq=nullptr;float * out=nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dq,q.size()*sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc((void**)&out,no*4));
    CUDA_CHECK(cudaMemcpy(dq,q.data(),q.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice));
    const auto launch=[&](int mode,int copy) {
        if(!mode)ggml_cuda_op_mul_mat_vec_q(ctx,wt,xt,yt,weights[copy],nullptr,(const char*)dq,out,0,no,1,ni,stream);
        const auto one=init_fastdiv_values(1);
#define TILE(M,W,R) if(mode==M)dsv4f_q8_tile_launch<W,R>(weights[copy],dq,out,ni,no,ni/32,1,1,0,0,no,0,0,no,one,one,stream)
        TILE(1,1,4);TILE(2,2,4);TILE(3,4,2);TILE(4,8,1);TILE(5,1,8);TILE(6,2,8);TILE(7,1,16);
#undef TILE
    };
    std::vector<float> reference(no),actual(no);launch(0,0);CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(reference.data(),out,no*4,cudaMemcpyDeviceToHost));
    bool ok=true;
    cudaGraph_t graphs[8];cudaGraphExec_t execs[8];std::vector<double> samples[8];
    for(int mode=0;mode<8;++mode) {
        launch(mode,0);CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(actual.data(),out,no*4,cudaMemcpyDeviceToHost));
        double e=0,n=0,oe=0,on=0,maximum=0;
        for(int r=0;r<no;++r){const double d=actual[r]-reference[r];e+=d*d;n+=double(reference[r])*reference[r];ok&=std::isfinite(actual[r]);}
        for(int r=0;r<no;r+=std::max(1,no/17)) {
            double expected=0;
            for(int b=0;b<ni/32;++b) {
                const auto & w=host[size_t(r)*(ni/32)+b];ggml_fp16_t wd,qd;
                std::memcpy(&wd,&w.d,2);std::memcpy(&qd,&q[b].ds,2);
                int dot=0;for(int k=0;k<32;++k)dot+=int(w.qs[k])*int(q[b].qs[k]);
                expected+=double(dot)*ggml_fp16_to_fp32(wd)*ggml_fp16_to_fp32(qd);
            }
            const double d=actual[r]-expected;oe+=d*d;on+=expected*expected;maximum=std::max(maximum,std::abs(d));
        }
        const double rel=std::sqrt(e/std::max(n,1e-30)),oracle=std::sqrt(oe/std::max(on,1e-30));
        const bool exact=std::memcmp(reference.data(),actual.data(),no*4)==0;
        ok&=rel<2e-6 && oracle<2e-6 && maximum<.001 && (mode!=4 || exact);
        std::printf("Q8 parity K=%d M=%d mode=%d relative=%.9g oracle=%.9g max_abs=%.9g exact=%d %s\n",ni,no,mode,rel,oracle,maximum,int(exact),ok?"PASS":"FAIL");
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));
        for(int i=0;i<copies;++i)launch(mode,i);
        CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[mode]));
        CUDA_CHECK(cudaGraphInstantiate(&execs[mode],graphs[mode],nullptr,nullptr,0));
    }
    cudaEvent_t begin,end;CUDA_CHECK(hipEventCreate(&begin));CUDA_CHECK(hipEventCreate(&end));
    for(int rep=-3;rep<12;++rep)for(int k=0;k<8;++k) {
        const int mode=(rep+8+k)%8;
        CUDA_CHECK(cudaEventRecord(begin,stream));CUDA_CHECK(cudaGraphLaunch(execs[mode],stream));
        CUDA_CHECK(cudaEventRecord(end,stream));CUDA_CHECK(cudaEventSynchronize(end));
        float ms;CUDA_CHECK(hipEventElapsedTime(&ms,begin,end));
        if(rep>=0)samples[mode].push_back(ms/copies);
    }
    for(int mode=0;mode<8;++mode) {
        auto & s=samples[mode];std::sort(s.begin(),s.end());
        std::printf("Q8 timing K=%d M=%d copies=%d mode=%d median_ms=%.6f effective_GBs=%.3f\n",ni,no,copies,mode,s[s.size()/2],bytes/1e6/s[s.size()/2]);
        CUDA_CHECK(cudaGraphExecDestroy(execs[mode]));CUDA_CHECK(cudaGraphDestroy(graphs[mode]));
    }
    CUDA_CHECK(cudaEventDestroy(begin));CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(out));CUDA_CHECK(cudaFree(dq));for(auto * p:weights)CUDA_CHECK(cudaFree(p));ggml_free(gc);
    return ok;
}

// Exercise the real fused gate/up entry point, including its input quantizer.
// Rotate enough independent weight storage to exceed the GPU cache.
static bool check_fused(ggml_backend_t gpu,int ni,int no) {
    auto & ctx=*static_cast<ggml_backend_cuda_context*>(gpu->context);auto stream=ctx.stream();
    const size_t weight_bytes=size_t(ni/32)*no*sizeof(block_q8_0);
    const int copies=std::min<size_t>(64,std::max<size_t>(1,((192ULL<<20)+2*weight_bytes-1)/(2*weight_bytes)));
    auto * gc=ggml_init({2<<20,nullptr,true});
    auto * x=ggml_new_tensor_2d(gc,GGML_TYPE_F32,ni,1);
    auto * y=ggml_new_tensor_2d(gc,GGML_TYPE_F32,no,1);
    std::vector<ggml_tensor*> up(copies),gate(copies);
    for(int i=0;i<copies;++i){up[i]=ggml_new_tensor_2d(gc,GGML_TYPE_Q8_0,ni,no);gate[i]=ggml_dup_tensor(gc,up[i]);}
    auto * buffer=ggml_backend_alloc_ctx_tensors(gc,gpu);if(!buffer)return false;
    ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(95534+ni+no);std::normal_distribution<float> normal(0,.03f);
    std::vector<float> input(ni);for(auto & v:input)v=normal(rng);
    ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
    std::vector<block_q8_0> hu(size_t(ni/32)*no),hg(hu.size());
    const float scales[]={0.f,.0003f,-.00125f,.000001f,.0075f};
    for(auto * plane:{&hu,&hg})for(size_t b=0;b<plane->size();++b) {
        auto h=ggml_fp32_to_fp16(scales[b%5]);std::memcpy(&(*plane)[b].d,&h,2);
        for(auto & q:(*plane)[b].qs)q=int(rng()%255)-127;
    }
    for(int i=0;i<copies;++i){ggml_backend_tensor_set(up[i],hu.data(),0,weight_bytes);ggml_backend_tensor_set(gate[i],hg.data(),0,weight_bytes);}
    const auto launch=[&](int i){ggml_cuda_mm_fusion_args_host f{};f.gate=gate[i];f.glu_op=GGML_GLU_OP_SWIGLU_CLAMP;f.glu_limit=1.25f;ggml_cuda_mul_mat_vec_q(ctx,up[i],x,nullptr,y,&f);};
    block_q8_1 * dq=nullptr;CUDA_CHECK(cudaMalloc((void**)&dq,ni/32*sizeof(block_q8_1)));
    quantize_row_q8_1_cuda((const float*)x->data,nullptr,dq,GGML_TYPE_Q8_0,ni,ni,1,1,ni,1,1,1,stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<block_q8_1> q(ni/32);CUDA_CHECK(cudaMemcpy(q.data(),dq,q.size()*sizeof(block_q8_1),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaFree(dq));
    auto dot=[&](const std::vector<block_q8_0> & plane,int row){double v=0;for(int b=0;b<ni/32;++b){const auto & w=plane[size_t(row)*(ni/32)+b];ggml_fp16_t wd,qd;std::memcpy(&wd,&w.d,2);std::memcpy(&qd,&q[b].ds,2);int sum=0;for(int k=0;k<32;++k)sum+=int(w.qs[k])*int(q[b].qs[k]);v+=double(sum)*ggml_fp16_to_fp32(wd)*ggml_fp16_to_fp32(qd);}return v;};
    std::vector<float> reference(no),actual(no);bool ok=true;
    cudaGraph_t graphs[8];cudaGraphExec_t execs[8];std::vector<double> samples[8];
    for(int mode=0;mode<8;++mode) {
        setenv("GGML_HIP_Q8_TILE",mode?"1":"0",1);const auto setting=std::to_string(mode);setenv("GGML_HIP_Q8_TILE_TUNE",setting.c_str(),1);
        launch(0);CUDA_CHECK(cudaStreamSynchronize(stream));ggml_backend_tensor_get(y,actual.data(),0,no*4);
        if(!mode)reference=actual;
        double e=0,n=0,oe=0,on=0,maximum=0;
        for(int r=0;r<no;++r){const double d=actual[r]-reference[r];e+=d*d;n+=double(reference[r])*reference[r];ok&=std::isfinite(actual[r]);}
        for(int r=0;r<no;r+=std::max(1,no/17)) {const double g=std::min(dot(hg,r),1.25),u=std::clamp(dot(hu,r),-1.25,1.25),expected=g/(1.+std::exp(-g))*u,d=actual[r]-expected;oe+=d*d;on+=expected*expected;maximum=std::max(maximum,std::abs(d));}
        const double rel=std::sqrt(e/std::max(n,1e-30)),oracle=std::sqrt(oe/std::max(on,1e-30));ok&=rel<2e-6 && oracle<2e-6 && maximum<.001;
        std::printf("Q8 fused parity K=%d M=%d mode=%d relative=%.9g oracle=%.9g max_abs=%.9g %s\n",ni,no,mode,rel,oracle,maximum,ok?"PASS":"FAIL");
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));for(int i=0;i<copies;++i)launch(i);CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[mode]));CUDA_CHECK(cudaGraphInstantiate(&execs[mode],graphs[mode],nullptr,nullptr,0));
    }
    cudaEvent_t begin,end;CUDA_CHECK(hipEventCreate(&begin));CUDA_CHECK(hipEventCreate(&end));
    for(int rep=-3;rep<24;++rep)for(int k=0;k<8;++k) {const int mode=(rep+8+k)%8;CUDA_CHECK(cudaEventRecord(begin,stream));CUDA_CHECK(cudaGraphLaunch(execs[mode],stream));CUDA_CHECK(cudaEventRecord(end,stream));CUDA_CHECK(cudaEventSynchronize(end));float ms;CUDA_CHECK(hipEventElapsedTime(&ms,begin,end));if(rep>=0)samples[mode].push_back(ms/copies);}
    for(int mode=0;mode<8;++mode){auto & v=samples[mode];std::sort(v.begin(),v.end());std::printf("Q8 fused timing K=%d M=%d copies=%d mode=%d median_ms=%.6f\n",ni,no,copies,mode,v[v.size()/2]);CUDA_CHECK(cudaGraphExecDestroy(execs[mode]));CUDA_CHECK(cudaGraphDestroy(graphs[mode]));}
    CUDA_CHECK(cudaEventDestroy(begin));CUDA_CHECK(cudaEventDestroy(end));ggml_backend_buffer_free(buffer);ggml_free(gc);return ok;
}

int main() {
    setenv("GGML_CUDA_MOE_CACHE","0",1);
    setenv("GGML_HIP_Q8_TILE","0",1); // Keep the reference on the preceding dispatch.
    ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 77;bool ok=true;
    for(auto shape: {std::pair{4096,1024},std::pair{4096,2048},std::pair{2048,4096},std::pair{128,7}})
        ok&=check(gpu,shape.first,shape.second);
    ok&=check_fused(gpu,4096,2048);
    ggml_backend_free(gpu);return ok?0:1;
}
