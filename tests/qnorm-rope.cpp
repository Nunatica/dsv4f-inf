#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/rope.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct fixture {
    ggml_backend_t gpu;ggml_context *ctx;ggml_backend_buffer_t buffer;
    ggml_tensor *base,*x,*positions,*factors,*norm,*out,*observer=nullptr;
    ggml_cgraph *graph;
    int d,nh,nq,ns,nd,off,type,alias,observed;
    float eps=1e-6f,freq_base=10000.f,freq_scale,ext,attn;
    std::vector<float> values,ff;
    std::vector<int32_t> pos;
    fixture(ggml_backend_t g,int heads,int queries,int streams=1,bool padded=false,
            int rope_type=0,int rotated=64,int offset=448,bool yarn=false,bool factor=false,
            int alias_mode=0,int observe=0,int width=512):
        gpu(g),d(width),nh(heads),nq(queries),ns(streams),nd(rotated),off(offset),type(rope_type),alias(alias_mode),observed(observe),
        freq_scale(yarn?.25f:1.f),ext(yarn?1.f:0.f),attn(yarn?.878f:1.f) {
        ctx=ggml_init({4*1024*1024,nullptr,true});
        if(padded) {
            base=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,d+16,nh+1,nq+1,ns);
            x=ggml_view_4d(ctx,base,d,nh,nq,ns,base->nb[1],base->nb[2],base->nb[3],16);
        } else {
            base=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,(int64_t)d*nh*nq*ns+4);
            x=ggml_view_4d(ctx,base,d,nh,nq,ns,d*4,d*nh*4,(int64_t)d*nh*nq*4,16);
        }
        positions=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,nq);
        factors=factor?ggml_new_tensor_1d(ctx,GGML_TYPE_F32,nd/2):nullptr;
        norm=ggml_rms_norm(ctx,x,eps);
        out=ggml_rope_ext(ctx,norm,positions,factors,nd,type,8192,freq_base,freq_scale,ext,attn,32.f,1.f);
        ggml_rope_set_offset(out,off);
        if(observe==1)ggml_set_output(norm);
        if(observe==2) {observer=ggml_scale(ctx,norm,.5f);ggml_set_output(observer);}
        ggml_set_output(out);graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        if(observer)ggml_build_forward_expand(graph,observer);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);if(!buffer)std::abort();
        if(alias) {out->data=x->data;out->buffer=x->buffer;}
        values.resize(ggml_nelements(base));pos.resize(nq);ff.resize(nd/2);
    }
    ~fixture(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    float at(int s,int q,int h,int c) const {
        const size_t bytes=x->view_offs+s*x->nb[3]+q*x->nb[2]+h*x->nb[1]+c*4;
        return values[bytes/4];
    }
    void upload(int iteration) {
        std::mt19937 rng(731+iteration);std::normal_distribution<float> normal(0,iteration==1?30.f:.3f);
        for(auto &v:values)v=iteration==2?0.f:iteration==3?normal(rng)*1e-4f:normal(rng);
        if(iteration==4)for(auto &v:values)v*=1e15f;
        if(iteration==5)values[x->view_offs/4]=INFINITY;
        if(iteration==6)values[x->view_offs/4]=NAN;
        const int p[]={0,7,2,262143,-5,8192,128,131071,17};
        for(int q=0;q<nq;++q)pos[q]=p[(q+iteration)%9];
        for(int i=0;i<nd/2;++i)ff[i]=.75f+.03f*(i%11);
        ggml_backend_tensor_set(base,values.data(),0,values.size()*4);
        ggml_backend_tensor_set(positions,pos.data(),0,pos.size()*4);
        if(factors)ggml_backend_tensor_set(factors,ff.data(),0,ff.size()*4);
    }
    void graph_run(int mode) {
        if(mode<0)unsetenv("GGML_HIP_QNORM_ROPE");
        else setenv("GGML_HIP_QNORM_ROPE",mode?"1":"0",1);
        if(ggml_backend_graph_compute(gpu,graph)!=GGML_STATUS_SUCCESS)std::abort();
    }
    void direct(int mode) {
        auto &c=*static_cast<ggml_backend_cuda_context *>(gpu->context);
        if(mode)ggml_cuda_op_qnorm_rope_fused(c,norm,out);
        else {ggml_cuda_op_rms_norm(c,norm);ggml_cuda_op_rope(c,out);}
    }
    std::vector<float> read(ggml_tensor *t=nullptr) {
        if(!t)t=out;ggml_backend_synchronize(gpu);std::vector<float> data(ggml_nelements(t));
        ggml_backend_tensor_get(t,data.data(),0,data.size()*4);return data;
    }
    // Independent FP64 RMS/rotation at low positions, where phase-rounding
    // error is small. High-position phase behavior is checked against GPU RoPE.
    double oracle(int s,int q,int h,int c) const {
        double sum=0;for(int i=0;i<d;++i)sum+=double(at(s,q,h,i))*at(s,q,h,i);
        const double scale=1/std::sqrt(sum/d+eps);
        if(c<off || c>=off+nd)return at(s,q,h,c)*scale;
        const int r=c-off;
        const int pair=type==GGML_ROPE_TYPE_NEOX?r%(nd/2):r/2;
        const int a=off+(type==GGML_ROPE_TYPE_NEOX?pair:2*pair);
        const int b=type==GGML_ROPE_TYPE_NEOX?a+nd/2:a+1;
        // FP32 phase construction is part of the existing backend contract.
        // Keep it explicit, then use FP64 RMS, YaRN and trigonometry independently.
        const float theta_scale=std::pow(freq_base,-2.f/nd);
        const float phase_f=(float(pos[q])*std::pow(theta_scale,float(pair)))/(factors?ff[pair]:1.f);
        double phase=phase_f;
        double theta=phase*freq_scale,mscale=attn;
        if(ext) {
            const double lo=std::max(0.,std::floor(nd*std::log(8192./(32.*2*M_PI))/(2*std::log(double(freq_base)))));
            const double hi=std::min(double(nd-1),std::ceil(nd*std::log(8192./(2*M_PI))/(2*std::log(double(freq_base)))));
            const double ramp=(1-std::min(1.,std::max(0.,(pair-lo)/std::max(.001,hi-lo))))*ext;
            theta=theta*(1-ramp)+phase*ramp;mscale*=1+.1*std::log(1./freq_scale);
        }
        const double cc=std::cos(theta)*mscale,ss=std::sin(theta)*mscale;
        return c==a?scale*at(s,q,h,a)*cc-scale*at(s,q,h,b)*ss:
                    scale*at(s,q,h,a)*ss+scale*at(s,q,h,b)*cc;
    }
};
static bool check(ggml_backend_t gpu,int mode,int heads,int queries,int streams=1,bool padded=false,int type=0,
                  int nd=64,int off=448,bool yarn=false,bool ff=false,int alias=0,int observer=0,int width=512) {
    fixture a(gpu,heads,queries,streams,padded,type,nd,off,yarn,ff,alias,observer,width);
    fixture b(gpu,heads,queries,streams,padded,type,nd,off,yarn,ff,alias,observer,width);
    if(mode<0)unsetenv("GGML_HIP_QNORM_ROPE");
    else setenv("GGML_HIP_QNORM_ROPE",mode?"1":"0",1);
    const bool eligible=width==512 && off>0 && off+nd==512 && alias!=2;
    bool ok=ggml_cuda_should_fuse_qnorm_rope(b.norm,b.out)==eligible;
    double max_diff=0,error=0,norm=0,max_oracle=0;int bit_diff=0;
    for(int iteration=0;iteration<7;++iteration) {
        a.upload(iteration);b.upload(iteration);a.graph_run(0);b.graph_run(mode);
        const auto ref=a.read(),out=b.read();
        for(size_t i=0;i<out.size();++i) {
            if(!std::isfinite(ref[i])) {ok&=std::isnan(ref[i])?std::isnan(out[i]):out[i]==ref[i];continue;}
            ok&=std::isfinite(out[i]);double e=out[i]-ref[i];max_diff=std::max(max_diff,std::abs(e));error+=e*e;norm+=double(ref[i])*ref[i];
            bit_diff+=std::memcmp(&ref[i],&out[i],4)!=0;
        }
        if(observer) {
            auto aa=a.read(observer==1?a.norm:a.observer),bb=b.read(observer==1?b.norm:b.observer);
            for(size_t i=0;i<aa.size();++i)ok&=std::isnan(aa[i])?std::isnan(bb[i]):std::abs(aa[i]-bb[i])<8e-6;
        }
        for(int s=0;s<streams;++s)for(int q=0;q<queries;++q)for(int h:{0,heads-1})for(int c=0;c<width;++c) {
            if(std::abs(b.pos[q])>128 && c>=off && c<off+nd)continue;
            const size_t index=((s*queries+q)*heads+h)*width+c;
            const double expected=b.oracle(s,q,h,c);
            if(std::isnan(expected))ok&=std::isnan(out[index]);
            else max_oracle=std::max(max_oracle,std::abs(out[index]-expected));
        }
    }
    const double nmse=error/std::max(norm,1e-30);ok&=max_diff<8e-6 && nmse<1e-12 && max_oracle<3e-5;
    std::printf("qnorm-rope mode=%d d=%d h=%d q=%d s=%d pad=%d type=%d rot=%d off=%d yarn=%d ff=%d alias=%d observer=%d max=%g nmse=%g oracle=%g bit_diff=%d %s\n",
        mode,width,heads,queries,streams,padded,type,nd,off,yarn,ff,alias,observer,max_diff,nmse,max_oracle,bit_diff,ok?"PASS":"FAIL");std::fflush(stdout);return ok;
}
static void bench(ggml_backend_t gpu) {
    auto &ctx=*static_cast<ggml_backend_cuda_context *>(gpu->context);auto stream=ctx.stream();
    for(int nq:{1,8,128,512,5120}) {
        fixture *f[]={new fixture(gpu,64,nq),new fixture(gpu,64,nq)};
        hipGraphExec_t execs[2];
        for(int mode=0;mode<2;++mode) {
            f[mode]->upload(0);f[mode]->direct(mode);ggml_backend_synchronize(gpu);hipGraph_t graph;
            CUDA_CHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
            for(int i=0;i<8;++i)f[mode]->direct(mode);
            CUDA_CHECK(hipStreamEndCapture(stream,&graph));
            CUDA_CHECK(hipGraphInstantiate(&execs[mode],graph,nullptr,nullptr,0));CUDA_CHECK(hipGraphDestroy(graph));
        }
        hipEvent_t start,end;CUDA_CHECK(hipEventCreate(&start));CUDA_CHECK(hipEventCreate(&end));
        for(int round=0;round<3;++round)for(int j=0;j<2;++j) {
            const int mode=round%2?1-j:j;
            for(int i=0;i<3;++i)CUDA_CHECK(hipGraphLaunch(execs[mode],stream));
            CUDA_CHECK(hipEventRecord(start,stream));
            for(int i=0;i<3;++i)CUDA_CHECK(hipGraphLaunch(execs[mode],stream));
            CUDA_CHECK(hipEventRecord(end,stream));CUDA_CHECK(hipEventSynchronize(end));float ms;
            CUDA_CHECK(hipEventElapsedTime(&ms,start,end));
            std::printf("BENCH heads=64 queries=%d mode=%d round=%d ms=%.8f\n",nq,mode,round,ms/24);std::fflush(stdout);
        }
        CUDA_CHECK(hipEventDestroy(start));CUDA_CHECK(hipEventDestroy(end));
        for(int i=0;i<2;++i){CUDA_CHECK(hipGraphExecDestroy(execs[i]));delete f[i];}
    }
}
int main(int argc,char **argv) {
    ggml_backend_load_all();auto gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);if(!gpu)return 2;
    if(argc>1 && !std::strcmp(argv[1],"--bench")){bench(gpu);ggml_backend_free(gpu);return 0;}
    bool ok=true;
    // A fresh fixture verifies both an absent variable and explicit enablement.
    for(int mode:{-1,1})for(int type:{GGML_ROPE_TYPE_NORMAL,GGML_ROPE_TYPE_NEOX}) {
        for(int q:{1,3,37})ok&=check(gpu,mode,64,q,1,false,type);
        ok&=check(gpu,mode,5,9,2,true,type,64,448,true,true);
        ok&=check(gpu,mode,5,9,2,true,type,128,384,true,false);
        ok&=check(gpu,mode,5,9,2,false,type,2,510,false,true);
        ok&=check(gpu,mode,5,9,2,false,type,510,2,true,false);
        ok&=check(gpu,mode,5,9,2,false,type,64,448,true,true,1);
        ok&=check(gpu,mode,5,9,2,true,type,64,448,false,false,2);
        for(int observer:{1,2})ok&=check(gpu,mode,5,9,1,false,type,64,448,true,true,0,observer);
        ok&=check(gpu,mode,5,9,1,false,type,512,0,true,false);
        ok&=check(gpu,mode,5,9,1,false,type,64,64,false,false);
        ok&=check(gpu,mode,5,9,1,false,type,64,64,true,true,0,0,128);
    }
    ggml_backend_free(gpu);return ok?0:1;
}
