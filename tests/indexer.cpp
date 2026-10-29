#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

struct ggml_backend_cuda_context;
void ggml_cuda_lightning_indexer(ggml_backend_cuda_context &, ggml_tensor *);

struct fixture {
    ggml_backend_t gpu;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * qs, *ks, *ws, *ms, *q, *k, *w, *m, *out[2];
    ggml_cgraph * graph;
    int nq, nk, ns, nh, nm;
    std::vector<float> qdata, kdata, wdata;
    std::vector<ggml_fp16_t> masks;
    fixture(ggml_backend_t gpu_, ggml_type type, int queries, int keys,
            int streams=1, bool separate_masks=false, int heads=64) :
        gpu(gpu_), nq(queries), nk(keys), ns(streams), nh(heads), nm(separate_masks?streams:1) {
        ctx=ggml_init({8*1024*1024,nullptr,true});
        // Deliberately padded head/query strides and a nonzero key view offset.
        qs=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,128,nh+1,nq+1,ns);
        q=ggml_view_4d(ctx,qs,128,nh,nq,ns,qs->nb[1],qs->nb[2],qs->nb[3],0);
        ks=ggml_new_tensor_4d(ctx,type,128,2,nk,ns);
        k=ggml_view_4d(ctx,ks,128,1,nk,ns,ks->nb[1],ks->nb[2],ks->nb[3],ks->nb[1]);
        ws=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,nh+8,nq+1,1,ns);
        w=ggml_view_4d(ctx,ws,nh,nq,1,ns,ws->nb[1],ws->nb[2],ws->nb[3],0);
        ms=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,nk,nq+1,1,nm);
        m=ggml_view_4d(ctx,ms,nk,nq,1,nm,ms->nb[1],ms->nb[2],ms->nb[3],0);
        for (int mode=0;mode<2;++mode) out[mode]=ggml_lightning_indexer(ctx,q,k,w,m);
        ggml_set_output(out[1]);
        graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out[1]);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);
        if (!buffer) std::abort();
    }
    ~fixture(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void upload(int iteration, bool special=false) {
        std::mt19937 rng(731+iteration);std::normal_distribution<float> normal(0,.2f);
        qdata.resize(ggml_nelements(qs));kdata.resize(ggml_nelements(ks));wdata.resize(ggml_nelements(ws));
        for(auto &x:qdata)x=normal(rng)*(iteration==1?4.f:1.f);
        for(auto &x:kdata)x=normal(rng);
        for(auto &x:wdata)x=normal(rng)*.05f;
        ggml_backend_tensor_set(qs,qdata.data(),0,ggml_nbytes(qs));
        ggml_backend_tensor_set(ws,wdata.data(),0,ggml_nbytes(ws));
        if(ks->type==GGML_TYPE_F32)ggml_backend_tensor_set(ks,kdata.data(),0,ggml_nbytes(ks));
        else {
            std::vector<char> bytes(ggml_nbytes(ks));
            ggml_quantize_chunk(ks->type,kdata.data(),bytes.data(),0,ggml_nrows(ks),128,nullptr);
            ggml_backend_tensor_set(ks,bytes.data(),0,bytes.size());
            ggml_get_type_traits(ks->type)->to_float(bytes.data(),kdata.data(),kdata.size());
        }
        masks.assign(ggml_nelements(ms),ggml_fp32_to_fp16(-INFINITY));
        for(int s=0;s<nm;++s)for(int t=0;t<nq;++t)for(int r=0;r<nk;++r) {
            // Fully masked query/key tiles, finite biases, changing causal frontier.
            const bool visible=t>=16 && (t>=32 ? r<nk-128 :
                r<std::min(nk,513+t*7+iteration*11) && r%257<129);
            if(visible)masks[(s*(nq+1)+t)*nk+r]=ggml_fp32_to_fp16(-.001f*(r%7+s));
        }
        if(special && nq>16 && nk>130) {
            masks[16*nk+129]=ggml_fp32_to_fp16(INFINITY);
            masks[16*nk+130]=ggml_fp32_to_fp16(NAN);
        }
        ggml_backend_tensor_set(ms,masks.data(),0,ggml_nbytes(ms));
    }
    void direct(int mode) {
        setenv("GGML_HIP_INDEXER_WMMA",mode?"1":"0",1);
        ggml_cuda_lightning_indexer(*static_cast<ggml_backend_cuda_context *>(gpu->context),out[mode]);
    }
    std::vector<float> read(int mode) {
        ggml_backend_synchronize(gpu);
        std::vector<float> result(ggml_nelements(out[mode]));
        ggml_backend_tensor_get(out[mode],result.data(),0,ggml_nbytes(out[mode]));return result;
    }
    double oracle(int s,int t,int r,bool rounded) {
        const float bias=ggml_fp16_to_fp32(masks[((s%nm)*(nq+1)+t)*nk+r]);
        const float * key=kdata.data()+((s*nk+r)*2+1)*128;
        double score=0;
        for(int h=0;h<nh;++h) {
            const float * query=qdata.data()+((s*(nq+1)+t)*(nh+1)+h)*128;
            double dot=0;
            for(int d=0;d<128;++d) {
                const float qv=rounded?ggml_fp16_to_fp32(ggml_fp32_to_fp16(query[d])):query[d];
                const float kv=rounded?ggml_fp16_to_fp32(ggml_fp32_to_fp16(key[d])):key[d];
                dot+=double(qv)*kv;
            }
            score+=std::max(dot,0.)*wdata[(s*(nq+1)+t)*(nh+8)+h];
        }
        return score+bias;
    }
};

static bool check(ggml_backend_t gpu,ggml_type type,int nq,int nk,int ns=1,bool separate=false,int nh=64) {
    fixture f(gpu,type,nq,nk,ns,separate,nh);
    bool ok=true;double error=0,norm=0,quant_error=0,quant_norm=0,max_error=0;int changed=0,selected=0;
    const bool candidate=nq>=32 && nh==64 && (type==GGML_TYPE_F16 || type==GGML_TYPE_Q8_0);
    for(int iteration=0;iteration<3;++iteration) {
        f.upload(iteration,true);f.direct(0);const auto reference=f.read(0);
        setenv("GGML_HIP_INDEXER_WMMA","1",1);
        ok&=ggml_backend_graph_compute(gpu,f.graph)==GGML_STATUS_SUCCESS;
        const auto actual=f.read(1);
        if(!candidate)ok&=std::memcmp(reference.data(),actual.data(),actual.size()*sizeof(float))==0;
        for(size_t i=0;i<actual.size();++i) {
            if(std::isnan(reference[i]))ok&=std::isnan(actual[i]);
            else if(std::isinf(reference[i]))ok&=actual[i]==reference[i];
            else {ok&=std::isfinite(actual[i]);max_error=std::max(max_error,double(std::abs(actual[i]-reference[i])));}
        }
        for(int s=0;s<ns;++s)for(int t:{0,16,nq-1})for(int r=0;r<nk;++r) {
            const size_t i=(s*nq+t)*nk+r;
            if(!std::isfinite(reference[i]))continue;
            const double exact=f.oracle(s,t,r,false), rounded=f.oracle(s,t,r,candidate);
            error+=(actual[i]-exact)*(actual[i]-exact);norm+=exact*exact;
            quant_error+=(actual[i]-rounded)*(actual[i]-rounded);quant_norm+=rounded*rounded;
        }
        if(nk>512)for(int s=0;s<ns;++s)for(int t:{16,nq-1}) {
            std::vector<int> a,b;const size_t begin=(s*nq+t)*nk;
            for(int r=0;r<nk;++r)if(std::isfinite(reference[begin+r]))a.push_back(r);
            b=a;const int count=std::min<int>(512,a.size());if(!count)continue;
            auto sort=[&](auto &ids,const auto &scores){std::partial_sort(ids.begin(),ids.begin()+count,ids.end(),[&](int i,int j){return scores[begin+i]>scores[begin+j] || (scores[begin+i]==scores[begin+j] && i<j);});ids.resize(count);};
            sort(a,reference);sort(b,actual);
            const float threshold=reference[begin+a.back()];
            for(int id:b) {
                const bool common=std::find(a.begin(),a.end(),id)!=a.end();changed+=!common;selected++;
                if(!common)ok&=threshold-reference[begin+id]<=2*max_error+1e-7;
            }
        }
    }
    const double relative=std::sqrt(error/std::max(norm,1e-30));
    const double rounded_relative=std::sqrt(quant_error/std::max(quant_norm,1e-30));
    ok&=relative<5e-4 && rounded_relative<1e-5;
    std::printf("indexer type=%s q=%d keys=%d streams=%d masks=%d heads=%d rel=%g rounded_rel=%g max_diff=%g top512_changed=%d/%d %s\n",
        ggml_type_name(type),nq,nk,ns,separate?ns:1,nh,relative,rounded_relative,max_error,changed,selected,ok?"PASS":"FAIL");
    return ok;
}

static void bench(ggml_backend_t gpu) {
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_Q8_0}) {
        fixture f(gpu,type,4096,8192);f.upload(0);
        // A fully live mask isolates matrix work from the future-tile skip.
        std::fill(f.masks.begin(),f.masks.end(),ggml_fp32_to_fp16(0));
        ggml_backend_tensor_set(f.ms,f.masks.data(),0,ggml_nbytes(f.ms));
        for(int round=0;round<3;++round)for(int mode:round%2?std::vector<int>{1,0}:std::vector<int>{0,1}) {
            f.direct(mode);ggml_backend_synchronize(gpu);
            const auto begin=std::chrono::steady_clock::now();
            for(int i=0;i<5;++i)f.direct(mode);
            ggml_backend_synchronize(gpu);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/5;
            std::printf("BENCH type=%s q=4096 keys=8192 mode=%d round=%d ms=%g\n",ggml_type_name(type),mode,round,ms);std::fflush(stdout);
        }
    }
}

int main(int argc,char **argv) {
    ggml_backend_load_all();auto *gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);if(!gpu)return 2;
    if(argc>1 && !std::strcmp(argv[1],"--bench")){bench(gpu);ggml_backend_free(gpu);return 0;}
    bool ok=true;
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_Q8_0}) {
        for(int q:{31,32,33})ok&=check(gpu,type,q,137);
        ok&=check(gpu,type,35,1025,2,false);
        ok&=check(gpu,type,35,1025,2,true);
    }
    ok&=check(gpu,GGML_TYPE_F32,33,137);
    ok&=check(gpu,GGML_TYPE_Q8_0,33,137,1,false,32);
    ok&=check(gpu,GGML_TYPE_F16,64,2048);
    ok&=check(gpu,GGML_TYPE_Q8_0,257,4097);
    ggml_backend_free(gpu);return ok?0:1;
}
