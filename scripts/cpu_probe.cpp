#include "moe.h"
#include "ggml-cpu.h"
#include "ggml-backend-moe-cache.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    // Synthetic DRAM-sized working set; no GPU backend, model routing, or attention.
    const char * env_threads = std::getenv("DSV4F_CPU_THREADS");
    const int threads = env_threads ? std::max(1, std::min(128, std::atoi(env_threads))) : 11;
    const int used = argc > 1 ? std::max(1, std::min(6, std::atoi(argv[1]))) : 3;
    const bool service = !std::getenv("DSV4F_PROBE_FALLBACK");
    ggml_cpu_init();
    for (auto type : {GGML_TYPE_MXFP4, GGML_TYPE_Q4_K}) {
        constexpr int ne=4096,nf=2048,experts=64;
        auto * ctx=ggml_init({size_t(1150)*1024*1024,nullptr,false});
        auto * g=ggml_new_tensor_3d(ctx,type,ne,nf,experts);
        auto * u=ggml_new_tensor_3d(ctx,type,ne,nf,experts);
        auto * d=ggml_new_tensor_3d(ctx,type,nf,ne,experts);
        auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ne,1);
        auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,1);
        auto * w=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,used,1);
        for(auto * t : {g,u,d}) {
            std::vector<float> row(t->ne[0]);
            for(size_t i=0;i<row.size();++i)row[i]=0.08f*std::sin(float(i)*1.17f);
            std::vector<char> qr(ggml_row_size(type,t->ne[0]));
            ggml_quantize_chunk(type,row.data(),qr.data(),0,1,t->ne[0],nullptr);
            for(int e=0;e<experts;++e)for(int r=0;r<t->ne[1];++r)
                std::memcpy((char*)t->data+e*t->nb[2]+r*t->nb[1],qr.data(),qr.size());
        }
        for(int i=0;i<ne;++i)((float*)x->data)[i]=std::cos(float(i)*0.13f);
        for(int i=0;i<used;++i)((float*)w->data)[i]=1.0f/used;
        auto * out=dsv4f_moe(ctx,x,ids,w,g,u,d,10.f);
        auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        std::vector<double> times;
        for(int n=0;n<1056;++n) {
            for(int i=0;i<used;++i)((int32_t*)ids->data)[i]=(n*7+i*19)%experts;
            const auto start=std::chrono::steady_clock::now();
            if(service) ggml_dsv4f_cpu_experts(g,u,d,(float*)x->data,(int32_t*)ids->data,(float*)w->data,used,0,10.f,(float*)out->data);
            else if(ggml_graph_compute_with_ctx(ctx,graph,1)!=GGML_STATUS_SUCCESS)return 1;
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(n>=32)times.push_back(ms);
        }
        std::sort(times.begin(),times.end());
        double sum=0;for(double ms:times)sum+=ms;
        printf("path=%s type=%s weights_mib=%.2f threads=%d experts_per_call=%d calls=%zu mean_ms=%.4f median_ms=%.4f p90_ms=%.4f effective_weight_GBps=%.2f checksum=%g\n",
            service?"service":"fallback",ggml_type_name(type),double(ggml_nbytes(g)+ggml_nbytes(u)+ggml_nbytes(d))/(1<<20),threads,used,times.size(),sum/times.size(),times[times.size()/2],times[times.size()*9/10],
            double(used)*(g->nb[2]+u->nb[2]+d->nb[2])/(sum/times.size())/1e6,((float*)out->data)[0]);
        fflush(stdout);ggml_free(ctx);
    }
}
