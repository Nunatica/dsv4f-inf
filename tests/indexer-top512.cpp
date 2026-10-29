#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-dsv4f-topk.h"
#include "ggml-cuda/common.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
struct ggml_backend_cuda_context;
void ggml_cuda_op_top_k(ggml_backend_cuda_context &,ggml_tensor *);
static uint32_t key(float f) {
    uint32_t u;std::memcpy(&u,&f,4);return u ^ (uint32_t(-(int32_t)(u>>31))|0x80000000U);
}
struct fixture {
    ggml_backend_t gpu;ggml_context *ctx;ggml_backend_buffer_t buffer;
    ggml_tensor *src,*out;ggml_cgraph *graph;int n,rows,k;
    std::vector<float> data;
    fixture(ggml_backend_t g,int width,int r,int count=512,bool hint=true):gpu(g),n(width),rows(r),k(count) {
        ctx=ggml_init({4*1024*1024,nullptr,true});
        // Contiguous view with a nonzero backing offset, multiple batch dimensions.
        auto *base=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,(int64_t)n*rows+32);
        src=ggml_view_4d(ctx,base,n,1,1,rows,n*4,n*4,n*4,32*4);
        out=hint?ggml_dsv4f_indexer_top_k(ctx,src,k):ggml_top_k(ctx,src,k);
        ggml_set_output(out);graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);if(!buffer)std::abort();
        data.resize((size_t)n*rows);
    }
    ~fixture(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void upload(int iteration) {
        std::mt19937 rng(421+iteration);
        for(int r=0;r<rows;++r)for(int i=0;i<n;++i) {
            float v=float(int(rng()%10000001)-5000000)*.00001f;
            if(iteration==0) v += .00000001f*i;
            if(iteration==1) v = i%7 < 3 ? -INFINITY : float(i%13);
            if(iteration==2) v = -INFINITY;
            if(iteration==3) v = i%2 ? -0.f : 0.f;
            if(iteration==4 && i>r*31+127) v = -INFINITY;
            if(iteration==5) { // Float-bit total order, including both NaN signs/payloads.
                uint32_t bits[]={0x7fc00001,0xffc00002,0x7f800000,0xff800000,0x00000001,0x80000001};
                if(i%17<6)std::memcpy(&v,bits+i%17,4);
            }
            data[(size_t)r*n+i]=v;
        }
        ggml_backend_tensor_set(src,data.data(),0,data.size()*4);
    }
    void direct(int mode) {
        if(mode<0)unsetenv("GGML_HIP_INDEXER_TOP512");
        else setenv("GGML_HIP_INDEXER_TOP512",std::to_string(mode).c_str(),1);
        ggml_cuda_op_top_k(*static_cast<ggml_backend_cuda_context *>(gpu->context),out);
    }
    void compute(int mode) {
        if(mode<0)unsetenv("GGML_HIP_INDEXER_TOP512");
        else setenv("GGML_HIP_INDEXER_TOP512",std::to_string(mode).c_str(),1);
        if(ggml_backend_graph_compute(gpu,graph)!=GGML_STATUS_SUCCESS)std::abort();
    }
    std::vector<int> read() {
        ggml_backend_synchronize(gpu);std::vector<int> ids((size_t)k*rows);
        ggml_backend_tensor_get(out,ids.data(),0,ids.size()*4);return ids;
    }
};
static bool check(ggml_backend_t gpu,int n,int rows,int mode,int k=512,bool hint=true) {
    fixture f(gpu,n,rows,k,hint);bool ok=true;
    const bool candidate=hint && k==512 && n>1024 && n<=(mode==2?65536:8192);
    for(int iteration=0;iteration<7;++iteration) {
        f.upload(iteration);
        // Baseline is checked too; bitonic's small-width NaN order isn't radix's.
        if(iteration!=5 || n>1024) {
            f.direct(0);auto ref=f.read();
            for(int r=0;r<rows;++r) {
                std::vector<uint32_t> expected;for(int i=0;i<n;++i)expected.push_back(key(f.data[(size_t)r*n+i]));
                std::sort(expected.begin(),expected.end(),std::greater<uint32_t>());
                std::vector<uint32_t> actual;
                std::vector<int> ids;
                for(int j=0;j<k;++j) {int id=ref[r*k+j];if(id<0||id>=n){ok=false;continue;}ids.push_back(id);actual.push_back(key(f.data[(size_t)r*n+id]));}
                std::sort(ids.begin(),ids.end());ok&=std::adjacent_find(ids.begin(),ids.end())==ids.end();
                std::sort(actual.begin(),actual.end(),std::greater<uint32_t>());
                // Small bitonic compares zeros as equal, whereas radix distinguishes them.
                if(n>1024 || iteration!=3)ok&=std::equal(actual.begin(),actual.end(),expected.begin());
            }
        }
        f.compute(mode);const auto result=f.read();
        for(int r=0;r<rows;++r) {
            std::vector<int> expected(n);std::iota(expected.begin(),expected.end(),0);
            std::partial_sort(expected.begin(),expected.begin()+k,expected.end(),[&](int a,int b){
                auto ka=key(f.data[(size_t)r*n+a]),kb=key(f.data[(size_t)r*n+b]);return ka>kb || (ka==kb && a<b);});
            expected.resize(k);std::vector<int> actual(result.begin()+r*k,result.begin()+(r+1)*k);
            std::sort(actual.begin(),actual.end());ok&=std::adjacent_find(actual.begin(),actual.end())==actual.end();
            if(candidate){std::sort(expected.begin(),expected.end());ok&=actual==expected;}
            else {
                for(int id:actual)ok&=id>=0 && id<n;
                if(n>1024 || (iteration!=3 && iteration!=5)) {
                    std::vector<uint32_t> a,b;
                    for(int id:actual)if(id>=0 && id<n)a.push_back(key(f.data[(size_t)r*n+id]));
                    for(int id:expected)b.push_back(key(f.data[(size_t)r*n+id]));
                    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());ok&=a==b;
                }
            }
        }
    }
    std::printf("top512 width=%d rows=%d k=%d hint=%d mode=%d %s\n",n,rows,k,hint,mode,ok?"PASS":"FAIL");std::fflush(stdout);return ok;
}
static void bench(ggml_backend_t gpu) {
    auto & ctx=*static_cast<ggml_backend_cuda_context *>(gpu->context);
    auto stream=ctx.stream();
    for(int n:{1280,2560,4480,6144,8192,16384,32768,65536})for(int rows:{1,512,5120}) {
        if(rows==5120 && n!=4480 && n!=6144 && n!=8192)continue;
        for(int pattern:{0,4}) {
            std::vector<fixture *> fs;std::vector<hipGraphExec_t> execs;
            for(int mode=0;mode<3;++mode) {
                auto *f=new fixture(gpu,n,rows);f->upload(pattern);f->direct(mode);ggml_backend_synchronize(gpu);
                hipGraph_t graph;hipGraphExec_t exec;
                CUDA_CHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
                for(int i=0;i<16;++i)f->direct(mode);
                CUDA_CHECK(hipStreamEndCapture(stream,&graph));
                CUDA_CHECK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                CUDA_CHECK(hipGraphDestroy(graph));execs.push_back(exec);fs.push_back(f);
            }
            hipEvent_t start,end;CUDA_CHECK(hipEventCreate(&start));CUDA_CHECK(hipEventCreate(&end));
            for(int round=0;round<3;++round) for(int j=0;j<3;++j) {
                const int mode=round%2?2-j:j;
                for(int i=0;i<3;++i)CUDA_CHECK(hipGraphLaunch(execs[mode],stream));
                CUDA_CHECK(hipEventRecord(start,stream));
                for(int i=0;i<3;++i)CUDA_CHECK(hipGraphLaunch(execs[mode],stream));
                CUDA_CHECK(hipEventRecord(end,stream));CUDA_CHECK(hipEventSynchronize(end));
                float ms;CUDA_CHECK(hipEventElapsedTime(&ms,start,end));ms/=48;
                std::printf("BENCH width=%d rows=%d pattern=%d mode=%d round=%d ms=%.8f\n",n,rows,pattern,mode,round,ms);std::fflush(stdout);
            }
            CUDA_CHECK(hipEventDestroy(start));CUDA_CHECK(hipEventDestroy(end));
            for(auto exec:execs)CUDA_CHECK(hipGraphExecDestroy(exec));
            for(auto *f:fs)delete f;
        }
    }
}
int main(int argc,char **argv) {
    ggml_backend_load_all();auto gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);if(!gpu)return 2;
    if(argc>1 && !std::strcmp(argv[1],"--bench")){bench(gpu);ggml_backend_free(gpu);return 0;}
    bool ok=true;
    // -1 verifies the binary default with the variable absent, including fallback shapes.
    for(int mode:{-1,1,2}) {
        for(int n:{512,1024,1025,2048,4096,4097,4480,8192,8193,32769,65536,65537})ok&=check(gpu,n,3,mode);
        ok&=check(gpu,4480,1,mode);ok&=check(gpu,4480,37,mode);
        ok&=check(gpu,4480,3,mode,511);ok&=check(gpu,4480,3,mode,513);ok&=check(gpu,4480,3,mode,512,false);
    }
    ggml_backend_free(gpu);return ok?0:1;
}
