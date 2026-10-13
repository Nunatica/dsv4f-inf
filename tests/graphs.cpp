#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void host_join(ggml_tensor * dst, int, int, void *) {
    auto * out = static_cast<float *>(dst->data);
    for (int i = 0; i < 256; ++i) {
        out[i] = static_cast<float *>(dst->src[0]->data)[i] +
                 static_cast<float *>(dst->src[1]->data)[i] -
                 static_cast<float *>(dst->src[2]->data)[i];
    }
}

static bool check_transfers(ggml_backend_t gpu, ggml_backend_t cpu) {
    ggml_backend_t backends[] = {gpu, cpu};
    auto * host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu));
    ggml_backend_buffer_type_t bufts[] = {ggml_backend_get_default_buffer_type(gpu),
                                        host_buft ? host_buft : ggml_backend_get_default_buffer_type(cpu)};
    auto * sched = ggml_backend_sched_new(backends, bufts, 2, 256, false, true);
    ggml_context * ctx[2];
    ggml_tensor * inputs[2][3], * output[2];
    ggml_cgraph * graph[2];
    for (int k = 0; k < 2; ++k) {
        ctx[k] = ggml_init({2*1024*1024, nullptr, true});
        ggml_tensor * args[3];
        for (int j = 0; j < 3; ++j) {
            inputs[k][j] = ggml_new_tensor_1d(ctx[k], GGML_TYPE_F32, 256);
            ggml_set_input(inputs[k][j]);
            args[j] = ggml_scale(ctx[k], inputs[k][j], 0.5f*(j+1));
            if (k) ggml_backend_sched_set_tensor_backend(sched, args[j], gpu);
        }
        auto * joined = ggml_custom_4d(ctx[k], GGML_TYPE_F32, 256, 1, 1, 1, args, 3, host_join, 1, nullptr);
        if (k) ggml_backend_sched_set_tensor_backend(sched, joined, cpu);
        output[k] = ggml_silu(ctx[k], joined);
        if (k) ggml_backend_sched_set_tensor_backend(sched, output[k], gpu);
        ggml_set_output(output[k]);
        graph[k] = ggml_new_graph_custom(ctx[k], 256, false);
        ggml_build_forward_expand(graph[k], output[k]);
    }
    auto * ref_buffer = ggml_backend_alloc_ctx_tensors(ctx[0], cpu);
    bool ok = ggml_backend_sched_alloc_graph(sched, graph[1]);
    float maximum = 0;
    std::vector<float> input(256), expected(256), actual(256);
    for (int iteration = 0; ok && iteration < 16; ++iteration) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 256; ++i) input[i] = std::sin(0.23f*i + 0.71f*iteration + j);
            for (int k = 0; k < 2; ++k) ggml_backend_tensor_set(inputs[k][j], input.data(), 0, 256*sizeof(float));
        }
        ok &= ggml_backend_graph_compute(cpu, graph[0]) == GGML_STATUS_SUCCESS;
        ok &= ggml_backend_sched_graph_compute(sched, graph[1]) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(output[0], expected.data(), 0, 256*sizeof(float));
        ggml_backend_tensor_get(output[1], actual.data(), 0, 256*sizeof(float));
        for (int i = 0; i < 256; ++i) {
            maximum = std::max(maximum, std::abs(expected[i]-actual[i]));
            ok &= std::isfinite(actual[i]);
        }
    }
    ok &= maximum < 1e-5f;
    std::printf("Batched H2D/D2H across GPU/CPU/GPU splits: max_error=%g %s\n", maximum, ok ? "PASS" : "FAIL");
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(ref_buffer);
    for (auto * c : ctx) ggml_free(c);
    return ok;
}

static bool check_hc_mix(ggml_backend_t gpu, ggml_backend_t cpu, int nt, int iterations) {
    ggml_context * ctx[2];
    ggml_backend_buffer_t buffer[2];
    ggml_tensor * mixes[2], * scale[2], * base[2], * packed[2];
    ggml_cgraph * graphs[2];
    ggml_tensor * reference = nullptr;
    ggml_cgraph * ref_graph = nullptr;
    constexpr float eps=1e-6f;
    const float scales[]={0.3f,1.2f,0.7f};
    std::vector<float> values(24*nt), biases(24);
    for (int i=0;i<24*nt;++i) values[i]=8.0f*std::sin(0.37f*i);
    for (int i=0;i<24;++i) biases[i]=0.1f*std::cos(0.2f*i);
    for (int k=0;k<2;++k) {
        ctx[k]=ggml_init({2*1024*1024,nullptr,true});
        mixes[k]=ggml_new_tensor_2d(ctx[k],GGML_TYPE_F32,24,nt);
        scale[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,3);
        base[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,24);
        packed[k]=ggml_dsv4_hc_mix(ctx[k],mixes[k],scale[k],base[k],eps,iterations);
        graphs[k]=ggml_new_graph(ctx[k]);ggml_build_forward_expand(graphs[k],packed[k]);
        if (!k) {
            reference=ggml_dsv4_hc_comb(ctx[k],mixes[k],scale[k],base[k],eps,iterations);
            ref_graph=ggml_new_graph(ctx[k]);ggml_build_forward_expand(ref_graph,reference);
        }
        buffer[k]=ggml_backend_alloc_ctx_tensors(ctx[k],k?gpu:cpu);
        ggml_backend_tensor_set(mixes[k],values.data(),0,values.size()*sizeof(float));
        ggml_backend_tensor_set(scale[k],scales,0,sizeof(scales));
        ggml_backend_tensor_set(base[k],biases.data(),0,biases.size()*sizeof(float));
    }
    bool ok=ggml_backend_graph_compute(cpu,ref_graph)==GGML_STATUS_SUCCESS;
    std::vector<float> combination(16*nt), expected(24*nt), actual(24*nt);
    ggml_backend_tensor_get(reference,combination.data(),0,combination.size()*sizeof(float));
    for (int t=0;t<nt;++t) {
        for (int i=0;i<8;++i) {
            const float scaled=values[t*24+i]*scales[i/4];
            const float v=1.0f/(1.0f+std::exp(-(scaled+biases[i])));
            expected[t*24+i]=i<4?v+eps:2.0f*v;
        }
        std::copy_n(combination.data()+t*16,16,expected.data()+t*24+8);
    }
    for (int k=0;k<2;++k) {
        ok &= ggml_backend_graph_compute(k?gpu:cpu,graphs[k])==GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(packed[k],actual.data(),0,actual.size()*sizeof(float));
        float maximum=0;
        for(size_t i=0;i<actual.size();++i) {
            maximum=std::max(maximum,std::abs(actual[i]-expected[i]));
            ok &= std::isfinite(actual[i]);
        }
        ok &= maximum<3e-6f;
        std::printf("Fused HC pre/post/Sinkhorn %s tokens=%d iterations=%d max_error=%g\n",k?"GPU":"CPU",nt,iterations,maximum);
        ggml_backend_buffer_free(buffer[k]);ggml_free(ctx[k]);
    }
    return ok;
}

static bool check_hc_full(ggml_backend_t gpu,ggml_backend_t cpu,int ne,int iters) {
    ggml_context * ctx[2];ggml_backend_buffer_t buffers[2];ggml_tensor * r[2],*fn[2],*sc[2],*base[2],*norm[2],*out[2],*packed=nullptr;
    ggml_cgraph * graphs[2];constexpr float norm_eps=1e-5f,hc_eps=2e-6f;
    std::vector<float> matrix(4*ne*24),bias(24),nw(ne),input(4*ne),expected(ne+24),actual(ne+24);
    for(size_t i=0;i<matrix.size();++i)matrix[i]=.01f*std::sin(.173f*i);for(int i=0;i<24;++i)bias[i]=.1f*std::cos(.37f*i);for(int i=0;i<ne;++i)nw[i]=.8f+.2f*std::cos(.03f*i);
    const float scales[]={.3f,1.2f,.7f};
    for(int k=0;k<2;++k){ctx[k]=ggml_init({2*1024*1024,nullptr,true});r[k]=ggml_new_tensor_3d(ctx[k],GGML_TYPE_F32,ne,4,1);fn[k]=ggml_new_tensor_2d(ctx[k],GGML_TYPE_F32,4*ne,24);sc[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,3);base[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,24);norm[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,ne);
        if(!k){auto * flat=ggml_rms_norm(ctx[k],ggml_reshape_2d(ctx[k],r[k],4*ne,1),norm_eps);packed=ggml_dsv4_hc_mix(ctx[k],ggml_mul_mat(ctx[k],fn[k],flat),sc[k],base[k],hc_eps,iters);auto * pre=ggml_view_2d(ctx[k],packed,4,1,packed->nb[1],0);auto * collapsed=ggml_dsv4_hc_pre(ctx[k],r[k],pre);out[k]=ggml_mul(ctx[k],ggml_rms_norm(ctx[k],collapsed,norm_eps),norm[k]);}
        else out[k]=ggml_dsv4_hc_full(ctx[k],r[k],fn[k],sc[k],base[k],norm[k],norm_eps,hc_eps,iters);
        graphs[k]=ggml_new_graph(ctx[k]);ggml_build_forward_expand(graphs[k],out[k]);buffers[k]=ggml_backend_alloc_ctx_tensors(ctx[k],k?gpu:cpu);
        ggml_backend_tensor_set(fn[k],matrix.data(),0,matrix.size()*4);ggml_backend_tensor_set(sc[k],scales,0,sizeof(scales));ggml_backend_tensor_set(base[k],bias.data(),0,bias.size()*4);ggml_backend_tensor_set(norm[k],nw.data(),0,nw.size()*4);
    }
    bool ok=true;float maximum=0;
    for(int step=0;step<12;++step){for(int i=0;i<4*ne;++i)input[i]=std::sin(.013f*i+.37f*step)+.3f*std::cos(.23f*i);
        for(int k=0;k<2;++k){ggml_backend_tensor_set(r[k],input.data(),0,input.size()*4);ok&=ggml_backend_graph_compute(k?gpu:cpu,graphs[k])==GGML_STATUS_SUCCESS;}
        ggml_backend_tensor_get(out[0],expected.data(),0,ne*4);ggml_backend_tensor_get(packed,expected.data()+ne,0,24*4);ggml_backend_tensor_get(out[1],actual.data(),0,actual.size()*4);
        for(size_t i=0;i<actual.size();++i){maximum=std::max(maximum,std::abs(actual[i]-expected[i]));ok&=std::isfinite(actual[i]);}
    }
    ok&=maximum<5e-5f;std::printf("Full F32 HC projection/Sinkhorn/pre/norm ne=%d iters=%d max_error=%g %s\n",ne,iters,maximum,ok?"PASS":"FAIL");
    for(int k=0;k<2;++k){ggml_backend_buffer_free(buffers[k]);ggml_free(ctx[k]);}return ok;
}

// Real scheduler allocation may reuse intermediate buffers. Retain the post
// residual for a consumer AFTER the full HC operation, just as the model does.
static bool check_hc_boundary(ggml_backend_t gpu,ggml_backend_t cpu,int ne,int iters,bool use_comb) {
    ggml_backend_t backends[]={gpu,cpu};
    ggml_backend_buffer_type_t bufts[]={ggml_backend_get_default_buffer_type(gpu),ggml_backend_get_default_buffer_type(cpu)};
    auto * sched=ggml_backend_sched_new(backends,bufts,2,256,false,true);
    ggml_context * ctx[2];ggml_cgraph * graphs[2];
    ggml_tensor * r[2],*x[2],*p[2],*c[2],*fn[2],*sc[2],*bias[2],*norm[2],*joined[2],*packed[2];
    std::vector<float> matrix(4*ne*24),input((ne+16)*4),branch(ne),weight(ne),base(24),post(4),comb(16);
    for(size_t i=0;i<matrix.size();++i)matrix[i]=.01f*std::sin(.173f*i);
    for(int i=0;i<ne;++i)weight[i]=.8f+.2f*std::cos(.03f*i);
    for(int i=0;i<24;++i)base[i]=.1f*std::cos(.37f*i);
    const float scales[]={.3f,1.2f,.7f};
    for(int k=0;k<2;++k) {
        ctx[k]=ggml_init({2*1024*1024,nullptr,true});
        r[k]=ggml_new_tensor_2d(ctx[k],GGML_TYPE_F32,ne+16,4);
        x[k]=ggml_new_tensor_2d(ctx[k],GGML_TYPE_F32,ne,1);
        p[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,4);
        c[k]=ggml_new_tensor_2d(ctx[k],GGML_TYPE_F32,4,4);
        fn[k]=ggml_new_tensor_2d(ctx[k],GGML_TYPE_F32,4*ne,24);
        sc[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,3);
        bias[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,24);
        norm[k]=ggml_new_tensor_1d(ctx[k],GGML_TYPE_F32,ne);
        for(auto * t:{r[k],x[k],p[k],c[k],fn[k],sc[k],bias[k],norm[k]}) {
            ggml_set_input(t);
            if(k)ggml_backend_sched_set_tensor_backend(sched,t,gpu);
        }
        auto * rv=ggml_view_3d(ctx[k],r[k],ne,4,1,(ne+16)*sizeof(float),4*(ne+16)*sizeof(float),0);
        auto * residual=ggml_dsv4_hc_post(ctx[k],x[k],rv,p[k],use_comb?ggml_transpose(ctx[k],c[k]):nullptr);
        ggml_tensor * normalized;
        if(k) {
            packed[k]=ggml_dsv4_hc_full(ctx[k],residual,fn[k],sc[k],bias[k],norm[k],1e-5f,2e-6f,iters);
            normalized=ggml_view_2d(ctx[k],packed[k],ne,1,packed[k]->nb[1],0);
        }else {
            auto * flat=ggml_rms_norm(ctx[k],ggml_reshape_2d(ctx[k],residual,4*ne,1),1e-5f);
            packed[k]=ggml_dsv4_hc_mix(ctx[k],ggml_mul_mat(ctx[k],fn[k],flat),sc[k],bias[k],2e-6f,iters);
            auto * pre=ggml_view_2d(ctx[k],packed[k],4,1,packed[k]->nb[1],0);
            normalized=ggml_mul(ctx[k],ggml_rms_norm(ctx[k],ggml_dsv4_hc_pre(ctx[k],residual,pre),1e-5f),norm[k]);
        }
        joined[k]=ggml_add(ctx[k],residual,ggml_repeat(ctx[k],ggml_reshape_3d(ctx[k],normalized,ne,1,1),residual));
        ggml_set_output(packed[k]);ggml_set_output(joined[k]);
        graphs[k]=ggml_new_graph_custom(ctx[k],256,false);ggml_build_forward_expand(graphs[k],joined[k]);
        if(k)for(int i=0;i<ggml_graph_n_nodes(graphs[k]);++i)
            ggml_backend_sched_set_tensor_backend(sched,ggml_graph_node(graphs[k],i),gpu);
    }
    auto * ref_buffer=ggml_backend_alloc_ctx_tensors(ctx[0],cpu);
    bool ok=ggml_backend_sched_alloc_graph(sched,graphs[1]);float maximum=0;
    for(int step=0;ok && step<16;++step) {
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(.013f*i+.37f*step)+.3f*std::cos(.23f*i);
        for(int i=0;i<ne;++i)branch[i]=.3f*std::cos(.17f*i+.31f*step);
        for(int i=0;i<4;++i)post[i]=.4f+.2f*std::sin(i+.7f*step);
        for(int i=0;i<16;++i)comb[i]=.25f+.03f*std::cos(i+.2f*step);
        for(int k=0;k<2;++k) {
            ggml_backend_tensor_set(r[k],input.data(),0,input.size()*sizeof(float));
            ggml_backend_tensor_set(x[k],branch.data(),0,branch.size()*sizeof(float));
            ggml_backend_tensor_set(p[k],post.data(),0,post.size()*sizeof(float));
            if(use_comb)ggml_backend_tensor_set(c[k],comb.data(),0,comb.size()*sizeof(float));
            ggml_backend_tensor_set(fn[k],matrix.data(),0,matrix.size()*sizeof(float));
            ggml_backend_tensor_set(sc[k],scales,0,sizeof(scales));
            ggml_backend_tensor_set(bias[k],base.data(),0,base.size()*sizeof(float));
            ggml_backend_tensor_set(norm[k],weight.data(),0,weight.size()*sizeof(float));
        }
        ok &= ggml_backend_graph_compute(cpu,graphs[0])==GGML_STATUS_SUCCESS;
        ok &= ggml_backend_sched_graph_compute(sched,graphs[1])==GGML_STATUS_SUCCESS;
        for(int part=0;part<2;++part) {
            const size_t n=part?24:4*ne;
            std::vector<float> expected(n),actual(n);
            ggml_backend_tensor_get(part?packed[0]:joined[0],expected.data(),0,n*sizeof(float));
            ggml_backend_tensor_get(part?packed[1]:joined[1],actual.data(),part?ne*sizeof(float):0,n*sizeof(float));
            for(size_t i=0;i<n;++i){maximum=std::max(maximum,std::abs(expected[i]-actual[i]));ok &= std::isfinite(actual[i]);}
        }
    }
    ok &= maximum<5e-5f;
    printf("HC post -> full with scheduler reuse ne=%d iters=%d comb=%d error=%g %s\n",ne,iters,use_comb,maximum,ok?"PASS":"FAIL");
    ggml_backend_sched_free(sched);ggml_backend_buffer_free(ref_buffer);for(auto * t:ctx)ggml_free(t);return ok;
}

// Repeated input updates must remain visible after the HIP/CUDA graph is captured.
// Use F32 weights to compare graph replay against CPU without quantization drift.
struct fixture {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * x;
    ggml_tensor * y;
    ggml_cgraph * graph;
    fixture(ggml_backend_t backend) {
        ctx = ggml_init({2*1024*1024, nullptr, true});
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        ggml_set_input(x);
        auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 128);
        auto * n = ggml_rms_norm(ctx, x, 1e-5f);
        auto * p = ggml_mul_mat(ctx, w, n);
        y = ggml_swiglu_split(ctx, p, ggml_scale(ctx, p, 0.75f));
        ggml_set_output(y);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> weights(256*128);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.01f*std::sin(float(i)*0.37f);
        ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    }
    ~fixture() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
};
int main() {
    setenv("DSV4F_BATCH_TRANSFERS", "1", 0);
    ggml_backend_load_all();
    auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) { std::fprintf(stderr, "GPU and CPU backends are required\n"); return 2; }
    bool ok = true;
    {
        fixture ref(cpu), test(gpu);
        std::vector<float> input(256), expected(128), actual(128), previous(128);
        float maximum = 0;
        for (int iteration = 0; iteration < 16; ++iteration) {
            for (int i = 0; i < 256; ++i) input[i] = std::sin(float(i)*0.23f + iteration*0.71f) + 0.1f;
            for (auto * f : {&ref, &test}) ggml_backend_tensor_set(f->x, input.data(), 0, ggml_nbytes(f->x));
            ok &= ggml_backend_graph_compute(cpu, ref.graph) == GGML_STATUS_SUCCESS;
            ok &= ggml_backend_graph_compute(gpu, test.graph) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(ref.y, expected.data(), 0, ggml_nbytes(ref.y));
            ggml_backend_tensor_get(test.y, actual.data(), 0, ggml_nbytes(test.y));
            float change = 0;
            for (int i = 0; i < 128; ++i) {
                maximum = std::max(maximum, std::abs(expected[i] - actual[i]));
                change = std::max(change, std::abs(previous[i] - actual[i]));
                ok &= std::isfinite(actual[i]);
            }
            if (iteration) ok &= change > 1e-5f;
            previous = actual;
        }
        ok &= maximum < 1e-4f;
        std::printf("GPU graph replay with changing inputs: max_error=%g %s\n", maximum, ok ? "PASS" : "FAIL");
    }
    ok &= check_transfers(gpu, cpu);
    for (int nt : {1,7,128}) for (int iterations : {1,20}) ok &= check_hc_mix(gpu,cpu,nt,iterations);
    for(int ne:{256,4096})for(int it:{1,20})ok &= check_hc_full(gpu,cpu,ne,it);
    for(int ne:{256,4096})for(int it:{1,20})for(bool comb:{false,true})ok &= check_hc_boundary(gpu,cpu,ne,it,comb);
    ggml_backend_free(gpu); ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
