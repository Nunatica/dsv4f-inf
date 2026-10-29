#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-dsv4f-kv.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <vector>

static bool check(ggml_backend_t gpu, ggml_type type, int queries, int streams, int stream0, bool sinks = true, int heads = 4, int rows = 512, bool indexed = false, int selected = 16, bool measure = false, bool resident = false) {
    constexpr int dim = 512, raw_rows = 256;
    const int slots=std::max(32,selected*2), raw_bound=measure?128:24;
    auto * dev = ggml_backend_get_device(gpu);
    const auto get_api = (ggml_dsv4f_kv_api_fn) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_dsv4f_kv_api");
    if (!get_api) return false;
    const auto * api = get_api();
    auto * host_ctx = ggml_init({1024*1024, nullptr, true});
    auto * host = ggml_new_tensor_3d(host_ctx, type, dim, rows, stream0 + streams);
    auto * host_buf = ggml_backend_alloc_ctx_tensors_from_buft(host_ctx, ggml_backend_dev_host_buffer_type(dev));
    if (!host_buf) return false;
    ggml_backend_buffer_clear(host_buf, 0);
    void * state = api->create(dev, host, slots);
    bool ok = api->bytes(state) < ggml_nbytes(host);

    auto * ctx = ggml_init({8*1024*1024, nullptr, true});
    auto * q_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, queries, streams);
    auto * q = ggml_permute(ctx, q_storage, 0, 2, 1, 3);
    auto * raw = ggml_new_tensor_4d(ctx, type, dim, raw_rows, 1, streams);
    auto * comp = ggml_new_tensor_3d(ctx, type, dim, rows, stream0 + streams);
    auto * cv = ggml_view_4d(ctx, comp, dim, rows, 1, streams, comp->nb[1], comp->nb[2], comp->nb[2], stream0*comp->nb[2]);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, raw_rows + rows, GGML_PAD(queries, 16), 1, streams);
    auto * sink = sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads) : nullptr;
    const float scale = 1/std::sqrt(float(dim));
    auto * full = ggml_concat(ctx, raw, cv, 1);
    auto * expected = ggml_flash_attn_ext(ctx, q, full, full, mask, scale, 0, 0);
    ggml_flash_attn_ext_add_sinks(expected, sink);
    ggml_prec_set_acc(expected, GGML_PREC_F32);
    auto * updates = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    auto * stream_write = ggml_dsv4f_kv_write(ctx, state, updates, ids);
    auto * rm=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,raw_rows,mask->ne[1],1,streams);
    auto * cm=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,rows,mask->ne[1],1,streams);
    auto * ts=ggml_new_tensor_4d(ctx,GGML_TYPE_I32,selected+7,queries,1,streams);
    auto * tk=ggml_view_4d(ctx,ts,std::min(512,selected+3),queries,1,streams,ts->nb[1],ts->nb[2],ts->nb[3],0);
    auto * actual = resident?ggml_dsv4f_csa_attn(ctx,q,raw,cv,rm,cm,tk,sink,raw_bound,scale):
        ggml_dsv4f_kv_attn(ctx,state,stream0,q,raw,indexed?rm:mask,sink,raw_rows+selected,scale,stream_write,
            indexed?tk:nullptr,indexed?cm:nullptr,indexed?raw_bound:0);
    auto * control=measure?ggml_dsv4f_kv_attn(ctx,state,stream0,q,raw,mask,sink,raw_rows+selected,scale,stream_write):nullptr;
    auto * control_graph=measure?ggml_new_graph(ctx):nullptr;
    if(control)ggml_build_forward_expand(control_graph,control);
    ggml_set_output(expected); ggml_set_output(actual);
    auto * reference_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(reference_graph, expected);
    auto * stream_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(stream_graph, actual);

    // Overwrite both a resident row and a previously evicted row using exactly
    // the same GPU set_rows quantizer as the full-resident reference.
    auto * reference_write = ggml_set_rows(ctx, ggml_reshape_2d(ctx, comp, dim, rows*(stream0 + streams)), updates, ids);
    auto * write_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(write_graph, reference_write);
    ggml_build_forward_expand(write_graph, stream_write);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    ok &= buffer != nullptr;
    ok &= ggml_backend_buffer_get_alloc_size(buffer, actual) > ggml_nbytes(actual);
    auto upload = [&](ggml_tensor * t, float phase) {
        std::vector<float> values(ggml_nelements(t));
        for (size_t i = 0; i < values.size(); ++i) values[i] = .3f*std::sin(i*.013f + phase);
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        if (t->type == GGML_TYPE_F32) std::memcpy(bytes.data(), values.data(), bytes.size());
        else ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        return bytes;
    };
    const auto qbytes=upload(q_storage,.1f),rawbytes=upload(raw,.3f);
    const auto sinkbytes=sink?upload(sink,.9f):std::vector<uint8_t>();
    std::vector<float> qvalues(ggml_nelements(q_storage)),sinkvalues(heads);
    std::memcpy(qvalues.data(),qbytes.data(),qbytes.size());
    if(sink)std::memcpy(sinkvalues.data(),sinkbytes.data(),sinkbytes.size());
    auto backing = upload(comp, .7f);
    ggml_backend_tensor_set(host, backing.data(), 0, backing.size());
    float maximum = 0;
    double error2 = 0, reference2 = 0;
    double oracle_error2=0,oracle_reference2=0;float oracle_max=0;
    std::vector<float> a(ggml_nelements(actual)), e(a.size());
    for (int iteration = 0; iteration < 7 && ok; ++iteration) {
        // Read first/last/old rows; distinct query selections overflow the pool
        // several times in prefill, then revisit rows on the next graph replay.
        const int base = iteration == 6 ? 0 : iteration*71;
        std::vector<ggml_fp16_t> m(ggml_nelements(mask), ggml_fp32_to_fp16(-INFINITY));
        std::vector<ggml_fp16_t> rmask(ggml_nelements(rm),ggml_fp32_to_fp16(-INFINITY));
        std::vector<ggml_fp16_t> cmask(ggml_nelements(cm),ggml_fp32_to_fp16(-.03f));
        std::vector<int32_t> tids(ggml_nelements(ts),-1);
        for (int s = 0; s < streams; ++s) for (int j = 0; j < queries; ++j) {
            if (sinks && iteration == 4 && j == 0) continue; // all masked, sink-only
            const auto off = (s*mask->ne[1] + j)*mask->ne[0];
            for (int i = 0; i < (measure?128:19); ++i) m[off + (j + i*7)%raw_rows] = ggml_fp32_to_fp16(0);
            for (int i = 0; i < selected; ++i) {
                const int row = i == 0 ? rows - 1 : (base + j*23 + s*17 + i*13)%(rows - 1);
                m[off + raw_rows + row] = ggml_fp32_to_fp16(-.1f*(i%3));
                cmask[(s*cm->ne[1]+j)*rows+row]=m[off+raw_rows+row];
                // Include masked choices, duplicates, invalid IDs and a padded
                // top-k stride. Unselected compressed rows still have finite bias.
                if(indexed && iteration==2 && i==3)m[off+raw_rows+row]=cmask[(s*cm->ne[1]+j)*rows+row]=ggml_fp32_to_fp16(-INFINITY);
                tids[(s*queries+j)*(selected+7)+(selected-1-i)]=row;
            }
            tids[(s*queries+j)*(selected+7)+selected]=rows-1;
            tids[(s*queries+j)*(selected+7)+selected+2]=rows+17;
            std::copy_n(m.data()+off,raw_rows,rmask.data()+(s*rm->ne[1]+j)*raw_rows);
        }
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
        if(indexed) {
            ggml_backend_tensor_set(rm,rmask.data(),0,ggml_nbytes(rm));
            ggml_backend_tensor_set(cm,cmask.data(),0,ggml_nbytes(cm));
            ggml_backend_tensor_set(ts,tids.data(),0,ggml_nbytes(ts));
        }
        upload(updates, iteration*.37f);
        const int64_t row_ids[] = {int64_t(stream0 + streams - 1)*rows + rows - 1,
                                  int64_t(stream0)*rows + (base + 13)%rows};
        ggml_backend_tensor_set(ids, row_ids, 0, sizeof(row_ids));
        ok &= ggml_backend_graph_compute(gpu, write_graph) == GGML_STATUS_SUCCESS;
        ok &= ggml_backend_graph_compute(gpu, reference_graph) == GGML_STATUS_SUCCESS;
        ok &= ggml_backend_graph_compute(gpu, stream_graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(expected, e.data(), 0, ggml_nbytes(expected));
        ggml_backend_tensor_get(actual, a.data(), 0, ggml_nbytes(actual));
        for (size_t i = 0; i < a.size(); ++i) {
            ok &= std::isfinite(a[i]) && std::isfinite(e[i]);
            const float d = a[i] - e[i]; maximum = std::max(maximum, std::abs(d));
            error2 += double(d)*d; reference2 += double(e[i])*e[i];
        }
        if(indexed)for(int s=0;s<streams;++s)for(int j:{0,queries/2,queries-1})for(int h:{0,heads-1}) {
            const auto off=(s*mask->ne[1]+j)*mask->ne[0];
            const float * qr=qvalues.data()+((s*queries+j)*heads+h)*dim;
            std::vector<double> scores(mask->ne[0],-INFINITY),expected_cpu(dim,0);
            std::vector<float> key(dim);
            auto load_key=[&](int r) {
                const void * p=r<raw_rows?(const void *)(rawbytes.data()+(s*raw_rows+r)*raw->nb[1]):
                    (const void *)((const char *)host->data+((stream0+s)*rows+r-raw_rows)*host->nb[1]);
                ggml_get_type_traits(type)->to_float(p,key.data(),dim);
            };
            double mx=sinks?sinkvalues[h]:-INFINITY;
            for(int r=0;r<mask->ne[0];++r)if(std::isfinite(ggml_fp16_to_fp32(m[off+r]))) {
                load_key(r);double dot=0;for(int d=0;d<dim;++d)dot+=double(qr[d])*key[d];
                scores[r]=dot*scale+ggml_fp16_to_fp32(m[off+r]);mx=std::max(mx,scores[r]);
            }
            double sum=sinks?std::exp(double(sinkvalues[h])-mx):0;
            for(int r=0;r<mask->ne[0];++r)if(std::isfinite(scores[r])) {
                load_key(r);const double p=std::exp(scores[r]-mx);sum+=p;
                for(int d=0;d<dim;++d)expected_cpu[d]+=p*key[d];
            }
            for(int d=0;d<dim;++d) {
                const double v=sum?expected_cpu[d]/sum:0,diff=a[((s*queries+j)*heads+h)*dim+d]-v;
                oracle_error2+=diff*diff;oracle_reference2+=v*v;oracle_max=std::max(oracle_max,float(std::abs(diff)));
            }
        }
        if (iteration == 5) {
            // A restored conversation changes host backing without graph writes.
            backing = upload(comp, 2.7f);
            ggml_backend_tensor_set(host, backing.data(), 0, backing.size());
            api->reset(state);
        }
    }
    const double nmse = error2/std::max(reference2, 1e-30);
    const double oracle_nmse=oracle_error2/std::max(oracle_reference2,1e-30);
    ok &= maximum < .002f && nmse < 1e-5 && oracle_max<.002f && oracle_nmse<1e-5;
    std::printf("KV stream %s indexed=%d resident=%d queries=%d streams=%d offset=%d sinks=%d pool=%d rows=%d error=%g nmse=%g oracle_max=%g oracle_nmse=%g %s\n",
                ggml_type_name(type), int(indexed), int(resident), queries, streams, stream0, int(sinks), slots, rows, maximum, nmse,oracle_max,oracle_nmse, ok ? "PASS" : "FAIL");
    if(measure && ok)for(int round=0;round<3;++round)for(int index=0;index<2;++index) {
        const int mode=round%2?1-index:index;
        auto * graph=mode?stream_graph:resident?reference_graph:control_graph;
        for(int i=0;i<3;++i)ok&=ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(gpu);const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<5;++i)ok&=ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(gpu);
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/5;
        std::printf("CSA_BENCH type=%s resident=%d queries=%d rows=%d topk=%d mode=%d round=%d ms=%g\n",ggml_type_name(type),int(resident),queries,rows,selected,mode,round,ms);std::fflush(stdout);
    }
    ggml_backend_synchronize(gpu);
    api->destroy(state);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    ggml_backend_buffer_free(host_buf); ggml_free(host_ctx);
    return ok;
}
int main(int argc,char **argv) {
    ggml_backend_load_all();
    auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) return 1;
    bool ok = true;
    if(argc>1 && !std::strcmp(argv[1],"--bench")) {
        for(auto type:{GGML_TYPE_F16,GGML_TYPE_Q8_0})for(bool resident:{false,true})ok&=check(gpu,type,512,1,0,true,64,8192,true,512,true,resident);
        ggml_backend_free(gpu);return ok?0:1;
    }
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        ok &= check(gpu, type, 1, 1, 0);
        ok &= check(gpu, type, 8, 1, 0);
        ok &= check(gpu, type, 37, 2, 1);
        ok &= check(gpu, type, 1, 1, 0, false);
        ok &= check(gpu, type, 13, 1, 0, true, 64);
        ok &= check(gpu, type, 4, 1, 0, true, 64, 4096);
        ok &= check(gpu, type, 37, 1, 0, true, 64, 4096);
        for(int n:{1,4,8,13,37})ok &= check(gpu,type,n,1,0,true,64,4096,true);
        ok &= check(gpu,type,37,2,1,true,64,4096,true);
        ok &= check(gpu,type,37,1,0,false,64,4096,true);
        for(int n:{1,4,8,13,37})ok &= check(gpu,type,n,1,0,true,64,4096,true,16,false,true);
        ok &= check(gpu,type,37,2,1,true,64,4096,true,16,false,true);
        ok &= check(gpu,type,37,1,0,false,64,4096,true,16,false,true);
    }
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
