#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

// Model a cache that has consumed all unreserved VRAM. This reproduces the
// missing-workspace failure independently of desktop/display allocations.
struct unavailable_pool : ggml_cuda_pool {
    void * alloc(size_t, size_t *) override { throw std::runtime_error("unreserved MMQ allocation"); }
    void free(void *, size_t) override { std::abort(); }
};

static float value(uint32_t i, uint32_t seed) {
    i += seed; i ^= i >> 16; i *= 0x7feb352d; i ^= i >> 15; i *= 0x846ca68b; i ^= i >> 16;
    return float(i & 0xffffff)/8388608.0f - 1;
}

static bool check(ggml_backend_t gpu, ggml_backend_t cpu, int width, int tokens,
                  bool routed, bool broadcast, bool batched = false) {
    constexpr int outputs = 32, experts = 7, used = 3;
    const int channels = batched ? 2 : 1, samples = batched ? 2 : 1;
    ggml_context * contexts[2];
    ggml_backend_buffer_t buffers[2];
    ggml_tensor * weights[2], * input[2], * ids[2] = {}, * result[2];
    ggml_cgraph * graphs[2];
    for (int path = 0; path < 2; ++path) {
        auto * ctx = contexts[path] = ggml_init({2*1024*1024, nullptr, true});
        weights[path] = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, width, outputs, routed ? experts : channels, 1);
        if (routed) {
            input[path] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, broadcast ? 1 : used, tokens);
            ids[path] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
            result[path] = ggml_mul_mat_id(ctx, weights[path], input[path], ids[path]);
        } else {
            input[path] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, tokens, channels, samples);
            result[path] = ggml_mul_mat(ctx, weights[path], input[path]);
        }
        ggml_set_output(result[path]);
        graphs[path] = ggml_new_graph(ctx);
        ggml_build_forward_expand(graphs[path], result[path]);
        buffers[path] = ggml_backend_alloc_ctx_tensors(ctx, path ? gpu : cpu);
        if (!buffers[path]) return false;
    }
    std::vector<float> w(ggml_nelements(weights[0]));
    for (size_t i = 0; i < w.size(); ++i) w[i] = .1f*value(uint32_t(i), 731);
    std::vector<uint8_t> bytes(ggml_nbytes(weights[0]));
    ggml_quantize_chunk(weights[0]->type, w.data(), bytes.data(), 0, ggml_nrows(weights[0]), width, nullptr);
    for (int path = 0; path < 2; ++path) ggml_backend_tensor_set(weights[path], bytes.data(), 0, bytes.size());
    bool ok = ggml_backend_buffer_get_alloc_size(buffers[1], result[1]) > ggml_nbytes(result[1]);
    std::vector<float> x(ggml_nelements(input[0])), reference(ggml_nelements(result[0])), actual(reference.size());
    double worst = 0;
    for (int iteration = 0; iteration < 5; ++iteration) {
        for (size_t i = 0; i < x.size(); ++i) x[i] = value(uint32_t(i), 123456 + iteration*317);
        std::vector<int32_t> selected(used*tokens);
        for (int t = 0; t < tokens; ++t) for (int j = 0; j < used; ++j) selected[t*used+j] = (t+j+iteration)%experts;
        for (int path = 0; path < 2; ++path) {
            ggml_backend_tensor_set(input[path], x.data(), 0, x.size()*sizeof(float));
            if (routed) ggml_backend_tensor_set(ids[path], selected.data(), 0, selected.size()*sizeof(int32_t));
        }
        ok &= ggml_backend_graph_compute(cpu, graphs[0]) == GGML_STATUS_SUCCESS;
        ok &= ggml_backend_graph_compute(gpu, graphs[1]) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(result[0], reference.data(), 0, ggml_nbytes(result[0]));
        ggml_backend_tensor_get(result[1], actual.data(), 0, ggml_nbytes(result[1]));
        double error = 0, norm = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            ok &= std::isfinite(actual[i]);
            const double delta = actual[i] - reference[i];
            error += delta*delta; norm += double(reference[i])*reference[i];
        }
        worst = std::max(worst, error/std::max(norm, 1e-30));
    }
    ok &= worst < .0003;
    std::printf("MMQ reserved workspace width=%d tokens=%d routed=%d broadcast=%d batched=%d nmse=%g %s\n",
        width, tokens, int(routed), int(broadcast), int(batched), worst, ok ? "PASS" : "FAIL");
    for (int path = 0; path < 2; ++path) { ggml_backend_buffer_free(buffers[path]); ggml_free(contexts[path]); }
    return ok;
}
int main() {
    ggml_backend_load_all();
    auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) return 2;
    auto * context = (ggml_backend_cuda_context *) gpu->context;
    for (auto & device : context->pools) for (auto & pool : device) pool = std::make_unique<unavailable_pool>();
    bool ok = true;
    try {
        ok &= check(gpu, cpu, 512, 33, false, false);
        ok &= check(gpu, cpu, 544, 127, false, false);
        ok &= check(gpu, cpu, 8192, 4096, false, false);
        ok &= check(gpu, cpu, 512, 33, true, true);
        ok &= check(gpu, cpu, 512, 33, true, false);
        ok &= check(gpu, cpu, 512, 17, false, false, true);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "MMQ workspace FAIL: %s\n", e.what()); ok = false;
    }
    ggml_backend_free(gpu); ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
