#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

// Dense tile reference on identical inputs. Changing KV and masks on repeated
// executions exercises graph replay and pooled scratch lifetimes as well.
static bool check(ggml_backend_t gpu, ggml_type type, int queries, int heads,
                  int rows, int bound, bool shared, bool sinks) {
    constexpr int dim = 512;
    auto * ctx = ggml_init({8*1024*1024, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, queries, heads*64);
    auto cache = [&] {
        auto * storage = ggml_new_tensor_3d(ctx, type, dim, 2*heads, rows);
        auto * view = ggml_view_3d(ctx, storage, dim, heads, rows,
                                  storage->nb[1], storage->nb[2], storage->nb[1]);
        return std::make_pair(storage, ggml_permute(ctx, view, 0, 2, 1, 3));
    };
    const auto k = cache();
    const auto v = shared ? k : cache();
    // FLASH_ATTN_EXT's public API requires a contiguous mask.
    const int mask_stride = rows;
    auto * mask_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mask_stride, queries + 2);
    auto * mask = mask_storage;
    auto * sink = sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads*64) : nullptr;
    ggml_tensor * out[2];
    ggml_cgraph * graph[2];
    for (int path = 0; path < 2; ++path) {
        out[path] = ggml_flash_attn_ext(ctx, q, k.second, v.second, mask, 1/std::sqrt(float(dim)), 0, 0);
        ggml_flash_attn_ext_set_n_kv_max(out[path], path ? bound : 0);
        ggml_flash_attn_ext_add_sinks(out[path], sink);
        ggml_prec_set_acc(out[path], GGML_PREC_F32);
        ggml_set_output(out[path]);
        graph[path] = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph[path], out[path]);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    bool ok = buffer != nullptr;
    float maximum = 0;
    double error2 = 0, reference2 = 0, ms[2] = {};
    std::vector<float> expected(ggml_nelements(out[0])), actual(expected.size()), previous(expected.size());
    for (int iteration = 0; ok && iteration < 5; ++iteration) {
        auto upload = [&](ggml_tensor * t, float phase) {
            std::vector<float> values(ggml_nelements(t));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = .4f*std::sin(i*.013f + phase + iteration*.31f);
            }
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, values.data(), 0, bytes.size());
            } else {
                ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0,
                                    ggml_nrows(t), t->ne[0], nullptr);
                ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
            }
        };
        upload(q, .1f);
        upload(k.first, .3f);
        if (!shared) {
            upload(v.first, .7f);
        }
        if (sink) {
            upload(sink, .9f);
        }
        std::vector<ggml_fp16_t> masks(ggml_nelements(mask_storage), ggml_fp32_to_fp16(-INFINITY));
        for (int j = 0; j < queries; ++j) {
            const int count = sinks && iteration == 3 && j == 0 ? 0 : bound - (iteration + j)%17;
            for (int i = 0; i < count; ++i) {
                const int index = (i*37 + j*113 + iteration*71)%rows;
                masks[j*mask_stride + index] = ggml_fp32_to_fp16(-.1f*(i%5));
            }
        }
        ggml_backend_tensor_set(mask_storage, masks.data(), 0, ggml_nbytes(mask_storage));
        for (int path = 0; path < 2; ++path) {
            const auto start = std::chrono::steady_clock::now();
            ok &= ggml_backend_graph_compute(gpu, graph[path]) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(gpu);
            if (iteration >= 2) {
                ms[path] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()/3;
            }
        }
        ggml_backend_tensor_get(out[0], expected.data(), 0, ggml_nbytes(out[0]));
        ggml_backend_tensor_get(out[1], actual.data(), 0, ggml_nbytes(out[1]));
        float change = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            ok &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
            const float error = actual[i] - expected[i];
            maximum = std::max(maximum, std::abs(error));
            error2 += double(error)*error;
            reference2 += double(expected[i])*expected[i];
            change = std::max(change, std::abs(actual[i] - previous[i]));
        }
        if (iteration) {
            ok &= change > 1e-5f;
        }
        previous = actual;
    }
    const double nmse = error2/std::max(reference2, 1e-30);
    ok &= maximum < .002f && nmse < 1e-5;
    std::printf("sparse FA %s q=%d heads=%d kv=%d bound=%d shared=%d sinks=%d max_error=%g nmse=%g dense_ms=%.3f sparse_ms=%.3f %s\n",
        ggml_type_name(type), queries, heads, rows, bound, int(shared), int(sinks),
        maximum, nmse, ms[0], ms[1], ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) {
        std::fprintf(stderr, "GPU backend required\n");
        return 1;
    }
    bool ok = true;
    for (auto type : { GGML_TYPE_Q8_0, GGML_TYPE_F16 }) {
        ok &= check(gpu, type, 1, 1, 4096, 640, true, true);
        ok &= check(gpu, type, 2, 1, 8192, 640, true, true);
        ok &= check(gpu, type, 4, 2, 8192, 257, false, true);
        ok &= check(gpu, type, 1, 1, 65536, 640, true, false);
    }
    ok &= check(gpu, GGML_TYPE_Q8_0, 8, 1, 16384, 513, true, true);
    // Dense fallback: too many queries or insufficient savings.
    ok &= check(gpu, GGML_TYPE_Q8_0, 9, 1, 4096, 64, true, true);
    ok &= check(gpu, GGML_TYPE_Q8_0, 2, 1, 2048, 640, true, true);
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
