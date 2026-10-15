#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <utility>
#include <vector>

struct ggml_backend_cuda_context;
void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context &, ggml_tensor *);

// The original tile is useful for exact skip-only parity. WMMA instead uses
// an independent double-precision oracle: the original tile also accumulates
// VKQ in FP16 and is not an exact numerical reference.
static bool check(ggml_backend_t gpu, ggml_type type, int queries, int rows,
                  int kv_heads, bool shared, bool sinks, bool dense, bool wmma) {
    constexpr int dim = 512, gqa = 64;
    const int heads = kv_heads*gqa;
    auto * ctx = ggml_init({8*1024*1024, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, queries, heads);
    auto cache = [&] {
        auto * storage = ggml_new_tensor_3d(ctx, type, dim, 2*kv_heads, rows);
        auto * view = ggml_view_3d(ctx, storage, dim, kv_heads, rows,
                                  storage->nb[1], storage->nb[2], storage->nb[1]);
        return std::make_pair(storage, ggml_permute(ctx, view, 0, 2, 1, 3));
    };
    const auto k = cache();
    const auto v = shared ? k : cache();
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, queries + 2);
    auto * sink = sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads) : nullptr;
    auto * out = ggml_flash_attn_ext(ctx, q, k.second, v.second, mask, 1/std::sqrt(float(dim)), 0, 0);
    ggml_flash_attn_ext_add_sinks(out, sink);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_set_output(out);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    bool ok = buffer != nullptr;
    double error2 = 0, reference2 = 0;
    float maximum = 0, tile_difference = 0;
    std::vector<float> actual(ggml_nelements(out)), tile(actual.size());
    uint32_t random = 731;
    for (int iteration = 0; ok && iteration < 3; ++iteration) {
        const float amplitude = iteration == 1 ? 1.5f : .2f;
        auto upload = [&](ggml_tensor * t) {
            std::vector<float> values(ggml_nelements(t));
            for (auto & x : values) {
                random = 1664525*random + 1013904223;
                x = (float(random >> 8)/16777216.f - .5f)*(2*amplitude);
            }
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
            } else {
                std::vector<uint8_t> bytes(ggml_nbytes(t));
                ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0, ggml_nrows(t), dim, nullptr);
                ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
                ggml_get_type_traits(t->type)->to_float(bytes.data(), values.data(), values.size());
            }
            return values;
        };
        const auto qdata = upload(q);
        const auto kdata = upload(k.first);
        const auto vdata = shared ? kdata : upload(v.first);
        const auto sdata = sinks ? upload(sink) : std::vector<float>();
        std::vector<ggml_fp16_t> masks(ggml_nelements(mask), ggml_fp32_to_fp16(-INFINITY));
        for (int j = 0; j < queries; ++j) {
            // Exercise a whole masked tile, plus an empty partial final tile
            // whose unused query lanes would wrap to nonempty input rows.
            if (sinks && ((iteration == 2 && j < 4) || (iteration == 1 && j == queries-1))) {
                continue;
            }
            const int count = dense ? rows : 640;
            for (int i = 0; i < count; ++i) {
                const int raw = std::min(4352, rows-1024);
                const int row = dense ? i : i < 128 ? (i + j*13 + iteration*71)%raw :
                    raw + ((i-128)*37 + j*13 + iteration*71)%(rows-raw);
                masks[size_t(j)*rows + row] = ggml_fp32_to_fp16(-.1f*(i%5));
            }
        }
        ggml_backend_tensor_set(mask, masks.data(), 0, ggml_nbytes(mask));
        ok &= ggml_backend_graph_compute(gpu, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(gpu);
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        for (float x : actual) {
            ok &= std::isfinite(x);
        }
        // Sample both ends, partial query tiles and distinct KV head groups.
        for (int j : {0, queries/2, queries-1}) {
            for (int h : {0, 7, heads-1}) {
                const int kh = h/gqa;
                const float * qr = qdata.data() + (size_t(h)*queries + j)*dim;
                std::vector<double> scores(rows, -INFINITY), expected(dim, 0);
                double mx = sinks ? sdata[h] : -INFINITY;
                for (int r = 0; r < rows; ++r) {
                    const float bias = ggml_fp16_to_fp32(masks[size_t(j)*rows + r]);
                    if (!std::isfinite(bias)) {
                        continue;
                    }
                    const float * kr = kdata.data() + (size_t(r)*2*kv_heads + 1 + kh)*dim;
                    double dot = 0;
                    for (int d = 0; d < dim; ++d) {
                        dot += double(qr[d])*kr[d];
                    }
                    scores[r] = dot/std::sqrt(double(dim)) + bias;
                    mx = std::max(mx, scores[r]);
                }
                double sum = sinks ? std::exp(double(sdata[h]) - mx) : 0;
                for (int r = 0; r < rows; ++r) {
                    if (!std::isfinite(scores[r])) {
                        continue;
                    }
                    const double weight = std::exp(scores[r] - mx);
                    const float * vr = vdata.data() + (size_t(r)*2*kv_heads + 1 + kh)*dim;
                    sum += weight;
                    for (int d = 0; d < dim; ++d) {
                        expected[d] += weight*vr[d];
                    }
                }
                for (int d = 0; d < dim; ++d) {
                    const double ref = expected[d]/sum;
                    const double error = actual[(size_t(j)*heads + h)*dim + d] - ref;
                    error2 += error*error;
                    reference2 += ref*ref;
                    maximum = std::max(maximum, float(std::abs(error)));
                }
            }
        }
        auto & cuda = *static_cast<ggml_backend_cuda_context *>(gpu->context);
        ggml_cuda_flash_attn_ext_tile(cuda, out);
        ggml_backend_synchronize(gpu);
        ggml_backend_tensor_get(out, tile.data(), 0, ggml_nbytes(out));
        for (size_t i = 0; i < tile.size(); ++i) {
            tile_difference = std::max(tile_difference, std::abs(tile[i] - actual[i]));
        }
    }
    const double nmse = error2/std::max(reference2, 1e-30);
    // Small query batches deliberately fall back to the original tile path.
    ok &= (wmma && queries >= 32) ? maximum < .002f && nmse < 1e-5 : tile_difference == 0;
    std::printf("prefill FA %s q=%d kv=%d heads=%d shared=%d sinks=%d dense=%d cpu_max=%g cpu_nmse=%g tile_diff=%g %s\n",
        ggml_type_name(type), queries, rows, kv_heads, int(shared), int(sinks), int(dense), maximum, nmse,
        tile_difference, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) {
        return 1;
    }
    const char * mode = std::getenv("GGML_HIP_PREFILL_FA");
    const bool wmma = !mode || mode[0] == '2';
    bool ok = true;
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        ok &= check(gpu, type, 33, 5376, 1, true, true, false, wmma);
        ok &= check(gpu, type, 64, 8960, 2, false, true, false, wmma);
        ok &= check(gpu, type, 32, 2048, 1, false, false, true, wmma);
    }
    ok &= check(gpu, GGML_TYPE_Q8_0, 8, 4096, 1, true, true, false, wmma);
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
