#pragma once

#include <atomic>
#include <cstdio>
#include <cstring>

#if defined(GGML_USE_HIP)

// 0: original tile; 1: skip fully masked tile chunks; 2 (default): RDNA3 WMMA
// prefill. Decode retains its selected-KV gather and existing dispatch.
static bool ggml_hip_flash_attn_ext_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const int mode = [] {
        const char * value = std::getenv("GGML_HIP_PREFILL_FA");
        return !value ? 2 : std::strcmp(value, "1") == 0 ? 1 : std::strcmp(value, "2") == 0 ? 2 : 0;
    }();
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * m = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!mode || !GGML_CUDA_CC_IS_RDNA3_0(cc) || q->ne[0] != 512 || k->ne[0] != 512 || v->ne[0] != 512 ||
        q->ne[1] < 32 || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        k->ne[1] < 2048 || k->ne[1] % FATTN_KQ_STRIDE != 0 ||
        k->ne[2] != v->ne[2] || k->ne[1] != v->ne[1] || q->ne[2] % (8*k->ne[2]) != 0 ||
        !m || m->type != GGML_TYPE_F16 || m->nb[0] != sizeof(half) ||
        m->ne[0] != k->ne[1] || m->ne[1] < q->ne[1] || m->ne[2] != 1 || m->ne[3] != 1) {
        return false;
    }
    for (const auto * t : { k, v }) {
        if ((t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_Q8_0) ||
            t->nb[0] != ggml_type_size(t->type)) {
            return false;
        }
    }
    if (mode == 2) {
        // Match the alignment requirement of the ordinary MMA dispatcher.
        // Quantized KV is converted to a contiguous FP16 buffer by launch_fattn.
        for (const auto * t : { q, k, v, m }) {
            if (!ggml_is_quantized(t->type)) {
                for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                    if (t->nb[i] % 16 != 0) {
                        return false;
                    }
                }
            }
        }
    }
    float max_bias, softcap;
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    static std::atomic<bool> reported{false};
    const char * debug = std::getenv("GGML_HIP_PREFILL_FA_DEBUG");
    if (debug && std::strcmp(debug, "0") != 0 && !reported.exchange(true)) {
        std::fprintf(stderr, "[hip-prefill-fa] mode=%d queries=%lld kv=%lld types=%s/%s\n",
            mode, (long long) q->ne[1], (long long) k->ne[1], ggml_type_name(k->type), ggml_type_name(v->type));
    }
    if (mode == 2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 4, 8>(ctx, dst);
    } else {
        constexpr int ncols1 = 4, ncols2 = 8;
        const int nwarps = ggml_cuda_fattn_tile_get_nthreads(512, 512, 32, cc)/32;
        const int nbatch = ggml_cuda_fattn_tile_get_nbatch_fa(512, 512, 32, cc);
        fattn_kernel_t kernel = flash_attn_tile<512, 512, ncols1, ncols2, false, true>;
        launch_fattn<512, ncols1, ncols2>(ctx, dst, kernel, nwarps, 0, nbatch, true, true, false, false, 32);
    }
    return true;
}

#endif
