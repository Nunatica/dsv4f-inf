#pragma once
#include "ggml.h"
#include <cstddef>

// Activation quantization and routed-row metadata belong to the graph, so both
// target and MTP contexts reserve them before the expert cache consumes VRAM.
// Q8_1 is the upper bound: native FP4 uses fewer activation bytes.
struct ggml_dsv4f_mmq_workspace {
    static constexpr size_t alignment = 256;
    static constexpr size_t row_padding = 512;
    static constexpr size_t tile_padding = 32768;
    bool enabled;
    size_t size, quant = 0, quant_bytes = 0, scale = 0;
    size_t ids_src = 0, ids_dst = 0, bounds = 0;

    size_t take(size_t bytes) {
        const size_t offset = size;
        size += GGML_PAD(bytes, alignment);
        return offset;
    }
    explicit ggml_dsv4f_mmq_workspace(const ggml_tensor * dst) {
        const bool routed = dst->op == GGML_OP_MUL_MAT_ID;
        enabled = (routed || dst->op == GGML_OP_MUL_MAT) &&
                  dst->src[0] && dst->src[1] && ggml_is_quantized(dst->src[0]->type) &&
                  dst->src[1]->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
        size = ggml_nbytes(dst);
        if (!enabled) return;
        size = GGML_PAD(size, alignment);
        const auto * x = dst->src[1];
        const size_t rows = routed ? size_t(x->ne[2])*dst->src[2]->ne[0] : size_t(ggml_nrows(x));
        // block_q8_1_mmq is 144 bytes per 128 values. The largest J=128
        // tile's unconditional read and load-chunk alignment fit in 32 KiB.
        quant_bytes = rows*GGML_PAD(size_t(x->ne[0]), row_padding)*144/128 + tile_padding;
        quant = take(quant_bytes);
        if (dst->src[0]->type == GGML_TYPE_NVFP4) scale = take(rows*sizeof(float));
        if (routed) {
            ids_src = take(rows*sizeof(int32_t));
            ids_dst = take((rows + 127)*sizeof(int32_t));
            bounds = take((dst->src[0]->ne[2] + 1)*sizeof(int32_t));
        }
    }
};
