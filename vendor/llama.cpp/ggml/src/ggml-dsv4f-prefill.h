#pragma once
#include "ggml.h"
#include <algorithm>
#include <cstddef>
#include <cstdlib>

// Shared by graph construction and the GPU implementation. Scratch is a graph
// operand so worst-case scheduler reservation accounts for it before the decode
// cache is sized. No unbudgeted device allocations are made by the copy stream.
struct ggml_dsv4f_prefill_layout {
    static constexpr int ring = 2;
    size_t size = 0, stage, gate, up, down, xq, hq, inverse, identity, bounds;
    size_t stage_stride, gbytes, ubytes, dbytes;
    size_t take(size_t bytes) {
        const size_t off = size;
        size += (bytes + 255) / 256 * 256;
        return off;
    }
    static size_t qbytes(int64_t rows, int64_t cols) {
        // MMQ Q8_1: 144 bytes / 128 elements, plus a full maximum-width tile.
        return rows * ((cols + 511) / 512 * 512) * 144 / 128 + 128 * 144;
    }
    ggml_dsv4f_prefill_layout(const ggml_tensor * g, const ggml_tensor * u,
                            const ggml_tensor * d, int64_t rows, int group) {
        gbytes = g->nb[2]; ubytes = u->nb[2]; dbytes = d->nb[2];
        stage_stride = group * (gbytes + ubytes + dbytes) + 512;
        stage_stride = (stage_stride + 255) / 256 * 256;
        stage = take(ring * stage_stride);
        gate = take(rows * g->ne[1] * sizeof(float));
        up = take(rows * u->ne[1] * sizeof(float));
        down = take(rows * d->ne[1] * sizeof(float));
        xq = take(qbytes(rows, g->ne[0]));
        hq = take(qbytes(rows, d->ne[0]));
        inverse = take(rows * sizeof(int32_t));
        identity = take((rows + 128) * sizeof(int32_t));
        bounds = take((g->ne[2] + 1) * sizeof(int32_t));
    }
};

inline int ggml_dsv4f_prefill_group() {
    const char * value = std::getenv("DSV4F_PREFILL_GROUP");
    return value ? std::max(1, std::min(256, std::atoi(value))) : 16;
}
