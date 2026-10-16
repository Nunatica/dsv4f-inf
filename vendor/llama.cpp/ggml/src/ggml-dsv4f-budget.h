#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>

constexpr size_t ggml_dsv4f_cache_guard = 256ULL << 20;

// Saturate rather than wrapping when an allocation plus its reserve exceeds
// size_t. This is also used for allocations made after cache initialization.
inline size_t ggml_dsv4f_vram_deficit(size_t free, size_t headroom, size_t allocation) {
    if (allocation > std::numeric_limits<size_t>::max() - headroom) return std::numeric_limits<size_t>::max();
    const size_t required = allocation + headroom;
    return required > free ? required - free : 0;
}

inline size_t ggml_dsv4f_cache_remaining(size_t ceiling, size_t existing) {
    return ceiling==std::numeric_limits<size_t>::max()?ceiling:(ceiling>existing?ceiling-existing:0);
}

// Requested cache bytes are a ceiling. Current free VRAM already excludes all
// loaded models/contexts and other applications. Account for slot metadata too.
// The guard absorbs library workspaces and lazy GPU code-object loading, which
// do not pass through the engine allocator (up to ~300 MiB on the target HIP GPU).
inline size_t ggml_dsv4f_cache_slots(size_t free, size_t headroom, size_t requested,
                                   size_t stride, size_t max_slots, size_t metadata = 0,
                                   size_t guard = ggml_dsv4f_cache_guard) {
    if (!stride || stride > std::numeric_limits<size_t>::max() - metadata ||
        free <= headroom || free - headroom <= guard) return 0;
    return std::min({(free - headroom - guard) / (stride + metadata), requested / stride, max_slots});
}
