#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>

inline size_t ggml_dsv4f_cache_remaining(size_t ceiling, size_t existing) {
    return ceiling==std::numeric_limits<size_t>::max()?ceiling:(ceiling>existing?ceiling-existing:0);
}

// Requested cache bytes are a ceiling. Current free VRAM already excludes all
// loaded models/contexts and other applications. Account for slot metadata too.
inline size_t ggml_dsv4f_cache_slots(size_t free, size_t headroom, size_t requested,
                                   size_t stride, size_t max_slots, size_t metadata = 0,
                                   size_t guard = 64ULL << 20) {
    if (!stride || stride > std::numeric_limits<size_t>::max() - metadata ||
        free <= headroom || free - headroom <= guard) return 0;
    return std::min({(free - headroom - guard) / (stride + metadata), requested / stride, max_slots});
}
