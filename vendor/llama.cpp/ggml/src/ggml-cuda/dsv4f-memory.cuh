#pragma once
#include "common.cuh"
#include "../ggml-dsv4f-budget.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

inline size_t ggml_dsv4f_env_mib(const char * name, size_t fallback) {
    const char * text = std::getenv(name);
    if (!text) return fallback;
    char * end = nullptr;
    errno = 0;
    const auto value = std::strtoull(text, &end, 10);
    if (errno || end == text || *end || *text == '-' || value > (std::numeric_limits<size_t>::max() >> 20)) {
        throw std::runtime_error(std::string("invalid MiB value for ") + name);
    }
    return size_t(value) << 20;
}

inline size_t ggml_dsv4f_free_vram() {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
#if defined(GGML_USE_HIP) && defined(__linux__)
    // HIP can overestimate free VRAM occupied by the desktop/display driver.
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    char pci[32] = {};
    if (cudaDeviceGetPCIBusId(pci, sizeof(pci), dev) == cudaSuccess) {
        const std::string base = std::string("/sys/bus/pci/devices/") + pci + "/";
        size_t physical_total = 0, used = 0;
        std::ifstream tf(base + "mem_info_vram_total"), uf(base + "mem_info_vram_used");
        if ((tf >> physical_total) && (uf >> used) && used <= physical_total) free = std::min(free, physical_total - used);
    } else {
        (void) cudaGetLastError();
    }
#endif
    return free;
}

inline size_t ggml_dsv4f_headroom() {
    return std::max<size_t>(128ULL << 20, ggml_dsv4f_env_mib("DSV4F_VRAM_HEADROOM_MIB", 1024ULL << 20));
}

inline size_t ggml_dsv4f_guard() {
    return ggml_dsv4f_env_mib("DSV4F_VRAM_GUARD_MIB", ggml_dsv4f_cache_guard);
}

inline size_t ggml_dsv4f_allocate_cache(char ** slab, size_t stride, size_t max_slots,
                                       size_t metadata, const char * tag, size_t existing_bytes = 0) {
    const size_t headroom = ggml_dsv4f_headroom();
    const size_t guard = ggml_dsv4f_guard();
    const size_t ceiling = ggml_dsv4f_env_mib("DSV4F_EXPERT_CACHE_MIB", std::numeric_limits<size_t>::max());
    const size_t requested = ggml_dsv4f_cache_remaining(ceiling,existing_bytes);
    const size_t free = ggml_dsv4f_free_vram();
    size_t slots = ggml_dsv4f_cache_slots(free, headroom, requested, stride, max_slots, metadata, guard);
    while (slots && cudaMalloc((void **) slab, slots * stride) != cudaSuccess) {
        (void) cudaGetLastError();
        // Another process may allocate between measurement and allocation.
        const size_t retry = ggml_dsv4f_cache_slots(ggml_dsv4f_free_vram(), headroom, requested, stride, max_slots, metadata, guard);
        const size_t step = std::max<size_t>(1, (64ULL << 20) / stride);
        slots = std::min(retry, slots > step ? slots - step : 0);
        std::fprintf(stderr, "[%s] cache allocation retry: %zu slots\n", tag, slots);
    }
    /*
    std::fprintf(stderr, "[%s] VRAM budget: free=%.0f MiB requested=%.0f MiB actual=%.2f MiB headroom=%.0f MiB guard=%.0f MiB slots=%zu\n",
                 tag, free / 1048576.0, (requested == std::numeric_limits<size_t>::max() ? free : requested) / 1048576.0,
                 (slots * stride) / 1048576.0, headroom / 1048576.0, guard / 1048576.0, slots);
    */
    return slots;
}
