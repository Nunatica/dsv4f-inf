#pragma once
#include "ggml-backend-moe-cache.h"
#include <cstdint>

// floor(log2(1 + count)), at most 63 on x86-64. A bounded historical prior
// preserves adaptation in the existing recency/decaying-frequency policies.
inline uint64_t ggml_dsv4f_routing_bonus(intptr_t count) {
    if(count<=0)return 0;
    // Add in unsigned arithmetic so INTPTR_MAX + 1 cannot overflow.
    uint64_t value=uint64_t(count)+1;
#if defined(__GNUC__) || defined(__clang__)
    // value is nonzero. A leading-zero count computes the same integer log
    // without a shift loop in every RAM/VRAM eviction candidate's score.
    return 63-__builtin_clzll(value);
#else
    uint64_t bonus=0;
    for(;value>1;value>>=1)++bonus;
    return bonus;
#endif
}
inline intptr_t ggml_dsv4f_routing_count(const void * row,int expert) {
    return row && ggml_dsv4f_routing.count?ggml_dsv4f_routing.count(row,expert):0;
}
inline void * ggml_dsv4f_routing_bind(const ggml_tensor * tensor) {
    return ggml_dsv4f_routing.bind?ggml_dsv4f_routing.bind(tensor):nullptr;
}
inline void ggml_dsv4f_routing_observe(void * row,const int32_t * ids,const float * weights,int n) {
    if(row && ggml_dsv4f_routing.observe)ggml_dsv4f_routing.observe(row,ids,weights,n);
}
