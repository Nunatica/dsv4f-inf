#pragma once
#include "ggml-dsv4f-cache-policy.h"
#include <cstdint>
#include <unordered_set>
#include <vector>

struct ggml_dsv4f_warm_candidate {int key;intptr_t count;};
struct ggml_dsv4f_warm_copy {int key,slot;};

// Startup only. Rank raw historical counts, retain selected residents, and use
// only replaceable slots. Reader ownership is still checked by the GPU loader.
inline std::vector<ggml_dsv4f_warm_copy> ggml_dsv4f_warm_start_plan(
    const ggml_dsv4f_cache_policy & policy,std::vector<ggml_dsv4f_warm_candidate> candidates,
    size_t reserved_slots=0) {
    std::vector<int> movable;
    std::unordered_set<int> fixed;
    for(size_t i=0;i<policy.slots.size();++i) {
        const auto & s=policy.slots[i];
        if(i>=reserved_slots && s.ready)movable.push_back(int(i));
        else if(s.key>=0)fixed.insert(s.key);
    }
    candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[&](const ggml_dsv4f_warm_candidate & c) {
        return c.key<0 || c.key>=ggml_dsv4f_cache_policy::key_count || c.count<=0 || fixed.count(c.key);
    }),candidates.end());
    std::sort(candidates.begin(),candidates.end(),[](const ggml_dsv4f_warm_candidate & a,const ggml_dsv4f_warm_candidate & b) {
        return a.count>b.count || (a.count==b.count && a.key<b.key);
    });
    if(candidates.size()>movable.size())candidates.resize(movable.size());
    std::unordered_set<int> wanted;
    for(const auto & c:candidates)wanted.insert(c.key);
    movable.erase(std::remove_if(movable.begin(),movable.end(),[&](int i){return wanted.count(policy.slots[i].key);}),movable.end());
    std::sort(movable.begin(),movable.end(),[&](int a,int b) {
        const int ka=policy.slots[a].key,kb=policy.slots[b].key;
        if((ka<0)!=(kb<0))return ka<0;
        const auto sa=policy.score(ka),sb=policy.score(kb);
        return sa<sb || (sa==sb && a<b);
    });
    std::vector<ggml_dsv4f_warm_copy> copies;
    for(const auto & c:candidates)if(!policy.contains(c.key)) {
        assert(copies.size()<movable.size());
        copies.push_back({c.key,movable[copies.size()]});
    }
    return copies;
}
