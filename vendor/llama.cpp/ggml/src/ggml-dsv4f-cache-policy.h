#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Host-only metadata. GPU reader/exclusive ownership remains authoritative;
// callers serialize these operations with the service's loader mutex.
class ggml_dsv4f_cache_policy {
public:
    static constexpr int key_count=256*256;
    enum class mode { legacy, scored, aged };
    struct slot { int key=-1; bool ready=true; };
    struct decision { int key=-1,index=-1; };
    std::vector<slot> slots;

    void configure(mode value,uint64_t decay_requests=43*256,double replace_ratio=1.25) {
        clear();kind=value;period=std::max<uint64_t>(1,decay_requests);ratio=std::max(1.0,replace_ratio);
    }
    void clear() {
        slots.clear();legacy_counts.clear();std::fill(counts.begin(),counts.end(),0);
        std::fill(location.begin(),location.end(),-1);cursor=clock=0;
    }
    void observe(const int * keys,int n) {
        if(kind==mode::aged && ++clock%period==0)for(auto & c:counts)c>>=1;
        for(int i=0;i<n;++i) {
            assert(keys[i]>=0 && keys[i]<key_count);
            if(kind==mode::legacy)++legacy_counts[keys[i]];
            else ++counts[keys[i]];
        }
    }
    uint64_t score(int key) const {
        if(key<0)return 0;
        if(kind!=mode::legacy)return counts[key];
        const auto it=legacy_counts.find(key);return it==legacy_counts.end()?0:it->second;
    }
    bool contains(int key) const {return location[key]>=0 && slots[location[key]].ready;}
    void rebuild() {
        std::fill(location.begin(),location.end(),-1);cursor=0;
        for(size_t i=0;i<slots.size();++i)if(slots[i].key>=0) {
            assert(location[slots[i].key]<0);location[slots[i].key]=int(i);
        }
    }
    void publish(int index,int key) {
        if(slots[index].key>=0)location[slots[index].key]=-1;
        slots[index]={key,true};location[key]=index;
    }
    decision select(const int * misses,int n,const std::unordered_set<int> & pending) {
        if(slots.empty())return {};
        if(kind==mode::legacy) {
            // Keep the original first-admissible-miss and rotating victim scan
            // as an end-to-end baseline, including its linear membership scan.
            for(int i=0;i<n;++i) {
                const int key=misses[i];if(pending.count(key))continue;
                bool resident=false;for(const auto & s:slots)if(s.key==key){resident=true;break;}
                if(resident)continue;
                const int index=victim();
                if(index>=0 && (slots[index].key<0 || score(key)>score(slots[index].key)))return {key,index};
            }
            return {};
        }
        int best=-1;
        for(int i=0;i<n;++i) {
            const int key=misses[i];
            if(location[key]>=0 || pending.count(key))continue;
            if(best<0 || score(key)>score(best))best=key;
        }
        if(best<0)return {};
        const int index=victim();if(index<0)return {};
        const uint64_t cold=score(slots[index].key),hot=score(best);
        if(slots[index].key>=0 && (kind==mode::aged?((long double)hot<=(long double)cold*ratio):hot<=cold))return {};
        return {best,index};
    }

private:
    mode kind=mode::legacy;
    uint64_t period=43*256,clock=0,cursor=0;
    double ratio=1.25;
    std::unordered_map<int,uint64_t> legacy_counts;
    std::vector<uint64_t> counts=std::vector<uint64_t>(key_count,0);
    std::vector<int> location=std::vector<int>(key_count,-1);
    int victim() {
        int best=-1;uint64_t low=std::numeric_limits<uint64_t>::max();
        // Dense scores make an exact minimum inexpensive. The legacy hash-map
        // path retains its original bounded 128-slot scan for comparison.
        const size_t count=kind==mode::legacy?std::min<size_t>(128,slots.size()):slots.size();
        for(size_t i=0;i<count;++i) {
            const int index=int(cursor++%slots.size());const auto & s=slots[index];if(!s.ready)continue;
            if(s.key<0)return index;
            const auto count=score(s.key);if(count<low){low=count;best=index;}
        }
        return best;
    }
};
