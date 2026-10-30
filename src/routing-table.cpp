#include "routing-table.h"
#include "ggml-backend-moe-cache.h"
#include "nlohmann/json.hpp"
#include <array>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <limits>
#include <numeric>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace {
using json=nlohmann::json;
constexpr int max_experts=512;
struct row {
    // Advisory frequencies: relaxed atomics keep cache readers and observers race-free.
    std::array<std::atomic<intptr_t>,max_experts> counts;
    row(){for(auto & c:counts)c.store(0,std::memory_order_relaxed);}
};
struct table {
    std::mutex mutex,save_mutex;
    std::map<std::string,std::unique_ptr<row>> rows;
    std::atomic<bool> recording{false};
    std::string model,path;
    bool enabled=false;
    bool loaded_history=false;
    int pruning=0;
};
table & history(){static auto * t=new table;return *t;}
std::string row_name(const std::string & name) {
    for(const char * projection:{"ffn_gate_exps","ffn_up_exps","ffn_down_exps"}) {
        const auto at=name.find(projection);
        if(at!=std::string::npos)return name.substr(0,at)+"ffn_exps"+name.substr(at+std::strlen(projection));
    }
    return {};
}
void * bind(const ggml_tensor * tensor) {
    if(tensor->ne[2]<1 || tensor->ne[2]>max_experts)return nullptr;
    const auto name=row_name(tensor->name);if(name.empty())return nullptr;
    auto & t=history();std::lock_guard<std::mutex> lock(t.mutex);
    auto & p=t.rows[name];if(!p)p=std::make_unique<row>();return p.get();
}
void observe(void * handle,const int32_t * ids,const float * weights,int n) {
    if(!history().recording.load(std::memory_order_relaxed))return;
    auto & r=*(row *)handle;
    for(int i=0;i<n;++i)if(ids[i]>=0 && ids[i]<max_experts && weights[i]!=0) {
        auto & c=r.counts[ids[i]];
        auto value=c.load(std::memory_order_relaxed);
        while(value<INTPTR_MAX && !c.compare_exchange_weak(value,value+1,std::memory_order_relaxed)){}
    }
}
intptr_t count(const void * handle,int id) {
    if(id<0 || id>=max_experts)return 0;
    const auto & r=*(const row *)handle;
    return r.counts[id].load(std::memory_order_relaxed);
}
}

void dsv4f_routing_init(const std::string & model_path) {
    auto & t=history();std::lock_guard<std::mutex> saving(t.save_mutex);
    std::lock_guard<std::mutex> lock(t.mutex);
    t.recording=false;ggml_dsv4f_routing={};t.rows.clear();t.enabled=false;t.loaded_history=false;t.pruning=0;
    const char * env=std::getenv("DSV4F_ADAPTIVE_ROUTING");
    if(env && std::strcmp(env,"0") && std::strcmp(env,"1"))throw std::runtime_error("DSV4F_ADAPTIVE_ROUTING must be 0 or 1");
    const char * prune=std::getenv("DSV4F_ADAPTIVE_PRUNING");
    if(prune) {
        const std::string value=prune;
        if(value.empty() || value.size()>2 || value.find_first_not_of("0123456789")!=std::string::npos ||
           (value.size()>1 && value[0]=='0') || std::stoi(value)>50)
            throw std::runtime_error("DSV4F_ADAPTIVE_PRUNING must be an integer in 0..50");
        t.pruning=std::stoi(value);
    }
    const bool routing=env && !std::strcmp(env,"1");
    if(routing && t.pruning)throw std::runtime_error("DSV4F_ADAPTIVE_ROUTING and DSV4F_ADAPTIVE_PRUNING cannot both be enabled; set DSV4F_ADAPTIVE_ROUTING=0");
    if(!routing && !t.pruning)return;
    t.model=std::filesystem::path(model_path).stem().string();
    // Identify split models by the first GGUF shard's filename, keeping the
    // shard suffix. A different entry shard still resolves to that same name.
    t.model=std::regex_replace(t.model,std::regex("-[0-9]{5}-of-([0-9]{5})$"),"-00001-of-$1");
    if(t.model.empty())throw std::runtime_error("adaptive routing requires a model filename");
    t.path=(std::filesystem::path("models")/(t.model+".routing.json")).string();
    try {
        if(!std::filesystem::exists(t.path) && t.pruning)throw std::runtime_error("routing history file is required");
        if(std::filesystem::exists(t.path)) {
            if(std::filesystem::file_size(t.path)>64*1024*1024)throw std::runtime_error("routing file exceeds 64 MiB");
            std::ifstream input(t.path);if(!input)throw std::runtime_error("cannot open routing file");
            json data;input>>data;
            if(data.at("version")!=1 || data.at("model")!=t.model || !data.at("experts").is_object())
                throw std::runtime_error("routing schema/model mismatch");
            const auto & experts=data.at("experts");
            if(experts.size()>4096)throw std::runtime_error("too many routing rows");
            std::map<std::string,std::unique_ptr<row>> loaded;
            bool has_counts=false;
            const std::regex valid_name("blk\\.[0-9]+\\.ffn_exps(\\.[a-z0-9_]+)?\\.weight");
            for(auto it=experts.begin();it!=experts.end();++it) {
                if(!std::regex_match(it.key(),valid_name) || !it.value().is_object())throw std::runtime_error("invalid routing row");
                auto p=std::make_unique<row>();
                for(auto c=it.value().begin();c!=it.value().end();++c) {
                    const auto & id=c.key();
                    if(id.empty() || id.size()>3 || id.find_first_not_of("0123456789")!=std::string::npos ||
                       (id.size()>1 && id[0]=='0') || !c.value().is_number_integer())throw std::runtime_error("invalid expert counter");
                    const int eid=std::stoi(id);if(eid>=max_experts)throw std::runtime_error("expert ID out of range");
                    if(c.value().is_number_unsigned()?c.value().get<uint64_t>()>uint64_t(INTPTR_MAX):c.value().get<int64_t>()<0)
                        throw std::runtime_error("expert counter must be in 0..INTPTR_MAX");
                    p->counts[eid]=c.value().get<intptr_t>();
                    has_counts|=p->counts[eid]>0;
                }
                loaded.emplace(it.key(),std::move(p));
            }
            t.rows=std::move(loaded);
            t.loaded_history=has_counts;
            std::fprintf(stderr,"[dsv4f-routing] loaded %zu rows from %s\n",t.rows.size(),t.path.c_str());
        }
    } catch(const std::exception & e) {
        if(t.pruning)throw std::runtime_error("adaptive pruning cannot load "+t.path+": "+e.what());
        t.rows.clear();std::fprintf(stderr,"[dsv4f-routing] ignoring %s: %s\n",t.path.c_str(),e.what());
    }
    if(t.pruning) {
        if(!t.loaded_history)throw std::runtime_error("adaptive pruning requires nonzero routing history: "+t.path);
        std::fprintf(stderr,"[dsv4f-pruning] enabled: target %d%% of routed expert weight bytes from %s (read-only)\n",t.pruning,t.path.c_str());
        return;
    }
    t.enabled=true;ggml_dsv4f_routing={bind,observe,count};
    std::fprintf(stderr,"[dsv4f-routing] enabled: %s\n",t.path.c_str());
}
std::vector<std::vector<float>> dsv4f_pruning_masks(const std::vector<dsv4f_pruning_layer> & layers) {
    auto & t=history();std::lock_guard<std::mutex> lock(t.mutex);
    if(!t.pruning)return {};
    struct candidate {size_t slot;int id;uint64_t bytes;double share,score;};
    std::vector<candidate> candidates;
    std::vector<std::vector<float>> masks;
    std::vector<int> active;
    std::vector<double> excluded_share(layers.size());
    std::vector<uint64_t> layer_bytes(layers.size());
    std::vector<std::multiset<std::pair<uint64_t,int>>> remaining(layers.size());
    std::vector<uint64_t> min_keep(layers.size());
    std::map<int,bool> layer_ids;
    uint64_t total_bytes=0;
    for(size_t slot=0;slot<layers.size();++slot) {
        const auto & l=layers[slot];const int ne=l.n_experts;
        if(ne<1 || ne>max_experts || l.n_used<1 || l.n_used>ne || l.layer<0 || !layer_ids.emplace(l.layer,true).second)
            throw std::runtime_error("adaptive pruning: invalid/duplicate layer or expert dimensions");
        std::vector<intptr_t> counts(ne);
        std::vector<uint64_t> bytes(ne);
        std::vector<bool> seen(ne);
        double calls=0;
        for(const auto & g:l.groups) {
            const auto it=t.rows.find(row_name(g.gate_name));
            if(int(g.local.size())!=ne || !g.expert_bytes)
                throw std::runtime_error("adaptive pruning: invalid expert mapping/bytes");
            std::vector<bool> locals(ne);
            int nlocal=0;
            for(int id=0;id<ne;++id)if(g.local[id]>=0) {
                const int local=g.local[id];
                if(local>=ne || seen[id] || locals[local])throw std::runtime_error("adaptive pruning: invalid/duplicate expert ID");
                seen[id]=locals[local]=true;++nlocal;
                counts[id]=it==t.rows.end()?0:it->second->counts[local].load(std::memory_order_relaxed);bytes[id]=g.expert_bytes;
            }
            for(int i=0;i<nlocal;++i)if(!locals[i])throw std::runtime_error("adaptive pruning: noncontiguous format-local IDs");
            if(it!=t.rows.end())for(int i=nlocal;i<max_experts;++i)if(it->second->counts[i])
                throw std::runtime_error("adaptive pruning: routing counter exceeds expert format size");
        }
        for(int id=0;id<ne;++id)calls+=double(counts[id]);
        if(calls==0 || std::find(seen.begin(),seen.end(),false)!=seen.end())
            throw std::runtime_error("adaptive pruning: incomplete/empty routing history for layer "+std::to_string(l.layer));
        masks.emplace_back(ne,0.f);active.push_back(ne);
        for(int id=0;id<ne;++id) {
            if(bytes[id]>std::numeric_limits<uint64_t>::max()-total_bytes)
                throw std::runtime_error("adaptive pruning: expert byte size overflow");
            total_bytes+=bytes[id];
            remaining[slot].insert({bytes[id],id});
            const double share=double(counts[id])/calls;
            candidates.push_back({slot,id,bytes[id],share,share/double(bytes[id])});
        }
    }
    if(!total_bytes)throw std::runtime_error("adaptive pruning: no routed expert weights");
    // Greedy historical call-share cost per byte. Binary64 arithmetic and
    // layer/global-ID ties are mirrored by scripts/model_info.py.
    std::sort(candidates.begin(),candidates.end(),[&](const candidate & a,const candidate & b){
        if(a.score!=b.score)return a.score<b.score;
        if(layers[a.slot].layer!=layers[b.slot].layer)return layers[a.slot].layer<layers[b.slot].layer;
        return a.id<b.id;
    });
    const uint64_t target=total_bytes/100*t.pruning+(total_bytes%100*t.pruning+99)/100;
    // Keep enough byte capacity available to meet the target. Otherwise a
    // cheap small expert can consume the last removable slot in a layer and
    // prevent a feasible target from being reached with a larger expert.
    uint64_t retained_floor=0;
    for(size_t i=0;i<layers.size();++i) {
        auto it=remaining[i].begin();
        for(int k=0;k<layers[i].n_used;++k,++it)min_keep[i]+=it->first;
        retained_floor+=min_keep[i];
    }
    if(total_bytes-retained_floor<target)
        throw std::runtime_error("adaptive pruning cannot reach memory target while retaining top-k in every layer");
    uint64_t saved=0;
    for(const auto & c:candidates) {
        if(saved>=target)break;
        if(active[c.slot]<=layers[c.slot].n_used)continue;
        auto & choices=remaining[c.slot];choices.erase(choices.find({c.bytes,c.id}));
        uint64_t next_keep=0;auto it=choices.begin();
        for(int k=0;k<layers[c.slot].n_used;++k,++it)next_keep+=it->first;
        const uint64_t next_floor=retained_floor-min_keep[c.slot]+next_keep;
        if(total_bytes-next_floor<target){choices.insert({c.bytes,c.id});continue;}
        retained_floor=next_floor;min_keep[c.slot]=next_keep;
        masks[c.slot][c.id]=-std::numeric_limits<float>::infinity();--active[c.slot];
        saved+=c.bytes;layer_bytes[c.slot]+=c.bytes;excluded_share[c.slot]+=c.share;
    }
    if(saved<target)throw std::runtime_error("adaptive pruning cannot reach memory target while retaining top-k in every layer");
    for(size_t i=0;i<layers.size();++i) {
        const auto & l=layers[i];
        std::fprintf(stderr,"[dsv4f-pruning] layer %d: disabled %d/%d experts, active %d, top-k %d, skipped_bytes=%llu, historical_call_share=%.6f%%\n",
            l.layer,l.n_experts-active[i],l.n_experts,active[i],l.n_used,(unsigned long long)layer_bytes[i],100*excluded_share[i]);
    }
    std::fprintf(stderr,"[dsv4f-pruning] memory target: original_bytes=%llu target_bytes=%llu skipped_bytes=%llu saved=%.6f%% (whole experts)\n",
        (unsigned long long)total_bytes,(unsigned long long)target,(unsigned long long)saved,100*double(saved)/double(total_bytes));
    return masks;
}
void dsv4f_routing_start(){
    if(history().enabled && history().loaded_history && ggml_dsv4f_experts.gpu_warm_start)
        ggml_dsv4f_experts.gpu_warm_start();
    if(history().enabled && ggml_dsv4f_experts.gpu_drain)ggml_dsv4f_experts.gpu_drain();
    history().recording.store(history().enabled,std::memory_order_relaxed);
}
bool dsv4f_routing_save() {
    auto & t=history();std::lock_guard<std::mutex> saving(t.save_mutex);
    if(!t.enabled)return true;
    std::string temporary;int fd=-1;
    try {
        // GPU completion can precede the service worker's final all-hit
        // observation. Drain those records before taking a counter snapshot.
        if(ggml_dsv4f_experts.gpu_drain)ggml_dsv4f_experts.gpu_drain();
        json data={{"version",1},{"model",t.model},{"experts",json::object()}};
        {
            std::lock_guard<std::mutex> lock(t.mutex);
            for(const auto & item:t.rows) {
                auto counters=json::object();
                for(int id=0;id<max_experts;++id) {
                    const auto n=item.second->counts[id].load(std::memory_order_relaxed);
                    if(n)counters[std::to_string(id)]=n;
                }
                if(!counters.empty())data["experts"][item.first]=std::move(counters);
            }
        }
        std::filesystem::create_directories(std::filesystem::path(t.path).parent_path());
        std::string pattern=t.path+".tmp.XXXXXX";std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
        fd=mkstemp(name.data());if(fd<0)throw std::runtime_error(std::strerror(errno));temporary=name.data();
        const auto bytes=data.dump(2)+"\n";size_t at=0;
        while(at<bytes.size()) {
            const auto n=write(fd,bytes.data()+at,bytes.size()-at);
            if(n<0 && errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("cannot write routing counters");
            at+=size_t(n);
        }
        if(fsync(fd))throw std::runtime_error("cannot sync routing counters");
        const int error=close(fd);fd=-1;if(error)throw std::runtime_error("cannot close routing counters");
        if(rename(temporary.c_str(),t.path.c_str()))throw std::runtime_error(std::strerror(errno));
        return true;
    } catch(const std::exception & e) {
        if(fd>=0)close(fd);
        if(!temporary.empty())unlink(temporary.c_str());
        std::fprintf(stderr,"[dsv4f-routing] save failed for %s: %s\n",t.path.c_str(),e.what());return false;
    }
}

void dsv4f_pruning_repair_hash(std::vector<int32_t> & ids,int top_k,
    const std::string & gate_name,const std::vector<float> & mask) {
    auto & t=history();std::lock_guard<std::mutex> lock(t.mutex);
    if(top_k<1 || ids.size()%top_k)throw std::runtime_error("invalid hash routing geometry");
    const auto row=t.rows.find(row_name(gate_name));
    if(row==t.rows.end())throw std::runtime_error("missing hash routing history");
    std::vector<int32_t> active;
    for(size_t id=0;id<mask.size();++id)if(mask[id]==0)active.push_back(id);
    if(active.size()<size_t(top_k))throw std::runtime_error("hash pruning requires top-k active experts");
    std::sort(active.begin(),active.end(),[&](int a,int b){
        const auto ca=row->second->counts[a].load(std::memory_order_relaxed),cb=row->second->counts[b].load(std::memory_order_relaxed);return ca!=cb?ca>cb:a<b;
    });
    for(size_t at=0;at<ids.size();at+=top_k) {
        std::set<int32_t> used;
        for(int k=0;k<top_k;++k) {
            const int id=ids[at+k];
            if(id<0 || size_t(id)>=mask.size())throw std::runtime_error("hash expert ID out of range");
            if(mask[id]==0 && !used.insert(id).second)throw std::runtime_error("duplicate hash expert ID");
        }
        for(int k=0;k<top_k;++k)if(mask[ids[at+k]]<0) {
            const auto found=std::find_if(active.begin(),active.end(),[&](int id){return !used.count(id);});
            ids[at+k]=*found;used.insert(*found);
        }
    }
}
