#include "ggml-dsv4f-cache-policy.h"
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
using policy=ggml_dsv4f_cache_policy;
static void require(bool value,const char * message){if(!value){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}}
int main() {
    std::unordered_set<int> pending;
    for(auto kind:{policy::mode::legacy,policy::mode::scored,policy::mode::aged}) {
        policy p;p.configure(kind,8,2);p.slots.resize(2);
        int keys[]={0,1};p.observe(keys,2);
        auto a=p.select(keys,2,pending);require(a.index>=0,"empty-slot admission");
        p.slots[a.index].ready=false;pending.insert(a.key);
        auto b=p.select(keys,2,pending);require(b.index>=0 && b.index!=a.index && b.key!=a.key,"exclude pending key and loader-owned slot");
        p.publish(a.index,a.key);pending.clear();p.publish(b.index,b.key);
        require(p.select(keys,2,pending).index<0,"resident exclusion");
        p.slots[0].ready=p.slots[1].ready=false;int other=2;p.observe(&other,1);
        require(p.select(&other,1,pending).index<0,"all slots busy");
        p.slots[0].ready=p.slots[1].ready=true;
        // Simulate phase compaction moving an expert from the extension.
        p.slots.resize(3);p.publish(2,other);p.slots[0].key=other;p.slots.resize(2);p.rebuild();
        require(p.select(&other,1,pending).index<0,"rebuild membership after D2D compaction");
        p.clear();require(p.slots.empty() && p.score(other)==0,"reset state");
        require(p.select(keys,2,pending).index<0,"zero cache");
    }
    policy scored;scored.configure(policy::mode::scored);scored.slots.resize(1);
    int low=4,high=5;scored.observe(&low,1);for(int i=0;i<5;++i)scored.observe(&high,1);
    int candidates[]={low,high};require(scored.select(candidates,2,pending).key==high,"highest-value miss takes priority");
    policy aged,legacy;aged.configure(policy::mode::aged,8,2);legacy.configure(policy::mode::legacy);
    for(auto * p:{&aged,&legacy}) {
        p->slots.resize(1);int old=0;p->publish(0,old);
        for(int i=0;i<64;++i)p->observe(&old,1);
    }
    int fresh=1;aged.observe(&fresh,1);
    require(aged.select(&fresh,1,pending).index<0,"hysteresis rejects low-value replacement");
    bool adapted=false;
    for(int i=0;i<16;++i) {
        aged.observe(&fresh,1);legacy.observe(&fresh,1);
        adapted|=aged.select(&fresh,1,pending).index==0;
    }
    require(adapted,"aged counts adapt to changed hot set");
    require(legacy.select(&fresh,1,pending).index<0,"cumulative baseline retains stale hot set");
    policy threshold;threshold.configure(policy::mode::aged,1000,2);threshold.slots.resize(1);int cold=0;threshold.observe(&cold,1);threshold.publish(0,cold);
    threshold.observe(&fresh,1);threshold.observe(&fresh,1);
    require(threshold.select(&fresh,1,pending).index<0,"strict replacement ratio boundary");
    threshold.observe(&fresh,1);require(threshold.select(&fresh,1,pending).index==0,"replacement ratio exceeded");
    policy exact;exact.configure(policy::mode::scored);exact.slots.resize(256);
    for(int k=0;k<256;++k){exact.publish(k,k);if(k!=200)exact.observe(&k,1);}
    int incoming=256;exact.observe(&incoming,1);
    require(exact.select(&incoming,1,pending).index==200,"exact minimum beyond first 128 slots");
    exact.slots[200].ready=false;require(exact.select(&incoming,1,pending).index<0,"global minimum remains loader protected");
    policy fractional;fractional.configure(policy::mode::aged,1000,1.25);fractional.slots.resize(1);fractional.publish(0,cold);
    for(int i=0;i<4;++i)fractional.observe(&cold,1);for(int i=0;i<5;++i)fractional.observe(&fresh,1);
    require(fractional.select(&fresh,1,pending).index<0,"fractional margin strict boundary");
    fractional.observe(&fresh,1);require(fractional.select(&fresh,1,pending).index==0,"fractional margin exceeded");
    std::puts("Cache policy admission, pending ownership, compaction, hysteresis and hot-set adaptation PASS");
}
