#include "ggml-dsv4f-cache-policy.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
using policy=ggml_dsv4f_cache_policy;
int main(int argc,char ** argv) {
    if(argc<5){std::cerr<<"usage: replay TRACE legacy|scored|aged DECAY_TOKENS RATIO [COPY_DELAY_ROUTES=2]\n";return 2;}
    const std::string name=argv[2];policy::mode mode;
    if(name=="legacy")mode=policy::mode::legacy;else if(name=="scored")mode=policy::mode::scored;else if(name=="aged")mode=policy::mode::aged;else return 2;
    const unsigned age=std::stoul(argv[3]),delay=argc>5?std::stoul(argv[5]):2;
    const double ratio=std::stod(argv[4]);
    policy p;p.configure(mode,43ULL*age,ratio);
    struct queued{uint64_t when;int index,key;};std::deque<queued> queue;std::unordered_set<int> pending;
    uint64_t step=0,routes=0,hits=0,observed_hits=0,copies=0;int transitions=0;
    auto complete=[&](bool all){while(!queue.empty() && (all || queue.front().when<=step)){auto j=queue.front();queue.pop_front();p.publish(j.index,j.key);pending.erase(j.key);}};
    std::ifstream input(argv[1]);if(!input){std::cerr<<"cannot open trace\n";return 2;}std::string line;
    while(std::getline(input,line)) {
        std::istringstream row(line);char tag;row>>tag;
        if(tag=='S') {
            int base,total;row>>base>>total;complete(true);
            if(total<int(p.slots.size())) {
                std::vector<int> low(base),high(p.slots.size()-base);std::iota(low.begin(),low.end(),0);std::iota(high.begin(),high.end(),base);
                auto score=[&](int i){return p.score(p.slots[i].key);};
                std::sort(low.begin(),low.end(),[&](int a,int b){return score(a)<score(b);});
                std::sort(high.begin(),high.end(),[&](int a,int b){return score(a)>score(b);});
                for(size_t i=0;i<std::min(low.size(),high.size());++i) {
                    if(p.slots[high[i]].key<0 || (p.slots[low[i]].key>=0 && score(high[i])<=score(low[i])))break;
                    p.slots[low[i]].key=p.slots[high[i]].key;
                }
                ++transitions;
            }
            const bool shrinking=total<int(p.slots.size());
            p.slots.resize(total);if(shrinking)p.rebuild();
        }else if(tag=='R') {
            ++step;complete(false);uint64_t mask;row>>mask;std::vector<int> keys,misses;int key;
            while(row>>key)keys.push_back(key);
            for(size_t i=0;i<keys.size();++i){++routes;observed_hits+=(mask>>i)&1;if(p.contains(keys[i]))++hits;else misses.push_back(keys[i]);}
            p.observe(keys.data(),int(keys.size()));
            if(queue.size()>=8)continue;
            auto chosen=p.select(misses.data(),int(misses.size()),pending);if(chosen.index<0)continue;
            pending.insert(chosen.key);p.slots[chosen.index].ready=false;queue.push_back({step+delay,chosen.index,chosen.key});++copies;
        }
    }
    std::cout<<"{\"policy\":\""<<name<<"\",\"decay_tokens\":"<<age<<",\"ratio\":"<<ratio<<",\"copy_delay_routes\":"<<delay
             <<",\"routes\":"<<routes<<",\"hits\":"<<hits<<",\"hit_percent\":"<<(routes?100.*hits/routes:0)
             <<",\"observed_hit_percent\":"<<(routes?100.*observed_hits/routes:0)<<",\"admissions\":"<<copies<<",\"shrink_transitions\":"<<transitions<<"}\n";
}
