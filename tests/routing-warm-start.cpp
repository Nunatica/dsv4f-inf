#include "ggml-dsv4f-warm-start.h"
#include <stdexcept>
#include <cstdio>

static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static void apply_plan(ggml_dsv4f_cache_policy & p,const std::vector<ggml_dsv4f_warm_copy> & plan) {
    for(const auto & copy:plan)p.publish(copy.slot,copy.key);
}
int main() {
    ggml_dsv4f_cache_policy p;p.configure(ggml_dsv4f_cache_policy::mode::aged);p.slots.resize(3);
    const std::vector<ggml_dsv4f_warm_candidate> counts={{511,INTPTR_MAX},{20,101},{10,100},{30,99},{40,0}};
    auto plan=ggml_dsv4f_warm_start_plan(p,counts);
    require(plan.size()==3 && plan[0].key==511 && plan[1].key==20 && plan[2].key==10,"raw counts determine top capacity, including maximum and high IDs");
    apply_plan(p,plan);
    require(ggml_dsv4f_warm_start_plan(p,counts).empty(),"selected residents need no transfers on repeated startup");
    plan=ggml_dsv4f_warm_start_plan(p,{{10,100},{20,100},{30,100},{511,99}});
    require(plan.size()==1 && plan[0].key==30 && p.slots[plan[0].slot].key==511,"ties use key order and retain other hot residents");
    apply_plan(p,plan);
    require(ggml_dsv4f_warm_start_plan(p,{{1,0},{2,-1},{-1,99},{p.key_count,99}}).empty(),"absent/invalid history leaves warmup residents untouched");
    p.clear();p.slots.resize(5);p.publish(0,1);p.publish(1,2);p.publish(2,3);p.slots[2].ready=false;p.slots[1].ready=false;
    plan=ggml_dsv4f_warm_start_plan(p,{{1,999},{2,999},{3,999},{8,80},{7,70},{6,60}},1);
    require(plan.size()==2 && plan[0].key==8 && plan[1].key==7,"reserved, leased and busy residents excluded from warm budget");
    for(const auto & copy:plan)require(copy.slot>=3,"reserved, leased and busy slots cannot be overwritten");
    apply_plan(p,plan);require(p.slots[0].key==1 && p.slots[1].key==2 && p.slots[2].key==3,"protected contents retained");
    require(ggml_dsv4f_warm_start_plan(p,{{8,80},{7,70},{6,60}},5).empty(),"fully reserved cache makes no transfers");
    p.clear();require(ggml_dsv4f_warm_start_plan(p,counts).empty(),"zero capacity makes no transfers");
    std::puts("Routing VRAM preload ranking, retention and slot protection PASS");
}
