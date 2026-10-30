#include "routing-table.h"
#include "moe.h"
#include "ggml-backend.h"
#include "ggml-dsv4f-routing.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using json=nlohmann::json;
static void require(bool v,const char * why){if(!v)throw std::runtime_error(why);}
template<class F> static void rejects(F f){bool failed=false;try{f();}catch(const std::exception &){failed=true;}require(failed,"expected rejection");}
static const char * path="models/Pruning.routing.json";
static void write(const json & j){std::ofstream f(path);f<<j.dump();}
static const std::vector<dsv4f_pruning_group> groups={
    {"blk.0.ffn_gate_exps.q3_k.weight",{0,-1,1,-1,2,-1,3,-1},10},
    {"blk.0.ffn_gate_exps.ptq1_0.weight",{-1,0,-1,1,-1,2,-1,3},10}};
static void init(){dsv4f_routing_init("Pruning.gguf");}
static std::vector<float> mask(){auto m=dsv4f_pruning_masks({{0,8,3,groups}});return m.empty()?std::vector<float>{}:m.front();}

static void router(ggml_backend_t backend,const std::vector<float> & mask,int nt,bool fused) {
    auto * ctx=ggml_init({2<<20,nullptr,true});require(ctx,"context");
    constexpr int ne=8,nk=3;constexpr float scale=1.7f;
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ne,nt);
    auto * b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,ne);
    ggml_tensor * ids,*weights;
    if(fused)dsv4f_router(ctx,x,b,nk,true,scale,&ids,&weights);
    else {
        auto * s=ggml_sqrt(ctx,ggml_softplus(ctx,x));
        ids=ggml_cont(ctx,ggml_top_k(ctx,ggml_add(ctx,s,b),nk));
        auto * p=ggml_reshape_2d(ctx,ggml_get_rows(ctx,ggml_reshape_3d(ctx,s,1,ne,nt),ids),nk,nt);
        weights=ggml_scale(ctx,ggml_div(ctx,p,ggml_clamp(ctx,ggml_sum_rows(ctx,p),6.103515625e-5f,INFINITY)),scale);
    }
    auto * graph=ggml_new_graph_custom(ctx,1024,false);
    ggml_build_forward_expand(graph,ids);ggml_build_forward_expand(graph,weights);
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"allocation");
    std::vector<float> logits(ne*nt),bias(ne),actual(nk*nt),scores(ne);
    // At 50%, the first four ranks are pruned: top-1, 2, 3, 4 must be skipped.
    const float values[]={100,4,3,80,90,2,1,70};
    for(int i=0;i<ne;++i)bias[i]=mask[i]!=0?mask[i]:i*.03f;
    for(int t=0;t<nt;++t)for(int i=0;i<ne;++i)logits[t*ne+i]=values[i]+t*.125f;
    ggml_backend_tensor_set(x,logits.data(),0,logits.size()*4);ggml_backend_tensor_set(b,bias.data(),0,bias.size()*4);
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");
    std::vector<int32_t> chosen(nk*nt);ggml_backend_tensor_get(ids,chosen.data(),0,chosen.size()*4);
    ggml_backend_tensor_get(weights,actual.data(),0,actual.size()*4);
    for(int t=0;t<nt;++t) {
        std::vector<int> order;
        for(int i=0;i<ne;++i){double v=logits[t*ne+i];scores[i]=std::sqrt(v>20?v:std::log1p(std::exp(v)));if(mask[i]==0)order.push_back(i);}
        std::sort(order.begin(),order.end(),[&](int a,int c){return scores[a]+bias[a]>scores[c]+bias[c];});
        float sum=0;for(int k=0;k<nk;++k)sum+=scores[order[k]];
        std::vector<int> selected(chosen.begin()+t*nk,chosen.begin()+(t+1)*nk),expected(order.begin(),order.begin()+nk);
        std::sort(selected.begin(),selected.end());std::sort(expected.begin(),expected.end());
        require(selected==expected,"skip pruned ranks and replenish active top-k");
        for(int k=0;k<nk;++k) {
            require(std::abs(actual[t*nk+k]-scale*scores[chosen[t*nk+k]]/sum)<2e-6f,"renormalize remaining weights");
        }
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}
int main(int argc,char ** argv) {
    json cases;
    if(argc==2){std::ifstream input(argv[1]);input>>cases;}

    const auto original=std::filesystem::current_path();char temp[]="/tmp/dsv4f-pruning-XXXXXX";
    require(mkdtemp(temp),"temporary directory");std::filesystem::current_path(temp);std::filesystem::create_directory("models");
    if(argc==2) {
        json results=json::array();
        setenv("DSV4F_ADAPTIVE_ROUTING","0",1);
        for(const auto & test:cases) {
            json data={{"version",1},{"model","Pruning"},{"experts",json::object()}};
            std::vector<dsv4f_pruning_layer> layers;
            for(const auto & item:test["layers"]) {
                const std::vector<intptr_t> counts=item["counts"];
                const std::vector<uint64_t> bytes=item["bytes"];
                dsv4f_pruning_layer l{item["layer"],int(counts.size()),item["top_k"],{}};
                for(size_t i=0;i<counts.size();++i) {
                    auto name="blk."+std::to_string(l.layer)+".ffn_gate_exps.t"+std::to_string(i)+".weight";
                    auto row=name;row.replace(row.find("ffn_gate_exps"),13,"ffn_exps");
                    data["experts"][row]["0"]=counts[i];
                    std::vector<int32_t> local(counts.size(),-1);local[i]=0;
                    l.groups.push_back({name,local,bytes[i]});
                }
                layers.push_back(std::move(l));
            }
            write(data);setenv("DSV4F_ADAPTIVE_PRUNING",std::to_string(test["percentage"].get<int>()).c_str(),1);
            try {
                init();const auto masks=dsv4f_pruning_masks(layers);json result=json::array();
                for(size_t l=0;l<layers.size();++l){json ids=json::array();if(!masks.empty())for(size_t i=0;i<masks[l].size();++i)if(masks[l][i]<0)ids.push_back(i);result.push_back(ids);}
                results.push_back(result);
            }catch(const std::exception &){results.push_back(nullptr);}
        }
        std::cout<<results.dump()<<std::endl;
        std::filesystem::current_path(original);std::filesystem::remove_all(temp);return 0;
    }
    setenv("DSV4F_ADAPTIVE_ROUTING","0",1);unsetenv("DSV4F_ADAPTIVE_PRUNING");init();require(mask().empty(),"default disabled");
    for(const char * v:{"-1","51","1.5","bad","","01","999999999999999999999"}){setenv("DSV4F_ADAPTIVE_PRUNING",v,1);rejects(init);}
    setenv("DSV4F_ADAPTIVE_PRUNING","25",1);setenv("DSV4F_ADAPTIVE_ROUTING","1",1);rejects(init);
    unsetenv("DSV4F_ADAPTIVE_ROUTING");rejects(init);
    {std::ofstream f(path);f<<"{bad";}rejects(init);
    json data={{"version",1},{"model","Pruning"},{"experts",{
        {"blk.0.ffn_exps.q3_k.weight",{{"0",1},{"1",100},{"2",2},{"3",200}}},
        {"blk.0.ffn_exps.ptq1_0.weight",{{"0",300},{"1",3},{"2",400},{"3",4}}}}}};
    auto bad=data;bad["model"]="Other";write(bad);rejects(init);
    bad=data;bad["experts"].erase("blk.0.ffn_exps.ptq1_0.weight");write(bad);init();
    {const auto sparse=mask();require(sparse[1]<0 && sparse[3]<0,"absent format rows represent zero calls");}
    write(data);std::vector<float> m;
    for(int pct=1;pct<=50;++pct) {
        setenv("DSV4F_ADAPTIVE_PRUNING",std::to_string(pct).c_str(),1);init();m=mask();
        require(std::count_if(m.begin(),m.end(),[](float v){return v<0;})==(8*pct+99)/100,"whole expert rounding reaches byte target");
        if(pct==25)require(m[0]<0 && m[4]<0 && m[2]==0,"mixed-tier local to global mapping");
    }
    require(m[0]<0 && m[4]<0 && m[3]<0 && m[7]<0,"bottom half mapping");
    rejects([]{dsv4f_pruning_masks({{0,8,5,groups}});});
    require(!ggml_dsv4f_routing.bind && !ggml_dsv4f_routing.observe,"no collection hooks");
    dsv4f_routing_start();require(dsv4f_routing_save(),"save no-op");
    {std::ifstream f(path);json saved;f>>saved;require(saved==data,"history remains unchanged");}
    ggml_backend_load_all();auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);require(cpu,"CPU");
    router(cpu,m,4,true);router(cpu,m,8,false);
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(gpu){router(gpu,m,4,true);router(gpu,m,8,false);ggml_backend_free(gpu);}
    data["experts"]["blk.0.ffn_exps.q3_k.weight"]={{"3",200}};
    data["experts"]["blk.0.ffn_exps.ptq1_0.weight"]={{"2",400}};write(data);init();m=mask();
    for(int i=0;i<8;++i)require((m[i]<0)==(i<4),"unobserved counts zero; ties use global ID");
    setenv("DSV4F_ADAPTIVE_PRUNING","0",1);init();require(mask().empty(),"zero disables after reinit");
    ggml_backend_free(cpu);std::filesystem::current_path(original);std::filesystem::remove_all(temp);
    std::puts(gpu?"Adaptive pruning validation, mixed tiers, CPU/GPU fused and prefill routing PASS":"Adaptive pruning validation, mixed tiers, CPU fused and prefill routing PASS (GPU unavailable)");
}
