#include "routing-table.h"
#include "direct-weights.h"
#include "ggml-dsv4f-routing.h"
#include "nlohmann/json.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

using json=nlohmann::json;
static void require(bool value,const char * why){if(!value)throw std::runtime_error(why);}
static json read(const std::string & path){std::ifstream f(path);json j;f>>j;return j;}
static void write(const std::string & path,const json & j){std::ofstream f(path);f<<j.dump();}
static int drains=0;
static void drain(){++drains;}
static int warm_starts=0;
static void * warm_row=nullptr;
static size_t warm_start(){
    ++warm_starts;
    const auto before=ggml_dsv4f_routing_count(warm_row,63);
    const int32_t id=63;const float weight=1;
    ggml_dsv4f_routing_observe(warm_row,&id,&weight,1);
    require(ggml_dsv4f_routing_count(warm_row,63)==before,"VRAM preload happens before recording begins");
    return 1;
}

int main() {
    const auto original=std::filesystem::current_path();
    char directory[]="/tmp/dsv4f-routing-XXXXXX";
    require(mkdtemp(directory),"temporary directory");std::filesystem::current_path(directory);
    unsetenv("DSV4F_ADAPTIVE_ROUTING");
    unsetenv("DSV4F_ADAPTIVE_PRUNING");
    const std::string model="Example-00001-of-00009.gguf",path="models/Example-00001-of-00009.routing.json";
    const std::string group="blk.2.ffn_exps.q3_k.weight";
    auto * ctx=ggml_init({1<<20,nullptr,true});require(ctx,"context");
    auto * gate=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,128,64);
    auto * up=ggml_dup_tensor(ctx,gate),* down=ggml_dup_tensor(ctx,gate);
    ggml_set_name(gate,"blk.2.ffn_gate_exps.q3_k.weight");
    ggml_set_name(up,"blk.2.ffn_up_exps.q3_k.weight");
    ggml_set_name(down,"blk.2.ffn_down_exps.q3_k.weight");
    dsv4f_routing_init(model);dsv4f_routing_start();
    require(!ggml_dsv4f_routing.bind && !ggml_dsv4f_routing.observe && !ggml_dsv4f_routing.count,"disabled by default");
    require(dsv4f_routing_save() && !std::filesystem::exists(path),"default creates no routing file");
    setenv("DSV4F_ADAPTIVE_ROUTING","1",1);
    dsv4f_routing_init(model);require(ggml_dsv4f_routing.bind,"explicitly enabled");
    auto * row=ggml_dsv4f_routing_bind(gate);
    require(row && row==ggml_dsv4f_routing_bind(up) && row==ggml_dsv4f_routing_bind(down),"projection counters shared");
    const int32_t ids[]={63,0,1,511};const float weights[]={1,0,1,1};
    ggml_dsv4f_routing_observe(row,ids,weights,4);
    require(!ggml_dsv4f_routing_count(row,63),"warmup excluded");
    ggml_dsv4f_experts.gpu_warm_start=warm_start;
    ggml_dsv4f_experts.gpu_drain=drain;dsv4f_routing_start();require(drains==1,"warmup observations drained");
    require(warm_starts==0,"no file skips VRAM preload");
    std::vector<std::thread> threads;
    for(int i=0;i<8;++i)threads.emplace_back([&]{
        for(int n=0;n<128;++n)ggml_dsv4f_routing_observe(row,ids,weights,4);
    });
    for(auto & t:threads)t.join();
    const intptr_t observed=ggml_dsv4f_routing_count(row,63);
    require(observed>0 && observed<=1024,"concurrent counters allow lost increments");
    require(!ggml_dsv4f_routing_count(row,0),"zero-weight exclusion");
    const intptr_t observed_last=ggml_dsv4f_routing_count(row,511);
    require(observed_last>0 && observed_last<=1024,"512-expert approximate counting");
    require(dsv4f_routing_save() && drains==2,"save drains observations");
    ggml_dsv4f_experts.gpu_drain=nullptr;
    auto saved=read(path);require(saved["experts"][group]["63"]==observed,"JSON counters and model filename");
    require(saved["experts"][group]["511"]==observed_last,"512-expert geometry");
    dsv4f_routing_init("Example-00002-of-00009.gguf");row=ggml_dsv4f_routing_bind(gate);warm_row=row;dsv4f_routing_start();
    require(warm_starts==1,"loaded positive history triggers VRAM preload");
    ggml_dsv4f_experts.gpu_warm_start=nullptr;
    require(ggml_dsv4f_routing_count(row,63)==observed,"restart uses first-shard history even from another entry shard");

    // Only the hottest high-ID expert fits ahead of the physical prefix.
    // Plane copies/preload must never increment routing history.
    setenv("DSV4F_RAM_BUDGET_MIB","131072",1);setenv("DSV4F_RAM_CACHE_MIB","1",1);
    setenv("DSV4F_RAM_RESERVE_MIB","4",1);setenv("DSV4F_RAM_PRELOAD","1",1);
    const int fd=::open("weights",O_CREAT|O_RDWR,0600);require(fd>=0,"weight fixture");
    constexpr size_t plane=65536;
    require(ftruncate(fd,plane*64)==0,"size weight fixture");
    // Expert 1 has a historical count too; remove it so the expected prefix is exact.
    saved["experts"][group].erase("1");saved["experts"][group].erase("511");write(path,saved);
    dsv4f_routing_init(model);row=ggml_dsv4f_routing_bind(gate);dsv4f_routing_start();
    auto * buffer=dsv4f_direct_weights_buffer({{gate,fd,0}});
    const auto preload=dsv4f_preload_experts();require(preload.planes==16 && preload.budget_limited,"bounded preload");
    auto present=[&](int id){const void * p=(char*)gate->data+id*plane;size_t size=plane;bool hit=false;ggml_dsv4f_host_weights.resident_many(&p,&size,&hit,1);return hit;};
    require(present(63) && present(0) && present(14) && !present(15),"preload descending count with physical tie order");
    {ggml_dsv4f_host_lease lease((char*)gate->data+16*plane,plane);}
    require(present(63) && !present(0) && present(16),"RAM eviction combines history and LRU");
    const void * pointers[]={(char*)gate->data+17*plane,(char*)gate->data+18*plane};
    const size_t sizes[]={plane,plane};const void * output[2]{};
    ggml_dsv4f_host_weights.acquire_many(pointers,sizes,output,2);
    for(auto * p:output)ggml_dsv4f_host_weights.release(p);
    require(present(63) && !present(1) && !present(2),"batched eviction uses same historical score");
    require(ggml_dsv4f_routing_count(row,63)==observed,"weight copies do not count as demand");
    ggml_backend_buffer_free(buffer);close(fd);

    ggml_dsv4f_experts.gpu_warm_start=warm_start;
    setenv("DSV4F_ADAPTIVE_ROUTING","0",1);dsv4f_routing_init(model);dsv4f_routing_start();
    require(warm_starts==1,"OFF skips VRAM preload");
    require(!ggml_dsv4f_routing.bind && !ggml_dsv4f_routing.count,"OFF clears all hooks");
    require(dsv4f_routing_save() && read(path)==saved,"OFF leaves routing file unchanged");
    dsv4f_routing_init("Disabled.gguf");require(dsv4f_routing_save() && !std::filesystem::exists("models/Disabled.routing.json"),"OFF creates no file");
    setenv("DSV4F_ADAPTIVE_ROUTING","invalid",1);bool rejected=false;
    try{dsv4f_routing_init(model);}catch(const std::exception &){rejected=true;}require(rejected,"invalid setting rejected");
    setenv("DSV4F_ADAPTIVE_ROUTING","1",1);
    {std::ofstream f(path);f<<"{bad JSON";}
    dsv4f_routing_init(model);row=ggml_dsv4f_routing_bind(gate);
    require(!ggml_dsv4f_routing_count(row,63),"corrupt file ignored safely");
    dsv4f_routing_start();require(warm_starts==1,"corrupt history skips VRAM preload");
    write(path,{{"version",1},{"model","Example-00001-of-00009"},{"experts",{{group,{{"63",0}}}}}});
    dsv4f_routing_init(model);dsv4f_routing_start();require(warm_starts==1,"zero history skips VRAM preload");
    ggml_dsv4f_experts.gpu_warm_start=nullptr;
    saved["model"]="Different";write(path,saved);dsv4f_routing_init(model);
    require(!ggml_dsv4f_routing_count(ggml_dsv4f_routing_bind(gate),63),"model identity checked");
    saved["model"]="Example-00001-of-00009";saved["experts"][group]["63"]=-1;write(path,saved);dsv4f_routing_init(model);
    require(!ggml_dsv4f_routing_count(ggml_dsv4f_routing_bind(gate),63),"negative count rejected");
    saved["experts"][group]["63"]=uint64_t(INTPTR_MAX)+1;write(path,saved);dsv4f_routing_init(model);
    require(!ggml_dsv4f_routing_count(ggml_dsv4f_routing_bind(gate),63),"count above INTPTR_MAX rejected");
    saved["experts"][group]["63"]=INTPTR_MAX-1;write(path,saved);dsv4f_routing_init(model);dsv4f_routing_start();row=ggml_dsv4f_routing_bind(gate);
    ggml_dsv4f_routing_observe(row,ids,weights,4);
    require(ggml_dsv4f_routing_count(row,63)==INTPTR_MAX,"counter reaches signed maximum");
    ggml_dsv4f_routing_observe(row,ids,weights,4);
    require(ggml_dsv4f_routing_count(row,63)==INTPTR_MAX,"signed counter saturates without overflowing");
    require(ggml_dsv4f_routing_bonus(INTPTR_MAX)==sizeof(intptr_t)*8-1,"historical bonus bounded");
    require(dsv4f_routing_save(),"save signed maximum");dsv4f_routing_init(model);row=ggml_dsv4f_routing_bind(gate);
    require(ggml_dsv4f_routing_count(row,63)==INTPTR_MAX,"signed maximum survives save and reload");
    // A failed save preserves the existing routing file and remains recoverable.
    const auto before=read(path);std::filesystem::rename("models","backup");{std::ofstream f("models");f<<"blocked";}
    require(!dsv4f_routing_save() && read("backup/Example-00001-of-00009.routing.json")==before,"save failure preserves previous history");
    std::filesystem::remove("models");std::filesystem::rename("backup","models");
    require(dsv4f_routing_save(),"save recovers after filesystem failure");
    setenv("DSV4F_ADAPTIVE_ROUTING","0",1);dsv4f_routing_init(model);ggml_free(ctx);
    std::filesystem::current_path(original);std::filesystem::remove_all(directory);
    std::puts("Adaptive routing counters, persistence, preload, RAM eviction, OFF and recovery PASS");
}
