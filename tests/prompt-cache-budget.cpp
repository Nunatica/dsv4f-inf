#include "direct-weights.h"
#include "server-task.h"
#include "ggml-backend-moe-cache.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

constexpr uint64_t MiB=1ull<<20;
static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static uint64_t rss() {
    unsigned long pages=0,resident=0;FILE * f=std::fopen("/proc/self/statm","r");require(f,"RSS fixture");
    const int n=std::fscanf(f,"%lu %lu",&pages,&resident);std::fclose(f);require(n==2,"RSS values");
    return uint64_t(resident)*uint64_t(sysconf(_SC_PAGESIZE));
}
static auto stats() {ggml_dsv4f_host_weights_stats s{};ggml_dsv4f_host_weights.statistics(&s);return s;}
int main() {
    setenv("DSV4F_RAM_BUDGET_MIB","0",1);
    {dsv4f_ram_reservation legacy(UINT64_MAX);require(bool(legacy),"resident loader reservation must be a no-op");}
    setenv("DSV4F_RAM_BUDGET_MIB","131072",1);
    setenv("DSV4F_RAM_RESERVE_MIB","1",1); // Scale the production 1 GiB headroom down for this fixture.
    setenv("DSV4F_RAM_CACHE_MIB","48",1);setenv("DSV4F_RAM_PRELOAD","1",1);
    char path[]="/tmp/dsv4f-prompt-budget-XXXXXX";
    const int fd=mkstemp(path);require(fd>=0,"weight fixture");require(ftruncate(fd,64*MiB)==0,"size fixture");
    auto * ctx=ggml_init({1<<20,nullptr,true});require(ctx,"context");
    auto * tensor=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,512,512,64);
    auto * buffer=dsv4f_direct_weights_buffer({{tensor,fd,0}});
    server_prompt prompt;for(int id:{2,3,4})prompt.tokens.push_back(id);
    prompt.checkpoints.emplace_back();prompt.checkpoints.back().data_tgt.resize(MiB,42);
    const uint64_t budget_mib=(rss()+MiB-1)/MiB+33;
    setenv("DSV4F_RAM_BUDGET_MIB",std::to_string(budget_mib).c_str(),1);
    dsv4f_preload_experts();const auto before=stats();
    require(before.resident_bytes>=24*MiB,"fixture must fill the runtime RAM allowance");

    // A snapshot much larger than the fixed headroom must displace weights,
    // including the draft vector and copied checkpoint, rather than OOM.
    server_prompt_cache cache(64,128);
    auto * saved=cache.alloc(prompt,12*MiB,4*MiB);
    require(saved && saved->data.main.size()==12*MiB && saved->data.drft.size()==4*MiB,"prompt state allocation");
    require(saved->prompt.checkpoints.back().data_tgt.front()==42,"checkpoint copied");
    const auto after=stats();
    require(after.evictions>before.evictions && after.resident_bytes+12*MiB<before.resident_bytes,"snapshot must reclaim expert RAM");
    require(rss()<(budget_mib+1)*MiB,"runtime allocation must stay inside the RAM budget");
    cache.states.clear();

    // Two untouch-ed reservations must both survive a background preload.
    dsv4f_preload_experts();const auto full=stats();
    {
        dsv4f_ram_reservation first(8*MiB);require(bool(first),"first pending allocation");
        dsv4f_ram_reservation second(8*MiB);require(bool(second),"second pending allocation");
        dsv4f_preload_experts();
        require(stats().resident_bytes+12*MiB<full.resident_bytes,"loader refilled pending runtime allocations");
    }
    dsv4f_preload_experts();require(stats().resident_bytes>after.resident_bytes,"released reservations must permit expert refill");

    // Reclaim must never invalidate live CPU/DMA leases. When only active
    // or non-expert memory remains, skip the optional prompt cache entry.
    cache.limit_size=0;
    {
        ggml_dsv4f_host_lease pinned(tensor->data,MiB);
        const auto before_failure=stats();
        require(!cache.alloc(prompt,(budget_mib-2)*MiB,0),"unaffordable prompt state must be skipped");
        require(*(const unsigned char *)pinned.data==0,"reclaim invalidated a live lease");
        require(stats().evictions>before_failure.evictions,"reclaim did not try idle experts");
        dsv4f_ram_reservation retry(MiB);require(bool(retry),"failed reservation leaked budget accounting");
    }
    require(!cache.alloc(prompt,SIZE_MAX,1),"overflowing prompt state must be skipped");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);close(fd);unlink(path);
    std::puts("Prompt cache RAM budget: reclaim, target/draft/checkpoints, pending allocations, live leases and graceful skip PASS");
}
