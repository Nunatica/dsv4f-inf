#include "direct-weights.h"
#include "llama-mmap.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
int main() {
    setenv("DSV4F_RAM_BUDGET_MIB","131072",1);
    setenv("DSV4F_RAM_CACHE_MIB","1",1);
    setenv("DSV4F_RAM_RESERVE_MIB","4",1);
    setenv("DSV4F_RAM_UNLOCK_IO","1",1);
    char path[]="/tmp/dsv4f-direct-XXXXXX";
    const int fd=mkstemp(path);require(fd>=0,"create fixture");
    constexpr size_t offset=123,plane=65536,count=64;
    std::vector<unsigned char> expected(offset+plane*count);
    for(size_t i=0;i<expected.size();++i)expected[i]=(i*71+i/97)%251;
    require(write(fd,expected.data(),expected.size())==(ssize_t)expected.size(),"write fixture");
    require(fsync(fd)==0,"flush fixture");
    require(posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)==0,"drop test file pages");
    const size_t file_pages=(expected.size()+4095)/4096;
    void * file_map=mmap(nullptr,file_pages*4096,PROT_READ,MAP_SHARED,fd,0);
    require(file_map!=MAP_FAILED,"mincore baseline mapping");
    std::vector<unsigned char> baseline(file_pages);
    require(mincore(file_map,file_pages*4096,baseline.data())==0,"mincore baseline");
    auto * ctx=ggml_init({1<<20,nullptr,true});require(ctx,"context");
    auto * t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,128,count);
    auto * b=dsv4f_direct_weights_buffer({{t,fd,offset}});
    unsetenv("DSV4F_RAM_PRELOAD");
    auto preload=dsv4f_preload_experts();
    require(!preload.planes && !preload.read_bytes,"default performed preload");
    setenv("DSV4F_RAM_PRELOAD","0",1);
    preload=dsv4f_preload_experts();
    ggml_dsv4f_host_weights_stats disabled{};
    ggml_dsv4f_host_weights.statistics(&disabled);
    require(!preload.planes && !disabled.reads && !disabled.resident_bytes,"disabled preload read or cached weights");
    setenv("DSV4F_RAM_PRELOAD","1",1);
    // Startup must keep the first fitting prefix, without reading the entire
    // model into a rolling LRU that would leave only the last experts cached.
    const size_t allocated_plane=plane+4096,fitting=(1<<20)/allocated_plane;
    preload=dsv4f_preload_experts();
    require(preload.budget_limited && preload.planes==fitting,"preload fitting prefix");
    require(preload.resident_bytes==fitting*allocated_plane && preload.read_bytes==preload.resident_bytes,"preload accounting");
    ggml_dsv4f_host_weights_stats preloaded{},reused{};
    ggml_dsv4f_host_weights.statistics(&preloaded);
    for(size_t i=0;i<fitting;++i) {
        ggml_dsv4f_host_lease lease((char*)t->data+i*plane,plane);
        require(!std::memcmp(lease.data,expected.data()+offset+i*plane,plane),"preloaded bytes");
    }
    preload=dsv4f_preload_experts();
    ggml_dsv4f_host_weights.statistics(&reused);
    require(!preload.read_bytes && reused.reads==preloaded.reads && reused.evictions==preloaded.evictions,"preload reread/evicted existing prefix");
    setenv("DSV4F_RAM_BUDGET_MIB","0",1);
    preload=dsv4f_preload_experts();
    require(!preload.planes && !preload.read_bytes,"zero budget performed preload");
    setenv("DSV4F_RAM_BUDGET_MIB","131072",1);
    // GPU handoff drops duplicate RAM copies without invalidating live DMA
    // leases. Residency queries must neither read SSD nor promote transient data.
    require(ggml_dsv4f_host_weights.resident_many && ggml_dsv4f_host_weights.discard,"GPU handoff API");
    {
        const void * pointers[]={t->data,(char*)t->data+plane*63};
        size_t sizes[]={plane,plane};bool present[2]{};
        ggml_dsv4f_host_weights_stats before{},after{};
        ggml_dsv4f_host_weights.statistics(&before);
        ggml_dsv4f_host_weights.resident_many(pointers,sizes,present,2);
        require(present[0] && !present[1],"residency snapshot");
        sizes[0]++;ggml_dsv4f_host_weights.resident_many(pointers,sizes,present,2);
        require(!present[0],"residency accepted partial plane");sizes[0]--;
        {
            ggml_dsv4f_host_lease first(t->data,plane),second(t->data,plane);
            ggml_dsv4f_host_weights.discard(t->data,plane);
            ggml_dsv4f_host_weights.resident_many(pointers,sizes,present,2);
            require(!present[0],"handoff still advertises RAM residency");
            require(first.data==second.data && !std::memcmp(first.data,expected.data()+offset,plane),"handoff invalidated live leases");
        }
        ggml_dsv4f_host_weights.statistics(&after);
        require(after.reads==before.reads && !after.transient_bytes,"handoff read SSD or leaked a lease");
        require(before.resident_bytes-after.resident_bytes==allocated_plane,"handoff did not return RAM");
        ggml_dsv4f_host_weights.discard((char*)t->data+plane,plane);
        pointers[0]=(char*)t->data+plane;
        ggml_dsv4f_host_weights.resident_many(pointers,sizes,present,2);
        require(!present[0],"idle handoff retained RAM");
    }
    // Cold loads for distinct experts may run concurrently. Requests for the
    // same cold expert coalesce onto one in-flight read and share its lease.
    for(bool same:{false,true}) {
        std::atomic<int> ready{0};std::atomic<bool> go{false},ok{true};
        std::vector<std::thread> threads;
        for(int i=0;i<12;++i)threads.emplace_back([&,i] {
            ++ready;while(!go.load(std::memory_order_acquire))std::this_thread::yield();
            try {
                const size_t eid=same?12:size_t(i);
                ggml_dsv4f_host_lease lease((char*)t->data+eid*plane,plane);
                if(std::memcmp(lease.data,expected.data()+offset+eid*plane,plane))ok=false;
            } catch(...) {ok=false;}
        });
        while(ready.load()!=12)std::this_thread::yield();go.store(true,std::memory_order_release);
        for(auto & thread:threads)thread.join();require(ok,same?"coalesced cold leases":"parallel cold leases");
    }
    // A speculative hint must remain non-blocking for its caller. A demand
    // lease may either hit the completed entry or coalesce with its O_DIRECT
    // load, and source teardown below must drain any remaining hints safely.
    require(ggml_dsv4f_host_weights.prefetch,"direct prefetch API");
    ggml_dsv4f_host_weights.prefetch((char*)t->data+13*plane,plane);
    {
        ggml_dsv4f_host_lease lease((char*)t->data+13*plane,plane);
        require(!std::memcmp(lease.data,expected.data()+offset+13*plane,plane),"prefetched expert bytes");
    }
    require(ggml_dsv4f_host_weights.acquire_many,"batch lease API");
    {
        // Mix cold planes, a duplicate key, an ordinary resident pointer and a
        // concurrent single-plane consumer. Each returned pointer is a lease.
        const void * pointers[10];size_t sizes[10];const void * outputs[10]{};
        for(int i=0;i<8;++i){pointers[i]=(char*)t->data+(20+i)*plane;sizes[i]=plane;}
        pointers[8]=pointers[0];sizes[8]=plane;pointers[9]=expected.data();sizes[9]=16;
        std::atomic<bool> ok{true};
        std::thread consumer([&]{
            try{ggml_dsv4f_host_lease lease(pointers[0],plane);if(std::memcmp(lease.data,expected.data()+offset+20*plane,plane))ok=false;}
            catch(...){ok=false;}
        });
        ggml_dsv4f_host_weights.acquire_many(pointers,sizes,outputs,10);
        consumer.join();require(ok,"single/batch coalescing");
        for(int i=0;i<8;++i)require(!std::memcmp(outputs[i],expected.data()+offset+(20+i)*plane,plane),"batch expert bytes");
        require(outputs[8]==outputs[0] && outputs[9]==expected.data(),"batch duplicate/resident pointers");
        for(const void * p:outputs)ggml_dsv4f_host_weights.release(p);
    }
    {
        // A validation or budget failure must roll back the complete batch,
        // including previously reserved cold loads and cache-hit refcounts.
        const void * pointers[16];size_t sizes[16];const void * outputs[16]{};
        for(int i=0;i<16;++i){pointers[i]=(char*)t->data+(32+i)*plane;sizes[i]=plane;}
        sizes[1]=plane+1;bool failed=false;
        try{ggml_dsv4f_host_weights.acquire_many(pointers,sizes,outputs,2);}catch(const std::runtime_error &){failed=true;}
        require(failed,"invalid batch accepted");sizes[1]=plane;failed=false;
        try{ggml_dsv4f_host_weights.acquire_many(pointers,sizes,outputs,16);}catch(const std::runtime_error &){failed=true;}
        require(failed,"over-budget batch accepted");
        ggml_dsv4f_host_weights.acquire_many(pointers,sizes,outputs,8);
        for(int i=0;i<8;++i){require(!std::memcmp(outputs[i],expected.data()+offset+(32+i)*plane,plane),"batch rollback recovery");ggml_dsv4f_host_weights.release(outputs[i]);}
    }
    // A live lease survives many evictions and is shared by simultaneous readers.
    {
        ggml_dsv4f_host_lease first(t->data,plane);
        for(size_t i=1;i<count;++i) {
            ggml_dsv4f_host_lease lease((char*)t->data+i*plane,plane);
            require(!std::memcmp(lease.data,expected.data()+offset+i*plane,plane),"direct expert bytes");
        }
        require(!std::memcmp(first.data,expected.data()+offset,plane),"live lease was evicted");
        std::atomic<bool> ok{true};
        std::vector<std::thread> threads;
        for(int i=0;i<4;++i)threads.emplace_back([&]{
            for(int j=0;j<40;++j){ggml_dsv4f_host_lease lease(t->data,plane);if(lease.data!=first.data || std::memcmp(lease.data,expected.data()+offset,plane))ok=false;}
        });
        for(auto & thread:threads)thread.join();require(ok,"concurrent leases");
    }
    // Generic read copies through leases too; malformed/truncated payloads fail closed.
    std::vector<unsigned char> got(plane*2+111);
    ggml_backend_tensor_get(t,got.data(),plane-13,got.size());
    require(!std::memcmp(got.data(),expected.data()+offset+plane-13,got.size()),"partial generic read");
    bool rejected=false;
    try{ggml_dsv4f_host_lease invalid(t->data,plane+1);}catch(const std::runtime_error &){rejected=true;}
    require(rejected,"invalid range accepted");
    // Repeated dense scans larger than the cache must keep a stable resident
    // set. Overflow bytes disappear at last release; decode may evict normally.
    require(ggml_dsv4f_host_weights.acquire_prefill && ggml_dsv4f_host_weights.statistics,"scan API");
    ggml_dsv4f_host_weights_stats previous{};bool saw_transient=false;
    for(int pass=0;pass<3;++pass) {
        for(size_t i=0;i<count;i+=2) {
            const void * pointers[]={(char*)t->data+i*plane,(char*)t->data+(i+1)*plane};
            const size_t sizes[]={plane,plane};const void * outputs[2]{};
            ggml_dsv4f_host_weights.acquire_prefill(pointers,sizes,outputs,2);
            ggml_dsv4f_host_weights_stats live{};ggml_dsv4f_host_weights.statistics(&live);
            require(live.resident_bytes+live.transient_bytes<=1<<20,"scan exceeded cache cap");
            saw_transient|=live.transient_bytes!=0;
            for(int j=0;j<2;++j) {
                require(!std::memcmp(outputs[j],expected.data()+offset+(i+j)*plane,plane),"scan expert bytes");
                ggml_dsv4f_host_weights.release(outputs[j]);
            }
        }
        ggml_dsv4f_host_weights_stats current{};ggml_dsv4f_host_weights.statistics(&current);
        require(!current.transient_bytes,"scan transient leak");
        if(pass) {
            require(current.evictions==previous.evictions,"scan evicted the resident set");
            require(current.reads-previous.reads==current.transient_reads-previous.transient_reads,"scan readmiss admitted into stable set");
            require(current.resident_bytes==previous.resident_bytes,"scan resident set changed");
            require(current.reads-previous.reads<count,"scan did not reuse resident experts");
        }
        previous=current;
    }
    require(saw_transient,"scan never streamed overflow");
    for(size_t i=0;i<count;++i){ggml_dsv4f_host_lease decode((char*)t->data+i*plane,plane);}
    ggml_dsv4f_host_weights_stats after_decode{};ggml_dsv4f_host_weights.statistics(&after_decode);
    require(after_decode.evictions>previous.evictions,"decode did not restore ordinary LRU");
    ggml_backend_buffer_free(b);ggml_free(ctx);
    // Packed pruning IDs address nonconsecutive original file planes in all
    // demand paths, including cold batches and background prefetch.
    {
        auto * packed_ctx=ggml_init({1<<20,nullptr,true});
        auto * packed=ggml_new_tensor_3d(packed_ctx,GGML_TYPE_F32,128,128,3);
        const std::vector<int32_t> original_ids={1,7,21};
        auto * packed_buffer=dsv4f_direct_weights_buffer({{packed,fd,offset,0,original_ids}});
        ggml_dsv4f_host_weights_stats before{},after{};ggml_dsv4f_host_weights.statistics(&before);
        const void * pointers[]={packed->data,(char*)packed->data+plane};
        const size_t sizes[]={plane,plane};const void * outputs[2]{};
        ggml_dsv4f_host_weights.acquire_many(pointers,sizes,outputs,2);
        for(int i=0;i<2;++i) {
            require(!std::memcmp(outputs[i],expected.data()+offset+original_ids[i]*plane,plane),"packed cold batch original offsets");
            ggml_dsv4f_host_weights.release(outputs[i]);
        }
        ggml_dsv4f_host_weights.prefetch((char*)packed->data+2*plane,plane);
        {ggml_dsv4f_host_lease lease((char*)packed->data+2*plane,plane);
            require(!std::memcmp(lease.data,expected.data()+offset+original_ids[2]*plane,plane),"packed prefetch original offset");}
        ggml_dsv4f_host_weights.statistics(&after);
        require(after.reads-before.reads==3 && after.resident_bytes==3*allocated_plane,"packed cache contains retained planes only");
        preload=dsv4f_preload_experts();require(preload.planes==3 && !preload.read_bytes && !preload.budget_limited,"packed preload reuses demand planes");
        ggml_backend_buffer_free(packed_buffer);ggml_free(packed_ctx);
    }
    // Descriptor/tensor name order can differ from on-disk order. Shard index
    // takes precedence over offset, even with the later shard registered first.
    {
        auto * order_ctx=ggml_init({1<<20,nullptr,true});
        auto * later=ggml_new_tensor_3d(order_ctx,GGML_TYPE_F32,128,1280,1);
        auto * later_offset=ggml_new_tensor_3d(order_ctx,GGML_TYPE_F32,128,1280,1);
        auto * earlier=ggml_new_tensor_3d(order_ctx,GGML_TYPE_F32,128,1280,1);
        auto * order_buffer=dsv4f_direct_weights_buffer({{later,fd,offset,1},
            {later_offset,fd,offset+plane*30,0},{earlier,fd,offset+plane*20,0}});
        ggml_dsv4f_host_weights_stats before{},after{};
        preload=dsv4f_preload_experts();
        require(preload.planes==1 && preload.budget_limited,"preload shard order count");
        ggml_dsv4f_host_weights.statistics(&before);
        {ggml_dsv4f_host_lease lease(earlier->data,earlier->nb[2]);
            require(!std::memcmp(lease.data,expected.data()+offset+plane*20,earlier->nb[2]),"preload earlier shard bytes");}
        ggml_dsv4f_host_weights.statistics(&after);
        require(after.reads==before.reads,"preload did not start with earlier shard");
        setenv("DSV4F_RAM_CACHE_MIB","2",1);
        preload=dsv4f_preload_experts();
        require(preload.planes==3 && !preload.budget_limited,"all experts fit preload");
        ggml_backend_buffer_free(order_buffer);ggml_free(order_ctx);
        setenv("DSV4F_RAM_CACHE_MIB","1",1);
    }
    // The engine never faults weight file pages into the OS file cache,
    // including during startup preload.
    std::vector<unsigned char> residency(file_pages);
    require(mincore(file_map,file_pages*4096,residency.data())==0,"mincore");
    for(size_t i=0;i<file_pages;++i)require(!(residency[i]&1) || (baseline[i]&1),"O_DIRECT populated the file page cache");
    munmap(file_map,file_pages*4096);
    // Consecutive unaligned direct chunks must keep the logical file cursor.
    {
        llama_file input(path,"rb",true);
        require(input.has_direct_io(),"direct chunk reader");
        input.seek(offset,SEEK_SET);
        input.read_raw(got.data(),17001);
        input.read_raw(got.data()+17001,19003);
        require(input.tell()==offset+36004,"direct logical cursor");
        require(!std::memcmp(got.data(),expected.data()+offset,36004),"consecutive direct chunks");
    }
    // Source removal must leave the resident path usable.
    {ggml_dsv4f_host_lease ordinary(expected.data(),16);require(ordinary.data==expected.data(),"resident fallback");}
    const int direct=open(path,O_RDONLY|O_DIRECT);require(direct>=0,"direct fd");
    std::vector<unsigned char> a(257),c(819),d(33);
    const dsv4f_direct_request batch[]={{direct,a.data(),a.size(),offset+17},{direct,c.data(),c.size(),offset+plane+91},{direct,d.data(),d.size(),offset+3*plane-11}};
    dsv4f_direct_read_many(batch,3);
    require(!std::memcmp(a.data(),expected.data()+offset+17,a.size()) &&
            !std::memcmp(c.data(),expected.data()+offset+plane+91,c.size()) &&
            !std::memcmp(d.data(),expected.data()+offset+3*plane-11,d.size()),"batched direct chunks");
    require(ftruncate(fd,16)==0,"truncate fixture");rejected=false;
    try{dsv4f_direct_read(direct,got.data(),plane,offset);}catch(const std::runtime_error &){rejected=true;}
    require(rejected,"truncation accepted");
    auto * bad_ctx=ggml_init({1<<20,nullptr,true});auto * bad=ggml_new_tensor_3d(bad_ctx,GGML_TYPE_F32,128,128,2);
    auto * bad_buffer=dsv4f_direct_weights_buffer({{bad,fd,offset}});
    const void * invalid[]={bad->data,(char*)bad->data+plane};const size_t invalid_sizes[]={plane,plane};const void * invalid_output[2]{};
    rejected=false;
    try{ggml_dsv4f_host_weights.acquire_many(invalid,invalid_sizes,invalid_output,2);}catch(const std::runtime_error &){rejected=true;}
    require(rejected,"batch truncation accepted");ggml_backend_buffer_free(bad_buffer);ggml_free(bad_ctx);
    close(direct);close(fd);unlink(path);
    std::puts("PASS: direct SSD bytes, bounded eviction, concurrent leases, EOF, no file page cache");
}
