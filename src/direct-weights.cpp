#include "direct-weights.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-dsv4f-routing.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/aio_abi.h>
#endif

namespace {
constexpr size_t alignment=4096;
constexpr uint64_t MiB=1ull<<20;
uint64_t process_rss() {
    unsigned long pages=0,resident=0;
    if(FILE * f=std::fopen("/proc/self/statm","r")) {
        const int n=std::fscanf(f,"%lu %lu",&pages,&resident);std::fclose(f);
        if(n==2)return uint64_t(resident)*uint64_t(sysconf(_SC_PAGESIZE));
    }
    throw std::runtime_error("cannot read process RAM usage");
}
// Check every cgroup ancestor: charged GPU/shared/file memory and the
// supervisor are part of the hard limit too, even when absent from our RSS.
// Only used at large runtime-allocation boundaries, not on expert reads.
uint64_t runtime_headroom(uint64_t budget,uint64_t reserve) {
    const auto available=[&](uint64_t limit,uint64_t used) {
        return used<limit && reserve<limit-used?limit-used-reserve:0;
    };
    uint64_t result=available(budget,process_rss());
    std::ifstream groups("/proc/self/cgroup");std::string line;
    while(std::getline(groups,line))if(line.compare(0,3,"0::")==0) {
        auto path=std::filesystem::path("/sys/fs/cgroup")/std::filesystem::path(line.substr(3)).relative_path();
        for(;;path=path.parent_path()) {
            std::ifstream max_file(path/"memory.max"),current_file(path/"memory.current");
            std::string maximum;uint64_t current=0;
            if(max_file>>maximum && maximum!="max" && current_file>>current) {
                try {result=std::min(result,available(std::stoull(maximum),current));}
                catch(const std::exception &){throw std::runtime_error("invalid cgroup RAM limit");}
            }
            if(path=="/sys/fs/cgroup" || path==path.root_path())break;
        }
        break;
    }
    return result;
}
uint64_t env_mib(const char * name,uint64_t fallback) {
    const char * v=std::getenv(name);
    if(!v)return fallback;
    if(!*v)throw std::runtime_error(std::string(name)+" must be an integer in MiB");
    uint64_t n=0;
    for(const char * p=v;*p;++p) {
        if(*p<'0'||*p>'9'||n>(std::numeric_limits<uint64_t>::max()/MiB-(*p-'0'))/10)
            throw std::runtime_error(std::string(name)+" must be a nonnegative integer in MiB");
        n=n*10+(*p-'0');
    }
    return n*MiB;
}
bool unlocked_io() {
    static const bool value=[] {
        const char * v=std::getenv("DSV4F_RAM_UNLOCK_IO");
        if(!v || !std::strcmp(v,"1"))return true;
        if(!std::strcmp(v,"0"))return false;
        throw std::runtime_error("DSV4F_RAM_UNLOCK_IO must be 0 or 1");
    }();
    return value;
}
size_t rounded(size_t n) {
    if(n>SIZE_MAX-alignment)throw std::runtime_error("direct I/O size overflow");
    return (n+alignment-1)&~(alignment-1);
}
struct region {
    void * data=MAP_FAILED;
    size_t bytes=0;
    explicit region(size_t n,int prot=PROT_READ|PROT_WRITE):bytes(rounded(n)) {
        data=mmap(nullptr,bytes,prot,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
        if(data==MAP_FAILED)throw std::runtime_error("cannot reserve direct I/O memory");
    }
    ~region(){if(data!=MAP_FAILED)munmap(data,bytes);}
    void populate() {
#if defined(MADV_POPULATE_WRITE)
        if(madvise(data,bytes,MADV_POPULATE_WRITE)==0)return;
        if(errno!=EINVAL && errno!=ENOSYS && errno!=EOPNOTSUPP)
            throw std::runtime_error(std::string("cannot populate preload memory: ")+std::strerror(errno));
#endif
        // Materialize anonymous pages before the reader uses this batch.
        auto * p=static_cast<volatile unsigned char *>(data);
        for(size_t at=0;at<bytes;at+=alignment)p[at]=0;
    }
};
void read_aligned(int fd,void * dst,size_t n,uint64_t off,size_t required) {
    // A short final sector is legal only when every requested payload byte arrived.
    size_t done=0;
    while(done<n) {
        const size_t chunk=std::min<size_t>(n-done,64*MiB);
        ssize_t got=pread(fd,(char*)dst+done,chunk,off+done);
        if(got<0 && errno==EINTR)continue;
        if(got<0)throw std::runtime_error(std::string("O_DIRECT read failed: ")+std::strerror(errno));
        if(got==0 || (size_t(got)<chunk && done+got<required))throw std::runtime_error("truncated direct I/O payload");
        done+=got;
        if(size_t(got)<chunk)break;
    }
}
struct source {
    std::unique_ptr<region> address;
    struct span {
        uintptr_t begin;size_t bytes,expert;int fd;uint64_t offset;uint32_t file_index;void * routing;
        std::vector<int32_t> expert_ids;
        uint64_t file_offset(size_t local) const {
            return offset+(expert_ids.empty()?local:size_t(expert_ids.at(local/expert))*expert);
        }
    };
    std::map<uintptr_t,span> spans;
    std::vector<int> fds;
    bool closing=false;
    size_t prefetches=0;
    uint64_t sequence=0;
    ~source(){for(int fd:fds)close(fd);}
};
struct load {
    std::unique_ptr<region> storage;
    source * owner=nullptr;
    const void * pointer=nullptr;
    size_t users=1;
    bool done=false;
    bool transient=false;
    bool populated=false;
    void * routing=nullptr;
    int expert=0;
    std::exception_ptr error;
    std::condition_variable ready;
};
struct entry {
    std::unique_ptr<region> storage;
    const void * pointer=nullptr;
    size_t users=0;
    size_t bytes=0;
    source * owner=nullptr;
    bool transient=false;
    void * routing=nullptr;
    int expert=0;
    std::list<uintptr_t>::iterator age;
};
struct cache {
    std::mutex mutex;
    std::condition_variable changed;
    std::map<uintptr_t,source*> sources;
    std::map<uintptr_t,entry> entries;
    std::map<uintptr_t,std::shared_ptr<load>> loads;
    std::map<const void*,uintptr_t> leases;
    std::list<uintptr_t> ages;
    struct prefetch_job { source * owner; uintptr_t key; size_t bytes; };
    std::condition_variable prefetch_ready;
    std::deque<prefetch_job> prefetch_queue;
    std::unordered_set<uintptr_t> prefetch_pending;
    std::vector<std::thread> prefetch_workers;
    uint64_t used=0,reserved=0,hits=0,reads=0,bytes=0,evictions=0;
    uint64_t transient_used=0,transient_reserved=0,transient_reads=0,transient_bytes=0;
    uint64_t prefetch_queued=0,prefetch_completed=0,prefetch_dropped=0,prefetch_failed=0;
    uint64_t next_source=0;
    uint64_t runtime_reserved=0;
    uint64_t populated_reserved=0;
    ~cache()=default;
    uint64_t limit() const {
        const uint64_t budget=dsv4f_ram_budget();
        const uint64_t reserve=env_mib("DSV4F_RAM_RESERVE_MIB",1024*MiB);
        if(budget<=reserve)throw std::runtime_error("RAM budget must exceed DSV4F_RAM_RESERVE_MIB");
        uint64_t ceiling=std::min(env_mib("DSV4F_RAM_CACHE_MIB",budget-reserve),budget-reserve);
        // Account for actual process RSS (KV/history, Engram, GPU host buffers, stacks, etc.).
        // mmap/munmap for cache entries returns evicted memory immediately to the OS.
        const uint64_t rss=process_rss(),cache_rss=used+populated_reserved;
        const uint64_t other=rss>cache_rss?rss-cache_rss:0;
        const uint64_t room=budget-reserve>other?budget-reserve-other:0;
        ceiling=std::min(ceiling,room>runtime_reserved?room-runtime_reserved:0);
        return ceiling;
    }
    std::list<uintptr_t>::iterator victim(bool resident_only=false) {
        auto best=ages.end();uint64_t low=UINT64_MAX;size_t rank=0;
        for(auto it=ages.begin();it!=ages.end();++it) {
            const auto & e=entries.at(*it);
            if(e.users || (resident_only && e.transient))continue;
            if(!ggml_dsv4f_routing.count)return it; // Preserve ordinary LRU when disabled.
            const uint64_t score=rank+8*ggml_dsv4f_routing_bonus(ggml_dsv4f_routing_count(e.routing,e.expert));
            if(!score)return it; // No later LRU rank can improve on zero.
            if(score<low){low=score;best=it;}
            if(++rank==128)break;
        }
        return best;
    }
    void erase(std::map<uintptr_t,entry>::iterator it) {
        if(it->second.transient)transient_used-=it->second.storage->bytes;
        else ++evictions;
        used-=it->second.storage->bytes;leases.erase(it->second.pointer);
        ages.erase(it->second.age);entries.erase(it);
    }
    void release_locked(const void * ptr) {
        auto lease=leases.find(ptr);if(lease==leases.end())return;
        auto it=entries.find(lease->second);GGML_ASSERT(it!=entries.end() && it->second.users);
        if(!--it->second.users) {
            if(it->second.transient)erase(it);
            changed.notify_all();
        }
    }
};
// Outlives model/backend static teardown; it contains no allocations after model unload.
cache & pool(){static auto * p=new cache;return *p;}
const void * acquire(const void * ptr,size_t n);
const void * acquire_impl(const void * ptr,size_t n,bool preload);
void release(const void * ptr);
void prefetch_worker() {
    auto & c=pool();
    for(;;) {
        cache::prefetch_job job{};
        {
            std::unique_lock<std::mutex> lock(c.mutex);
            c.prefetch_ready.wait(lock,[&]{return !c.prefetch_queue.empty();});
            job=c.prefetch_queue.front();c.prefetch_queue.pop_front();
        }
        bool ok=false;
        try {const void * p=acquire((const void *)job.key,job.bytes);release(p);ok=true;}
        catch(...) {}
        {
            std::lock_guard<std::mutex> lock(c.mutex);
            c.prefetch_pending.erase(job.key);
            GGML_ASSERT(job.owner->prefetches);--job.owner->prefetches;
            if(ok)++c.prefetch_completed;else ++c.prefetch_failed;
            c.changed.notify_all();
        }
    }
}
void prefetch(const void * ptr,size_t n) {
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    const auto key=(uintptr_t)ptr;
    auto s=c.sources.upper_bound(key);if(s==c.sources.begin())return;--s;
    source * owner=s->second;
    if(key-s->first>=owner->address->bytes || owner->closing)return;
    auto sp=owner->spans.upper_bound(key);if(sp==owner->spans.begin())return;--sp;
    const auto & span=sp->second;const size_t local=key-span.begin;
    if(local>=span.bytes || n!=span.expert || local%span.expert || n>span.bytes-local)return;
    if(c.entries.count(key) || c.loads.count(key) || c.prefetch_pending.count(key))return;
    // Predictions are useful for only the immediately following layer. Keep
    // the newest hints when SSD workers fall behind instead of draining stale
    // layers and displacing demand-loaded experts.
    constexpr size_t queue_limit=96;
    while(c.prefetch_queue.size()>=queue_limit) {
        const auto old=c.prefetch_queue.front();c.prefetch_queue.pop_front();
        c.prefetch_pending.erase(old.key);GGML_ASSERT(old.owner->prefetches);
        --old.owner->prefetches;++c.prefetch_dropped;
    }
    c.prefetch_pending.insert(key);c.prefetch_queue.push_back({owner,key,n});
    ++owner->prefetches;++c.prefetch_queued;
    if(c.prefetch_workers.empty())for(int i=0;i<3;++i)c.prefetch_workers.emplace_back(prefetch_worker);
    c.prefetch_ready.notify_all();
}
const void * acquire(const void * ptr,size_t n) {
    return acquire_impl(ptr,n,false);
}
const void * acquire_impl(const void * ptr,size_t n,bool preload) {
    auto & c=pool();std::unique_lock<std::mutex> lock(c.mutex);
    const auto key=(uintptr_t)ptr;
    auto s=c.sources.upper_bound(key);
    if(s==c.sources.begin())return ptr;
    --s;source * owner=s->second;
    if(key-s->first>=owner->address->bytes)return ptr;
    if(owner->closing)throw std::runtime_error("direct weight source is closing");
    auto sp=owner->spans.upper_bound(key);
    if(sp==owner->spans.begin())throw std::runtime_error("invalid direct weight address");
    --sp;const auto & span=sp->second;
    const size_t local=key-span.begin;
    if(local>=span.bytes || n!=span.expert || local%span.expert || n>span.bytes-local)
        throw std::runtime_error("direct weights must be leased one complete expert plane at a time");
    auto it=c.entries.find(key);
    if(it!=c.entries.end()) {
        ++it->second.users;++c.hits;c.ages.splice(c.ages.end(),c.ages,it->second.age);
        return it->second.pointer;
    }
    auto loading=c.loads.find(key);
    if(loading!=c.loads.end()) {
        auto request=loading->second;++request->users;
        request->ready.wait(lock,[&]{return request->done;});
        if(request->error)std::rethrow_exception(request->error);
        ++c.hits;
        return request->pointer;
    }
    const uint64_t offset=span.file_offset(local);
    const size_t delta=offset%alignment,total=rounded(delta+n);
    const uint64_t cap=c.limit();
    // Startup fills only unused room. Never scan/evict the already loaded
    // prefix just to replace it with later weights when the cache is full.
    if(preload && (total>cap || c.used>cap-total || c.reserved>cap-total-c.used))return nullptr;
    if(total>cap)throw std::runtime_error("RAM budget has no room for an expert plane; reduce context/batch or reserve");
    while(c.used>cap-total || c.reserved>cap-total-c.used) {
        auto victim=c.victim();
        if(victim==c.ages.end())throw std::runtime_error("RAM expert cache is exhausted by active leases");
        c.erase(c.entries.find(*victim));
    }
    auto storage=std::make_unique<region>(total);
    if(!unlocked_io()) {
        read_aligned(span.fd,storage->data,total,offset-delta,delta+n);
        const void * output=(char*)storage->data+delta;
        c.ages.push_back(key);
        entry e;e.storage=std::move(storage);e.pointer=output;e.users=1;e.bytes=n;e.owner=owner;e.routing=span.routing;e.expert=int(local/span.expert);e.age=std::prev(c.ages.end());
        c.entries.emplace(key,std::move(e));c.leases.emplace(output,key);c.used+=total;++c.reads;c.bytes+=total;
        if(c.reads%4096==0)std::fprintf(stderr,"[dsv4f-ram] resident=%.2f MiB hit=%llu reads=%llu direct_mib=%.2f evictions=%llu\n",c.used/double(MiB),(unsigned long long)c.hits,(unsigned long long)c.reads,c.bytes/double(MiB),(unsigned long long)c.evictions);
        return output;
    }
    auto request=std::make_shared<load>();request->storage=std::move(storage);request->owner=owner;
    c.loads.emplace(key,request);c.reserved+=total;
    lock.unlock();
    try {read_aligned(span.fd,request->storage->data,total,offset-delta,delta+n);}
    catch(...) {request->error=std::current_exception();}
    lock.lock();c.reserved-=total;
    if(request->error) {
        request->storage.reset();request->done=true;c.loads.erase(key);
        lock.unlock();request->ready.notify_all();c.changed.notify_all();
        std::rethrow_exception(request->error);
    }
    const void * output=(char*)request->storage->data+delta;
    c.ages.push_back(key);
    entry e;e.storage=std::move(request->storage);e.pointer=output;e.users=request->users;e.bytes=n;e.owner=owner;e.routing=span.routing;e.expert=int(local/span.expert);e.age=std::prev(c.ages.end());
    c.entries.emplace(key,std::move(e));c.leases.emplace(output,key);c.used+=total;++c.reads;c.bytes+=total;
    request->pointer=output;request->done=true;c.loads.erase(key);
    if(c.reads%4096==0)std::fprintf(stderr,"[dsv4f-ram] resident=%.2f MiB hit=%llu reads=%llu direct_mib=%.2f evictions=%llu\n",c.used/double(MiB),(unsigned long long)c.hits,(unsigned long long)c.reads,c.bytes/double(MiB),(unsigned long long)c.evictions);
    lock.unlock();request->ready.notify_all();c.changed.notify_all();
    return output;
}
void release(const void * ptr) {
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    c.release_locked(ptr);
}

struct batch_read {
    uintptr_t key;
    std::shared_ptr<load> request;
    int fd;
    uint64_t offset;
    size_t delta,bytes,required;
};
void read_batch(const batch_read * reads,size_t count) {
    if(!count)return;
#if defined(__linux__) && defined(__NR_io_setup) && defined(__NR_io_submit) && defined(__NR_io_getevents) && defined(__NR_io_destroy)
    struct aio_state {
        aio_context_t context=0;
        aio_state(){if(syscall(__NR_io_setup,128,&context)<0)context=0;}
        void drain(){if(context){syscall(__NR_io_destroy,context);context=0;}}
        ~aio_state(){drain();}
    };
    thread_local aio_state aio;
    if(aio.context && count>1 && count<=128) {
        std::vector<iocb> blocks(count);std::vector<iocb*> pointers(count);
        std::vector<io_event> events(count);
        for(size_t i=0;i<count;++i) {
            auto & b=blocks[i];b.aio_data=i;b.aio_lio_opcode=IOCB_CMD_PREAD;b.aio_fildes=reads[i].fd;
            b.aio_buf=(uintptr_t)reads[i].request->storage->data;b.aio_nbytes=reads[i].bytes;
            b.aio_offset=reads[i].offset;pointers[i]=&b;
        }
        size_t submitted=0;
        while(submitted<count) {
            const long n=syscall(__NR_io_submit,aio.context,count-submitted,pointers.data()+submitted);
            if(n>0){submitted+=size_t(n);continue;}
            if(n<0 && errno==EINTR)continue;
            break; // Drain submitted work, then pread any unsubmitted requests.
        }
        size_t completed=0;
        while(completed<submitted) {
            const long n=syscall(__NR_io_getevents,aio.context,1,submitted-completed,events.data()+completed,nullptr);
            if(n>0){completed+=size_t(n);continue;}
            if(n<0 && errno==EINTR)continue;
            const int error=errno;
            aio.drain(); // Never free a cache allocation while the kernel owns it.
            throw std::runtime_error(std::string("expert batch completion failed: ")+std::strerror(error));
        }
        for(size_t k=0;k<submitted;++k) {
            const auto & event=events[k];const size_t i=event.data;
            if(i>=count || event.res<0 || uint64_t(event.res)<reads[i].required)
                throw std::runtime_error(event.res<0?std::string("expert batch O_DIRECT read failed: ")+std::strerror(int(-event.res)):"truncated expert batch payload");
        }
        if(submitted) {
            static std::once_flag announced;std::call_once(announced,[&]{
                std::fprintf(stderr,"[dsv4f-ram-batch] reader=linux-aio depth=%zu destination=cache\n",submitted);
            });
            for(size_t i=submitted;i<count;++i)
                read_aligned(reads[i].fd,reads[i].request->storage->data,reads[i].bytes,reads[i].offset,reads[i].required);
            return;
        }
    }
#endif
    for(size_t i=0;i<count;++i)
        read_aligned(reads[i].fd,reads[i].request->storage->data,reads[i].bytes,reads[i].offset,reads[i].required);
}

struct preload_plane {uintptr_t pointer;size_t bytes;intptr_t count;};
void preload_pipelined(const std::vector<preload_plane> & planes,dsv4f_preload_result & result,
                      uint64_t batch_limit,uint64_t initial_bytes) {
    auto & c=pool();
    using batch=std::vector<batch_read>;
    auto complete=[&](const std::shared_ptr<batch> & reads,std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(c.mutex);
        for(const auto & r:*reads) {
            auto & request=*r.request;
            if(request.populated){c.populated_reserved-=r.bytes;request.populated=false;}
            c.reserved-=r.bytes;
            if(error) {request.error=error;request.storage.reset();}
            else {
                const void * output=(char*)request.storage->data+r.delta;c.ages.push_back(r.key);
                entry e;e.storage=std::move(request.storage);e.pointer=output;e.users=request.users;
                e.bytes=r.required-r.delta;e.owner=request.owner;e.routing=request.routing;e.expert=request.expert;
                e.age=std::prev(c.ages.end());c.entries.emplace(r.key,std::move(e));c.leases.emplace(output,r.key);
                c.used+=r.bytes;++c.reads;c.bytes+=r.bytes;request.pointer=output;
                c.release_locked(output); // Drop only the preload's own lease.
            }
            request.done=true;c.loads.erase(r.key);request.ready.notify_all();
        }
        c.changed.notify_all();
    };
    std::shared_ptr<batch> prepared;
    std::future<size_t> reader;
    const auto start=std::chrono::steady_clock::now();auto progress=start;
    try {
        size_t next=0,batches=0;
        while(next<planes.size() && !result.budget_limited) {
            prepared=std::make_shared<batch>();prepared->reserve(128);uint64_t allocated=0;
            while(next<planes.size() && prepared->size()<128 && (prepared->empty() || allocated<batch_limit)) {
                const auto plane=planes[next];
                std::unique_lock<std::mutex> lock(c.mutex);
                const auto key=plane.pointer;
                auto it=c.entries.find(key);
                if(it!=c.entries.end()) {
                    ++c.hits;c.ages.splice(c.ages.end(),c.ages,it->second.age);++result.planes;++next;continue;
                }
                auto loading=c.loads.find(key);
                if(loading!=c.loads.end()) {
                    auto request=loading->second;++request->users;
                    request->ready.wait(lock,[&]{return request->done;});
                    if(request->error)std::rethrow_exception(request->error);
                    ++c.hits;c.release_locked(request->pointer);++result.planes;++next;continue;
                }
                auto source_it=c.sources.upper_bound(key);
                if(source_it==c.sources.begin())throw std::runtime_error("missing preload source");
                auto * owner=std::prev(source_it)->second;
                if(owner->closing)throw std::runtime_error("preload source is closing");
                auto span_it=owner->spans.upper_bound(key);
                if(span_it==owner->spans.begin())throw std::runtime_error("missing preload span");
                const auto & span=std::prev(span_it)->second;
                const size_t local=key-span.begin;
                if(local>=span.bytes || plane.bytes!=span.expert || local%span.expert || plane.bytes>span.bytes-local)
                    throw std::runtime_error("invalid preload expert plane");
                const uint64_t offset=span.file_offset(local);
                const size_t delta=offset%alignment,total=rounded(delta+plane.bytes);
                const uint64_t cap=c.limit();
                if(total>cap || c.used>cap-total || c.reserved>cap-total-c.used){result.budget_limited=true;break;}
                if(!prepared->empty() && total>batch_limit-std::min(allocated,batch_limit))break;
                auto request=std::make_shared<load>();request->owner=owner;request->routing=span.routing;
                request->expert=int(local/span.expert);
                c.loads.emplace(key,request);c.reserved+=total;
                prepared->push_back({key,request,span.fd,offset-delta,delta,total,delta+plane.bytes});
                lock.unlock();
                // The preceding batch's reader is in flight while these pages
                // are allocated/populated. Both batches count against the cap.
                request->storage=std::make_unique<region>(total);request->storage->populate();
                lock.lock();request->populated=true;c.populated_reserved+=total;
                allocated+=total;++next;
            }
            if(reader.valid())result.planes+=reader.get();
            if(!prepared->empty()) {
                const auto work=prepared;
                reader=std::async(std::launch::async,[&,work] {
                    try {read_batch(work->data(),work->size());}
                    catch(...){complete(work,std::current_exception());throw;}
                    complete(work,nullptr);return work->size();
                });
                prepared.reset();++batches;
            }
            const auto now=std::chrono::steady_clock::now();
            if(now-progress>=std::chrono::seconds(5)) {
                std::lock_guard<std::mutex> lock(c.mutex);
                std::fprintf(stderr,"[dsv4f-ram-preload] planes=%llu resident=%.2f GiB read=%.2f GiB pipeline_batches=%zu\n",
                    (unsigned long long)result.planes,c.used/double(1ull<<30),(c.bytes-initial_bytes)/double(1ull<<30),batches);
                progress=now;
            }
        }
        if(reader.valid())result.planes+=reader.get();
    } catch(...) {
        const auto error=std::current_exception();
        if(reader.valid())try{reader.get();}catch(...){}
        if(prepared && !prepared->empty())complete(prepared,error);
        throw;
    }
}

void acquire_many_impl(const void * const * pointers,const size_t * sizes,const void ** output,size_t count,bool scan) {
    if(!count)return;
    // Preserve the lock-held diagnostic mode, and keep single-plane calls cheap.
    if(!scan && (count==1 || !unlocked_io())) {
        size_t held=0;
        try{for(;held<count;++held)output[held]=acquire(pointers[held],sizes[held]);}
        catch(...){while(held)release(output[--held]);throw;}
        return;
    }
    struct ticket {const void * pointer=nullptr;std::shared_ptr<load> request;};
    std::vector<ticket> tickets(count);std::vector<batch_read> reads;reads.reserve(count);
    auto & c=pool();std::unique_lock<std::mutex> lock(c.mutex);
    const uint64_t cap=c.limit();
    struct source_guard {
        cache & c;
        std::vector<source*> owners;
        source_guard(cache & p,size_t n):c(p){owners.reserve(n);}
        void retain(source * p){if(std::find(owners.begin(),owners.end(),p)==owners.end()){owners.push_back(p);++p->prefetches;}}
        ~source_guard(){for(auto * p:owners){GGML_ASSERT(p->prefetches);--p->prefetches;}c.changed.notify_all();}
    } sources(c,count);
    uint64_t stream_room=0,resident_cap=cap;
    if(scan) {
        uint64_t batch_bytes=0;
        for(size_t i=0;i<count;++i) {
            if(sizes[i]>SIZE_MAX-alignment)throw std::runtime_error("prefill plane size overflow");
            const size_t n=rounded(sizes[i]+alignment-1);
            if(n>cap || batch_bytes>cap-n)throw std::runtime_error("prefill group exceeds RAM cache budget");
            batch_bytes+=n;
        }
        // Keep a bounded streaming window INSIDE the expert cache allowance.
        // Two groups cover the staging ring; ordinary group=16 needs <1 GiB.
        stream_room=std::min(cap,std::max(2*batch_bytes,std::min<uint64_t>(1024*MiB,cap/4)));
        resident_cap=cap-stream_room;
        // Reclaim only when existing residency exceeds the new allowance
        // (e.g. first prefill after decode, a smaller budget, larger group/KV).
        while(c.used-c.transient_used+c.reserved-c.transient_reserved>resident_cap) {
            auto victim=c.victim(true);
            if(victim==c.ages.end())break;
            c.erase(c.entries.find(*victim));
        }
    }
    auto rollback=[&] {
        for(auto & t:tickets) {
            if(t.request && !t.request->done){GGML_ASSERT(t.request->users);--t.request->users;continue;}
            const void * p=t.request?t.request->pointer:t.pointer;
            c.release_locked(p);
        }
        c.changed.notify_all();
    };
    auto fail=[&](const batch_read & r,const std::exception_ptr & error) {
        r.request->error=error;r.request->storage.reset();r.request->done=true;
        if(r.request->transient)c.transient_reserved-=r.bytes;
        c.reserved-=r.bytes;c.loads.erase(r.key);r.request->ready.notify_all();
    };
    try {
        for(size_t i=0;i<count;++i) {
            const auto key=(uintptr_t)pointers[i];const size_t n=sizes[i];
            auto s=c.sources.upper_bound(key);
            if(s==c.sources.begin()){tickets[i].pointer=pointers[i];continue;}
            --s;source * owner=s->second;
            if(key-s->first>=owner->address->bytes){tickets[i].pointer=pointers[i];continue;}
            if(owner->closing)throw std::runtime_error("direct weight source is closing");
            sources.retain(owner);
            auto sp=owner->spans.upper_bound(key);
            if(sp==owner->spans.begin())throw std::runtime_error("invalid direct weight address");
            --sp;const auto & span=sp->second;const size_t local=key-span.begin;
            if(local>=span.bytes || n!=span.expert || local%span.expert || n>span.bytes-local)
                throw std::runtime_error("batch weights must be complete expert planes");
            auto it=c.entries.find(key);
            if(it!=c.entries.end()) {
                ++it->second.users;++c.hits;c.ages.splice(c.ages.end(),c.ages,it->second.age);
                tickets[i].pointer=it->second.pointer;continue;
            }
            auto loading=c.loads.find(key);
            if(loading!=c.loads.end()) {
                tickets[i].request=loading->second;++loading->second->users;++c.hits;continue;
            }
            const uint64_t offset=span.file_offset(local);
            const size_t delta=offset%alignment,total=rounded(delta+n);
            if(total>cap)throw std::runtime_error("RAM budget has no room for a batched expert plane");
            while(c.used>cap-total || c.reserved>cap-total-c.used) {
                if(!scan) {
                    auto victim=c.victim();
                    if(victim==c.ages.end())throw std::runtime_error("RAM expert batch is exhausted by active leases");
                    c.erase(c.entries.find(*victim));
                } else {
                    if(c.used-c.transient_used+c.reserved-c.transient_reserved>resident_cap) {
                        auto victim=c.victim(true);
                        if(victim!=c.ages.end()){c.erase(c.entries.find(*victim));continue;}
                    }
                    // Previous groups' copy callbacks can free transient space.
                    // Do not evict the resident scan set to admit these misses.
                    if(c.changed.wait_for(lock,std::chrono::seconds(30))==std::cv_status::timeout)
                        throw std::runtime_error("prefill streaming window exhausted by active leases");
                    if(owner->closing)throw std::runtime_error("direct weight source is closing");
                }
            }
            // A callback/demand thread can acquire this key while a scan waits
            // for streaming space. Recheck before registering a new load.
            it=c.entries.find(key);
            if(it!=c.entries.end()) {
                ++it->second.users;++c.hits;c.ages.splice(c.ages.end(),c.ages,it->second.age);
                tickets[i].pointer=it->second.pointer;continue;
            }
            loading=c.loads.find(key);
            if(loading!=c.loads.end()) {
                tickets[i].request=loading->second;++loading->second->users;++c.hits;continue;
            }
            auto request=std::make_shared<load>();request->storage=std::make_unique<region>(total);request->owner=owner;
            request->routing=span.routing;request->expert=int(local/span.expert);
            request->transient=scan && c.used-c.transient_used+c.reserved-c.transient_reserved+total>resident_cap;
            c.loads.emplace(key,request);c.reserved+=total;tickets[i].request=request;
            if(request->transient)c.transient_reserved+=total;
            reads.push_back({key,request,span.fd,offset-delta,delta,total,delta+n});
        }
    } catch(...) {
        const auto error=std::current_exception();for(const auto & r:reads)fail(r,error);
        rollback();throw;
    }
    std::exception_ptr error;
    lock.unlock();
    try {for(size_t i=0;i<reads.size();i+=128)read_batch(reads.data()+i,std::min<size_t>(128,reads.size()-i));}
    catch(...){error=std::current_exception();}
    lock.lock();
    for(const auto & r:reads) {
        if(error){fail(r,error);continue;}
        const void * p=(char*)r.request->storage->data+r.delta;c.ages.push_back(r.key);
        entry e;e.storage=std::move(r.request->storage);e.pointer=p;e.users=r.request->users;e.bytes=r.required-r.delta;
        e.owner=r.request->owner;e.routing=r.request->routing;e.expert=r.request->expert;e.age=std::prev(c.ages.end());e.transient=r.request->transient;
        c.entries.emplace(r.key,std::move(e));c.leases.emplace(p,r.key);
        c.reserved-=r.bytes;c.used+=r.bytes;++c.reads;c.bytes+=r.bytes;
        if(r.request->transient){c.transient_reserved-=r.bytes;c.transient_used+=r.bytes;++c.transient_reads;c.transient_bytes+=r.bytes;}
        r.request->pointer=p;r.request->done=true;c.loads.erase(r.key);r.request->ready.notify_all();
        if(c.reads%4096==0)std::fprintf(stderr,"[dsv4f-ram] resident=%.2f MiB hit=%llu reads=%llu direct_mib=%.2f evictions=%llu\n",c.used/double(MiB),(unsigned long long)c.hits,(unsigned long long)c.reads,c.bytes/double(MiB),(unsigned long long)c.evictions);
    }
    if(error){rollback();std::rethrow_exception(error);}
    for(auto & t:tickets)if(t.request) {
        t.request->ready.wait(lock,[&]{return t.request->done;});
        if(t.request->error){rollback();std::rethrow_exception(t.request->error);}
    }
    for(size_t i=0;i<count;++i)output[i]=tickets[i].request?tickets[i].request->pointer:tickets[i].pointer;
    c.changed.notify_all();
}
void acquire_many(const void * const * p,const size_t * n,const void ** out,size_t count) {
    acquire_many_impl(p,n,out,count,false);
}
void acquire_prefill(const void * const * p,const size_t * n,const void ** out,size_t count) {
    acquire_many_impl(p,n,out,count,true);
}
void statistics(ggml_dsv4f_host_weights_stats * out) {
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    *out={c.used-c.transient_used,c.transient_used,c.reads,c.bytes,c.evictions,c.transient_reads,c.transient_bytes};
}
void resident_many(const void * const * pointers,const size_t * sizes,bool * output,size_t count) {
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    for(size_t i=0;i<count;++i) {
        auto it=c.entries.find((uintptr_t)pointers[i]);
        output[i]=it!=c.entries.end() && !it->second.transient && it->second.bytes==sizes[i];
    }
}
void discard(const void * pointer,size_t bytes) {
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    auto it=c.entries.find((uintptr_t)pointer);
    if(it==c.entries.end() || it->second.bytes!=bytes)return;
    if(!it->second.users)c.erase(it);
    else if(!it->second.transient) {
        it->second.transient=true;c.transient_used+=it->second.storage->bytes;
    }
    c.changed.notify_all();
}
void free_buffer(ggml_backend_buffer_t buffer) {
    auto * owner=(source*)buffer->context;
    auto & c=pool();
    {std::unique_lock<std::mutex> lock(c.mutex);owner->closing=true;
        for(auto it=c.prefetch_queue.begin();it!=c.prefetch_queue.end();) {
            if(it->owner==owner) {
                c.prefetch_pending.erase(it->key);GGML_ASSERT(owner->prefetches);
                --owner->prefetches;++c.prefetch_dropped;it=c.prefetch_queue.erase(it);
            } else ++it;
        }
        c.changed.wait(lock,[&] {
            if(owner->prefetches)return false;
            for(const auto & item:c.loads)if(item.second->owner==owner)return false;
            for(const auto & item:c.entries)if(item.second.owner==owner && item.second.users)return false;
            return true;
        });
        for(auto it=c.entries.begin();it!=c.entries.end();) {
            if(it->second.owner==owner){auto old=it++;c.erase(old);}else ++it;
        }
        c.sources.erase((uintptr_t)owner->address->data);
        if(c.sources.empty()) {
            std::fprintf(stderr,"[dsv4f-ram-prefetch] queued=%llu completed=%llu dropped=%llu failed=%llu\n",
                (unsigned long long)c.prefetch_queued,(unsigned long long)c.prefetch_completed,
                (unsigned long long)c.prefetch_dropped,(unsigned long long)c.prefetch_failed);
            ggml_dsv4f_host_weights={};
        }
    }
    delete owner;
}
void * base(ggml_backend_buffer_t b){return ((source*)b->context)->address->data;}
void get_tensor(ggml_backend_buffer_t,const ggml_tensor * t,void * output,size_t off,size_t n) {
    // Unexpected generic tensor uploads cannot dereference the protected descriptor address.
    size_t done=0;
    while(done<n) {
        const size_t eid=(off+done)/t->nb[2],delta=(off+done)%t->nb[2];
        ggml_dsv4f_host_lease lease((char*)t->data+eid*t->nb[2],t->nb[2]);
        const size_t count=std::min(n-done,t->nb[2]-delta);
        std::memcpy((char*)output+done,(const char*)lease.data+delta,count);done+=count;
    }
}
}
uint64_t dsv4f_ram_budget(){return env_mib("DSV4F_RAM_BUDGET_MIB",0);}
bool dsv4f_direct_weights_enabled(){return dsv4f_ram_budget()!=0;}
uint64_t dsv4f_load_pipeline_bytes() {
    const uint64_t bytes=env_mib("DSV4F_LOAD_PIPELINE_MIB",1024*MiB);
    if(bytes>65536*MiB)throw std::runtime_error("DSV4F_LOAD_PIPELINE_MIB must be an integer in 0..65536");
    return bytes;
}
dsv4f_ram_reservation::dsv4f_ram_reservation(uint64_t requested) {
    const uint64_t budget=dsv4f_ram_budget();
    if(!budget || !requested){acquired=true;return;}
    const uint64_t reserve=env_mib("DSV4F_RAM_RESERVE_MIB",1024*MiB);
    if(budget<=reserve || requested>budget-reserve)return;
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    if(c.runtime_reserved>budget-reserve-requested)return;
    const uint64_t pending=requested+c.runtime_reserved;
    if(c.reserved>budget-reserve-pending)return;
    const uint64_t needed=pending+c.reserved;
    uint64_t reclaimed=0;
    for(;;) {
        const uint64_t room=runtime_headroom(budget,reserve);
        if(room>=needed)break;
        const uint64_t deficit=needed-room;uint64_t freed=0;
        while(freed<deficit) {
            const auto victim=c.victim();
            if(victim==c.ages.end()) {
                std::fprintf(stderr,"[dsv4f-ram-reserve] denied: request=%.2f MiB reclaimed=%.2f MiB; no idle experts\n",
                    requested/double(MiB),reclaimed/double(MiB));
                return;
            }
            const auto entry=c.entries.find(*victim);freed+=entry->second.storage->bytes;c.erase(entry);
        }
        reclaimed+=freed;
    }
    c.runtime_reserved+=requested;bytes=requested;acquired=true;
    std::fprintf(stderr,"[dsv4f-ram-reserve] request=%.2f MiB reclaimed=%.2f MiB reserve=%.2f MiB\n",
        requested/double(MiB),reclaimed/double(MiB),reserve/double(MiB));
}
dsv4f_ram_reservation::~dsv4f_ram_reservation() {
    if(!bytes)return;
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    GGML_ASSERT(c.runtime_reserved>=bytes);c.runtime_reserved-=bytes;c.changed.notify_all();
}
dsv4f_preload_result dsv4f_preload_experts() {
    dsv4f_preload_result result;
    const char * enabled=std::getenv("DSV4F_RAM_PRELOAD");
    if(!enabled || !std::strcmp(enabled,"0"))return result;
    if(std::strcmp(enabled,"1"))throw std::runtime_error("DSV4F_RAM_PRELOAD must be 0 or 1");
    if(!dsv4f_direct_weights_enabled())return result;
    auto & c=pool();
    struct source_hold {
        cache & c;
        std::vector<source*> owners;
        ~source_hold() {
            std::lock_guard<std::mutex> lock(c.mutex);
            for(auto * s:owners){GGML_ASSERT(s->prefetches);--s->prefetches;}
            c.changed.notify_all();
        }
    } hold{c,{}};
    struct ordered_span { uint64_t sequence;source::span span; };
    std::vector<ordered_span> spans;
    uint64_t initial_bytes=0;
    {
        std::lock_guard<std::mutex> lock(c.mutex);
        for(const auto & item:c.sources) {
            auto * s=item.second;if(s->closing)continue;
            hold.owners.push_back(s);++s->prefetches;
            for(const auto & item:s->spans)spans.push_back({s->sequence,item.second});
        }
        if(spans.empty())return result;
        initial_bytes=c.bytes;
        std::fprintf(stderr,"[dsv4f-ram-preload] begin: resident=%.2f GiB ceiling=%.2f GiB order=%s O_DIRECT\n",
            c.used/double(1ull<<30),c.limit()/double(1ull<<30),ggml_dsv4f_routing.count?"routing-count,shard,offset":"shard,offset");
    }
    std::sort(spans.begin(),spans.end(),[](const ordered_span & a,const ordered_span & b) {
        return std::tie(a.sequence,a.span.file_index,a.span.offset)<std::tie(b.sequence,b.span.file_index,b.span.offset);
    });
    std::vector<preload_plane> planes;
    for(const auto & item:spans) {
        const auto & span=item.span;
        if(!span.expert || span.bytes%span.expert)throw std::runtime_error("invalid expert geometry for startup preload");
        for(size_t at=0;at<span.bytes;at+=span.expert)
            planes.push_back({span.begin+at,span.expert,ggml_dsv4f_routing_count(span.routing,int(at/span.expert))});
    }
    std::stable_sort(planes.begin(),planes.end(),[](const preload_plane & a,const preload_plane & b){return a.count>b.count;});
    const auto start=std::chrono::steady_clock::now();auto progress=start;
    const uint64_t batch_limit=dsv4f_load_pipeline_bytes();
    if(batch_limit && unlocked_io()) {
        std::fprintf(stderr,"[dsv4f-ram-preload] pipelined allocation/read: batch=%.0f MiB, max_planes=128, depth=2\n",batch_limit/double(MiB));
        preload_pipelined(planes,result,batch_limit,initial_bytes);
    } else for(const auto & plane:planes) {
        const void * p=acquire_impl((const void*)plane.pointer,plane.bytes,true);
        if(!p){result.budget_limited=true;break;}
        release(p);++result.planes;
        const auto now=std::chrono::steady_clock::now();
        if(now-progress>=std::chrono::seconds(5)) {
            std::lock_guard<std::mutex> lock(c.mutex);
            std::fprintf(stderr,"[dsv4f-ram-preload] planes=%llu resident=%.2f GiB read=%.2f GiB\n",
                (unsigned long long)result.planes,c.used/double(1ull<<30),(c.bytes-initial_bytes)/double(1ull<<30));
            progress=now;
        }
    }
    {
        std::lock_guard<std::mutex> lock(c.mutex);
        result.read_bytes=c.bytes-initial_bytes;result.resident_bytes=c.used;
        std::fprintf(stderr,"[dsv4f-ram-preload] complete: planes=%llu read=%.2f GiB resident=%.2f GiB ceiling=%.2f GiB stop=%s elapsed=%.2f s\n",
            (unsigned long long)result.planes,result.read_bytes/double(1ull<<30),result.resident_bytes/double(1ull<<30),
            c.limit()/double(1ull<<30),result.budget_limited?"budget":"all-experts",
            std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    }
    return result;
}
void dsv4f_direct_read(int fd,void * output,size_t n,uint64_t off) {
    if(!n)return;
    const size_t delta=off%alignment;
    thread_local std::unique_ptr<region> scratch;
    const size_t count=rounded(delta+n);
    if(!scratch || scratch->bytes<count)scratch=std::make_unique<region>(count);
    read_aligned(fd,scratch->data,count,off-delta,delta+n);
    std::memcpy(output,(char*)scratch->data+delta,n);
}
void dsv4f_direct_read_many(const dsv4f_direct_request * requests,size_t count) {
    if(!count)return;
    if(count==1){dsv4f_direct_read(requests[0].fd,requests[0].output,requests[0].bytes,requests[0].offset);return;}
#if defined(__linux__) && defined(__NR_io_setup) && defined(__NR_io_submit) && defined(__NR_io_getevents) && defined(__NR_io_destroy)
    struct aio_state {
        aio_context_t context=0;
        aio_state(){if(syscall(__NR_io_setup,128,&context)<0)context=0;}
        ~aio_state(){if(context)syscall(__NR_io_destroy,context);}
    };
    thread_local aio_state aio;
    if(aio.context && count<=128) {
        size_t total=0;
        std::vector<size_t> positions(count),deltas(count),sizes(count);
        for(size_t i=0;i<count;++i) {
            deltas[i]=requests[i].offset%alignment;sizes[i]=rounded(deltas[i]+requests[i].bytes);
            positions[i]=total;if(total>SIZE_MAX-sizes[i])throw std::runtime_error("direct I/O batch size overflow");total+=sizes[i];
        }
        thread_local std::unique_ptr<region> batch_storage;
        if(!batch_storage || batch_storage->bytes<total)batch_storage=std::make_unique<region>(total);
        auto * storage=batch_storage.get();
        std::vector<iocb> blocks(count);std::vector<iocb*> pointers(count);
        for(size_t i=0;i<count;++i) {
            auto & b=blocks[i];b.aio_data=i;b.aio_lio_opcode=IOCB_CMD_PREAD;b.aio_fildes=requests[i].fd;
            b.aio_buf=(uintptr_t)((char*)storage->data+positions[i]);b.aio_nbytes=sizes[i];
            b.aio_offset=requests[i].offset-deltas[i];pointers[i]=&b;
        }
        size_t submitted=0;int submit_error=0;
        while(submitted<count) {
            const long n=syscall(__NR_io_submit,aio.context,count-submitted,pointers.data()+submitted);
            if(n>0){submitted+=size_t(n);continue;}
            if(n<0 && errno==EINTR)continue;
            submit_error=n<0?errno:EIO;break;
        }
        if(submitted) {
            std::vector<io_event> events(submitted);size_t completed=0;
            while(completed<submitted) {
                const long n=syscall(__NR_io_getevents,aio.context,1,submitted-completed,events.data()+completed,nullptr);
                if(n>0){completed+=size_t(n);continue;}
                if(n<0 && errno==EINTR)continue;
                throw std::runtime_error(std::string("O_DIRECT batch completion failed: ")+std::strerror(errno));
            }
            for(const auto & event:events) {
                const size_t i=event.data;
                if(i>=count || event.res<0 || uint64_t(event.res)<deltas[i]+requests[i].bytes)
                    throw std::runtime_error(event.res<0?std::string("O_DIRECT batch read failed: ")+std::strerror(int(-event.res)):"truncated direct I/O batch payload");
            }
            if(submit_error || submitted!=count)
                throw std::runtime_error(std::string("O_DIRECT batch submit failed: ")+std::strerror(submit_error?submit_error:EIO));
            for(size_t i=0;i<count;++i)std::memcpy(requests[i].output,(char*)storage->data+positions[i]+deltas[i],requests[i].bytes);
            static std::once_flag native_log;std::call_once(native_log,[&]{std::fprintf(stderr,"[dsv4f-ram] O_DIRECT batch reader: linux-aio depth=%zu\n",count);});
            return;
        }
    }
#endif
    static std::once_flag fallback_log;std::call_once(fallback_log,[]{std::fprintf(stderr,"[dsv4f-ram] O_DIRECT batch reader unavailable; using sequential pread\n");});
    for(size_t i=0;i<count;++i)dsv4f_direct_read(requests[i].fd,requests[i].output,requests[i].bytes,requests[i].offset);
}
ggml_backend_buffer_t dsv4f_direct_weights_buffer(const std::vector<dsv4f_direct_tensor> & tensors) {
    auto owner=std::make_unique<source>();size_t bytes=0;
    for(const auto & t:tensors)bytes+=rounded(ggml_nbytes(t.tensor));
    owner->address=std::make_unique<region>(bytes,PROT_NONE);
    ggml_backend_buffer_i iface{};iface.free_buffer=free_buffer;iface.get_base=base;iface.get_tensor=get_tensor;
    std::map<int,int> opened;size_t at=0;
    // Prepare everything that can fail before registering the source.
    for(const auto & t:tensors) {
        // Entire quantization tiers can disappear after pruning. They have no
        // source span, but still receive a descriptor address below.
        if(!ggml_nbytes(t.tensor))continue;
        if(!t.expert_ids.empty()) {
            if(t.expert_ids.size()!=size_t(t.tensor->ne[2]))throw std::runtime_error("invalid direct expert map size");
            int32_t previous=-1;
            for(int32_t id:t.expert_ids) {
                if(id<=previous)throw std::runtime_error("invalid direct expert map order");
                previous=id;
            }
        }
        if(!opened.count(t.fd)) {
            const std::string path="/proc/self/fd/"+std::to_string(t.fd);
            const int fd=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);
            if(fd<0)throw std::runtime_error(std::string("cannot open O_DIRECT weight source: ")+std::strerror(errno));
            owner->fds.push_back(fd);opened[t.fd]=fd;
        }
        const auto key=(uintptr_t)owner->address->data+at;
        owner->spans.emplace(key,source::span{key,ggml_nbytes(t.tensor),t.tensor->nb[2],opened.at(t.fd),t.offset,t.file_index,ggml_dsv4f_routing_bind(t.tensor),t.expert_ids});
        at+=rounded(ggml_nbytes(t.tensor));
    }
    auto & c=pool();
    std::lock_guard<std::mutex> lock(c.mutex);
    (void)c.limit();
    auto * b=ggml_backend_buffer_init(ggml_backend_cpu_buffer_type(),iface,owner.get(),owner->address->bytes);
    at=0;for(const auto & t:tensors){ggml_backend_tensor_alloc(b,t.tensor,(char*)owner->address->data+at);at+=rounded(ggml_nbytes(t.tensor));}
    c.sources.emplace((uintptr_t)owner->address->data,owner.get());
    owner->sequence=c.next_source++;
    ggml_dsv4f_host_weights.acquire=acquire;ggml_dsv4f_host_weights.release=release;ggml_dsv4f_host_weights.prefetch=prefetch;
    ggml_dsv4f_host_weights.acquire_many=acquire_many;
    ggml_dsv4f_host_weights.acquire_prefill=acquire_prefill;ggml_dsv4f_host_weights.statistics=statistics;
    ggml_dsv4f_host_weights.resident_many=resident_many;ggml_dsv4f_host_weights.discard=discard;
    std::fprintf(stderr,"[dsv4f-ram] O_DIRECT expert source: %.2f GiB on SSD, process budget %.2f GiB, cache ceiling %.2f GiB, io_lock=%s\n",bytes/double(1ull<<30),dsv4f_ram_budget()/double(1ull<<30),c.limit()/double(1ull<<30),unlocked_io()?"released":"held");
    owner.release();return b;
}
