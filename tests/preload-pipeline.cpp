#include "direct-weights.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-cpu.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
constexpr size_t plane=1<<20;
std::atomic<bool> enabled{false};
std::mutex mutex;
std::condition_variable changed;
bool second_allocation=false,reading=false,overlap=false;
int allocations=0;
void require(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
}
// Interpose only the startup test's single-plane operations. No production
// hooks are needed: the first reader waits for the next physical allocation.
extern "C" int madvise(void * address,size_t size,int advice) noexcept {
#if defined(MADV_POPULATE_WRITE)
    if(enabled && advice==MADV_POPULATE_WRITE && size==plane) {
        std::unique_lock<std::mutex> lock(mutex);
        if(++allocations==2) {
            second_allocation=true;changed.notify_all();
            if(!changed.wait_for(lock,std::chrono::seconds(5),[]{return reading;})){errno=ETIMEDOUT;return -1;}
        }
    }
#endif
    return syscall(SYS_madvise,address,size,advice);
}
extern "C" ssize_t pread(int fd,void * data,size_t size,off_t offset) {
    if(enabled && size==plane && offset==0) {
        std::unique_lock<std::mutex> lock(mutex);reading=true;changed.notify_all();
        if(!changed.wait_for(lock,std::chrono::seconds(5),[]{return second_allocation;})){errno=ETIMEDOUT;return -1;}
        overlap=true;
    }
    return syscall(SYS_pread64,fd,data,size,offset);
}

int main() {
    try {
        setenv("DSV4F_RAM_BUDGET_MIB","131072",1);setenv("DSV4F_RAM_CACHE_MIB","8",1);
        setenv("DSV4F_RAM_RESERVE_MIB","4",1);setenv("DSV4F_RAM_UNLOCK_IO","1",1);setenv("DSV4F_RAM_PRELOAD","1",1);
        char path[]="/tmp/dsv4f-preload-pipeline-XXXXXX";const int fd=mkstemp(path);require(fd>=0,"fixture file");
        std::vector<unsigned char> reference(3*plane);
        for(size_t i=0;i<reference.size();++i)reference[i]=(i*17+i/65536)%251;
        require(write(fd,reference.data(),reference.size())==ssize_t(reference.size()),"write fixture");require(!fsync(fd),"sync fixture");
        for(const char * batch:{"1","0"}) {
            setenv("DSV4F_LOAD_PIPELINE_MIB",batch,1);
            auto * ctx=ggml_init({1<<20,nullptr,true});auto * t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,512,512,3);
            auto * buffer=dsv4f_direct_weights_buffer({{t,fd,0}});
#if defined(MADV_POPULATE_WRITE)
            enabled=batch[0]=='1';
#endif
            const auto result=dsv4f_preload_experts();enabled=false;
            require(result.planes==3 && !result.budget_limited && result.read_bytes==3*plane && result.resident_bytes==3*plane,"preload accounting");
#if defined(MADV_POPULATE_WRITE)
            if(batch[0]=='1')require(overlap,"allocation did not overlap the reader");
#endif
            for(int i=0;i<3;++i){ggml_dsv4f_host_lease lease((char*)t->data+i*plane,plane);require(!std::memcmp(lease.data,reference.data()+i*plane,plane),"preload byte parity");}
            ggml_backend_buffer_free(buffer);ggml_free(ctx);
        }
        // A failed second batch must drain I/O and release all reservations so
        // unloading the source cannot hang and a new model can preload again.
        setenv("DSV4F_LOAD_PIPELINE_MIB","1",1);
        require(!ftruncate(fd,plane+17),"truncate fixture");
        auto * ctx=ggml_init({1<<20,nullptr,true});auto * t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,512,512,3);
        auto * buffer=dsv4f_direct_weights_buffer({{t,fd,0}});bool failed=false;
        try{dsv4f_preload_experts();}catch(const std::runtime_error &){failed=true;}
        require(failed,"truncated preload accepted");ggml_backend_buffer_free(buffer);ggml_free(ctx);
        require(!ftruncate(fd,reference.size()),"restore fixture");require(pwrite(fd,reference.data(),reference.size(),0)==ssize_t(reference.size()),"rewrite fixture");
        ctx=ggml_init({1<<20,nullptr,true});t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,512,512,3);buffer=dsv4f_direct_weights_buffer({{t,fd,0}});
        const auto recovered=dsv4f_preload_experts();require(recovered.planes==3 && recovered.resident_bytes==3*plane,"recovery leaked memory/reservations");
        ggml_backend_buffer_free(buffer);ggml_free(ctx);close(fd);unlink(path);
        std::puts("PASS preload allocation/read overlap, bytes, sequential fallback, error drain and recovery");
    } catch(const std::exception & e){enabled=false;std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
