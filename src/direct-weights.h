#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include <cstddef>
#include <vector>

// Zero/unset keeps the original resident loader. Positive budgets enable Linux direct I/O.
uint64_t dsv4f_ram_budget();
bool dsv4f_direct_weights_enabled();
// Shared resident/preload allocation-read pipeline batch ceiling; 0 disables.
// At least one tensor/plane is processed, even when larger than the ceiling.
uint64_t dsv4f_load_pipeline_bytes();
// Temporarily reserve headroom for a runtime allocation by reclaiming idle
// expert planes. Keeps background loaders from refilling the reclaimed space
// until the allocation is resident. False means the optional allocation must
// be skipped. A zero RAM budget makes this a successful no-op.
class dsv4f_ram_reservation {
public:
    explicit dsv4f_ram_reservation(uint64_t bytes);
    ~dsv4f_ram_reservation();
    explicit operator bool() const {return acquired;}
    dsv4f_ram_reservation(const dsv4f_ram_reservation &)=delete;
    dsv4f_ram_reservation & operator=(const dsv4f_ram_reservation &)=delete;
private:
    uint64_t bytes=0;
    bool acquired=false;
};
struct dsv4f_direct_tensor {
    ggml_tensor * tensor;
    int fd;
    uint64_t offset;
    uint32_t file_index=0;
};
ggml_backend_buffer_t dsv4f_direct_weights_buffer(const std::vector<dsv4f_direct_tensor> & tensors);
struct dsv4f_preload_result {
    uint64_t planes=0,read_bytes=0,resident_bytes=0;
    bool budget_limited=false;
};
// Call after runtime buffers/warmup, before accepting requests. Fill remaining
// cache space by descending routing count (source/shard/offset breaks ties),
// without evicting existing entries. Disabled routing retains physical order.
// Disabled unless DSV4F_RAM_PRELOAD=1; a zero RAM budget is also a no-op.
// Payloads use the ordinary O_DIRECT cache path. A nonzero
// DSV4F_LOAD_PIPELINE_MIB overlaps the next batch's population with reads;
// both pending batches count against the cache ceiling.
dsv4f_preload_result dsv4f_preload_experts();
// Aligned O_DIRECT payload read, including unaligned tensor/row offsets and the last file sector.
void dsv4f_direct_read(int fd, void * output, size_t bytes, uint64_t offset);
struct dsv4f_direct_request { int fd; void * output; size_t bytes; uint64_t offset; };
// Submit independent aligned O_DIRECT reads together. Falls back to the
// synchronous reader when native Linux AIO is unavailable.
void dsv4f_direct_read_many(const dsv4f_direct_request * requests, size_t count);
