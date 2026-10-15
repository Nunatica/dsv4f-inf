#include "dsv4f-kv.cuh"
#include "set-rows.cuh"
#include <memory>

void ggml_cuda_dsv4f_kv_reset(void * ptr) {
    auto & s = *(dsv4f_kv_state *) ptr;
    ggml_cuda_set_device(s.device);
    // Lifecycle boundaries run outside capture. Order reset with every graph
    // stream before host-side restore/copy can change authoritative storage.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(s.map, 0xff, size_t(s.rows)*s.streams*sizeof(int)));
    CUDA_CHECK(cudaMemset(s.owner, 0xff, s.slots*sizeof(int)));
    CUDA_CHECK(cudaMemset(s.stamp, 0, s.slots*sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(s.ref, 0, s.slots*sizeof(int)));
    CUDA_CHECK(cudaMemset(s.ctl, 0, 2*sizeof(unsigned)));
    CUDA_CHECK(cudaDeviceSynchronize());
}
size_t ggml_cuda_dsv4f_kv_bytes(void * ptr) {
    auto & s = *(dsv4f_kv_state *) ptr;
    return s.slots*(s.row_bytes + 3*sizeof(int)) + size_t(s.rows)*s.streams*sizeof(int) + 2*sizeof(unsigned);
}
void ggml_cuda_dsv4f_kv_destroy(void * ptr) {
    auto * s = (dsv4f_kv_state *) ptr;
    ggml_cuda_set_device(s->device);
    CUDA_CHECK(cudaFree(s->cache));
    CUDA_CHECK(cudaFree(s->map));
    CUDA_CHECK(cudaFree(s->owner));
    CUDA_CHECK(cudaFree(s->stamp));
    CUDA_CHECK(cudaFree(s->ref));
    CUDA_CHECK(cudaFree(s->ctl));
    delete s;
}
void * ggml_cuda_dsv4f_kv_create(int device, ggml_tensor * host, uint32_t slots) {
    GGML_ASSERT(host->ne[0] == 512 && (host->type == GGML_TYPE_Q8_0 || host->type == GGML_TYPE_F16));
    ggml_cuda_set_device(device);
    auto s = std::make_unique<dsv4f_kv_state>();
    s->device = device; s->rows = host->ne[1]; s->streams = host->ne[2];
    s->slots = slots; s->type = host->type; s->row_bytes = host->nb[1];
    s->host_cpu = (char *) host->data;
    // Host allocators can fall back to pageable memory: fail explicitly in that
    // case instead of passing an inaccessible host address to a captured kernel.
    const auto err = cudaHostGetDevicePointer((void **) &s->host, host->data, 0);
    if (err != cudaSuccess) {
        GGML_ABORT("KV streaming needs pinned mapped host storage (unset GGML_CUDA_NO_PINNED)");
    }
    CUDA_CHECK(cudaMalloc((void **) &s->cache, s->slots*s->row_bytes));
    CUDA_CHECK(cudaMalloc((void **) &s->map, size_t(s->rows)*s->streams*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &s->owner, s->slots*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &s->stamp, s->slots*sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc((void **) &s->ref, s->slots*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &s->ctl, 2*sizeof(unsigned)));
    ggml_cuda_dsv4f_kv_reset(s.get());
    return s.release();
}
static __global__ void kv_write_through(const int64_t * ids, const char * host, char * cache,
        const int * map, int rows, int bytes) {
    const int row = int(ids[blockIdx.x]);
    assert(row >= 0 && row < rows);
    const int slot = map[row];
    if (slot >= 0) {
        for (int i = threadIdx.x; i < bytes; i += blockDim.x) {
            cache[int64_t(slot)*bytes + i] = host[int64_t(row)*bytes + i];
        }
    }
}
void ggml_cuda_dsv4f_kv_write(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto & s = *(dsv4f_kv_state *) ggml_dsv4f_kv_state(dst);
    GGML_ASSERT(ctx.device == s.device);
    const auto * rows = dst->src[0];
    const auto * ids = dst->src[1];
    GGML_ASSERT(rows->type == GGML_TYPE_F32 && rows->ne[0] == 512 && ids->type == GGML_TYPE_I64);
    // Reuse the exact same quantization kernel as resident caches, including its
    // rounding rules. The destination here is GPU-mapped host memory.
    ggml_tensor target = *rows;
    target.type = s.type;
    target.ne[1] = int64_t(s.rows)*s.streams;
    target.ne[2] = target.ne[3] = 1;
    target.nb[0] = ggml_type_size(s.type);
    target.nb[1] = s.row_bytes;
    target.nb[2] = target.nb[3] = target.nb[1]*target.ne[1];
    target.data = s.host;
    target.src[0] = dst->src[0]; target.src[1] = dst->src[1];
    ggml_cuda_op_set_rows(ctx, &target);
    // Kernel completion orders host writes before the write-through reads.
    kv_write_through<<<ids->ne[0], 256, 0, ctx.stream()>>>((const int64_t *) ids->data,
        s.host, s.cache, s.map, s.rows*s.streams, int(s.row_bytes));
    CUDA_CHECK(cudaGetLastError());
}
