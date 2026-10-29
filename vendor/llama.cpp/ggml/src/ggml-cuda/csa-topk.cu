#include "fattn-mma-f16.cuh"
void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context &, ggml_tensor *);

// Reuse the validated D512/32-column FP32 accumulator layout. Here all 32
// columns are heads of one query, rather than eight heads of four queries.
void ggml_cuda_csa_topk_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool prefill) {
#if defined(GGML_USE_HIP)
    if (prefill && GGML_CUDA_CC_IS_RDNA3_0(ggml_cuda_info().devices[ctx.device].cc) && dst->src[0]->ne[2]%32==0) {
        ggml_cuda_flash_attn_ext_mma_f16_case<512,512,1,32>(ctx,dst);
        return;
    }
#endif
    ggml_cuda_flash_attn_ext_tile(ctx,dst);
}
