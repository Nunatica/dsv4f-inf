#include "mxfp4.h"
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DSV4F_AVX2 __attribute__((target("avx2,fma,f16c")))
#define DSV4F_AVX512 __attribute__((target("avx2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma,f16c")))

// The AVX2 path also runs on AVX-512 systems without requiring those extensions.
// Reuse activation loads and half conversions across two/four independent rows.
template<int N> DSV4F_AVX2 static void dot_rows(int n,const void * const * pointers,const void * activation,float * result) {
    const auto * y=static_cast<const block_q8_0 *>(activation);
    const block_mxfp4 * x[N];for(int r=0;r<N;++r)x[r]=static_cast<const block_mxfp4 *>(pointers[r]);
    const __m128i values=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const __m128i mask=_mm_set1_epi8(15);const __m256i ones=_mm256_set1_epi16(1);
    __m256 even[N],odd[N];for(int r=0;r<N;++r)even[r]=odd[r]=_mm256_setzero_ps();
    const int blocks=n/32;int b=0;
    for(;b+1<blocks;b+=2) {
        const __m256i q8[2]={_mm256_loadu_si256((const __m256i *)y[b].qs),_mm256_loadu_si256((const __m256i *)y[b+1].qs)};
        const float dy[2]={_cvtsh_ss(y[b].d),_cvtsh_ss(y[b+1].d)};
        for(int r=0;r<N;++r)for(int j=0;j<2;++j) {
            const __m128i bits=_mm_loadu_si128((const __m128i *)x[r][b+j].qs);
            const __m128i lo=_mm_shuffle_epi8(values,_mm_and_si128(bits,mask));
            const __m128i hi=_mm_shuffle_epi8(values,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m256i q4=_mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);
            const __m256i product=_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(q4,q4),_mm256_sign_epi8(q8[j],q4)),ones);
            const __m256 scale=_mm256_set1_ps(dy[j]*ggml_e8m0_to_fp32_half(x[r][b+j].e));
            if(j)odd[r]=_mm256_fmadd_ps(scale,_mm256_cvtepi32_ps(product),odd[r]);
            else even[r]=_mm256_fmadd_ps(scale,_mm256_cvtepi32_ps(product),even[r]);
        }
    }
    for(int r=0;r<N;++r) {
        const __m256 sum=_mm256_add_ps(even[r],odd[r]);
        __m128 v=_mm_add_ps(_mm256_extractf128_ps(sum,1),_mm256_castps256_ps128(sum));
        v=_mm_add_ps(v,_mm_movehl_ps(v,v));v=_mm_add_ss(v,_mm_movehdup_ps(v));
        result[r]=_mm_cvtss_f32(v);
        // Match the reference scalar tail for an odd number of 32-value blocks.
        if(b<blocks) {
            static constexpr int8_t table[]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
            int a=0,c=0;for(int j=0;j<16;++j){a+=y[b].qs[j]*table[x[r][b].qs[j]&15];c+=y[b].qs[j+16]*table[x[r][b].qs[j]>>4];}
            result[r]+=_cvtsh_ss(y[b].d)*ggml_e8m0_to_fp32_half(x[r][b].e)*(a+c);
        }
    }
}
template<int N> DSV4F_AVX512 static void dot_rows512(int n,const void * const * pointers,const void * activation,float * result) {
    const auto * y=static_cast<const block_q8_0 *>(activation);
    const block_mxfp4 * x[N];for(int r=0;r<N;++r)x[r]=static_cast<const block_mxfp4 *>(pointers[r]);
    const __m128i table=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const __m512i values=_mm512_broadcast_i32x4(table),zero=_mm512_setzero_si512();
    const __m256i mask=_mm256_set1_epi8(15);
    __m512 accum[N];for(int r=0;r<N;++r)accum[r]=_mm512_setzero_ps();
    const int blocks=n/32;int b=0;
    for(;b+1<blocks;b+=2) {
        const __m512i q8=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)y[b].qs)),_mm256_loadu_si256((const __m256i *)y[b+1].qs),1);
        const float dy0=_cvtsh_ss(y[b].d),dy1=_cvtsh_ss(y[b+1].d);
        for(int r=0;r<N;++r) {
            const __m256i bits=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)x[r][b].qs)),_mm_loadu_si128((const __m128i *)x[r][b+1].qs),1);
            const __m256i lo=_mm256_and_si256(bits,mask),hi=_mm256_and_si256(_mm256_srli_epi16(bits,4),mask);
            const __m512i nibbles=_mm512_shuffle_i64x2(_mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1),_mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1),0xd8);
            const __m512i q4=_mm512_shuffle_epi8(values,nibbles);
            const __m512i signed_y=_mm512_mask_sub_epi8(q8,_mm512_movepi8_mask(q4),zero,q8);
            const __m512i dot=_mm512_dpbusd_epi32(zero,_mm512_abs_epi8(q4),signed_y);
            const float d0=dy0*ggml_e8m0_to_fp32_half(x[r][b].e),d1=dy1*ggml_e8m0_to_fp32_half(x[r][b+1].e);
            const __m512 scale=_mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(d0)),_mm256_set1_ps(d1),1);
            accum[r]=_mm512_fmadd_ps(scale,_mm512_cvtepi32_ps(dot),accum[r]);
        }
    }
    for(int r=0;r<N;++r) {
        const __m256 sum=_mm256_add_ps(_mm512_castps512_ps256(accum[r]),_mm512_extractf32x8_ps(accum[r],1));
        __m128 v=_mm_add_ps(_mm256_extractf128_ps(sum,1),_mm256_castps256_ps128(sum));
        v=_mm_add_ps(v,_mm_movehl_ps(v,v));v=_mm_add_ss(v,_mm_movehdup_ps(v));result[r]=_mm_cvtss_f32(v);
        if(b<blocks) {
            static constexpr int8_t lookup[]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
            int a=0,c=0;for(int j=0;j<16;++j){a+=y[b].qs[j]*lookup[x[r][b].qs[j]&15];c+=y[b].qs[j+16]*lookup[x[r][b].qs[j]>>4];}
            result[r]+=_cvtsh_ss(y[b].d)*ggml_e8m0_to_fp32_half(x[r][b].e)*(a+c);
        }
    }
}
bool dsv4f_mxfp4_rows_supported(){return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");}
const char * dsv4f_mxfp4_rows_isa() {
    static const char * selected=[] {
        const char * value=std::getenv("DSV4F_CPU_MXFP4_ISA");if(!value || !std::strcmp(value,"avx2"))return "avx2";
        if(std::strcmp(value,"avx512") && std::strcmp(value,"auto"))throw std::runtime_error("DSV4F_CPU_MXFP4_ISA must be avx2, avx512 or auto");
        if(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vnni"))return "avx512";
        std::fprintf(stderr,"[dsv4f-cpu] AVX-512 VNNI unavailable; using AVX2 multi-row kernel\n");return "avx2";
    }();return selected;
}
dsv4f_mxfp4_kernel dsv4f_mxfp4_kernel_for(int rows) {
    static const dsv4f_mxfp4_kernel avx2[]={dot_rows<1>,dot_rows<2>,dot_rows<3>,dot_rows<4>};
    static const dsv4f_mxfp4_kernel avx512[]={dot_rows512<1>,dot_rows512<2>,dot_rows512<3>,dot_rows512<4>};
    static const auto selected=std::strcmp(dsv4f_mxfp4_rows_isa(),"avx512")?avx2:avx512;
    GGML_ASSERT(rows>=1 && rows<=4);return selected[rows-1];
}
#else
bool dsv4f_mxfp4_rows_supported(){return false;}
const char * dsv4f_mxfp4_rows_isa(){return "unavailable";}
dsv4f_mxfp4_kernel dsv4f_mxfp4_kernel_for(int){GGML_ABORT("MXFP4 multi-row kernel is unavailable");}
#endif
void dsv4f_mxfp4_rows(int n,const void * const * weights,const void * activation,float * output,int rows) {
    GGML_ASSERT(n%32==0);dsv4f_mxfp4_kernel_for(rows)(n,weights,activation,output);
}
