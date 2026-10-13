#include "mxfp4.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main() {
    ggml_cpu_init();if(!dsv4f_mxfp4_rows_supported()){std::puts("multi-row AVX2 unavailable; fallback selected");return 0;}
    const auto * dot=ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);const auto * quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::mt19937 rng(392);std::normal_distribution<float> normal(0,.7f);double worst=0;int different=0;
    for(int n:{32,64,96,256,2048,4096})for(int trial=0;trial<16;++trial) {
        std::vector<float> x(n);for(auto & v:x)v=trial==0?0:normal(rng);
        std::vector<uint8_t> activation(ggml_row_size(GGML_TYPE_Q8_0,n)+3);quant->from_float(x.data(),activation.data()+3,n);
        std::vector<uint8_t> weight[4];const void * pointers[4];float reference[4];
        for(int r=0;r<4;++r){for(auto & v:x)v=normal(rng)*(r+1);weight[r].resize(ggml_row_size(GGML_TYPE_MXFP4,n)+5);pointers[r]=weight[r].data()+5;ggml_quantize_chunk(GGML_TYPE_MXFP4,x.data(),weight[r].data()+5,0,1,n,nullptr);dot->vec_dot(n,reference+r,0,pointers[r],0,activation.data()+3,0,1);}
        for(int rows=1;rows<=4;++rows){float result[4];dsv4f_mxfp4_rows(n,pointers,activation.data()+3,result,rows);
            for(int r=0;r<rows;++r){const double error=std::abs(result[r]-reference[r]);worst=std::max(worst,error);different+=std::memcmp(result+r,reference+r,sizeof(float))!=0;
                if(!std::isfinite(result[r]) || error>1e-6*std::max(1.f,std::abs(reference[r]))){std::fprintf(stderr,"FAIL n=%d rows=%d reference=%g result=%g\n",n,rows,reference[r],result[r]);return 1;}}
        }
    }
    std::printf("MXFP4 multi-row: odd/even block counts, unaligned rows, zero/random input; max_error=%g differing=%d PASS\n",worst,different);
}
