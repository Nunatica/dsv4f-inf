#pragma once
#include <cstddef>

// Several weight rows share one quantized Q8_0 activation. The original MXFP4
// bytes, scale semantics and two-block accumulation order are retained.
bool dsv4f_mxfp4_rows_supported();
const char * dsv4f_mxfp4_rows_isa();
using dsv4f_mxfp4_kernel=void (*)(int,const void * const *,const void *,float *);
dsv4f_mxfp4_kernel dsv4f_mxfp4_kernel_for(int rows);
void dsv4f_mxfp4_rows(int n,const void * const * weights,const void * activation,float * output,int rows);
