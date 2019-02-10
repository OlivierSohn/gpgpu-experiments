#include "cplx.c"

#define N_LOCAL_BUTTERFLIES       replace_N_LOCAL_BUTTERFLIES // must be a power of 2
#define N_GLOBAL_BUTTERFLIES      replace_N_GLOBAL_BUTTERFLIES // must be a power of 2, and >= N_LOCAL_BUTTERFLIES
#define LOG2_N_GLOBAL_BUTTERFLIES replace_LOG2_N_GLOBAL_BUTTERFLIES
#define MINUS_PI_over_N_GLOBAL_BUTTERFLIES replace_MINUS_PI_over_N_GLOBAL_BUTTERFLIES


inline int expand(int idxL, int log2N1, int mm) {
  return ((idxL-mm) << 1) + mm;
}

__kernel void kernel_func(__global const float *input,
                          __global struct cplx *global_output,
                          __local struct cplx* pingpong) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;

  __local struct cplx *prev = pingpong;
  __local struct cplx *next = pingpong + 2*N_GLOBAL_BUTTERFLIES;

  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    prev[m] = complexFromReal(input[m]);
  }

  for(int i=1, LOG2_N_GLOBAL_BUTTERFLIES_over_i = LOG2_N_GLOBAL_BUTTERFLIES, log2i = 0;
      i <= N_GLOBAL_BUTTERFLIES;
      i <<= 1, --LOG2_N_GLOBAL_BUTTERFLIES_over_i, ++log2i)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m = base_idx + j;
      int const mm = m & (i-1);
      
      int idxD = expand(m, log2i, mm);
      int const tIdx = mm << LOG2_N_GLOBAL_BUTTERFLIES_over_i;
      
      butterfly_outofplace(m,idxD,prev,next, N_GLOBAL_BUTTERFLIES, i,
                           polar(tIdx * MINUS_PI_over_N_GLOBAL_BUTTERFLIES));
    }

    // swap(prev,next)
    {
      __local struct cplx * tmp = prev;
      prev = next;
      next = tmp;
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory write
    global_output[m] = prev[m];
  }
}
