#include "cplx.c"

#define N_LOCAL_BUTTERFLIES       replace_N_LOCAL_BUTTERFLIES // must be a power of 2
#define LOG2_N_GLOBAL_BUTTERFLIES replace_LOG2_N_GLOBAL_BUTTERFLIES


inline int expand(int idxL, int log2N1, int mm) {
  return ((idxL-mm) << 1) + mm;
}

__kernel void kernel_func(__global const float *input,
                          __global const struct cplx *twiddle,
                          __global struct cplx *global_output,
                          __local struct cplx* pingpong) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;
  int const n_global_butterflies = get_global_size(0) * N_LOCAL_BUTTERFLIES;

  __local struct cplx *prev = pingpong;
  __local struct cplx *next = pingpong + 2*n_global_butterflies;

  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    prev[m] = complexFromReal(input[m]);
  }

  for(int i=1, LOG2_N_GLOBAL_BUTTERFLIES_over_i = LOG2_N_GLOBAL_BUTTERFLIES, log2i = 0;
      i <= n_global_butterflies;
      i <<= 1, --LOG2_N_GLOBAL_BUTTERFLIES_over_i, ++log2i)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m = base_idx + j;
      int const mm = m & (i-1);
      
      int idxD = expand(m, log2i, mm);
      
      butterfly_outofplace(m,idxD,prev,next, n_global_butterflies, i,
                           twiddle[mm << LOG2_N_GLOBAL_BUTTERFLIES_over_i]);
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
