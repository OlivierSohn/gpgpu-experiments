#include "cplx.c"

#define N_LOCAL_BUTTERFLIES replace_N_LOCAL_BUTTERFLIES // must be a power of 2


inline int expand(int idxL, int N1, int N2) {
  return (idxL/N1)*N1*N2 + (idxL%N1);
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

  
  // at each cycle, a bank can only provide access to a single adress.
  for(int i=1; i<=n_global_butterflies; i <<= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m = base_idx + j;
      
      int idxD = expand(m, i, 2);
      
      butterfly_outofplace(m,idxD,prev,next, n_global_butterflies, i,
                           twiddle[(m%i)*(n_global_butterflies/i)]);
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
