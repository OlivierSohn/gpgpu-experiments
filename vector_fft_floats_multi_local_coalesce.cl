#include "cplx.c"

#define N_LOCAL_BUTTERFLIES replace_N_LOCAL_BUTTERFLIES // must be a power of 2

__kernel void kernel_func(__local struct cplx* output, __global const float *input, __global const struct cplx *twiddle, __global struct cplx *global_output) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;
  
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    output[m] = complexFromReal(input[m]);
  }

  int const n_global_butterflies = get_global_size(0) * N_LOCAL_BUTTERFLIES;
  
  // at each cycle, a bank can only provide access to a single adress.
  for(int i=1; i<=n_global_butterflies; i <<= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m = base_idx + j;
      int const tmp = i*(m/i);
      int const idx = tmp + m;

      //assert(idx+i < Sz);
      int const ri = m - tmp;
      int const tIdx = ri*(n_global_butterflies/i);
      
      butterfly(output+idx, i, twiddle[tIdx]);
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory write
    global_output[m] = output[m];
  }
}
