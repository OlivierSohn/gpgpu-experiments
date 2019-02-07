#include "cplx.c"

#define N_LOCAL_BUTTERFLIES replace_N_LOCAL_BUTTERFLIES // must be a power of 2

__kernel void kernel_func(__local struct cplx* output, __global const float *input, __global const struct cplx *twiddle, __global struct cplx *global_output) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;
  
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = 2*base_idx + j;
    output[m] = complexFromReal(input[m]);
  }

  int const n_global_butterflies = get_global_size(0) * N_LOCAL_BUTTERFLIES;
  
  for(int i=1; i<=n_global_butterflies; i <<= 1)
  {
    // For the first iterations, there is no need for a memory barrier
    // because we only use memory locations where our thread has written to.
    if(i>N_LOCAL_BUTTERFLIES) {
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    
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
  
  // We write back what we computed on the last iteration of the loop
  // so we don't need a barrier
  //
  // Note that we could merge this writeback with the last iteration of the previous
  // loop, to save some local memory writes and reads, but profiling shows that the
  // performance is worse. Maybe because this leads to writing global memory not in a
  // linear way?
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = 2*base_idx + j;
    global_output[m] = output[m];
  }
}
