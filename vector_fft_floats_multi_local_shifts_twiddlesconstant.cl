
#include "cplx.c"

#define N_LOCAL_BUTTERFLIES       replace_N_LOCAL_BUTTERFLIES // must be a power of 2
#define LOG2_N_GLOBAL_BUTTERFLIES replace_LOG2_N_GLOBAL_BUTTERFLIES

__kernel void kernel_func(__local struct cplx* output, __global const float *input, __constant const struct cplx *twiddle, __global struct cplx *global_output) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;
  
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = 2*base_idx + j;
    output[m] = complexFromReal(input[m]);
  }

  int const n_global_butterflies = get_global_size(0) * N_LOCAL_BUTTERFLIES;
  
  // i = size of a butterfly half
  // LOG2_N_GLOBAL_BUTTERFLIES_over_i = log2(n_global_butterflies / i)
  for(int i=1, LOG2_N_GLOBAL_BUTTERFLIES_over_i = LOG2_N_GLOBAL_BUTTERFLIES;
      i <= n_global_butterflies;
      i <<= 1, --LOG2_N_GLOBAL_BUTTERFLIES_over_i)
  {
    // During the first iterations, there is no need for synchronisation
    // because we only use memory locations where our thread has written to.
    if(i>N_LOCAL_BUTTERFLIES) {
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m = base_idx + j;
      //int const ri = m & (i-1);// m - tmp;
      //int const tmp = m - ri;// m & ~(i-1); // (m >> iPow) << iPow; // = i*(m/i)
      
      // if N_LOCAL_BUTTERFLIES <= i, we know that m & ~(i-1) == base_idx & ~(i-1)
      // so in that case, we could compute that part out of the loop.
      int const idx = m + (m & ~(i-1));//2*m - ri;//tmp + m;

      //assert(idx+i < Sz);
      int const tIdx = (m & (i-1)) << LOG2_N_GLOBAL_BUTTERFLIES_over_i;
      
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
