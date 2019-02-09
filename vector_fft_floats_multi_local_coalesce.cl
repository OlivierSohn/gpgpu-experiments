#include "cplx.c"

#define N_LOCAL_BUTTERFLIES replace_N_LOCAL_BUTTERFLIES // must be a power of 2

/*
 Future work:
 - To restore the barrier omission for first levels, we can change the arrangement
   of the input so that, in case get_global_size(0)==4, indexes are:
       0  4 8 12  1 5 9 13  2 6 10 14  3 7 11 15
 LEVEL1:
 t0    1          2         3          4
 t1       1         2         3          4
 ..
 LEVEL2:
 t0    1          3         2          4
 t1       1         3         2          4
 ..
 LEVEL3:
 t0    1  2       3 4
 t1
 
 n_global_butterflies = 8
 
   This way, our thread will have written to local memory all the memory locations
   used for the first levels.
   Note that this will also allow to remove the last barrier.
 */
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
