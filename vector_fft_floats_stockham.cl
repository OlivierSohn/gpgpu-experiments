#include "cplx.c"

inline int expand(int idxL, int N1, int N2) {
  return (idxL/N1)*N1*N2 + (idxL%N1);
}

__kernel void kernel_func(__global const float *input,
                          __global const struct cplx *twiddle,
                          __global struct cplx *output,
                          __local struct cplx *pingpong) {
  
  // Get the index of the current element to be processed
  int const k = get_global_id(0); // we use a single dimension hence we pass 0
  
  int const n_global_butterflies = get_global_size(0);
  int const szInput = n_global_butterflies * 2;

  __local struct cplx *prev = pingpong;
  __local struct cplx *next = pingpong+szInput;

  prev[2*k]   = complexFromReal(input[2*k]);
  prev[2*k+1] = complexFromReal(input[2*k+1]);

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int i=1; i<=n_global_butterflies; i <<= 1) {

    //assert(k+n_global_butterflies < Sz);
    
    int idxD = expand(k, i, 2);

    butterfly_outofplace(k,idxD,prev,next, n_global_butterflies, i, twiddle[(k%i)*(n_global_butterflies/i)]);

    // swap(prev,next)
    {
      __local struct cplx * tmp = prev;
      prev = next;
      next = tmp;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  // writeback
  output[2*k]   = prev[2*k];
  output[2*k+1] = prev[2*k+1];
}
