#include "cplx.c"

#define N_LOCAL_BUTTERFLIES       replace_N_LOCAL_BUTTERFLIES // must be a power of 2
#define N_GLOBAL_BUTTERFLIES      replace_N_GLOBAL_BUTTERFLIES // must be a power of 2, and >= N_LOCAL_BUTTERFLIES
#define LOG2_N_GLOBAL_BUTTERFLIES replace_LOG2_N_GLOBAL_BUTTERFLIES
#define MINUS_PI_over_N_GLOBAL_BUTTERFLIES replace_MINUS_PI_over_N_GLOBAL_BUTTERFLIES

__constant sampler_t input_sampler =
  CLK_NORMALIZED_COORDS_FALSE |
  CLK_ADDRESS_NONE |
  CLK_FILTER_NEAREST;



inline int expand(int idxL, int log2N1, int mm) {
  return ((idxL-mm) << 1) + mm;
}

__kernel void kernel_func(__read_only image2d_t input_image, // real floats
                          __write_only image2d_t output_image, // cplx floats
                          __local struct cplx* pingpong) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;

  __local struct cplx *prev = pingpong;
  __local struct cplx *next = pingpong + 2*N_GLOBAL_BUTTERFLIES;

  {
    if(N_LOCAL_BUTTERFLIES==1) { // half the threads will read.
      if((k&1) == 0) {
        int const m = k/2;
        // coalesced global memory read, local memory write with some bank conflicts.
        float4 f = read_imagef(input_image, input_sampler, (int2)(m,0));
        prev[4*m+0] = complexFromReal(f[0]);
        prev[4*m+1] = complexFromReal(f[1]);
        prev[4*m+2] = complexFromReal(f[2]);
        prev[4*m+3] = complexFromReal(f[3]);
      }
    }
    else { // all threads read
      for(int j=0; j<N_LOCAL_BUTTERFLIES/2; ++j) {
        int const m = get_global_size(0) * j + k;
        // coalesced global memory read, local memory write with some bank conflicts.
        float4 f = read_imagef(input_image, input_sampler, (int2)(m,0));
        prev[4*m+0] = complexFromReal(f[0]);
        prev[4*m+1] = complexFromReal(f[1]);
        prev[4*m+2] = complexFromReal(f[2]);
        prev[4*m+3] = complexFromReal(f[3]);
      }
    }
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

  for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory write
    write_imagef(output_image,
                (int2)(m,0),
                (float4)(prev[2*m].real,
                         prev[2*m].imag,
                         prev[2*m+1].real,
                         prev[2*m+1].imag)
                );
  }
}
