
#include "cplx.c"

#define N_LOCAL_BUTTERFLIES       replace_N_LOCAL_BUTTERFLIES // must be a power of 2
#define N_GLOBAL_BUTTERFLIES      replace_N_GLOBAL_BUTTERFLIES // must be a power of 2, and >= N_LOCAL_BUTTERFLIES
#define LOG2_N_GLOBAL_BUTTERFLIES replace_LOG2_N_GLOBAL_BUTTERFLIES
#define MINUS_PI_over_N_GLOBAL_BUTTERFLIES replace_MINUS_PI_over_N_GLOBAL_BUTTERFLIES
#define INPUT_SIZE                replace_INPUT_SIZE // must be = 2*N_GLOBAL_BUTTERFLIES
#define IMAG_OFFSET               INPUT_SIZE

inline void butterfly_with_writeback(int const idx,
                                     __global float *g, // we will write back the results to this
                                     const __local struct cplx * const v,
                                     int const i,
                                     const struct cplx twiddle) {
  struct cplx const t = cplxMult(v[idx+i], twiddle);

  g[idx]               = v[idx].real + t.real;
  g[idx+i]             = v[idx].real - t.real;
  g[idx+IMAG_OFFSET ]  = v[idx].imag + t.imag;
  g[idx+i+IMAG_OFFSET] = v[idx].imag - t.imag;
}

__kernel void kernel_func(__local struct cplx * output, // cplx buffer
                          __global const float * input, // real buffer
                          __global float * global_output // real buffer, imag buffer
                          ) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;

  // copy real part
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    output[m]            = complexFromReal(input[m]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // i = size of a butterfly half
  // LOG2_N_GLOBAL_BUTTERFLIES_over_i = log2(N_GLOBAL_BUTTERFLIES / i)
  for(int i=1, LOG2_N_GLOBAL_BUTTERFLIES_over_i = LOG2_N_GLOBAL_BUTTERFLIES;
      i <= N_GLOBAL_BUTTERFLIES;
      i <<= 1, --LOG2_N_GLOBAL_BUTTERFLIES_over_i)
  {
    // During the first iterations, there is no need for synchronisation
    // because we only use memory locations that our thread has written to.
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
      
      if(i == N_GLOBAL_BUTTERFLIES) {
        butterfly_with_writeback(idx,global_output, output, i, polar(tIdx * MINUS_PI_over_N_GLOBAL_BUTTERFLIES));
      }
      else {
        butterfly(output+idx, i, polar(tIdx * MINUS_PI_over_N_GLOBAL_BUTTERFLIES));
      }
    }
  }
  
  // we omit the write back which has been done in the last iteration
  /*
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory write
    global_output[m] = output[m].real;
    global_output[m+IMAG_OFFSET] = output[m].imag;
  }*/
/*  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory write
    global_output[m+IMAG_OFFSET] = output[m].imag;
  }*/
}
