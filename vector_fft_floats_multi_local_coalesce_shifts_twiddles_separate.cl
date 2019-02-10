
#include "cplx.c"

#define N_LOCAL_BUTTERFLIES       replace_N_LOCAL_BUTTERFLIES // must be a power of 2
#define N_GLOBAL_BUTTERFLIES      replace_N_GLOBAL_BUTTERFLIES // must be a power of 2, and >= N_LOCAL_BUTTERFLIES
#define LOG2_N_GLOBAL_BUTTERFLIES replace_LOG2_N_GLOBAL_BUTTERFLIES
#define MINUS_PI_over_N_GLOBAL_BUTTERFLIES replace_MINUS_PI_over_N_GLOBAL_BUTTERFLIES
#define INPUT_SIZE                replace_INPUT_SIZE // must be = 2*N_GLOBAL_BUTTERFLIES
#define IMAG_OFFSET               INPUT_SIZE

////////////////////////////////////////////////////////////////////
// Functions used when performing the butterfly on local memory with
// a separate representation for complex numbers
////////////////////////////////////////////////////////////////////

inline void butterfly_separate(__local float*v,
                               int const i,
                               const struct cplx twiddle) {
  //struct cplx const t = cplxMult(v[i], twiddle); // x
  
  //v[i] = cplxSub(v[0], t); // a
  //cplxAddAssign(&v[0], t); // b
  
  float tReal = v[i] * twiddle.real - v[i+IMAG_OFFSET] * twiddle.imag; // x
  float tImag = v[i] * twiddle.imag + v[i+IMAG_OFFSET] * twiddle.real; // x
  v[i]             = v[0]             - tReal; // a
  v[0]             += tReal;                   // b
  v[i+IMAG_OFFSET] = v[0+IMAG_OFFSET] - tImag; // a
  v[0+IMAG_OFFSET] += tImag;                   // b
  
}


inline void butterfly_with_writeback(int const idx,
                                     __global float *g, // we will write back the results to this
                                     const __local float * const v,
                                     int const i,
                                     const struct cplx twiddle) {
//  struct cplx const t = cplxMult(v[idx+i], twiddle); // x
  float tReal = v[idx+i] * twiddle.real - v[idx+i+IMAG_OFFSET] * twiddle.imag; // x
  float tImag = v[idx+i] * twiddle.imag + v[idx+i+IMAG_OFFSET] * twiddle.real; // x

  g[idx]               = v[idx] + tReal;
  g[idx+i]             = v[idx] - tReal;
  g[idx+IMAG_OFFSET ]  = v[idx+IMAG_OFFSET] + tImag;
  g[idx+i+IMAG_OFFSET] = v[idx+IMAG_OFFSET] - tImag;
}

__kernel void kernel_func(__local float * output, // real buffer, imag buffer
                          __global const float * input, // real buffer
                          __global float * global_output // real buffer, imag buffer
                          ) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;

  // copy real part
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    output[m]            = input[m];
//    output[m+IMAG_OFFSET] = 0.f;
  }
  // copy imaginary part
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // local memory write with no bank conflict.
    output[m+IMAG_OFFSET] = 0.f;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // i = size of a butterfly half
  // LOG2_N_GLOBAL_BUTTERFLIES_over_i = log2(N_GLOBAL_BUTTERFLIES / i)
  for(int i=1, LOG2_N_GLOBAL_BUTTERFLIES_over_i = LOG2_N_GLOBAL_BUTTERFLIES;
      i <= N_GLOBAL_BUTTERFLIES;
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
      
      if(i == N_GLOBAL_BUTTERFLIES) {
        butterfly_with_writeback(idx,global_output, output, i, polar(tIdx * MINUS_PI_over_N_GLOBAL_BUTTERFLIES));
      }
      else {
        butterfly_separate(output+idx, i, polar(tIdx * MINUS_PI_over_N_GLOBAL_BUTTERFLIES));
      }
    }
  }
  
/*  barrier(CLK_LOCAL_MEM_FENCE);

  for(int j=0; j<4*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_global_size(0) * j + k;
    // coalesced global memory write
    global_output[m] = output[m];
  }*/
}
