// Generalization of 'vector_fft_floats.cl' : here, each thread
// handles 2^n butterflies at each fft level, instead of
// a single butterfly in 'vector_fft_floats.cl'.

#define N_LOCAL_BUTTERFLIES replaceThisBeforeCompiling // must be a power of 2

// There is no support for complex in OpenCL so I make my own:
struct cplx {
  float real;
  float imag;
};

inline struct cplx complexFromReal(float r) {
  struct cplx c;
  c.real = r;
  c.imag = 0.f;
  return c;
}

inline struct cplx cplxMult(struct cplx const a, struct cplx const b) {
  struct cplx r;
  
  r.real = b.real * a.real - b.imag * a.imag;
  r.imag = b.real * a.imag + b.imag * a.real;
  
  return r;
}

inline struct cplx cplxSub(struct cplx const a, struct cplx const b) {
  struct cplx r;
  
  r.real = a.real - b.real;
  r.imag = a.imag - b.imag;
  
  return r;
}

inline void cplxAddAssign(__global struct cplx * a, struct cplx const b) {
  a->real += b.real;
  a->imag += b.imag;
}

inline void butterfly(__global struct cplx *v, int i,  struct cplx twiddle) {
  struct cplx const t = cplxMult(v[i], twiddle);
  v[i] = cplxSub(v[0], t);
  cplxAddAssign(&v[0], t);
}

__kernel void kernel_func(__global const float *input, __global const struct cplx *twiddle, __global struct cplx *output) {
  int const k = get_global_id(0);
  int const base_idx = k * N_LOCAL_BUTTERFLIES;
  
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = 2*base_idx + j;
    output[m] = complexFromReal(input[m]);
  }
  
  barrier(CLK_GLOBAL_MEM_FENCE);
  
  int const n_global_butterflies = get_global_size(0) * N_LOCAL_BUTTERFLIES;

  for(int i=1; i<=n_global_butterflies; i <<= 1)
  {
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m = base_idx + j;
      int const tmp = i*(m/i);
      int const idx = tmp + m;

      //assert(idx+i < Sz);
      int const ri = m - tmp;
      int const tIdx = ri*(n_global_butterflies/i); // TODO use shifts, i is a power of 2
      
      butterfly(output+idx, i, twiddle[tIdx]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}
