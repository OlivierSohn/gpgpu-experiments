
#include "cplx.c"

#define N_LOCAL_BUTTERFLIES       replace_N_LOCAL_BUTTERFLIES // must be a power of 2
#define N_GLOBAL_BUTTERFLIES      replace_N_GLOBAL_BUTTERFLIES // must be a power of 2, and >= N_LOCAL_BUTTERFLIES
#define LOG2_N_GLOBAL_BUTTERFLIES replace_LOG2_N_GLOBAL_BUTTERFLIES
#define MINUS_PI_over_N_GLOBAL_BUTTERFLIES replace_MINUS_PI_over_N_GLOBAL_BUTTERFLIES

__kernel void first_butterflies(__local struct cplx* output,
                                __global const float *input,
                                __global struct cplx *global_output) {
  int const k = get_local_id(0);
  
  int const workgroupIdxOffset = 2 * N_LOCAL_BUTTERFLIES * get_local_size(0) * get_group_id(0);
  int const lastSz = (2 * N_GLOBAL_BUTTERFLIES) / get_num_groups(0);

  // we copy a contiguous chunk of input to local memory
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_local_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    output[m] = complexFromReal(input[workgroupIdxOffset+m]);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  int const global_base_idx = get_global_id(0) * N_LOCAL_BUTTERFLIES;

  // i = size of a butterfly half
  // LOG2_N_GLOBAL_BUTTERFLIES_over_i = log2(N_GLOBAL_BUTTERFLIES / i)
  for(int i=1, LOG2_N_GLOBAL_BUTTERFLIES_over_i = LOG2_N_GLOBAL_BUTTERFLIES;
      i <= lastSz;
      i <<= 1, --LOG2_N_GLOBAL_BUTTERFLIES_over_i)
  {
    // During the first iterations, there is no need for synchronisation
    // because we only use memory locations where our thread has written to.
    if(i>N_LOCAL_BUTTERFLIES) {
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m = global_base_idx + j;
      
      int const idx = m + (m & ~(i-1));

      int const tIdx = (m & (i-1)) << LOG2_N_GLOBAL_BUTTERFLIES_over_i;
      
      butterfly(output+idx-workgroupIdxOffset, i, polar(tIdx * MINUS_PI_over_N_GLOBAL_BUTTERFLIES));
    }
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // we copy contiguous chunks of input to local memory
  for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_local_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    global_output[workgroupIdxOffset+m] = output[m];
  }
}


/*
 Precondition: '2*N_LOCAL_BUTTERFLIES' is a multiple of 'get_num_groups(0)'
 */
__kernel void last_butterflies(__local struct cplx* output,
                               __global struct cplx *global_output) {
  int const nGroups = get_num_groups(0);
  int const group = get_group_id(0);
  int const k = get_local_id(0);
  
  int const group_sz = 2 * N_LOCAL_BUTTERFLIES * get_local_size(0);
  int const workgroup_offset_idx = (2 * N_LOCAL_BUTTERFLIES * get_local_size(0) * group) / nGroups;
  
  // we copy contiguous chunks of input to local memory
  for(int g=0; g<nGroups; ++g)
  {
    int const base_global_index = 2 * N_LOCAL_BUTTERFLIES * get_local_size(0) * g;
    for(int j=0; j<(2*N_LOCAL_BUTTERFLIES)/nGroups; ++j)
    {
      int const m = base_global_index + workgroup_offset_idx + get_local_size(0) * j + k;
      // coalesced global memory read, local memory write with no bank conflict.
      output[base_global_index/nGroups + get_local_size(0) * j + k] = global_output[m];
    }
  }
  
  int const firstSz = (2 * N_GLOBAL_BUTTERFLIES) / nGroups; // 8192
  int const firstLocalSz = (2 * N_LOCAL_BUTTERFLIES * get_local_size(0)) / nGroups; // 2048

  // i = size of a butterfly half
  // LOG2_N_GLOBAL_BUTTERFLIES_over_i = log2(N_GLOBAL_BUTTERFLIES / i)
  for(int i=firstSz,
      N_GLOBAL_BUTTERFLIES_over_i = get_num_groups(0)/2,
      i_local=firstLocalSz;
      
      i <= N_GLOBAL_BUTTERFLIES;
      
      i <<= 1, i_local <<= 1,
      N_GLOBAL_BUTTERFLIES_over_i >>= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m_local = k*N_LOCAL_BUTTERFLIES + j;
      
      int const idx = m_local + (m_local & ~(i_local-1));
      
      // TODO use shifts
      int const countFirstLocalSz = idx / firstLocalSz;
      int const remainder = idx - countFirstLocalSz * firstLocalSz;
      int const m = countFirstLocalSz * group_sz + group * firstLocalSz + remainder;
      
      int const tIdx = (m & (i-1)) * N_GLOBAL_BUTTERFLIES_over_i;
      butterfly(output+idx, i_local, polar(tIdx * MINUS_PI_over_N_GLOBAL_BUTTERFLIES));
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // we copy contiguous chunks of local memory to output
  for(int g=0; g<nGroups; ++g)
  {
    int const base_global_index = 2 * N_LOCAL_BUTTERFLIES * get_local_size(0) * g;
    for(int j=0; j<(2*N_LOCAL_BUTTERFLIES)/nGroups; ++j)
    {
      int const m = base_global_index + workgroup_offset_idx + get_local_size(0) * j + k;
      // coalesced global memory write, local memory read with no bank conflict.
      global_output[m] = output[base_global_index/nGroups + get_local_size(0) * j + k];
    }
  }

}
