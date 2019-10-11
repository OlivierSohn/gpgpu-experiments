
#include "cplx.c"

#define N_LOCAL_BUTTERFLIES       replace_N_LOCAL_BUTTERFLIES // must be a power of 2
#define N_GLOBAL_BUTTERFLIES      replace_N_GLOBAL_BUTTERFLIES // must be a power of 2, and >= N_LOCAL_BUTTERFLIES
#define LOG2_N_GLOBAL_BUTTERFLIES replace_LOG2_N_GLOBAL_BUTTERFLIES
#define MINUS_PI_over_N_GLOBAL_BUTTERFLIES replace_MINUS_PI_over_N_GLOBAL_BUTTERFLIES

inline int expand(int idxL, int log2N1, int mm) {
  return ((idxL-mm) << 1) + mm;
}

__kernel void first_butterflies(__global const float *input,
                                __global struct cplx *global_output,
                                __local struct cplx* pingpong) {
  int const k = get_local_id(0);
  int const localBufSz = 2 * N_LOCAL_BUTTERFLIES * get_local_size(0);

  __local struct cplx *prev = pingpong;
  __local struct cplx *next = pingpong + localBufSz;

  int const workgroupIdxOffset = (localBufSz/2) * get_group_id(0);
  int const lastSz = N_GLOBAL_BUTTERFLIES / get_num_groups(0);

  // TODO this approach is probably wrong, for nGroups=2 we want odd/even split.
  // so by generalizing that, maybe we need to take one input every nGroups sample.
  // And to have coalescent reads and writes, we could write
  // 2 kernels to interleave / deinterleave the data
  
  // we copy two contiguous chunks of input to shared memory:
  //                                             i          j
  // global memory :                        ...+++///%%% ...+++///%%%
  //                                          i'j'
  // shared memory (here for second group): +++ +++
  //                         i' = i-workgroupIdxOffset
  //                         j' = j-workgroupIdxOffset-N_GLOBAL_BUTTERFLIES+localBufSz/2
  // the offset for the first index is 'workgroupIdxOffset'
  // the offset for the middle index is 'workgroupIdxOffset + N_GLOBAL_BUTTERFLIES'
  for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_local_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    prev[m] = complexFromReal(input[workgroupIdxOffset+m]);
  }
  for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j) {
    int const m = get_local_size(0) * j + k;
    // coalesced global memory read, local memory write with no bank conflict.
    prev[localBufSz/2 + m] = complexFromReal(input[N_GLOBAL_BUTTERFLIES+workgroupIdxOffset+m]);
  }

  int const global_base_idx = get_global_id(0) * N_LOCAL_BUTTERFLIES;
  int const local_base_idx = get_local_id(0) * N_LOCAL_BUTTERFLIES;
  
  
  // i = size of a butterfly half
  // LOG2_N_GLOBAL_BUTTERFLIES_over_i = log2(N_GLOBAL_BUTTERFLIES / i)
  int log2i = 0;
  for(int i=1, LOG2_N_GLOBAL_BUTTERFLIES_over_i = LOG2_N_GLOBAL_BUTTERFLIES;
      i <= lastSz;
      i <<= 1, --LOG2_N_GLOBAL_BUTTERFLIES_over_i, ++log2i)
  {
    if(log2i==2) {
      break;
    }

 barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int j=0; j<N_LOCAL_BUTTERFLIES; ++j)
    {
      int const m = global_base_idx + j;
//      int const tIdx = (m & (i-1)) << LOG2_N_GLOBAL_BUTTERFLIES_over_i;
      int const tIdx = 0;
      struct cplx one = {
        .real = 1.f, .imag = 0.f
      };
      
      int const local_m = local_base_idx + j;
      int const idxD = expand(local_m, log2i, local_m & (i-1));
      //next[idxD] = complexFromReal(i);
      //next[idxD+i] = complexFromReal(i);
      butterfly_outofplace(local_m, // local_m is in the first half of the buffer
                           idxD, // could be either in the first half or the second half of the buffer.
                           prev,next,
                           lastSz,
                           i,
                           one/*polar(tIdx * MINUS_PI_over_N_GLOBAL_BUTTERFLIES)*/);//*/
    }

    // swap(prev,next)
    {
      __local struct cplx * tmp = prev;
      prev = next;
      next = tmp;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  {
    int const m_first_workitem = (get_global_id(0) - get_local_id(0)) * N_LOCAL_BUTTERFLIES;
    int idxDGlobal = expand(m_first_workitem,
                            log2i-1,
                            m_first_workitem & (lastSz-1));//*/
    // we copy local memory to a single contiguous chunk of output
    for(int j=0; j<2*N_LOCAL_BUTTERFLIES; ++j) {
      int const m = get_local_size(0) * j + k;
      // coalesced global memory read, local memory write with no bank conflict.
      global_output[idxDGlobal+m] = prev[m];
    }
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
  
  int const firstSz = N_GLOBAL_BUTTERFLIES / nGroups;
  int const firstLocalSz = (N_LOCAL_BUTTERFLIES * get_local_size(0)) / nGroups;

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
