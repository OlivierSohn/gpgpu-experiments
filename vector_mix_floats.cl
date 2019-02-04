__kernel void kernel_func(__global const float *input, __global float *output) {
  
  // Get the index of the current element to be processed
  int i = get_global_id(0); // we use a single dimension hence we pass 0

  // Note: In this very naive implementation, threads of odd indexes won't do anything.
  //       To optimize the use of GPU resources, we could use half the number of threads
  //       and multiply the index by 2 instead.
  if(i & 0x1) {
    return;
  }
  
  float a = input[i];
  float b = input[i+1];
  
  output[i] = b;
  output[i+1] = a;
}
