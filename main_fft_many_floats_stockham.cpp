

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                        // Times for 4096 fft //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//constexpr auto kernel_file = "vector_fft_floats_stockham_multi_local_coalesce.cl";          // 208 us
constexpr auto kernel_file = "vector_fft_floats_stockham_multi_local_coalesce_shift.cl";      // 176-205 us

bool withInput(cl_context context,
               cl_device_id device_id,
               cl_command_queue command_queue,
               cl_kernel kernel,
               int nButterfliesPerThread,
               std::vector<float> const & input,
               bool verifyResults
               )
{
  using namespace imajuscule;
  using namespace imajuscule::fft;

  verify(is_power_of_two(input.size()) && input.size() >= 2);
  
  std::vector<std::complex<float>> output;
  output.resize(input.size());
  
  cl_ulong local_mem_sz;
  cl_int ret = clGetDeviceInfo(device_id,
                               CL_DEVICE_LOCAL_MEM_SIZE,
                               sizeof(local_mem_sz), &local_mem_sz, NULL);
  if(local_mem_sz < 2 * output.size() * sizeof(decltype(output[0]))) { // factor 2 because we ping pong between buffers
    std::cout << "not enough local memory on the device!" << std::endl;
    return false;
  }

  // Our GPU kernel doesn't do bit-reversal of the input, so this should be done on the host.
  // In this scope, we verify that when the input is bit-reversed prior to being fed to 'cpu_func',
  // we get the expected result:
  if(verifyResults)
  {
    std::cout << "- make ref 1" << std::endl;
    auto refForwardFft = makeRefForwardFft(input); // this implementation has been well unit-tested in another project
    std::cout << "- make ref 2" << std::endl;
    auto cpuForwardFft = cpu_fft_norecursion(bitReversePermutation(input));
    std::cout << "- verify consistency" << std::endl;
    verifyVectorsAreEqual(refForwardFft, cpuForwardFft);
    std::cout << "- ok" << std::endl;
  }
  
  auto twiddle = imajuscule::compute_roots_of_unity<float>(input.size());

  // Create memory buffers on the device for each vector
  cl_mem input_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        input.size() * sizeof(decltype(input[0])), NULL, &ret);
  CHECK_CL_ERROR(ret);
  cl_mem twiddle_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                          twiddle.size() * sizeof(decltype(twiddle[0])), NULL, &ret);
  CHECK_CL_ERROR(ret);
  cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         output.size() * sizeof(decltype(output[0])), NULL, &ret);
  CHECK_CL_ERROR(ret);
  
  // Copy the input and twiddles to their respective memory buffers.
  // This can crash if the GPU has not enough memory.
  ret = clEnqueueWriteBuffer(command_queue, input_mem_obj, CL_TRUE, 0,
                             input.size() * sizeof(decltype(input[0])), &input[0], 0, NULL, NULL);
  CHECK_CL_ERROR(ret);
  ret = clEnqueueWriteBuffer(command_queue, twiddle_mem_obj, CL_TRUE, 0,
                             twiddle.size()*sizeof(decltype(twiddle[0])), &twiddle[0], 0, NULL, NULL);
  CHECK_CL_ERROR(ret);
  
  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&twiddle_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel, 3, 2*sizeof(float) * 2*input.size(), NULL); // pingpong buffer
  CHECK_CL_ERROR(ret);

  // Execute the OpenCL kernel
  size_t global_item_size = input.size()/(2*nButterfliesPerThread);
  size_t local_item_size = global_item_size;
  std::cout << "run kernels using global size : " << global_item_size << std::endl;
  
  double elapsed = 0.;

  constexpr int nIterations = 3000;
  constexpr int nSkipIterations = 5;
  for(int i=0; i<nSkipIterations+nIterations; ++i)
  {
    cl_event event;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                 &global_item_size,
                                 &local_item_size,
                                 0, NULL, &event);
    CHECK_CL_ERROR(ret);
    
    ret = clWaitForEvents(1, &event);
    CHECK_CL_ERROR(ret);

    cl_ulong time_start, time_end;
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    CHECK_CL_ERROR(ret);
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    CHECK_CL_ERROR(ret);

    if(i>=nSkipIterations) {
      elapsed += time_end - time_start;
    }
  }
  std::cout << "avg kernel duration (us) : " << (int)(elapsed/(double)nIterations)/1000 << std::endl;

  // Read the memory buffer output_mem_obj on the device to the local variable output
  ret = clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0,
                            output.size() * sizeof(decltype(output[0])), &output[0], 0, NULL, NULL);
  
  CHECK_CL_ERROR(ret);

  if(verifyResults) {
    std::cout << "verifying results... " << std::endl;
    // The output produced by the gpu is the same as the output produced by the cpu:
    verifyVectorsAreEqual(output,
                          makeRefForwardFft(input),
                          // getFFTEpsilon is assuming that the floating point errors "add up"
                          // at every butterfly operation, but like said here :
                          // https://floating-point-gui.de/errors/propagation/
                          // this is true for multiplications, but not for additions
                          // which are used in butterfly operations.
                          // Hence I replace the following line with 0.001f:
                          //20.f*getFFTEpsilon<float>(input.size()),
                          0.001f
                          );
  }
  
  // Cleanup
  ret = clReleaseMemObject(input_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clReleaseMemObject(twiddle_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clReleaseMemObject(output_mem_obj);
  CHECK_CL_ERROR(ret);
  
  return true;
}

std::string ReplaceString(std::string subject, const std::string& search,
                          const std::string& replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return subject;
}

struct ScopedKernel {
  
  cl_program program;
  cl_kernel kernel;
  int nButterfliesPerThread;

  ScopedKernel(cl_context context, cl_device_id device_id, std::string const & kernel_src, size_t const input_size) {
    using namespace imajuscule;
    int const nButterflies = input_size/2;
    
    cl_int ret;

    // TODO if local memory can hold the output, use the kernel with local memory,
    // else use the kernel with global memory
    // We could think of a mixed approach where we compute the fft by parts:
    //   do the first levels by blocks, using local memory + writeback,
    //   omit the last writeback, use the local memory + global memory for
    //     the other levels
    //   do the writeback of the omitted portion

    for(nButterfliesPerThread = 1;;) {
      std::string const replaced_str = ReplaceString(ReplaceString(kernel_src,
                                                                   "replace_LOG2_N_GLOBAL_BUTTERFLIES",
                                                                   std::to_string(power_of_two_exponent(nButterflies))),
                                                     "replace_N_LOCAL_BUTTERFLIES",
                                                     std::to_string(nButterfliesPerThread));
      size_t const replaced_source_size = replaced_str.size();
      const char * rep_src = replaced_str.data();

      // Create a program from the kernel source
      program = clCreateProgramWithSource(context, 1,
                                          (const char **)&rep_src, (const size_t *)&replaced_source_size, &ret);
      CHECK_CL_ERROR(ret);
      
      // Build the program
      ret = clBuildProgram(program, 1, &device_id,
                           "-I /Users/Olivier/Dev/gpgpu/ -cl-denorms-are-zero -cl-strict-aliasing -cl-fast-relaxed-math",
                           NULL, NULL);
      CHECK_CL_ERROR(ret);
      
      // Create the OpenCL kernel
      kernel = clCreateKernel(program, "kernel_func", &ret);
      CHECK_CL_ERROR(ret);
      
      size_t workgroup_max_sz;
      ret = clGetKernelWorkGroupInfo(kernel,
                                     device_id,
                                     CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(workgroup_max_sz), &workgroup_max_sz, NULL);
      CHECK_CL_ERROR(ret);
      std::cout << "workgroup max size: " << workgroup_max_sz << " for " << nButterfliesPerThread << " butterfly per thread." << std::endl;
      
      if(nButterflies > nButterfliesPerThread * workgroup_max_sz) {
        release();
        // To estimate the next value of 'nButterfliesPerThread',
        // we make the reasonnable assumption that "work group max size"
        // won't be bigger if we increase 'nButterfliesPerThread':
        nButterfliesPerThread = nButterflies / workgroup_max_sz;
        continue;
      }
      break;
    }
  }
  
  ~ScopedKernel() {
    release();
  }

private:
  void release() {
    cl_int ret = clReleaseKernel(kernel);
    CHECK_CL_ERROR(ret);
    ret = clReleaseProgram(program);
    CHECK_CL_ERROR(ret);
    kernel = 0;
    program = 0;
  }
  
  ScopedKernel(const ScopedKernel&) = delete;
  ScopedKernel& operator=(const ScopedKernel&) = delete;
  ScopedKernel(ScopedKernel&&) = delete;
  ScopedKernel& operator=(ScopedKernel&&) = delete;
};

int main(void) {
  srand(0); // we use rand() as random number generator and we want reproducible results so we use a fixed seeed.
  
  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  CHECK_CL_ERROR(ret);
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                       &device_id, &ret_num_devices);
  CHECK_CL_ERROR(ret);
  
  // Create an OpenCL context
  cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  CHECK_CL_ERROR(ret);
  
  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id,
                                                        CL_QUEUE_PROFILING_ENABLE, &ret);
  CHECK_CL_ERROR(ret);

  // read the kernel code
  auto kernel_src = read_kernel(kernel_file);

  // Note that if the GPU has not enough memory available, it will crash.
  // On my system, the limit is reached at size 134217728.
  for(int sz=2; sz < 10000000; sz *= 2) {
    std::cout << std::endl << "* input size: " << sz << std::endl;
    
    // Create the input vector
    std::vector<float> input;
    input.reserve(sz);
    for(int i=0; i<sz; ++i) {
      input.push_back(rand_float(0.f,1.f));
    }
    
    const ScopedKernel sc(context, device_id, kernel_src, input.size());

    if(!withInput(context,
              device_id,
              command_queue,
              sc.kernel,
              sc.nButterfliesPerThread,
              input,
              true // set this to true to verify results
                  )) {
      break;
    }
  }
  
  // Clean up
  ret = clFlush(command_queue);
  CHECK_CL_ERROR(ret);
  ret = clFinish(command_queue);
  CHECK_CL_ERROR(ret);
  ret = clReleaseCommandQueue(command_queue);
  CHECK_CL_ERROR(ret);
  ret = clReleaseContext(context);
  CHECK_CL_ERROR(ret);
  
  return 0;
}
