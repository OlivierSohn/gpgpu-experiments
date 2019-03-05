
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                   // Times for 4096 8192 fft //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr auto kernel_file = "vector_fft_floats_huge_local_coalesce_shifts_twiddles.cl";

using T = float;

std::vector<std::complex<T>> test_one_level(int const sz, int const num_workgroups) {
  std::vector<std::complex<T>> res;
  res.reserve(sz);
  int sz2 = sz / (num_workgroups*num_workgroups);
  verify(sz2 *num_workgroups * num_workgroups == sz);
  for(int g=0; g<num_workgroups/2; ++g) {
    for(int h=0; h<num_workgroups; ++h) {
      for(int i=0; i<sz2; ++i) {
        res.emplace_back(i/16, h);
      }
    }
  }
  for(int g=0; g<num_workgroups/2; ++g) {
    for(int h=0; h<num_workgroups; ++h) {
      for(int i=sz2; i<2*sz2; ++i) {
        res.emplace_back(i/16, h);
      }
    }
  }

  return res;
}

bool withInput(cl_context context,
               cl_device_id device_id,
               cl_command_queue command_queue,
               cl_kernel kernel1,
               cl_kernel kernel2,
               int nButterfliesPerThread,
               std::vector<T> const & input,
               int nWorkgroups,
               bool verifyResults
               )
{
  using namespace imajuscule;
  using namespace imajuscule::fft;

  verify(is_power_of_two(input.size()) && input.size() >= 2);
  
  std::vector<std::complex<T>> output;
  output.resize(input.size());
  
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
  
  // Create memory buffers on the device for each vector
  cl_int ret;
  cl_mem input_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        input.size() * sizeof(decltype(input[0])), NULL, &ret);
  CHECK_CL_ERROR(ret);
  cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         output.size() * sizeof(decltype(output[0])), NULL, &ret);
  CHECK_CL_ERROR(ret);

  // Copy the input to its memory buffers.
  // This can crash if the GPU has not enough memory.
  ret = clEnqueueWriteBuffer(command_queue, input_mem_obj, CL_TRUE, 0,
                             input.size() * sizeof(decltype(input[0])), &input[0], 0, NULL, NULL);
  CHECK_CL_ERROR(ret);
  
  // Set the arguments of the kernels
  ret = clSetKernelArg(kernel1, 0, (output.size() * sizeof(decltype(output[0]))) / nWorkgroups, NULL); // local memory
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&input_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&output_mem_obj);
  CHECK_CL_ERROR(ret);

  ret = clSetKernelArg(kernel2, 0, (output.size() * sizeof(decltype(output[0]))) / nWorkgroups, NULL); // local memory
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&output_mem_obj);
  CHECK_CL_ERROR(ret);

  // Execute the OpenCL kernel
  size_t global_item_size = input.size()/(2*nButterfliesPerThread);
  size_t local_item_size = global_item_size/nWorkgroups;
  std::cout << "run kernels using global size : " << global_item_size << std::endl;
  
  double elapsed = 0.;

  int nIterations = 3000;
  constexpr int nSkipIterations = 1;
  for(int i=0; i<nSkipIterations+nIterations; ++i)
  {
    cl_event event[2];
    int nKernels = (nWorkgroups == 1) ? 1 : 2;
    if(nWorkgroups == 1) {
      ret = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL,
                                   &global_item_size,
                                   &local_item_size,
                                   0, NULL, event);
    }
    else {
      ret = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL,
                                   &global_item_size,
                                   &local_item_size,
                                   0, NULL, event);
      CHECK_CL_ERROR(ret);
      ret = clEnqueueNDRangeKernel(command_queue, kernel2, 1, NULL,
                                   &global_item_size,
                                   &local_item_size,
                                   0, NULL, event+1);
      CHECK_CL_ERROR(ret);
    }

    ret = clWaitForEvents(1, event+nKernels-1);
    CHECK_CL_ERROR(ret);

    // skip first measurements
    if(i<nSkipIterations) {
      continue;
    }
    cl_ulong delta = 0;
    for(int j=0; j<nKernels; ++j) {
      cl_ulong time_start, time_end;
      ret = clGetEventProfilingInfo(event[j], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
      CHECK_CL_ERROR(ret);
      ret = clGetEventProfilingInfo(event[j], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
      CHECK_CL_ERROR(ret);
      delta += time_end - time_start;
    }
    elapsed += delta;
    
    // stop if the test is too long
    {
      cl_ulong sofarMilliSec = elapsed/1000000;
      if(sofarMilliSec > 1000) {
        nIterations = 1+i-nSkipIterations;
        break;
      }
    }
  }
  std::cout << "avg kernel duration (us) : " << (elapsed/(double)nIterations)/1000 <<
  " over " << nIterations << " iterations. " << std::endl;

  ret = clEnqueueReadBuffer(command_queue,
                            output_mem_obj, CL_TRUE, 0,
                            output.size() * sizeof(decltype(output[0])), &output[0], 0, NULL, NULL);
  
  CHECK_CL_ERROR(ret);
  
  if(verifyResults) {
    std::cout << "verifying results... " << std::endl;
    // The output produced by the gpu is the same as the output produced by the cpu:
    verifyVectorsAreEqual(output,
                          cpu_fft_norecursion(input),
                          // getFFTEpsilon is assuming that the floating point errors "add up"
                          // at every butterfly operation, but like said here :
                          // https://floating-point-gui.de/errors/propagation/
                          // this is true for multiplications, but not for additions
                          // which are used in butterfly operations.
                          // Hence I replace the following line with 0.001f:
                          //20.f*getFFTEpsilon<T>(input.size()),
                          0.001f
                          );
  }

  
  // Cleanup
  ret = clReleaseMemObject(input_mem_obj);
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
  cl_kernel kernel1, kernel2;
  int nButterfliesPerThread;

  ScopedKernel(cl_context context, cl_device_id device_id, std::string const & kernel_src, size_t const input_size, size_t const nWorkgroups) {
    using namespace imajuscule;
    int const nButterflies = input_size/2;
    char buf[256];
    memset(buf, 0, sizeof(buf));
    snprintf(buf, sizeof(buf), "%a", (T)(-M_PI/nButterflies));

    cl_int ret;

    // 2*nButterfliesPerThread must be a multiple of 'nWorkgroups'
    for(nButterfliesPerThread = std::max(1,(int)nWorkgroups/2);;) {
      
      if(!isMultiple(2*nButterfliesPerThread, nWorkgroups)) {
        throw std::runtime_error("the number of local butterflies must be a multiple of the number of groups");
      }
      
      std::string const replaced_str = ReplaceString(ReplaceString(ReplaceString(ReplaceString(kernel_src,
                                                                                               "replace_MINUS_PI_over_N_GLOBAL_BUTTERFLIES",
                                                                                               buf),
                                                                                 "replace_N_GLOBAL_BUTTERFLIES",
                                                                                 std::to_string(nButterflies)),
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
                           // -cl-fast-relaxed-math makes the twiddle fators computation a little faster
                           // but a little less accurate too.
                           "-I /Users/Olivier/Dev/gpgpu/ -cl-denorms-are-zero -cl-strict-aliasing -cl-fast-relaxed-math",
                           NULL, NULL);
      CHECK_CL_ERROR(ret);
      
      // Create the OpenCL kernel
      kernel1 = clCreateKernel(program, "first_butterflies", &ret);
      CHECK_CL_ERROR(ret);
      kernel2 = clCreateKernel(program, "last_butterflies", &ret);
      CHECK_CL_ERROR(ret);

      size_t workgroup_max_sz1, workgroup_max_sz2;
      ret = clGetKernelWorkGroupInfo(kernel1,
                                     device_id,
                                     CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(workgroup_max_sz1), &workgroup_max_sz1, NULL);
      CHECK_CL_ERROR(ret);
      ret = clGetKernelWorkGroupInfo(kernel2,
                                     device_id,
                                     CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(workgroup_max_sz2), &workgroup_max_sz2, NULL);
      CHECK_CL_ERROR(ret);
      verify(workgroup_max_sz2 == workgroup_max_sz1);
      size_t workgroup_max_sz = std::min(workgroup_max_sz1, workgroup_max_sz2);
      std::cout << "workgroup max size: " << workgroup_max_sz << " for " << nButterfliesPerThread << " butterfly per thread." << std::endl;
      
      if(nButterflies > nButterfliesPerThread * workgroup_max_sz * nWorkgroups) {
        release();
        // To estimate the next value of 'nButterfliesPerThread',
        // we make the reasonnable assumption that "work group max size"
        // won't be bigger if we increase 'nButterfliesPerThread':
        nButterfliesPerThread = nButterflies / (workgroup_max_sz * nWorkgroups);
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
    cl_int ret = clReleaseKernel(kernel1);
    CHECK_CL_ERROR(ret);
    ret = clReleaseKernel(kernel2);
    CHECK_CL_ERROR(ret);
    ret = clReleaseProgram(program);
    CHECK_CL_ERROR(ret);
    kernel1 = 0;
    kernel2 = 0;
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
  
  cl_ulong local_mem_sz;
  ret = clGetDeviceInfo(device_id,
                        CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(local_mem_sz), &local_mem_sz, NULL);
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
  // We could query CL_DEVICE_GLOBAL_MEM_SIZE and deduce an upper bound for 'sz'.
  for(int sz=2; sz < 100000000; sz *= 2) {
    std::cout << std::endl << "* input size: " << sz << std::endl;
    
    // Create the input vector
    std::vector<T> input;
    input.reserve(sz);
    for(int i=0; i<sz; ++i) {
      input.push_back(rand_float(0.f,1.f));
    }
    
    int const minNWorkgroups = 1 + (input.size() * sizeof(std::complex<T>) - 1) / local_mem_sz;
    std::cout << "using " << minNWorkgroups << " workgroup(s)." << std::endl;

    const ScopedKernel sc(context, device_id, kernel_src, input.size(), minNWorkgroups);
    
    if(!withInput(context,
                  device_id,
                  command_queue,
                  sc.kernel1,
                  sc.kernel2,
                  sc.nButterfliesPerThread,
                  input,
                  minNWorkgroups,
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
