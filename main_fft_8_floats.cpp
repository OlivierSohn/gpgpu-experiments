
constexpr auto kernel_file = "vector_fft_floats.cl";

int main(void) {
  // Create the input vector
  std::vector<float> input{2.5f, 9.f, -3.f, 5.f, 10.f, 4.f, 1.f, 7.f};

  // Our GPU kernel doesn't do bit-reversal of the input, so this should be done on the host.
  // In this scope, we verify that when the input is bit-reversed prior to being fed to 'cpu_func',
  // we get the expected result:
  {
    auto refForwardFft = makeRefForwardFft(input); // this implementation has been well unit-tested in another project
    auto cpuForwardFft = cpu_fft_norecursion(bitReversePermutation(input));
    verifyVectorsAreEqual(refForwardFft, cpuForwardFft);
  }
  
  const unsigned int Sz = input.size(); // is assumed to be a power of 2
  
  auto twiddle = imajuscule::compute_roots_of_unity<float>(Sz);

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
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  CHECK_CL_ERROR(ret);

  auto bufSz = input.size() * sizeof(decltype(input[0]));
  
  // Create memory buffers on the device for each vector
  cl_mem input_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        bufSz, NULL, &ret);
  cl_mem twiddle_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        bufSz*2, NULL, &ret);
  CHECK_CL_ERROR(ret);
  cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    bufSz*2 /* because these are complex<float>*/, NULL, &ret);
  CHECK_CL_ERROR(ret);

  // Copy the lists A and B to their respective memory buffers
  ret = clEnqueueWriteBuffer(command_queue, input_mem_obj, CL_TRUE, 0,
                             bufSz, input.data(), 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, twiddle_mem_obj, CL_TRUE, 0,
                             twiddle.size()*sizeof(decltype(twiddle[0])), twiddle.data(), 0, NULL, NULL);
  CHECK_CL_ERROR(ret);

  // Create a program from the kernel source
  auto kernel_src = read_kernel(kernel_file);
  auto kernel_c_src = kernel_src.c_str();
  auto source_size = kernel_src.size();
  cl_program program = clCreateProgramWithSource(context, 1,
                                                 (const char **)&kernel_c_src, (const size_t *)&source_size, &ret);
  CHECK_CL_ERROR(ret);

  // Build the program
  std::string options = std::string("-I ") + src_root() +
    " -cl-denorms-are-zero -cl-strict-aliasing -cl-fast-relaxed-math";
  ret = clBuildProgram(program, 1, &device_id, options.c_str(), NULL, NULL);
  CHECK_CL_ERROR(ret);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "kernel_func", &ret);
  CHECK_CL_ERROR(ret);

  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&twiddle_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem_obj);
  CHECK_CL_ERROR(ret);

  // Execute the OpenCL kernel on the list
  size_t global_item_size = input.size()/2; // the number of butterfly operations per fft level
  size_t local_item_size = global_item_size;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                               &global_item_size, &local_item_size, 0, NULL, NULL);
  CHECK_CL_ERROR(ret);

  // Read the memory buffer output_mem_obj on the device to the local variable output
  std::vector<std::complex<float>> output;
  output.resize(input.size());

  ret = clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0,
                            bufSz*2, output.data(), 0, NULL, NULL);
  CHECK_CL_ERROR(ret);

  // The output produced by the gpu is the same as the output produced by the cpu:
  verifyVectorsAreEqual(output, cpu_fft_norecursion(input));
  
  // Display the result to the screen
  /*
  for(auto e : input)
    printf("i: %f\n", e);
  for(auto e : output)
    printf("o: %f\n", e);
  */
   
  // Clean up
  ret = clFlush(command_queue);
  CHECK_CL_ERROR(ret);
  ret = clFinish(command_queue);
  CHECK_CL_ERROR(ret);
  ret = clReleaseKernel(kernel);
  CHECK_CL_ERROR(ret);
  ret = clReleaseProgram(program);
  CHECK_CL_ERROR(ret);
  ret = clReleaseMemObject(input_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clReleaseMemObject(twiddle_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clReleaseMemObject(output_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clReleaseCommandQueue(command_queue);
  CHECK_CL_ERROR(ret);
  ret = clReleaseContext(context);
  CHECK_CL_ERROR(ret);
  return 0;
}
