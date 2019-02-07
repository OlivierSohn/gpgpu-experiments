
constexpr auto kernel_file = "vector_mix_floats.cl";

/*
 This is the cpu version of the gpu kernel, to test that our program works as intended.
 */
auto cpu_func(std::vector<float> const & input) {
  std::vector<float> output;
  output.resize(input.size());
  for(int i=0;i<input.size(); i+=2) {
    output[i] = input[i+1];
    output[i+1] = input[i];
  }
  return output;
}

int main(void) {
  // Create the input vector
  std::vector<float> input{2.5f, 9.f, -3.f, 5.f, 10.f, 4.f, 1.f, 7.f};
  
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
  CHECK_CL_ERROR(ret);
  cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    bufSz, NULL, &ret);
  CHECK_CL_ERROR(ret);

  // Copy the lists A and B to their respective memory buffers
  ret = clEnqueueWriteBuffer(command_queue, input_mem_obj, CL_TRUE, 0,
                             bufSz, input.data(), 0, NULL, NULL);
  CHECK_CL_ERROR(ret);

  // Create a program from the kernel source
  auto kernel_src = read_kernel(kernel_file);
  auto kernel_c_src = kernel_src.c_str();
  auto source_size = kernel_src.size();
  cl_program program = clCreateProgramWithSource(context, 1,
                                                 (const char **)&kernel_c_src, (const size_t *)&source_size, &ret);
  CHECK_CL_ERROR(ret);

  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  CHECK_CL_ERROR(ret);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "kernel_func", &ret);
  CHECK_CL_ERROR(ret);

  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_mem_obj);
  CHECK_CL_ERROR(ret);

  // Execute the OpenCL kernel on the list
  size_t global_item_size = input.size(); // Process the entire lists
  size_t local_item_size = input.size();
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                               &global_item_size, &local_item_size, 0, NULL, NULL);
  CHECK_CL_ERROR(ret);

  // Read the memory buffer output_mem_obj on the device to the local variable output
  decltype(input) output;
  output.resize(input.size());

  ret = clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0,
                            bufSz, output.data(), 0, NULL, NULL);
  CHECK_CL_ERROR(ret);

  // The output produced by the gpu is the same as the output produced by the cpu:
  verifyVectorsAreEqual(output, cpu_func(input));
  
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
  ret = clReleaseMemObject(output_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clReleaseCommandQueue(command_queue);
  CHECK_CL_ERROR(ret);
  ret = clReleaseContext(context);
  CHECK_CL_ERROR(ret);
  return 0;
}
