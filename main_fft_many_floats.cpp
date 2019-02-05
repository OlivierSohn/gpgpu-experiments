
#include <complex>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "error_check.cpp"

#include "cpu_fft.cpp"
#include "cpu_fft_norecursion.cpp"
#include "bitReverse.cpp"

#define MAX_SOURCE_SIZE (0x100000)

constexpr auto kernel_file = "/Users/Olivier/Dev/gpgpu/vector_fft_floats_multi.cl";

float rand_float(float minVal, float maxVal) {
  return minVal + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxVal-minVal)));
}

void withInput(cl_context context,
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
  
  // Our GPU kernel doesn't do bit-reversal of the input, so this should be done on the host.
  // In this scope, we verify that when the input is bit-reversed prior to being fed to 'cpu_func',
  // we get the expected result:
  if(verifyResults)
  {
    std::cout << "- make ref 1" << std::endl;
    auto refForwardFft = makeRefForwardFft(input); // this implementation has been well unit-tested in another project
    std::cout << "- make ref 2" << std::endl;
    auto cpuForwardFft = cpu_func(bitReversePermutation(input));
    std::cout << "- verify consistency" << std::endl;
    verifyVectorsAreEqual(refForwardFft, cpuForwardFft);
    std::cout << "- ok" << std::endl;
  }
  
  auto twiddle = imajuscule::compute_roots_of_unity<float>(input.size());

  std::vector<std::complex<float>> output;
  output.resize(input.size());

  cl_int ret;
  
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
  
  // Execute the OpenCL kernel
  size_t global_item_size = input.size()/(2*nButterfliesPerThread);
  size_t local_item_size = global_item_size;
  std::cout << "run kernels using global size : " << global_item_size << std::endl;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                               &global_item_size,
                               &local_item_size,
                               0, NULL, NULL);
  CHECK_CL_ERROR(ret);
  
  // Read the memory buffer output_mem_obj on the device to the local variable output
  
  std::cout << "waiting for gpu results... " << std::endl;
  ret = clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0,
                            output.size() * sizeof(decltype(output[0])), &output[0], 0, NULL, NULL);
  CHECK_CL_ERROR(ret);
  std::cout << "done" << std::endl;

  if(verifyResults) {
    std::cout << "verifying results... " << std::endl;
    // The output produced by the gpu is the same as the output produced by the cpu:
    verifyVectorsAreEqual(output,
                          cpu_func(input),
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

  ScopedKernel(cl_context context, cl_device_id device_id, char* const source_str, size_t const source_size, size_t const input_size) {
    int const nButterflies = input_size/2;

    for(nButterfliesPerThread = 1;;) {
      std::string const replaced_str = ReplaceString(std::string(source_str, source_size),
                                                     "replaceThisBeforeCompiling",
                                                     std::to_string(nButterfliesPerThread));
      size_t const replaced_source_size = replaced_str.size();
      const char * rep_src = replaced_str.data();
      
      cl_int ret;
      // Create a program from the kernel source
      program = clCreateProgramWithSource(context, 1,
                                          (const char **)&rep_src, (const size_t *)&replaced_source_size, &ret);
      CHECK_CL_ERROR(ret);
      
      // Build the program
      ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
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
  
  // Load the kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;
  
  fp = fopen(kernel_file, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  verify(source_size < MAX_SOURCE_SIZE);
  
  fclose( fp );
  
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
    
    const ScopedKernel sc(context, device_id, source_str, source_size, input.size());

    withInput(context,
              command_queue,
              sc.kernel,
              sc.nButterfliesPerThread,
              input,
              false // set this to true to verify results
              );
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

  free(source_str);
  return 0;
}
