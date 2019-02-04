
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

#define MAX_SOURCE_SIZE (0x100000)

constexpr auto kernel_file = "/Users/Olivier/Dev/gpgpu/vector_fft_floats.cl";

void kill() {
  assert(0);
  throw "program error";
}
void CHECK_CL_ERROR(int res) {
  if(res==CL_SUCCESS) {
    return;
  }
  fprintf(stderr, "OpenCL Error %d\n", res);
  kill();
}

template<typename T>
void verifyVectorsAreEqual(std::vector<T> const & a, std::vector<T> const & b) {
  if(a.size() != b.size()) {
    kill();
  }
  for(int i=0; i<a.size(); ++i) {
    if(a[i] != b[i]) {
      std::cout << a[i] << " != " << b[i] << std::endl;
      kill();
    }
  }
}


template<typename T>
static std::complex<T> make_root_of_unity(unsigned int index, unsigned int size) {
  return std::polar(1.f, -2 * static_cast<T>(M_PI) * index / size);
}

template<typename T>
void compute_roots_of_unity(unsigned int N, std::vector<std::complex<T>> & res) {
  auto n_roots = N/2;
  res.reserve(n_roots);
  for(unsigned int i=0; i<n_roots; ++i) {
    res.push_back(make_root_of_unity<T>(i,N));
  }
}

template<typename T>
auto compute_roots_of_unity(unsigned int N) {
  std::vector<std::complex<T>> res;
  compute_roots_of_unity(N, res);
  return std::move(res);
}

void cpu_butterfly(std::complex<float> & a, std::complex<float> & b, std::complex<float> twiddle) {
  auto const t = b * twiddle;
  b = a - t;
  a += t;
}

auto complexify(std::vector<float> const & v) {
  std::vector<std::complex<float>> output;
  
  output.reserve(v.size());
  for(auto i : v) {
    output.emplace_back(i);
  }
  return output;
}

/*
 This is the cpu version of the gpu kernel, to test that our program works as intended.
 */
auto cpu_func(std::vector<float> const & input) {
  const unsigned int Sz = input.size(); // is assumed to be a power of 2

  std::vector<std::complex<float>> output = complexify(input);
  
  auto twiddle = compute_roots_of_unity<float>(Sz);
  
  for(int i=1; i<Sz; i <<= 1) {
    for(int k = 0; k<Sz; k += 2*i) {
      for(int l=0; l<i; ++l) {
        int idx = k+l;
        auto tIdx = l*((Sz/2)/i); // TODO verify
        //std::cout << tIdx << std::endl;
        cpu_butterfly(output[idx], output[idx+i], twiddle[tIdx]);
      }
    }
  }

  return output;
}

int main(void) {
  // Create the input vector
  std::vector<float> input{2.5f, 9.f, -3.f, 5.f, 10.f, 4.f, 1.f, 7.f};
  const unsigned int Sz = input.size(); // is assumed to be a power of 2
  
  auto twiddle = compute_roots_of_unity<float>(Sz);

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
  cl_program program = clCreateProgramWithSource(context, 1,
                                                 (const char **)&source_str, (const size_t *)&source_size, &ret);
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
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&twiddle_mem_obj);
  CHECK_CL_ERROR(ret);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem_obj);
  CHECK_CL_ERROR(ret);

  // Execute the OpenCL kernel on the list
  size_t global_item_size = input.size(); // Process the entire lists
  size_t local_item_size = input.size();
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
  verifyVectorsAreEqual(output, cpu_func(input));
  
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
