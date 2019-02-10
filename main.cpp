
// common includes

#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "error_check.cpp"

#include "read_kernel_source.cpp"

#include "math.cpp"
#include "bitReverse.cpp"
#include "rand.cpp"

#include "cpu_fft.cpp"
#include "cpu_fft_norecursion.cpp"



//
// GPGPU algorithms, by order of increasing (gpu kernel) complexities:
//


// 1. This example adds two vectors of 1024 ints:
//
//#include "main_add_1024_ints.cpp"



// 2. This example performs a permutation of adjacent elements
//    in a vector of 8 elements:
//
//#include "main_mix_8_floats.cpp"



// 3. This example computes an fft (Cooley-Tuckey radix-2, no bit-reversal of the input)
//    on a vector of 8 elements:
//
//#include "main_fft_8_floats.cpp"



// 4. This example computes an fft (Cooley-Tuckey radix-2, no bit-reversal of the input)
//    on vectors of large sizes:
//
//#include "main_fft_many_floats.cpp"           // for 8192 fft: 1500 us kernel time



// 5. This example computes an fft (Cooley-Tuckey radix-2, no bit-reversal of the input)
//    on vectors of large sizes, using local memory to speed up the kernel.
//
//    Several variations are possible, by using different kernels (see inside the source)
//
//#include "main_fft_many_floats_local.cpp"    // for 8192 fft: 1300 us kernel time

// 6. This example computes an fft (Cooley-Tuckey radix-2, no bit-reversal of the input)
//    on vectors of large sizes, using local memory to speed up the kernel,
//    and computing twiddle factors on the fly instead of reading them from memory:
//
//#include "main_fft_many_floats_local_twiddles.cpp"


// 7. This example computes an fft (Cooley-Tuckey radix-2, no bit-reversal of the input)
//    on vectors of large sizes, using local memory to speed up the kernel,
//    computing twiddle factors on the fly instead of reading them from memory,
//    and where a separate representation for complex numbers is used to avoid bank conflicts:
//
//#include "main_fft_many_floats_local_twiddles_separate.cpp"


// 8. This example computes an fft (Stockham radix-2)
//    on a vector of 8 elements:
//
//#include "main_fft_8_floats_stockham.cpp"

// 9. This example computes an fft (Stockham radix-2)
//    on vectors of large sizes.
//
//#include "main_fft_many_floats_stockham.cpp"

// 10. This example computes an fft (Stockham radix-2)
//    on vectors of large sizes,
//    computing twiddle factors on the fly instead of reading them from memory:
//
//#include "main_fft_many_floats_stockham_twiddles.cpp"

// 11. This example computes an fft (Stockham radix-2)
//    on vectors of large sizes,
//    computing twiddle factors on the fly instead of reading them from memory,
//    and where a separate representation for complex numbers is used to avoid bank conflicts:
//
#include "main_fft_many_floats_stockham_twiddles_separate.cpp"
