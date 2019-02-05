
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
//    on vectors of arbitrary sizes:
//
#include "main_fft_many_floats.cpp"


