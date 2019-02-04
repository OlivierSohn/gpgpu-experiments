
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



// 3. This example computes the "butterfly" stage of the Cooley-Tuckey
//    fft algorithm (bit-reversal of the input is omitted) on a vector
//    of 8 elements:
//
#include "main_fft_8_floats.cpp"


