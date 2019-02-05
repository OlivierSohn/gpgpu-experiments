
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
  
  auto twiddle = imajuscule::compute_roots_of_unity<float>(Sz);
  
  for(int i=1; i<Sz; i <<= 1) {
    for(int k = 0; k<Sz; k += 2*i) {
      for(int l=0; l<i; ++l) {
        int idx = k+l;
        auto tIdx = l*((Sz/2)/i);
        cpu_butterfly(output[idx], output[idx+i], twiddle[tIdx]);
      }
    }
  }
  
  return output;
}

