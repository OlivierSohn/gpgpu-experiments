
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
 Converts from a representation where real and imaginary parts are stored in separate buffers
 to a std::complex representation
 */
template<typename T>
std::vector<std::complex<T>> unseparate(std::vector<T> const & v) {
  verify(v.size() % 2 == 0);
  
  std::vector<std::complex<T>> res;
  res.resize(v.size()/2);
  
  for(int i=0; i<v.size()/2; ++i) {
    res[i] = {v[i],v[i+v.size()/2]};
  }
  return res;
}

/*
 This is the cpu version of the gpu kernel, to test that our program works as intended.
 */
auto cpu_fft_norecursion(std::vector<float> const & input, int maxLevel = -1) {
  const int Sz = input.size(); // is assumed to be a power of 2
  
  std::vector<std::complex<float>> output = complexify(input);
  
  auto twiddle = imajuscule::compute_roots_of_unity<float>(Sz);
  
  if(maxLevel < 0) {
    maxLevel = Sz;
  }
  else {
    maxLevel = std::min(maxLevel, Sz);
  }
  
  for(int i=1; i<maxLevel; i <<= 1) {
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

inline int expand(int idxL, int N1, int N2) {
  return (idxL/N1)*N1*N2 + (idxL%N1);
  // TODO optimize
}

auto cpu_fft_norecursion_stockham(std::vector<float> const & input, int maxLevel = -1) {
  const int Sz = input.size(); // is assumed to be a power of 2

  if(maxLevel < 0) {
    maxLevel = Sz;
  }
  else {
    maxLevel = std::min(maxLevel, Sz);
  }

  auto a=complexify(input);
  decltype(a) b;
  b.resize(input.size());
  std::vector<std::complex<float>> * prev, *next;
  prev = &a;
  next = &b;
  
  auto twiddle = imajuscule::compute_roots_of_unity<float>(Sz);
  
  for(int i=1; i<maxLevel; i <<= 1) {
    for(int k = 0; k<Sz/2; k++) {
      std::complex<float> v[2];
//      float angle = -2.f*M_PI*(float)(k%i)/(float)(2*i);
      float angle = 0.f;
      auto twiddle = std::polar(1.f,angle);
      v[0] = (*prev)[k];
      v[1] = (*prev)[k+Sz/2] * twiddle;
      auto v0 = v[0];
      v[0] += v[1];
      v[1] = v0 - v[1];
      int idxD = expand(k, i, 2);
      (*next)[idxD] = v[0];
      (*next)[idxD+i] = v[1];
    }
    std::swap(prev,next);
  }
  
  return *prev;
}
