
#include <iostream>

// When debugging your program it will be useful
// to put a breakpoint in kill()
// so as to be able to inspect the stack.
void kill() {
  throw "program error";
}
void CHECK_CL_ERROR(int res) {
  if(res==CL_SUCCESS) {
    return;
  }
  fprintf(stderr, "OpenCL Error %d\n", res);
  kill();
}

void verify(bool b) {
  if(b) {
    return;
  }
  kill();
}

template<typename T>
void verifyVectorsAreEqual(std::vector<T> const & a, std::vector<T> const & b, float epsilon = 1e-5) {
  verify(a.size() == b.size());
  
  for(int i=0; i<a.size(); ++i) {
    auto range = std::abs(a[i]) + std::abs(b[i]);
    if(range == 0.f) {
      continue;
    }
    if(std::abs(a[i]-b[i]) / range > epsilon) {
      std::cout << i << ": " << a[i] << " != " << b[i] << std::endl;
      kill();
    }
  }
}
