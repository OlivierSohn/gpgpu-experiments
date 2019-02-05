
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
