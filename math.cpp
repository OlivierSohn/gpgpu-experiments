
namespace imajuscule {
  constexpr bool is_power_of_two(size_t n) { return ((n != 0) && !(n & (n - 1))); }
  
  constexpr unsigned int power_of_two_exponent(unsigned int v) {
    unsigned int n = 0;
    while (v >>= 1) {
      ++n;
    }
    return n;
  }
  
  constexpr bool isMultiple(int a, int b) {
    return (a/b)*b == a;
  }
}
