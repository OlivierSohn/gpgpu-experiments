
/* Knuth's algorithm from http://www.hackersdelight.org/revisions.pdf. Retrieved 8/19/2015 */
inline uint32_t reverseBits (uint32_t a)
{
  uint32_t t;
  a = (a << 15) | (a >> 17);
  t = (a ^ (a >> 10)) & 0x003f801f;
  a = (t + (t << 10)) ^ a;
  t = (a ^ (a >>  4)) & 0x0e038421;
  a = (t + (t <<  4)) ^ a;
  t = (a ^ (a >>  2)) & 0x22488842;
  a = (t + (t <<  2)) ^ a;
  return a;
}

auto bitReversePermutation(std::vector<float> const & v) {
  using namespace imajuscule;
  
  std::vector<float> res;
  res.resize(v.size());
  assert(is_power_of_two(v.size()));
  uint32_t const e = power_of_two_exponent(v.size());
  for(int i=0; i<v.size(); ++i) {
    uint32_t ri = reverseBits(i);
    ri >>= (uint32_t)(8*sizeof(uint32_t) - e);
    res[ri] = v[i];
  }
  return res;
}
