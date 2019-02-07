
float rand_float(float minVal, float maxVal) {
  return minVal + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxVal-minVal)));
}
