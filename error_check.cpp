
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
bool close(T const & a, T const & b, float epsilon) {
  
  auto range = std::abs(a) + std::abs(b);
  if(range == 0.f) {
    return true;
  }
  else if(std::abs(a-b) / range > epsilon) {
    //std::cout << i << ": " << a << " != " << b << std::endl;
    return false;
  }
  else {
    return true;
  }
}


template<typename T>
std::vector<std::pair<std::pair<int, int>, bool>>
equalRanges(std::vector<T> const & a, std::vector<T> const & b, float epsilon) {
  verify(a.size() == b.size());
  
  std::vector<std::pair<std::pair<int, int>, bool>> ranges;
  
  auto success = [&ranges]() {
    if(ranges.empty()) {
      ranges.emplace_back(std::make_pair(0,1),true);
      return;
    }
    auto & last = ranges.back();
    if(last.second) {
      last.first.second++;
    }
    else {
      ranges.emplace_back(std::make_pair(last.first.second,last.first.second+1),true);
    }
  };
  auto error = [&ranges]() {
    if(ranges.empty()) {
      ranges.emplace_back(std::make_pair(0,1),false);
      return;
    }
    auto & last = ranges.back();
    if(!last.second) {
      last.first.second++;
    }
    else {
      ranges.emplace_back(std::make_pair(last.first.second,last.first.second+1),false);
    }
  };
  
  for(int i=0; i<a.size(); ++i) {
    if(close(a[i],b[i], epsilon)) {
      success();
    }
    else {
      error();
    }
  }
  
  return ranges;
}


/*
 Used in "diff" operations between vectors
 */
struct correspondance {
  correspondance(int idxA, int idxB) : idxA(idxA), idxB(idxB), length(1) {}
  
  int idxA; // idx in "A" vector
  int idxB; // idx in "B" vector
  int length; // length of the correspondance
  
  int endA() const {
    return idxA + length;
  }
  int endB() const {
    return idxB + length;
  }
};

template<typename T>
std::vector<correspondance>
correspondances(std::vector<T> const & a, std::vector<T> const & b, int min_correspondance_length, float epsilon) {
  std::vector<correspondance> res;
  std::unordered_set<int> b_used;
  b_used.reserve(b.size());

  int const sza = a.size();
  int const szb = b.size();
  
  bool in_correspondance = false;
  for(int i=0; i<sza; ++i) {
    int j=0;
    if(in_correspondance) {
      int k = res.back().endB();
      if(k<szb && close(a[i], b[k], epsilon)) {
        // extend the length by one
        ++res.back().length;
        b_used.emplace(k);
        continue;
      }
      // the next item is not a match.
      in_correspondance = false;
      if(res.back().length < min_correspondance_length) {
        // the correspondance is short, so we rewind:
        i -= res.back().length;
        for(int l = res.back().idxB; l<res.back().endB(); ++l) {
          b_used.erase(l);
        }
        // and we will start the search for a longer correspondance right after the beginning of the short one:
        j = res.back().idxB + 1;
        // and we drop the short correspondance
        res.pop_back();
      }
    }
    // find the first matching j in b that is not already used in res
    for( ;j < szb; ++j) {
      if(b_used.find(j) == b_used.end() &&
         close(a[i],b[j],epsilon)) {
        // we found a match
        b_used.emplace(j);
        in_correspondance = true;
        res.emplace_back(i,j);
        break;
      }
    }
  }
  return res;
}

template<typename T>
void verifyVectorsAreEqual(std::vector<T> const & a, std::vector<T> const & b, float epsilon = 1e-5) {
  verify(a.size() == b.size());
  
  
  auto ranges = equalRanges(a, b, epsilon);
  
  if(ranges.empty() || (ranges.size() == 1 && ranges[0].second)) {
    return;
  }
  for(auto [r,ok]:ranges) {
    int n = r.second - r.first;
    std::cout << r.first << ": " << n << " " << (ok ? "success" : "error") << std::endl;
  }
  
  auto corr = correspondances(a, b, 1, epsilon);
  int total = 0;
  for(auto const & c : corr) {
    std::cout << c.idxA << " - " << c.endA() << " -> " << c.idxB << " " << c.endB() << std::endl;
    total += c.length;
  }
  std::cout << "Total matches : " << total << std::endl;

  kill();

}
