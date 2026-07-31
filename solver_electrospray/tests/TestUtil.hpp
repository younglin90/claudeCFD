#pragma once

#include "fvm/Benchmarks.hpp"
#include <iostream>

inline void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    std::exit(1);
  }
}

inline double l2(const fvm::ScalarField& a, const fvm::ScalarField& b) {
  double e = 0.0, n = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    e += fvm::sqr(a[i] - b[i]);
    n += fvm::sqr(b[i]);
  }
  return std::sqrt(e / std::max(n, 1e-30));
}
