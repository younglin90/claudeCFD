#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace fvm {

using Vec = Eigen::Vector2d;
using VectorField = std::vector<Vec>;
using ScalarField = std::vector<double>;
using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

inline double sqr(double x) { return x * x; }

inline void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}

inline double vanLeer(double r) {
  return (r + std::abs(r)) / (1.0 + std::abs(r) + 1e-30);
}

}
