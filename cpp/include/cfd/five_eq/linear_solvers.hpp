// Small one-dimensional banded linear solvers.
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cfd::five_eq {

inline std::vector<double> solve_tridiag(std::vector<double> lower,
                                         std::vector<double> diagonal,
                                         std::vector<double> upper,
                                         std::vector<double> rhs) {
    const int n = static_cast<int>(diagonal.size());
    if (n == 0) return {};
    if (static_cast<int>(lower.size()) != n - 1 || static_cast<int>(upper.size()) != n - 1 ||
        static_cast<int>(rhs.size()) != n) throw std::invalid_argument("invalid tridiagonal dimensions");
    for (int i = 1; i < n; ++i) {
        if (std::fabs(diagonal[i - 1]) < 1.e-30) throw std::runtime_error("zero tridiagonal pivot");
        const double factor = lower[i - 1] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    if (std::fabs(diagonal[n - 1]) < 1.e-30) throw std::runtime_error("zero tridiagonal pivot");
    std::vector<double> result(n);
    result[n - 1] = rhs[n - 1] / diagonal[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        if (std::fabs(diagonal[i]) < 1.e-30) throw std::runtime_error("zero tridiagonal pivot");
        result[i] = (rhs[i] - upper[i] * result[i + 1]) / diagonal[i];
    }
    return result;
}

inline std::vector<double> solve_periodic_tridiag(const std::vector<double>& lower,
                                                  const std::vector<double>& diagonal,
                                                  const std::vector<double>& upper,
                                                  const std::vector<double>& rhs,
                                                  double corner_lu, double corner_ul) {
    const int n = static_cast<int>(diagonal.size());
    if (n < 2) throw std::invalid_argument("periodic tridiagonal requires n >= 2");
    auto x0 = solve_tridiag(lower, diagonal, upper, rhs);
    std::vector<double> e0(n, 0.0), en(n, 0.0);
    e0[0] = 1.0; en[n - 1] = 1.0;
    const auto y0 = solve_tridiag(lower, diagonal, upper, e0);
    const auto y1 = solve_tridiag(lower, diagonal, upper, en);
    const double b0 = corner_lu * x0[n - 1];
    const double b1 = corner_ul * x0[0];
    const double m00 = 1.0 + corner_lu * y0[n - 1];
    const double m01 = corner_lu * y1[n - 1];
    const double m10 = corner_ul * y0[0];
    const double m11 = 1.0 + corner_ul * y1[0];
    const double determinant = m00 * m11 - m01 * m10;
    if (std::fabs(determinant) < 1.e-30) throw std::runtime_error("singular cyclic correction");
    const double z0 = (m11 * b0 - m01 * b1) / determinant;
    const double z1 = (-m10 * b0 + m00 * b1) / determinant;
    for (int i = 0; i < n; ++i) x0[i] -= y0[i] * z0 + y1[i] * z1;
    return x0;
}

} // namespace cfd::five_eq
