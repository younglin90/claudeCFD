// Periodic pressure Helmholtz assembly/solve.
#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "cfd/five_eq/linear_solvers.hpp"

namespace cfd::five_eq {

struct PeriodicHelmholtz {
    std::vector<double> lower, diagonal, upper;
    double corner_lu = 0.0, corner_ul = 0.0;
};

inline PeriodicHelmholtz assemble_helmholtz_periodic(const std::vector<double>& sigma_pp,
                                                     const std::vector<double>& rho_eff,
                                                     double gamma_dt, double dx) {
    const int n = static_cast<int>(sigma_pp.size());
    if (n < 2 || static_cast<int>(rho_eff.size()) != n || gamma_dt <= 0.0 || dx <= 0.0)
        throw std::invalid_argument("invalid periodic Helmholtz inputs");
    std::vector<double> k_face(n);
    for (int i = 0; i < n; ++i) {
        const double rho_face = 0.5 * (rho_eff[i] + rho_eff[(i + 1) % n]);
        k_face[i] = gamma_dt / (std::max(rho_face, 1.e-30) * dx * dx);
    }
    PeriodicHelmholtz result;
    result.lower.resize(n - 1); result.upper.resize(n - 1); result.diagonal.resize(n);
    for (int i = 0; i < n; ++i) result.diagonal[i] = sigma_pp[i] / gamma_dt + k_face[i] + k_face[(i + n - 1) % n];
    for (int i = 0; i < n - 1; ++i) result.lower[i] = result.upper[i] = -k_face[i];
    result.corner_lu = result.corner_ul = -k_face[n - 1];
    return result;
}

inline std::vector<double> solve_helmholtz_periodic(const std::vector<double>& sigma_pp,
                                                     const std::vector<double>& rho_eff,
                                                     double gamma_dt, double dx,
                                                     const std::vector<double>& rhs) {
    const auto matrix = assemble_helmholtz_periodic(sigma_pp, rho_eff, gamma_dt, dx);
    return solve_periodic_tridiag(matrix.lower, matrix.diagonal, matrix.upper, rhs,
                                  matrix.corner_lu, matrix.corner_ul);
}

} // namespace cfd::five_eq
