#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace electrospray {

struct AxisymmetricRadialSolution {
  std::vector<double> rCenters;
  std::vector<double> phi;
  std::vector<double> eRadial;
};

inline AxisymmetricRadialSolution solveRadialPoissonAxisymmetric(
    const std::vector<double>& rFaces,
    const std::vector<double>& epsilonCells,
    double phiOuter,
    const std::vector<double>& chargeDensityCells = {}) {
  if (rFaces.size() != epsilonCells.size() + 1) {
    throw std::runtime_error("r_faces must be one-dimensional with len(epsilon_cells)+1 entries");
  }
  if (rFaces.empty() || rFaces.front() != 0.0) throw std::runtime_error("r_faces must start at zero and increase");
  for (size_t i = 1; i < rFaces.size(); ++i) {
    if (!(rFaces[i] > rFaces[i - 1])) throw std::runtime_error("r_faces must start at zero and increase");
  }
  for (double eps : epsilonCells) {
    if (!(eps > 0.0)) throw std::runtime_error("epsilon must be positive");
  }
  if (!chargeDensityCells.empty() && chargeDensityCells.size() != epsilonCells.size()) {
    throw std::runtime_error("charge_density_cells must match epsilon_cells");
  }

  const int n = static_cast<int>(epsilonCells.size());
  std::vector<double> dr(static_cast<size_t>(n), 0.0);
  std::vector<double> rc(static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    dr[static_cast<size_t>(i)] = rFaces[static_cast<size_t>(i + 1)] - rFaces[static_cast<size_t>(i)];
    rc[static_cast<size_t>(i)] = 0.5 * (rFaces[static_cast<size_t>(i)] + rFaces[static_cast<size_t>(i + 1)]);
  }

  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; ++i) {
    const double rho = chargeDensityCells.empty() ? 0.0 : chargeDensityCells[static_cast<size_t>(i)];
    rhs(i) = rho * M_PI *
             (rFaces[static_cast<size_t>(i + 1)] * rFaces[static_cast<size_t>(i + 1)] -
              rFaces[static_cast<size_t>(i)] * rFaces[static_cast<size_t>(i)]);

    double gw = 0.0;
    if (i > 0) {
      gw = 2.0 * M_PI * rFaces[static_cast<size_t>(i)] /
           (0.5 * dr[static_cast<size_t>(i - 1)] / epsilonCells[static_cast<size_t>(i - 1)] +
            0.5 * dr[static_cast<size_t>(i)] / epsilonCells[static_cast<size_t>(i)]);
      matrix(i, i - 1) -= gw;
    }
    double ge = 0.0;
    if (i == n - 1) {
      ge = 2.0 * M_PI * rFaces[static_cast<size_t>(i + 1)] * epsilonCells[static_cast<size_t>(i)] /
           (0.5 * dr[static_cast<size_t>(i)]);
      rhs(i) += ge * phiOuter;
    } else {
      ge = 2.0 * M_PI * rFaces[static_cast<size_t>(i + 1)] /
           (0.5 * dr[static_cast<size_t>(i)] / epsilonCells[static_cast<size_t>(i)] +
            0.5 * dr[static_cast<size_t>(i + 1)] / epsilonCells[static_cast<size_t>(i + 1)]);
      matrix(i, i + 1) -= ge;
    }
    matrix(i, i) += gw + ge;
  }

  Eigen::VectorXd phiVec = matrix.colPivHouseholderQr().solve(rhs);
  std::vector<double> phi(static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) phi[static_cast<size_t>(i)] = phiVec(i);

  std::vector<double> facePhi(static_cast<size_t>(n + 1), 0.0);
  facePhi.front() = phi.front();
  facePhi.back() = phiOuter;
  for (int i = 1; i < n; ++i) {
    const double conductancePerArea =
        1.0 / (0.5 * dr[static_cast<size_t>(i - 1)] / epsilonCells[static_cast<size_t>(i - 1)] +
               0.5 * dr[static_cast<size_t>(i)] / epsilonCells[static_cast<size_t>(i)]);
    const double displacement = -conductancePerArea * (phi[static_cast<size_t>(i)] - phi[static_cast<size_t>(i - 1)]);
    facePhi[static_cast<size_t>(i)] =
        phi[static_cast<size_t>(i - 1)] -
        displacement * 0.5 * dr[static_cast<size_t>(i - 1)] / epsilonCells[static_cast<size_t>(i - 1)];
  }

  std::vector<double> eRadial(static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    eRadial[static_cast<size_t>(i)] =
        -(facePhi[static_cast<size_t>(i + 1)] - facePhi[static_cast<size_t>(i)]) / dr[static_cast<size_t>(i)];
  }
  return {rc, phi, eRadial};
}

}  // namespace electrospray
