#pragma once

#include "fvm/FvMatrix.hpp"
#include "fvm/Mesh3D.hpp"

namespace fvm {

struct TaylorGreen3DReport {
  double finalTime = 0.0;
  double computedEnergy = 0.0;
  double analyticEnergy = 0.0;
  double energyError = 0.0;
  double computedEnstrophy = 0.0;
  double analyticEnstrophy = 0.0;
  double enstrophyError = 0.0;
};

inline int periodicId3D(int n, int i, int j, int k) {
  i = (i + n) % n;
  j = (j + n) % n;
  k = (k + n) % n;
  return k * n * n + j * n + i;
}

inline Eigen::VectorXd solvePeriodicScalarDiffusion3D(int n, const Eigen::VectorXd& old,
                                                      double nu, double dt) {
  const double h = 1.0 / static_cast<double>(n);
  const double a0 = h * h * h / dt;
  const double aN = nu * h;
  FvMatrix m(n * n * n);
  std::vector<Triplet> trips;
  trips.reserve(static_cast<size_t>(7 * n * n * n));
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        int c = periodicId3D(n, i, j, k);
        trips.emplace_back(c, c, a0 + 6.0 * aN);
        trips.emplace_back(c, periodicId3D(n, i + 1, j, k), -aN);
        trips.emplace_back(c, periodicId3D(n, i - 1, j, k), -aN);
        trips.emplace_back(c, periodicId3D(n, i, j + 1, k), -aN);
        trips.emplace_back(c, periodicId3D(n, i, j - 1, k), -aN);
        trips.emplace_back(c, periodicId3D(n, i, j, k + 1), -aN);
        trips.emplace_back(c, periodicId3D(n, i, j, k - 1), -aN);
        m.b[c] = a0 * old[c];
      }
    }
  }
  m.A.setFromTriplets(trips.begin(), trips.end());
  return m.solveBiCGSTABILUT(1e-11, 10000);
}

inline TaylorGreen3DReport runTaylorGreen3D(int n, double nu, double tEnd, double dt) {
  Eigen::VectorXd ux(n * n * n), uy(n * n * n), uz(n * n * n);
  const double h = 1.0 / static_cast<double>(n);
  for (int k = 0; k < n; ++k) {
    double z = (static_cast<double>(k) + 0.5) * h;
    for (int j = 0; j < n; ++j) {
      double y = (static_cast<double>(j) + 0.5) * h;
      for (int i = 0; i < n; ++i) {
        double x = (static_cast<double>(i) + 0.5) * h;
        int c = periodicId3D(n, i, j, k);
        ux[c] = std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y) * std::cos(2.0 * M_PI * z);
        uy[c] = -std::cos(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y) * std::cos(2.0 * M_PI * z);
        uz[c] = 0.0;
      }
    }
  }
  int steps = static_cast<int>(std::ceil(tEnd / dt));
  double actualDt = tEnd / std::max(steps, 1);
  for (int step = 0; step < steps; ++step) {
    ux = solvePeriodicScalarDiffusion3D(n, ux, nu, actualDt);
    uy = solvePeriodicScalarDiffusion3D(n, uy, nu, actualDt);
    uz = solvePeriodicScalarDiffusion3D(n, uz, nu, actualDt);
  }

  double energy = 0.0;
  double enstrophy = 0.0;
  const double inv2h = 0.5 / h;
  const double cellV = h * h * h;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        int c = periodicId3D(n, i, j, k);
        energy += 0.5 * (ux[c] * ux[c] + uy[c] * uy[c] + uz[c] * uz[c]) * cellV;
        auto vx = [&](int ii, int jj, int kk) { return ux[periodicId3D(n, ii, jj, kk)]; };
        auto vy = [&](int ii, int jj, int kk) { return uy[periodicId3D(n, ii, jj, kk)]; };
        auto vz = [&](int ii, int jj, int kk) { return uz[periodicId3D(n, ii, jj, kk)]; };
        double wx = (vz(i, j + 1, k) - vz(i, j - 1, k)) * inv2h -
                    (vy(i, j, k + 1) - vy(i, j, k - 1)) * inv2h;
        double wy = (vx(i, j, k + 1) - vx(i, j, k - 1)) * inv2h -
                    (vz(i + 1, j, k) - vz(i - 1, j, k)) * inv2h;
        double wz = (vy(i + 1, j, k) - vy(i - 1, j, k)) * inv2h -
                    (vx(i, j + 1, k) - vx(i, j - 1, k)) * inv2h;
        enstrophy += 0.5 * (wx * wx + wy * wy + wz * wz) * cellV;
      }
    }
  }

  const double decay = std::exp(-12.0 * M_PI * M_PI * nu * tEnd);
  const double analyticEnergy = 0.125 * decay * decay;
  const double analyticEnstrophy = 12.0 * M_PI * M_PI * analyticEnergy;
  TaylorGreen3DReport r;
  r.finalTime = tEnd;
  r.computedEnergy = energy;
  r.analyticEnergy = analyticEnergy;
  r.energyError = std::abs(energy - analyticEnergy) / std::max(analyticEnergy, 1e-30);
  r.computedEnstrophy = enstrophy;
  r.analyticEnstrophy = analyticEnstrophy;
  r.enstrophyError = std::abs(enstrophy - analyticEnstrophy) / std::max(analyticEnstrophy, 1e-30);
  return r;
}

}
