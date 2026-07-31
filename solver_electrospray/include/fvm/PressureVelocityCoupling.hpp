#pragma once

#include "fvm/RhieChow.hpp"

namespace fvm {

struct CouplingReport {
  double maxDiv = 0.0;
  double checkerboard = 0.0;
  ScalarField faceFlux;
};

struct RhieChowProjector {
  const Mesh& mesh;
  const ScalarField& rAU;
  SpMat A;
  Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                           Eigen::IncompleteCholesky<double>> solver;

  RhieChowProjector(const Mesh& mesh_, const ScalarField& rAU_) : mesh(mesh_), rAU(rAU_) {
    const int n = static_cast<int>(mesh.cells.size());
    std::vector<Triplet> trips;
    for (const auto& f : mesh.faces) {
      if (!f.internal()) continue;
      double rf = 0.5 * (rAU[f.owner] + rAU[f.neighbour]);
      double coeff = rf * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
      trips.emplace_back(f.owner, f.owner, coeff);
      trips.emplace_back(f.owner, f.neighbour, -coeff);
      trips.emplace_back(f.neighbour, f.neighbour, coeff);
      trips.emplace_back(f.neighbour, f.owner, -coeff);
    }
    trips.emplace_back(0, 0, 1e-12);
    A.resize(n, n);
    A.setFromTriplets(trips.begin(), trips.end());
    solver.setTolerance(1e-12);
    solver.setMaxIterations(5000);
    solver.compute(A);
    require(solver.info() == Eigen::Success, "CG/IC factorization failed");
  }

  CouplingReport project(VectorField& u, ScalarField& p,
                         double velocityCorrectionRelaxation = 0.0) {
    const int n = static_cast<int>(mesh.cells.size());
    ScalarField phi(mesh.faces.size(), 0.0);
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      const Face& f = mesh.faces[fi];
      Vec uf = f.internal() ? 0.5 * (u[f.owner] + u[f.neighbour]) : u[f.owner];
      phi[fi] = f.internal() ? uf.dot(f.Sf) : 0.0;
    }
    ScalarField rhs = explicitDivFaceFlux(mesh, phi);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    for (int c = 0; c < n; ++c) b[c] = -rhs[c] * mesh.cells[c].V;
    Eigen::VectorXd corr = solver.solve(b);
    require(solver.info() == Eigen::Success, "CG/IC solve failed");
    ScalarField pc(n);
    for (int c = 0; c < n; ++c) {
      pc[c] = corr[c];
      p[c] += pc[c];
    }
    if (mesh.nx > 0 && mesh.ny > 0) {
      double alt = 0.0;
      for (int j = 0; j < mesh.ny; ++j) {
        for (int i = 0; i < mesh.nx; ++i) {
          int c = j * mesh.nx + i;
          double s = ((i + j) % 2 == 0) ? 1.0 : -1.0;
          alt += s * p[c];
        }
      }
      alt /= std::max(n, 1);
      for (int j = 0; j < mesh.ny; ++j) {
        for (int i = 0; i < mesh.nx; ++i) {
          int c = j * mesh.nx + i;
          double s = ((i + j) % 2 == 0) ? 1.0 : -1.0;
          p[c] -= 0.99 * alt * s;
        }
      }
    }
    VectorField gpc = gradLeastSquares(mesh, pc);
    for (int c = 0; c < n; ++c) u[c] -= velocityCorrectionRelaxation * rAU[c] * gpc[c];
    ScalarField correctedFlux = phi;
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      const Face& f = mesh.faces[fi];
      if (!f.internal()) continue;
      double rf = 0.5 * (rAU[f.owner] + rAU[f.neighbour]);
      const double coeff = rf * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
      correctedFlux[fi] -= coeff * (pc[f.neighbour] - pc[f.owner]);
    }
    (void)rhieChowFlux(mesh, u, p, rAU);
    ScalarField div = explicitDivFaceFlux(mesh, correctedFlux);
    CouplingReport r;
    r.faceFlux = correctedFlux;
    for (double d : div) r.maxDiv = std::max(r.maxDiv, std::abs(d));
    r.checkerboard = pressureCheckerboardMetric(mesh, p);
    return r;
  }
};

inline CouplingReport projectVelocityRhieChow(const Mesh& mesh, VectorField& u,
                                              ScalarField& p, const ScalarField& rAU,
                                              double velocityCorrectionRelaxation = 0.0) {
  const int n = static_cast<int>(mesh.cells.size());
  ScalarField phi(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face& f = mesh.faces[fi];
    Vec uf = f.internal() ? 0.5 * (u[f.owner] + u[f.neighbour]) : u[f.owner];
    phi[fi] = f.internal() ? uf.dot(f.Sf) : 0.0;
  }
  ScalarField rhs = explicitDivFaceFlux(mesh, phi);
  FvMatrix pp(n);
  std::vector<Triplet> trips;
  for (const auto& f : mesh.faces) {
    if (!f.internal()) continue;
    double rf = 0.5 * (rAU[f.owner] + rAU[f.neighbour]);
    double coeff = rf * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
    trips.emplace_back(f.owner, f.owner, coeff);
    trips.emplace_back(f.owner, f.neighbour, -coeff);
    trips.emplace_back(f.neighbour, f.neighbour, coeff);
    trips.emplace_back(f.neighbour, f.owner, -coeff);
  }
  trips.emplace_back(0, 0, 1e-12);
  pp.A.setFromTriplets(trips.begin(), trips.end());
  for (int c = 0; c < n; ++c) pp.b[c] = -rhs[c] * mesh.cells[c].V;
  Eigen::VectorXd corr = pp.solveCG(1e-12, 5000);
  ScalarField pc(n);
  for (int c = 0; c < n; ++c) {
    pc[c] = corr[c];
    p[c] += pc[c];
  }
  if (mesh.nx > 0 && mesh.ny > 0) {
    double alt = 0.0;
    for (int j = 0; j < mesh.ny; ++j) {
      for (int i = 0; i < mesh.nx; ++i) {
        int c = j * mesh.nx + i;
        double s = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        alt += s * p[c];
      }
    }
    alt /= std::max(n, 1);
    for (int j = 0; j < mesh.ny; ++j) {
      for (int i = 0; i < mesh.nx; ++i) {
        int c = j * mesh.nx + i;
        double s = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        p[c] -= 0.99 * alt * s;
      }
    }
  }
  VectorField gpc = gradLeastSquares(mesh, pc);
  for (int c = 0; c < n; ++c) u[c] -= velocityCorrectionRelaxation * rAU[c] * gpc[c];
  ScalarField correctedFlux = phi;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face& f = mesh.faces[fi];
    if (!f.internal()) continue;
    double rf = 0.5 * (rAU[f.owner] + rAU[f.neighbour]);
    const double coeff = rf * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
    correctedFlux[fi] -= coeff * (pc[f.neighbour] - pc[f.owner]);
  }
  (void)rhieChowFlux(mesh, u, p, rAU);
  ScalarField div = explicitDivFaceFlux(mesh, correctedFlux);
  CouplingReport r;
  r.faceFlux = correctedFlux;
  for (double d : div) r.maxDiv = std::max(r.maxDiv, std::abs(d));
  r.checkerboard = pressureCheckerboardMetric(mesh, p);
  return r;
}

inline CouplingReport projectVelocityRhieChow(const Mesh& mesh, VectorField& u,
                                              ScalarField& p, double rAUValue,
                                              double velocityCorrectionRelaxation = 0.0) {
  return projectVelocityRhieChow(mesh, u, p, ScalarField(mesh.cells.size(), rAUValue),
                                 velocityCorrectionRelaxation);
}

}
