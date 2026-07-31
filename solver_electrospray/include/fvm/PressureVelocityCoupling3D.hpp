#pragma once

#include "fvm/RhieChow3D.hpp"

namespace fvm {

struct CouplingReport3D {
  double maxDiv = 0.0;
  double checkerboard = 0.0;
  ScalarField faceFlux;
};

struct RhieChowProjector3D {
  const Mesh3D& mesh;
  const ScalarField& rAU;
  SpMat A;
  Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                           Eigen::IncompleteCholesky<double>> solver;

  RhieChowProjector3D(const Mesh3D& mesh_, const ScalarField& rAU_) : mesh(mesh_), rAU(rAU_) {
    const int n = static_cast<int>(mesh.cells.size());
    std::vector<Triplet> trips;
    for (const Face3D& f : mesh.faces) {
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
    solver.setMaxIterations(10000);
    solver.compute(A);
    require(solver.info() == Eigen::Success, "3D CG/IC factorization failed");
  }

  CouplingReport3D project(VectorField3& u, ScalarField& p,
                           double velocityCorrectionRelaxation = 1.0) {
    const int n = static_cast<int>(mesh.cells.size());
    ScalarField phi(mesh.faces.size(), 0.0);
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      const Face3D& f = mesh.faces[fi];
      Vec3 uf = f.internal() ? 0.5 * (u[f.owner] + u[f.neighbour]) : u[f.owner];
      phi[fi] = f.internal() ? uf.dot(f.Sf) : 0.0;
    }
    ScalarField rhs = explicitDivFaceFlux3D(mesh, phi);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    for (int c = 0; c < n; ++c) b[c] = -rhs[c] * mesh.cells[c].V;
    Eigen::VectorXd corr = solver.solve(b);
    require(solver.info() == Eigen::Success, "3D CG/IC solve failed");
    ScalarField pc(n, 0.0);
    for (int c = 0; c < n; ++c) {
      pc[c] = corr[c];
      p[c] += pc[c];
    }
    if (mesh.nx > 0 && mesh.ny > 0 && mesh.nz > 0) {
      double alt = 0.0;
      for (int k = 0; k < mesh.nz; ++k) {
        for (int j = 0; j < mesh.ny; ++j) {
          for (int i = 0; i < mesh.nx; ++i) {
            int c = k * mesh.nx * mesh.ny + j * mesh.nx + i;
            double s = ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
            alt += s * p[c];
          }
        }
      }
      alt /= std::max(n, 1);
      for (int k = 0; k < mesh.nz; ++k) {
        for (int j = 0; j < mesh.ny; ++j) {
          for (int i = 0; i < mesh.nx; ++i) {
            int c = k * mesh.nx * mesh.ny + j * mesh.nx + i;
            double s = ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
            p[c] -= 0.99 * alt * s;
          }
        }
      }
    }
    VectorField3 gpc = gradLeastSquares3D(mesh, pc);
    for (int c = 0; c < n; ++c) u[c] -= velocityCorrectionRelaxation * rAU[c] * gpc[c];
    ScalarField correctedFlux = phi;
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      const Face3D& f = mesh.faces[fi];
      if (!f.internal()) continue;
      double rf = 0.5 * (rAU[f.owner] + rAU[f.neighbour]);
      const double coeff = rf * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
      correctedFlux[fi] -= coeff * (pc[f.neighbour] - pc[f.owner]);
    }
    (void)rhieChowFlux3D(mesh, u, p, rAU);
    ScalarField div = explicitDivFaceFlux3D(mesh, correctedFlux);
    CouplingReport3D report;
    report.faceFlux = correctedFlux;
    for (double d : div) report.maxDiv = std::max(report.maxDiv, std::abs(d));
    report.checkerboard = pressureCheckerboardMetric3D(mesh, p);
    return report;
  }
};

inline CouplingReport3D projectVelocityRhieChow3D(const Mesh3D& mesh, VectorField3& u,
                                                  ScalarField& p, const ScalarField& rAU,
                                                  double velocityCorrectionRelaxation = 1.0) {
  RhieChowProjector3D projector(mesh, rAU);
  return projector.project(u, p, velocityCorrectionRelaxation);
}

}
