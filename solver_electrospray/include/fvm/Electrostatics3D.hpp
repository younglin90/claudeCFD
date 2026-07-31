#pragma once

#include "fvm/FieldOperators3D.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <algorithm>

namespace fvm {

struct PotentialBoundary3D {
  std::vector<char> faceDirichlet;
  ScalarField faceValue;
  std::vector<char> cellFixed;
  ScalarField cellValue;
};

struct PotentialSolveReport3D {
  ScalarField phi;
  VectorField3 E;
  double residual = 0.0;
  int iterations = 0;
};

struct ChargeTransportReport3D {
  ScalarField charge;
  double initialMass = 0.0;
  double finalMass = 0.0;
  double relativeMassDrift = 0.0;
  double minCharge = 0.0;
  double maxCharge = 0.0;
};

inline double harmonicMean(double a, double b) {
  if (a <= 0.0 || b <= 0.0) return 0.0;
  return 2.0 * a * b / (a + b);
}

inline ScalarField facePermittivityHarmonic3D(const Mesh3D& mesh, const ScalarField& eps) {
  require(eps.size() == mesh.cells.size(), "3D eps field size mismatch");
  ScalarField epsF(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    epsF[fi] = f.internal() ? harmonicMean(eps[f.owner], eps[f.neighbour]) : eps[f.owner];
  }
  return epsF;
}

inline double integratedScalar3D(const Mesh3D& mesh, const ScalarField& q) {
  double sum = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) sum += q[c] * mesh.cells[c].V;
  return sum;
}

inline PotentialSolveReport3D solvePotential3D(const Mesh3D& mesh,
                                               const ScalarField& eps,
                                               const ScalarField& rhoE,
                                               const PotentialBoundary3D& bc,
                                               double tol = 1e-12,
                                               int maxIt = 10000) {
  const int n = static_cast<int>(mesh.cells.size());
  require(rhoE.size() == mesh.cells.size(), "3D charge density size mismatch");
  require(bc.faceDirichlet.empty() || bc.faceDirichlet.size() == mesh.faces.size(),
          "3D potential face BC flag size mismatch");
  require(bc.faceValue.empty() || bc.faceValue.size() == mesh.faces.size(),
          "3D potential face BC value size mismatch");
  require(bc.cellFixed.empty() || bc.cellFixed.size() == mesh.cells.size(),
          "3D potential fixed-cell flag size mismatch");
  require(bc.cellValue.empty() || bc.cellValue.size() == mesh.cells.size(),
          "3D potential fixed-cell value size mismatch");

  ScalarField epsF = facePermittivityHarmonic3D(mesh, eps);
  std::vector<Triplet> trips;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
  for (int c = 0; c < n; ++c) b[c] = rhoE[c] * mesh.cells[c].V;

  auto fixed = [&](int c) {
    return !bc.cellFixed.empty() && bc.cellFixed[c] != 0;
  };
  auto fixedValue = [&](int c) {
    return bc.cellValue.empty() ? 0.0 : bc.cellValue[c];
  };

  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    const double coeff = epsF[fi] * std::abs(f.Sf.dot(f.d)) /
                         std::max(f.d.squaredNorm(), 1e-30);
    if (coeff == 0.0) continue;
    if (f.internal()) {
      const bool of = fixed(f.owner);
      const bool nf = fixed(f.neighbour);
      if (!of && !nf) {
        trips.emplace_back(f.owner, f.owner, coeff);
        trips.emplace_back(f.owner, f.neighbour, -coeff);
        trips.emplace_back(f.neighbour, f.neighbour, coeff);
        trips.emplace_back(f.neighbour, f.owner, -coeff);
      } else if (!of && nf) {
        trips.emplace_back(f.owner, f.owner, coeff);
        b[f.owner] += coeff * fixedValue(f.neighbour);
      } else if (of && !nf) {
        trips.emplace_back(f.neighbour, f.neighbour, coeff);
        b[f.neighbour] += coeff * fixedValue(f.owner);
      }
    } else if (!fixed(f.owner) && !bc.faceDirichlet.empty() && bc.faceDirichlet[fi]) {
      trips.emplace_back(f.owner, f.owner, coeff);
      b[f.owner] += coeff * (bc.faceValue.empty() ? 0.0 : bc.faceValue[fi]);
    }
  }

  bool anyFree = false;
  for (int c = 0; c < n; ++c) {
    if (fixed(c)) {
      trips.emplace_back(c, c, 1.0);
      b[c] = fixedValue(c);
    } else {
      anyFree = true;
    }
  }
  if (anyFree && trips.empty()) trips.emplace_back(0, 0, 1.0);

  SpMat A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                           Eigen::IncompleteCholesky<double>> solver;
  solver.setTolerance(tol);
  solver.setMaxIterations(maxIt);
  solver.compute(A);
  require(solver.info() == Eigen::Success, "3D potential CG/IC factorization failed");
  Eigen::VectorXd x = solver.solve(b);
  require(solver.info() == Eigen::Success, "3D potential CG/IC solve failed");

  PotentialSolveReport3D report;
  report.phi.assign(mesh.cells.size(), 0.0);
  for (int c = 0; c < n; ++c) report.phi[c] = x[c];
  VectorField3 gradPhi = gradLeastSquares3D(mesh, report.phi);
  report.E.resize(mesh.cells.size(), Vec3::Zero());
  for (size_t c = 0; c < mesh.cells.size(); ++c) report.E[c] = -gradPhi[c];
  report.residual = (A * x - b).norm() / std::max(b.norm(), 1e-30);
  report.iterations = solver.iterations();
  return report;
}

inline ChargeTransportReport3D transportChargeBounded3D(const Mesh3D& mesh,
                                                        const ScalarField& q0,
                                                        const ScalarField& faceFlux,
                                                        double dt,
                                                        double qMin,
                                                        double qMax) {
  require(q0.size() == mesh.cells.size(), "3D charge field size mismatch");
  require(faceFlux.size() == mesh.faces.size(), "3D charge flux size mismatch");
  ScalarField q = q0;
  const double initial = integratedScalar3D(mesh, q0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    if (!f.internal()) continue;
    const double mdot = faceFlux[fi];
    const int up = mdot >= 0.0 ? f.owner : f.neighbour;
    const double flux = mdot * q0[up] * dt;
    q[f.owner] -= flux / mesh.cells[f.owner].V;
    q[f.neighbour] += flux / mesh.cells[f.neighbour].V;
  }
  for (double& v : q) v = std::clamp(v, qMin, qMax);
  const double clippedMass = integratedScalar3D(mesh, q);
  const double diff = initial - clippedMass;
  double capacity = 0.0;
  if (diff > 0.0) {
    for (size_t c = 0; c < mesh.cells.size(); ++c) capacity += (qMax - q[c]) * mesh.cells[c].V;
    if (capacity > 1e-30) {
      for (size_t c = 0; c < mesh.cells.size(); ++c) q[c] += diff * (qMax - q[c]) / capacity;
    }
  } else if (diff < 0.0) {
    for (size_t c = 0; c < mesh.cells.size(); ++c) capacity += (q[c] - qMin) * mesh.cells[c].V;
    if (capacity > 1e-30) {
      for (size_t c = 0; c < mesh.cells.size(); ++c) q[c] += diff * (q[c] - qMin) / capacity;
    }
  }
  for (double& v : q) v = std::clamp(v, qMin, qMax);
  ChargeTransportReport3D report;
  report.charge = q;
  report.initialMass = initial;
  report.finalMass = integratedScalar3D(mesh, q);
  report.relativeMassDrift =
      std::abs(report.finalMass - report.initialMass) / std::max(std::abs(report.initialMass), 1e-30);
  report.minCharge = *std::min_element(q.begin(), q.end());
  report.maxCharge = *std::max_element(q.begin(), q.end());
  return report;
}

}
