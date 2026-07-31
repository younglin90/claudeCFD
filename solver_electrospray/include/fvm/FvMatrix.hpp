#pragma once

#include "fvm/FieldOperators.hpp"
#include <cmath>
#include <limits>
#include <string>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace fvm {

struct LinearSolveReport {
  Eigen::VectorXd x;
  int iterations = 0;
  double estimatedError = std::numeric_limits<double>::infinity();
  double relativeResidual = std::numeric_limits<double>::infinity();
  bool factorizationOk = false;
  bool solveOk = false;
  bool finite = false;
  std::string solverName;
};

inline bool finiteVector(const Eigen::VectorXd& x) {
  for (int i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i])) return false;
  }
  return true;
}

struct FvMatrix {
  SpMat A;
  Eigen::VectorXd b;

  explicit FvMatrix(int n = 0) : A(n, n), b(Eigen::VectorXd::Zero(n)) {}

  LinearSolveReport solveCGWithDiagnostics(double tol = 1e-11, int maxIt = 2000) const {
    Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                             Eigen::IncompleteCholesky<double>> solver;
    solver.setTolerance(tol);
    solver.setMaxIterations(maxIt);
    solver.compute(A);
    LinearSolveReport report;
    report.solverName = "CG/IC";
    report.factorizationOk = solver.info() == Eigen::Success;
    require(report.factorizationOk, "CG/IC factorization failed");
    report.x = solver.solve(b);
    report.iterations = solver.iterations();
    report.estimatedError = solver.error();
    report.solveOk = solver.info() == Eigen::Success;
    report.finite = finiteVector(report.x);
    report.relativeResidual =
        (A * report.x - b).norm() / std::max(b.norm(), 1e-30);
    require(report.solveOk, "CG/IC solve failed");
    require(report.finite, "CG/IC solve produced non-finite values");
    return report;
  }

  Eigen::VectorXd solveCG(double tol = 1e-11, int maxIt = 2000) const {
    return solveCGWithDiagnostics(tol, maxIt).x;
  }

  LinearSolveReport solveBiCGSTABILUTWithDiagnostics(double tol = 1e-11, int maxIt = 2000) const {
    LinearSolveReport report;
    report.solverName = "BiCGSTAB/ILUT";
    if (b.norm() == 0.0) {
      report.x = Eigen::VectorXd::Zero(b.size());
      report.iterations = 0;
      report.estimatedError = 0.0;
      report.relativeResidual = 0.0;
      report.factorizationOk = true;
      report.solveOk = true;
      report.finite = true;
      return report;
    }
    Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>> solver;
    solver.preconditioner().setDroptol(1e-6);
    solver.preconditioner().setFillfactor(50);
    solver.setTolerance(tol);
    solver.setMaxIterations(maxIt);
    solver.compute(A);
    report.factorizationOk = solver.info() == Eigen::Success;
    require(report.factorizationOk, "BiCGSTAB/ILUT factorization failed");
    report.x = solver.solve(b);
    report.iterations = solver.iterations();
    report.estimatedError = solver.error();
    report.solveOk = solver.info() == Eigen::Success;
    report.finite = finiteVector(report.x);
    report.relativeResidual = (A * report.x - b).norm() / std::max(b.norm(), 1e-30);
    require(report.solveOk || report.relativeResidual < std::max(1e-8, 10.0 * tol),
            "BiCGSTAB/ILUT solve failed rel=" + std::to_string(report.relativeResidual));
    require(report.finite, "BiCGSTAB/ILUT solve produced non-finite values");
    return report;
  }

  Eigen::VectorXd solveBiCGSTABILUT(double tol = 1e-11, int maxIt = 2000) const {
    return solveBiCGSTABILUTWithDiagnostics(tol, maxIt).x;
  }
};

inline FvMatrix ddtMatrix(const Mesh& mesh, double rho, double dt, const ScalarField& old) {
  const int n = static_cast<int>(mesh.cells.size());
  std::vector<Triplet> t;
  FvMatrix m(n);
  for (int c = 0; c < n; ++c) {
    double a = rho * mesh.cells[c].V / dt;
    t.emplace_back(c, c, a);
    m.b[c] += a * old[c];
  }
  m.A.setFromTriplets(t.begin(), t.end());
  return m;
}

inline void addImplicitDivergenceUpwind(FvMatrix& m, const Mesh& mesh,
                                        const ScalarField& faceFlux) {
  std::vector<Triplet> t;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face& f = mesh.faces[fi];
    if (!f.internal()) continue;
    const double mdot = faceFlux[fi];
    if (mdot >= 0.0) {
      t.emplace_back(f.owner, f.owner, mdot);
      t.emplace_back(f.neighbour, f.owner, -mdot);
    } else {
      t.emplace_back(f.owner, f.neighbour, mdot);
      t.emplace_back(f.neighbour, f.neighbour, -mdot);
    }
  }
  SpMat D(m.A.rows(), m.A.cols());
  D.setFromTriplets(t.begin(), t.end());
  m.A += D;
}

inline void addLaplacian(FvMatrix& m, const Mesh& mesh, double gamma,
                         const ScalarField* boundary = nullptr) {
  std::vector<Triplet> t;
  for (const auto& f : mesh.faces) {
    double coeff = gamma * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
    if (f.internal()) {
      t.emplace_back(f.owner, f.owner, coeff);
      t.emplace_back(f.owner, f.neighbour, -coeff);
      t.emplace_back(f.neighbour, f.neighbour, coeff);
      t.emplace_back(f.neighbour, f.owner, -coeff);
    } else {
      t.emplace_back(f.owner, f.owner, coeff);
      if (boundary) m.b[f.owner] += coeff * (*boundary)[f.owner];
    }
  }
  SpMat L(m.A.rows(), m.A.cols());
  L.setFromTriplets(t.begin(), t.end());
  m.A += L;
}

inline FvMatrix diffusionMatrix(const Mesh& mesh, double gamma) {
  FvMatrix m(static_cast<int>(mesh.cells.size()));
  m.A.resize(mesh.cells.size(), mesh.cells.size());
  addLaplacian(m, mesh, gamma);
  return m;
}

inline FvMatrix poissonMatrix(const Mesh& mesh) {
  FvMatrix m = diffusionMatrix(mesh, 1.0);
  std::vector<Triplet> t;
  for (int k = 0; k < m.A.outerSize(); ++k) {
    for (SpMat::InnerIterator it(m.A, k); it; ++it) t.emplace_back(it.row(), it.col(), it.value());
  }
  t.emplace_back(0, 0, 1.0);
  m.A.setFromTriplets(t.begin(), t.end());
  m.b[0] = 0.0;
  return m;
}

}
