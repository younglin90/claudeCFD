#include "TestUtil.hpp"
#include "fvm/FvMatrix.hpp"
#include <cmath>
#include <iostream>

int main() {
  {
    fvm::FvMatrix m(3);
    std::vector<fvm::Triplet> t;
    t.emplace_back(0, 0, 4.0);
    t.emplace_back(0, 1, -1.0);
    t.emplace_back(1, 0, -1.0);
    t.emplace_back(1, 1, 4.0);
    t.emplace_back(1, 2, -1.0);
    t.emplace_back(2, 1, -1.0);
    t.emplace_back(2, 2, 3.0);
    m.A.setFromTriplets(t.begin(), t.end());
    m.b << 15.0, 10.0, 10.0;
    const auto report = m.solveCGWithDiagnostics(1e-12, 100);
    check(report.solverName == "CG/IC", "CG diagnostic solver name");
    check(report.factorizationOk && report.solveOk && report.finite, "CG diagnostic success flags");
    check(report.iterations <= 100, "CG diagnostic iteration count bounded");
    check(report.relativeResidual < 1e-10, "CG diagnostic relative residual");
  }

  {
    fvm::FvMatrix m(3);
    std::vector<fvm::Triplet> t;
    t.emplace_back(0, 0, 3.0);
    t.emplace_back(0, 1, 1.0);
    t.emplace_back(1, 1, 4.0);
    t.emplace_back(1, 2, 2.0);
    t.emplace_back(2, 0, 1.0);
    t.emplace_back(2, 2, 5.0);
    m.A.setFromTriplets(t.begin(), t.end());
    m.b << 1.0, 2.0, 3.0;
    const auto report = m.solveBiCGSTABILUTWithDiagnostics(1e-12, 100);
    check(report.solverName == "BiCGSTAB/ILUT", "BiCGSTAB diagnostic solver name");
    check(report.factorizationOk && report.finite, "BiCGSTAB diagnostic success flags");
    check(report.relativeResidual < 1e-10, "BiCGSTAB diagnostic relative residual");
  }

  {
    fvm::FvMatrix m(2);
    std::vector<fvm::Triplet> t;
    t.emplace_back(0, 0, 2.0);
    t.emplace_back(1, 1, 3.0);
    m.A.setFromTriplets(t.begin(), t.end());
    m.b = Eigen::VectorXd::Zero(2);
    const auto report = m.solveBiCGSTABILUTWithDiagnostics(1e-12, 10);
    check(report.solveOk && report.finite && report.iterations == 0, "zero RHS shortcut diagnostic");
    check(report.relativeResidual == 0.0, "zero RHS residual diagnostic");
  }

  Eigen::VectorXd bad(2);
  bad << 1.0, std::numeric_limits<double>::quiet_NaN();
  check(!fvm::finiteVector(bad), "finiteVector detects NaN");

  std::cout << "linear_solver_diagnostics=pass\n";
}
