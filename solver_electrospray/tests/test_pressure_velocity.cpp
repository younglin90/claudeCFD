#include "TestUtil.hpp"
#include <fstream>

int main() {
  auto mesh = fvm::Mesh::quadGrid(24, 24, 1.0, 1.0, 0.08);
  fvm::VectorField u(mesh.cells.size(), fvm::Vec::Zero());
  fvm::ScalarField p(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double x = mesh.cells[c].centroid.x();
    double y = mesh.cells[c].centroid.y();
    u[c] = {0.08 * std::sin(M_PI * x) * std::cos(M_PI * y),
            -0.05 * std::cos(M_PI * x) * std::sin(M_PI * y)};
    p[c] = ((static_cast<int>(c) / mesh.nx + static_cast<int>(c) % mesh.nx) % 2 == 0) ? 1.0 : -1.0;
  }
  double before = fvm::pressureCheckerboardMetric(mesh, p);
  auto report = fvm::projectVelocityRhieChow(mesh, u, p, 0.01);
  fvm::ensureLogDir();
  std::ofstream csv("benchmark_logs/continuity_checkerboard.csv");
  csv << "max_div,checkerboard_before,checkerboard_after\n";
  csv << report.maxDiv << "," << before << "," << report.checkerboard << "\n";
  std::cout << "continuity_max_div=" << report.maxDiv << " checkerboard_before=" << before
            << " checkerboard_after=" << report.checkerboard << "\n";
  check(report.maxDiv < 1e-9, "projection continuity residual at solver tolerance on skew collocated mesh");
  check(report.checkerboard < before, "Rhie-Chow damps checkerboard pressure mode");
}
