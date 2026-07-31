#include "TestUtil.hpp"
#include "fvm/PressureVelocityCoupling3D.hpp"

int main() {
  auto mesh = fvm::Mesh3D::hexGrid(6, 5, 4, 1.0, 1.0, 1.0, 0.05);
  fvm::VectorField3 u(mesh.cells.size(), fvm::Vec3::Zero());
  fvm::ScalarField p(mesh.cells.size(), 0.0), rAU(mesh.cells.size(), 0.01);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    u[c] = {0.03 * std::sin(M_PI * x.x()) * std::cos(M_PI * x.y()),
            -0.02 * std::cos(M_PI * x.x()) * std::sin(M_PI * x.y()),
            0.01 * std::sin(M_PI * x.z())};
    int i = static_cast<int>(c) % mesh.nx;
    int j = (static_cast<int>(c) / mesh.nx) % mesh.ny;
    int k = static_cast<int>(c) / (mesh.nx * mesh.ny);
    p[c] = ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
  }
  double beforeChecker = fvm::pressureCheckerboardMetric3D(mesh, p);
  auto report = fvm::projectVelocityRhieChow3D(mesh, u, p, rAU, 1.0);
  auto flux = fvm::rhieChowFlux3D(mesh, fvm::VectorField3(mesh.cells.size(), fvm::Vec3::Zero()),
                                  p, rAU);
  auto checkerDiv = fvm::explicitDivFaceFlux3D(mesh, flux);
  double maxCheckerDiv = 0.0;
  for (double d : checkerDiv) maxCheckerDiv = std::max(maxCheckerDiv, std::abs(d));
  check(report.maxDiv <= 1e-10, "3D Rhie-Chow projection face continuity at tolerance");
  check(report.checkerboard < beforeChecker, "3D pressure checkerboard mode is damped");
  check(maxCheckerDiv > 1e-5, "3D Rhie-Chow creates damping flux for checkerboard pressure");
  std::cout << "pressure3d_max_div=" << report.maxDiv
            << " checkerboard_before=" << beforeChecker
            << " checkerboard_after=" << report.checkerboard
            << " checkerboard_flux_div=" << maxCheckerDiv << "\n";
}
