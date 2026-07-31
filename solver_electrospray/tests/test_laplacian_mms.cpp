#include "TestUtil.hpp"
#include <fstream>

static double mmsError(int n) {
  auto mesh = fvm::Mesh::quadGrid(n, n, 1.0, 1.0, 0.22);
  fvm::ScalarField phi(mesh.cells.size()), exact(mesh.cells.size());
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double x = mesh.cells[c].centroid.x();
    double y = mesh.cells[c].centroid.y();
    phi[c] = std::sin(M_PI * x) * std::sin(M_PI * y);
    exact[c] = -2.0 * M_PI * M_PI * phi[c];
  }
  auto lap = fvm::laplacianExplicit(mesh, phi);
  double e = 0.0, v = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    if (mesh.cells[c].centroid.x() < 1.0 / n || mesh.cells[c].centroid.x() > 1.0 - 1.0 / n ||
        mesh.cells[c].centroid.y() < 1.0 / n || mesh.cells[c].centroid.y() > 1.0 - 1.0 / n) continue;
    e += fvm::sqr(lap[c] - exact[c]) * mesh.cells[c].V;
    v += mesh.cells[c].V;
  }
  return std::sqrt(e / std::max(v, 1e-30));
}

int main() {
  fvm::ensureLogDir();
  std::ofstream csv("benchmark_logs/mms_diffusion.csv");
  csv << "n,l2_error\n";
  double e16 = mmsError(16);
  double e32 = mmsError(32);
  double e64 = mmsError(64);
  csv << "16," << e16 << "\n32," << e32 << "\n64," << e64 << "\n";
  double slope = std::log(e16 / e64) / std::log(4.0);
  check(slope >= 1.85, "MMS diffusion convergence is near second order or better");
  std::cout << "mms_diffusion e16=" << e16 << " e32=" << e32 << " e64=" << e64
            << " slope=" << slope << "\n";
}
