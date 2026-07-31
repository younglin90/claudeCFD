#include "TestUtil.hpp"
#include "fvm/FieldOperators3D.hpp"

static double mmsError(int n) {
  auto mesh = fvm::Mesh3D::hexGrid(n, n, n, 1.0, 1.0, 1.0, 0.18);
  fvm::ScalarField phi(mesh.cells.size(), 0.0);
  fvm::ScalarField exact(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    phi[c] = std::sin(M_PI * x.x()) * std::sin(M_PI * x.y()) * std::sin(M_PI * x.z());
    exact[c] = -3.0 * M_PI * M_PI * phi[c];
  }
  auto lap = fvm::laplacianExplicit3D(mesh, phi);
  double e2 = 0.0;
  double vol = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    double margin = 1.5 / static_cast<double>(n);
    if (x.x() < margin || x.x() > 1.0 - margin ||
        x.y() < margin || x.y() > 1.0 - margin ||
        x.z() < margin || x.z() > 1.0 - margin) {
      continue;
    }
    e2 += fvm::sqr(lap[c] - exact[c]) * mesh.cells[c].V;
    vol += mesh.cells[c].V;
  }
  return std::sqrt(e2 / std::max(vol, 1e-30));
}

int main() {
  double e6 = mmsError(6);
  double e8 = mmsError(8);
  double e10 = mmsError(10);
  double slope = std::log(e6 / e10) / std::log(10.0 / 6.0);
  check(std::isfinite(slope), "3D MMS slope finite");
  check(slope >= 1.9, "3D skewed-mesh diffusion MMS slope >= 1.9");
  std::cout << "mms3d_e6=" << e6 << " mms3d_e8=" << e8
            << " mms3d_e10=" << e10 << " mms3d_slope=" << slope << "\n";
}
