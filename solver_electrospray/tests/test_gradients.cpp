#include "TestUtil.hpp"

int main() {
  auto mesh = fvm::Mesh::quadGrid(32, 24, 1.0, 1.0, 0.2);
  fvm::ScalarField phi(mesh.cells.size());
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto x = mesh.cells[c].centroid.x();
    const auto y = mesh.cells[c].centroid.y();
    phi[c] = 2.0 + 3.0 * x - 4.0 * y;
  }
  auto gg = fvm::gradGreenGauss(mesh, phi);
  auto ls = fvm::gradLeastSquares(mesh, phi);
  double errLS = 0.0;
  double errGG = 0.0;
  int count = 0;
  for (int j = 1; j < mesh.ny - 1; ++j) {
    for (int i = 1; i < mesh.nx - 1; ++i) {
      int c = j * mesh.nx + i;
      errLS = std::max(errLS, (ls[c] - fvm::Vec(3.0, -4.0)).norm());
      errGG = std::max(errGG, (gg[c] - fvm::Vec(3.0, -4.0)).norm());
      ++count;
    }
  }
  check(count > 0, "interior cells exist");
  check(errLS < 1e-10, "least-squares exact for linear field");
  check(errGG < 0.45, "Green-Gauss bounded on skew mesh");
  std::cout << "gradients ls_max_error=" << errLS << " gg_interior_max_error=" << errGG << "\n";
}
