#include "TestUtil.hpp"
#include "fvm/FieldOperators3D.hpp"

int main() {
  auto mesh = fvm::Mesh3D::hexGrid(2, 3, 4);
  double totalVolume = 0.0;
  double maxSplitError = 0.0;
  int internalFaces = 0;
  int boundaryFaces = 0;
  for (const auto& c : mesh.cells) totalVolume += c.V;
  for (const auto& f : mesh.faces) {
    maxSplitError = std::max(maxSplitError, (f.Sf - f.Delta - f.k).norm());
    if (f.internal()) ++internalFaces;
    else ++boundaryFaces;
  }
  check(std::abs(totalVolume - 1.0) < 1e-12, "3D hex mesh total volume is unit cube");
  check(internalFaces == 46, "3D hex mesh internal face count");
  check(boundaryFaces == 52, "3D hex mesh boundary face count");
  check(maxSplitError < 1e-14, "3D over-relaxed Sf decomposition is exact");

  auto stretched = fvm::Mesh3D::hexGrid(2, 3, 4, 6.0, 7.25, 5.5, 0.03);
  check(static_cast<int>(stretched.patches[0].faces.size()) == 12,
        "3D stretched mesh xmin patch count");
  check(static_cast<int>(stretched.patches[1].faces.size()) == 12,
        "3D stretched mesh xmax patch count");
  check(static_cast<int>(stretched.patches[2].faces.size()) == 8,
        "3D stretched mesh ymin patch count");
  check(static_cast<int>(stretched.patches[3].faces.size()) == 8,
        "3D stretched mesh ymax patch count");
  check(static_cast<int>(stretched.patches[4].faces.size()) == 6,
        "3D stretched mesh zmin patch count");
  check(static_cast<int>(stretched.patches[5].faces.size()) == 6,
        "3D stretched mesh zmax patch count");

  fvm::ScalarField phi(mesh.cells.size(), 0.0);
  fvm::VectorField3 u(mesh.cells.size(), fvm::Vec3::Zero());
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    phi[c] = x.x() + 2.0 * x.y() - 0.5 * x.z();
    u[c] = {x.x(), -x.y(), 0.0};
  }
  auto grad = fvm::gradLeastSquares3D(mesh, phi);
  auto div = fvm::divergence3D(mesh, u);
  double maxGradError = 0.0;
  double maxDivInterior = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    maxGradError = std::max(maxGradError, (grad[c] - fvm::Vec3(1.0, 2.0, -0.5)).norm());
    const auto& x = mesh.cells[c].centroid;
    if (x.x() > 0.26 && x.x() < 0.74 && x.y() > 0.2 && x.y() < 0.8) {
      maxDivInterior = std::max(maxDivInterior, std::abs(div[c]));
    }
  }
  check(maxGradError < 1e-12, "3D least-squares gradient recovers linear field");
  check(maxDivInterior < 1e-12, "3D divergence recovers zero interior divergence");
  std::cout << "mesh3d_total_volume=" << totalVolume
            << " internal_faces=" << internalFaces
            << " boundary_faces=" << boundaryFaces
            << " max_grad_error=" << maxGradError
            << " max_div_interior=" << maxDivInterior << "\n";
}
