#include "TestUtil.hpp"
#include "fvm/FieldOperators3D.hpp"

int main() {
  auto mesh = fvm::Mesh3D::tetraGrid(2, 2, 2, 1.0, 1.0, 1.0, 0.04);
  double totalVolume = 0.0;
  int internalFaces = 0;
  int boundaryFaces = 0;
  double maxSplitResidual = 0.0;
  for (const auto& c : mesh.cells) totalVolume += c.V;
  for (const auto& f : mesh.faces) {
    if (f.internal()) ++internalFaces;
    else ++boundaryFaces;
    maxSplitResidual = std::max(maxSplitResidual, (f.Sf - f.Delta - f.k).norm());
    check(f.area > 0.0, "tetra face area positive");
    check(f.magD > 0.0, "tetra face delta positive");
  }
  check(mesh.cells.size() == 48, "2x2x2 tetra grid has 48 tetra cells");
  check(internalFaces > 0, "tetra grid has internal faces");
  check(boundaryFaces > 0, "tetra grid has boundary faces");
  check(std::abs(totalVolume - 1.0) < 5e-13, "tetra grid preserves unit volume");
  check(maxSplitResidual < 1e-14, "tetra Sf decomposition residual small");

  fvm::ScalarField phi(mesh.cells.size(), 0.0);
  fvm::VectorField3 vec(mesh.cells.size(), fvm::Vec3::Zero());
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    phi[c] = 1.2 * x.x() - 0.7 * x.y() + 0.4 * x.z() + 0.3;
    vec[c] = {0.8 * x.x(), -0.2 * x.y(), 0.5 * x.z()};
  }
  auto grad = fvm::gradLeastSquares3D(mesh, phi);
  double maxInteriorGradError = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    if (mesh.cells[c].faces.size() < 4) continue;
    bool boundary = false;
    for (int fi : mesh.cells[c].faces) boundary = boundary || !mesh.faces[fi].internal();
    if (!boundary) {
      maxInteriorGradError = std::max(maxInteriorGradError,
                                      (grad[c] - fvm::Vec3{1.2, -0.7, 0.4}).norm());
    }
  }
  auto div = fvm::divergence3D(mesh, vec);
  double maxDivError = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    bool boundary = false;
    for (int fi : mesh.cells[c].faces) boundary = boundary || !mesh.faces[fi].internal();
    if (!boundary) maxDivError = std::max(maxDivError, std::abs(div[c] - 1.1));
  }
  check(maxInteriorGradError < 1e-11, "tetra LS gradient exact for linear field on interior cells");
  check(maxDivError < 0.5, "tetra divergence remains bounded for linear vector field");
  std::cout << "tetra3d_cells=" << mesh.cells.size()
            << " tetra3d_internal_faces=" << internalFaces
            << " tetra3d_boundary_faces=" << boundaryFaces
            << " tetra3d_total_volume=" << totalVolume
            << " tetra3d_max_grad_error=" << maxInteriorGradError
            << " tetra3d_max_div_error=" << maxDivError << "\n";
}
