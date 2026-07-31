#include "TestUtil.hpp"
#include "fvm/FieldOperators3D.hpp"

int main() {
  auto mesh = fvm::Mesh3D::hexGrid(20, 1, 1);
  fvm::ScalarField phi(mesh.cells.size(), 0.0);
  fvm::ScalarField faceFlux(mesh.faces.size(), 0.0);
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) {
    phi[c] = mesh.cells[c].centroid.x() < 0.5 ? 1.0 : 0.0;
  }
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const auto& f = mesh.faces[fi];
    faceFlux[fi] = f.Sf.x();
  }

  auto flux = fvm::convectionFaceFluxUpwindTVD3D(mesh, phi, faceFlux, 1.0);
  double minFace = 1e300;
  double maxFace = -1e300;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const auto& f = mesh.faces[fi];
    if (!f.internal() || std::abs(faceFlux[fi]) < 1e-14) continue;
    double faceValue = flux[fi] / faceFlux[fi];
    minFace = std::min(minFace, faceValue);
    maxFace = std::max(maxFace, faceValue);
  }
  check(minFace >= -1e-12, "3D TVD face interpolation does not undershoot step");
  check(maxFace <= 1.0 + 1e-12, "3D TVD face interpolation does not overshoot step");

  fvm::ScalarField linear(mesh.cells.size(), 0.0);
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) {
    linear[c] = 2.0 * mesh.cells[c].centroid.x() + 1.0;
  }
  auto div = fvm::divConvectionUpwindTVD3D(mesh, linear, faceFlux, 1.0);
  double maxInteriorError = 0.0;
  for (int c = 1; c < static_cast<int>(mesh.cells.size()) - 1; ++c) {
    maxInteriorError = std::max(maxInteriorError, std::abs(div[c] - 2.0));
  }
  check(maxInteriorError < 0.25, "3D TVD deferred correction recovers linear convection gradient");
  std::cout << "convection_tvd3d min_face=" << minFace
            << " max_face=" << maxFace
            << " linear_div_max_error=" << maxInteriorError << "\n";
}
