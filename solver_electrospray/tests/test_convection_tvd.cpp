#include "TestUtil.hpp"

int main() {
  auto mesh = fvm::Mesh::quadGrid(20, 1, 1.0, 1.0, 0.0);
  fvm::ScalarField phi(mesh.cells.size(), 0.0);
  fvm::VectorField u(mesh.cells.size(), fvm::Vec(1.0, 0.0));
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) {
    phi[c] = mesh.cells[c].centroid.x() < 0.5 ? 1.0 : 0.0;
  }
  auto flux = fvm::convectionFaceFluxUpwindTVD(mesh, phi, u, 1.0);
  double minFace = 1e300;
  double maxFace = -1e300;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const auto& f = mesh.faces[fi];
    if (!f.internal() || std::abs(f.Sf.x()) < 1e-12) continue;
    double mdot = 0.5 * (u[f.owner] + u[f.neighbour]).dot(f.Sf);
    if (std::abs(mdot) < 1e-14) continue;
    double faceValue = flux[fi] / mdot;
    minFace = std::min(minFace, faceValue);
    maxFace = std::max(maxFace, faceValue);
  }
  check(minFace >= -1e-12, "TVD face interpolation does not undershoot step");
  check(maxFace <= 1.0 + 1e-12, "TVD face interpolation does not overshoot step");

  fvm::ScalarField linear(mesh.cells.size(), 0.0);
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) {
    linear[c] = 2.0 * mesh.cells[c].centroid.x() + 1.0;
  }
  auto div = fvm::divConvectionUpwindTVD(mesh, linear, u, 1.0);
  fvm::ScalarField upwindFaceFlux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const auto& f = mesh.faces[fi];
    if (!f.internal()) continue;
    upwindFaceFlux[fi] = (0.5 * (u[f.owner] + u[f.neighbour])).dot(f.Sf);
  }
  fvm::FvMatrix implicitDiv(static_cast<int>(mesh.cells.size()));
  implicitDiv.A.resize(mesh.cells.size(), mesh.cells.size());
  fvm::addImplicitDivergenceUpwind(implicitDiv, mesh, upwindFaceFlux);
  Eigen::VectorXd phiVec(mesh.cells.size());
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) phiVec[c] = phi[c];
  Eigen::VectorXd implicitFlux = implicitDiv.A * phiVec;
  fvm::ScalarField firstOrderFlux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const auto& f = mesh.faces[fi];
    if (!f.internal()) continue;
    int up = upwindFaceFlux[fi] >= 0.0 ? f.owner : f.neighbour;
    firstOrderFlux[fi] = upwindFaceFlux[fi] * phi[up];
  }
  auto explicitFirstOrder = fvm::explicitDivFaceFlux(mesh, firstOrderFlux);
  double maxImplicitMismatch = 0.0;
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) {
    maxImplicitMismatch = std::max(maxImplicitMismatch,
        std::abs(implicitFlux[c] / mesh.cells[c].V - explicitFirstOrder[c]));
  }
  double maxInteriorError = 0.0;
  for (int c = 1; c < static_cast<int>(mesh.cells.size()) - 1; ++c) {
    maxInteriorError = std::max(maxInteriorError, std::abs(div[c] - 2.0));
  }
  check(maxImplicitMismatch < 1e-12, "implicit upwind div assembly matches face-loop flux divergence");
  check(maxInteriorError < 0.25, "TVD deferred correction recovers linear convection gradient");
  std::cout << "convection_tvd min_face=" << minFace << " max_face=" << maxFace
            << " implicit_div_mismatch=" << maxImplicitMismatch
            << " linear_div_max_error=" << maxInteriorError << "\n";
}
