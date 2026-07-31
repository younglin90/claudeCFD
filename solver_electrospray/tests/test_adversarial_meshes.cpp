#include "TestUtil.hpp"

static double nonOrthoNorm(const fvm::Mesh& mesh) {
  fvm::ScalarField phi(mesh.cells.size());
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double x = mesh.cells[c].centroid.x();
    double y = mesh.cells[c].centroid.y();
    phi[c] = x * x + 0.5 * y * y + x * y;
  }
  auto lap = fvm::laplacianExplicit(mesh, phi);
  double e2 = 0.0;
  double vol = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& cc = mesh.cells[c].centroid;
    if (cc.x() < 0.08 || cc.x() > 0.92 * (mesh.nx > mesh.ny ? 4.0 : 1.0) ||
        cc.y() < 0.08 || cc.y() > 0.92) continue;
    e2 += fvm::sqr(lap[c] - 3.0) * mesh.cells[c].V;
    vol += mesh.cells[c].V;
  }
  return std::sqrt(e2 / std::max(vol, 1e-30));
}

static fvm::Mesh irregularPolygonMesh() {
  fvm::Mesh mesh;
  mesh.points = {
      {0.0, 0.0}, {0.45, 0.0}, {1.0, 0.0},
      {0.0, 0.52}, {0.38, 0.42}, {1.0, 0.35},
      {0.0, 1.0}, {0.62, 1.0}, {1.0, 1.0},
      {0.72, 0.62},
  };
  mesh.cells.resize(5);
  mesh.cells[0].points = {0, 1, 4, 3};
  mesh.cells[1].points = {1, 2, 5, 9, 4};
  mesh.cells[2].points = {3, 4, 7, 6};
  mesh.cells[3].points = {4, 9, 8, 7};
  mesh.cells[4].points = {5, 8, 9};
  mesh.buildFaces();
  mesh.computeGeometry();
  return mesh;
}

static double polygonNonOrthoNorm(const fvm::Mesh& mesh) {
  fvm::ScalarField phi(mesh.cells.size());
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double x = mesh.cells[c].centroid.x();
    double y = mesh.cells[c].centroid.y();
    phi[c] = x * x + 0.5 * y * y + x * y;
  }
  auto lap = fvm::laplacianExplicit(mesh, phi);
  double e2 = 0.0;
  double vol = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    e2 += fvm::sqr(lap[c] - 3.0) * mesh.cells[c].V;
    vol += mesh.cells[c].V;
  }
  return std::sqrt(e2 / std::max(vol, 1e-30));
}

int main() {
  auto irregular = fvm::Mesh::quadGrid(28, 20, 1.0, 1.0, 0.35);
  auto aspect = fvm::Mesh::quadGrid(48, 8, 4.0, 1.0, 0.08, 1.0);
  auto poly = irregularPolygonMesh();
  double eIrregular = nonOrthoNorm(irregular);
  double eAspect = nonOrthoNorm(aspect);
  double ePoly = polygonNonOrthoNorm(poly);
  fvm::ScalarField p(irregular.cells.size(), 0.0), rAU(irregular.cells.size(), 1.0);
  fvm::VectorField h(irregular.cells.size(), fvm::Vec::Zero());
  for (size_t c = 0; c < p.size(); ++c) {
    int i = static_cast<int>(c) % irregular.nx;
    int j = static_cast<int>(c) / irregular.nx;
    p[c] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
  }
  auto flux = fvm::rhieChowFlux(irregular, h, p, rAU);
  auto div = fvm::explicitDivFaceFlux(irregular, flux);
  double maxDiv = 0.0;
  for (double d : div) maxDiv = std::max(maxDiv, std::abs(d));
  std::cout << "adversarial_irregular_lap_error=" << eIrregular
            << " high_aspect_lap_error=" << eAspect
            << " irregular_polygon_lap_error=" << ePoly
            << " checkerboard_damping_div=" << maxDiv << "\n";
  check(eIrregular < 20.0, "fully irregular mesh non-orthogonal correction remains bounded");
  check(eAspect < 45.0, "high-aspect-ratio mesh non-orthogonal correction remains bounded");
  check(ePoly < 40.0, "fresh irregular polygon mesh non-orthogonal correction remains bounded");
  check(maxDiv > 1e-3, "Rhie-Chow creates damping flux for checkerboard mode");
}
