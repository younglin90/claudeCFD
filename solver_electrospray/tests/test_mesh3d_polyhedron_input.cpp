#include "TestUtil.hpp"
#include "fvm/FieldOperators3D.hpp"
#include <filesystem>
#include <fstream>

int main() {
  std::vector<fvm::Vec3> pts = {
      {0.0, 0.0, 0.0}, {0.48, 0.03, 0.02}, {0.47, 0.95, -0.01}, {-0.02, 1.0, 0.04},
      {0.03, -0.02, 1.0}, {0.50, 0.02, 1.05}, {0.46, 0.98, 0.97}, {0.00, 1.02, 1.02},
      {1.0, 0.0, 0.0}, {1.03, 1.0, 0.03}, {1.0, -0.03, 1.0}, {1.02, 0.96, 1.03}};

  std::vector<std::vector<std::vector<int>>> cellFaces = {
      {
          {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
          {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5},
      },
      {
          {1, 5, 6, 2}, {8, 9, 11, 10}, {1, 8, 10, 5},
          {2, 6, 11, 9}, {1, 2, 9, 8}, {5, 10, 11, 6},
      },
  };

  auto mesh = fvm::Mesh3D::fromCellFaces(pts, cellFaces);
  double totalVolume = 0.0;
  int internalFaces = 0;
  int boundaryFaces = 0;
  double maxSplitResidual = 0.0;
  for (const auto& c : mesh.cells) totalVolume += c.V;
  for (const auto& f : mesh.faces) {
    if (f.internal()) ++internalFaces;
    else ++boundaryFaces;
    maxSplitResidual = std::max(maxSplitResidual, (f.Sf - f.Delta - f.k).norm());
    check(f.area > 0.0, "polyhedron face area positive");
    check(f.magD > 0.0, "polyhedron face d positive");
  }
  check(mesh.cells.size() == 2, "polyhedron input has two cells");
  check(mesh.faces.size() == 11, "polyhedron input deduplicates one shared face");
  check(internalFaces == 1, "polyhedron input detects one internal face");
  check(boundaryFaces == 10, "polyhedron input keeps boundary faces");
  check(totalVolume > 0.85 && totalVolume < 1.15, "polyhedron input volume plausible");
  check(maxSplitResidual < 1e-14, "polyhedron Sf decomposition residual small");

  fvm::ScalarField phi(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    phi[c] = 0.8 * x.x() - 0.4 * x.y() + 0.2 * x.z();
  }
  auto grad = fvm::gradLeastSquares3D(mesh, phi);
  for (const auto& g : grad) {
    check(std::isfinite(g.x()) && std::isfinite(g.y()) && std::isfinite(g.z()),
          "polyhedron LS gradient finite");
  }
  auto lap = fvm::laplacianExplicit3D(mesh, phi);
  double maxAbsLap = 0.0;
  for (double v : lap) {
    check(std::isfinite(v), "polyhedron laplacian finite");
    maxAbsLap = std::max(maxAbsLap, std::abs(v));
  }
  check(maxAbsLap < 10.0, "polyhedron laplacian bounded on linear field");

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/mesh3d_polyhedron_input.csv");
  csv << "case,cells,faces,internal_faces,boundary_faces,total_volume,"
         "max_sf_split_residual,max_lap_linear,from_cell_faces\n";
  csv << "two_cell_polyhedron," << mesh.cells.size() << "," << mesh.faces.size()
      << "," << internalFaces << "," << boundaryFaces << "," << totalVolume
      << "," << maxSplitResidual << "," << maxAbsLap << ",1\n";

  std::cout << "poly3d_cells=" << mesh.cells.size()
            << " poly3d_faces=" << mesh.faces.size()
            << " poly3d_internal_faces=" << internalFaces
            << " poly3d_boundary_faces=" << boundaryFaces
            << " poly3d_total_volume=" << totalVolume
            << " poly3d_max_lap_linear=" << maxAbsLap << "\n";
}
