#include "TestUtil.hpp"
#include "fvm/VofTransport3D.hpp"
#include <filesystem>
#include <fstream>

static double aggregatePlaneCutVolume(const fvm::Mesh3D& mesh, const fvm::Vec3& normal,
                                      double globalCut) {
  double wetVolume = 0.0;
  double totalVolume = 0.0;
  fvm::Vec3 n = normal.normalized();
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) {
    const double localCut = globalCut - n.dot(mesh.cells[c].centroid);
    const double frac = fvm::exactPlicPlaneCutVolumeFraction3D(mesh, c, n, localCut);
    wetVolume += frac * mesh.cells[c].V;
    totalVolume += mesh.cells[c].V;
    check(std::isfinite(frac), "exact PLIC plane-cut fraction finite");
    check(frac >= -1e-14 && frac <= 1.0 + 1e-14,
          "exact PLIC plane-cut fraction bounded");
  }
  return wetVolume / std::max(totalVolume, 1e-30);
}

static fvm::Mesh3D irregularBipyramidMesh() {
  std::vector<fvm::Vec3> pts = {
      {0.0, 0.0, 0.5}, {1.0, 0.0, 0.48}, {1.02, 1.0, 0.52}, {-0.03, 1.0, 0.51},
      {0.48, 0.52, -0.05}, {0.53, 0.47, 1.04}};
  std::vector<std::vector<std::vector<int>>> cellFaces = {
      {
          {0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
      },
      {
          {0, 1, 2, 3}, {0, 5, 1}, {1, 5, 2}, {2, 5, 3}, {3, 5, 0},
      },
  };
  return fvm::Mesh3D::fromCellFaces(pts, cellFaces);
}

int main() {
  const auto singleHex = fvm::Mesh3D::hexGrid(1, 1, 1);
  const int c = 0;
  const double xHalf =
      fvm::exactPlicPlaneCutVolumeFraction3D(singleHex, c, fvm::Vec3::UnitX(), 0.0);
  const double xQuarter =
      fvm::exactPlicPlaneCutVolumeFraction3D(singleHex, c, fvm::Vec3::UnitX(), 0.25);
  const double xThreeQuarter =
      fvm::exactPlicPlaneCutVolumeFraction3D(singleHex, c, fvm::Vec3::UnitX(), -0.25);
  const double diagonalHalf =
      fvm::exactPlicPlaneCutVolumeFraction3D(singleHex, c, fvm::Vec3(1.0, 1.0, 1.0), 0.0);
  const double empty =
      fvm::exactPlicPlaneCutVolumeFraction3D(singleHex, c, fvm::Vec3::UnitX(), 10.0);
  const double full =
      fvm::exactPlicPlaneCutVolumeFraction3D(singleHex, c, fvm::Vec3::UnitX(), -10.0);

  const auto hexGrid = fvm::Mesh3D::hexGrid(4, 3, 2);
  const auto tetGrid = fvm::Mesh3D::tetraGrid(4, 3, 2);
  const double hexAggregate = aggregatePlaneCutVolume(hexGrid, fvm::Vec3::UnitX(), 0.5);
  const double tetAggregate = aggregatePlaneCutVolume(tetGrid, fvm::Vec3::UnitX(), 0.5);
  const double diagonalAggregate =
      aggregatePlaneCutVolume(hexGrid, fvm::Vec3(1.0, 1.0, 1.0), 1.5 / std::sqrt(3.0));

  const auto twoHex = fvm::Mesh3D::hexGrid(2, 1, 1);
  int internalFace = -1;
  for (int fi = 0; fi < static_cast<int>(twoHex.faces.size()); ++fi) {
    if (twoHex.faces[fi].internal()) internalFace = fi;
  }
  check(internalFace >= 0, "two-cell hex mesh has an internal face");
  const fvm::Face3D& sweptFace = twoHex.faces[internalFace];
  const int leftCell = sweptFace.owner;
  fvm::IsoSurfaceReconstruction3D sweptIso;
  sweptIso.alpha = 0.5;
  sweptIso.mixed = true;
  sweptIso.normal = fvm::Vec3::UnitX();
  sweptIso.cellCentroid = twoHex.cells[leftCell].centroid;
  sweptIso.interfaceCentroid = {0.45, 0.5, 0.5};
  sweptIso.cut = 0.45 - sweptIso.cellCentroid.x();
  const double sweptHalf =
      fvm::exactSweptFacePlicWetFraction3D(twoHex, internalFace, leftCell, 0.1, 1.0, sweptIso);
  sweptIso.cut = 0.39 - sweptIso.cellCentroid.x();
  const double sweptFull =
      fvm::exactSweptFacePlicWetFraction3D(twoHex, internalFace, leftCell, 0.1, 1.0, sweptIso);
  sweptIso.cut = 0.51 - sweptIso.cellCentroid.x();
  const double sweptEmpty =
      fvm::exactSweptFacePlicWetFraction3D(twoHex, internalFace, leftCell, 0.1, 1.0, sweptIso);

  const auto irregular = irregularBipyramidMesh();
  int irregularInternalFace = -1;
  int unsupportedCells = 0;
  for (int ci = 0; ci < static_cast<int>(irregular.cells.size()); ++ci) {
    if (!fvm::cellSupportsExactPlicPlaneCut3D(irregular.cells[ci])) ++unsupportedCells;
  }
  for (int fi = 0; fi < static_cast<int>(irregular.faces.size()); ++fi) {
    if (irregular.faces[fi].internal()) irregularInternalFace = fi;
  }
  check(irregularInternalFace >= 0, "irregular polyhedral diagnostic has an internal face");
  const int irregularUpwind = irregular.faces[irregularInternalFace].owner;
  fvm::IsoSurfaceReconstruction3D irregularIso;
  irregularIso.alpha = 0.5;
  irregularIso.mixed = true;
  irregularIso.normal = fvm::Vec3::UnitZ();
  irregularIso.cellCentroid = irregular.cells[irregularUpwind].centroid;
  irregularIso.interfaceCentroid = irregularIso.cellCentroid;
  irregularIso.cut = 0.0;
  const double irregularSweptWet = fvm::exactSweptFacePlicWetFraction3D(
      irregular, irregularInternalFace, irregularUpwind, 0.05, 0.25, irregularIso);

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/plic_plane_cut3d.csv");
  csv << "case,value,expected,abs_error\n";
  auto write = [&](const char* name, double value, double expected) {
    csv << name << "," << value << "," << expected << ","
        << std::abs(value - expected) << "\n";
  };
  write("single_hex_x_half", xHalf, 0.5);
  write("single_hex_x_quarter_wet", xQuarter, 0.25);
  write("single_hex_x_three_quarter_wet", xThreeQuarter, 0.75);
  write("single_hex_diagonal_half", diagonalHalf, 0.5);
  write("single_hex_empty", empty, 0.0);
  write("single_hex_full", full, 1.0);
  write("hex_grid_x_half", hexAggregate, 0.5);
  write("tet_grid_x_half", tetAggregate, 0.5);
  write("hex_grid_diagonal_half", diagonalAggregate, 0.5);
  write("swept_face_half", sweptHalf, 0.5);
  write("swept_face_full", sweptFull, 1.0);
  write("swept_face_empty", sweptEmpty, 0.0);

  std::ofstream irregularCsv("benchmark_logs/plic_irregular_swept_coverage3d.csv");
  irregularCsv << "case,cells,faces,unsupported_cells,internal_face,wet_fraction,status\n";
  irregularCsv << "irregular_bipyramid_swept_face," << irregular.cells.size() << ","
               << irregular.faces.size() << "," << unsupportedCells << ","
               << irregularInternalFace << "," << irregularSweptWet
               << ",IRREGULAR_SWEPT_DIAGNOSTIC_NOT_EXACT_CELL_CUT\n";

  check(std::abs(xHalf - 0.5) < 1e-13, "single hex x-half exact PLIC cut");
  check(std::abs(xQuarter - 0.25) < 1e-13, "single hex x-quarter exact PLIC cut");
  check(std::abs(xThreeQuarter - 0.75) < 1e-13,
        "single hex x-three-quarter exact PLIC cut");
  check(std::abs(diagonalHalf - 0.5) < 1e-13, "single hex diagonal exact PLIC cut");
  check(std::abs(empty) < 1e-13, "single hex empty exact PLIC cut");
  check(std::abs(full - 1.0) < 1e-13, "single hex full exact PLIC cut");
  check(std::abs(hexAggregate - 0.5) < 1e-13, "hex grid aggregate exact PLIC cut");
  check(std::abs(tetAggregate - 0.5) < 1e-13, "tet grid aggregate exact PLIC cut");
  check(std::abs(diagonalAggregate - 0.5) < 1e-13,
        "hex grid diagonal aggregate exact PLIC cut");
  check(std::abs(sweptHalf - 0.5) < 1e-13, "exact swept PLIC half-slab wet fraction");
  check(std::abs(sweptFull - 1.0) < 1e-13, "exact swept PLIC full-slab wet fraction");
  check(std::abs(sweptEmpty) < 1e-13, "exact swept PLIC empty-slab wet fraction");
  check(unsupportedCells > 0, "irregular polyhedral swept diagnostic covers unsupported cell shape");
  check(std::isfinite(irregularSweptWet) && irregularSweptWet >= -1e-14 &&
            irregularSweptWet <= 1.0 + 1e-14,
        "irregular polyhedral swept diagnostic remains finite and bounded");

  std::cout << "plic_plane_cut3d_single_half=" << xHalf
            << " single_quarter=" << xQuarter
            << " single_three_quarter=" << xThreeQuarter
            << " diagonal_half=" << diagonalHalf
            << " hex_aggregate=" << hexAggregate
            << " tet_aggregate=" << tetAggregate
            << " swept_half=" << sweptHalf
            << " irregular_swept_wet=" << irregularSweptWet << "\n";
}
