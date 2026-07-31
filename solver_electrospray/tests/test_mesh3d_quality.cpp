#include "TestUtil.hpp"
#include "fvm/MeshQuality3D.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
  const auto regular = fvm::Mesh3D::hexGrid(4, 4, 4, 1.0, 1.0, 1.0, 0.0);
  const auto regularQ = fvm::meshQualityReport3D(regular);
  check(regularQ.finite, "regular mesh quality finite");
  check(regularQ.nonPositiveVolumeCount == 0, "regular mesh positive volumes");
  check(regularQ.zeroAreaFaceCount == 0, "regular mesh positive face areas");
  check(regularQ.maxNonOrthogonalityDeg < 1e-9, "regular mesh orthogonal");
  check(regularQ.maxAspectRatio > 1.7 && regularQ.maxAspectRatio < 1.8, "regular hexahedron diagonal aspect ratio");

  const auto skewed = fvm::Mesh3D::hexGrid(5, 4, 3, 1.0, 1.0, 1.0, 0.35);
  const auto skewedQ = fvm::meshQualityReport3D(skewed);
  check(skewedQ.finite, "skewed mesh quality finite");
  check(skewedQ.maxNonOrthogonalityDeg > 0.01, "skewed mesh non-orthogonality detected");
  check(skewedQ.maxSkewness > 1e-5, "skewed mesh skewness detected");

  const auto stretched = fvm::Mesh3D::hexGridFromCoordinates({0.0, 10.0, 20.0},
                                                            {0.0, 0.1, 0.2},
                                                            {0.0, 0.1, 0.2});
  const auto stretchedQ = fvm::meshQualityReport3D(stretched);
  check(stretchedQ.finite, "stretched mesh quality finite");
  check(stretchedQ.maxAspectRatio > 100.0, "high aspect ratio detected");

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/mesh3d_quality.csv");
  csv << "case,cells,faces,max_non_ortho_deg,mean_non_ortho_deg,max_skewness,"
         "max_aspect_ratio,min_volume,max_volume\n";
  auto write = [&](const char* name, const fvm::MeshQualityReport3D& q) {
    csv << name << "," << q.cells << "," << q.faces << "," << q.maxNonOrthogonalityDeg
        << "," << q.meanNonOrthogonalityDeg << "," << q.maxSkewness << ","
        << q.maxAspectRatio << "," << q.minVolume << "," << q.maxVolume << "\n";
  };
  write("regular", regularQ);
  write("skewed", skewedQ);
  write("stretched", stretchedQ);

  std::cout << "mesh3d_quality_regular_non_ortho=" << regularQ.maxNonOrthogonalityDeg
            << " mesh3d_quality_skewed_non_ortho=" << skewedQ.maxNonOrthogonalityDeg
            << " mesh3d_quality_stretched_aspect=" << stretchedQ.maxAspectRatio << "\n";
}
