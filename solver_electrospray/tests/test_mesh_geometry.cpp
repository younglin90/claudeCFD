#include "TestUtil.hpp"

int main() {
  auto mesh = fvm::Mesh::quadGrid(8, 6, 1.0, 1.0, 0.12);
  double volume = 0.0;
  int internal = 0, boundary = 0;
  double maxDecomp = 0.0;
  for (const auto& c : mesh.cells) volume += c.V;
  for (const auto& f : mesh.faces) {
    internal += f.internal() ? 1 : 0;
    boundary += f.internal() ? 0 : 1;
    maxDecomp = std::max(maxDecomp, (f.Sf - f.Delta - f.k).norm());
  }
  check(std::abs(volume - 1.0) < 2e-3, "mesh volume conservation");
  check(internal == (8 - 1) * 6 + (6 - 1) * 8, "internal face count");
  check(boundary == 2 * (8 + 6), "boundary face count");
  check(maxDecomp < 1e-14, "over-relaxed Sf decomposition");
  auto stretched = fvm::Mesh::stretchedQuadGrid(12, 10, 1.4);
  double stretchedVolume = 0.0;
  double minCell = 1e300;
  double maxCell = 0.0;
  for (const auto& c : stretched.cells) {
    stretchedVolume += c.V;
    minCell = std::min(minCell, c.V);
    maxCell = std::max(maxCell, c.V);
  }
  check(std::abs(stretchedVolume - 1.0) < 1e-12, "stretched mesh volume conservation");
  check(maxCell / minCell > 2.0, "stretched mesh clusters cells near walls");
  std::cout << "mesh_geometry volume=" << volume << " internal=" << internal
            << " boundary=" << boundary << " max_decomp=" << maxDecomp
            << " stretched_volume=" << stretchedVolume
            << " stretched_ratio=" << maxCell / minCell << "\n";
}
