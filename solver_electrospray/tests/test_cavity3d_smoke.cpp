#include "TestUtil.hpp"
#include "fvm/IncompressibleSolver3D.hpp"
#include <filesystem>
#include <fstream>

int main() {
  fvm::Cavity3DCase cfg;
  auto mesh = fvm::Mesh3D::hexGrid(cfg.n, cfg.n, cfg.n);
  auto sol = fvm::solveCavityProjection3D(cfg);
  double topLayerUx = 0.0;
  double coreUx = 0.0;
  double maxSpeed = 0.0;
  int topCount = 0;
  int coreCount = 0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    maxSpeed = std::max(maxSpeed, sol.u[c].norm());
    if (x.y() > 0.80) {
      topLayerUx += sol.u[c].x();
      ++topCount;
    }
    if (x.x() > 0.35 && x.x() < 0.65 && x.y() > 0.35 && x.y() < 0.65 &&
        x.z() > 0.35 && x.z() < 0.65) {
      coreUx += sol.u[c].x();
      ++coreCount;
    }
  }
  topLayerUx /= std::max(topCount, 1);
  coreUx /= std::max(coreCount, 1);
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/cavity3d_smoke.csv");
  csv << "case,n,Re,steps,dt,top_layer_ux,core_ux,max_speed,max_div,max_courant\n";
  csv << "baseline," << cfg.n << "," << cfg.Re << "," << cfg.steps << "," << cfg.dt << ","
      << topLayerUx << "," << coreUx
      << "," << maxSpeed << "," << sol.maxDiv << "," << sol.maxCourant << "\n";
  check(std::isfinite(maxSpeed), "3D cavity smoke velocity finite");
  check(maxSpeed > 1e-3, "3D cavity smoke responds to moving lid");
  check(topLayerUx > coreUx, "3D cavity lid drives stronger top-layer x velocity than core");
  check(sol.maxDiv <= 1e-10, "3D cavity smoke Rhie-Chow face continuity at tolerance");

  fvm::Cavity3DLid referenceLid{0, 0.0, 1, 1.0};
  std::vector<fvm::Cavity3DStage> courantStages = {{1000, 20, 0.02, 1.0},
                                                   {1000, 20, 0.01, 1.0},
                                                   {1000, 30, 0.08, 1.0}};
  auto courantSafe = fvm::solveCavityProjection3DContinuation(8, courantStages, 1.0, true,
                                                              -1.0, 2, referenceLid,
                                                              16, 16, 10, true);
  csv << "courant_safe_16x16x10,16,1000," << courantSafe.steps << ",0.08,nan,nan,nan,"
      << courantSafe.maxDiv << "," << courantSafe.maxCourant << "\n";
  check(courantSafe.maxDiv <= 1e-10, "3D cavity Courant-safe smoke continuity at tolerance");
  check(courantSafe.maxCourant <= 1.0, "3D cavity Courant-safe smoke respects requested Courant limit");

  std::cout << "cavity3d_top_layer_ux=" << topLayerUx
            << " cavity3d_core_ux=" << coreUx
            << " cavity3d_max_speed=" << maxSpeed
            << " cavity3d_max_div=" << sol.maxDiv
            << " cavity3d_courant_safe_max_div=" << courantSafe.maxDiv
            << " cavity3d_courant_safe_max_courant=" << courantSafe.maxCourant << "\n";
}
