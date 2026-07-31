#include "TestUtil.hpp"
#include "fvm/IncompressibleSolver3D.hpp"
#include <filesystem>
#include <fstream>

int main() {
  fvm::Cavity3DCase cfg;
  auto mesh = fvm::Mesh3D::hexGrid(cfg.n, cfg.n, cfg.n);
  auto sol = fvm::solveCavityProjection3D(cfg);
  fvm::VelocityBC3D bc{[](const fvm::Vec3& x) { return fvm::cavityVelocityBC3D(x); }};
  auto samples = fvm::sampleCavityCenterlineProfiles3D(mesh, sol.u, bc, 8);

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/cavity3d_profiles.csv");
  csv << "n,Re,steps,dt,axis,coord,component,value\n";

  double centerUx = 0.0;
  double topUx = 0.0;
  double bottomUx = 0.0;
  double midUy = 0.0;
  double maxAbsWOnCenterlines = 0.0;
  for (const auto& sample : samples) {
    csv << cfg.n << "," << cfg.Re << "," << cfg.steps << "," << cfg.dt << ","
        << sample.axis << "," << sample.coord << "," << sample.component << ","
        << sample.value << "\n";
    if (sample.axis == "z_center" && sample.component == "uz") {
      maxAbsWOnCenterlines = std::max(maxAbsWOnCenterlines, std::abs(sample.value));
    }
    if (sample.axis == "y_center" && sample.component == "ux" && sample.coord == 0.0) {
      bottomUx = sample.value;
    }
    if (sample.axis == "y_center" && sample.component == "ux" && sample.coord == 0.5) {
      centerUx = sample.value;
    }
    if (sample.axis == "y_center" && sample.component == "ux" && sample.coord == 1.0) {
      topUx = sample.value;
    }
    if (sample.axis == "x_center" && sample.component == "uy" && sample.coord == 0.5) {
      midUy = sample.value;
    }
  }

  check(samples.size() == 27, "3D cavity profile sampler returns three 9-point centerlines");
  check(std::abs(bottomUx) < 1e-12, "3D cavity lower wall profile velocity obeys no-slip BC");
  check(std::abs(topUx - 1.0) < 1e-12, "3D cavity upper wall profile velocity obeys lid BC");
  check(centerUx < 0.0, "3D cavity profile has mid-cell recirculation in x velocity");
  check(std::abs(midUy) < 0.25, "3D cavity centerline y velocity remains bounded");
  check(std::isfinite(maxAbsWOnCenterlines) && maxAbsWOnCenterlines < 0.25,
        "3D cavity centerline z velocity remains finite and bounded");
  check(sol.maxDiv <= 1e-10, "3D cavity profile run Rhie-Chow continuity at tolerance");
  std::cout << "cavity3d_profile_center_ux=" << centerUx
            << " cavity3d_profile_top_ux=" << topUx
            << " cavity3d_profile_mid_uy=" << midUy
            << " cavity3d_profile_max_abs_w=" << maxAbsWOnCenterlines
            << " cavity3d_profile_max_div=" << sol.maxDiv << "\n";
}
