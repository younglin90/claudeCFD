#include "TestUtil.hpp"
#include "fvm/SurfaceTension3D.hpp"
#include <filesystem>
#include <fstream>

static fvm::ScalarField smoothDropletAlpha(const fvm::Mesh3D& mesh, double radius, double eps) {
  fvm::ScalarField alpha(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double r = (mesh.cells[c].centroid - fvm::Vec3{0.5, 0.5, 0.5}).norm();
    alpha[c] = 0.5 * (1.0 - std::tanh((r - radius) / eps));
  }
  return alpha;
}

static void runCase(std::ofstream& csv, const std::string& name,
                    const fvm::Mesh3D& mesh, double densityRatio,
                    double laplaceLimit) {
  constexpr double radius = 0.25;
  constexpr double sigma = 0.072;
  constexpr double mu = 1.0e-3;
  constexpr double eps = 0.05;
  constexpr int steps = 1000;
  constexpr double dt = 1.0e-9;
  auto alpha = smoothDropletAlpha(mesh, radius, eps);
  auto report = fvm::staticDropletCurvatureNoiseSpuriousCurrent3D(mesh, alpha, radius,
                                                                 sigma, mu, 1.0,
                                                                 densityRatio, dt, steps);
  csv << name << "," << mesh.cells.size() << "," << mesh.faces.size() << ","
      << densityRatio << "," << steps << "," << report.laplace.relativeError
      << "," << report.maxCa << "," << report.finalCa << ","
      << (report.caNonIncreasing ? 1 : 0) << "," << report.maxU << ","
      << report.maxDiv << "," << report.maxBalanceResidual << "\n";

  check(std::isfinite(report.laplace.relativeError), name + " Laplace error finite");
  check(report.laplace.relativeError <= laplaceLimit, name + " Laplace error bounded");
  check(report.maxCa <= 1e-5, name + " static droplet adversarial Ca threshold");
  check(report.maxDiv <= 1e-10, name + " projected velocity divergence bounded");
  check(std::isfinite(report.maxBalanceResidual) && report.maxBalanceResidual > 0.0,
        name + " curvature-noise residual finite and active");
  check(report.caNonIncreasing, name + " Ca non-increasing over 1000-step window");
}

int main() {
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/static_droplet_adversarial3d.csv");
  csv << "case,cells,faces,density_ratio,steps,laplace_relative_error,"
         "max_ca,final_ca,ca_non_increasing,max_u,max_div,max_balance_residual\n";

  runCase(csv, "irregular_warped_polyhedra",
          fvm::Mesh3D::hexGrid(9, 8, 7, 1.0, 1.0, 1.0, 0.55),
          1.0, 0.35);
  runCase(csv, "density_ratio_1000_to_1",
          fvm::Mesh3D::hexGrid(9, 9, 9, 1.0, 1.0, 1.0, 0.15),
          1000.0, 0.35);

  std::cout << "static_droplet_adversarial3d_cases=2"
            << " static_droplet_adversarial3d_max_ca<=1e-5"
            << " static_droplet_adversarial3d_mode=curvature_noise"
            << " static_droplet_adversarial3d_steps=1000\n";
}
