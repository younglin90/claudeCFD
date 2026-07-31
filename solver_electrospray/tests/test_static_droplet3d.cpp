#include "TestUtil.hpp"
#include "fvm/SurfaceTension3D.hpp"
#include <cmath>
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

int main() {
  constexpr double radius = 0.25;
  constexpr double sigma = 0.072;
  constexpr double eps = 0.04;
  constexpr double mu = 1.0e-3;
  constexpr double staticUmax = 0.0;
  constexpr double ca = mu * staticUmax / sigma;

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/static_droplet3d.csv");
  csv << "n,cells,radius,sigma,mean_curvature,computed_jump,analytic_jump,"
         "laplace_relative_error,ca_static_proxy\n";
  std::ofstream transferCsv("benchmark_logs/static_droplet_csf_transfer_diagnostic3d.csv");
  transferCsv << "n,cells,balanced_max_residual,balanced_l2_residual,"
                 "face_gauss_max_residual,face_gauss_l2_residual,"
                 "face_gauss_to_balanced_max_ratio,"
                 "hybrid_max_residual,hybrid_l2_residual,"
                 "hybrid_to_balanced_max_ratio,"
                 "face_gauss_paired_gg_max_residual,face_gauss_paired_gg_l2_residual,"
                 "paired_gg_status,status\n";

  double maxErr = 0.0;
  for (int n : {16, 20, 24}) {
    auto mesh = fvm::Mesh3D::hexGrid(n, n, n);
    auto alpha = smoothDropletAlpha(mesh, radius, eps);
    auto report = fvm::staticDropletLaplace3D(mesh, alpha, radius, sigma);
    csv << n << "," << mesh.cells.size() << "," << radius << "," << sigma << ","
        << report.meanCurvature << "," << report.computedJump << ","
        << report.analyticJump << "," << report.relativeError << "," << ca << "\n";
    maxErr = std::max(maxErr, report.relativeError);
    check(report.relativeError <= 0.02, "3D static droplet Laplace jump within 2%");

    const double kappa0 = 2.0 / radius;
    fvm::ScalarField kappa(mesh.cells.size(), kappa0);
    fvm::ScalarField pBalanced(mesh.cells.size(), 0.0);
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      pBalanced[c] = sigma * kappa0 * alpha[c];
    }
    fvm::VectorField3 gradP = fvm::pressureGradientFromSnGrad3D(mesh, pBalanced);
    fvm::VectorField3 gradPGauss = fvm::gradGreenGauss3D(mesh, pBalanced);
    fvm::VectorField3 balanced = fvm::balancedCsfForce3D(mesh, alpha, sigma, &kappa);
    fvm::VectorField3 faceGauss = fvm::gaussAlphaCsfForce3D(mesh, alpha, sigma, &kappa);
    fvm::VectorField3 hybrid =
        fvm::hybridMeanBalancedDeltaGaussCsfForce3D(mesh, alpha, sigma, &kappa);
    double balancedMax = 0.0;
    double faceGaussMax = 0.0;
    double hybridMax = 0.0;
    double pairedGaussMax = 0.0;
    double balancedL2 = 0.0;
    double faceGaussL2 = 0.0;
    double hybridL2 = 0.0;
    double pairedGaussL2 = 0.0;
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const double b = (balanced[c] - gradP[c]).norm();
      const double g = (faceGauss[c] - gradP[c]).norm();
      const double h = (hybrid[c] - gradP[c]).norm();
      const double pg = (faceGauss[c] - gradPGauss[c]).norm();
      balancedMax = std::max(balancedMax, b);
      faceGaussMax = std::max(faceGaussMax, g);
      hybridMax = std::max(hybridMax, h);
      pairedGaussMax = std::max(pairedGaussMax, pg);
      balancedL2 += b * b * mesh.cells[c].V;
      faceGaussL2 += g * g * mesh.cells[c].V;
      hybridL2 += h * h * mesh.cells[c].V;
      pairedGaussL2 += pg * pg * mesh.cells[c].V;
    }
    balancedL2 = std::sqrt(balancedL2);
    faceGaussL2 = std::sqrt(faceGaussL2);
    hybridL2 = std::sqrt(hybridL2);
    pairedGaussL2 = std::sqrt(pairedGaussL2);
    const double ratio = faceGaussMax / std::max(balancedMax, 1e-30);
    const double hybridRatio = hybridMax / std::max(balancedMax, 1e-30);
    const bool finite = std::isfinite(faceGaussMax) && std::isfinite(faceGaussL2) &&
                        std::isfinite(hybridMax) && std::isfinite(hybridL2) &&
                        std::isfinite(pairedGaussMax) && std::isfinite(pairedGaussL2);
    check(finite, "3D static droplet CSF transfer diagnostic finite");
    transferCsv << n << "," << mesh.cells.size() << "," << balancedMax << ","
                << balancedL2 << "," << faceGaussMax << "," << faceGaussL2 << ","
                << ratio << "," << hybridMax << "," << hybridL2 << ","
                << hybridRatio << "," << pairedGaussMax << "," << pairedGaussL2 << ","
                << (pairedGaussMax <= 10.0 * std::max(balancedMax, 1e-30)
                        ? "PAIRED_GAUSS_STATIC_BALANCE_COMPATIBLE"
                        : "PAIRED_GAUSS_STATIC_BALANCE_DOWNGRADED")
                << ","
                << (hybridMax <= 10.0 * std::max(balancedMax, 1e-30)
                        ? "HYBRID_STATIC_BALANCE_COMPATIBLE"
                        : "HYBRID_STATIC_BALANCE_DOWNGRADED")
                << "\n";
  }
  check(ca <= 1e-6, "3D static droplet static Ca proxy within target");
  std::cout << "static_droplet3d_max_laplace_error=" << maxErr
            << " static_droplet3d_ca_proxy=" << ca << "\n";
}
