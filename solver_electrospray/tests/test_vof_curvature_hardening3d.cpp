#include "TestUtil.hpp"
#include "fvm/SurfaceTension3D.hpp"
#include "fvm/VofTransport3D.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

fvm::ScalarField sphereAlpha(const fvm::Mesh3D& mesh, double radius, double width) {
  fvm::ScalarField alpha(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const double r = (mesh.cells[c].centroid - fvm::Vec3{0.5, 0.5, 0.5}).norm();
    alpha[c] = std::clamp(0.5 * (1.0 - std::tanh((r - radius) / width)), 0.0, 1.0);
  }
  return alpha;
}

struct CurvatureMetric {
  std::string method;
  std::string surface = "sphere_curvature";
  int n = 0;
  int cells = 0;
  double relativeL2 = 0.0;
  double relativeMean = 0.0;
  double meanKappa = 0.0;
  double mixedVolume = 0.0;
  double fallbackFraction = 0.0;
  int fittedCells = 0;
  int conditionedCells = 0;
  double meanStencilCondition = 0.0;
  double p95StencilCondition = 0.0;
  double maxStencilCondition = 0.0;
  int illConditionedFallbackCells = 0;
  int curvatureClampCells = 0;
};

CurvatureMetric curvatureMetricOnMesh(const fvm::Mesh3D& mesh, int n, double radius,
                                      const std::string& method,
                                      const std::string& surface = "sphere_curvature") {
  const double h = 1.0 / static_cast<double>(n);
  const auto alpha = sphereAlpha(mesh, radius, 1.25 * h);
  fvm::ScalarField kappa;
  double fallbackFraction = 0.0;
  int fittedCells = 0;
  int conditionedCells = 0;
  double meanStencilCondition = 0.0;
  double p95StencilCondition = 0.0;
  double maxStencilCondition = 0.0;
  int illConditionedFallbackCells = 0;
  int curvatureClampCells = 0;
  if (method == "equivalent_sphere") {
    kappa = fvm::curvatureFromEquivalentSphere3D(mesh, alpha);
  } else if (method == "local_plic_quadric") {
    auto report = fvm::curvatureFromLocalPlicQuadricReport3D(mesh, alpha, 36);
    kappa = report.kappa;
    fallbackFraction = report.fallbackFraction;
    fittedCells = report.fittedCells;
    conditionedCells = report.conditionedCells;
    meanStencilCondition = report.meanStencilCondition;
    p95StencilCondition = report.p95StencilCondition;
    maxStencilCondition = report.maxStencilCondition;
    illConditionedFallbackCells = report.illConditionedFallbackCells;
    curvatureClampCells = report.curvatureClampCells;
  } else {
    kappa = fvm::curvatureFromIsoRDF3D(mesh, alpha, 2);
  }
  const double exact = 2.0 / radius;
  double wsum = 0.0;
  double esum = 0.0;
  double ksum = 0.0;
  for (size_t c = 0; c < alpha.size(); ++c) {
    const double w = alpha[c] * (1.0 - alpha[c]) * mesh.cells[c].V;
    wsum += w;
    const double kmag = std::abs(kappa[c]);
    esum += w * fvm::sqr(kmag - exact);
    ksum += w * kmag;
    check(std::isfinite(kappa[c]), "curvature finite on refined sphere");
  }
  check(wsum > 0.0, "curvature metric has mixed interface volume");
  CurvatureMetric m;
  m.method = method;
  m.surface = surface;
  m.n = n;
  m.cells = static_cast<int>(mesh.cells.size());
  m.relativeL2 = std::sqrt(esum / wsum) / exact;
  m.meanKappa = ksum / wsum;
  m.relativeMean = std::abs(m.meanKappa - exact) / exact;
  m.mixedVolume = wsum;
  m.fallbackFraction = fallbackFraction;
  m.fittedCells = fittedCells;
  m.conditionedCells = conditionedCells;
  m.meanStencilCondition = meanStencilCondition;
  m.p95StencilCondition = p95StencilCondition;
  m.maxStencilCondition = maxStencilCondition;
  m.illConditionedFallbackCells = illConditionedFallbackCells;
  m.curvatureClampCells = curvatureClampCells;
  return m;
}

CurvatureMetric curvatureMetric(int n, double radius, const std::string& method) {
  return curvatureMetricOnMesh(fvm::Mesh3D::hexGrid(n, n, n, 1.0, 1.0, 1.0, 0.04),
                               n, radius, method);
}

double curvaturePerturbationResponse(const fvm::Mesh3D& mesh, const fvm::ScalarField& alpha) {
  auto perturbed = alpha;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    const double bump = 0.015 * std::sin(2.0 * M_PI * x.x()) *
                        std::sin(2.0 * M_PI * x.y()) *
                        std::sin(2.0 * M_PI * x.z());
    if (alpha[c] > 1e-6 && alpha[c] < 1.0 - 1e-6) {
      perturbed[c] = std::clamp(alpha[c] + bump, 0.0, 1.0);
    }
  }
  const auto k0 = fvm::curvatureFromIsoRDF3D(mesh, alpha, 2);
  const auto k1 = fvm::curvatureFromIsoRDF3D(mesh, perturbed, 2);
  double maxDiff = 0.0;
  for (size_t c = 0; c < k0.size(); ++c) {
    maxDiff = std::max(maxDiff, std::abs(k1[c] - k0[c]));
  }
  return maxDiff;
}

struct TimeSensitivityMetric {
  double coarseDrift = 0.0;
  double fineDrift = 0.0;
  double relativeL1Difference = 0.0;
  double minAlpha = 0.0;
  double maxAlpha = 0.0;
};

TimeSensitivityMetric vofTimeSensitivity() {
  const auto mesh = fvm::Mesh3D::hexGrid(12, 12, 10, 1.0, 1.0, 1.0, 0.06);
  const auto initial = sphereAlpha(mesh, 0.23, 0.035);
  const auto faceFlux = fvm::divergenceFreeBoxFlux3D(mesh, 0.03);
  fvm::VofTransportOptions3D opt;
  opt.scheme = fvm::VofAdvectionScheme3D::IsoAdvector;
  opt.correctionSweeps = 5;
  auto coarse = initial;
  auto fine = initial;
  fvm::VofTransportReport3D coarseReport;
  fvm::VofTransportReport3D fineReport;
  for (int i = 0; i < 8; ++i) coarseReport = fvm::advectVof3D(mesh, coarse, faceFlux, 0.004, opt);
  for (int i = 0; i < 16; ++i) fineReport = fvm::advectVof3D(mesh, fine, faceFlux, 0.002, opt);

  double l1 = 0.0;
  double ref = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    l1 += std::abs(coarse[c] - fine[c]) * mesh.cells[c].V;
    ref += std::abs(initial[c]) * mesh.cells[c].V;
  }
  const auto [aminC, amaxC] = fvm::vofBounds3D(coarse);
  const auto [aminF, amaxF] = fvm::vofBounds3D(fine);
  return {coarseReport.relativeMassDrift,
          fineReport.relativeMassDrift,
          l1 / std::max(ref, 1e-30),
          std::min(aminC, aminF),
          std::max(amaxC, amaxF)};
}

std::string trendStatus(const CurvatureMetric& a, const CurvatureMetric& b, const CurvatureMetric& c) {
  if (c.relativeL2 < b.relativeL2 && b.relativeL2 < a.relativeL2) return "UPHELD_DECREASING";
  if (c.relativeL2 < a.relativeL2) return "APPROXIMATE_NET_DECREASING";
  return "DOWNGRADED_NONMONOTONE";
}

}  // namespace

int main() {
  const double radius = 0.24;
  const auto rdf8 = curvatureMetric(8, radius, "iso_rdf");
  const auto rdf12 = curvatureMetric(12, radius, "iso_rdf");
  const auto rdf16 = curvatureMetric(16, radius, "iso_rdf");
  const auto m8 = curvatureMetric(8, radius, "equivalent_sphere");
  const auto m12 = curvatureMetric(12, radius, "equivalent_sphere");
  const auto m16 = curvatureMetric(16, radius, "equivalent_sphere");

  const auto probeMesh = fvm::Mesh3D::hexGrid(12, 12, 12, 1.0, 1.0, 1.0, 0.05);
  const auto probeAlpha = sphereAlpha(probeMesh, radius, 1.25 / 12.0);
  const double perturbResponse = curvaturePerturbationResponse(probeMesh, probeAlpha);
  const auto dtMetric = vofTimeSensitivity();
  const std::string status = trendStatus(m8, m12, m16);
  const std::string rdfStatus = trendStatus(rdf8, rdf12, rdf16);

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/vof_curvature_hardening3d.csv");
  csv << std::setprecision(16);
  const auto q8 = curvatureMetric(8, radius, "local_plic_quadric");
  const auto q12 = curvatureMetric(12, radius, "local_plic_quadric");
  const auto q16 = curvatureMetric(16, radius, "local_plic_quadric");
  const std::string quadricStatus = trendStatus(q8, q12, q16);

  const auto skewMesh = fvm::Mesh3D::hexGrid(14, 14, 14, 1.0, 1.0, 1.0, 0.12);
  const auto skewRdf = curvatureMetricOnMesh(skewMesh, 14, radius, "iso_rdf", "skewed_sphere_curvature");
  const auto skewQuadric = curvatureMetricOnMesh(skewMesh, 14, radius, "local_plic_quadric",
                                                 "skewed_sphere_curvature");

  csv << "surface,method,n,cells,relative_l2,relative_mean,mean_kappa,mixed_volume,"
         "fallback_fraction,fitted_cells,conditioned_cells,mean_stencil_condition,"
         "p95_stencil_condition,max_stencil_condition,ill_conditioned_fallback_cells,"
         "curvature_clamp_cells,status\n";
  for (const auto& m : {rdf8, rdf12, rdf16}) {
    csv << m.surface << "," << m.method << "," << m.n << "," << m.cells << "," << m.relativeL2
        << "," << m.relativeMean << "," << m.meanKappa << "," << m.mixedVolume
        << "," << m.fallbackFraction
        << "," << m.fittedCells << "," << m.conditionedCells << "," << m.meanStencilCondition
        << "," << m.p95StencilCondition << "," << m.maxStencilCondition
        << "," << m.illConditionedFallbackCells << "," << m.curvatureClampCells
        << "," << rdfStatus << "\n";
  }
  for (const auto& m : {m8, m12, m16, q8, q12, q16}) {
    const std::string rowStatus = m.method == "local_plic_quadric" ? quadricStatus : status;
    csv << m.surface << "," << m.method << "," << m.n << "," << m.cells << "," << m.relativeL2
        << "," << m.relativeMean << "," << m.meanKappa << "," << m.mixedVolume
        << "," << m.fallbackFraction
        << "," << m.fittedCells << "," << m.conditionedCells << "," << m.meanStencilCondition
        << "," << m.p95StencilCondition << "," << m.maxStencilCondition
        << "," << m.illConditionedFallbackCells << "," << m.curvatureClampCells
        << "," << rowStatus << "\n";
  }
  for (const auto& m : {skewRdf, skewQuadric}) {
    csv << m.surface << "," << m.method << "," << m.n << "," << m.cells << "," << m.relativeL2
        << "," << m.relativeMean << "," << m.meanKappa << "," << m.mixedVolume
        << "," << m.fallbackFraction
        << "," << m.fittedCells << "," << m.conditionedCells << "," << m.meanStencilCondition
        << "," << m.p95StencilCondition << "," << m.maxStencilCondition
        << "," << m.illConditionedFallbackCells << "," << m.curvatureClampCells
        << "," << (skewQuadric.relativeL2 < skewRdf.relativeL2 ? "LOCAL_BEATS_RDF" : "LOCAL_NOT_BETTER")
        << "\n";
  }
  csv << "curvature_perturbation,iso_rdf,12," << probeMesh.cells.size() << ","
      << perturbResponse << ",0,0,0,0,0,0,0,0,0,0,0,RESPONDS_TO_ALPHA\n";
  csv << "vof_time_sensitivity,isoadvector,12," << 12 * 12 * 10 << ","
      << dtMetric.relativeL1Difference << "," << dtMetric.coarseDrift << ","
      << dtMetric.fineDrift << "," << dtMetric.maxAlpha << ",0,0,0,0,0,0,0,0,BOUNDED_ISOADVECTOR\n";
  csv.flush();

  check(std::isfinite(m8.relativeL2) && std::isfinite(m12.relativeL2) && std::isfinite(m16.relativeL2),
        "curvature refinement metrics finite");
  check(status == "UPHELD_DECREASING", "equivalent-sphere curvature error decreases under refinement");
  check(m16.relativeL2 < 0.10, "equivalent-sphere curvature final error below diagnostic ceiling");
  check(quadricStatus == "UPHELD_DECREASING", "local PLIC quadric curvature error decreases under refinement");
  check(q16.relativeL2 < rdf16.relativeL2, "local PLIC quadric improves fine-grid local curvature over RDF");
  check(q16.fallbackFraction < 0.25, "local PLIC quadric fallback fraction bounded on fine grid");
  check(skewQuadric.relativeL2 < skewRdf.relativeL2, "local PLIC quadric improves skewed-grid curvature over RDF");
  check(perturbResponse > 1e-6, "curvature responds to alpha perturbation");
  check(dtMetric.coarseDrift <= 1e-3 && dtMetric.fineDrift <= 1e-3, "isoAdvector mass drift bounded in dt sensitivity");
  check(dtMetric.minAlpha >= -1e-14 && dtMetric.maxAlpha <= 1.0 + 1e-14, "isoAdvector alpha remains bounded");
  check(dtMetric.relativeL1Difference < 0.35, "isoAdvector dt sensitivity bounded by diagnostic envelope");

  std::cout << "vof_curvature_hardening_status=" << status
            << " local_plic_quadric_status=" << quadricStatus
            << " relL2_8=" << m8.relativeL2
            << " relL2_12=" << m12.relativeL2
            << " relL2_16=" << m16.relativeL2
            << " local_relL2_8=" << q8.relativeL2
            << " local_relL2_12=" << q12.relativeL2
            << " local_relL2_16=" << q16.relativeL2
            << " local_fallback_16=" << q16.fallbackFraction
            << " skew_rdf=" << skewRdf.relativeL2
            << " skew_local=" << skewQuadric.relativeL2
            << " perturb_response=" << perturbResponse
            << " dt_l1=" << dtMetric.relativeL1Difference
            << " mass_drift_coarse=" << dtMetric.coarseDrift
            << " mass_drift_fine=" << dtMetric.fineDrift << "\n";
}
