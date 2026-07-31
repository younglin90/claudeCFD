#include "TestUtil.hpp"
#include "fvm/SurfaceTension3D.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

int main() {
  auto mesh = fvm::Mesh3D::hexGrid(12, 11, 10, 1.0, 1.0, 1.0, 0.08);
  fvm::ScalarField alpha(mesh.cells.size(), 0.0), p(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    double r = (x - fvm::Vec3{0.5, 0.5, 0.5}).norm();
    alpha[c] = 0.5 * (1.0 - std::tanh((r - 0.24) / 0.035));
    p[c] = std::sin(M_PI * x.x()) * std::cos(M_PI * x.y()) * std::sin(M_PI * x.z());
  }

  auto snAlpha = fvm::faceSnGrad3D(mesh, alpha);
  auto snPressure = fvm::faceSnGrad3D(mesh, p);
  auto report = fvm::auditBalancedSnGradInvariant3D(mesh, alpha, p);
  auto state = fvm::buildBalancedForceSurfaceTensionState3D(mesh, alpha, 0.072);
  auto force = fvm::balancedCsfForce3D(mesh, alpha, 0.072);
  auto gp = fvm::pressureGradientFromSnGrad3D(mesh, p);
  const fvm::Vec3 wallNormal = fvm::Vec3::UnitY();
  const fvm::Vec3 rawInterfaceNormal{1.0, 0.2, 0.4};
  const double contactAngleDeg = 51.0;
  const fvm::Vec3 adjustedNormal =
      fvm::contactAngleAdjustedNormal3D(rawInterfaceNormal, wallNormal, contactAngleDeg);
  const double adjustedAngleDeg =
      std::acos(std::clamp(adjustedNormal.dot(wallNormal), -1.0, 1.0)) * 180.0 / M_PI;
  fvm::Vec3 rawTangent = rawInterfaceNormal - rawInterfaceNormal.dot(wallNormal) * wallNormal;
  rawTangent.normalize();
  fvm::Vec3 adjustedTangent = adjustedNormal - adjustedNormal.dot(wallNormal) * wallNormal;
  adjustedTangent.normalize();
  const double tangentAlignment = rawTangent.dot(adjustedTangent);

  double maxSnAlpha = 0.0, maxSnPressure = 0.0, maxForce = 0.0, maxGp = 0.0;
  for (double v : snAlpha) maxSnAlpha = std::max(maxSnAlpha, std::abs(v));
  for (double v : snPressure) maxSnPressure = std::max(maxSnPressure, std::abs(v));
  for (const auto& v : force) maxForce = std::max(maxForce, v.norm());
  for (const auto& v : gp) maxGp = std::max(maxGp, v.norm());

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/surface_tension3d.csv");
  csv << "case,cells,faces,max_snGrad_difference,max_kappa,max_force,max_pressure_grad,"
         "max_sn_alpha,max_sn_pressure,state_snGrad_difference,state_balance_residual,"
         "contact_angle_deg,adjusted_angle_deg,tangent_alignment\n";
  csv << "snGrad_invariant," << mesh.cells.size() << "," << mesh.faces.size() << ","
      << report.maxSnGradDifference << "," << report.maxCurvatureMagnitude << ","
      << maxForce << "," << maxGp << "," << maxSnAlpha << "," << maxSnPressure
      << "," << state.maxSnGradDifference << "," << state.maxBalanceResidual << ","
      << contactAngleDeg << "," << adjustedAngleDeg << "," << tangentAlignment << "\n";

  check(report.maxSnGradDifference == 0.0, "3D CSF and pressure paths reuse identical snGrad operator");
  check(state.maxSnGradDifference == 0.0, "3D solver-facing surface-tension state reuses identical snGrad operator");
  check(std::isfinite(state.maxBalanceResidual), "3D solver-facing surface-tension state balance residual finite");
  check(std::isfinite(report.maxCurvatureMagnitude), "3D CSF curvature finite");
  check(std::isfinite(maxForce) && maxForce > 0.0, "3D CSF force finite and active");
  check(std::isfinite(maxGp) && maxGp > 0.0, "3D pressure snGrad gradient finite and active");
  check(std::abs(adjustedAngleDeg - contactAngleDeg) < 1e-12,
        "3D contact-angle adjusted normal satisfies requested wall angle");
  check(tangentAlignment > 1.0 - 1e-12,
        "3D contact-angle adjusted normal preserves tangential interface direction");

  auto wallMesh = fvm::Mesh3D::hexGrid(14, 10, 14, 1.0, 0.7, 1.0, 0.06);
  fvm::ScalarField wallAlpha(wallMesh.cells.size(), 0.0);
  const fvm::Vec3 capCenter{0.5, 0.04, 0.5};
  const double capRadius = 0.28;
  const double capThickness = 0.035;
  for (size_t c = 0; c < wallMesh.cells.size(); ++c) {
    const auto& x = wallMesh.cells[c].centroid;
    const double r = (x - capCenter).norm();
    wallAlpha[c] = 0.5 * (1.0 - std::tanh((r - capRadius) / capThickness));
  }
  const double wallYMax = 0.14;
  const auto baselineCurvature =
      fvm::curvatureFromLocalPlicQuadricReport3D(wallMesh, wallAlpha, 28);
  const auto contactCurvature = fvm::curvatureFromLocalPlicQuadricReport3D(
      wallMesh, wallAlpha, 28, &wallNormal, contactAngleDeg, wallYMax);
  int wallMixedCells = 0;
  int changedWallCurvatureCells = 0;
  double maxWallCurvatureDelta = 0.0;
  double meanWallCurvatureDelta = 0.0;
  for (int ci = 0; ci < static_cast<int>(wallMesh.cells.size()); ++ci) {
    const double a = wallAlpha[ci];
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    if (wallMesh.cells[ci].centroid.y() > wallYMax) continue;
    const double delta = std::abs(contactCurvature.kappa[ci] - baselineCurvature.kappa[ci]);
    ++wallMixedCells;
    meanWallCurvatureDelta += delta;
    maxWallCurvatureDelta = std::max(maxWallCurvatureDelta, delta);
    if (delta > 1e-12) ++changedWallCurvatureCells;
  }
  if (wallMixedCells > 0) meanWallCurvatureDelta /= static_cast<double>(wallMixedCells);
  auto weightedMeanAbsKappa = [&](const fvm::ScalarField& kappa, bool wallOnly) {
    double weighted = 0.0;
    double weightSum = 0.0;
    for (int ci = 0; ci < static_cast<int>(wallMesh.cells.size()); ++ci) {
      const double a = std::clamp(wallAlpha[ci], 0.0, 1.0);
      if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
      if (wallOnly && wallMesh.cells[ci].centroid.y() > wallYMax) continue;
      const double w = std::max(a * (1.0 - a) * wallMesh.cells[ci].V, 0.0);
      weighted += w * std::abs(kappa[ci]);
      weightSum += w;
    }
    return weightSum > 0.0 ? weighted / weightSum : 0.0;
  };
  const double analyticCapCurvature = 2.0 / capRadius;
  const double baselineMeanKappa = weightedMeanAbsKappa(baselineCurvature.kappa, false);
  const double contactMeanKappa = weightedMeanAbsKappa(contactCurvature.kappa, false);
  const double baselineWallMeanKappa = weightedMeanAbsKappa(baselineCurvature.kappa, true);
  const double contactWallMeanKappa = weightedMeanAbsKappa(contactCurvature.kappa, true);
  const auto relativeErrorPercent = [&](double value) {
    return 100.0 * (value - analyticCapCurvature) / std::max(analyticCapCurvature, 1e-30);
  };
  const double baselineMeanErrorPercent = relativeErrorPercent(baselineMeanKappa);
  const double contactMeanErrorPercent = relativeErrorPercent(contactMeanKappa);
  const double baselineWallErrorPercent = relativeErrorPercent(baselineWallMeanKappa);
  const double contactWallErrorPercent = relativeErrorPercent(contactWallMeanKappa);
  const std::string analyticStatus =
      std::abs(contactWallErrorPercent) <= 20.0 ? "APPROXIMATE_WITHIN_20_PERCENT"
                                               : "DOWNGRADED_ANALYTIC_CAP_ERROR";

  std::ofstream wallCsv("benchmark_logs/surface_tension_contact_angle_curvature3d.csv");
  wallCsv << "case,cells,wall_mixed_cells,changed_wall_curvature_cells,"
             "baseline_fitted_cells,contact_fitted_cells,baseline_fallback_fraction,"
             "contact_fallback_fraction,baseline_p95_condition,contact_p95_condition,"
             "baseline_max_abs_curvature,contact_max_abs_curvature,"
             "mean_wall_curvature_delta,max_wall_curvature_delta,"
             "analytic_cap_curvature,baseline_weighted_mean_abs_kappa,"
             "contact_weighted_mean_abs_kappa,baseline_wall_weighted_mean_abs_kappa,"
             "contact_wall_weighted_mean_abs_kappa,baseline_mean_error_percent,"
             "contact_mean_error_percent,baseline_wall_error_percent,"
             "contact_wall_error_percent,status\n";
  wallCsv << "sessile_cap_diagnostic," << wallMesh.cells.size() << "," << wallMixedCells << ","
          << changedWallCurvatureCells << "," << baselineCurvature.fittedCells << ","
          << contactCurvature.fittedCells << "," << baselineCurvature.fallbackFraction << ","
          << contactCurvature.fallbackFraction << "," << baselineCurvature.p95StencilCondition << ","
          << contactCurvature.p95StencilCondition << "," << baselineCurvature.maxAbsCurvature << ","
          << contactCurvature.maxAbsCurvature << "," << meanWallCurvatureDelta << ","
          << maxWallCurvatureDelta << "," << analyticCapCurvature << ","
          << baselineMeanKappa << "," << contactMeanKappa << ","
          << baselineWallMeanKappa << "," << contactWallMeanKappa << ","
          << baselineMeanErrorPercent << "," << contactMeanErrorPercent << ","
          << baselineWallErrorPercent << "," << contactWallErrorPercent << ","
          << analyticStatus << "\n";

  check(wallMixedCells > 0, "3D contact-angle curvature diagnostic has wall mixed cells");
  check(changedWallCurvatureCells > 0,
        "3D contact-angle curvature diagnostic changes wall-adjacent curvature");
  check(contactCurvature.fallbackFraction >= 0.0 && contactCurvature.fallbackFraction <= 1.0,
        "3D contact-angle curvature diagnostic fallback fraction bounded");
  check(std::isfinite(contactCurvature.maxAbsCurvature),
        "3D contact-angle curvature diagnostic curvature finite");
  check(std::isfinite(contactWallErrorPercent),
        "3D contact-angle curvature analytic cap diagnostic finite");

  struct SessileCapRefinementRow {
    int n = 0;
    int cells = 0;
    int wallMixed = 0;
    double baselineWallError = 0.0;
    double contactWallError = 0.0;
    double baselineFallback = 0.0;
    double contactFallback = 0.0;
  };
  auto runSessileCapRefinement = [&](int n) {
    SessileCapRefinementRow row;
    row.n = n;
    const int ny = std::max(8, static_cast<int>(std::round(0.72 * static_cast<double>(n))));
    auto m = fvm::Mesh3D::hexGrid(n, ny, n, 1.0, 0.7, 1.0, 0.06);
    fvm::ScalarField a(m.cells.size(), 0.0);
    for (size_t ci = 0; ci < m.cells.size(); ++ci) {
      const double r = (m.cells[ci].centroid - capCenter).norm();
      a[ci] = 0.5 * (1.0 - std::tanh((r - capRadius) / capThickness));
    }
    const double yMax = 0.7 / std::max(ny, 1) * 2.0;
    const auto base = fvm::curvatureFromLocalPlicQuadricReport3D(m, a, 28);
    const auto contact = fvm::curvatureFromLocalPlicQuadricReport3D(
        m, a, 28, &wallNormal, contactAngleDeg, yMax);
    auto wallWeighted = [&](const fvm::ScalarField& kappa) {
      double weighted = 0.0;
      double weightSum = 0.0;
      int mixed = 0;
      for (int ci = 0; ci < static_cast<int>(m.cells.size()); ++ci) {
        const double ai = std::clamp(a[ci], 0.0, 1.0);
        if (ai <= 1e-6 || ai >= 1.0 - 1e-6) continue;
        if (m.cells[ci].centroid.y() > yMax) continue;
        const double w = std::max(ai * (1.0 - ai) * m.cells[ci].V, 0.0);
        weighted += w * std::abs(kappa[ci]);
        weightSum += w;
        ++mixed;
      }
      row.wallMixed = mixed;
      return weightSum > 0.0 ? weighted / weightSum : 0.0;
    };
    row.cells = static_cast<int>(m.cells.size());
    row.baselineWallError = relativeErrorPercent(wallWeighted(base.kappa));
    row.contactWallError = relativeErrorPercent(wallWeighted(contact.kappa));
    row.baselineFallback = base.fallbackFraction;
    row.contactFallback = contact.fallbackFraction;
    return row;
  };
  const std::vector<SessileCapRefinementRow> capRows = {
      runSessileCapRefinement(10),
      runSessileCapRefinement(14),
      runSessileCapRefinement(18)};
  const bool contactAbsErrorNetDecreasing =
      std::abs(capRows.back().contactWallError) < std::abs(capRows.front().contactWallError);
  std::ofstream refineCsv("benchmark_logs/surface_tension_contact_angle_curvature_refinement3d.csv");
  refineCsv << "n,cells,wall_mixed_cells,baseline_wall_error_percent,"
               "contact_wall_error_percent,baseline_fallback_fraction,"
               "contact_fallback_fraction,status\n";
  for (const auto& row : capRows) {
    const std::string rowStatus =
        contactAbsErrorNetDecreasing ? "CONTACT_ERROR_NET_DECREASING"
                                     : "DOWNGRADED_CONTACT_ERROR_NOT_DECREASING";
    refineCsv << row.n << "," << row.cells << "," << row.wallMixed << ","
              << row.baselineWallError << "," << row.contactWallError << ","
              << row.baselineFallback << "," << row.contactFallback << ","
              << rowStatus << "\n";
    check(row.wallMixed > 0, "3D contact-angle refinement has wall mixed cells");
    check(std::isfinite(row.contactWallError),
          "3D contact-angle refinement wall curvature error finite");
  }

  std::cout << "surface_tension3d_snGrad_diff=" << report.maxSnGradDifference
            << " surface_tension3d_max_kappa=" << report.maxCurvatureMagnitude
            << " surface_tension3d_max_force=" << maxForce
            << " surface_tension3d_max_pressure_grad=" << maxGp
            << " surface_tension3d_state_balance_residual=" << state.maxBalanceResidual << "\n";
}
