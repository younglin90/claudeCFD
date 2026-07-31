#include "TestUtil.hpp"
#include "electrospray/CandidoTaylorConeJet3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace {

struct AxialStats {
  int developedSamples = 0;
  double meanCurrent = 0.0;
  double meanAbsCharge = 0.0;
  double meanAbsUy = 0.0;
};

struct BoundaryStats {
  double total = 0.0;
  double nozzle = 0.0;
  double collector = 0.0;
  double lateral = 0.0;
  double nozzlePeak = 0.0;
  double collectorPeak = 0.0;
  double lateralPeak = 0.0;
};

double ratio(double high, double low) {
  return std::abs(high) / std::max(std::abs(low), 1e-30);
}

AxialStats alpha05Stats(const electrospray::CandidoConeJetSmokeReport3D& r,
                        double minAreaDi2) {
  AxialStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
    ++s.developedSamples;
    s.meanCurrent += std::abs(h.developedJetAlpha05ConvectiveCurrent);
    s.meanAbsCharge += h.developedJetMeanAlpha05AbsCharge;
    s.meanAbsUy += h.developedJetMeanAlpha05AbsUy;
  }
  if (s.developedSamples > 0) {
    const double n = static_cast<double>(s.developedSamples);
    s.meanCurrent /= n;
    s.meanAbsCharge /= n;
    s.meanAbsUy /= n;
  }
  return s;
}

BoundaryStats boundaryStats(const electrospray::CandidoConeJetSmokeReport3D& r) {
  BoundaryStats s;
  s.total = r.cumulativeConductiveBoundaryChargeFlux;
  s.nozzle = r.cumulativeConductiveBoundaryChargeFluxByPatch[2];
  s.collector = r.cumulativeConductiveBoundaryChargeFluxByPatch[3];
  s.lateral = r.cumulativeConductiveBoundaryChargeFluxByPatch[0] +
              r.cumulativeConductiveBoundaryChargeFluxByPatch[1] +
              r.cumulativeConductiveBoundaryChargeFluxByPatch[4] +
              r.cumulativeConductiveBoundaryChargeFluxByPatch[5];
  s.nozzlePeak = r.maxAbsConductiveBoundaryCurrentByPatch[2];
  s.collectorPeak = r.maxAbsConductiveBoundaryCurrentByPatch[3];
  s.lateralPeak = std::max({r.maxAbsConductiveBoundaryCurrentByPatch[0],
                            r.maxAbsConductiveBoundaryCurrentByPatch[1],
                            r.maxAbsConductiveBoundaryCurrentByPatch[4],
                            r.maxAbsConductiveBoundaryCurrentByPatch[5]});
  return s;
}

std::string dominantHighPatch(const BoundaryStats& high) {
  std::string name = "nozzle";
  double value = std::abs(high.nozzle);
  if (std::abs(high.collector) > value) {
    name = "collector";
    value = std::abs(high.collector);
  }
  if (std::abs(high.lateral) > value) {
    name = "lateral";
  }
  return name;
}

void checkSmokeNumerics(const electrospray::CandidoConeJetSmokeReport3D& r,
                        const std::string& label) {
  check(r.cells > 0 && r.faces > 0, label + " mesh is non-empty");
  check(std::isfinite(r.alphaMassDrift) && r.alphaMassDrift <= 1e-3,
        label + " keeps VoF mass bounded");
  check(std::isfinite(r.maxDiv) && r.maxDiv <= 1e-7,
        label + " keeps projection continuity bounded");
  check(std::isfinite(r.relativeChargeBudgetResidual),
        label + " has finite charge-budget residual");
}

void writeIsolationRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    const electrospray::CandidoConeJetSmokeReport3D& paperLow,
    const electrospray::CandidoConeJetSmokeReport3D& paperHigh,
    double minAreaDi2) {
  const AxialStats lowAxial = alpha05Stats(low, minAreaDi2);
  const AxialStats highAxial = alpha05Stats(high, minAreaDi2);
  const BoundaryStats lowBoundary = boundaryStats(low);
  const BoundaryStats highBoundary = boundaryStats(high);
  const BoundaryStats paperLowBoundary = boundaryStats(paperLow);
  const BoundaryStats paperHighBoundary = boundaryStats(paperHigh);
  const bool comparable = lowAxial.developedSamples > 0 &&
                          highAxial.developedSamples > 0;
  const double axialCurrentRatio =
      comparable ? ratio(highAxial.meanCurrent, lowAxial.meanCurrent)
                 : std::numeric_limits<double>::infinity();
  const double chargeRatio =
      comparable ? ratio(highAxial.meanAbsCharge, lowAxial.meanAbsCharge)
                 : std::numeric_limits<double>::infinity();
  const double velocityRatio =
      comparable ? ratio(highAxial.meanAbsUy, lowAxial.meanAbsUy)
                 : std::numeric_limits<double>::infinity();
  const double totalRatio = ratio(highBoundary.total, lowBoundary.total);
  const double nozzleRatio = ratio(highBoundary.nozzle, lowBoundary.nozzle);
  const double collectorRatio =
      ratio(highBoundary.collector, lowBoundary.collector);
  const double lateralRatio = ratio(highBoundary.lateral, lowBoundary.lateral);
  const double lowTotalVsPaper = ratio(lowBoundary.total, paperLowBoundary.total);
  const double highTotalVsPaper = ratio(highBoundary.total, paperHighBoundary.total);
  const double lowNozzleVsPaper =
      ratio(lowBoundary.nozzle, paperLowBoundary.nozzle);
  const double highNozzleVsPaper =
      ratio(highBoundary.nozzle, paperHighBoundary.nozzle);
  const double lowCollectorVsPaper =
      ratio(lowBoundary.collector, paperLowBoundary.collector);
  const double highCollectorVsPaper =
      ratio(highBoundary.collector, paperHighBoundary.collector);
  const double maxOptionEffectDeviation =
      std::max({std::abs(lowTotalVsPaper - 1.0),
                std::abs(highTotalVsPaper - 1.0),
                std::abs(lowNozzleVsPaper - 1.0),
                std::abs(highNozzleVsPaper - 1.0),
                std::abs(lowCollectorVsPaper - 1.0),
                std::abs(highCollectorVsPaper - 1.0)});
  const std::string dominant = dominantHighPatch(highBoundary);
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    if (name != "paper_charge_boundary" && maxOptionEffectDeviation <= 1e-9) {
      status = "DOWNGRADED_BOUNDARY_SWITCH_NO_EFFECT_UNDER_IMPLICIT_OHMIC_PROJECTION";
    } else if (axialCurrentRatio <= 2.0 && totalRatio <= 2.0) {
      status = "APPROXIMATE_ELECTRODE_BOUNDARY_WEAK_CURRENT_SENSITIVITY";
    } else if (dominant == "nozzle" && nozzleRatio > 2.0) {
      status = "DOWNGRADED_NOZZLE_ELECTRODE_CURRENT_DOMINATES";
    } else if (dominant == "collector" && collectorRatio > 2.0) {
      status = "DOWNGRADED_COLLECTOR_ELECTRODE_CURRENT_DOMINATES";
    } else if (dominant == "lateral" && lateralRatio > 2.0) {
      status = "DOWNGRADED_LATERAL_BOUNDARY_CURRENT_DOMINATES";
    } else {
      status = "DOWNGRADED_BOUNDARY_CURRENT_NOT_SOLE_LIMITER";
    }
  }
  csv << name << "," << low.targetCaE << "," << high.targetCaE << ","
      << minAreaDi2 << "," << lowAxial.developedSamples << ","
      << highAxial.developedSamples << "," << axialCurrentRatio << ","
      << chargeRatio << "," << velocityRatio << "," << lowBoundary.total << ","
      << highBoundary.total << "," << totalRatio << "," << lowBoundary.nozzle
      << "," << highBoundary.nozzle << "," << nozzleRatio << ","
      << lowBoundary.collector << "," << highBoundary.collector << ","
      << collectorRatio << "," << lowBoundary.lateral << ","
      << highBoundary.lateral << "," << lateralRatio << ","
      << lowBoundary.nozzlePeak << "," << highBoundary.nozzlePeak << ","
      << ratio(highBoundary.nozzlePeak, lowBoundary.nozzlePeak) << ","
      << lowBoundary.collectorPeak << "," << highBoundary.collectorPeak << ","
      << ratio(highBoundary.collectorPeak, lowBoundary.collectorPeak) << ","
      << lowBoundary.lateralPeak << "," << highBoundary.lateralPeak << ","
      << ratio(highBoundary.lateralPeak, lowBoundary.lateralPeak) << ","
      << lowTotalVsPaper << "," << highTotalVsPaper << ","
      << lowNozzleVsPaper << "," << highNozzleVsPaper << ","
      << lowCollectorVsPaper << "," << highCollectorVsPaper << ","
      << maxOptionEffectDeviation << ","
      << low.relativeChargeBudgetResidual << ","
      << high.relativeChargeBudgetResidual << "," << low.alphaMassDrift << ","
      << high.alphaMassDrift << "," << low.maxDiv << "," << high.maxDiv
      << "," << dominant << "," << status << "\n";
  csv.flush();
}

}  // namespace

int main() {
  std::filesystem::create_directories("benchmark_logs");

  electrospray::CandidoTaylorConeJetSetup setup;
  electrospray::CandidoConeJetSmokeOptions3D paper;
  paper.usePoissonFaceConductiveCurrent = true;
  paper.usePoissonFaceMaxwellForce = true;
  paper.implicitOhmicChargeProjection = true;
  paper.useDimensionalElectricalScaling = true;
  paper.useElectricRelaxationTimeStepLimit = true;
  paper.electricRelaxationTimeStepSafety = 1.0;
  paper.electricDriveCaExponent = 0.0;
  paper.useBoundaryChargeAdvection = true;
  paper.useVofInletBoundaryAlpha = true;
  paper.suppressNozzleConductiveChargeFlux = true;

  electrospray::CandidoConeJetSmokeOptions3D nozzleAllowed = paper;
  nozzleAllowed.suppressNozzleConductiveChargeFlux = false;

  electrospray::CandidoConeJetSmokeOptions3D collectorOnly = paper;
  collectorOnly.suppressNozzleConductiveChargeFlux = false;
  collectorOnly.collectorOnlyConductiveChargeFlux = true;

  electrospray::CandidoConeJetSmokeOptions3D implicitFilteredPaper = paper;
  implicitFilteredPaper.applyConductiveBoundaryFiltersInImplicitOhmic = true;

  electrospray::CandidoConeJetSmokeOptions3D implicitFilteredCollectorOnly =
      collectorOnly;
  implicitFilteredCollectorOnly.applyConductiveBoundaryFiltersInImplicitOhmic =
      true;

  const auto paperLow = electrospray::runCandidoConeJetSmoke3D(0.25, setup, paper);
  const auto paperHigh = electrospray::runCandidoConeJetSmoke3D(0.42, setup, paper);
  const auto nozzleLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, nozzleAllowed);
  const auto nozzleHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, nozzleAllowed);
  const auto collectorLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, collectorOnly);
  const auto collectorHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, collectorOnly);
  const auto filteredPaperLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, implicitFilteredPaper);
  const auto filteredPaperHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, implicitFilteredPaper);
  const auto filteredCollectorLow = electrospray::runCandidoConeJetSmoke3D(
      0.25, setup, implicitFilteredCollectorOnly);
  const auto filteredCollectorHigh = electrospray::runCandidoConeJetSmoke3D(
      0.42, setup, implicitFilteredCollectorOnly);

  checkSmokeNumerics(paperLow, "paper-charge boundary low-CaE");
  checkSmokeNumerics(paperHigh, "paper-charge boundary high-CaE");
  checkSmokeNumerics(nozzleLow, "nozzle-allowed boundary low-CaE");
  checkSmokeNumerics(nozzleHigh, "nozzle-allowed boundary high-CaE");
  checkSmokeNumerics(collectorLow, "collector-only boundary low-CaE");
  checkSmokeNumerics(collectorHigh, "collector-only boundary high-CaE");
  checkSmokeNumerics(filteredPaperLow, "implicit-filtered paper boundary low-CaE");
  checkSmokeNumerics(filteredPaperHigh, "implicit-filtered paper boundary high-CaE");
  checkSmokeNumerics(filteredCollectorLow,
                     "implicit-filtered collector-only boundary low-CaE");
  checkSmokeNumerics(filteredCollectorHigh,
                     "implicit-filtered collector-only boundary high-CaE");

  std::ofstream csv(
      "benchmark_logs/candido_electrode_surface_current_boundary_isolation3d.csv");
  csv << "case,low_ca_e,high_ca_e,min_area_di2,"
         "low_developed_samples,high_developed_samples,"
         "axial_alpha05_current_ratio,charge_ratio,velocity_ratio,"
         "low_total_cumulative_conductive_flux,"
         "high_total_cumulative_conductive_flux,total_cumulative_ratio,"
         "low_nozzle_cumulative_flux,high_nozzle_cumulative_flux,nozzle_ratio,"
         "low_collector_cumulative_flux,high_collector_cumulative_flux,"
         "collector_ratio,low_lateral_cumulative_flux,"
         "high_lateral_cumulative_flux,lateral_ratio,"
         "low_nozzle_peak,high_nozzle_peak,nozzle_peak_ratio,"
         "low_collector_peak,high_collector_peak,collector_peak_ratio,"
         "low_lateral_peak,high_lateral_peak,lateral_peak_ratio,"
         "low_total_vs_paper_ratio,high_total_vs_paper_ratio,"
         "low_nozzle_vs_paper_ratio,high_nozzle_vs_paper_ratio,"
         "low_collector_vs_paper_ratio,high_collector_vs_paper_ratio,"
         "max_option_effect_deviation,"
         "low_relative_charge_budget_residual,"
         "high_relative_charge_budget_residual,"
         "low_alpha_mass_drift,high_alpha_mass_drift,"
         "low_max_div,high_max_div,dominant_high_patch,status\n";
  constexpr double minAreaDi2 = 1e-4;
  writeIsolationRow(csv, "paper_charge_boundary", paperLow, paperHigh,
                    paperLow, paperHigh, minAreaDi2);
  writeIsolationRow(csv, "nozzle_allowed_boundary", nozzleLow, nozzleHigh,
                    paperLow, paperHigh, minAreaDi2);
  writeIsolationRow(csv, "collector_only_boundary", collectorLow, collectorHigh,
                    paperLow, paperHigh, minAreaDi2);
  writeIsolationRow(csv, "implicit_filtered_paper_charge_boundary",
                    filteredPaperLow, filteredPaperHigh, paperLow, paperHigh,
                    minAreaDi2);
  writeIsolationRow(csv, "implicit_filtered_collector_only_boundary",
                    filteredCollectorLow, filteredCollectorHigh, paperLow,
                    paperHigh, minAreaDi2);
  check(csv.good(), "electrode boundary isolation CSV written");

  std::cout << "electrode_surface_current_boundary_isolation=written\n";
  return 0;
}
