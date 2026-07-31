#include "TestUtil.hpp"
#include "electrospray/CandidoTaylorConeJet3D.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace {

struct AxialStats {
  int tailSamples = 0;
  int developedSamples = 0;
  double meanCurrent = 0.0;
  double meanAbsCharge = 0.0;
  double meanAbsUy = 0.0;
  double meanAbsElectricSource = 0.0;
};

double ratio(double high, double low) {
  return high / std::max(low, 1e-30);
}

AxialStats alpha05Stats(const electrospray::CandidoConeJetSmokeReport3D& r,
                        double minAreaDi2) {
  AxialStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    ++s.tailSamples;
    if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
    ++s.developedSamples;
    s.meanCurrent += std::abs(h.developedJetAlpha05ConvectiveCurrent);
    s.meanAbsCharge += h.developedJetMeanAlpha05AbsCharge;
    s.meanAbsUy += h.developedJetMeanAlpha05AbsUy;
    s.meanAbsElectricSource += h.developedJetMeanAlpha05AbsElectricMomentumSourceY;
  }
  if (s.developedSamples > 0) {
    const double n = static_cast<double>(s.developedSamples);
    s.meanCurrent /= n;
    s.meanAbsCharge /= n;
    s.meanAbsUy /= n;
    s.meanAbsElectricSource /= n;
  }
  return s;
}

void removeInterruptedCandidoCsvs(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir)) return;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const std::filesystem::path path = entry.path();
    const std::string name = path.filename().string();
    if (name.rfind("candido_", 0) != 0 || path.extension() != ".csv") continue;
    if (std::filesystem::file_size(path) == 0) {
      std::filesystem::remove(path);
    }
  }
}

void checkSmokeNumerics(const electrospray::CandidoConeJetSmokeReport3D& r,
                        const std::string& label) {
  check(r.cells > 0 && r.faces > 0, label + " mesh is non-empty");
  check(std::isfinite(r.alphaMassDrift) && r.alphaMassDrift <= 1e-3,
        label + " keeps VoF mass bounded");
  check(std::isfinite(r.maxDiv) && r.maxDiv <= 1e-7,
        label + " keeps projection continuity bounded");
  check(std::isfinite(r.maxImplicitOhmicChargeResidual),
        label + " has finite implicit Ohmic residual");
  check(std::isfinite(r.relativeChargeBudgetResidual),
        label + " has finite charge-budget residual");
}

}  // namespace

int main() {
  std::filesystem::create_directories("benchmark_logs");
  removeInterruptedCandidoCsvs("benchmark_logs");

  electrospray::CandidoTaylorConeJetSetup setup;
  electrospray::CandidoConeJetSmokeOptions3D baseline;
  baseline.usePoissonFaceConductiveCurrent = true;
  baseline.usePoissonFaceMaxwellForce = true;
  baseline.implicitOhmicChargeProjection = true;
  baseline.useDimensionalElectricalScaling = true;
  baseline.useElectricRelaxationTimeStepLimit = true;
  baseline.electricRelaxationTimeStepSafety = 1.0;
  baseline.electricDriveCaExponent = 0.0;
  baseline.useBoundaryChargeAdvection = true;
  baseline.useVofInletBoundaryAlpha = true;
  baseline.suppressNozzleConductiveChargeFlux = true;

  electrospray::CandidoConeJetSmokeOptions3D candidate = baseline;
  candidate.conservativeChargeBounding = true;
  candidate.useInterfaceLocalizedChargeRedistribution = true;
  candidate.interfaceChargeRedistributionLiquidFloor = 0.02;
  candidate.refreshPotentialAfterChargeAdvance = true;

  const auto baselineLow = electrospray::runCandidoConeJetSmoke3D(0.25, setup, baseline);
  const auto baselineHigh = electrospray::runCandidoConeJetSmoke3D(0.42, setup, baseline);
  const auto candidateLow = electrospray::runCandidoConeJetSmoke3D(0.25, setup, candidate);
  const auto candidateHigh = electrospray::runCandidoConeJetSmoke3D(0.42, setup, candidate);

  checkSmokeNumerics(baselineLow, "baseline low-CaE");
  checkSmokeNumerics(baselineHigh, "baseline high-CaE");
  checkSmokeNumerics(candidateLow, "conservative surface-charge low-CaE");
  checkSmokeNumerics(candidateHigh, "conservative surface-charge high-CaE");

  constexpr double minAreaDi2 = 1e-4;
  const AxialStats baseLow = alpha05Stats(baselineLow, minAreaDi2);
  const AxialStats baseHigh = alpha05Stats(baselineHigh, minAreaDi2);
  const AxialStats candLow = alpha05Stats(candidateLow, minAreaDi2);
  const AxialStats candHigh = alpha05Stats(candidateHigh, minAreaDi2);
  const bool baselineComparable =
      baseLow.developedSamples > 0 && baseHigh.developedSamples > 0;
  const bool candidateComparable =
      candLow.developedSamples > 0 && candHigh.developedSamples > 0;
  const double baselineCurrentRatio =
      baselineComparable ? ratio(baseHigh.meanCurrent, baseLow.meanCurrent)
                         : std::numeric_limits<double>::infinity();
  const double candidateCurrentRatio =
      candidateComparable ? ratio(candHigh.meanCurrent, candLow.meanCurrent)
                          : std::numeric_limits<double>::infinity();
  const double candidateChargeRatio =
      candidateComparable ? ratio(candHigh.meanAbsCharge, candLow.meanAbsCharge)
                          : std::numeric_limits<double>::infinity();
  const double candidateVelocityRatio =
      candidateComparable ? ratio(candHigh.meanAbsUy, candLow.meanAbsUy)
                          : std::numeric_limits<double>::infinity();
  const double candidateElectricSourceRatio =
      candidateComparable ? ratio(candHigh.meanAbsElectricSource,
                                  candLow.meanAbsElectricSource)
                          : std::numeric_limits<double>::infinity();

  const bool budgetOk =
      std::max(std::abs(candidateLow.relativeChargeBudgetResidual),
               std::abs(candidateHigh.relativeChargeBudgetResidual)) <= 1.0;
  const bool numericsOk = candidateLow.alphaMassDrift <= 1e-3 &&
                          candidateHigh.alphaMassDrift <= 1e-3 &&
                          candidateLow.maxDiv <= 1e-7 &&
                          candidateHigh.maxDiv <= 1e-7 &&
                          candidateLow.maxImplicitOhmicChargeResidual <= 1e-8 &&
                          candidateHigh.maxImplicitOhmicChargeResidual <= 1e-8;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (candidateComparable) {
    if (!budgetOk || !numericsOk) {
      status = "DOWNGRADED_CONSERVATIVE_SURFACE_CHARGE_BUDGET_OR_NUMERICS";
    } else if (candidateCurrentRatio <= 2.0 &&
               candidateCurrentRatio <= baselineCurrentRatio) {
      status = "APPROXIMATE_CONSERVATIVE_SURFACE_CHARGE_REDUCES_CURRENT_SENSITIVITY";
    } else {
      status = "DOWNGRADED_CONSERVATIVE_SURFACE_CHARGE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }

  std::ofstream csv("benchmark_logs/candido_conservative_surface_charge_closure3d.csv");
  csv << "case,low_ca_e,high_ca_e,min_area_di2,"
         "baseline_low_developed_samples,baseline_high_developed_samples,"
         "candidate_low_developed_samples,candidate_high_developed_samples,"
         "baseline_axial_alpha05_current_ratio,"
         "candidate_axial_alpha05_current_ratio,candidate_charge_ratio,"
         "candidate_velocity_ratio,candidate_electric_source_ratio,"
         "candidate_low_alpha05_current,candidate_high_alpha05_current,"
         "candidate_low_mean_abs_charge,candidate_high_mean_abs_charge,"
         "candidate_low_mean_abs_uy,candidate_high_mean_abs_uy,"
         "candidate_low_mean_abs_electric_source,"
         "candidate_high_mean_abs_electric_source,"
         "candidate_low_implicit_ohmic_residual,"
         "candidate_high_implicit_ohmic_residual,"
         "candidate_low_charge_clamp_l1,candidate_high_charge_clamp_l1,"
         "candidate_low_redistribution_residual,"
         "candidate_high_redistribution_residual,"
         "candidate_low_redistribution_deficit_l1,"
         "candidate_high_redistribution_deficit_l1,"
         "candidate_low_relative_charge_budget_residual,"
         "candidate_high_relative_charge_budget_residual,"
         "candidate_low_alpha_mass_drift,candidate_high_alpha_mass_drift,"
         "candidate_low_max_div,candidate_high_max_div,"
         "candidate_low_morphology_error_percent,"
         "candidate_high_max_radial_asymmetry,status\n";
  csv << "conservative_surface_charge_closure,"
      << baselineLow.targetCaE << "," << baselineHigh.targetCaE << ","
      << minAreaDi2 << "," << baseLow.developedSamples << ","
      << baseHigh.developedSamples << "," << candLow.developedSamples << ","
      << candHigh.developedSamples << "," << baselineCurrentRatio << ","
      << candidateCurrentRatio << "," << candidateChargeRatio << ","
      << candidateVelocityRatio << "," << candidateElectricSourceRatio << ","
      << candLow.meanCurrent << "," << candHigh.meanCurrent << ","
      << candLow.meanAbsCharge << "," << candHigh.meanAbsCharge << ","
      << candLow.meanAbsUy << "," << candHigh.meanAbsUy << ","
      << candLow.meanAbsElectricSource << "," << candHigh.meanAbsElectricSource
      << "," << candidateLow.maxImplicitOhmicChargeResidual << ","
      << candidateHigh.maxImplicitOhmicChargeResidual << ","
      << candidateLow.cumulativeChargeClampCorrectionL1 << ","
      << candidateHigh.cumulativeChargeClampCorrectionL1 << ","
      << candidateLow.maxChargeRedistributionResidual << ","
      << candidateHigh.maxChargeRedistributionResidual << ","
      << candidateLow.cumulativeChargeRedistributionDeficitL1 << ","
      << candidateHigh.cumulativeChargeRedistributionDeficitL1 << ","
      << candidateLow.relativeChargeBudgetResidual << ","
      << candidateHigh.relativeChargeBudgetResidual << ","
      << candidateLow.alphaMassDrift << "," << candidateHigh.alphaMassDrift
      << "," << candidateLow.maxDiv << "," << candidateHigh.maxDiv << ","
      << 100.0 * std::abs(candidateLow.finalMidplaneJetRadius -
                          baselineLow.finalMidplaneJetRadius) /
             std::max(std::abs(baselineLow.finalMidplaneJetRadius), 1e-30)
      << "," << candidateHigh.finalRadialAsymmetry << "," << status << "\n";
  csv.flush();

  check(csv.good(), "conservative surface-charge diagnostic CSV written");
  std::cout << "conservative_surface_charge_status=" << status << "\n";
  std::cout << "baseline_alpha05_current_ratio=" << baselineCurrentRatio << "\n";
  std::cout << "candidate_alpha05_current_ratio=" << candidateCurrentRatio << "\n";
  std::cout << "candidate_charge_budget_residuals="
            << candidateLow.relativeChargeBudgetResidual << ","
            << candidateHigh.relativeChargeBudgetResidual << "\n";
  return 0;
}
