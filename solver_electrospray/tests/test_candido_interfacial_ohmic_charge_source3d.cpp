#include "TestUtil.hpp"
#include "electrospray/CandidoTaylorConeJet3D.hpp"

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
  double meanAbsElectricSource = 0.0;
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
    if (std::filesystem::file_size(path) == 0) std::filesystem::remove(path);
  }
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
  candidate.useInterfacialOhmicChargeSource = true;

  const auto baselineLow = electrospray::runCandidoConeJetSmoke3D(0.25, setup, baseline);
  const auto baselineHigh = electrospray::runCandidoConeJetSmoke3D(0.42, setup, baseline);
  const auto candidateLow = electrospray::runCandidoConeJetSmoke3D(0.25, setup, candidate);
  const auto candidateHigh = electrospray::runCandidoConeJetSmoke3D(0.42, setup, candidate);

  checkSmokeNumerics(baselineLow, "baseline low-CaE");
  checkSmokeNumerics(baselineHigh, "baseline high-CaE");
  checkSmokeNumerics(candidateLow, "interfacial Ohmic source low-CaE");
  checkSmokeNumerics(candidateHigh, "interfacial Ohmic source high-CaE");

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
  const bool sourceActive = candidateLow.maxInterfacialOhmicChargeSourceCells > 0 &&
                            candidateHigh.maxInterfacialOhmicChargeSourceCells > 0 &&
                            candidateLow.maxInterfacialOhmicChargeSourceDensity > 0.0 &&
                            candidateHigh.maxInterfacialOhmicChargeSourceDensity > 0.0;
  const bool budgetOk =
      std::max(std::abs(candidateLow.relativeChargeBudgetResidual),
               std::abs(candidateHigh.relativeChargeBudgetResidual)) <= 1.0;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (candidateComparable) {
    if (!sourceActive) {
      status = "BLOCKED_INTERFACIAL_OHMIC_SOURCE_INACTIVE";
    } else if (!budgetOk) {
      status = "DOWNGRADED_INTERFACIAL_OHMIC_SOURCE_BREAKS_CHARGE_BUDGET";
    } else if (candidateCurrentRatio <= 2.0 &&
               candidateCurrentRatio <= baselineCurrentRatio) {
      status = "APPROXIMATE_INTERFACIAL_OHMIC_SOURCE_REDUCES_CURRENT_SENSITIVITY";
    } else {
      status = "DOWNGRADED_INTERFACIAL_OHMIC_SOURCE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }

  std::ofstream csv("benchmark_logs/candido_interfacial_ohmic_charge_source3d.csv");
  csv << "case,low_ca_e,high_ca_e,min_area_di2,"
         "baseline_low_developed_samples,baseline_high_developed_samples,"
         "candidate_low_developed_samples,candidate_high_developed_samples,"
         "baseline_axial_alpha05_current_ratio,"
         "candidate_axial_alpha05_current_ratio,candidate_charge_ratio,"
         "candidate_velocity_ratio,candidate_electric_source_ratio,"
         "candidate_low_source_cells,candidate_high_source_cells,"
         "candidate_low_max_source_density,candidate_high_max_source_density,"
         "candidate_low_applied_source_charge,"
         "candidate_high_applied_source_charge,"
         "candidate_low_source_clamp_l1,candidate_high_source_clamp_l1,"
         "candidate_low_relative_charge_budget_residual,"
         "candidate_high_relative_charge_budget_residual,"
         "candidate_low_alpha_mass_drift,candidate_high_alpha_mass_drift,"
         "candidate_low_max_div,candidate_high_max_div,"
         "candidate_low_post_source_potential_residual,"
         "candidate_high_post_source_potential_residual,"
         "candidate_low_post_source_gauss_residual,"
         "candidate_high_post_source_gauss_residual,status\n";
  csv << "interfacial_ohmic_charge_source,"
      << baselineLow.targetCaE << "," << baselineHigh.targetCaE << ","
      << minAreaDi2 << "," << baseLow.developedSamples << ","
      << baseHigh.developedSamples << "," << candLow.developedSamples << ","
      << candHigh.developedSamples << "," << baselineCurrentRatio << ","
      << candidateCurrentRatio << "," << candidateChargeRatio << ","
      << candidateVelocityRatio << "," << candidateElectricSourceRatio << ","
      << candidateLow.maxInterfacialOhmicChargeSourceCells << ","
      << candidateHigh.maxInterfacialOhmicChargeSourceCells << ","
      << candidateLow.maxInterfacialOhmicChargeSourceDensity << ","
      << candidateHigh.maxInterfacialOhmicChargeSourceDensity << ","
      << candidateLow.cumulativeInterfacialOhmicChargeSource << ","
      << candidateHigh.cumulativeInterfacialOhmicChargeSource << ","
      << candidateLow.cumulativeInterfacialOhmicChargeClampL1 << ","
      << candidateHigh.cumulativeInterfacialOhmicChargeClampL1 << ","
      << candidateLow.relativeChargeBudgetResidual << ","
      << candidateHigh.relativeChargeBudgetResidual << ","
      << candidateLow.alphaMassDrift << "," << candidateHigh.alphaMassDrift
      << "," << candidateLow.maxDiv << "," << candidateHigh.maxDiv << ","
      << candidateLow.maxPostChargePotentialResidual << ","
      << candidateHigh.maxPostChargePotentialResidual << ","
      << candidateLow.maxPostChargeRelativeGaussLawResidual << ","
      << candidateHigh.maxPostChargeRelativeGaussLawResidual << ","
      << status << "\n";
  csv.flush();

  check(csv.good(), "interfacial Ohmic charge-source diagnostic CSV written");
  std::cout << "interfacial_ohmic_charge_source_status=" << status << "\n";
  std::cout << "baseline_alpha05_current_ratio=" << baselineCurrentRatio << "\n";
  std::cout << "candidate_alpha05_current_ratio=" << candidateCurrentRatio << "\n";
  std::cout << "candidate_source_cells="
            << candidateLow.maxInterfacialOhmicChargeSourceCells << ","
            << candidateHigh.maxInterfacialOhmicChargeSourceCells << "\n";
  return 0;
}
