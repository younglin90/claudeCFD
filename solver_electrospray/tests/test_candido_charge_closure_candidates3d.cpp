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
  double meanAlpha05Convective = 0.0;
  double meanAbsCharge = 0.0;
  double meanAbsUy = 0.0;
};

double ratio(double highValue, double lowValue) {
  return highValue / std::max(lowValue, 1e-30);
}

AxialStats axialStats(const electrospray::CandidoConeJetSmokeReport3D& r,
                      double minAreaDi2) {
  AxialStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
    ++s.developedSamples;
    s.meanAlpha05Convective += std::abs(h.developedJetAlpha05ConvectiveCurrent);
    s.meanAbsCharge += h.developedJetMeanAlpha05AbsCharge;
    s.meanAbsUy += h.developedJetMeanAlpha05AbsUy;
  }
  if (s.developedSamples > 0) {
    const double inv = 1.0 / static_cast<double>(s.developedSamples);
    s.meanAlpha05Convective *= inv;
    s.meanAbsCharge *= inv;
    s.meanAbsUy *= inv;
  }
  return s;
}

double tailMeanAlpha05ElectricSource(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  double sum = 0.0;
  int samples = 0;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
    sum += h.developedJetMeanAlpha05AbsElectricMomentumSourceY;
    ++samples;
  }
  return samples > 0 ? sum / static_cast<double>(samples) : 0.0;
}

const electrospray::CandidoConeJetHistorySample3D* nearestHistoryAtMs(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double timeMs) {
  const auto* best = &r.history.front();
  double bestDistance = std::numeric_limits<double>::max();
  for (const auto& h : r.history) {
    const double hMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
    const double distance = std::abs(hMs - timeMs);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = &h;
    }
  }
  return best;
}

double morphologyErrorAtMs(const electrospray::CandidoConeJetSmokeReport3D& r,
                           double timeMs,
                           double referenceVolumeDi3) {
  const auto* h = nearestHistoryAtMs(r, timeMs);
  return 100.0 * (h->morphologyVolumeDi3 - referenceVolumeDi3) /
         std::max(std::abs(referenceVolumeDi3), 1e-30);
}

double maxMorphologyError04_07(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double e04 = morphologyErrorAtMs(r, 0.4, 1.2826510303495016);
  const double e07 = morphologyErrorAtMs(r, 0.7, 1.2550259882802302);
  return std::max(std::abs(e04), std::abs(e07));
}

double maxRadialAsymmetry(const electrospray::CandidoConeJetSmokeReport3D& r) {
  double value = 0.0;
  for (const auto& h : r.history) value = std::max(value, h.radialAsymmetry);
  return value;
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

void writeChargeSubcyclingDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baseline,
    const electrospray::CandidoConeJetSmokeReport3D& subcycled,
    int subcycles) {
  const double clampRatio =
      subcycled.cumulativeChargeClampCorrectionL1 /
      std::max(baseline.cumulativeChargeClampCorrectionL1, 1e-30);
  const double residualRatio =
      subcycled.relativeChargeBudgetResidual /
      std::max(baseline.relativeChargeBudgetResidual, 1e-30);
  const double currentRatio =
      subcycled.maxConvectiveCurrent /
      std::max(baseline.maxConvectiveCurrent, 1e-30);
  const std::string status =
      (clampRatio < 1.0 && residualRatio < 1.0)
          ? "SUBCYCLING_REDUCES_CHARGE_CLIPPING"
          : "DOWNGRADED_SUBCYCLING_DOES_NOT_REDUCE_CLIPPING";
  csv << baseline.targetCaE << "," << subcycles << ","
      << baseline.relativeChargeBudgetResidual << ","
      << subcycled.relativeChargeBudgetResidual << "," << residualRatio << ","
      << baseline.cumulativeChargeClampCorrectionL1 << ","
      << subcycled.cumulativeChargeClampCorrectionL1 << "," << clampRatio << ","
      << baseline.maxChargeClampedCells << ","
      << subcycled.maxChargeClampedCells << ","
      << baseline.maxUnclampedAbsCharge << ","
      << subcycled.maxUnclampedAbsCharge << ","
      << baseline.maxConvectiveCurrent << ","
      << subcycled.maxConvectiveCurrent << "," << currentRatio << ","
      << baseline.maxVelocity << "," << subcycled.maxVelocity << ","
      << status << "\n";
}

void writeChargeConservativeBoundingDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baseline,
    const electrospray::CandidoConeJetSmokeReport3D& bounded) {
  const double residualRatio =
      bounded.relativeChargeBudgetResidual /
      std::max(baseline.relativeChargeBudgetResidual, 1e-30);
  const double currentRatio =
      bounded.maxConvectiveCurrent / std::max(baseline.maxConvectiveCurrent, 1e-30);
  const double clampRatio =
      bounded.cumulativeChargeClampCorrectionL1 /
      std::max(baseline.cumulativeChargeClampCorrectionL1, 1e-30);
  const std::string status =
      residualRatio < 1e-6 ? "CONSERVATIVE_BOUNDING_CLOSES_CHARGE_BUDGET"
                           : "DOWNGRADED_CONSERVATIVE_BOUNDING_RESIDUAL";
  csv << baseline.targetCaE << ","
      << baseline.relativeChargeBudgetResidual << ","
      << bounded.relativeChargeBudgetResidual << "," << residualRatio << ","
      << baseline.cumulativeChargeClampCorrectionL1 << ","
      << bounded.cumulativeChargeClampCorrectionL1 << "," << clampRatio << ","
      << bounded.maxChargeRedistributionResidual << ","
      << baseline.maxChargeClampedCells << "," << bounded.maxChargeClampedCells
      << "," << baseline.maxUnclampedAbsCharge << ","
      << bounded.maxUnclampedAbsCharge << "," << baseline.maxConvectiveCurrent
      << "," << bounded.maxConvectiveCurrent << "," << currentRatio << ","
      << baseline.maxVelocity << "," << bounded.maxVelocity << ","
      << status << "\n";
}

void writeChargeLimitSensitivityRow(
    std::ofstream& csv,
    double chargeLimitBase,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  csv << r.targetCaE << "," << chargeLimitBase << ","
      << r.relativeChargeBudgetResidual << ","
      << r.cumulativeChargeClampCorrectionL1 << ","
      << r.maxChargeClampedCells << "," << r.maxUnclampedAbsCharge << ","
      << r.maxCharge << "," << r.minCharge << "," << r.maxConvectiveCurrent
      << "," << r.maxVelocity << "," << r.alphaMassDrift << "," << r.maxDiv
      << ",DIAGNOSTIC_Q_LIMIT_SENSITIVITY\n";
}

electrospray::CandidoConeJetSmokeOptions3D paperChargeBoundaryOptions() {
  electrospray::CandidoConeJetSmokeOptions3D opt;
  opt.conservativeChargeBounding = true;
  opt.useRayleighChargeLimit = true;
  opt.usePoissonFaceConductiveCurrent = true;
  opt.usePoissonFaceMaxwellForce = true;
  opt.implicitOhmicChargeProjection = true;
  opt.useDimensionalElectricalScaling = true;
  opt.useElectricRelaxationTimeStepLimit = true;
  opt.electricRelaxationTimeStepSafety = 1.0;
  opt.electricDriveCaExponent = 0.0;
  opt.useBoundaryChargeAdvection = true;
  opt.useVofInletBoundaryAlpha = true;
  opt.suppressNozzleConductiveChargeFlux = true;
  return opt;
}

struct ClosureStats {
  AxialStats baselineLow;
  AxialStats baselineHigh;
  AxialStats candidateLow;
  AxialStats candidateHigh;
  double baselineAxialRatio = std::numeric_limits<double>::infinity();
  double candidateAxialRatio = std::numeric_limits<double>::infinity();
  double candidateChargeRatio = std::numeric_limits<double>::infinity();
  double candidateVelocityRatio = std::numeric_limits<double>::infinity();
  double candidateLowElectricSource = 0.0;
  double candidateHighElectricSource = 0.0;
  double candidateElectricSourceRatio = std::numeric_limits<double>::infinity();
  double candidateMorphologyError = std::numeric_limits<double>::infinity();
  double candidateHighAsymmetry = 0.0;
  bool baseComparable = false;
  bool candidateComparable = false;
};

ClosureStats closureStats(
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& candidateLow,
    const electrospray::CandidoConeJetSmokeReport3D& candidateHigh,
    double minAreaDi2) {
  ClosureStats s;
  s.baselineLow = axialStats(baselineLow, minAreaDi2);
  s.baselineHigh = axialStats(baselineHigh, minAreaDi2);
  s.candidateLow = axialStats(candidateLow, minAreaDi2);
  s.candidateHigh = axialStats(candidateHigh, minAreaDi2);
  s.baseComparable =
      s.baselineLow.developedSamples > 0 && s.baselineHigh.developedSamples > 0;
  s.candidateComparable =
      s.candidateLow.developedSamples > 0 && s.candidateHigh.developedSamples > 0;
  if (s.baseComparable) {
    s.baselineAxialRatio = ratio(s.baselineHigh.meanAlpha05Convective,
                                 s.baselineLow.meanAlpha05Convective);
  }
  if (s.candidateComparable) {
    s.candidateAxialRatio = ratio(s.candidateHigh.meanAlpha05Convective,
                                  s.candidateLow.meanAlpha05Convective);
    s.candidateChargeRatio =
        ratio(s.candidateHigh.meanAbsCharge, s.candidateLow.meanAbsCharge);
    s.candidateVelocityRatio =
        ratio(s.candidateHigh.meanAbsUy, s.candidateLow.meanAbsUy);
    s.candidateLowElectricSource =
        tailMeanAlpha05ElectricSource(candidateLow, minAreaDi2);
    s.candidateHighElectricSource =
        tailMeanAlpha05ElectricSource(candidateHigh, minAreaDi2);
    s.candidateElectricSourceRatio =
        ratio(s.candidateHighElectricSource, s.candidateLowElectricSource);
  }
  s.candidateMorphologyError = maxMorphologyError04_07(candidateLow);
  s.candidateHighAsymmetry = maxRadialAsymmetry(candidateHigh);
  return s;
}

void writePostChargePotentialRefreshRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& candidateLow,
    const electrospray::CandidoConeJetSmokeReport3D& candidateHigh,
    double minAreaDi2) {
  const ClosureStats s =
      closureStats(baselineLow, baselineHigh, candidateLow, candidateHigh,
                   minAreaDi2);
  const bool numericalQuality =
      candidateLow.alphaMassDrift <= 1e-3 && candidateHigh.alphaMassDrift <= 1e-3 &&
      candidateLow.maxDiv <= 1e-7 && candidateHigh.maxDiv <= 1e-7 &&
      candidateLow.maxPostChargePotentialResidual <= 1e-7 &&
      candidateHigh.maxPostChargePotentialResidual <= 1e-7;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (s.candidateComparable) {
    if (!numericalQuality) {
      status = "DOWNGRADED_POST_CHARGE_REFRESH_NUMERICAL_QUALITY";
    } else if (s.candidateAxialRatio <= 2.0 &&
               s.candidateMorphologyError <= 10.0 &&
               s.candidateHighAsymmetry >= 0.05) {
      status = "APPROXIMATE_POST_CHARGE_REFRESH_ALL_GUARDS_GREEN";
    } else if (s.candidateAxialRatio < s.baselineAxialRatio) {
      status = "APPROXIMATE_POST_CHARGE_REFRESH_REDUCES_CURRENT_SENSITIVITY";
    } else {
      status = "DOWNGRADED_POST_CHARGE_REFRESH_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }
  csv << "post_charge_potential_refresh," << baselineLow.targetCaE << ","
      << baselineHigh.targetCaE << "," << minAreaDi2 << ","
      << s.baselineLow.developedSamples << ","
      << s.baselineHigh.developedSamples << ","
      << s.candidateLow.developedSamples << ","
      << s.candidateHigh.developedSamples << "," << s.baselineAxialRatio
      << "," << s.candidateAxialRatio << "," << s.candidateChargeRatio << ","
      << s.candidateVelocityRatio << "," << s.candidateElectricSourceRatio
      << "," << s.candidateLow.meanAlpha05Convective << ","
      << s.candidateHigh.meanAlpha05Convective << ","
      << s.candidateLow.meanAbsCharge << ","
      << s.candidateHigh.meanAbsCharge << "," << s.candidateLow.meanAbsUy
      << "," << s.candidateHigh.meanAbsUy << ","
      << s.candidateLowElectricSource << "," << s.candidateHighElectricSource
      << "," << candidateLow.maxPostChargePotentialResidual << ","
      << candidateHigh.maxPostChargePotentialResidual << ","
      << candidateLow.maxPostChargeRelativeGaussLawResidual << ","
      << candidateHigh.maxPostChargeRelativeGaussLawResidual << ","
      << candidateLow.alphaMassDrift << "," << candidateHigh.alphaMassDrift
      << "," << candidateLow.maxDiv << "," << candidateHigh.maxDiv << ","
      << s.candidateMorphologyError << "," << s.candidateHighAsymmetry << ","
      << status << "\n";
}

void writeInterfaceChargeTransportRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& candidateLow,
    const electrospray::CandidoConeJetSmokeReport3D& candidateHigh,
    double minAreaDi2) {
  const ClosureStats s =
      closureStats(baselineLow, baselineHigh, candidateLow, candidateHigh,
                   minAreaDi2);
  const bool numericalQuality =
      candidateLow.alphaMassDrift <= 1e-3 && candidateHigh.alphaMassDrift <= 1e-3 &&
      candidateLow.maxDiv <= 1e-7 && candidateHigh.maxDiv <= 1e-7;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (s.candidateComparable) {
    if (!numericalQuality) {
      status = "DOWNGRADED_INTERFACE_CHARGE_NUMERICAL_QUALITY";
    } else if (s.candidateAxialRatio <= 2.0 &&
               s.candidateMorphologyError <= 10.0 &&
               s.candidateHighAsymmetry >= 0.05) {
      status = "APPROXIMATE_INTERFACE_CHARGE_CANDIDATE_ALL_GUARDS_GREEN";
    } else if (s.candidateAxialRatio <= 2.0) {
      status =
          "DOWNGRADED_INTERFACE_CHARGE_WEAK_CURRENT_WITH_MORPHOLOGY_OR_WHIP_TRADEOFF";
    } else if (s.candidateAxialRatio < s.baselineAxialRatio) {
      status =
          "APPROXIMATE_INTERFACE_CHARGE_REDUCES_CURRENT_SENSITIVITY_DIAGNOSTIC_ONLY";
    } else {
      status = "DOWNGRADED_INTERFACE_CHARGE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }
  csv << "interface_localized_charge_transport," << baselineLow.targetCaE
      << "," << baselineHigh.targetCaE << "," << minAreaDi2 << ","
      << s.baselineLow.developedSamples << ","
      << s.baselineHigh.developedSamples << ","
      << s.candidateLow.developedSamples << ","
      << s.candidateHigh.developedSamples << "," << s.baselineAxialRatio
      << "," << s.candidateAxialRatio << "," << s.candidateChargeRatio << ","
      << s.candidateVelocityRatio << ","
      << s.candidateLow.meanAlpha05Convective << ","
      << s.candidateHigh.meanAlpha05Convective << ","
      << s.candidateLow.meanAbsCharge << ","
      << s.candidateHigh.meanAbsCharge << "," << s.candidateLow.meanAbsUy
      << "," << s.candidateHigh.meanAbsUy << ","
      << candidateLow.cumulativeChargeClampCorrectionL1 << ","
      << candidateHigh.cumulativeChargeClampCorrectionL1 << ","
      << candidateLow.cumulativeChargeRedistributionDeficitL1 << ","
      << candidateHigh.cumulativeChargeRedistributionDeficitL1 << ","
      << candidateLow.maxChargeRedistributionWeightedCells << ","
      << candidateHigh.maxChargeRedistributionWeightedCells << ","
      << candidateLow.maxChargeRedistributionWeightedCapacity << ","
      << candidateHigh.maxChargeRedistributionWeightedCapacity << ","
      << candidateLow.relativeChargeBudgetResidual << ","
      << candidateHigh.relativeChargeBudgetResidual << ","
      << candidateLow.alphaMassDrift << "," << candidateHigh.alphaMassDrift
      << "," << candidateLow.maxDiv << "," << candidateHigh.maxDiv << ","
      << s.candidateMorphologyError << "," << s.candidateHighAsymmetry << ","
      << status << "\n";
}

void writeConductivityPotentialClosureRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& candidateLow,
    const electrospray::CandidoConeJetSmokeReport3D& candidateHigh,
    double minAreaDi2) {
  const ClosureStats s =
      closureStats(baselineLow, baselineHigh, candidateLow, candidateHigh,
                   minAreaDi2);
  const bool numericalQuality =
      candidateLow.alphaMassDrift <= 1e-3 && candidateHigh.alphaMassDrift <= 1e-3 &&
      candidateLow.maxDiv <= 1e-7 && candidateHigh.maxDiv <= 1e-7 &&
      candidateLow.maxConductivityPotentialResidual <= 1e-7 &&
      candidateHigh.maxConductivityPotentialResidual <= 1e-7;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (s.candidateComparable) {
    if (!numericalQuality) {
      status = "DOWNGRADED_CONDUCTIVITY_CLOSURE_NUMERICAL_QUALITY";
    } else if (s.candidateAxialRatio <= 2.0 &&
               s.candidateMorphologyError <= 10.0 &&
               s.candidateHighAsymmetry >= 0.05) {
      status = "APPROXIMATE_CONDUCTIVITY_CLOSURE_ALL_GUARDS_GREEN";
    } else if (s.candidateAxialRatio < s.baselineAxialRatio) {
      status = "APPROXIMATE_CONDUCTIVITY_CLOSURE_REDUCES_CURRENT_SENSITIVITY";
    } else {
      status = "DOWNGRADED_CONDUCTIVITY_CLOSURE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }
  csv << "conductivity_potential_charge_closure," << baselineLow.targetCaE
      << "," << baselineHigh.targetCaE << "," << minAreaDi2 << ","
      << s.baselineLow.developedSamples << ","
      << s.baselineHigh.developedSamples << ","
      << s.candidateLow.developedSamples << ","
      << s.candidateHigh.developedSamples << "," << s.baselineAxialRatio
      << "," << s.candidateAxialRatio << "," << s.candidateChargeRatio << ","
      << s.candidateVelocityRatio << "," << s.candidateElectricSourceRatio
      << "," << s.candidateLow.meanAlpha05Convective << ","
      << s.candidateHigh.meanAlpha05Convective << ","
      << s.candidateLow.meanAbsCharge << ","
      << s.candidateHigh.meanAbsCharge << "," << s.candidateLow.meanAbsUy
      << "," << s.candidateHigh.meanAbsUy << ","
      << s.candidateLowElectricSource << "," << s.candidateHighElectricSource
      << "," << candidateLow.maxConductivityPotentialResidual << ","
      << candidateHigh.maxConductivityPotentialResidual << ","
      << candidateLow.cumulativeConductivityClosureClampCorrectionL1 << ","
      << candidateHigh.cumulativeConductivityClosureClampCorrectionL1 << ","
      << candidateLow.relativeChargeBudgetResidual << ","
      << candidateHigh.relativeChargeBudgetResidual << ","
      << candidateLow.alphaMassDrift << "," << candidateHigh.alphaMassDrift
      << "," << candidateLow.maxDiv << "," << candidateHigh.maxDiv << ","
      << s.candidateMorphologyError << "," << s.candidateHighAsymmetry << ","
      << status << "\n";
}

}  // namespace

int main() {
  std::filesystem::create_directories("benchmark_logs");

  electrospray::CandidoTaylorConeJetSetup setup;
  constexpr double minAreaDi2 = 1e-4;

  const auto baselineOpt = paperChargeBoundaryOptions();
  auto postRefreshOpt = baselineOpt;
  postRefreshOpt.refreshPotentialAfterChargeAdvance = true;
  auto conductivityClosureOpt = baselineOpt;
  conductivityClosureOpt.useConductivityPotentialChargeClosure = true;
  auto interfaceChargeTransportOpt = baselineOpt;
  interfaceChargeTransportOpt.useInterfaceLocalizedChargeRedistribution = true;
  interfaceChargeTransportOpt.interfaceChargeRedistributionLiquidFloor = 0.02;
  auto subcycledOpt = baselineOpt;
  subcycledOpt.chargeSubcycles = 4;
  auto unboundedOpt = baselineOpt;
  unboundedOpt.conservativeChargeBounding = false;
  auto combinedBoundingSubcyclingOpt = baselineOpt;
  combinedBoundingSubcyclingOpt.chargeSubcycles = 4;

  const auto baselineLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, baselineOpt);
  const auto baselineHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, baselineOpt);
  const auto postLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, postRefreshOpt);
  const auto postHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, postRefreshOpt);
  const auto conductivityLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             conductivityClosureOpt);
  const auto conductivityHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             conductivityClosureOpt);
  const auto interfaceChargeLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             interfaceChargeTransportOpt);
  const auto interfaceChargeHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             interfaceChargeTransportOpt);
  const auto subcycledHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, subcycledOpt);
  const auto unboundedHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, unboundedOpt);
  const auto combinedBoundingSubcyclingHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             combinedBoundingSubcyclingOpt);

  checkSmokeNumerics(baselineLow, "baseline low-CaE");
  checkSmokeNumerics(baselineHigh, "baseline high-CaE");
  checkSmokeNumerics(postLow, "post-charge refresh low-CaE");
  checkSmokeNumerics(postHigh, "post-charge refresh high-CaE");
  checkSmokeNumerics(conductivityLow, "conductivity closure low-CaE");
  checkSmokeNumerics(conductivityHigh, "conductivity closure high-CaE");
  checkSmokeNumerics(interfaceChargeLow, "interface charge transport low-CaE");
  checkSmokeNumerics(interfaceChargeHigh, "interface charge transport high-CaE");
  checkSmokeNumerics(subcycledHigh, "charge subcycled high-CaE");
  checkSmokeNumerics(unboundedHigh, "unbounded charge high-CaE");
  checkSmokeNumerics(combinedBoundingSubcyclingHigh,
                     "combined bounding/subcycling high-CaE");

  std::ofstream interfaceCsv(
      "benchmark_logs/candido_interface_charge_transport_diagnostic3d.csv");
  interfaceCsv << "case,low_ca_e,high_ca_e,min_area_di2,"
                  "baseline_low_developed_samples,"
                  "baseline_high_developed_samples,"
                  "candidate_low_developed_samples,"
                  "candidate_high_developed_samples,"
                  "baseline_axial_alpha05_current_ratio,"
                  "candidate_axial_alpha05_current_ratio,"
                  "candidate_charge_ratio,candidate_velocity_ratio,"
                  "candidate_low_alpha05_current,"
                  "candidate_high_alpha05_current,"
                  "candidate_low_mean_abs_charge,"
                  "candidate_high_mean_abs_charge,"
                  "candidate_low_mean_abs_uy,candidate_high_mean_abs_uy,"
                  "candidate_low_clamp_l1,candidate_high_clamp_l1,"
                  "candidate_low_redistribution_deficit_l1,"
                  "candidate_high_redistribution_deficit_l1,"
                  "candidate_low_weighted_cells,candidate_high_weighted_cells,"
                  "candidate_low_weighted_capacity,"
                  "candidate_high_weighted_capacity,"
                  "candidate_low_relative_charge_budget_residual,"
                  "candidate_high_relative_charge_budget_residual,"
                  "candidate_low_alpha_mass_drift,"
                  "candidate_high_alpha_mass_drift,"
                  "candidate_low_max_div,candidate_high_max_div,"
                  "candidate_low_morphology_error_percent,"
                  "candidate_high_max_radial_asymmetry,status\n";
  writeInterfaceChargeTransportRow(interfaceCsv, baselineLow, baselineHigh,
                                   interfaceChargeLow, interfaceChargeHigh,
                                   minAreaDi2);
  interfaceCsv.flush();

  std::ofstream postCsv(
      "benchmark_logs/candido_post_charge_potential_refresh_diagnostic3d.csv");
  postCsv << "case,low_ca_e,high_ca_e,min_area_di2,"
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
             "candidate_low_post_charge_potential_residual,"
             "candidate_high_post_charge_potential_residual,"
             "candidate_low_post_charge_relative_gauss_residual,"
             "candidate_high_post_charge_relative_gauss_residual,"
             "candidate_low_alpha_mass_drift,candidate_high_alpha_mass_drift,"
             "candidate_low_max_div,candidate_high_max_div,"
             "candidate_low_morphology_error_percent,"
             "candidate_high_max_radial_asymmetry,status\n";
  writePostChargePotentialRefreshRow(postCsv, baselineLow, baselineHigh, postLow,
                                     postHigh, minAreaDi2);
  postCsv.flush();

  std::ofstream conductivityCsv(
      "benchmark_logs/candido_conductivity_potential_charge_closure3d.csv");
  conductivityCsv << "case,low_ca_e,high_ca_e,min_area_di2,"
                     "baseline_low_developed_samples,"
                     "baseline_high_developed_samples,"
                     "candidate_low_developed_samples,"
                     "candidate_high_developed_samples,"
                     "baseline_axial_alpha05_current_ratio,"
                     "candidate_axial_alpha05_current_ratio,"
                     "candidate_charge_ratio,candidate_velocity_ratio,"
                     "candidate_electric_source_ratio,"
                     "candidate_low_alpha05_current,"
                     "candidate_high_alpha05_current,"
                     "candidate_low_mean_abs_charge,"
                     "candidate_high_mean_abs_charge,"
                     "candidate_low_mean_abs_uy,candidate_high_mean_abs_uy,"
                     "candidate_low_mean_abs_electric_source,"
                     "candidate_high_mean_abs_electric_source,"
                     "candidate_low_conductivity_potential_residual,"
                     "candidate_high_conductivity_potential_residual,"
                     "candidate_low_closure_clamp_l1,"
                     "candidate_high_closure_clamp_l1,"
                     "candidate_low_relative_charge_budget_residual,"
                     "candidate_high_relative_charge_budget_residual,"
                     "candidate_low_alpha_mass_drift,"
                     "candidate_high_alpha_mass_drift,"
                     "candidate_low_max_div,candidate_high_max_div,"
                     "candidate_low_morphology_error_percent,"
                     "candidate_high_max_radial_asymmetry,status\n";
  writeConductivityPotentialClosureRow(conductivityCsv, baselineLow,
                                       baselineHigh, conductivityLow,
                                       conductivityHigh, minAreaDi2);
  conductivityCsv.flush();

  std::ofstream subcyclingCsv(
      "benchmark_logs/candido_charge_subcycling_diagnostic3d.csv");
  subcyclingCsv << "target_ca_e,subcycles,"
                   "baseline_relative_charge_budget_residual,"
                   "subcycled_relative_charge_budget_residual,residual_ratio,"
                   "baseline_clamp_correction_l1,subcycled_clamp_correction_l1,"
                   "clamp_correction_ratio,baseline_max_clamped_cells,"
                   "subcycled_max_clamped_cells,baseline_max_unclamped_abs_charge,"
                   "subcycled_max_unclamped_abs_charge,"
                   "baseline_max_convective_current,"
                   "subcycled_max_convective_current,current_ratio,"
                   "baseline_max_velocity,subcycled_max_velocity,status\n";
  writeChargeSubcyclingDiagnosticRow(subcyclingCsv, baselineHigh, subcycledHigh,
                                     subcycledOpt.chargeSubcycles);
  subcyclingCsv.flush();

  std::ofstream conservativeBoundingCsv(
      "benchmark_logs/candido_charge_conservative_bounding_diagnostic3d.csv");
  conservativeBoundingCsv
      << "target_ca_e,baseline_relative_charge_budget_residual,"
         "bounded_relative_charge_budget_residual,residual_ratio,"
         "baseline_clamp_correction_l1,bounded_clamp_correction_l1,"
         "clamp_correction_ratio,bounded_max_redistribution_residual,"
         "baseline_max_clamped_cells,bounded_max_clamped_cells,"
         "baseline_max_unclamped_abs_charge,bounded_max_unclamped_abs_charge,"
         "baseline_max_convective_current,bounded_max_convective_current,"
         "current_ratio,baseline_max_velocity,bounded_max_velocity,status\n";
  writeChargeConservativeBoundingDiagnosticRow(conservativeBoundingCsv,
                                               unboundedHigh, baselineHigh);
  conservativeBoundingCsv.flush();

  std::ofstream combinedCsv(
      "benchmark_logs/candido_charge_combined_bounding_subcycling3d.csv");
  combinedCsv
      << "target_ca_e,baseline_relative_charge_budget_residual,"
         "bounded_relative_charge_budget_residual,residual_ratio,"
         "baseline_clamp_correction_l1,bounded_clamp_correction_l1,"
         "clamp_correction_ratio,bounded_max_redistribution_residual,"
         "baseline_max_clamped_cells,bounded_max_clamped_cells,"
         "baseline_max_unclamped_abs_charge,bounded_max_unclamped_abs_charge,"
         "baseline_max_convective_current,bounded_max_convective_current,"
         "current_ratio,baseline_max_velocity,bounded_max_velocity,status\n";
  writeChargeConservativeBoundingDiagnosticRow(
      combinedCsv, unboundedHigh, combinedBoundingSubcyclingHigh);
  combinedCsv.flush();

  std::ofstream chargeLimitCsv(
      "benchmark_logs/candido_charge_limit_sensitivity3d.csv");
  chargeLimitCsv << "target_ca_e,charge_limit_base,"
                    "relative_charge_budget_residual,"
                    "cumulative_charge_clamp_correction_l1,"
                    "max_charge_clamped_cells,max_unclamped_abs_charge,"
                    "max_charge,min_charge,max_convective_current,max_velocity,"
                    "alpha_mass_drift,max_div,status\n";
  for (const double chargeLimitBase : {5.0, 50.0, 500.0}) {
    auto limitOpt = baselineOpt;
    limitOpt.chargeLimitBase = chargeLimitBase;
    const auto limitRun =
        electrospray::runCandidoConeJetSmoke3D(0.42, setup, limitOpt);
    checkSmokeNumerics(limitRun, "charge-limit sensitivity high-CaE");
    writeChargeLimitSensitivityRow(chargeLimitCsv, chargeLimitBase, limitRun);
  }
  chargeLimitCsv.flush();

  std::ofstream chargeReferenceGapCsv(
      "benchmark_logs/candido_charge_model_reference_gap3d.csv");
  chargeReferenceGapCsv
      << "item,missing_requirement,evidence,status\n"
      << "bulk_charge_conservation,"
         "paper-faithful conservative bounded charge transport over the full "
         "developed cone-jet window,"
         "short conservative-bounding fixture closes charge budget locally but "
         "does not calibrate current,"
         "DOCUMENTED_NOT_RESOLVED\n"
      << "current_boundary_treatment,"
         "published electrode/nozzle/outlet surface-current boundary closure "
         "for the Candido configuration,"
         "boundary-switch and implicit-filter diagnostics do not reduce the "
         "developed alpha05 current sensitivity,"
         "DOCUMENTED_NOT_RESOLVED\n"
      << "voltage_sensitivity,"
         "external Fig. 8(b)-consistent q_e U dot n current validation with "
         "nonzero developed alpha05 liquid-jet observable,"
         "paper-current standalone rows are DOWNGRADED or zero-current BLOCKED,"
         "DOCUMENTED_NOT_RESOLVED\n";
  chargeReferenceGapCsv.flush();

  check(postCsv.good(), "post-charge potential-refresh CSV written");
  check(conductivityCsv.good(), "conductivity-potential closure CSV written");
  check(interfaceCsv.good(), "interface charge-transport CSV written");
  check(subcyclingCsv.good(), "charge-subcycling CSV written");
  check(conservativeBoundingCsv.good(), "conservative charge-bounding CSV written");
  check(combinedCsv.good(), "combined bounding/subcycling CSV written");
  check(chargeLimitCsv.good(), "charge-limit sensitivity CSV written");
  check(chargeReferenceGapCsv.good(), "charge reference-gap CSV written");
  return 0;
}
