#include "TestUtil.hpp"
#include "electrospray/CandidoTaylorConeJet3D.hpp"

#include <algorithm>
#include <filesystem>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

static void writeRow(std::ofstream& csv, const std::string& name,
                     const electrospray::CandidoConeJetSmokeReport3D& r) {
  csv << name << "," << r.cells << "," << r.faces << "," << r.steps << ","
      << r.targetCaE << "," << r.voltage << "," << r.computedCaE << ","
      << r.electricWeber << "," << r.hydrodynamicTimeScale << ","
      << r.inletVelocity << "," << r.dt << "," << r.initialMass << ","
      << r.finalMass << "," << r.alphaMassDrift << ","
      << r.cumulativeBoundaryLiquidFlux << "," << r.cumulativeBoundaryLiquidInflow << ","
      << r.cumulativeBoundaryLiquidOutflow << "," << r.massBudgetExpectedFinal << ","
      << r.massBudgetResidual << "," << r.relativeMassBudgetResidual << ","
      << r.minAlpha << "," << r.maxAlpha << "," << r.initialTipY << "," << r.finalTipY << ","
      << r.tipDisplacement << "," << r.finalCentroidY << "," << r.maxDiv << ","
      << r.maxPotentialResidual << "," << r.maxElectricForce << ","
      << r.maxCsfForce << "," << r.maxCurvature << ","
      << r.curvatureFallbackFraction << "," << r.minCharge << ","
      << r.maxCharge << "," << r.maxConductiveCurrent << ","
      << r.maxConvectiveCurrent << "," << r.finalRadialAsymmetry << ","
      << r.finalMidplaneJetRadius << "," << r.maxVelocity << "\n";
  csv.flush();
}

static void writeHistoryRows(std::ofstream& csv, const std::string& name,
                             const electrospray::CandidoConeJetSmokeReport3D& r) {
  for (const auto& h : r.history) {
    csv << name << "," << h.step << "," << h.time << "," << h.mass << ","
        << h.minAlpha << "," << h.maxAlpha << "," << h.tipY << ","
        << h.centroidY << "," << h.radialAsymmetry << "," << h.maxDiv << ","
        << h.potentialResidual << "," << h.electricForce << "," << h.csfForce << ","
        << h.curvature << "," << h.conductiveCurrent << "," << h.convectiveCurrent << ","
        << h.liquidConvectiveCurrent << "," << h.alpha05ConvectiveCurrent << ","
        << h.midplaneLiquidAreaDi2 << "," << h.midplaneAlpha05AreaDi2 << ","
        << h.developedJetYOverDi << "," << h.developedJetAlpha05AreaDi2 << ","
        << h.developedJetConvectiveCurrent << ","
        << h.developedJetLiquidConvectiveCurrent << ","
        << h.developedJetAlpha05ConvectiveCurrent << ","
        << h.developedJetTotalCurrent << ","
        << h.developedJetLiquidTotalCurrent << ","
        << h.developedJetAlpha05TotalCurrent << ","
        << h.developedJetAlpha05ConductiveCurrent << ","
        << h.developedJetMeanAlpha05Charge << ","
        << h.developedJetMeanAlpha05AbsCharge << ","
        << h.developedJetMeanAlpha05Uy << ","
        << h.developedJetMeanAlpha05AbsUy << ","
        << h.developedJetMeanAlpha05AbsElectricMomentumSourceY << ","
        << h.developedJetMeanAlpha05AbsSurfaceMomentumSourceY << ","
        << h.developedJetMeanAlpha05AbsMomentumSourceY << ","
        << h.developedJetMeanAlpha05AbsMomentumAccelerationY << ","
        << h.developedJetAlpha05CurrentShapeFactor << ","
        << h.totalCurrent << "," << h.maxVelocity << "," << h.waveYOverDi << ","
        << h.waveAsymmetry << ","
        << h.morphologyVolumeDi3 << "," << h.connectedMorphologyVolumeDi3 << ","
        << h.alpha05SilhouetteVolumeDi3 << ","
        << h.rayAlpha05SilhouetteVolumeDi3 << ","
        << h.allLiquidRayAlpha05SilhouetteVolumeDi3 << ","
        << h.rayAlpha05CellBoundarySilhouetteVolumeDi3 << ","
        << h.linearRayAlpha05SilhouetteVolumeDi3 << ","
        << h.plicContourSilhouetteVolumeDi3 << ","
        << h.plicPolygonSilhouetteVolumeDi3 << ","
        << h.plicSectorMedianSilhouetteVolumeDi3 << ","
        << h.plicRayPlaneSilhouetteVolumeDi3 << ","
        << h.plicRayPlaneQ25SilhouetteVolumeDi3 << ","
        << h.plicFirstExitSilhouetteVolumeDi3 << ","
        << h.poissonFaceConvectiveCurrent << ","
        << h.poissonFaceConductiveCurrent << ","
        << h.poissonFaceTotalCurrent << ","
        << h.poissonFaceAlpha05ConvectiveCurrent << ","
        << h.poissonFaceAlpha05ConductiveCurrent << ","
        << h.poissonFaceAlpha05TotalCurrent << ","
        << h.poissonFaceDevelopedYOverDi << ","
        << h.poissonFaceDevelopedAlpha05AreaDi2 << ","
        << h.poissonFaceDevelopedAlpha05ConvectiveCurrent << ","
        << h.poissonFaceDevelopedAlpha05ConductiveCurrent << ","
        << h.poissonFaceDevelopedAlpha05TotalCurrent << ","
        << h.poissonFaceDevelopedAlpha05MeanAbsUpwindCharge << ","
        << h.poissonFaceDevelopedAlpha05MeanAbsFaceFlux << ","
        << h.poissonFaceDevelopedAlpha05MeanAbsConvectiveFlux << ","
        << h.poissonFaceDevelopedAlpha05MaxAbsUpwindCharge << ","
        << h.poissonFaceDevelopedAlpha05MaxAbsFaceFlux << ","
        << h.rawVelocityFaceDevelopedAlpha05ConvectiveCurrent << ","
        << h.rawVelocityFaceDevelopedAlpha05MeanAbsUpwindCharge << ","
        << h.rawVelocityFaceDevelopedAlpha05MeanAbsFaceFlux << ","
        << h.rawVelocityFaceDevelopedAlpha05MeanAbsConvectiveFlux << ","
        << h.rawVelocityFaceDevelopedAlpha05MaxAbsUpwindCharge << ","
        << h.rawVelocityFaceDevelopedAlpha05MaxAbsFaceFlux << "\n";
  }
  csv.flush();
}

static void writeMorphologyObservableAuditRows(
    std::ofstream& csv, const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  for (const auto& h : r.history) {
    const double physicalTimeMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
    const double connectedDelta =
        h.connectedMorphologyVolumeDi3 - h.morphologyVolumeDi3;
    const double silhouetteDelta =
        h.alpha05SilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double raySilhouetteDelta =
        h.rayAlpha05SilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double allLiquidRaySilhouetteDelta =
        h.allLiquidRayAlpha05SilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double rayCellBoundarySilhouetteDelta =
        h.rayAlpha05CellBoundarySilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double linearRaySilhouetteDelta =
        h.linearRayAlpha05SilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double plicContourDelta =
        h.plicContourSilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double plicPolygonDelta =
        h.plicPolygonSilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double plicSectorMedianDelta =
        h.plicSectorMedianSilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double plicRayPlaneDelta =
        h.plicRayPlaneSilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double plicRayPlaneQ25Delta =
        h.plicRayPlaneQ25SilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const double plicFirstExitDelta =
        h.plicFirstExitSilhouetteVolumeDi3 - h.morphologyVolumeDi3;
    const std::string status =
        (std::isfinite(h.connectedMorphologyVolumeDi3) &&
         std::isfinite(h.alpha05SilhouetteVolumeDi3) &&
         std::isfinite(h.rayAlpha05SilhouetteVolumeDi3) &&
         std::isfinite(h.allLiquidRayAlpha05SilhouetteVolumeDi3) &&
         std::isfinite(h.rayAlpha05CellBoundarySilhouetteVolumeDi3) &&
         std::isfinite(h.linearRayAlpha05SilhouetteVolumeDi3) &&
         std::isfinite(h.plicContourSilhouetteVolumeDi3) &&
         std::isfinite(h.plicPolygonSilhouetteVolumeDi3) &&
         std::isfinite(h.plicSectorMedianSilhouetteVolumeDi3) &&
         std::isfinite(h.plicRayPlaneSilhouetteVolumeDi3) &&
         std::isfinite(h.plicRayPlaneQ25SilhouetteVolumeDi3) &&
         std::isfinite(h.plicFirstExitSilhouetteVolumeDi3))
            ? "DIAGNOSTIC_ONLY_NOT_PAPER_VALIDATED"
            : "DOWNGRADED_NONFINITE_OBSERVABLE";
    csv << name << "," << h.step << "," << physicalTimeMs << ","
        << h.morphologyVolumeDi3 << "," << h.connectedMorphologyVolumeDi3 << ","
        << h.alpha05SilhouetteVolumeDi3 << "," << h.rayAlpha05SilhouetteVolumeDi3
        << "," << h.allLiquidRayAlpha05SilhouetteVolumeDi3 << ","
        << h.rayAlpha05CellBoundarySilhouetteVolumeDi3 << ","
        << h.linearRayAlpha05SilhouetteVolumeDi3 << ","
        << connectedDelta << "," << silhouetteDelta << ","
        << raySilhouetteDelta << "," << allLiquidRaySilhouetteDelta << ","
        << rayCellBoundarySilhouetteDelta << ","
        << linearRaySilhouetteDelta << ","
        << h.plicContourSilhouetteVolumeDi3 << ","
        << plicContourDelta << "," << h.plicPolygonSilhouetteVolumeDi3 << ","
        << plicPolygonDelta << "," << h.plicSectorMedianSilhouetteVolumeDi3 << ","
        << plicSectorMedianDelta << "," << h.plicRayPlaneSilhouetteVolumeDi3 << ","
        << plicRayPlaneDelta << "," << h.plicRayPlaneQ25SilhouetteVolumeDi3 << ","
        << plicRayPlaneQ25Delta << "," << h.plicFirstExitSilhouetteVolumeDi3 << ","
        << plicFirstExitDelta << "," << status << "\n";
  }
  csv.flush();
}

static void writePhysicalTimeRows(std::ofstream& csv, const std::string& name,
                                  const electrospray::CandidoConeJetSmokeReport3D& r) {
  for (const auto& h : r.history) {
    const double physicalTime = h.time * r.hydrodynamicTimeScale;
    csv << name << "," << h.step << "," << h.time << "," << physicalTime << ","
        << physicalTime * 1.0e3 << "," << r.hydrodynamicTimeScale << ","
        << h.tipY << "," << h.centroidY << "," << h.radialAsymmetry << ","
        << h.maxVelocity << "\n";
  }
  csv.flush();
}

static double candidoGananCalvoCurrentScale(
    const electrospray::CandidoTaylorConeJetSetup& setup) {
  return std::sqrt(setup.surfaceTension * setup.liquidConductivity *
                   setup.validationFlowRate);
}

static double candidoPoissonChargeScale(
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  return electrospray::candidoPoissonChargeScale(setup, r.voltage);
}

static double candidoPoissonCurrentScale(
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  return candidoPoissonChargeScale(setup, r) /
         std::max(r.hydrodynamicTimeScale, 1e-300);
}

static void writeCurrentScalingValidationRow(
    std::ofstream& csv, const std::string& name,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double reference = candidoGananCalvoCurrentScale(setup);
  const double conductiveRatio = r.maxConductiveCurrent / std::max(reference, 1e-30);
  const double convectiveRatio = r.maxConvectiveCurrent / std::max(reference, 1e-30);
  const bool convectiveOrderOfMagnitude = convectiveRatio >= 0.1 && convectiveRatio <= 10.0;
  const std::string status =
      convectiveOrderOfMagnitude ? "ORDER_OF_MAGNITUDE" : "DOWNGRADED_OUT_OF_SCALE";
  csv << name << "," << r.targetCaE << "," << r.voltage << ","
      << setup.validationFlowRate << "," << setup.surfaceTension << ","
      << setup.liquidConductivity << "," << reference << ","
      << r.maxConductiveCurrent << "," << conductiveRatio << ","
      << r.maxConvectiveCurrent << "," << convectiveRatio << ","
      << "I_ref=sqrt(gamma*K*Q); convective_current_is_paper_comparable;"
         "conductive_row_is_max_face_flux_diagnostic_only"
      << "," << status << "\n";
  csv.flush();
}

enum class CandidoCurrentSensitivityObservable {
  Convective,
  Total,
  LiquidConvective,
  Alpha05Convective,
  PoissonFaceTotal,
  PoissonFaceAlpha05Convective,
  PoissonFaceAlpha05Total
};

static double candidoHistoryCurrentValue(
    const electrospray::CandidoConeJetHistorySample3D& h,
    CandidoCurrentSensitivityObservable observable) {
  switch (observable) {
    case CandidoCurrentSensitivityObservable::Total:
      return h.totalCurrent;
    case CandidoCurrentSensitivityObservable::LiquidConvective:
      return h.liquidConvectiveCurrent;
    case CandidoCurrentSensitivityObservable::Alpha05Convective:
      return h.alpha05ConvectiveCurrent;
    case CandidoCurrentSensitivityObservable::PoissonFaceTotal:
      return h.poissonFaceTotalCurrent;
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective:
      return h.poissonFaceAlpha05ConvectiveCurrent;
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total:
      return h.poissonFaceAlpha05TotalCurrent;
    case CandidoCurrentSensitivityObservable::Convective:
    default:
      return h.convectiveCurrent;
  }
}

static double candidoHistoryAxialDevelopedCurrentValue(
    const electrospray::CandidoConeJetHistorySample3D& h,
    CandidoCurrentSensitivityObservable observable) {
  switch (observable) {
    case CandidoCurrentSensitivityObservable::Total:
      return h.developedJetTotalCurrent;
    case CandidoCurrentSensitivityObservable::LiquidConvective:
      return h.developedJetLiquidConvectiveCurrent;
    case CandidoCurrentSensitivityObservable::Alpha05Convective:
      return h.developedJetAlpha05ConvectiveCurrent;
    case CandidoCurrentSensitivityObservable::PoissonFaceTotal:
      return h.poissonFaceTotalCurrent;
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective:
      return h.poissonFaceDevelopedAlpha05ConvectiveCurrent;
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total:
      return h.poissonFaceDevelopedAlpha05TotalCurrent;
    case CandidoCurrentSensitivityObservable::Convective:
    default:
      return h.developedJetConvectiveCurrent;
  }
}

static double candidoHistoryAxialDevelopedAreaValue(
    const electrospray::CandidoConeJetHistorySample3D& h,
    CandidoCurrentSensitivityObservable observable) {
  switch (observable) {
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective:
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total:
      return h.poissonFaceDevelopedAlpha05AreaDi2;
    case CandidoCurrentSensitivityObservable::PoissonFaceTotal:
    case CandidoCurrentSensitivityObservable::Total:
    case CandidoCurrentSensitivityObservable::LiquidConvective:
    case CandidoCurrentSensitivityObservable::Alpha05Convective:
    case CandidoCurrentSensitivityObservable::Convective:
    default:
      return h.developedJetAlpha05AreaDi2;
  }
}

static double candidoHistoryAxialDevelopedYValue(
    const electrospray::CandidoConeJetHistorySample3D& h,
    CandidoCurrentSensitivityObservable observable) {
  switch (observable) {
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective:
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total:
      return h.poissonFaceDevelopedYOverDi;
    case CandidoCurrentSensitivityObservable::PoissonFaceTotal:
    case CandidoCurrentSensitivityObservable::Total:
    case CandidoCurrentSensitivityObservable::LiquidConvective:
    case CandidoCurrentSensitivityObservable::Alpha05Convective:
    case CandidoCurrentSensitivityObservable::Convective:
    default:
      return h.developedJetYOverDi;
  }
}

static const char* candidoCurrentObservableSource(
    CandidoCurrentSensitivityObservable observable) {
  switch (observable) {
    case CandidoCurrentSensitivityObservable::Total:
      return "Candido_Fig8b_text_average_current_not_influenced_by_voltage;"
             "total_current=rho_e_u_plus_sigma_E_midplane";
    case CandidoCurrentSensitivityObservable::LiquidConvective:
      return "Candido_Fig8b_current_ie=int_S_qe_U_dot_n_dS;"
             "alpha_weighted_liquid_jet_cross_section";
    case CandidoCurrentSensitivityObservable::Alpha05Convective:
      return "Candido_Fig8b_current_ie=int_S_qe_U_dot_n_dS;"
             "alpha05_liquid_jet_cross_section";
    case CandidoCurrentSensitivityObservable::PoissonFaceTotal:
      return "face_consistent_total_current=rho_e_phiFlux_plus_sigma_gradphi_flux;"
             "uses_Poisson_snGrad_flux";
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective:
      return "face_consistent_alpha05_convective_current=rho_e_phiFlux;"
             "uses_projected_Rhie_Chow_Poisson_face_flux";
    case CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total:
      return "face_consistent_alpha05_total_current=rho_e_phiFlux_plus_sigma_gradphi_flux;"
             "uses_Poisson_snGrad_flux";
    case CandidoCurrentSensitivityObservable::Convective:
    default:
      return "Candido_Fig8b_text_average_current_not_influenced_by_voltage";
  }
}

static double candidoMeanTailCurrentForObservable(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    CandidoCurrentSensitivityObservable observable =
        CandidoCurrentSensitivityObservable::Convective) {
  double meanTail = 0.0;
  int tailCount = 0;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    meanTail += std::abs(candidoHistoryCurrentValue(r.history[i], observable));
    ++tailCount;
  }
  return meanTail / std::max(tailCount, 1);
}

static void writeCurrentVoltageSensitivityRowForObservable(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    CandidoCurrentSensitivityObservable observable) {
  auto stats = [observable](const electrospray::CandidoConeJetSmokeReport3D& r) {
    double maxCurrent = 0.0;
    double meanTail = 0.0;
    double meanAll = 0.0;
    int tailCount = 0;
    int allCount = 0;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = 0; i < r.history.size(); ++i) {
      const double value =
          std::abs(candidoHistoryCurrentValue(r.history[i], observable));
      maxCurrent = std::max(maxCurrent, value);
      meanAll += value;
      ++allCount;
      if (i >= tailStart) {
        meanTail += value;
        ++tailCount;
      }
    }
    meanAll /= std::max(allCount, 1);
    meanTail /= std::max(tailCount, 1);
    return std::tuple<double, double, double>{maxCurrent, meanAll, meanTail};
  };
  const auto [lowMax, lowMeanAll, lowMeanTail] = stats(low);
  const auto [highMax, highMeanAll, highMeanTail] = stats(high);
  const double tailMeanRatio = highMeanTail / std::max(lowMeanTail, 1e-30);
  const double peakRatio = highMax / std::max(lowMax, 1e-30);
  const bool weakVoltageSensitivity = tailMeanRatio <= 2.0;
  const std::string status =
      weakVoltageSensitivity ? "APPROXIMATE_WEAK_AVERAGE_VOLTAGE_SENSITIVITY"
                             : "DOWNGRADED_AVERAGE_CURRENT_TOO_VOLTAGE_SENSITIVE";
  csv << low.targetCaE << "," << high.targetCaE << ","
      << lowMax << "," << highMax << "," << peakRatio << ","
      << lowMeanAll << "," << highMeanAll << ","
      << lowMeanTail << "," << highMeanTail << "," << tailMeanRatio << ","
      << candidoCurrentObservableSource(observable) << ","
      << status << "\n";
}

static void writeCurrentVoltageSensitivityRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    bool useTotalCurrent = false) {
  writeCurrentVoltageSensitivityRowForObservable(
      csv, low, high,
      useTotalCurrent ? CandidoCurrentSensitivityObservable::Total
                      : CandidoCurrentSensitivityObservable::Convective);
}

static void writeMidplaneCurrentReachDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  struct Stats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanDevelopedAlpha05Current = 0.0;
    double peakDevelopedAlpha05Current = 0.0;
    double maxAlpha05AreaDi2 = 0.0;
    double maxTipY = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    Stats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      ++s.tailSamples;
      s.maxAlpha05AreaDi2 = std::max(s.maxAlpha05AreaDi2, h.midplaneAlpha05AreaDi2);
      s.maxTipY = std::max(s.maxTipY, h.tipY);
      if (h.midplaneAlpha05AreaDi2 < minAreaDi2) continue;
      const double current = std::abs(h.alpha05ConvectiveCurrent);
      s.meanDevelopedAlpha05Current += current;
      s.peakDevelopedAlpha05Current = std::max(s.peakDevelopedAlpha05Current, current);
      ++s.developedSamples;
    }
    if (s.developedSamples > 0) {
      s.meanDevelopedAlpha05Current /= s.developedSamples;
    }
    return s;
  };
  const Stats lowStats = stats(low);
  const Stats highStats = stats(high);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double meanRatio =
      comparable ? highStats.meanDevelopedAlpha05Current /
                       std::max(lowStats.meanDevelopedAlpha05Current, 1e-30)
                 : std::numeric_limits<double>::infinity();
  const double peakRatio =
      comparable ? highStats.peakDevelopedAlpha05Current /
                       std::max(lowStats.peakDevelopedAlpha05Current, 1e-30)
                 : std::numeric_limits<double>::infinity();
  std::string status;
  if (low.alphaMassDrift > 1e-3 || high.alphaMassDrift > 1e-3) {
    status = "DOWNGRADED_EXTENDED_WINDOW_MASS_FAILURE";
  } else if (!comparable) {
    status = "BLOCKED_UNDEVELOPED_FIXED_MIDPLANE_CURRENT";
  } else if (meanRatio <= 2.0) {
    status = "APPROXIMATE_FIXED_MIDPLANE_CURRENT_WEAK_SENSITIVITY";
  } else {
    status = "DOWNGRADED_FIXED_MIDPLANE_CURRENT_TOO_VOLTAGE_SENSITIVE";
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << low.steps << "," << high.steps << "," << minAreaDi2 << ","
      << lowStats.tailSamples << "," << highStats.tailSamples << ","
      << lowStats.developedSamples << "," << highStats.developedSamples << ","
      << lowStats.maxAlpha05AreaDi2 << "," << highStats.maxAlpha05AreaDi2 << ","
      << lowStats.maxTipY << "," << highStats.maxTipY << ","
      << lowStats.meanDevelopedAlpha05Current << ","
      << highStats.meanDevelopedAlpha05Current << "," << meanRatio << ","
      << lowStats.peakDevelopedAlpha05Current << ","
      << highStats.peakDevelopedAlpha05Current << "," << peakRatio << ","
      << low.alphaMassDrift << "," << high.alphaMassDrift << ","
      << low.maxDiv << "," << high.maxDiv << "," << status << "\n";
  csv.flush();
}

static void writeReducedCollectorCurrentFixtureRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  struct Stats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanDevelopedCurrent = 0.0;
    double peakDevelopedCurrent = 0.0;
    double maxAreaDi2 = 0.0;
    double maxTipY = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    Stats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      ++s.tailSamples;
      s.maxAreaDi2 = std::max(s.maxAreaDi2, h.midplaneAlpha05AreaDi2);
      s.maxTipY = std::max(s.maxTipY, h.tipY);
      if (h.midplaneAlpha05AreaDi2 < minAreaDi2) continue;
      const double current = std::abs(h.alpha05ConvectiveCurrent);
      s.meanDevelopedCurrent += current;
      s.peakDevelopedCurrent = std::max(s.peakDevelopedCurrent, current);
      ++s.developedSamples;
    }
    if (s.developedSamples > 0) {
      s.meanDevelopedCurrent /= s.developedSamples;
    }
    return s;
  };
  const Stats lowStats = stats(low);
  const Stats highStats = stats(high);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double meanRatio =
      comparable ? highStats.meanDevelopedCurrent /
                       std::max(lowStats.meanDevelopedCurrent, 1e-30)
                 : std::numeric_limits<double>::infinity();
  const double peakRatio =
      comparable ? highStats.peakDevelopedCurrent /
                       std::max(lowStats.peakDevelopedCurrent, 1e-30)
                 : std::numeric_limits<double>::infinity();
  std::string status;
  if (low.alphaMassDrift > 1e-3 || high.alphaMassDrift > 1e-3 ||
      low.maxDiv > 1e-7 || high.maxDiv > 1e-7) {
    status = "DOWNGRADED_REDUCED_DISTANCE_NUMERICAL_FAILURE";
  } else if (!comparable) {
    status = "BLOCKED_REDUCED_DISTANCE_UNDEVELOPED_FIXED_PLANE";
  } else if (meanRatio <= 2.0) {
    status = "APPROXIMATE_REDUCED_DISTANCE_WEAK_SENSITIVITY_NOT_PAPER_GEOMETRY";
  } else {
    status = "DOWNGRADED_REDUCED_DISTANCE_TOO_VOLTAGE_SENSITIVE";
  }
  const double collectorOverDi = setup.collectorDistance /
                                 std::max(setup.innerDiameter, 1e-30);
  csv << caseName << "," << setup.collectorDistance << ","
      << collectorOverDi << "," << 0.5 * collectorOverDi << ","
      << low.targetCaE << "," << high.targetCaE << ","
      << low.steps << "," << high.steps << "," << minAreaDi2 << ","
      << lowStats.tailSamples << "," << highStats.tailSamples << ","
      << lowStats.developedSamples << "," << highStats.developedSamples << ","
      << lowStats.maxAreaDi2 << "," << highStats.maxAreaDi2 << ","
      << lowStats.maxTipY << "," << highStats.maxTipY << ","
      << lowStats.meanDevelopedCurrent << "," << highStats.meanDevelopedCurrent
      << "," << meanRatio << "," << lowStats.peakDevelopedCurrent << ","
      << highStats.peakDevelopedCurrent << "," << peakRatio << ","
      << low.alphaMassDrift << "," << high.alphaMassDrift << ","
      << low.maxDiv << "," << high.maxDiv << "," << status << "\n";
  csv.flush();
}

static int candidoTailDevelopedSamplesAtMidplane(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  int developedSamples = 0;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    if (r.history[i].midplaneAlpha05AreaDi2 >= minAreaDi2) ++developedSamples;
  }
  return developedSamples;
}

static double candidoMaxTailTipY(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  double maxTipY = 0.0;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    maxTipY = std::max(maxTipY, r.history[i].tipY);
  }
  return maxTipY;
}

static double candidoPeakCurrentRatioForObservable(
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    CandidoCurrentSensitivityObservable observable) {
  auto peak = [observable](const electrospray::CandidoConeJetSmokeReport3D& r) {
    double maxCurrent = 0.0;
    for (const auto& h : r.history) {
      maxCurrent =
          std::max(maxCurrent, std::abs(candidoHistoryCurrentValue(h, observable)));
    }
    return maxCurrent;
  };
  return peak(high) / std::max(peak(low), 1e-30);
}

static void writeFig8bCurrentBlockerRow(
    std::ofstream& csv,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoTaylorConeJetSetup& reducedSetup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    const electrospray::CandidoConeJetSmokeReport3D& paperChargeLow,
    const electrospray::CandidoConeJetSmokeReport3D& paperChargeHigh,
    const electrospray::CandidoConeJetSmokeReport3D& inletVelocityLow,
    const electrospray::CandidoConeJetSmokeReport3D& inletVelocityHigh,
    const electrospray::CandidoConeJetSmokeReport3D& reducedLow,
    const electrospray::CandidoConeJetSmokeReport3D& reducedHigh,
    double minAreaDi2) {
  auto tailRatio = [](const electrospray::CandidoConeJetSmokeReport3D& low,
                      const electrospray::CandidoConeJetSmokeReport3D& high) {
    return candidoMeanTailCurrentForObservable(high) /
           std::max(candidoMeanTailCurrentForObservable(low), 1e-30);
  };
  const double chargeTailRatio = tailRatio(paperChargeLow, paperChargeHigh);
  const double inletTailRatio = tailRatio(inletVelocityLow, inletVelocityHigh);
  const double bestTailRatio = std::min(chargeTailRatio, inletTailRatio);
  const double chargePeakRatio = candidoPeakCurrentRatioForObservable(
      paperChargeLow, paperChargeHigh, CandidoCurrentSensitivityObservable::Convective);
  const double inletPeakRatio = candidoPeakCurrentRatioForObservable(
      inletVelocityLow, inletVelocityHigh, CandidoCurrentSensitivityObservable::Convective);
  const double bestPeakRatio = std::min(chargePeakRatio, inletPeakRatio);
  const int paperLowMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(inletVelocityLow, minAreaDi2);
  const int paperHighMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(inletVelocityHigh, minAreaDi2);
  const int reducedLowMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(reducedLow, minAreaDi2);
  const int reducedHighMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(reducedHigh, minAreaDi2);
  const double midplaneYOverDi =
      0.5 * setup.collectorDistance / std::max(setup.innerDiameter, 1e-30);
  const double reducedMidplaneYOverDi =
      0.5 * reducedSetup.collectorDistance / std::max(reducedSetup.innerDiameter, 1e-30);
  const bool paperFixedPlaneDeveloped =
      paperLowMidplaneSamples > 0 && paperHighMidplaneSamples > 0;
  const bool reducedPlaneDeveloped =
      reducedLowMidplaneSamples > 0 && reducedHighMidplaneSamples > 0;
  const bool weakTailSensitivity = bestTailRatio <= 2.0;
  const std::string status =
      !paperFixedPlaneDeveloped && !reducedPlaneDeveloped
          ? "BLOCKED_COARSE_FIXTURE_FIG8B_CURRENT_UNDEVELOPED_FIXED_PLANE"
          : (!weakTailSensitivity
                 ? "DOWNGRADED_FIG8B_CURRENT_TOO_VOLTAGE_SENSITIVE"
                 : "APPROXIMATE_FIG8B_CURRENT_WEAK_SENSITIVITY");
  csv << "coarse_smoke_fig8b_current" << "," << opt.nx << "," << opt.ny << ","
      << opt.nz << "," << minAreaDi2 << "," << midplaneYOverDi << ","
      << reducedMidplaneYOverDi << "," << paperLowMidplaneSamples << ","
      << paperHighMidplaneSamples << "," << reducedLowMidplaneSamples << ","
      << reducedHighMidplaneSamples << "," << candidoMaxTailTipY(inletVelocityLow)
      << "," << candidoMaxTailTipY(inletVelocityHigh) << ","
      << candidoMaxTailTipY(reducedLow) << "," << candidoMaxTailTipY(reducedHigh)
      << "," << chargeTailRatio << "," << inletTailRatio << ","
      << bestTailRatio << "," << chargePeakRatio << "," << inletPeakRatio << ","
      << bestPeakRatio << "," << status << "\n";
  csv.flush();
}

struct CandidoParetoAxialStats {
  int developedSamples = 0;
  double meanArea = 0.0;
  double meanAlpha05Convective = 0.0;
  double meanAlpha05Total = 0.0;
  double meanAbsCharge = 0.0;
  double meanAbsUy = 0.0;
};

static CandidoParetoAxialStats candidoParetoAxialStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  CandidoParetoAxialStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
    ++s.developedSamples;
    s.meanArea += h.developedJetAlpha05AreaDi2;
    s.meanAlpha05Convective += std::abs(h.developedJetAlpha05ConvectiveCurrent);
    s.meanAlpha05Total += std::abs(h.developedJetAlpha05TotalCurrent);
    s.meanAbsCharge += h.developedJetMeanAlpha05AbsCharge;
    s.meanAbsUy += h.developedJetMeanAlpha05AbsUy;
  }
  if (s.developedSamples > 0) {
    const double inv = 1.0 / static_cast<double>(s.developedSamples);
    s.meanArea *= inv;
    s.meanAlpha05Convective *= inv;
    s.meanAlpha05Total *= inv;
    s.meanAbsCharge *= inv;
    s.meanAbsUy *= inv;
  }
  return s;
}

static double candidoTailMeanAlpha05ElectricSource(
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

static double candidoRatio(double highValue, double lowValue) {
  return highValue / std::max(lowValue, 1e-30);
}

static double candidoMaxRadialAsymmetry(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  double value = 0.0;
  for (const auto& h : r.history) value = std::max(value, h.radialAsymmetry);
  return value;
}

static const electrospray::CandidoConeJetHistorySample3D* candidoNearestHistoryAtMs(
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

static double candidoMorphologyErrorAtMs(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double timeMs,
    double referenceVolumeDi3) {
  const auto* h = candidoNearestHistoryAtMs(r, timeMs);
  return 100.0 * (h->morphologyVolumeDi3 - referenceVolumeDi3) /
         std::max(std::abs(referenceVolumeDi3), 1e-30);
}

static double candidoMaxMorphologyError04_07(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double e04 =
      candidoMorphologyErrorAtMs(r, 0.4, 1.2826510303495016);
  const double e07 =
      candidoMorphologyErrorAtMs(r, 0.7, 1.2550259882802302);
  return std::max(std::abs(e04), std::abs(e07));
}

static void writePaperCurrentParetoTradeoffRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const CandidoParetoAxialStats lowStats =
      candidoParetoAxialStats(low, minAreaDi2);
  const CandidoParetoAxialStats highStats =
      candidoParetoAxialStats(high, minAreaDi2);
  const double tailRatio =
      candidoRatio(candidoMeanTailCurrentForObservable(high),
                   candidoMeanTailCurrentForObservable(low));
  const double peakRatio = candidoPeakCurrentRatioForObservable(
      low, high, CandidoCurrentSensitivityObservable::Convective);
  const bool axialComparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double axialConvectiveRatio =
      axialComparable ? candidoRatio(highStats.meanAlpha05Convective,
                                     lowStats.meanAlpha05Convective)
                      : std::numeric_limits<double>::infinity();
  const double axialTotalRatio =
      axialComparable ? candidoRatio(highStats.meanAlpha05Total,
                                     lowStats.meanAlpha05Total)
                      : std::numeric_limits<double>::infinity();
  const double chargeRatio =
      axialComparable ? candidoRatio(highStats.meanAbsCharge,
                                     lowStats.meanAbsCharge)
                      : std::numeric_limits<double>::infinity();
  const double velocityRatio =
      axialComparable ? candidoRatio(highStats.meanAbsUy, lowStats.meanAbsUy)
                      : std::numeric_limits<double>::infinity();
  const int lowMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(low, minAreaDi2);
  const int highMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(high, minAreaDi2);
  const double lowMorphologyError = candidoMaxMorphologyError04_07(low);
  const double highMaxAsymmetry = candidoMaxRadialAsymmetry(high);
  const bool finite =
      std::isfinite(tailRatio) && std::isfinite(peakRatio) &&
      std::isfinite(axialConvectiveRatio) && std::isfinite(axialTotalRatio) &&
      std::isfinite(lowMorphologyError) && std::isfinite(highMaxAsymmetry) &&
      std::isfinite(low.alphaMassDrift) && std::isfinite(high.alphaMassDrift) &&
      std::isfinite(low.maxDiv) && std::isfinite(high.maxDiv);
  std::string status = "DOWNGRADED_NONFINITE_PARETO_DIAGNOSTIC";
  if (finite) {
    const bool numericalQuality =
        low.alphaMassDrift <= 1e-3 && high.alphaMassDrift <= 1e-3 &&
        low.maxDiv <= 1e-7 && high.maxDiv <= 1e-7;
    const bool weakAllPhase = tailRatio <= 2.0 && peakRatio <= 2.0;
    const bool weakAxial = axialComparable && axialConvectiveRatio <= 2.0;
    const bool fixedPlaneDeveloped =
        lowMidplaneSamples > 0 && highMidplaneSamples > 0;
    const bool morphologyOk = lowMorphologyError <= 10.0;
    const bool whipOk = highMaxAsymmetry >= 0.05;
    if (!numericalQuality) {
      status = "DOWNGRADED_NUMERICAL_QUALITY";
    } else if (!weakAllPhase || !weakAxial) {
      status = "DOWNGRADED_CURRENT_RATIO_ABOVE_WEAK_BAR";
    } else if (!fixedPlaneDeveloped) {
      status = "BLOCKED_WEAK_CURRENT_BUT_FIXED_PLANE_UNDEVELOPED";
    } else if (!morphologyOk || !whipOk) {
      status = "DOWNGRADED_WEAK_CURRENT_WITH_MORPHOLOGY_OR_WHIP_TRADEOFF";
    } else {
      status = "APPROXIMATE_PARETO_CANDIDATE_ALL_GUARDS_GREEN";
    }
  }
  const double midplaneYOverDi =
      0.5 * setup.collectorDistance / std::max(setup.innerDiameter, 1e-30);
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << low.steps << "," << high.steps << "," << minAreaDi2 << ","
      << midplaneYOverDi << "," << low.alphaMassDrift << ","
      << high.alphaMassDrift << "," << low.maxDiv << "," << high.maxDiv << ","
      << lowMorphologyError << "," << highMaxAsymmetry << ","
      << tailRatio << "," << peakRatio << "," << lowMidplaneSamples << ","
      << highMidplaneSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << lowStats.meanAlpha05Convective << ","
      << highStats.meanAlpha05Convective << "," << axialConvectiveRatio << ","
      << lowStats.meanAlpha05Total << "," << highStats.meanAlpha05Total << ","
      << axialTotalRatio << "," << lowStats.meanAbsCharge << ","
      << highStats.meanAbsCharge << "," << chargeRatio << ","
      << lowStats.meanAbsUy << "," << highStats.meanAbsUy << ","
      << velocityRatio << "," << status << "\n";
}

static void writeOpenBoundaryCurrentDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const CandidoParetoAxialStats lowStats =
      candidoParetoAxialStats(low, minAreaDi2);
  const CandidoParetoAxialStats highStats =
      candidoParetoAxialStats(high, minAreaDi2);
  const double tailRatio =
      candidoRatio(candidoMeanTailCurrentForObservable(high),
                   candidoMeanTailCurrentForObservable(low));
  const double peakRatio = candidoPeakCurrentRatioForObservable(
      low, high, CandidoCurrentSensitivityObservable::Convective);
  const int lowMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(low, minAreaDi2);
  const int highMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(high, minAreaDi2);
  const bool axialComparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double axialConvectiveRatio =
      axialComparable ? candidoRatio(highStats.meanAlpha05Convective,
                                     lowStats.meanAlpha05Convective)
                      : std::numeric_limits<double>::infinity();
  const double axialTotalRatio =
      axialComparable ? candidoRatio(highStats.meanAlpha05Total,
                                     lowStats.meanAlpha05Total)
                      : std::numeric_limits<double>::infinity();
  const double lowMorphologyError = candidoMaxMorphologyError04_07(low);
  const double highMaxAsymmetry = candidoMaxRadialAsymmetry(high);
  const double lowBoundaryActivity =
      std::abs(low.cumulativeBoundaryLiquidInflow) +
      std::abs(low.cumulativeBoundaryLiquidOutflow);
  const double highBoundaryActivity =
      std::abs(high.cumulativeBoundaryLiquidInflow) +
      std::abs(high.cumulativeBoundaryLiquidOutflow);
  std::string status = "DOWNGRADED_NONFINITE_OPEN_BOUNDARY_DIAGNOSTIC";
  const bool finite =
      std::isfinite(tailRatio) && std::isfinite(peakRatio) &&
      std::isfinite(axialConvectiveRatio) && std::isfinite(axialTotalRatio) &&
      std::isfinite(lowMorphologyError) && std::isfinite(highMaxAsymmetry) &&
      std::isfinite(lowBoundaryActivity) && std::isfinite(highBoundaryActivity);
  if (finite) {
    const bool openBoundaryActive = lowBoundaryActivity > 0.0 || highBoundaryActivity > 0.0;
    const bool numericalQuality =
        low.alphaMassDrift <= 1e-3 && high.alphaMassDrift <= 1e-3 &&
        low.maxDiv <= 1e-7 && high.maxDiv <= 1e-7;
    const bool weakAllPhase = tailRatio <= 2.0 && peakRatio <= 2.0;
    const bool weakAxial = axialComparable && axialConvectiveRatio <= 2.0;
    const bool fixedPlaneDeveloped =
        lowMidplaneSamples > 0 && highMidplaneSamples > 0;
    const bool morphologyOk = lowMorphologyError <= 10.0;
    const bool whipOk = highMaxAsymmetry >= 0.05;
    if (!openBoundaryActive) {
      status = "BLOCKED_OPEN_BOUNDARY_NO_MEASURABLE_FLUX";
    } else if (!numericalQuality) {
      status = "DOWNGRADED_OPEN_BOUNDARY_NUMERICAL_QUALITY";
    } else if (!fixedPlaneDeveloped) {
      status = "BLOCKED_OPEN_BOUNDARY_FIXED_PLANE_UNDEVELOPED";
    } else if (!weakAllPhase || !weakAxial) {
      status = "DOWNGRADED_OPEN_BOUNDARY_CURRENT_RATIO_ABOVE_WEAK_BAR";
    } else if (!morphologyOk || !whipOk) {
      status = "DOWNGRADED_OPEN_BOUNDARY_MORPHOLOGY_OR_WHIP_TRADEOFF";
    } else {
      status = "APPROXIMATE_OPEN_BOUNDARY_CANDIDATE_ALL_GUARDS_GREEN";
    }
  }
  const double midplaneYOverDi =
      0.5 * setup.collectorDistance / std::max(setup.innerDiameter, 1e-30);
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << low.steps << "," << high.steps << "," << minAreaDi2 << ","
      << midplaneYOverDi << "," << low.alphaMassDrift << ","
      << high.alphaMassDrift << "," << low.maxDiv << "," << high.maxDiv << ","
      << low.cumulativeBoundaryLiquidInflow << ","
      << high.cumulativeBoundaryLiquidInflow << ","
      << low.cumulativeBoundaryLiquidOutflow << ","
      << high.cumulativeBoundaryLiquidOutflow << ","
      << low.cumulativeBoundaryLiquidFlux << ","
      << high.cumulativeBoundaryLiquidFlux << ","
      << low.massBudgetResidual << "," << high.massBudgetResidual << ","
      << lowMorphologyError << "," << highMaxAsymmetry << ","
      << tailRatio << "," << peakRatio << "," << lowMidplaneSamples << ","
      << highMidplaneSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << lowStats.meanAlpha05Convective << ","
      << highStats.meanAlpha05Convective << "," << axialConvectiveRatio << ","
      << lowStats.meanAlpha05Total << "," << highStats.meanAlpha05Total << ","
      << axialTotalRatio << "," << status << "\n";
  csv.flush();
}

struct CandidoMidplaneDevelopmentStats {
  int tailSamples = 0;
  int developedSamples = 0;
  int firstDevelopedStep = -1;
  double firstDevelopedTimeMs = std::numeric_limits<double>::quiet_NaN();
  double maxAlpha05AreaDi2 = 0.0;
  double meanDevelopedAlpha05Current = 0.0;
  double peakDevelopedAlpha05Current = 0.0;
  double maxTipY = 0.0;
};

static CandidoMidplaneDevelopmentStats candidoMidplaneDevelopmentStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  CandidoMidplaneDevelopmentStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = 0; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    s.maxAlpha05AreaDi2 = std::max(s.maxAlpha05AreaDi2, h.midplaneAlpha05AreaDi2);
    s.maxTipY = std::max(s.maxTipY, h.tipY);
    if (h.midplaneAlpha05AreaDi2 >= minAreaDi2 && s.firstDevelopedStep < 0) {
      s.firstDevelopedStep = h.step;
      s.firstDevelopedTimeMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
    }
    if (i < tailStart) continue;
    ++s.tailSamples;
    if (h.midplaneAlpha05AreaDi2 < minAreaDi2) continue;
    ++s.developedSamples;
    const double current = std::abs(h.alpha05ConvectiveCurrent);
    s.meanDevelopedAlpha05Current += current;
    s.peakDevelopedAlpha05Current =
        std::max(s.peakDevelopedAlpha05Current, current);
  }
  if (s.developedSamples > 0) {
    s.meanDevelopedAlpha05Current /=
        static_cast<double>(s.developedSamples);
  }
  return s;
}

static void writePaperCurrentDevelopmentTradeoffRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& stableLow,
    const electrospray::CandidoConeJetSmokeReport3D& stableHigh,
    const electrospray::CandidoConeJetSmokeReport3D& extendedLow,
    const electrospray::CandidoConeJetSmokeReport3D& extendedHigh,
    double minAreaDi2) {
  const CandidoMidplaneDevelopmentStats stableLowStats =
      candidoMidplaneDevelopmentStats(stableLow, minAreaDi2);
  const CandidoMidplaneDevelopmentStats stableHighStats =
      candidoMidplaneDevelopmentStats(stableHigh, minAreaDi2);
  const CandidoMidplaneDevelopmentStats extendedLowStats =
      candidoMidplaneDevelopmentStats(extendedLow, minAreaDi2);
  const CandidoMidplaneDevelopmentStats extendedHighStats =
      candidoMidplaneDevelopmentStats(extendedHigh, minAreaDi2);
  const bool extendedComparable =
      extendedLowStats.developedSamples > 0 &&
      extendedHighStats.developedSamples > 0;
  const double extendedMeanRatio =
      extendedComparable
          ? candidoRatio(extendedHighStats.meanDevelopedAlpha05Current,
                         extendedLowStats.meanDevelopedAlpha05Current)
          : std::numeric_limits<double>::infinity();
  const double extendedPeakRatio =
      extendedComparable
          ? candidoRatio(extendedHighStats.peakDevelopedAlpha05Current,
                         extendedLowStats.peakDevelopedAlpha05Current)
          : std::numeric_limits<double>::infinity();
  std::string status = "DOWNGRADED_NONFINITE_DEVELOPMENT_TRADEOFF";
  const bool finite =
      std::isfinite(stableLow.alphaMassDrift) &&
      std::isfinite(stableHigh.alphaMassDrift) &&
      std::isfinite(extendedLow.alphaMassDrift) &&
      std::isfinite(extendedHigh.alphaMassDrift) &&
      (!extendedComparable ||
       (std::isfinite(extendedMeanRatio) && std::isfinite(extendedPeakRatio)));
  if (finite) {
    const bool extendedQuality =
        extendedLow.alphaMassDrift <= 1e-3 &&
        extendedHigh.alphaMassDrift <= 1e-3 &&
        extendedLow.maxDiv <= 1e-7 && extendedHigh.maxDiv <= 1e-7;
    const bool stableUndeveloped =
        stableLowStats.developedSamples == 0 ||
        stableHighStats.developedSamples == 0;
    if (!extendedQuality) {
      status = "DOWNGRADED_EXTENDED_WINDOW_NUMERICAL_QUALITY";
    } else if (!extendedComparable) {
      status = stableUndeveloped
                   ? "BLOCKED_STABLE_AND_EXTENDED_FIXED_PLANE_UNDEVELOPED"
                   : "BLOCKED_EXTENDED_FIXED_PLANE_UNDEVELOPED";
    } else if (extendedMeanRatio > 2.0 || extendedPeakRatio > 2.0) {
      status = "DOWNGRADED_EXTENDED_WINDOW_CURRENT_RATIO_ABOVE_WEAK_BAR";
    } else {
      status = "APPROXIMATE_EXTENDED_WINDOW_CURRENT_WEAK_SENSITIVITY";
    }
  }
  csv << caseName << "," << minAreaDi2 << "," << stableLow.steps << ","
      << stableHigh.steps << "," << extendedLow.steps << ","
      << extendedHigh.steps << "," << stableLow.alphaMassDrift << ","
      << stableHigh.alphaMassDrift << "," << extendedLow.alphaMassDrift << ","
      << extendedHigh.alphaMassDrift << "," << stableLow.maxDiv << ","
      << stableHigh.maxDiv << "," << extendedLow.maxDiv << ","
      << extendedHigh.maxDiv << "," << stableLowStats.developedSamples << ","
      << stableHighStats.developedSamples << ","
      << extendedLowStats.developedSamples << ","
      << extendedHighStats.developedSamples << ","
      << stableLowStats.maxAlpha05AreaDi2 << ","
      << stableHighStats.maxAlpha05AreaDi2 << ","
      << extendedLowStats.maxAlpha05AreaDi2 << ","
      << extendedHighStats.maxAlpha05AreaDi2 << ","
      << stableLowStats.maxTipY << "," << stableHighStats.maxTipY << ","
      << extendedLowStats.maxTipY << "," << extendedHighStats.maxTipY << ","
      << extendedLowStats.firstDevelopedStep << ","
      << extendedHighStats.firstDevelopedStep << ","
      << extendedLowStats.firstDevelopedTimeMs << ","
      << extendedHighStats.firstDevelopedTimeMs << ","
      << extendedLowStats.meanDevelopedAlpha05Current << ","
      << extendedHighStats.meanDevelopedAlpha05Current << ","
      << extendedMeanRatio << ","
      << extendedLowStats.peakDevelopedAlpha05Current << ","
      << extendedHighStats.peakDevelopedAlpha05Current << ","
      << extendedPeakRatio << "," << status << "\n";
  csv.flush();
}

static void writePreconditionedCurrentPlaneDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const CandidoMidplaneDevelopmentStats lowStats =
      candidoMidplaneDevelopmentStats(low, minAreaDi2);
  const CandidoMidplaneDevelopmentStats highStats =
      candidoMidplaneDevelopmentStats(high, minAreaDi2);
  const CandidoParetoAxialStats lowAxial =
      candidoParetoAxialStats(low, minAreaDi2);
  const CandidoParetoAxialStats highAxial =
      candidoParetoAxialStats(high, minAreaDi2);
  const double allPhaseTailRatio =
      candidoRatio(candidoMeanTailCurrentForObservable(high),
                   candidoMeanTailCurrentForObservable(low));
  const double allPhasePeakRatio = candidoPeakCurrentRatioForObservable(
      low, high, CandidoCurrentSensitivityObservable::Convective);
  const bool fixedComparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const bool axialComparable =
      lowAxial.developedSamples > 0 && highAxial.developedSamples > 0;
  const double fixedMeanRatio =
      fixedComparable ? candidoRatio(highStats.meanDevelopedAlpha05Current,
                                     lowStats.meanDevelopedAlpha05Current)
                      : std::numeric_limits<double>::infinity();
  const double fixedPeakRatio =
      fixedComparable ? candidoRatio(highStats.peakDevelopedAlpha05Current,
                                     lowStats.peakDevelopedAlpha05Current)
                      : std::numeric_limits<double>::infinity();
  const double axialConvectiveRatio =
      axialComparable ? candidoRatio(highAxial.meanAlpha05Convective,
                                     lowAxial.meanAlpha05Convective)
                      : std::numeric_limits<double>::infinity();
  const double axialTotalRatio =
      axialComparable ? candidoRatio(highAxial.meanAlpha05Total,
                                     lowAxial.meanAlpha05Total)
                      : std::numeric_limits<double>::infinity();
  const double midplaneYOverDi =
      0.5 * setup.collectorDistance / std::max(setup.innerDiameter, 1e-30);
  const double tipYOverDi =
      opt.preconditionedJetTipYOverInnerDiameter > 0.0
          ? opt.preconditionedJetTipYOverInnerDiameter
          : midplaneYOverDi + 0.75;
  const bool finite =
      std::isfinite(allPhaseTailRatio) && std::isfinite(allPhasePeakRatio) &&
      std::isfinite(fixedMeanRatio) && std::isfinite(fixedPeakRatio) &&
      std::isfinite(axialConvectiveRatio) && std::isfinite(axialTotalRatio) &&
      std::isfinite(low.alphaMassDrift) && std::isfinite(high.alphaMassDrift) &&
      std::isfinite(low.maxDiv) && std::isfinite(high.maxDiv);
  std::string status = "DOWNGRADED_NONFINITE_PRECONDITIONED_CURRENT_DIAGNOSTIC";
  if (finite) {
    const bool numericalQuality =
        low.alphaMassDrift <= 1e-3 && high.alphaMassDrift <= 1e-3 &&
        low.maxDiv <= 1e-7 && high.maxDiv <= 1e-7;
    const bool weakFixed =
        fixedComparable && fixedMeanRatio <= 2.0 && fixedPeakRatio <= 2.0;
    if (!numericalQuality) {
      status = "DOWNGRADED_PRECONDITIONED_CURRENT_NUMERICAL_QUALITY";
    } else if (!fixedComparable) {
      status = "BLOCKED_PRECONDITIONED_FIXED_PLANE_STILL_UNDEVELOPED";
    } else if (!weakFixed) {
      status = "DOWNGRADED_PRECONDITIONED_FIXED_PLANE_CURRENT_TOO_SENSITIVE";
    } else {
      status = "APPROXIMATE_PRECONDITIONED_FIXED_PLANE_WEAK_SENSITIVITY_DIAGNOSTIC_ONLY";
    }
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << low.steps << "," << high.steps << "," << minAreaDi2 << ","
      << midplaneYOverDi << "," << tipYOverDi << ","
      << opt.preconditionedJetRadiusInnerDiameters << ","
      << opt.preconditionedJetInterfaceWidthInnerDiameters << ","
      << opt.preconditionedJetVelocityScale << "," << low.alphaMassDrift
      << "," << high.alphaMassDrift << "," << low.maxDiv << ","
      << high.maxDiv << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.maxAlpha05AreaDi2
      << "," << highStats.maxAlpha05AreaDi2 << ","
      << lowStats.meanDevelopedAlpha05Current << ","
      << highStats.meanDevelopedAlpha05Current << "," << fixedMeanRatio
      << "," << lowStats.peakDevelopedAlpha05Current << ","
      << highStats.peakDevelopedAlpha05Current << "," << fixedPeakRatio
      << "," << allPhaseTailRatio << "," << allPhasePeakRatio << ","
      << lowAxial.developedSamples << "," << highAxial.developedSamples
      << "," << lowAxial.meanAlpha05Convective << ","
      << highAxial.meanAlpha05Convective << "," << axialConvectiveRatio
      << "," << lowAxial.meanAlpha05Total << ","
      << highAxial.meanAlpha05Total << "," << axialTotalRatio << ","
      << status << "\n";
  csv.flush();
}

static void writeMovingCollectorBoundaryDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const CandidoParetoAxialStats lowStats =
      candidoParetoAxialStats(low, minAreaDi2);
  const CandidoParetoAxialStats highStats =
      candidoParetoAxialStats(high, minAreaDi2);
  const double tailRatio =
      candidoRatio(candidoMeanTailCurrentForObservable(high),
                   candidoMeanTailCurrentForObservable(low));
  const double peakRatio = candidoPeakCurrentRatioForObservable(
      low, high, CandidoCurrentSensitivityObservable::Convective);
  const int lowMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(low, minAreaDi2);
  const int highMidplaneSamples =
      candidoTailDevelopedSamplesAtMidplane(high, minAreaDi2);
  const bool axialComparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double axialConvectiveRatio =
      axialComparable ? candidoRatio(highStats.meanAlpha05Convective,
                                     lowStats.meanAlpha05Convective)
                      : std::numeric_limits<double>::infinity();
  const double axialTotalRatio =
      axialComparable ? candidoRatio(highStats.meanAlpha05Total,
                                     lowStats.meanAlpha05Total)
                      : std::numeric_limits<double>::infinity();
  const double lowMorphologyError = candidoMaxMorphologyError04_07(low);
  const double highMaxAsymmetry = candidoMaxRadialAsymmetry(high);
  const double collectorDimensionlessSpeed =
      electrospray::candidoDimensionlessCollectorVelocityScale(setup);
  const bool finite =
      std::isfinite(tailRatio) && std::isfinite(peakRatio) &&
      std::isfinite(axialConvectiveRatio) && std::isfinite(axialTotalRatio) &&
      std::isfinite(lowMorphologyError) && std::isfinite(highMaxAsymmetry) &&
      std::isfinite(collectorDimensionlessSpeed);
  std::string status = "DOWNGRADED_NONFINITE_MOVING_COLLECTOR_DIAGNOSTIC";
  if (finite) {
    const bool numericalQuality =
        low.alphaMassDrift <= 1e-3 && high.alphaMassDrift <= 1e-3 &&
        low.maxDiv <= 1e-7 && high.maxDiv <= 1e-7;
    const bool fixedPlaneDeveloped =
        lowMidplaneSamples > 0 && highMidplaneSamples > 0;
    const bool weakAllPhase = tailRatio <= 2.0 && peakRatio <= 2.0;
    const bool weakAxial = axialComparable && axialConvectiveRatio <= 2.0;
    const bool morphologyOk = lowMorphologyError <= 10.0;
    const bool whipOk = highMaxAsymmetry >= 0.05;
    if (!numericalQuality) {
      status = "DOWNGRADED_MOVING_COLLECTOR_NUMERICAL_QUALITY";
    } else if (!fixedPlaneDeveloped) {
      status = "BLOCKED_MOVING_COLLECTOR_FIXED_PLANE_UNDEVELOPED";
    } else if (!weakAllPhase || !weakAxial) {
      status = "DOWNGRADED_MOVING_COLLECTOR_CURRENT_RATIO_ABOVE_WEAK_BAR";
    } else if (!morphologyOk || !whipOk) {
      status = "DOWNGRADED_MOVING_COLLECTOR_MORPHOLOGY_OR_WHIP_TRADEOFF";
    } else {
      status = "APPROXIMATE_MOVING_COLLECTOR_CANDIDATE_ALL_GUARDS_GREEN";
    }
  }
  const double midplaneYOverDi =
      0.5 * setup.collectorDistance / std::max(setup.innerDiameter, 1e-30);
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << setup.collectorSpeed << "," << collectorDimensionlessSpeed << ","
      << low.steps << "," << high.steps << "," << minAreaDi2 << ","
      << midplaneYOverDi << "," << low.alphaMassDrift << ","
      << high.alphaMassDrift << "," << low.maxDiv << "," << high.maxDiv
      << "," << lowMorphologyError << "," << highMaxAsymmetry << ","
      << tailRatio << "," << peakRatio << "," << lowMidplaneSamples << ","
      << highMidplaneSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << lowStats.meanAlpha05Convective << ","
      << highStats.meanAlpha05Convective << "," << axialConvectiveRatio
      << "," << lowStats.meanAlpha05Total << ","
      << highStats.meanAlpha05Total << "," << axialTotalRatio << ","
      << status << "\n";
  csv.flush();
}

static void writeDevelopedJetCurrentWindowDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    CandidoCurrentSensitivityObservable observable,
    double minAreaDi2) {
  struct DevelopedStats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanDevelopedCurrent = 0.0;
    double meanDevelopedArea = 0.0;
    double meanTailArea = 0.0;
    double maxArea = 0.0;
  };
  auto stats = [observable, minAreaDi2](
                   const electrospray::CandidoConeJetSmokeReport3D& r) {
    DevelopedStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      const double area = h.midplaneAlpha05AreaDi2;
      ++s.tailSamples;
      s.meanTailArea += area;
      s.maxArea = std::max(s.maxArea, area);
      if (area < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanDevelopedArea += area;
      s.meanDevelopedCurrent +=
          std::abs(candidoHistoryCurrentValue(h, observable));
    }
    s.meanTailArea /= std::max(s.tailSamples, 1);
    if (s.developedSamples > 0) {
      s.meanDevelopedArea /= s.developedSamples;
      s.meanDevelopedCurrent /= s.developedSamples;
    }
    return s;
  };
  const DevelopedStats lowStats = stats(low);
  const DevelopedStats highStats = stats(high);
  const bool comparableWindow =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double ratio =
      comparableWindow ? highStats.meanDevelopedCurrent /
                             std::max(lowStats.meanDevelopedCurrent, 1e-30)
                       : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_UNDEVELOPED_LOW_OR_HIGH_JET_AT_CURRENT_PLANE";
  if (comparableWindow) {
    status = ratio <= 2.0 ? "APPROXIMATE_DEVELOPED_WINDOW_WEAK_SENSITIVITY"
                          : "DOWNGRADED_DEVELOPED_WINDOW_TOO_VOLTAGE_SENSITIVE";
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << candidoCurrentObservableSource(observable) << "," << minAreaDi2 << ","
      << lowStats.tailSamples << "," << highStats.tailSamples << ","
      << lowStats.developedSamples << "," << highStats.developedSamples << ","
      << lowStats.meanTailArea << "," << highStats.meanTailArea << ","
      << lowStats.maxArea << "," << highStats.maxArea << ","
      << lowStats.meanDevelopedArea << "," << highStats.meanDevelopedArea << ","
      << lowStats.meanDevelopedCurrent << "," << highStats.meanDevelopedCurrent << ","
      << ratio << "," << status << "\n";
}

static void writeAxialDevelopedJetCurrentWindowDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    CandidoCurrentSensitivityObservable observable,
    double minAreaDi2) {
  struct DevelopedStats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanDevelopedCurrent = 0.0;
    double meanDevelopedArea = 0.0;
    double meanDevelopedYOverDi = 0.0;
    double maxArea = 0.0;
  };
  auto stats = [observable, minAreaDi2](
                   const electrospray::CandidoConeJetSmokeReport3D& r) {
    DevelopedStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      ++s.tailSamples;
      const double developedArea = candidoHistoryAxialDevelopedAreaValue(h, observable);
      s.maxArea = std::max(s.maxArea, developedArea);
      if (developedArea < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanDevelopedArea += developedArea;
      s.meanDevelopedYOverDi += candidoHistoryAxialDevelopedYValue(h, observable);
      s.meanDevelopedCurrent +=
          std::abs(candidoHistoryAxialDevelopedCurrentValue(h, observable));
    }
    if (s.developedSamples > 0) {
      s.meanDevelopedArea /= s.developedSamples;
      s.meanDevelopedYOverDi /= s.developedSamples;
      s.meanDevelopedCurrent /= s.developedSamples;
    }
    return s;
  };
  const DevelopedStats lowStats = stats(low);
  const DevelopedStats highStats = stats(high);
  const bool comparableWindow =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double ratio =
      comparableWindow ? highStats.meanDevelopedCurrent /
                             std::max(lowStats.meanDevelopedCurrent, 1e-30)
                       : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparableWindow) {
    const double currentScale =
        std::max(lowStats.meanDevelopedCurrent, highStats.meanDevelopedCurrent);
    if (currentScale <= 1e-30) {
      status = "BLOCKED_ZERO_DEVELOPED_CURRENT_OBSERVABLE";
    } else {
      status = ratio <= 2.0 ? "APPROXIMATE_AXIAL_DEVELOPED_WINDOW_WEAK_SENSITIVITY"
                            : "DOWNGRADED_AXIAL_DEVELOPED_WINDOW_TOO_VOLTAGE_SENSITIVE";
    }
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << candidoCurrentObservableSource(observable) << "," << minAreaDi2 << ","
      << lowStats.tailSamples << "," << highStats.tailSamples << ","
      << lowStats.developedSamples << "," << highStats.developedSamples << ","
      << lowStats.maxArea << "," << highStats.maxArea << ","
      << lowStats.meanDevelopedArea << "," << highStats.meanDevelopedArea << ","
      << lowStats.meanDevelopedYOverDi << "," << highStats.meanDevelopedYOverDi << ","
      << lowStats.meanDevelopedCurrent << "," << highStats.meanDevelopedCurrent << ","
      << ratio << "," << status << "\n";
}

static void writeAxialCurrentFactorizationDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  struct FactorStats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanArea = 0.0;
    double meanCurrent = 0.0;
    double meanAbsCharge = 0.0;
    double meanAbsUy = 0.0;
    double meanShapeFactor = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    FactorStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      ++s.tailSamples;
      if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanArea += h.developedJetAlpha05AreaDi2;
      s.meanCurrent += std::abs(h.developedJetAlpha05ConvectiveCurrent);
      s.meanAbsCharge += h.developedJetMeanAlpha05AbsCharge;
      s.meanAbsUy += h.developedJetMeanAlpha05AbsUy;
      s.meanShapeFactor += h.developedJetAlpha05CurrentShapeFactor;
    }
    if (s.developedSamples > 0) {
      s.meanArea /= s.developedSamples;
      s.meanCurrent /= s.developedSamples;
      s.meanAbsCharge /= s.developedSamples;
      s.meanAbsUy /= s.developedSamples;
      s.meanShapeFactor /= s.developedSamples;
    }
    return s;
  };
  const FactorStats lowStats = stats(low);
  const FactorStats highStats = stats(high);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double areaRatio =
      comparable ? highStats.meanArea / std::max(lowStats.meanArea, 1e-30)
                 : std::numeric_limits<double>::infinity();
  const double chargeRatio =
      comparable ? highStats.meanAbsCharge /
                       std::max(lowStats.meanAbsCharge, 1e-30)
                 : std::numeric_limits<double>::infinity();
  const double velocityRatio =
      comparable ? highStats.meanAbsUy / std::max(lowStats.meanAbsUy, 1e-30)
                 : std::numeric_limits<double>::infinity();
  const double shapeRatio =
      comparable ? highStats.meanShapeFactor /
                       std::max(lowStats.meanShapeFactor, 1e-30)
                 : std::numeric_limits<double>::infinity();
  const double currentRatio =
      comparable ? highStats.meanCurrent / std::max(lowStats.meanCurrent, 1e-30)
                 : std::numeric_limits<double>::infinity();
  const double productRatio =
      comparable ? areaRatio * chargeRatio * velocityRatio * shapeRatio
                 : std::numeric_limits<double>::infinity();
  std::string dominant = "blocked";
  if (comparable) {
    dominant = "area";
    double best = areaRatio;
    if (chargeRatio > best) {
      best = chargeRatio;
      dominant = "charge";
    }
    if (velocityRatio > best) {
      best = velocityRatio;
      dominant = "velocity";
    }
    if (shapeRatio > best) {
      dominant = "charge_velocity_alignment";
    }
  }
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    status = currentRatio <= 2.0
                 ? "APPROXIMATE_AXIAL_ALPHA05_CURRENT_WEAK_SENSITIVITY"
                 : "DOWNGRADED_AXIAL_ALPHA05_CURRENT_FACTOR_" + dominant +
                       "_DOMINATES";
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << minAreaDi2 << "," << lowStats.tailSamples << ","
      << highStats.tailSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << areaRatio << ","
      << lowStats.meanAbsCharge << "," << highStats.meanAbsCharge << ","
      << chargeRatio << "," << lowStats.meanAbsUy << ","
      << highStats.meanAbsUy << "," << velocityRatio << ","
      << lowStats.meanShapeFactor << "," << highStats.meanShapeFactor << ","
      << shapeRatio << "," << lowStats.meanCurrent << ","
      << highStats.meanCurrent << "," << currentRatio << ","
      << productRatio << "," << dominant << "," << status << "\n";
}

static void writePoissonFaceConvectiveFactorizationDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  struct FactorStats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanArea = 0.0;
    double meanCurrent = 0.0;
    double meanAbsUpwindCharge = 0.0;
    double meanAbsFaceFlux = 0.0;
    double meanAbsConvectiveFlux = 0.0;
    double maxAbsUpwindCharge = 0.0;
    double maxAbsFaceFlux = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    FactorStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      ++s.tailSamples;
      if (h.poissonFaceDevelopedAlpha05AreaDi2 < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanArea += h.poissonFaceDevelopedAlpha05AreaDi2;
      s.meanCurrent += std::abs(h.poissonFaceDevelopedAlpha05ConvectiveCurrent);
      s.meanAbsUpwindCharge += h.poissonFaceDevelopedAlpha05MeanAbsUpwindCharge;
      s.meanAbsFaceFlux += h.poissonFaceDevelopedAlpha05MeanAbsFaceFlux;
      s.meanAbsConvectiveFlux += h.poissonFaceDevelopedAlpha05MeanAbsConvectiveFlux;
      s.maxAbsUpwindCharge =
          std::max(s.maxAbsUpwindCharge,
                   h.poissonFaceDevelopedAlpha05MaxAbsUpwindCharge);
      s.maxAbsFaceFlux =
          std::max(s.maxAbsFaceFlux, h.poissonFaceDevelopedAlpha05MaxAbsFaceFlux);
    }
    if (s.developedSamples > 0) {
      const double inv = 1.0 / static_cast<double>(s.developedSamples);
      s.meanArea *= inv;
      s.meanCurrent *= inv;
      s.meanAbsUpwindCharge *= inv;
      s.meanAbsFaceFlux *= inv;
      s.meanAbsConvectiveFlux *= inv;
    }
    return s;
  };
  auto ratio = [](double highValue, double lowValue) {
    return highValue / std::max(lowValue, 1e-30);
  };
  const FactorStats lowStats = stats(low);
  const FactorStats highStats = stats(high);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double areaRatio =
      comparable ? ratio(highStats.meanArea, lowStats.meanArea)
                 : std::numeric_limits<double>::infinity();
  const double currentRatio =
      comparable ? ratio(highStats.meanCurrent, lowStats.meanCurrent)
                 : std::numeric_limits<double>::infinity();
  const double chargeRatio =
      comparable ? ratio(highStats.meanAbsUpwindCharge,
                         lowStats.meanAbsUpwindCharge)
                 : std::numeric_limits<double>::infinity();
  const double faceFluxRatio =
      comparable ? ratio(highStats.meanAbsFaceFlux, lowStats.meanAbsFaceFlux)
                 : std::numeric_limits<double>::infinity();
  const double absConvectiveFluxRatio =
      comparable ? ratio(highStats.meanAbsConvectiveFlux,
                         lowStats.meanAbsConvectiveFlux)
                 : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    const double chargeScale =
        std::max(lowStats.meanAbsUpwindCharge, highStats.meanAbsUpwindCharge);
    const double faceFluxScale =
        std::max(lowStats.meanAbsFaceFlux, highStats.meanAbsFaceFlux);
    const double absConvectiveFluxScale =
        std::max(lowStats.meanAbsConvectiveFlux, highStats.meanAbsConvectiveFlux);
    const double signedCurrentScale =
        std::max(lowStats.meanCurrent, highStats.meanCurrent);
    if (chargeScale <= 1e-30) {
      status = "BLOCKED_ZERO_FACE_UPWIND_CHARGE";
    } else if (faceFluxScale <= 1e-30) {
      status = "BLOCKED_ZERO_FACE_FLUX";
    } else if (absConvectiveFluxScale <= 1e-30) {
      status = "BLOCKED_ZERO_FACE_CONVECTIVE_PRODUCT";
    } else if (signedCurrentScale <= 1e-30) {
      status = "BLOCKED_SIGN_CANCELLATION_IN_FACE_CURRENT";
    } else {
      status = currentRatio <= 2.0
                   ? "APPROXIMATE_FACE_CONVECTIVE_CURRENT_WEAK_SENSITIVITY"
                   : "DOWNGRADED_FACE_CONVECTIVE_CURRENT_TOO_VOLTAGE_SENSITIVE";
    }
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << minAreaDi2 << "," << lowStats.tailSamples << ","
      << highStats.tailSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << areaRatio << ","
      << lowStats.meanCurrent << "," << highStats.meanCurrent << ","
      << currentRatio << "," << lowStats.meanAbsUpwindCharge << ","
      << highStats.meanAbsUpwindCharge << "," << chargeRatio << ","
      << lowStats.meanAbsFaceFlux << "," << highStats.meanAbsFaceFlux << ","
      << faceFluxRatio << "," << lowStats.meanAbsConvectiveFlux << ","
      << highStats.meanAbsConvectiveFlux << "," << absConvectiveFluxRatio << ","
      << lowStats.maxAbsUpwindCharge << "," << highStats.maxAbsUpwindCharge << ","
      << lowStats.maxAbsFaceFlux << "," << highStats.maxAbsFaceFlux << ","
      << status << "\n";
}

static void writePoissonFaceVelocityProjectionFactorizationDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  struct FactorStats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanArea = 0.0;
    double projectedCurrent = 0.0;
    double projectedAbsUpwindCharge = 0.0;
    double projectedAbsFaceFlux = 0.0;
    double projectedAbsConvectiveFlux = 0.0;
    double rawCurrent = 0.0;
    double rawAbsUpwindCharge = 0.0;
    double rawAbsFaceFlux = 0.0;
    double rawAbsConvectiveFlux = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    FactorStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      ++s.tailSamples;
      if (h.poissonFaceDevelopedAlpha05AreaDi2 < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanArea += h.poissonFaceDevelopedAlpha05AreaDi2;
      s.projectedCurrent += std::abs(h.poissonFaceDevelopedAlpha05ConvectiveCurrent);
      s.projectedAbsUpwindCharge += h.poissonFaceDevelopedAlpha05MeanAbsUpwindCharge;
      s.projectedAbsFaceFlux += h.poissonFaceDevelopedAlpha05MeanAbsFaceFlux;
      s.projectedAbsConvectiveFlux += h.poissonFaceDevelopedAlpha05MeanAbsConvectiveFlux;
      s.rawCurrent += std::abs(h.rawVelocityFaceDevelopedAlpha05ConvectiveCurrent);
      s.rawAbsUpwindCharge += h.rawVelocityFaceDevelopedAlpha05MeanAbsUpwindCharge;
      s.rawAbsFaceFlux += h.rawVelocityFaceDevelopedAlpha05MeanAbsFaceFlux;
      s.rawAbsConvectiveFlux += h.rawVelocityFaceDevelopedAlpha05MeanAbsConvectiveFlux;
    }
    if (s.developedSamples > 0) {
      const double inv = 1.0 / static_cast<double>(s.developedSamples);
      s.meanArea *= inv;
      s.projectedCurrent *= inv;
      s.projectedAbsUpwindCharge *= inv;
      s.projectedAbsFaceFlux *= inv;
      s.projectedAbsConvectiveFlux *= inv;
      s.rawCurrent *= inv;
      s.rawAbsUpwindCharge *= inv;
      s.rawAbsFaceFlux *= inv;
      s.rawAbsConvectiveFlux *= inv;
    }
    return s;
  };
  auto ratio = [](double highValue, double lowValue) {
    return highValue / std::max(lowValue, 1e-30);
  };
  const FactorStats lowStats = stats(low);
  const FactorStats highStats = stats(high);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double areaRatio =
      comparable ? ratio(highStats.meanArea, lowStats.meanArea)
                 : std::numeric_limits<double>::infinity();
  const double projectedCurrentRatio =
      comparable ? ratio(highStats.projectedCurrent, lowStats.projectedCurrent)
                 : std::numeric_limits<double>::infinity();
  const double rawCurrentRatio =
      comparable ? ratio(highStats.rawCurrent, lowStats.rawCurrent)
                 : std::numeric_limits<double>::infinity();
  const double projectedFaceFluxRatio =
      comparable ? ratio(highStats.projectedAbsFaceFlux, lowStats.projectedAbsFaceFlux)
                 : std::numeric_limits<double>::infinity();
  const double rawFaceFluxRatio =
      comparable ? ratio(highStats.rawAbsFaceFlux, lowStats.rawAbsFaceFlux)
                 : std::numeric_limits<double>::infinity();
  const double projectedToRawCurrentLow =
      comparable ? ratio(lowStats.projectedCurrent, lowStats.rawCurrent)
                 : std::numeric_limits<double>::infinity();
  const double projectedToRawCurrentHigh =
      comparable ? ratio(highStats.projectedCurrent, highStats.rawCurrent)
                 : std::numeric_limits<double>::infinity();
  const double projectedToRawFaceFluxLow =
      comparable ? ratio(lowStats.projectedAbsFaceFlux, lowStats.rawAbsFaceFlux)
                 : std::numeric_limits<double>::infinity();
  const double projectedToRawFaceFluxHigh =
      comparable ? ratio(highStats.projectedAbsFaceFlux, highStats.rawAbsFaceFlux)
                 : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    const double projectedCurrentScale =
        std::max(lowStats.projectedCurrent, highStats.projectedCurrent);
    const double rawCurrentScale = std::max(lowStats.rawCurrent, highStats.rawCurrent);
    const double rawFaceFluxScale =
        std::max(lowStats.rawAbsFaceFlux, highStats.rawAbsFaceFlux);
    if (projectedCurrentScale <= 1e-30) {
      status = "BLOCKED_ZERO_PROJECTED_FACE_CURRENT";
    } else if (rawFaceFluxScale <= 1e-30) {
      status = "BLOCKED_ZERO_RAW_VELOCITY_FACE_FLUX";
    } else if (rawCurrentScale <= 1e-30) {
      status = "BLOCKED_ZERO_RAW_VELOCITY_CONVECTIVE_CURRENT";
    } else if (projectedCurrentRatio > 2.0 && rawCurrentRatio <= 2.0) {
      status = "DOWNGRADED_RHIE_CHOW_PROJECTION_CURRENT_SENSITIVITY";
    } else if (rawCurrentRatio > 2.0) {
      status = "DOWNGRADED_RAW_VELOCITY_CURRENT_TOO_VOLTAGE_SENSITIVE";
    } else {
      status = "APPROXIMATE_PROJECTED_AND_RAW_CURRENT_WEAK_SENSITIVITY";
    }
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << minAreaDi2 << "," << lowStats.tailSamples << ","
      << highStats.tailSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << areaRatio << ","
      << lowStats.projectedCurrent << "," << highStats.projectedCurrent << ","
      << projectedCurrentRatio << "," << lowStats.rawCurrent << ","
      << highStats.rawCurrent << "," << rawCurrentRatio << ","
      << lowStats.projectedAbsFaceFlux << "," << highStats.projectedAbsFaceFlux
      << "," << projectedFaceFluxRatio << "," << lowStats.rawAbsFaceFlux << ","
      << highStats.rawAbsFaceFlux << "," << rawFaceFluxRatio << ","
      << lowStats.projectedAbsUpwindCharge << ","
      << highStats.projectedAbsUpwindCharge << ","
      << lowStats.rawAbsUpwindCharge << "," << highStats.rawAbsUpwindCharge
      << "," << lowStats.projectedAbsConvectiveFlux << ","
      << highStats.projectedAbsConvectiveFlux << ","
      << lowStats.rawAbsConvectiveFlux << "," << highStats.rawAbsConvectiveFlux
      << "," << projectedToRawCurrentLow << "," << projectedToRawCurrentHigh
      << "," << projectedToRawFaceFluxLow << "," << projectedToRawFaceFluxHigh
      << "," << status << "\n";
}

static void writeMomentumSourceFactorizationDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  struct SourceStats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanArea = 0.0;
    double meanAbsUy = 0.0;
    double meanAbsElectricSource = 0.0;
    double meanAbsSurfaceSource = 0.0;
    double meanAbsSource = 0.0;
    double meanAbsAcceleration = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    SourceStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      ++s.tailSamples;
      if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanArea += h.developedJetAlpha05AreaDi2;
      s.meanAbsUy += h.developedJetMeanAlpha05AbsUy;
      s.meanAbsElectricSource +=
          h.developedJetMeanAlpha05AbsElectricMomentumSourceY;
      s.meanAbsSurfaceSource +=
          h.developedJetMeanAlpha05AbsSurfaceMomentumSourceY;
      s.meanAbsSource += h.developedJetMeanAlpha05AbsMomentumSourceY;
      s.meanAbsAcceleration +=
          h.developedJetMeanAlpha05AbsMomentumAccelerationY;
    }
    if (s.developedSamples > 0) {
      const double inv = 1.0 / static_cast<double>(s.developedSamples);
      s.meanArea *= inv;
      s.meanAbsUy *= inv;
      s.meanAbsElectricSource *= inv;
      s.meanAbsSurfaceSource *= inv;
      s.meanAbsSource *= inv;
      s.meanAbsAcceleration *= inv;
    }
    return s;
  };
  auto ratio = [](double highValue, double lowValue) {
    return highValue / std::max(lowValue, 1e-30);
  };
  const SourceStats lowStats = stats(low);
  const SourceStats highStats = stats(high);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double areaRatio =
      comparable ? ratio(highStats.meanArea, lowStats.meanArea)
                 : std::numeric_limits<double>::infinity();
  const double velocityRatio =
      comparable ? ratio(highStats.meanAbsUy, lowStats.meanAbsUy)
                 : std::numeric_limits<double>::infinity();
  const double electricSourceRatio =
      comparable ? ratio(highStats.meanAbsElectricSource,
                         lowStats.meanAbsElectricSource)
                 : std::numeric_limits<double>::infinity();
  const double surfaceSourceRatio =
      comparable ? ratio(highStats.meanAbsSurfaceSource,
                         lowStats.meanAbsSurfaceSource)
                 : std::numeric_limits<double>::infinity();
  const double sourceRatio =
      comparable ? ratio(highStats.meanAbsSource, lowStats.meanAbsSource)
                 : std::numeric_limits<double>::infinity();
  const double accelerationRatio =
      comparable ? ratio(highStats.meanAbsAcceleration,
                         lowStats.meanAbsAcceleration)
                 : std::numeric_limits<double>::infinity();
  std::string dominant = "none";
  if (comparable) {
    const double velocityDistance = std::abs(velocityRatio - 1.0);
    const double electricDistance = std::abs(electricSourceRatio - 1.0);
    const double surfaceDistance = std::abs(surfaceSourceRatio - 1.0);
    const double sourceDistance = std::abs(sourceRatio - 1.0);
    dominant = "total_source";
    double best = sourceDistance;
    if (electricDistance > best) {
      best = electricDistance;
      dominant = "electric_source";
    }
    if (surfaceDistance > best) {
      best = surfaceDistance;
      dominant = "surface_source";
    }
    if (velocityDistance > best) {
      dominant = "velocity_response";
    }
  }
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    const double sourceScale =
        std::max(lowStats.meanAbsSource, highStats.meanAbsSource);
    const double velocityScale = std::max(lowStats.meanAbsUy, highStats.meanAbsUy);
    if (sourceScale <= 1e-30) {
      status = "BLOCKED_ZERO_MOMENTUM_SOURCE";
    } else if (velocityScale <= 1e-30) {
      status = "BLOCKED_ZERO_ALPHA05_VELOCITY";
    } else if (velocityRatio > 1.5 && electricSourceRatio > 1.5) {
      status = "DOWNGRADED_ELECTRIC_SOURCE_DRIVES_VELOCITY_SENSITIVITY";
    } else if (velocityRatio > 1.5 && sourceRatio > 1.5) {
      status = "DOWNGRADED_TOTAL_SOURCE_DRIVES_VELOCITY_SENSITIVITY";
    } else {
      status = "APPROXIMATE_SOURCE_VELOCITY_WEAK_SENSITIVITY";
    }
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << minAreaDi2 << "," << lowStats.tailSamples << ","
      << highStats.tailSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << areaRatio << ","
      << lowStats.meanAbsUy << "," << highStats.meanAbsUy << ","
      << velocityRatio << "," << lowStats.meanAbsElectricSource << ","
      << highStats.meanAbsElectricSource << "," << electricSourceRatio << ","
      << lowStats.meanAbsSurfaceSource << "," << highStats.meanAbsSurfaceSource
      << "," << surfaceSourceRatio << "," << lowStats.meanAbsSource << ","
      << highStats.meanAbsSource << "," << sourceRatio << ","
      << lowStats.meanAbsAcceleration << "," << highStats.meanAbsAcceleration
      << "," << accelerationRatio << "," << dominant << "," << status << "\n";
}

static void writeInterfaceChargeTransportDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& candidateLow,
    const electrospray::CandidoConeJetSmokeReport3D& candidateHigh,
    double minAreaDi2) {
  const CandidoParetoAxialStats baseLowStats =
      candidoParetoAxialStats(baselineLow, minAreaDi2);
  const CandidoParetoAxialStats baseHighStats =
      candidoParetoAxialStats(baselineHigh, minAreaDi2);
  const CandidoParetoAxialStats candLowStats =
      candidoParetoAxialStats(candidateLow, minAreaDi2);
  const CandidoParetoAxialStats candHighStats =
      candidoParetoAxialStats(candidateHigh, minAreaDi2);
  const bool baseComparable =
      baseLowStats.developedSamples > 0 && baseHighStats.developedSamples > 0;
  const bool candComparable =
      candLowStats.developedSamples > 0 && candHighStats.developedSamples > 0;
  const double baselineAxialRatio =
      baseComparable ? candidoRatio(baseHighStats.meanAlpha05Convective,
                                    baseLowStats.meanAlpha05Convective)
                     : std::numeric_limits<double>::infinity();
  const double candidateAxialRatio =
      candComparable ? candidoRatio(candHighStats.meanAlpha05Convective,
                                    candLowStats.meanAlpha05Convective)
                     : std::numeric_limits<double>::infinity();
  const double candidateChargeRatio =
      candComparable ? candidoRatio(candHighStats.meanAbsCharge,
                                    candLowStats.meanAbsCharge)
                     : std::numeric_limits<double>::infinity();
  const double candidateVelocityRatio =
      candComparable ? candidoRatio(candHighStats.meanAbsUy,
                                    candLowStats.meanAbsUy)
                     : std::numeric_limits<double>::infinity();
  const double candidateMorphologyError =
      candidoMaxMorphologyError04_07(candidateLow);
  const double candidateHighAsymmetry = candidoMaxRadialAsymmetry(candidateHigh);
  const bool numericalQuality =
      candidateLow.alphaMassDrift <= 1e-3 && candidateHigh.alphaMassDrift <= 1e-3 &&
      candidateLow.maxDiv <= 1e-7 && candidateHigh.maxDiv <= 1e-7;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (candComparable) {
    if (!numericalQuality) {
      status = "DOWNGRADED_INTERFACE_CHARGE_NUMERICAL_QUALITY";
    } else if (candidateAxialRatio <= 2.0 && candidateMorphologyError <= 10.0 &&
               candidateHighAsymmetry >= 0.05) {
      status = "APPROXIMATE_INTERFACE_CHARGE_CANDIDATE_ALL_GUARDS_GREEN";
    } else if (candidateAxialRatio <= 2.0) {
      status = "DOWNGRADED_INTERFACE_CHARGE_WEAK_CURRENT_WITH_MORPHOLOGY_OR_WHIP_TRADEOFF";
    } else if (candidateAxialRatio < baselineAxialRatio) {
      status = "APPROXIMATE_INTERFACE_CHARGE_REDUCES_CURRENT_SENSITIVITY_DIAGNOSTIC_ONLY";
    } else {
      status = "DOWNGRADED_INTERFACE_CHARGE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }
  csv << caseName << "," << baselineLow.targetCaE << "," << baselineHigh.targetCaE
      << "," << minAreaDi2 << "," << baseLowStats.developedSamples << ","
      << baseHighStats.developedSamples << "," << candLowStats.developedSamples
      << "," << candHighStats.developedSamples << "," << baselineAxialRatio
      << "," << candidateAxialRatio << "," << candidateChargeRatio << ","
      << candidateVelocityRatio << "," << candLowStats.meanAlpha05Convective
      << "," << candHighStats.meanAlpha05Convective << ","
      << candLowStats.meanAbsCharge << "," << candHighStats.meanAbsCharge
      << "," << candLowStats.meanAbsUy << "," << candHighStats.meanAbsUy
      << "," << candidateLow.cumulativeChargeClampCorrectionL1 << ","
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
      << candidateMorphologyError << "," << candidateHighAsymmetry << ","
      << status << "\n";
}

static void writePostChargePotentialRefreshDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& candidateLow,
    const electrospray::CandidoConeJetSmokeReport3D& candidateHigh,
    double minAreaDi2) {
  const CandidoParetoAxialStats baseLowStats =
      candidoParetoAxialStats(baselineLow, minAreaDi2);
  const CandidoParetoAxialStats baseHighStats =
      candidoParetoAxialStats(baselineHigh, minAreaDi2);
  const CandidoParetoAxialStats candLowStats =
      candidoParetoAxialStats(candidateLow, minAreaDi2);
  const CandidoParetoAxialStats candHighStats =
      candidoParetoAxialStats(candidateHigh, minAreaDi2);
  const bool baseComparable =
      baseLowStats.developedSamples > 0 && baseHighStats.developedSamples > 0;
  const bool candComparable =
      candLowStats.developedSamples > 0 && candHighStats.developedSamples > 0;
  const double baselineAxialRatio =
      baseComparable ? candidoRatio(baseHighStats.meanAlpha05Convective,
                                    baseLowStats.meanAlpha05Convective)
                     : std::numeric_limits<double>::infinity();
  const double candidateAxialRatio =
      candComparable ? candidoRatio(candHighStats.meanAlpha05Convective,
                                    candLowStats.meanAlpha05Convective)
                     : std::numeric_limits<double>::infinity();
  const double candidateChargeRatio =
      candComparable ? candidoRatio(candHighStats.meanAbsCharge,
                                    candLowStats.meanAbsCharge)
                     : std::numeric_limits<double>::infinity();
  const double candidateVelocityRatio =
      candComparable ? candidoRatio(candHighStats.meanAbsUy,
                                    candLowStats.meanAbsUy)
                     : std::numeric_limits<double>::infinity();
  const double candidateLowElectricSource =
      candidoTailMeanAlpha05ElectricSource(candidateLow, minAreaDi2);
  const double candidateHighElectricSource =
      candidoTailMeanAlpha05ElectricSource(candidateHigh, minAreaDi2);
  const double candidateElectricSourceRatio =
      candComparable ? candidoRatio(candidateHighElectricSource,
                                    candidateLowElectricSource)
                     : std::numeric_limits<double>::infinity();
  const double candidateMorphologyError =
      candidoMaxMorphologyError04_07(candidateLow);
  const double candidateHighAsymmetry = candidoMaxRadialAsymmetry(candidateHigh);
  const bool numericalQuality =
      candidateLow.alphaMassDrift <= 1e-3 && candidateHigh.alphaMassDrift <= 1e-3 &&
      candidateLow.maxDiv <= 1e-7 && candidateHigh.maxDiv <= 1e-7 &&
      candidateLow.maxPostChargePotentialResidual <= 1e-7 &&
      candidateHigh.maxPostChargePotentialResidual <= 1e-7;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (candComparable) {
    if (!numericalQuality) {
      status = "DOWNGRADED_POST_CHARGE_REFRESH_NUMERICAL_QUALITY";
    } else if (candidateAxialRatio <= 2.0 && candidateMorphologyError <= 10.0 &&
               candidateHighAsymmetry >= 0.05) {
      status = "APPROXIMATE_POST_CHARGE_REFRESH_ALL_GUARDS_GREEN";
    } else if (candidateAxialRatio < baselineAxialRatio) {
      status = "APPROXIMATE_POST_CHARGE_REFRESH_REDUCES_CURRENT_SENSITIVITY";
    } else {
      status = "DOWNGRADED_POST_CHARGE_REFRESH_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }
  csv << caseName << "," << baselineLow.targetCaE << "," << baselineHigh.targetCaE
      << "," << minAreaDi2 << "," << baseLowStats.developedSamples << ","
      << baseHighStats.developedSamples << "," << candLowStats.developedSamples
      << "," << candHighStats.developedSamples << "," << baselineAxialRatio
      << "," << candidateAxialRatio << "," << candidateChargeRatio << ","
      << candidateVelocityRatio << "," << candidateElectricSourceRatio << ","
      << candLowStats.meanAlpha05Convective << ","
      << candHighStats.meanAlpha05Convective << ","
      << candLowStats.meanAbsCharge << "," << candHighStats.meanAbsCharge
      << "," << candLowStats.meanAbsUy << "," << candHighStats.meanAbsUy
      << "," << candidateLowElectricSource << ","
      << candidateHighElectricSource << ","
      << candidateLow.maxPostChargePotentialResidual << ","
      << candidateHigh.maxPostChargePotentialResidual << ","
      << candidateLow.maxPostChargeRelativeGaussLawResidual << ","
      << candidateHigh.maxPostChargeRelativeGaussLawResidual << ","
      << candidateLow.alphaMassDrift << "," << candidateHigh.alphaMassDrift
      << "," << candidateLow.maxDiv << "," << candidateHigh.maxDiv << ","
      << candidateMorphologyError << "," << candidateHighAsymmetry << ","
      << status << "\n";
}

static void writeConductivityPotentialChargeClosureDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& candidateLow,
    const electrospray::CandidoConeJetSmokeReport3D& candidateHigh,
    double minAreaDi2) {
  const CandidoParetoAxialStats baseLowStats =
      candidoParetoAxialStats(baselineLow, minAreaDi2);
  const CandidoParetoAxialStats baseHighStats =
      candidoParetoAxialStats(baselineHigh, minAreaDi2);
  const CandidoParetoAxialStats candLowStats =
      candidoParetoAxialStats(candidateLow, minAreaDi2);
  const CandidoParetoAxialStats candHighStats =
      candidoParetoAxialStats(candidateHigh, minAreaDi2);
  const bool baseComparable =
      baseLowStats.developedSamples > 0 && baseHighStats.developedSamples > 0;
  const bool candComparable =
      candLowStats.developedSamples > 0 && candHighStats.developedSamples > 0;
  const double baselineAxialRatio =
      baseComparable ? candidoRatio(baseHighStats.meanAlpha05Convective,
                                    baseLowStats.meanAlpha05Convective)
                     : std::numeric_limits<double>::infinity();
  const double candidateAxialRatio =
      candComparable ? candidoRatio(candHighStats.meanAlpha05Convective,
                                    candLowStats.meanAlpha05Convective)
                     : std::numeric_limits<double>::infinity();
  const double candidateChargeRatio =
      candComparable ? candidoRatio(candHighStats.meanAbsCharge,
                                    candLowStats.meanAbsCharge)
                     : std::numeric_limits<double>::infinity();
  const double candidateVelocityRatio =
      candComparable ? candidoRatio(candHighStats.meanAbsUy,
                                    candLowStats.meanAbsUy)
                     : std::numeric_limits<double>::infinity();
  const double candidateLowElectricSource =
      candidoTailMeanAlpha05ElectricSource(candidateLow, minAreaDi2);
  const double candidateHighElectricSource =
      candidoTailMeanAlpha05ElectricSource(candidateHigh, minAreaDi2);
  const double candidateElectricSourceRatio =
      candComparable ? candidoRatio(candidateHighElectricSource,
                                    candidateLowElectricSource)
                     : std::numeric_limits<double>::infinity();
  const double candidateMorphologyError =
      candidoMaxMorphologyError04_07(candidateLow);
  const double candidateHighAsymmetry = candidoMaxRadialAsymmetry(candidateHigh);
  const bool numericalQuality =
      candidateLow.alphaMassDrift <= 1e-3 && candidateHigh.alphaMassDrift <= 1e-3 &&
      candidateLow.maxDiv <= 1e-7 && candidateHigh.maxDiv <= 1e-7 &&
      candidateLow.maxConductivityPotentialResidual <= 1e-7 &&
      candidateHigh.maxConductivityPotentialResidual <= 1e-7;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (candComparable) {
    if (!numericalQuality) {
      status = "DOWNGRADED_CONDUCTIVITY_CLOSURE_NUMERICAL_QUALITY";
    } else if (candidateAxialRatio <= 2.0 && candidateMorphologyError <= 10.0 &&
               candidateHighAsymmetry >= 0.05) {
      status = "APPROXIMATE_CONDUCTIVITY_CLOSURE_ALL_GUARDS_GREEN";
    } else if (candidateAxialRatio < baselineAxialRatio) {
      status = "APPROXIMATE_CONDUCTIVITY_CLOSURE_REDUCES_CURRENT_SENSITIVITY";
    } else {
      status = "DOWNGRADED_CONDUCTIVITY_CLOSURE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }
  csv << caseName << "," << baselineLow.targetCaE << "," << baselineHigh.targetCaE
      << "," << minAreaDi2 << "," << baseLowStats.developedSamples << ","
      << baseHighStats.developedSamples << "," << candLowStats.developedSamples
      << "," << candHighStats.developedSamples << "," << baselineAxialRatio
      << "," << candidateAxialRatio << "," << candidateChargeRatio << ","
      << candidateVelocityRatio << "," << candidateElectricSourceRatio << ","
      << candLowStats.meanAlpha05Convective << ","
      << candHighStats.meanAlpha05Convective << ","
      << candLowStats.meanAbsCharge << "," << candHighStats.meanAbsCharge
      << "," << candLowStats.meanAbsUy << "," << candHighStats.meanAbsUy
      << "," << candidateLowElectricSource << ","
      << candidateHighElectricSource << ","
      << candidateLow.maxConductivityPotentialResidual << ","
      << candidateHigh.maxConductivityPotentialResidual << ","
      << candidateLow.cumulativeConductivityClosureClampCorrectionL1 << ","
      << candidateHigh.cumulativeConductivityClosureClampCorrectionL1 << ","
      << candidateLow.relativeChargeBudgetResidual << ","
      << candidateHigh.relativeChargeBudgetResidual << ","
      << candidateLow.alphaMassDrift << "," << candidateHigh.alphaMassDrift
      << "," << candidateLow.maxDiv << "," << candidateHigh.maxDiv << ","
      << candidateMorphologyError << "," << candidateHighAsymmetry << ","
      << status << "\n";
}

static void writeConservativeSurfaceChargeClosureDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& candidateLow,
    const electrospray::CandidoConeJetSmokeReport3D& candidateHigh,
    double minAreaDi2) {
  const CandidoParetoAxialStats baseLowStats =
      candidoParetoAxialStats(baselineLow, minAreaDi2);
  const CandidoParetoAxialStats baseHighStats =
      candidoParetoAxialStats(baselineHigh, minAreaDi2);
  const CandidoParetoAxialStats candLowStats =
      candidoParetoAxialStats(candidateLow, minAreaDi2);
  const CandidoParetoAxialStats candHighStats =
      candidoParetoAxialStats(candidateHigh, minAreaDi2);
  const bool baseComparable =
      baseLowStats.developedSamples > 0 && baseHighStats.developedSamples > 0;
  const bool candComparable =
      candLowStats.developedSamples > 0 && candHighStats.developedSamples > 0;
  const double baselineAxialRatio =
      baseComparable ? candidoRatio(baseHighStats.meanAlpha05Convective,
                                    baseLowStats.meanAlpha05Convective)
                     : std::numeric_limits<double>::infinity();
  const double candidateAxialRatio =
      candComparable ? candidoRatio(candHighStats.meanAlpha05Convective,
                                    candLowStats.meanAlpha05Convective)
                     : std::numeric_limits<double>::infinity();
  const double candidateChargeRatio =
      candComparable ? candidoRatio(candHighStats.meanAbsCharge,
                                    candLowStats.meanAbsCharge)
                     : std::numeric_limits<double>::infinity();
  const double candidateVelocityRatio =
      candComparable ? candidoRatio(candHighStats.meanAbsUy,
                                    candLowStats.meanAbsUy)
                     : std::numeric_limits<double>::infinity();
  const double candidateLowElectricSource =
      candidoTailMeanAlpha05ElectricSource(candidateLow, minAreaDi2);
  const double candidateHighElectricSource =
      candidoTailMeanAlpha05ElectricSource(candidateHigh, minAreaDi2);
  const double candidateElectricSourceRatio =
      candComparable ? candidoRatio(candidateHighElectricSource,
                                    candidateLowElectricSource)
                     : std::numeric_limits<double>::infinity();
  const double candidateMorphologyError =
      candidoMaxMorphologyError04_07(candidateLow);
  const double candidateHighAsymmetry = candidoMaxRadialAsymmetry(candidateHigh);
  const bool chargeBudgetOk =
      std::abs(candidateLow.relativeChargeBudgetResidual) <= 1e-6 &&
      std::abs(candidateHigh.relativeChargeBudgetResidual) <= 1e-6;
  const bool numericalQuality =
      candidateLow.alphaMassDrift <= 1e-3 && candidateHigh.alphaMassDrift <= 1e-3 &&
      candidateLow.maxDiv <= 1e-7 && candidateHigh.maxDiv <= 1e-7 &&
      candidateLow.maxImplicitOhmicChargeResidual <= 1e-7 &&
      candidateHigh.maxImplicitOhmicChargeResidual <= 1e-7 && chargeBudgetOk;
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (candComparable) {
    if (!numericalQuality) {
      status = "DOWNGRADED_CONSERVATIVE_SURFACE_CHARGE_BUDGET_OR_NUMERICS";
    } else if (candidateAxialRatio <= 2.0 && candidateMorphologyError <= 10.0 &&
               candidateHighAsymmetry >= 0.05) {
      status = "APPROXIMATE_CONSERVATIVE_SURFACE_CHARGE_ALL_GUARDS_GREEN";
    } else if (candidateAxialRatio < baselineAxialRatio) {
      status = "APPROXIMATE_CONSERVATIVE_SURFACE_CHARGE_REDUCES_CURRENT_SENSITIVITY";
    } else {
      status =
          "DOWNGRADED_CONSERVATIVE_SURFACE_CHARGE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY";
    }
  }
  csv << caseName << "," << baselineLow.targetCaE << "," << baselineHigh.targetCaE
      << "," << minAreaDi2 << "," << baseLowStats.developedSamples << ","
      << baseHighStats.developedSamples << "," << candLowStats.developedSamples
      << "," << candHighStats.developedSamples << "," << baselineAxialRatio
      << "," << candidateAxialRatio << "," << candidateChargeRatio << ","
      << candidateVelocityRatio << "," << candidateElectricSourceRatio << ","
      << candLowStats.meanAlpha05Convective << ","
      << candHighStats.meanAlpha05Convective << ","
      << candLowStats.meanAbsCharge << "," << candHighStats.meanAbsCharge
      << "," << candLowStats.meanAbsUy << "," << candHighStats.meanAbsUy
      << "," << candidateLowElectricSource << ","
      << candidateHighElectricSource << ","
      << candidateLow.maxImplicitOhmicChargeResidual << ","
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
      << candidateMorphologyError << "," << candidateHighAsymmetry << ","
      << status << "\n";
}

static void writeAxialTotalCurrentClosureDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  struct TotalStats {
    int tailSamples = 0;
    int developedSamples = 0;
    double meanArea = 0.0;
    double meanAlpha05Convective = 0.0;
    double meanAlpha05Conductive = 0.0;
    double meanAlpha05Total = 0.0;
    double meanAllPhaseTotal = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    TotalStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      ++s.tailSamples;
      if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanArea += h.developedJetAlpha05AreaDi2;
      s.meanAlpha05Convective += std::abs(h.developedJetAlpha05ConvectiveCurrent);
      s.meanAlpha05Conductive += std::abs(h.developedJetAlpha05ConductiveCurrent);
      s.meanAlpha05Total += std::abs(h.developedJetAlpha05TotalCurrent);
      s.meanAllPhaseTotal += std::abs(h.developedJetTotalCurrent);
    }
    if (s.developedSamples > 0) {
      s.meanArea /= s.developedSamples;
      s.meanAlpha05Convective /= s.developedSamples;
      s.meanAlpha05Conductive /= s.developedSamples;
      s.meanAlpha05Total /= s.developedSamples;
      s.meanAllPhaseTotal /= s.developedSamples;
    }
    return s;
  };
  const TotalStats lowStats = stats(low);
  const TotalStats highStats = stats(high);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  auto ratio = [](double highValue, double lowValue) {
    return highValue / std::max(lowValue, 1e-30);
  };
  const double convectiveRatio =
      comparable ? ratio(highStats.meanAlpha05Convective,
                         lowStats.meanAlpha05Convective)
                 : std::numeric_limits<double>::infinity();
  const double conductiveRatio =
      comparable ? ratio(highStats.meanAlpha05Conductive,
                         lowStats.meanAlpha05Conductive)
                 : std::numeric_limits<double>::infinity();
  const double alpha05TotalRatio =
      comparable ? ratio(highStats.meanAlpha05Total, lowStats.meanAlpha05Total)
                 : std::numeric_limits<double>::infinity();
  const double allPhaseTotalRatio =
      comparable ? ratio(highStats.meanAllPhaseTotal, lowStats.meanAllPhaseTotal)
                 : std::numeric_limits<double>::infinity();
  const double lowConductiveShare =
      comparable ? lowStats.meanAlpha05Conductive /
                       std::max(lowStats.meanAlpha05Total, 1e-30)
                 : std::numeric_limits<double>::infinity();
  const double highConductiveShare =
      comparable ? highStats.meanAlpha05Conductive /
                       std::max(highStats.meanAlpha05Total, 1e-30)
                 : std::numeric_limits<double>::infinity();
  std::string dominant = "blocked";
  if (comparable) {
    dominant = std::abs(alpha05TotalRatio - convectiveRatio) <
                       std::abs(alpha05TotalRatio - conductiveRatio)
                   ? "convective"
                   : "conductive";
  }
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    status = alpha05TotalRatio <= 2.0
                 ? "APPROXIMATE_AXIAL_TOTAL_CURRENT_WEAK_SENSITIVITY"
                 : "DOWNGRADED_AXIAL_TOTAL_CURRENT_" + dominant + "_DOMINATED";
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << minAreaDi2 << "," << lowStats.tailSamples << ","
      << highStats.tailSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << lowStats.meanAlpha05Convective << ","
      << highStats.meanAlpha05Convective << "," << convectiveRatio << ","
      << lowStats.meanAlpha05Conductive << ","
      << highStats.meanAlpha05Conductive << "," << conductiveRatio << ","
      << lowStats.meanAlpha05Total << "," << highStats.meanAlpha05Total << ","
      << alpha05TotalRatio << "," << lowStats.meanAllPhaseTotal << ","
      << highStats.meanAllPhaseTotal << "," << allPhaseTotalRatio << ","
      << lowConductiveShare << "," << highConductiveShare << ","
      << dominant << "," << status << "\n";
}

static void writeAxialCurrentThresholdSweepRows(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high) {
  const std::vector<double> thresholds = {0.0, 1e-8, 1e-6, 1e-4, 1e-3,
                                          1e-2, 0.1, 1.0, 2.0, 5.0};
  for (double minAreaDi2 : thresholds) {
    struct Stats {
      int tailSamples = 0;
      int developedSamples = 0;
      double meanArea = 0.0;
      double meanCurrent = 0.0;
    };
    auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
      Stats s;
      const size_t tailStart = r.history.size() / 2;
      for (size_t i = tailStart; i < r.history.size(); ++i) {
        const auto& h = r.history[i];
        ++s.tailSamples;
        if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
        ++s.developedSamples;
        s.meanArea += h.developedJetAlpha05AreaDi2;
        s.meanCurrent += std::abs(h.developedJetAlpha05ConvectiveCurrent);
      }
      if (s.developedSamples > 0) {
        s.meanArea /= s.developedSamples;
        s.meanCurrent /= s.developedSamples;
      }
      return s;
    };
    const Stats lowStats = stats(low);
    const Stats highStats = stats(high);
    const bool comparable =
        lowStats.developedSamples > 0 && highStats.developedSamples > 0;
    const double ratio =
        comparable ? highStats.meanCurrent / std::max(lowStats.meanCurrent, 1e-30)
                   : std::numeric_limits<double>::infinity();
    const double lowDevelopedFraction =
        lowStats.tailSamples > 0
            ? static_cast<double>(lowStats.developedSamples) /
                  static_cast<double>(lowStats.tailSamples)
            : 0.0;
    const double highDevelopedFraction =
        highStats.tailSamples > 0
            ? static_cast<double>(highStats.developedSamples) /
                  static_cast<double>(highStats.tailSamples)
            : 0.0;
    const std::string status =
        !comparable
            ? "BLOCKED_NO_COMPARABLE_AXIAL_DEVELOPED_WINDOW"
            : (ratio <= 2.0 ? "APPROXIMATE_WEAK_AVERAGE_VOLTAGE_SENSITIVITY"
                            : "DOWNGRADED_THRESHOLD_SWEEP_TOO_VOLTAGE_SENSITIVE");
    csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
        << minAreaDi2 << "," << lowStats.tailSamples << ","
        << highStats.tailSamples << "," << lowStats.developedSamples << ","
        << highStats.developedSamples << "," << lowDevelopedFraction << ","
        << highDevelopedFraction << "," << lowStats.meanArea << ","
        << highStats.meanArea << "," << lowStats.meanCurrent << ","
        << highStats.meanCurrent << "," << ratio << "," << status << "\n";
  }
}

static void writeElectricDriveScalingDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& testedLow,
    const electrospray::CandidoConeJetSmokeReport3D& testedHigh,
    double baselineExponent,
    double testedExponent,
    double minAreaDi2) {
  struct AxialStats {
    int developedSamples = 0;
    double meanArea = 0.0;
    double meanCurrent = 0.0;
    double meanAbsCharge = 0.0;
    double meanAbsUy = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    AxialStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanArea += h.developedJetAlpha05AreaDi2;
      s.meanCurrent += std::abs(h.developedJetAlpha05ConvectiveCurrent);
      s.meanAbsCharge += h.developedJetMeanAlpha05AbsCharge;
      s.meanAbsUy += h.developedJetMeanAlpha05AbsUy;
    }
    if (s.developedSamples > 0) {
      s.meanArea /= s.developedSamples;
      s.meanCurrent /= s.developedSamples;
      s.meanAbsCharge /= s.developedSamples;
      s.meanAbsUy /= s.developedSamples;
    }
    return s;
  };
  const AxialStats baseLowStats = stats(baselineLow);
  const AxialStats baseHighStats = stats(baselineHigh);
  const AxialStats testLowStats = stats(testedLow);
  const AxialStats testHighStats = stats(testedHigh);
  auto ratio = [](double high, double low) {
    return high / std::max(low, 1e-30);
  };
  const double baselineTailRatio =
      ratio(candidoMeanTailCurrentForObservable(baselineHigh),
            candidoMeanTailCurrentForObservable(baselineLow));
  const double testedTailRatio =
      ratio(candidoMeanTailCurrentForObservable(testedHigh),
            candidoMeanTailCurrentForObservable(testedLow));
  const bool baseComparable =
      baseLowStats.developedSamples > 0 && baseHighStats.developedSamples > 0;
  const bool testComparable =
      testLowStats.developedSamples > 0 && testHighStats.developedSamples > 0;
  const double baselineAxialRatio =
      baseComparable ? ratio(baseHighStats.meanCurrent, baseLowStats.meanCurrent)
                     : std::numeric_limits<double>::infinity();
  const double testedAxialRatio =
      testComparable ? ratio(testHighStats.meanCurrent, testLowStats.meanCurrent)
                     : std::numeric_limits<double>::infinity();
  const double baselineVelocityRatio =
      baseComparable ? ratio(baseHighStats.meanAbsUy, baseLowStats.meanAbsUy)
                     : std::numeric_limits<double>::infinity();
  const double testedVelocityRatio =
      testComparable ? ratio(testHighStats.meanAbsUy, testLowStats.meanAbsUy)
                     : std::numeric_limits<double>::infinity();
  const double baselineChargeRatio =
      baseComparable ? ratio(baseHighStats.meanAbsCharge, baseLowStats.meanAbsCharge)
                     : std::numeric_limits<double>::infinity();
  const double testedChargeRatio =
      testComparable ? ratio(testHighStats.meanAbsCharge, testLowStats.meanAbsCharge)
                     : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (testComparable) {
    status = testedAxialRatio <= 2.0
                 ? "DIAGNOSTIC_CA_INDEPENDENT_DRIVE_RECOVERS_WEAK_CURRENT_BAR"
                 : (testedVelocityRatio < baselineVelocityRatio
                        ? "APPROXIMATE_CA_INDEPENDENT_DRIVE_REDUCES_VELOCITY_BIAS"
                        : "DOWNGRADED_CA_INDEPENDENT_DRIVE_NOT_VELOCITY_FIX");
  }
  csv << caseName << "," << baselineLow.targetCaE << "," << baselineHigh.targetCaE << ","
      << baselineExponent << "," << testedExponent << ","
      << baselineTailRatio << "," << testedTailRatio << ","
      << baseLowStats.developedSamples << "," << baseHighStats.developedSamples << ","
      << testLowStats.developedSamples << "," << testHighStats.developedSamples << ","
      << baselineAxialRatio << "," << testedAxialRatio << ","
      << baselineVelocityRatio << "," << testedVelocityRatio << ","
      << baselineChargeRatio << "," << testedChargeRatio << ","
      << testedLow.alphaMassDrift << "," << testedHigh.alphaMassDrift << ","
      << testedLow.maxDiv << "," << testedHigh.maxDiv << ","
      << status << "\n";
}

static void writeHybridMaxwellLongWindowDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& hybridLow,
    const electrospray::CandidoConeJetSmokeReport3D& hybridHigh,
    double minAreaDi2) {
  struct AxialStats {
    int developedSamples = 0;
    double meanCurrent = 0.0;
    double meanAbsCharge = 0.0;
    double meanAbsUy = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
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
      s.meanCurrent /= s.developedSamples;
      s.meanAbsCharge /= s.developedSamples;
      s.meanAbsUy /= s.developedSamples;
    }
    return s;
  };
  auto ratio = [](double high, double low) {
    return high / std::max(low, 1e-30);
  };
  const AxialStats baseLowStats = stats(baselineLow);
  const AxialStats baseHighStats = stats(baselineHigh);
  const AxialStats hybridLowStats = stats(hybridLow);
  const AxialStats hybridHighStats = stats(hybridHigh);
  const bool baseComparable =
      baseLowStats.developedSamples > 0 && baseHighStats.developedSamples > 0;
  const bool hybridComparable =
      hybridLowStats.developedSamples > 0 && hybridHighStats.developedSamples > 0;
  const double baselineTailRatio =
      ratio(candidoMeanTailCurrentForObservable(baselineHigh),
            candidoMeanTailCurrentForObservable(baselineLow));
  const double hybridTailRatio =
      ratio(candidoMeanTailCurrentForObservable(hybridHigh),
            candidoMeanTailCurrentForObservable(hybridLow));
  const double baselineAxialRatio =
      baseComparable ? ratio(baseHighStats.meanCurrent, baseLowStats.meanCurrent)
                     : std::numeric_limits<double>::infinity();
  const double hybridAxialRatio =
      hybridComparable ? ratio(hybridHighStats.meanCurrent, hybridLowStats.meanCurrent)
                       : std::numeric_limits<double>::infinity();
  const double baselineVelocityRatio =
      baseComparable ? ratio(baseHighStats.meanAbsUy, baseLowStats.meanAbsUy)
                     : std::numeric_limits<double>::infinity();
  const double hybridVelocityRatio =
      hybridComparable ? ratio(hybridHighStats.meanAbsUy, hybridLowStats.meanAbsUy)
                       : std::numeric_limits<double>::infinity();
  const double baselineChargeRatio =
      baseComparable ? ratio(baseHighStats.meanAbsCharge, baseLowStats.meanAbsCharge)
                     : std::numeric_limits<double>::infinity();
  const double hybridChargeRatio =
      hybridComparable ? ratio(hybridHighStats.meanAbsCharge, hybridLowStats.meanAbsCharge)
                       : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (hybridComparable) {
    status = hybridAxialRatio <= 2.0
                 ? "DIAGNOSTIC_HYBRID_MAXWELL_RECOVERS_WEAK_CURRENT_BAR"
                 : (hybridVelocityRatio < baselineVelocityRatio
                        ? "APPROXIMATE_HYBRID_MAXWELL_REDUCES_VELOCITY_BIAS"
                        : "DOWNGRADED_HYBRID_MAXWELL_NOT_CURRENT_FIX");
  }
  csv << baselineLow.targetCaE << "," << baselineHigh.targetCaE << ","
      << baselineTailRatio << "," << hybridTailRatio << ","
      << baseLowStats.developedSamples << "," << baseHighStats.developedSamples << ","
      << hybridLowStats.developedSamples << "," << hybridHighStats.developedSamples << ","
      << baselineAxialRatio << "," << hybridAxialRatio << ","
      << baselineVelocityRatio << "," << hybridVelocityRatio << ","
      << baselineChargeRatio << "," << hybridChargeRatio << ","
      << baselineLow.maxElectricForce << "," << baselineHigh.maxElectricForce << ","
      << hybridLow.maxElectricForce << "," << hybridHigh.maxElectricForce << ","
      << hybridLow.alphaMassDrift << "," << hybridHigh.alphaMassDrift << ","
      << hybridLow.maxDiv << "," << hybridHigh.maxDiv << ","
      << status << "\n";
}

static void writeBoundedVectorMaxwellLongWindowDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& boundedLow,
    const electrospray::CandidoConeJetSmokeReport3D& boundedHigh,
    double minAreaDi2) {
  struct AxialStats {
    int developedSamples = 0;
    double meanCurrent = 0.0;
    double meanAbsCharge = 0.0;
    double meanAbsUy = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
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
      s.meanCurrent /= s.developedSamples;
      s.meanAbsCharge /= s.developedSamples;
      s.meanAbsUy /= s.developedSamples;
    }
    return s;
  };
  auto ratio = [](double high, double low) {
    return high / std::max(low, 1e-30);
  };
  const AxialStats baseLowStats = stats(baselineLow);
  const AxialStats baseHighStats = stats(baselineHigh);
  const AxialStats boundedLowStats = stats(boundedLow);
  const AxialStats boundedHighStats = stats(boundedHigh);
  const bool baseComparable =
      baseLowStats.developedSamples > 0 && baseHighStats.developedSamples > 0;
  const bool boundedComparable =
      boundedLowStats.developedSamples > 0 && boundedHighStats.developedSamples > 0;
  const double baselineTailRatio =
      ratio(candidoMeanTailCurrentForObservable(baselineHigh),
            candidoMeanTailCurrentForObservable(baselineLow));
  const double boundedTailRatio =
      ratio(candidoMeanTailCurrentForObservable(boundedHigh),
            candidoMeanTailCurrentForObservable(boundedLow));
  const double baselineAxialRatio =
      baseComparable ? ratio(baseHighStats.meanCurrent, baseLowStats.meanCurrent)
                     : std::numeric_limits<double>::infinity();
  const double boundedAxialRatio =
      boundedComparable ? ratio(boundedHighStats.meanCurrent, boundedLowStats.meanCurrent)
                        : std::numeric_limits<double>::infinity();
  const double baselineVelocityRatio =
      baseComparable ? ratio(baseHighStats.meanAbsUy, baseLowStats.meanAbsUy)
                     : std::numeric_limits<double>::infinity();
  const double boundedVelocityRatio =
      boundedComparable ? ratio(boundedHighStats.meanAbsUy, boundedLowStats.meanAbsUy)
                        : std::numeric_limits<double>::infinity();
  const double baselineChargeRatio =
      baseComparable ? ratio(baseHighStats.meanAbsCharge, baseLowStats.meanAbsCharge)
                     : std::numeric_limits<double>::infinity();
  const double boundedChargeRatio =
      boundedComparable ? ratio(boundedHighStats.meanAbsCharge, boundedLowStats.meanAbsCharge)
                        : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (boundedComparable) {
    status = boundedAxialRatio <= 2.0
                 ? "DIAGNOSTIC_BOUNDED_VECTOR_RECOVERS_WEAK_CURRENT_BAR"
                 : (boundedVelocityRatio < baselineVelocityRatio
                        ? "APPROXIMATE_BOUNDED_VECTOR_REDUCES_VELOCITY_BIAS"
                        : "DOWNGRADED_BOUNDED_VECTOR_NOT_CURRENT_FIX");
  }
  csv << baselineLow.targetCaE << "," << baselineHigh.targetCaE << ","
      << baselineTailRatio << "," << boundedTailRatio << ","
      << baseLowStats.developedSamples << "," << baseHighStats.developedSamples << ","
      << boundedLowStats.developedSamples << "," << boundedHighStats.developedSamples
      << "," << baselineAxialRatio << "," << boundedAxialRatio << ","
      << baselineVelocityRatio << "," << boundedVelocityRatio << ","
      << baselineChargeRatio << "," << boundedChargeRatio << ","
      << baselineLow.maxElectricForce << "," << baselineHigh.maxElectricForce << ","
      << boundedLow.maxElectricForce << "," << boundedHigh.maxElectricForce << ","
      << boundedLow.alphaMassDrift << "," << boundedHigh.alphaMassDrift << ","
      << boundedLow.maxDiv << "," << boundedHigh.maxDiv << "," << status << "\n";
}

static void writeCaIndependentCurrentResolutionRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  struct AxialStats {
    int developedSamples = 0;
    double meanArea = 0.0;
    double meanCurrent = 0.0;
    double meanAbsCharge = 0.0;
    double meanAbsUy = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
    AxialStats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = tailStart; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
      ++s.developedSamples;
      s.meanArea += h.developedJetAlpha05AreaDi2;
      s.meanCurrent += std::abs(h.developedJetAlpha05ConvectiveCurrent);
      s.meanAbsCharge += h.developedJetMeanAlpha05AbsCharge;
      s.meanAbsUy += h.developedJetMeanAlpha05AbsUy;
    }
    if (s.developedSamples > 0) {
      s.meanArea /= s.developedSamples;
      s.meanCurrent /= s.developedSamples;
      s.meanAbsCharge /= s.developedSamples;
      s.meanAbsUy /= s.developedSamples;
    }
    return s;
  };
  const AxialStats lowStats = stats(low);
  const AxialStats highStats = stats(high);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  auto ratio = [](double highValue, double lowValue) {
    return highValue / std::max(lowValue, 1e-30);
  };
  const double tailRatio =
      ratio(candidoMeanTailCurrentForObservable(high),
            candidoMeanTailCurrentForObservable(low));
  const double axialRatio =
      comparable ? ratio(highStats.meanCurrent, lowStats.meanCurrent)
                 : std::numeric_limits<double>::infinity();
  const double velocityRatio =
      comparable ? ratio(highStats.meanAbsUy, lowStats.meanAbsUy)
                 : std::numeric_limits<double>::infinity();
  const double chargeRatio =
      comparable ? ratio(highStats.meanAbsCharge, lowStats.meanAbsCharge)
                 : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    const bool enoughSamples =
        lowStats.developedSamples >= 3 && highStats.developedSamples >= 3;
    status = !enoughSamples
                 ? "BLOCKED_INSUFFICIENT_DEVELOPED_SAMPLES_FOR_GRID_TREND"
                 : (axialRatio <= 2.0
                        ? "APPROXIMATE_GRID_POINT_WITHIN_WEAK_CURRENT_BAR"
                        : "DOWNGRADED_GRID_POINT_ABOVE_WEAK_CURRENT_BAR");
  }
  csv << caseName << "," << opt.nx << "," << opt.ny << "," << opt.nz << ","
      << low.cells << "," << high.cells << "," << low.dt << "," << high.dt << ","
      << lowStats.developedSamples << "," << highStats.developedSamples << ","
      << lowStats.meanArea << "," << highStats.meanArea << ","
      << tailRatio << "," << axialRatio << "," << velocityRatio << ","
      << chargeRatio << "," << low.alphaMassDrift << "," << high.alphaMassDrift << ","
      << low.maxDiv << "," << high.maxDiv << "," << status << "\n";
}

static void writeCurrentBlowupDiagnosticRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const auto* peak = &r.history.front();
  const auto* firstOutOfScale =
      static_cast<const electrospray::CandidoConeJetHistorySample3D*>(nullptr);
  const double reference = candidoGananCalvoCurrentScale(setup);
  const double outOfScaleThreshold = 10.0 * reference;
  for (const auto& h : r.history) {
    if (std::abs(h.convectiveCurrent) > std::abs(peak->convectiveCurrent)) peak = &h;
    if (!firstOutOfScale && std::abs(h.convectiveCurrent) > outOfScaleThreshold) {
      firstOutOfScale = &h;
    }
  }
  const double peakMs = peak->time * r.hydrodynamicTimeScale * 1.0e3;
  const double firstOutMs =
      firstOutOfScale ? firstOutOfScale->time * r.hydrodynamicTimeScale * 1.0e3
                      : std::numeric_limits<double>::quiet_NaN();
  const double peakRatio = std::abs(peak->convectiveCurrent) / std::max(reference, 1e-30);
  const double firstOutRatio =
      firstOutOfScale ? std::abs(firstOutOfScale->convectiveCurrent) / std::max(reference, 1e-30)
                      : std::numeric_limits<double>::quiet_NaN();
  const double peakMassDrift =
      (peak->mass - r.initialMass) / std::max(std::abs(r.initialMass), 1e-30);
  const std::string status =
      firstOutOfScale ? "CURRENT_BLOWUP_AFTER_LONG_WINDOW_INSTABILITY"
                      : "CURRENT_REMAINS_WITHIN_ORDER_OF_MAGNITUDE";
  csv << name << "," << r.targetCaE << "," << reference << ","
      << peak->step << "," << peakMs << "," << peak->convectiveCurrent << ","
      << peakRatio << "," << peak->maxVelocity << "," << peak->mass << ","
      << peakMassDrift << "," << peak->minAlpha << "," << peak->maxAlpha << ","
      << peak->radialAsymmetry << "," << peak->tipY << ","
      << (firstOutOfScale ? firstOutOfScale->step : -1) << "," << firstOutMs << ","
      << firstOutRatio << "," << r.maxCharge << "," << r.minCharge << ","
      << status << "\n";
}

static void writeTomarConductingLongWindowDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& tomarLow,
    const electrospray::CandidoConeJetSmokeReport3D& tomarHigh,
    double minAreaDi2) {
  struct AxialStats {
    int developedSamples = 0;
    double meanCurrent = 0.0;
    double meanAbsCharge = 0.0;
    double meanAbsUy = 0.0;
  };
  auto stats = [minAreaDi2](const electrospray::CandidoConeJetSmokeReport3D& r) {
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
      s.meanCurrent /= s.developedSamples;
      s.meanAbsCharge /= s.developedSamples;
      s.meanAbsUy /= s.developedSamples;
    }
    return s;
  };
  auto ratio = [](double high, double low) {
    return high / std::max(low, 1e-30);
  };
  const AxialStats baseLowStats = stats(baselineLow);
  const AxialStats baseHighStats = stats(baselineHigh);
  const AxialStats tomarLowStats = stats(tomarLow);
  const AxialStats tomarHighStats = stats(tomarHigh);
  const bool tomarComparable =
      tomarLowStats.developedSamples > 0 && tomarHighStats.developedSamples > 0;
  const double baselineTailRatio =
      ratio(candidoMeanTailCurrentForObservable(baselineHigh),
            candidoMeanTailCurrentForObservable(baselineLow));
  const double tomarTailRatio =
      ratio(candidoMeanTailCurrentForObservable(tomarHigh),
            candidoMeanTailCurrentForObservable(tomarLow));
  const double baselineAxialRatio =
      ratio(baseHighStats.meanCurrent, baseLowStats.meanCurrent);
  const double tomarAxialRatio =
      ratio(tomarHighStats.meanCurrent, tomarLowStats.meanCurrent);
  const double tomarVelocityRatio =
      ratio(tomarHighStats.meanAbsUy, tomarLowStats.meanAbsUy);
  const double tomarChargeRatio =
      ratio(tomarHighStats.meanAbsCharge, tomarLowStats.meanAbsCharge);
  const bool finite =
      std::isfinite(tomarTailRatio) && std::isfinite(tomarAxialRatio) &&
      std::isfinite(tomarVelocityRatio) && std::isfinite(tomarChargeRatio) &&
      std::isfinite(tomarLow.maxElectricForce) &&
      std::isfinite(tomarHigh.maxElectricForce);
  const std::string status =
      !finite ? "DOWNGRADED_TOMAR_PRODUCTION_NONFINITE"
              : (!tomarComparable
                     ? "DOWNGRADED_TOMAR_NO_DEVELOPED_ALPHA05_WINDOW"
                     : (tomarAxialRatio > 1.0
                            ? "APPROXIMATE_TOMAR_PRODUCTION_FINITE_TREND_ONLY"
                            : "DOWNGRADED_TOMAR_PRODUCTION_CURRENT_TREND"));
  csv << baselineLow.targetCaE << "," << baselineHigh.targetCaE << ","
      << baselineTailRatio << "," << tomarTailRatio << ","
      << baseLowStats.developedSamples << "," << baseHighStats.developedSamples
      << "," << tomarLowStats.developedSamples << ","
      << tomarHighStats.developedSamples << "," << baselineAxialRatio << ","
      << tomarAxialRatio << "," << tomarVelocityRatio << ","
      << tomarChargeRatio << "," << tomarLow.maxElectricForce << ","
      << tomarHigh.maxElectricForce << "," << tomarLow.alphaMassDrift << ","
      << tomarHigh.alphaMassDrift << "," << tomarLow.maxDiv << ","
      << tomarHigh.maxDiv << "," << status << "\n";
}

static void writeLongWindowMassBudgetRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double openDomainGrowth =
      (r.finalMass - r.initialMass) / std::max(std::abs(r.initialMass), 1e-30);
  const double signedBoundaryGrowth =
      -r.cumulativeBoundaryLiquidFlux / std::max(std::abs(r.initialMass), 1e-30);
  const std::string status =
      r.relativeMassBudgetResidual <= 1e-10 ? "OPEN_BOUNDARY_BUDGET_CLOSED"
                                            : "DOWNGRADED_BUDGET_RESIDUAL";
  csv << name << "," << r.targetCaE << "," << r.steps << ","
      << r.initialMass << "," << r.finalMass << ","
      << r.cumulativeBoundaryLiquidFlux << ","
      << r.cumulativeBoundaryLiquidInflow << ","
      << r.cumulativeBoundaryLiquidOutflow << ","
      << r.massBudgetExpectedFinal << "," << r.massBudgetResidual << ","
      << r.relativeMassBudgetResidual << "," << openDomainGrowth << ","
      << signedBoundaryGrowth << "," << r.alphaMassDrift << ","
      << r.maxConvectiveCurrent << "," << r.maxVelocity << ","
      << status << "\n";
}

static void writeLongWindowChargeBudgetRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const bool finiteBudget =
      std::isfinite(r.initialIntegratedCharge) && std::isfinite(r.finalIntegratedCharge) &&
      std::isfinite(r.cumulativeBoundaryChargeFlux) &&
      std::isfinite(r.chargeBudgetResidual) &&
      std::isfinite(r.cumulativeChargeClampCorrectionL1);
  const std::string status =
      !finiteBudget ? "DOWNGRADED_NONFINITE_CHARGE_BUDGET"
                    : (r.maxChargeClampedCells > 0 || r.cumulativeChargeClampCorrectionL1 > 0.0)
                          ? "CHARGE_BUDGET_QUANTIFIED_WITH_CLAMPING"
                          : "CHARGE_BUDGET_QUANTIFIED_NO_CLAMPING";
  csv << name << "," << r.targetCaE << "," << r.steps << ","
      << r.initialIntegratedCharge << "," << r.finalIntegratedCharge << ","
      << r.cumulativeBoundaryChargeFlux << ","
      << r.cumulativeConductiveBoundaryChargeFlux << ","
      << r.cumulativeChargeRelaxationSink << ","
      << r.chargeBudgetExpectedFinal << "," << r.chargeBudgetResidual << ","
      << r.relativeChargeBudgetResidual << ","
      << r.cumulativeChargeClampCorrectionL1 << ","
      << r.maxChargeRedistributionResidual << ","
      << r.maxChargeClampedCells << "," << r.maxUnclampedAbsCharge << ","
      << r.maxCharge << "," << r.minCharge << ","
      << r.maxConductiveCurrent << "," << r.maxConvectiveCurrent << ","
      << r.maxVelocity << "," << status << "\n";
}

static void writeChargeSubcyclingDiagnosticRow(
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
      subcycled.maxConvectiveCurrent / std::max(baseline.maxConvectiveCurrent, 1e-30);
  const std::string status =
      (clampRatio < 1.0 && residualRatio < 1.0)
          ? "SUBCYCLING_REDUCES_CHARGE_CLIPPING"
          : "DOWNGRADED_SUBCYCLING_DOES_NOT_REDUCE_CLIPPING";
  csv << baseline.targetCaE << "," << subcycles << ","
      << baseline.relativeChargeBudgetResidual << ","
      << subcycled.relativeChargeBudgetResidual << ","
      << residualRatio << ","
      << baseline.cumulativeChargeClampCorrectionL1 << ","
      << subcycled.cumulativeChargeClampCorrectionL1 << ","
      << clampRatio << ","
      << baseline.maxChargeClampedCells << ","
      << subcycled.maxChargeClampedCells << ","
      << baseline.maxUnclampedAbsCharge << ","
      << subcycled.maxUnclampedAbsCharge << ","
      << baseline.maxConvectiveCurrent << ","
      << subcycled.maxConvectiveCurrent << ","
      << currentRatio << ","
      << baseline.maxVelocity << ","
      << subcycled.maxVelocity << ","
      << status << "\n";
}

static void writeChargeConservativeBoundingDiagnosticRow(
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
      << bounded.relativeChargeBudgetResidual << ","
      << residualRatio << ","
      << baseline.cumulativeChargeClampCorrectionL1 << ","
      << bounded.cumulativeChargeClampCorrectionL1 << ","
      << clampRatio << ","
      << bounded.maxChargeRedistributionResidual << ","
      << baseline.maxChargeClampedCells << ","
      << bounded.maxChargeClampedCells << ","
      << baseline.maxUnclampedAbsCharge << ","
      << bounded.maxUnclampedAbsCharge << ","
      << baseline.maxConvectiveCurrent << ","
      << bounded.maxConvectiveCurrent << ","
      << currentRatio << ","
      << baseline.maxVelocity << ","
      << bounded.maxVelocity << ","
      << status << "\n";
}

static void writeChargeLimitSensitivityRow(
    std::ofstream& csv,
    double chargeLimitBase,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  csv << r.targetCaE << "," << chargeLimitBase << ","
      << r.relativeChargeBudgetResidual << ","
      << r.cumulativeChargeClampCorrectionL1 << ","
      << r.maxChargeClampedCells << ","
      << r.maxUnclampedAbsCharge << ","
      << r.maxCharge << "," << r.minCharge << ","
      << r.maxConvectiveCurrent << "," << r.maxVelocity << ","
      << r.alphaMassDrift << "," << r.maxDiv << ","
      << "DIAGNOSTIC_Q_LIMIT_SENSITIVITY\n";
}

static void writeChargeScaleAuditRow(
    std::ofstream& csv,
    const std::string& name,
    double chargeLimitBase,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double rayleigh = electrospray::candidoRayleighChargeScale(setup);
  const double currentScale = candidoGananCalvoCurrentScale(setup);
  const double timeScale = rayleigh / std::max(currentScale, 1e-30);
  const double qLimit = chargeLimitBase * std::max(1.0, r.targetCaE / 0.25);
  const double maxIntegratedCharge =
      std::max(std::abs(r.initialIntegratedCharge), std::abs(r.finalIntegratedCharge));
  const std::string status =
      (r.cumulativeChargeClampCorrectionL1 > maxIntegratedCharge + 1e-30)
          ? "DOWNGRADED_CLAMP_DOMINATES_CHARGE_SCALE"
          : "CHARGE_SCALE_DIAGNOSTIC_ONLY";
  csv << name << "," << r.targetCaE << "," << chargeLimitBase << "," << qLimit << ","
      << rayleigh << "," << currentScale << "," << timeScale << ","
      << r.initialIntegratedCharge << "," << r.finalIntegratedCharge << ","
      << maxIntegratedCharge << "," << r.cumulativeChargeClampCorrectionL1 << ","
      << r.cumulativeChargeClampCorrectionL1 / std::max(rayleigh, 1e-30) << ","
      << r.maxUnclampedAbsCharge << "," << r.maxCharge << "," << r.minCharge << ","
      << r.maxConvectiveCurrent << "," << r.maxConvectiveCurrent / std::max(currentScale, 1e-30)
      << "," << status << "\n";
}

static void writeChargeUnitConsistencyRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double rayleigh = electrospray::candidoRayleighChargeScale(setup);
  const double currentScale = candidoGananCalvoCurrentScale(setup);
  const double timeScale = std::max(r.hydrodynamicTimeScale, 1e-30);
  const double maxIntegratedCharge =
      std::max(std::abs(r.initialIntegratedCharge), std::abs(r.finalIntegratedCharge));
  const double qUnitFromIntegratedCharge =
      rayleigh / std::max(maxIntegratedCharge, 1e-30);
  const double qUnitFromCurrent =
      currentScale * timeScale / std::max(r.maxConvectiveCurrent, 1e-30);
  const double lo = std::min(qUnitFromIntegratedCharge, qUnitFromCurrent);
  const double hi = std::max(qUnitFromIntegratedCharge, qUnitFromCurrent);
  const double consistencyRatio = hi / std::max(lo, 1e-300);
  const std::string status =
      (consistencyRatio <= 10.0)
          ? "APPROXIMATE_SINGLE_CHARGE_UNIT"
          : "DOWNGRADED_NO_SINGLE_CHARGE_UNIT_FOR_Q_AND_CURRENT";
  csv << name << "," << r.targetCaE << "," << rayleigh << "," << currentScale << ","
      << timeScale << "," << maxIntegratedCharge << "," << r.maxConvectiveCurrent << ","
      << qUnitFromIntegratedCharge << "," << qUnitFromCurrent << ","
      << consistencyRatio << "," << status << "\n";
}

static void writeNondimChargeScaleAuditRow(
    std::ofstream& csv,
    const std::string& name,
    double chargeLimitBase,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double qLimit = chargeLimitBase * std::max(1.0, r.targetCaE / 0.25);
  const double poissonQ = candidoPoissonChargeScale(setup, r);
  const double poissonI = candidoPoissonCurrentScale(setup, r);
  const double rayleigh = electrospray::candidoRayleighChargeScale(setup);
  const double ganan = candidoGananCalvoCurrentScale(setup);
  const double qLimitPhysical = qLimit * poissonQ;
  const double maxIntegratedPhysical =
      std::max(std::abs(r.initialIntegratedCharge), std::abs(r.finalIntegratedCharge)) *
      poissonQ;
  const double clampPhysical = r.cumulativeChargeClampCorrectionL1 * poissonQ;
  const double currentPhysical = r.maxConvectiveCurrent * poissonI;
  const std::string status =
      (qLimitPhysical <= 10.0 * rayleigh && currentPhysical <= 10.0 * ganan)
          ? "APPROXIMATE_POISSON_SCALE_ORDER_OF_MAGNITUDE"
          : "DOWNGRADED_POISSON_SCALE_STILL_NOT_CALIBRATED";
  csv << name << "," << r.targetCaE << "," << chargeLimitBase << "," << qLimit << ","
      << poissonQ << "," << poissonI << "," << rayleigh << "," << ganan << ","
      << qLimitPhysical << "," << qLimitPhysical / std::max(rayleigh, 1e-300) << ","
      << maxIntegratedPhysical << ","
      << maxIntegratedPhysical / std::max(rayleigh, 1e-300) << ","
      << clampPhysical << "," << clampPhysical / std::max(rayleigh, 1e-300) << ","
      << currentPhysical << "," << currentPhysical / std::max(ganan, 1e-300)
      << "," << status << "\n";
}

static void writeChargeFieldConsistencyRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const std::string status =
      (r.maxPotentialResidual <= 1e-8 &&
       r.maxRelativeGaussLawCellGradientResidual <= 0.1)
          ? "APPROXIMATE_CELL_GRADIENT_E_CONSISTENT"
          : "DOWNGRADED_CELL_GRADIENT_E_NOT_DISCRETE_GAUSS_CONSISTENT";
  csv << name << "," << r.targetCaE << "," << r.maxPotentialResidual << ","
      << r.maxGaussLawCellGradientResidual << ","
      << r.maxRelativeGaussLawCellGradientResidual << ","
      << r.maxCharge << "," << r.minCharge << ","
      << r.maxConvectiveCurrent << "," << r.maxConductiveCurrent << ","
      << r.relativeChargeBudgetResidual << ","
      << r.cumulativeChargeClampCorrectionL1 << "," << status << "\n";
}

static void writeElectricPropertyScalingAuditRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  constexpr double eps0 = 8.8541878128e-12;
  const double physicalLiquidEps = eps0 * setup.liquidRelativePermittivity;
  const double physicalGasEps = eps0 * setup.gasRelativePermittivity;
  const double physicalLiquidTau =
      physicalLiquidEps / std::max(setup.liquidConductivity, 1e-300);
  const double physicalGasTau =
      physicalGasEps / std::max(setup.gasConductivity, 1e-300);
  const double effectiveLiquidConductivity =
      opt.useDimensionalElectricalScaling
          ? electrospray::candidoDimensionlessConductivityFromPhysical(
                setup, setup.liquidConductivity)
          : opt.normalizedLiquidConductivity;
  const double effectiveGasConductivity =
      opt.useDimensionalElectricalScaling
          ? electrospray::candidoDimensionlessConductivityFromPhysical(
                setup, setup.gasConductivity)
          : opt.normalizedGasConductivity;
  const double normalizedLiquidTau =
      setup.liquidRelativePermittivity / std::max(effectiveLiquidConductivity, 1e-300);
  const double normalizedGasTau =
      setup.gasRelativePermittivity / std::max(effectiveGasConductivity, 1e-300);
  const double liquidTauOverHydro =
      physicalLiquidTau / std::max(r.hydrodynamicTimeScale, 1e-300);
  const double gasTauOverHydro =
      physicalGasTau / std::max(r.hydrodynamicTimeScale, 1e-300);
  const double simDtOverNormalizedLiquidTau = r.dt / std::max(normalizedLiquidTau, 1e-300);
  const double physicalDtOverLiquidTau =
      (r.dt * r.hydrodynamicTimeScale) / std::max(physicalLiquidTau, 1e-300);
  const bool normalizedMatchesPhysical =
      std::abs(std::log10(std::max(simDtOverNormalizedLiquidTau, 1e-300) /
                          std::max(physicalDtOverLiquidTau, 1e-300))) <= 1.0;
  csv << name << "," << r.targetCaE << "," << setup.liquidRelativePermittivity << ","
      << setup.liquidConductivity << "," << physicalLiquidTau << ","
      << physicalLiquidTau * 1.0e6 << "," << liquidTauOverHydro << ","
      << setup.gasRelativePermittivity << "," << setup.gasConductivity << ","
      << physicalGasTau << "," << gasTauOverHydro << ","
      << effectiveLiquidConductivity << "," << effectiveGasConductivity << ","
      << normalizedLiquidTau << "," << normalizedGasTau << ","
      << r.dt << "," << r.dt * r.hydrodynamicTimeScale << ","
      << simDtOverNormalizedLiquidTau << "," << physicalDtOverLiquidTau << ","
      << (normalizedMatchesPhysical ? "APPROXIMATE_RELAXATION_SCALE_MATCH"
                                    : "DOWNGRADED_NORMALIZED_CONDUCTIVITY_NOT_DIMENSIONAL")
      << "\n";
}

static void writeChargeRelaxationDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baseline,
    const electrospray::CandidoConeJetSmokeReport3D& relaxed) {
  const double currentRatio =
      relaxed.maxConvectiveCurrent / std::max(baseline.maxConvectiveCurrent, 1e-30);
  const double velocityRatio = relaxed.maxVelocity / std::max(baseline.maxVelocity, 1e-30);
  const std::string status =
      (relaxed.relativeChargeBudgetResidual <= 1e-8 && currentRatio <= 0.1)
          ? "RELAXATION_REDUCES_CURRENT_AND_CLOSES_BUDGET"
          : "DOWNGRADED_RELAXATION_DIAGNOSTIC_ONLY";
  csv << baseline.targetCaE << ","
      << baseline.relativeChargeBudgetResidual << ","
      << relaxed.relativeChargeBudgetResidual << ","
      << baseline.cumulativeChargeClampCorrectionL1 << ","
      << relaxed.cumulativeChargeClampCorrectionL1 << ","
      << relaxed.cumulativeChargeRelaxationSink << ","
      << baseline.maxUnclampedAbsCharge << ","
      << relaxed.maxUnclampedAbsCharge << ","
      << baseline.maxConvectiveCurrent << ","
      << relaxed.maxConvectiveCurrent << ","
      << currentRatio << ","
      << baseline.maxVelocity << ","
      << relaxed.maxVelocity << ","
      << velocityRatio << ","
      << relaxed.alphaMassDrift << ","
      << relaxed.maxDiv << ","
      << status << "\n";
}

static void writeDimensionalElectricalScalingDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baseline,
    const electrospray::CandidoConeJetSmokeReport3D& scaled) {
  const double currentRatio =
      scaled.maxConvectiveCurrent / std::max(baseline.maxConvectiveCurrent, 1e-30);
  const double clampRatio =
      scaled.cumulativeChargeClampCorrectionL1 /
      std::max(baseline.cumulativeChargeClampCorrectionL1, 1e-30);
  const double residualRatio =
      scaled.relativeChargeBudgetResidual /
      std::max(baseline.relativeChargeBudgetResidual, 1e-30);
  const double velocityRatio = scaled.maxVelocity / std::max(baseline.maxVelocity, 1e-30);
  const std::string status =
      (scaled.alphaMassDrift <= 1e-3 && scaled.maxDiv <= 1e-7 &&
       currentRatio <= 1.0 && clampRatio <= 1.0)
          ? "APPROXIMATE_DIMENSIONAL_SCALING_REDUCES_CURRENT"
          : "DOWNGRADED_DIMENSIONAL_SCALING_DIAGNOSTIC_ONLY";
  csv << baseline.targetCaE << "," << baseline.relativeChargeBudgetResidual << ","
      << scaled.relativeChargeBudgetResidual << "," << residualRatio << ","
      << baseline.cumulativeChargeClampCorrectionL1 << ","
      << scaled.cumulativeChargeClampCorrectionL1 << "," << clampRatio << ","
      << baseline.maxUnclampedAbsCharge << "," << scaled.maxUnclampedAbsCharge << ","
      << baseline.maxConvectiveCurrent << "," << scaled.maxConvectiveCurrent << ","
      << currentRatio << "," << baseline.maxVelocity << "," << scaled.maxVelocity << ","
      << velocityRatio << "," << scaled.alphaMassDrift << "," << scaled.maxDiv << ","
      << status << "\n";
}

static void writeElectricRelaxationTimestepDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& limitedLow,
    const electrospray::CandidoConeJetSmokeReport3D& limitedHigh) {
  const double baselineLowTail = candidoMeanTailCurrentForObservable(baselineLow);
  const double baselineHighTail = candidoMeanTailCurrentForObservable(baselineHigh);
  const double limitedLowTail = candidoMeanTailCurrentForObservable(limitedLow);
  const double limitedHighTail = candidoMeanTailCurrentForObservable(limitedHigh);
  const double baselineCurrentRatio =
      baselineHighTail / std::max(baselineLowTail, 1e-30);
  const double limitedCurrentRatio =
      limitedHighTail / std::max(limitedLowTail, 1e-30);
  const double lowDtReduction = limitedLow.dt / std::max(limitedLow.unrestrictedDt, 1e-30);
  const double highDtReduction =
      limitedHigh.dt / std::max(limitedHigh.unrestrictedDt, 1e-30);
  const std::string status =
      (limitedLow.electricRelaxationTimestepLimited != 0 &&
       limitedHigh.electricRelaxationTimestepLimited != 0 &&
       limitedCurrentRatio <= baselineCurrentRatio)
          ? "DIAGNOSTIC_RELAXATION_LIMIT_REDUCES_RATIO"
          : "DOWNGRADED_RELAXATION_LIMIT_NOT_CURRENT_FIX";
  csv << baselineLow.targetCaE << "," << baselineHigh.targetCaE << ","
      << baselineLow.dt << "," << baselineHigh.dt << ","
      << limitedLow.unrestrictedDt << "," << limitedLow.dt << ","
      << limitedLow.electricRelaxationDtLimit << ","
      << limitedLow.dtOverElectricRelaxationLimit << ","
      << limitedLow.electricRelaxationTimestepLimited << ","
      << limitedHigh.unrestrictedDt << "," << limitedHigh.dt << ","
      << limitedHigh.electricRelaxationDtLimit << ","
      << limitedHigh.dtOverElectricRelaxationLimit << ","
      << limitedHigh.electricRelaxationTimestepLimited << ","
      << lowDtReduction << "," << highDtReduction << ","
      << baselineLowTail << "," << baselineHighTail << ","
      << baselineCurrentRatio << "," << limitedLowTail << ","
      << limitedHighTail << "," << limitedCurrentRatio << ","
      << limitedLow.relativeChargeBudgetResidual << ","
      << limitedHigh.relativeChargeBudgetResidual << ","
      << limitedLow.chargeBudgetExpectedFinal << ","
      << limitedLow.chargeBudgetResidual << ","
      << limitedLow.cumulativeChargeClampCorrectionL1 << ","
      << limitedLow.maxChargeRedistributionResidual << ","
      << limitedHigh.chargeBudgetExpectedFinal << ","
      << limitedHigh.chargeBudgetResidual << ","
      << limitedHigh.cumulativeChargeClampCorrectionL1 << ","
      << limitedHigh.maxChargeRedistributionResidual << ","
      << limitedLow.alphaMassDrift << "," << limitedHigh.alphaMassDrift << ","
      << limitedLow.maxDiv << "," << limitedHigh.maxDiv << ","
      << status << "\n";
}

static void writeBoundaryChargeAdvectionDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baselineLow,
    const electrospray::CandidoConeJetSmokeReport3D& baselineHigh,
    const electrospray::CandidoConeJetSmokeReport3D& advectedLow,
    const electrospray::CandidoConeJetSmokeReport3D& advectedHigh) {
  const double baselineLowTail = candidoMeanTailCurrentForObservable(baselineLow);
  const double baselineHighTail = candidoMeanTailCurrentForObservable(baselineHigh);
  const double advectedLowTail = candidoMeanTailCurrentForObservable(advectedLow);
  const double advectedHighTail = candidoMeanTailCurrentForObservable(advectedHigh);
  const double baselineRatio =
      baselineHighTail / std::max(baselineLowTail, 1e-30);
  const double advectedRatio =
      advectedHighTail / std::max(advectedLowTail, 1e-30);
  const double lowResidualRatio =
      advectedLow.relativeChargeBudgetResidual /
      std::max(baselineLow.relativeChargeBudgetResidual, 1e-30);
  const double highResidualRatio =
      advectedHigh.relativeChargeBudgetResidual /
      std::max(baselineHigh.relativeChargeBudgetResidual, 1e-30);
  const std::string status =
      (advectedLow.relativeChargeBudgetResidual <= 1e-3 &&
       advectedHigh.relativeChargeBudgetResidual <= 1e-3 &&
       advectedRatio <= baselineRatio)
          ? "DIAGNOSTIC_BOUNDARY_CHARGE_ADVECTION_IMPROVES_OR_MATCHES_RATIO"
          : "DOWNGRADED_BOUNDARY_CHARGE_ADVECTION_NOT_CURRENT_FIX";
  csv << baselineLow.targetCaE << "," << baselineHigh.targetCaE << ","
      << baselineLowTail << "," << baselineHighTail << "," << baselineRatio << ","
      << advectedLowTail << "," << advectedHighTail << "," << advectedRatio << ","
      << baselineLow.relativeChargeBudgetResidual << ","
      << baselineHigh.relativeChargeBudgetResidual << ","
      << advectedLow.relativeChargeBudgetResidual << ","
      << advectedHigh.relativeChargeBudgetResidual << ","
      << lowResidualRatio << "," << highResidualRatio << ","
      << baselineLow.cumulativeBoundaryChargeFlux << ","
      << baselineHigh.cumulativeBoundaryChargeFlux << ","
      << advectedLow.cumulativeBoundaryChargeFlux << ","
      << advectedHigh.cumulativeBoundaryChargeFlux << ","
      << advectedLow.cumulativeConductiveBoundaryChargeFlux << ","
      << advectedHigh.cumulativeConductiveBoundaryChargeFlux << ","
      << advectedLow.alphaMassDrift << "," << advectedHigh.alphaMassDrift << ","
      << advectedLow.maxDiv << "," << advectedHigh.maxDiv << ","
      << "Candido_outlet_zero_gradient_charge_and_neutral_inflow_boundary_advection,"
      << status << "\n";
}

static void writeBoundaryCurrentDecompositionRows(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const std::array<const char*, 6> patchNames = {"xmin", "xmax", "ymin_nozzle",
                                                 "ymax_collector", "zmin", "zmax"};
  double lateralCumulative = 0.0;
  double lateralPeak = 0.0;
  for (int pi : {0, 1, 4, 5}) {
    lateralCumulative += r.cumulativeConductiveBoundaryChargeFluxByPatch[pi];
    lateralPeak = std::max(lateralPeak, r.maxAbsConductiveBoundaryCurrentByPatch[pi]);
  }
  for (size_t pi = 0; pi < patchNames.size(); ++pi) {
    const double cumulative = r.cumulativeConductiveBoundaryChargeFluxByPatch[pi];
    const double fraction =
        cumulative / std::max(std::abs(r.cumulativeConductiveBoundaryChargeFlux), 1e-30);
    csv << name << "," << r.targetCaE << "," << patchNames[pi] << ","
        << cumulative << "," << r.maxAbsConductiveBoundaryCurrentByPatch[pi] << ","
        << fraction << "," << r.cumulativeConductiveBoundaryChargeFlux << ","
        << r.maxConductiveCurrent << ",PATCH_RESOLVED_DIAGNOSTIC_ONLY\n";
  }
  csv << name << "," << r.targetCaE << ",lateral_sum,"
      << lateralCumulative << "," << lateralPeak << ","
      << lateralCumulative /
             std::max(std::abs(r.cumulativeConductiveBoundaryChargeFlux), 1e-30)
      << "," << r.cumulativeConductiveBoundaryChargeFlux << ","
      << r.maxConductiveCurrent << ",PATCH_RESOLVED_DIAGNOSTIC_ONLY\n";
}

static void writeBoundaryCurrentSensitivityRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high) {
  auto absRatio = [](double highValue, double lowValue) {
    return std::abs(highValue) / std::max(std::abs(lowValue), 1e-30);
  };
  auto lateralCumulative = [](const electrospray::CandidoConeJetSmokeReport3D& r) {
    return r.cumulativeConductiveBoundaryChargeFluxByPatch[0] +
           r.cumulativeConductiveBoundaryChargeFluxByPatch[1] +
           r.cumulativeConductiveBoundaryChargeFluxByPatch[4] +
           r.cumulativeConductiveBoundaryChargeFluxByPatch[5];
  };
  auto lateralPeak = [](const electrospray::CandidoConeJetSmokeReport3D& r) {
    return std::max({r.maxAbsConductiveBoundaryCurrentByPatch[0],
                     r.maxAbsConductiveBoundaryCurrentByPatch[1],
                     r.maxAbsConductiveBoundaryCurrentByPatch[4],
                     r.maxAbsConductiveBoundaryCurrentByPatch[5]});
  };
  auto fraction = [](double value, double total) {
    return value / std::max(std::abs(total), 1e-30);
  };
  const double lowNozzle = low.cumulativeConductiveBoundaryChargeFluxByPatch[2];
  const double highNozzle = high.cumulativeConductiveBoundaryChargeFluxByPatch[2];
  const double lowCollector = low.cumulativeConductiveBoundaryChargeFluxByPatch[3];
  const double highCollector = high.cumulativeConductiveBoundaryChargeFluxByPatch[3];
  const double lowLateral = lateralCumulative(low);
  const double highLateral = lateralCumulative(high);
  const double lowTotal = low.cumulativeConductiveBoundaryChargeFlux;
  const double highTotal = high.cumulativeConductiveBoundaryChargeFlux;
  const double highNozzleAbs = std::abs(highNozzle);
  const double highCollectorAbs = std::abs(highCollector);
  const double highLateralAbs = std::abs(highLateral);
  std::string dominant = "nozzle";
  double dominantAbs = highNozzleAbs;
  if (highCollectorAbs > dominantAbs) {
    dominant = "collector";
    dominantAbs = highCollectorAbs;
  }
  if (highLateralAbs > dominantAbs) {
    dominant = "lateral";
  }
  std::string status = "PATCH_CURRENT_DIAGNOSTIC_ONLY";
  const double totalRatio = absRatio(highTotal, lowTotal);
  const double nozzleRatio = absRatio(highNozzle, lowNozzle);
  const double collectorRatio = absRatio(highCollector, lowCollector);
  const double lateralRatio = absRatio(highLateral, lowLateral);
  if (dominant == "nozzle" && nozzleRatio > 2.0) {
    status = "DIAGNOSTIC_NOZZLE_BOUNDARY_CURRENT_DOMINATES_HIGH_CAE";
  } else if (dominant == "collector" && collectorRatio > 2.0) {
    status = "DIAGNOSTIC_COLLECTOR_BOUNDARY_CURRENT_DOMINATES_HIGH_CAE";
  } else if (dominant == "lateral" && lateralRatio > 2.0) {
    status = "DIAGNOSTIC_LATERAL_BOUNDARY_CURRENT_DOMINATES_HIGH_CAE";
  } else if (totalRatio <= 2.0) {
    status = "APPROXIMATE_BOUNDARY_CONDUCTIVE_CURRENT_WEAK_SENSITIVITY";
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << lowTotal << "," << highTotal << "," << totalRatio << ","
      << lowNozzle << "," << highNozzle << "," << nozzleRatio << ","
      << lowCollector << "," << highCollector << "," << collectorRatio << ","
      << lowLateral << "," << highLateral << "," << lateralRatio << ","
      << fraction(lowNozzle, lowTotal) << "," << fraction(highNozzle, highTotal) << ","
      << fraction(lowCollector, lowTotal) << "," << fraction(highCollector, highTotal) << ","
      << fraction(lowLateral, lowTotal) << "," << fraction(highLateral, highTotal) << ","
      << low.maxAbsConductiveBoundaryCurrentByPatch[2] << ","
      << high.maxAbsConductiveBoundaryCurrentByPatch[2] << ","
      << absRatio(high.maxAbsConductiveBoundaryCurrentByPatch[2],
                  low.maxAbsConductiveBoundaryCurrentByPatch[2])
      << "," << low.maxAbsConductiveBoundaryCurrentByPatch[3] << ","
      << high.maxAbsConductiveBoundaryCurrentByPatch[3] << ","
      << absRatio(high.maxAbsConductiveBoundaryCurrentByPatch[3],
                  low.maxAbsConductiveBoundaryCurrentByPatch[3])
      << "," << lateralPeak(low) << "," << lateralPeak(high) << ","
      << absRatio(lateralPeak(high), lateralPeak(low)) << ","
      << dominant << "," << status << "\n";
}

static void writeNozzleChargeBoundaryDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& baseline,
    const electrospray::CandidoConeJetSmokeReport3D& suppressed) {
  const double currentRatio =
      suppressed.maxConvectiveCurrent / std::max(baseline.maxConvectiveCurrent, 1e-30);
  const double velocityRatio =
      suppressed.maxVelocity / std::max(baseline.maxVelocity, 1e-30);
  const double nozzleFluxRatio =
      suppressed.cumulativeConductiveBoundaryChargeFluxByPatch[2] /
      std::max(std::abs(baseline.cumulativeConductiveBoundaryChargeFluxByPatch[2]), 1e-30);
  const std::string status =
      (suppressed.relativeChargeBudgetResidual <= 1e-8 && currentRatio <= 0.1)
          ? "NOZZLE_FLUX_SUPPRESSION_REDUCES_CURRENT_DIAGNOSTIC"
          : "DOWNGRADED_NOZZLE_BOUNDARY_DIAGNOSTIC_ONLY";
  csv << baseline.targetCaE << ","
      << baseline.relativeChargeBudgetResidual << ","
      << suppressed.relativeChargeBudgetResidual << ","
      << baseline.cumulativeConductiveBoundaryChargeFluxByPatch[2] << ","
      << suppressed.cumulativeConductiveBoundaryChargeFluxByPatch[2] << ","
      << nozzleFluxRatio << ","
      << baseline.cumulativeConductiveBoundaryChargeFlux << ","
      << suppressed.cumulativeConductiveBoundaryChargeFlux << ","
      << baseline.maxConvectiveCurrent << ","
      << suppressed.maxConvectiveCurrent << ","
      << currentRatio << ","
      << baseline.maxVelocity << ","
      << suppressed.maxVelocity << ","
      << velocityRatio << ","
      << suppressed.alphaMassDrift << ","
      << suppressed.maxDiv << ","
      << status << "\n";
}

static void writeWhippingDiagnosticRow(
    std::ofstream& csv, const std::string& name,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double paperInitiationYOverDi = 3.44;
  const double diOverDo = setup.innerDiameter / setup.outerDiameter;
  const double threshold = 0.05;
  const electrospray::CandidoConeJetHistorySample3D* maxAsym = &r.history.front();
  const electrospray::CandidoConeJetHistorySample3D* onset = nullptr;
  for (const auto& h : r.history) {
    if (h.radialAsymmetry > maxAsym->radialAsymmetry) maxAsym = &h;
    if (!onset && h.radialAsymmetry >= threshold) onset = &h;
  }
  const auto* h = onset ? onset : maxAsym;
  const double onsetMs = h->time * r.hydrodynamicTimeScale * 1.0e3;
  const double onsetTipYOverDi = h->tipY / std::max(diOverDo, 1e-30);
  const double onsetCentroidYOverDi = h->centroidY / std::max(diOverDo, 1e-30);
  const double maxAsymMs = maxAsym->time * r.hydrodynamicTimeScale * 1.0e3;
  const double maxAsymTipYOverDi = maxAsym->tipY / std::max(diOverDo, 1e-30);
  const double maxAsymCentroidYOverDi = maxAsym->centroidY / std::max(diOverDo, 1e-30);
  const electrospray::CandidoConeJetHistorySample3D* wavePeak = &r.history.front();
  for (const auto& sample : r.history) {
    if (sample.waveAsymmetry > wavePeak->waveAsymmetry) wavePeak = &sample;
  }
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  int fitCount = 0;
  const double waveFitFloor = 0.25 * std::max(wavePeak->waveAsymmetry, 1e-30);
  for (const auto& sample : r.history) {
    if (sample.waveAsymmetry < waveFitFloor) continue;
    const double t = sample.time;
    const double y = sample.waveYOverDi;
    sx += t;
    sy += y;
    sxx += t * t;
    sxy += t * y;
    ++fitCount;
  }
  const double denom = static_cast<double>(fitCount) * sxx - sx * sx;
  const double waveSpeedDiPerSh =
      (fitCount >= 2 && std::abs(denom) > 1e-30)
          ? (static_cast<double>(fitCount) * sxy - sx * sy) / denom
          : std::numeric_limits<double>::quiet_NaN();
  const electrospray::CandidoConeJetHistorySample3D* firstWave = nullptr;
  for (const auto& sample : r.history) {
    if (sample.waveAsymmetry >= waveFitFloor) {
      firstWave = &sample;
      break;
    }
  }
  const double earlyWaveSpeedDiPerSh =
      (firstWave && wavePeak->time > firstWave->time)
          ? (wavePeak->waveYOverDi - firstWave->waveYOverDi) /
                std::max(wavePeak->time - firstWave->time, 1e-30)
          : std::numeric_limits<double>::quiet_NaN();
  const double wavePeakMs = wavePeak->time * r.hydrodynamicTimeScale * 1.0e3;
  const double wavePeakLocationErrorPercent =
      100.0 * (wavePeak->waveYOverDi - paperInitiationYOverDi) /
      std::max(std::abs(paperInitiationYOverDi), 1e-30);
  const double locationErrorPercent =
      100.0 * (onsetTipYOverDi - paperInitiationYOverDi) /
      std::max(std::abs(paperInitiationYOverDi), 1e-30);
  const double maxAsymLocationErrorPercent =
      100.0 * (maxAsymTipYOverDi - paperInitiationYOverDi) /
      std::max(std::abs(paperInitiationYOverDi), 1e-30);
  const std::string status =
      !onset ? "DOWNGRADED_THRESHOLD_NOT_REACHED"
             : (earlyWaveSpeedDiPerSh > 0.0 ? "ASYMMETRY_THRESHOLD_REACHED"
                                             : "DOWNGRADED_NO_POSITIVE_WAVE_TRANSLATION");
  csv << name << "," << r.targetCaE << "," << r.voltage << ","
      << threshold << "," << maxAsym->radialAsymmetry << ","
      << onsetMs << "," << onsetTipYOverDi << "," << onsetCentroidYOverDi << ","
      << maxAsymMs << "," << maxAsymTipYOverDi << "," << maxAsymCentroidYOverDi << ","
      << wavePeakMs << "," << wavePeak->waveYOverDi << "," << wavePeak->waveAsymmetry << ","
      << waveSpeedDiPerSh << "," << earlyWaveSpeedDiPerSh << "," << fitCount << ","
      << paperInitiationYOverDi << "," << locationErrorPercent << ","
      << maxAsymLocationErrorPercent << ","
      << wavePeakLocationErrorPercent << ","
      << "Candido_Fig11_reports_wave_whip_initiation_y_over_Di_3.44,"
      << status << "\n";
  csv.flush();
}

static void writeMorphologyReferenceGapRows(std::ofstream& csv, const std::string& name,
                                            const electrospray::CandidoConeJetSmokeReport3D& r) {
  struct ReferencePoint {
    double timeMs;
    double paperReportedErrorPercent;
    double digitizedExperimentalVolumeDi3;
  };
  const std::vector<ReferencePoint> reference = {
      {0.0, 1.970, 1.1663230825622644},
      {0.4, 0.747, 1.2826510303495016},
      {0.7, -1.440, 1.2550259882802302},
      {0.8, -0.395, std::numeric_limits<double>::quiet_NaN()},
      {0.9, -0.948, std::numeric_limits<double>::quiet_NaN()}};
  const auto nearest = [&](double tMs) {
    const electrospray::CandidoConeJetHistorySample3D* best = &r.history.front();
    double bestDistance = std::numeric_limits<double>::max();
    for (const auto& h : r.history) {
      const double hMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
      const double distance = std::abs(hMs - tMs);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = &h;
      }
    }
    return best;
  };
  const double finalMs = r.history.back().time * r.hydrodynamicTimeScale * 1.0e3;
  for (const auto& ref : reference) {
    const auto* h = nearest(ref.timeMs);
    const double simMs = h->time * r.hydrodynamicTimeScale * 1.0e3;
    const bool inWindow = ref.timeMs <= finalMs + 1e-12;
    const bool hasDigitizedReference = std::isfinite(ref.digitizedExperimentalVolumeDi3);
    const double computedError =
        hasDigitizedReference
            ? 100.0 * (h->morphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double connectedError =
        hasDigitizedReference
            ? 100.0 * (h->connectedMorphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double disconnectedVolume =
        std::max(0.0, h->morphologyVolumeDi3 - h->connectedMorphologyVolumeDi3);
    const double disconnectedPercentOfReference =
        hasDigitizedReference
            ? 100.0 * disconnectedVolume /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double alpha05Error =
        hasDigitizedReference
            ? 100.0 * (h->alpha05SilhouetteVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double rayAlpha05Error =
        hasDigitizedReference
            ? 100.0 * (h->rayAlpha05SilhouetteVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double allLiquidRayAlpha05Error =
        hasDigitizedReference
            ? 100.0 * (h->allLiquidRayAlpha05SilhouetteVolumeDi3 -
                       ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double rayAlpha05CellBoundaryError =
        hasDigitizedReference
            ? 100.0 * (h->rayAlpha05CellBoundarySilhouetteVolumeDi3 -
                       ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double linearRayAlpha05Error =
        hasDigitizedReference
            ? 100.0 * (h->linearRayAlpha05SilhouetteVolumeDi3 -
                       ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double plicContourError =
        hasDigitizedReference
            ? 100.0 * (h->plicContourSilhouetteVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double plicPolygonError =
        hasDigitizedReference
            ? 100.0 * (h->plicPolygonSilhouetteVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double plicSectorMedianError =
        hasDigitizedReference
            ? 100.0 * (h->plicSectorMedianSilhouetteVolumeDi3 -
                       ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double plicRayPlaneError =
        hasDigitizedReference
            ? 100.0 * (h->plicRayPlaneSilhouetteVolumeDi3 -
                       ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double plicRayPlaneQ25Error =
        hasDigitizedReference
            ? 100.0 * (h->plicRayPlaneQ25SilhouetteVolumeDi3 -
                       ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double plicFirstExitError =
        hasDigitizedReference
            ? 100.0 * (h->plicFirstExitSilhouetteVolumeDi3 -
                       ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const double outerEnvelopeAlpha05Volume =
        std::max(h->rayAlpha05SilhouetteVolumeDi3,
                 h->plicRayPlaneQ25SilhouetteVolumeDi3);
    const double outerEnvelopeAlpha05Error =
        hasDigitizedReference
            ? 100.0 * (outerEnvelopeAlpha05Volume -
                       ref.digitizedExperimentalVolumeDi3) /
                  std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30)
            : std::numeric_limits<double>::quiet_NaN();
    const std::string status =
        !inWindow ? "OUT_OF_WINDOW_NOT_VALIDATED"
                  : (hasDigitizedReference ? "DIGITIZED_EXTERNAL_COMPARISON"
                                           : "BLOCKED_DIGITIZED_GEOMETRY_REQUIRED");
    csv << name << "," << ref.timeMs << "," << simMs << ","
        << ref.paperReportedErrorPercent << "," << h->tipY << ","
        << h->centroidY << "," << h->radialAsymmetry << ","
        << h->morphologyVolumeDi3 << "," << h->connectedMorphologyVolumeDi3 << ","
        << disconnectedVolume << "," << h->alpha05SilhouetteVolumeDi3 << ","
        << h->rayAlpha05SilhouetteVolumeDi3 << ","
        << h->allLiquidRayAlpha05SilhouetteVolumeDi3 << ","
        << h->rayAlpha05CellBoundarySilhouetteVolumeDi3 << ","
        << h->linearRayAlpha05SilhouetteVolumeDi3 << ","
        << h->plicContourSilhouetteVolumeDi3 << ","
        << h->plicPolygonSilhouetteVolumeDi3 << ","
        << h->plicSectorMedianSilhouetteVolumeDi3 << ","
        << h->plicRayPlaneSilhouetteVolumeDi3 << ","
        << h->plicRayPlaneQ25SilhouetteVolumeDi3 << ","
        << h->plicFirstExitSilhouetteVolumeDi3 << ","
        << outerEnvelopeAlpha05Volume << ","
        << ref.digitizedExperimentalVolumeDi3 << ","
        << computedError << "," << connectedError << "," << alpha05Error << ","
        << rayAlpha05Error << "," << allLiquidRayAlpha05Error << ","
        << rayAlpha05CellBoundaryError << ","
        << linearRayAlpha05Error << ","
        << plicContourError << ","
        << plicPolygonError << ","
        << plicSectorMedianError << ","
        << plicRayPlaneError << ","
        << plicRayPlaneQ25Error << ","
        << plicFirstExitError << ","
        << outerEnvelopeAlpha05Error << ","
        << disconnectedPercentOfReference << ","
        << "Candido_Fig3b_blue_experimental_points_digitized_from_local_pdf" << ","
        << status << "\n";
  }
  csv.flush();
}

static void writeMorphologySilhouetteBracketRows(
    std::ofstream& csv, const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  struct ReferencePoint {
    double timeMs;
    double digitizedExperimentalVolumeDi3;
  };
  const std::vector<ReferencePoint> reference = {
      {0.4, 1.2826510303495016},
      {0.7, 1.2550259882802302}};
  const auto nearest = [&](double tMs) {
    const electrospray::CandidoConeJetHistorySample3D* best = &r.history.front();
    double bestDistance = std::numeric_limits<double>::max();
    for (const auto& h : r.history) {
      const double hMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
      const double distance = std::abs(hMs - tMs);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = &h;
      }
    }
    return best;
  };
  for (const auto& ref : reference) {
    const auto* h = nearest(ref.timeMs);
    const double simMs = h->time * r.hydrodynamicTimeScale * 1.0e3;
    const double lower =
        std::min({h->rayAlpha05SilhouetteVolumeDi3,
                  h->allLiquidRayAlpha05SilhouetteVolumeDi3,
                  h->rayAlpha05CellBoundarySilhouetteVolumeDi3,
                  h->linearRayAlpha05SilhouetteVolumeDi3,
                  h->connectedMorphologyVolumeDi3,
                  h->plicRayPlaneQ25SilhouetteVolumeDi3,
                  h->plicFirstExitSilhouetteVolumeDi3});
    const double upper =
        std::max({h->rayAlpha05SilhouetteVolumeDi3,
                  h->allLiquidRayAlpha05SilhouetteVolumeDi3,
                  h->rayAlpha05CellBoundarySilhouetteVolumeDi3,
                  h->linearRayAlpha05SilhouetteVolumeDi3,
                  h->connectedMorphologyVolumeDi3,
                  h->plicRayPlaneQ25SilhouetteVolumeDi3,
                  h->plicFirstExitSilhouetteVolumeDi3});
    const bool bracketed = ref.digitizedExperimentalVolumeDi3 >= lower &&
                           ref.digitizedExperimentalVolumeDi3 <= upper;
    const double lowerError = 100.0 * (lower - ref.digitizedExperimentalVolumeDi3) /
                              std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30);
    const double upperError = 100.0 * (upper - ref.digitizedExperimentalVolumeDi3) /
                              std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30);
    csv << name << "," << ref.timeMs << "," << simMs << ","
        << h->rayAlpha05SilhouetteVolumeDi3 << ","
        << h->allLiquidRayAlpha05SilhouetteVolumeDi3 << ","
        << h->rayAlpha05CellBoundarySilhouetteVolumeDi3 << ","
        << h->linearRayAlpha05SilhouetteVolumeDi3 << ","
        << h->connectedMorphologyVolumeDi3 << ","
        << h->plicContourSilhouetteVolumeDi3 << ","
        << h->plicPolygonSilhouetteVolumeDi3 << ","
        << h->plicSectorMedianSilhouetteVolumeDi3 << ","
        << h->plicRayPlaneSilhouetteVolumeDi3 << ","
        << h->plicRayPlaneQ25SilhouetteVolumeDi3 << ","
        << h->plicFirstExitSilhouetteVolumeDi3 << ","
        << ref.digitizedExperimentalVolumeDi3 << ","
        << lower << "," << upper << "," << lowerError << "," << upperError << ","
        << (bracketed ? "REFERENCE_BRACKETED_BY_COARSE_OBSERVABLES"
                      : "REFERENCE_OUTSIDE_COARSE_OBSERVABLE_BRACKET")
        << "\n";
  }
  csv.flush();
}

static void writeLateMorphologyBlockerRows(std::ofstream& csv) {
  csv << "reference_time_ms,paper_reported_error_percent,status,visible_fig3b_times_ms,"
         "blocker,required_input\n";
  csv << "0.8,-0.395,BLOCKED_DIGITIZED_GEOMETRY_REQUIRED,"
      << "0.0;0.4;0.7,"
      << "Candido PDF text reports only relative error and rendered pages do not expose "
         "extractable experimental contour coordinates,"
      << "external digitized Fig3b contour or numerical/experimental morphology volume at 0.8 ms\n";
  csv << "0.9,-0.948,BLOCKED_DIGITIZED_GEOMETRY_REQUIRED,"
      << "0.0;0.4;0.7,"
      << "Candido PDF text reports only relative error and rendered pages do not expose "
         "extractable experimental contour coordinates,"
      << "external digitized Fig3b contour or numerical/experimental morphology volume at 0.9 ms\n";
  csv.flush();
}

static void writeMorphologyTimeAlignmentRows(
    std::ofstream& csv, const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  struct ReferencePoint {
    double timeMs;
    double digitizedExperimentalVolumeDi3;
  };
  const std::vector<ReferencePoint> reference = {
      {0.4, 1.2826510303495016},
      {0.7, 1.2550259882802302},
  };
  for (const auto& ref : reference) {
    const electrospray::CandidoConeJetHistorySample3D* nearestTime = &r.history.front();
    const electrospray::CandidoConeJetHistorySample3D* nearestVolume = &r.history.front();
    double bestTimeDistance = std::numeric_limits<double>::max();
    double bestVolumeError = std::numeric_limits<double>::max();
    for (const auto& h : r.history) {
      const double hMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
      const double timeDistance = std::abs(hMs - ref.timeMs);
      if (timeDistance < bestTimeDistance) {
        bestTimeDistance = timeDistance;
        nearestTime = &h;
      }
      const double volumeError = std::abs(
          100.0 * (h.morphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
          std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30));
      if (volumeError < bestVolumeError) {
        bestVolumeError = volumeError;
        nearestVolume = &h;
      }
    }
    const double fixedMs = nearestTime->time * r.hydrodynamicTimeScale * 1.0e3;
    const double bestMs = nearestVolume->time * r.hydrodynamicTimeScale * 1.0e3;
    const double fixedError =
        100.0 * (nearestTime->morphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
        std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30);
    const double alignedError =
        100.0 * (nearestVolume->morphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
        std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30);
    csv << name << "," << ref.timeMs << "," << ref.digitizedExperimentalVolumeDi3 << ","
        << fixedMs << "," << nearestTime->morphologyVolumeDi3 << "," << fixedError << ","
        << bestMs << "," << nearestVolume->morphologyVolumeDi3 << "," << alignedError << ","
        << bestMs - ref.timeMs << "\n";
  }
  csv.flush();
}

static void writeMorphologyPhaseLagDiagnosticRows(
    std::ofstream& csv, const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  struct ReferencePoint {
    double timeMs;
    double digitizedExperimentalVolumeDi3;
  };
  const std::vector<ReferencePoint> reference = {
      {0.4, 1.2826510303495016},
      {0.7, 1.2550259882802302},
  };
  for (const auto& ref : reference) {
    const electrospray::CandidoConeJetHistorySample3D* nearestTime = &r.history.front();
    const electrospray::CandidoConeJetHistorySample3D* nearestVolume = &r.history.front();
    double bestTimeDistance = std::numeric_limits<double>::max();
    double bestVolumeErrorPercent = std::numeric_limits<double>::max();
    for (const auto& h : r.history) {
      const double hMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
      const double timeDistance = std::abs(hMs - ref.timeMs);
      if (timeDistance < bestTimeDistance) {
        bestTimeDistance = timeDistance;
        nearestTime = &h;
      }
      const double volumeErrorPercent = std::abs(
          100.0 * (h.morphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
          std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30));
      if (volumeErrorPercent < bestVolumeErrorPercent) {
        bestVolumeErrorPercent = volumeErrorPercent;
        nearestVolume = &h;
      }
    }

    const double fixedMs = nearestTime->time * r.hydrodynamicTimeScale * 1.0e3;
    const double bestMs = nearestVolume->time * r.hydrodynamicTimeScale * 1.0e3;
    const double fixedErrorPercent =
        100.0 * (nearestTime->morphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
        std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30);
    const double alignedErrorPercent =
        100.0 * (nearestVolume->morphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3) /
        std::max(std::abs(ref.digitizedExperimentalVolumeDi3), 1e-30);
    const double lagMs = bestMs - ref.timeMs;
    double localSlopeDi3PerMs = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 1; i < r.history.size(); ++i) {
      const double prevMs = r.history[i - 1].time * r.hydrodynamicTimeScale * 1.0e3;
      const double currMs = r.history[i].time * r.hydrodynamicTimeScale * 1.0e3;
      if ((prevMs <= fixedMs && fixedMs <= currMs) ||
          (i == r.history.size() - 1 && fixedMs >= currMs)) {
        const double dtMs = std::max(currMs - prevMs, 1e-30);
        localSlopeDi3PerMs =
            (r.history[i].morphologyVolumeDi3 - r.history[i - 1].morphologyVolumeDi3) / dtMs;
        break;
      }
    }
    const double lagExplainedVolumeDi3 =
        std::isfinite(localSlopeDi3PerMs) ? localSlopeDi3PerMs * lagMs
                                          : std::numeric_limits<double>::quiet_NaN();
    const double fixedVolumeGapDi3 =
        nearestTime->morphologyVolumeDi3 - ref.digitizedExperimentalVolumeDi3;
    const double phaseExplainedFraction =
        (std::isfinite(lagExplainedVolumeDi3) && std::abs(fixedVolumeGapDi3) > 1e-30)
            ? lagExplainedVolumeDi3 / fixedVolumeGapDi3
            : std::numeric_limits<double>::quiet_NaN();
    const bool bestVolumeWithinOnePercent = std::abs(alignedErrorPercent) <= 1.0;
    const bool phaseLagDominates =
        bestVolumeWithinOnePercent && std::abs(fixedErrorPercent) > 5.0 &&
        std::abs(alignedErrorPercent) < 0.25 * std::abs(fixedErrorPercent);
    const std::string status =
        phaseLagDominates
            ? "PHASE_LAG_DOMINATES_FIXED_TIME_ERROR"
            : (std::abs(fixedErrorPercent) <= 10.0 ? "FIXED_TIME_WITHIN_10_PERCENT"
                                                   : "DOWNGRADED_SHAPE_OR_TIMING_ERROR");
    csv << name << "," << ref.timeMs << "," << ref.digitizedExperimentalVolumeDi3 << ","
        << fixedMs << "," << nearestTime->morphologyVolumeDi3 << ","
        << fixedErrorPercent << "," << bestMs << "," << nearestVolume->morphologyVolumeDi3
        << "," << alignedErrorPercent << "," << lagMs << ","
        << localSlopeDi3PerMs << "," << lagExplainedVolumeDi3 << ","
        << phaseExplainedFraction << "," << status << "\n";
  }
  csv.flush();
}

static void writeMorphologyTipSyncDiagnosticRow(
    std::ofstream& csv, const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const auto* maxTip = &r.history.front();
  const auto* firstTipJump = static_cast<const electrospray::CandidoConeJetHistorySample3D*>(nullptr);
  std::vector<double> uniqueTipLevels;
  for (const auto& h : r.history) {
    if (h.tipY > maxTip->tipY) maxTip = &h;
    bool newLevel = true;
    for (double y : uniqueTipLevels) {
      if (std::abs(y - h.tipY) < 1e-10) {
        newLevel = false;
        break;
      }
    }
    if (newLevel) uniqueTipLevels.push_back(h.tipY);
    if (!firstTipJump && std::abs(h.tipY - r.history.front().tipY) > 1e-10) {
      firstTipJump = &h;
    }
  }
  double minNonzeroTipStep = std::numeric_limits<double>::infinity();
  std::sort(uniqueTipLevels.begin(), uniqueTipLevels.end());
  for (size_t i = 1; i < uniqueTipLevels.size(); ++i) {
    const double dy = uniqueTipLevels[i] - uniqueTipLevels[i - 1];
    if (dy > 1e-10) minNonzeroTipStep = std::min(minNonzeroTipStep, dy);
  }
  if (!std::isfinite(minNonzeroTipStep)) minNonzeroTipStep = 0.0;
  const double maxTipMs = maxTip->time * r.hydrodynamicTimeScale * 1.0e3;
  const double firstJumpMs =
      firstTipJump ? firstTipJump->time * r.hydrodynamicTimeScale * 1.0e3
                   : std::numeric_limits<double>::quiet_NaN();
  const double paperSyncOffsetMs = 0.4 - maxTipMs;
  const auto nearestShiftedTime = [&](double paperTimeMs) {
    const double targetSimMs = paperTimeMs - paperSyncOffsetMs;
    const electrospray::CandidoConeJetHistorySample3D* best = &r.history.front();
    double bestDistance = std::numeric_limits<double>::max();
    for (const auto& h : r.history) {
      const double hMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
      const double distance = std::abs(hMs - targetSimMs);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = &h;
      }
    }
    return best;
  };
  const double reference04 = 1.2826510303495016;
  const double reference07 = 1.2550259882802302;
  const auto* sync04 = nearestShiftedTime(0.4);
  const auto* sync07 = nearestShiftedTime(0.7);
  const double sync04Ms = sync04->time * r.hydrodynamicTimeScale * 1.0e3;
  const double sync07Ms = sync07->time * r.hydrodynamicTimeScale * 1.0e3;
  const double sync04Error =
      100.0 * (sync04->morphologyVolumeDi3 - reference04) / std::abs(reference04);
  const double sync07Error =
      100.0 * (sync07->morphologyVolumeDi3 - reference07) / std::abs(reference07);
  const double sync04AllLiquidRayAlpha05Error =
      100.0 * (sync04->allLiquidRayAlpha05SilhouetteVolumeDi3 - reference04) /
      std::abs(reference04);
  const double sync07AllLiquidRayAlpha05Error =
      100.0 * (sync07->allLiquidRayAlpha05SilhouetteVolumeDi3 - reference07) /
      std::abs(reference07);
  const double sync04ConnectedRayAlpha05Error =
      100.0 * (sync04->rayAlpha05SilhouetteVolumeDi3 - reference04) / std::abs(reference04);
  const double sync07ConnectedRayAlpha05Error =
      100.0 * (sync07->rayAlpha05SilhouetteVolumeDi3 - reference07) / std::abs(reference07);
  const bool enoughTipLevels = uniqueTipLevels.size() >= 5;
  const std::string status =
      !enoughTipLevels
          ? "DOWNGRADED_TIP_QUANTIZED_COARSE_GRID"
          : ((std::abs(sync04Error) <= 10.0 && std::abs(sync07Error) <= 10.0)
                 ? "TIP_SYNC_MORPHOLOGY_WITHIN_10_PERCENT"
                 : "DOWNGRADED_TIP_SYNC_MORPHOLOGY_ERROR");
  const std::string alpha05Status =
      (std::abs(sync04AllLiquidRayAlpha05Error) <= 10.0 &&
       std::abs(sync07AllLiquidRayAlpha05Error) <= 10.0)
          ? "TIP_SYNC_ALL_LIQUID_ALPHA05_WITHIN_10_PERCENT"
          : "DOWNGRADED_TIP_SYNC_ALPHA05_INTERFACE_LOST_OR_MISMATCHED";
  csv << name << "," << r.history.size() << "," << uniqueTipLevels.size() << ","
      << r.history.front().tipY << "," << maxTip->tipY << ","
      << minNonzeroTipStep << "," << maxTipMs << "," << firstJumpMs << ","
      << paperSyncOffsetMs << "," << sync04Ms << "," << sync04->morphologyVolumeDi3 << ","
      << sync04Error << "," << sync04->rayAlpha05SilhouetteVolumeDi3 << ","
      << sync04ConnectedRayAlpha05Error << ","
      << sync04->allLiquidRayAlpha05SilhouetteVolumeDi3 << ","
      << sync04AllLiquidRayAlpha05Error << "," << sync07Ms << ","
      << sync07->morphologyVolumeDi3 << "," << sync07Error << ","
      << sync07->rayAlpha05SilhouetteVolumeDi3 << "," << sync07ConnectedRayAlpha05Error
      << "," << sync07->allLiquidRayAlpha05SilhouetteVolumeDi3 << ","
      << sync07AllLiquidRayAlpha05Error << ","
      << "Candido_text_sets_0.4ms_to_maximum_cone_length_without_jet_emission,"
      << alpha05Status << "," << status << "\n";
}

static void writeInterfacePreservationCandidateRow(
    std::ofstream& csv, const std::string& name,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const auto* maxTip = &r.history.front();
  std::vector<double> uniqueTipLevels;
  for (const auto& h : r.history) {
    if (h.tipY > maxTip->tipY) maxTip = &h;
    bool newLevel = true;
    for (double y : uniqueTipLevels) {
      if (std::abs(y - h.tipY) < 1e-10) {
        newLevel = false;
        break;
      }
    }
    if (newLevel) uniqueTipLevels.push_back(h.tipY);
  }
  const double maxTipMs = maxTip->time * r.hydrodynamicTimeScale * 1.0e3;
  const double paperSyncOffsetMs = 0.4 - maxTipMs;
  const auto nearestShiftedTime = [&](double paperTimeMs) {
    const double targetSimMs = paperTimeMs - paperSyncOffsetMs;
    const electrospray::CandidoConeJetHistorySample3D* best = &r.history.front();
    double bestDistance = std::numeric_limits<double>::max();
    for (const auto& h : r.history) {
      const double hMs = h.time * r.hydrodynamicTimeScale * 1.0e3;
      const double distance = std::abs(hMs - targetSimMs);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = &h;
      }
    }
    return best;
  };
  const double reference04 = 1.2826510303495016;
  const double reference07 = 1.2550259882802302;
  const auto* sync04 = nearestShiftedTime(0.4);
  const auto* sync07 = nearestShiftedTime(0.7);
  const double sync04VolumeError =
      100.0 * (sync04->morphologyVolumeDi3 - reference04) / std::abs(reference04);
  const double sync07VolumeError =
      100.0 * (sync07->morphologyVolumeDi3 - reference07) / std::abs(reference07);
  const double sync04Alpha05Error =
      100.0 * (sync04->allLiquidRayAlpha05SilhouetteVolumeDi3 - reference04) /
      std::abs(reference04);
  const double sync07Alpha05Error =
      100.0 * (sync07->allLiquidRayAlpha05SilhouetteVolumeDi3 - reference07) /
      std::abs(reference07);
  const bool alpha05Present =
      sync04->allLiquidRayAlpha05SilhouetteVolumeDi3 > 0.0 &&
      sync07->allLiquidRayAlpha05SilhouetteVolumeDi3 > 0.0;
  const std::string status =
      alpha05Present && std::abs(sync04Alpha05Error) <= 10.0 &&
              std::abs(sync07Alpha05Error) <= 10.0
          ? "UPHELD_ALPHA05_SYNC_MORPHOLOGY"
          : (alpha05Present ? "APPROXIMATE_ALPHA05_PRESENT_BUT_OUTSIDE_BAR"
                            : "DOWNGRADED_ALPHA05_INTERFACE_LOST");
  csv << name << "," << opt.vofCompression << "," << opt.vofPostSharpening << ","
      << opt.vofPostSharpeningSweeps << "," << (opt.useVofInletBoundaryAlpha ? 1 : 0)
      << "," << r.alphaMassDrift << "," << r.maxDiv << ","
      << r.maxAlpha << "," << uniqueTipLevels.size() << "," << maxTipMs << ","
      << paperSyncOffsetMs << ","
      << sync04->time * r.hydrodynamicTimeScale * 1.0e3 << ","
      << sync04->morphologyVolumeDi3 << "," << sync04VolumeError << ","
      << sync04->allLiquidRayAlpha05SilhouetteVolumeDi3 << "," << sync04Alpha05Error << ","
      << sync07->time * r.hydrodynamicTimeScale * 1.0e3 << ","
      << sync07->morphologyVolumeDi3 << "," << sync07VolumeError << ","
      << sync07->allLiquidRayAlpha05SilhouetteVolumeDi3 << "," << sync07Alpha05Error << ","
      << status << "\n";
  csv.flush();
}

static void checkSmokeReport(const electrospray::CandidoConeJetSmokeReport3D& r,
                             double targetCaE) {
  check(r.cells > 0 && r.faces > 0, "Candido cone-jet smoke mesh is non-empty");
  check(std::abs(r.computedCaE - targetCaE) / targetCaE < 1e-12,
        "Candido cone-jet voltage reproduces requested CaE");
  check(std::abs(r.electricWeber - 20.4) < 1e-12,
        "Candido cone-jet validation WeE from paper is recorded");
  check(std::abs(r.hydrodynamicTimeScale - 2.770163961e-4) < 5e-13,
        "Candido cone-jet hydrodynamic time scale matches paper properties");
  check(std::abs(r.inletVelocity - 8.007483074e-4) < 5e-13,
        "Candido cone-jet inlet velocity matches Qi/(pi Di^2/4)");
  check(r.alphaMassDrift <= 1e-3, "Candido cone-jet isoAdvector VoF mass drift bounded");
  check(r.minAlpha >= -1e-12 && r.maxAlpha <= 1.0 + 1e-12,
        "Candido cone-jet alpha remains bounded");
  check(r.maxPotentialResidual <= 1e-7, "Candido cone-jet potential solve converged");
  check(r.maxElectricForce > 0.0 && std::isfinite(r.maxElectricForce),
        "Candido cone-jet Maxwell force active");
  check(r.maxCsfForce > 0.0 && std::isfinite(r.maxCsfForce),
        "Candido cone-jet CSF force active");
  check(r.maxCurvature > 0.0 && std::isfinite(r.maxCurvature),
        "Candido cone-jet local curvature active");
  check(r.curvatureFallbackFraction >= 0.0 && r.curvatureFallbackFraction <= 1.0,
        "Candido cone-jet curvature fallback fraction reported");
  check(r.maxConductiveCurrent > 0.0 && std::isfinite(r.maxConductiveCurrent),
        "Candido cone-jet conductive current flux active");
  check(r.maxConvectiveCurrent >= 0.0 && std::isfinite(r.maxConvectiveCurrent),
        "Candido cone-jet convective current metric finite");
  check(r.maxDiv <= 1e-7, "Candido cone-jet Rhie-Chow projected continuity bounded");
  check(r.maxVelocity > 0.0 && std::isfinite(r.maxVelocity),
        "Candido cone-jet velocity response active");
  check(r.finalMidplaneJetRadius >= 0.0 && std::isfinite(r.finalMidplaneJetRadius),
        "Candido cone-jet midplane jet radius metric finite");
  check(static_cast<int>(r.history.size()) == r.steps + 1,
        "Candido cone-jet morphology history records every step plus initial state");
  check(r.history.front().step == 0 && r.history.back().step == r.steps,
        "Candido cone-jet morphology history step bounds are correct");
}

static void writeBoundaryConditionDiagnostics(const electrospray::CandidoTaylorConeJetSetup& setup,
                                              const electrospray::CandidoConeJetSmokeOptions3D& opt,
                                              double targetCaE) {
  const double ly = setup.collectorDistance / setup.outerDiameter;
  fvm::Mesh3D mesh = fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, opt.radialWindowOuterDiameters, ly,
                                          opt.radialWindowOuterDiameters, opt.skew);
  const double voltage = electrospray::candidoVoltageForElectricCapillary(setup, targetCaE);
  const double dimensionlessVoltage =
      voltage / (electrospray::candidoElectricFieldScale(setup, setup.validationVoltage) *
                 setup.outerDiameter);
  const auto bc = electrospray::candidoPotentialBoundary3D(mesh, setup, opt, dimensionlessVoltage);
  int nozzleElectrodeFaces = 0;
  int collectorFaces = 0;
  int openSideFaces = 0;
  int inletCandidateFaces = 0;
  const double yTol = ly / std::max(opt.ny, 1) * 0.55;
  const double axisX = 0.5 * opt.radialWindowOuterDiameters;
  const double axisZ = 0.5 * opt.radialWindowOuterDiameters;
  const double inletRadius = 0.5 * setup.innerDiameter / setup.outerDiameter;
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    const auto& face = mesh.faces[fi];
    if (face.internal()) continue;
    if (bc.faceDirichlet[fi] && bc.faceValue[fi] > 0.5 * dimensionlessVoltage) {
      ++nozzleElectrodeFaces;
    } else if (bc.faceDirichlet[fi]) {
      ++collectorFaces;
    } else {
      ++openSideFaces;
    }
    const double r = std::hypot(face.centroid.x() - axisX, face.centroid.z() - axisZ);
    double minVertexRadius = r;
    for (int pi : face.points) {
      minVertexRadius = std::min(
          minVertexRadius,
          std::hypot(mesh.points[pi].x() - axisX, mesh.points[pi].z() - axisZ));
    }
    if (face.centroid.y() <= yTol && minVertexRadius <= inletRadius) ++inletCandidateFaces;
  }
  std::ofstream csv("benchmark_logs/candido_boundary_conditions3d.csv");
  csv << "target_ca_e,faces,boundary_faces,nozzle_electrode_faces,collector_faces,"
         "open_side_faces,inlet_candidate_faces,voltage,dimensionless_voltage,contact_angle_deg\n";
  csv << targetCaE << "," << mesh.faces.size() << ","
      << nozzleElectrodeFaces + collectorFaces + openSideFaces << ","
      << nozzleElectrodeFaces << "," << collectorFaces << "," << openSideFaces << ","
      << inletCandidateFaces << "," << voltage << "," << dimensionlessVoltage << ","
      << setup.contactAngleDeg << "\n";
  check(nozzleElectrodeFaces > 0, "Candido nozzle electrode faces are present");
  check(collectorFaces > 0, "Candido collector electrode faces are present");
  check(openSideFaces > 0, "Candido open-side boundary faces are present");
}

static void writeContactAngleDiagnostic(const electrospray::CandidoTaylorConeJetSetup& setup,
                                        const electrospray::CandidoConeJetSmokeOptions3D& opt,
                                        double targetCaE) {
  const double ly = setup.collectorDistance / setup.outerDiameter;
  fvm::Mesh3D mesh = fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, opt.radialWindowOuterDiameters, ly,
                                          opt.radialWindowOuterDiameters, opt.skew);
  fvm::ScalarField alpha = electrospray::candidoInitialAlpha3D(mesh, setup, opt, targetCaE);
  fvm::VectorField3 gradAlpha = fvm::gradLeastSquares3D(mesh, alpha);
  const double yTol = ly / std::max(opt.ny, 1) * 1.5;
  int wallAdjacentMixedCells = 0;
  double maxAngleErrorDeg = 0.0;
  double minTangentAlignment = 1.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double a = alpha[ci];
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    if (mesh.cells[ci].centroid.y() > yTol) continue;
    if (gradAlpha[ci].norm() <= 1e-30) continue;
    const fvm::Vec3 rawNormal = gradAlpha[ci].normalized();
    const fvm::Vec3 wallNormal = fvm::Vec3::UnitY();
    const fvm::Vec3 adjusted =
        fvm::contactAngleAdjustedNormal3D(rawNormal, wallNormal, setup.contactAngleDeg);
    const double angleDeg =
        std::acos(std::clamp(adjusted.dot(wallNormal), -1.0, 1.0)) * 180.0 / M_PI;
    fvm::Vec3 rawTangent = rawNormal - rawNormal.dot(wallNormal) * wallNormal;
    fvm::Vec3 adjustedTangent = adjusted - adjusted.dot(wallNormal) * wallNormal;
    if (rawTangent.norm() > 1e-30 && adjustedTangent.norm() > 1e-30) {
      rawTangent.normalize();
      adjustedTangent.normalize();
      minTangentAlignment = std::min(minTangentAlignment, rawTangent.dot(adjustedTangent));
    }
    maxAngleErrorDeg = std::max(maxAngleErrorDeg, std::abs(angleDeg - setup.contactAngleDeg));
    ++wallAdjacentMixedCells;
  }
  std::ofstream csv("benchmark_logs/candido_contact_angle_diagnostic3d.csv");
  csv << "target_ca_e,contact_angle_deg,wall_adjacent_mixed_cells,"
         "max_projection_angle_error_deg,min_tangent_alignment,"
         "projection_utility_verified,production_curvature_enforced,status\n";
  const int projectionVerified =
      (maxAngleErrorDeg < 1e-10 && minTangentAlignment > 1.0 - 1e-10) ? 1 : 0;
  csv << targetCaE << "," << setup.contactAngleDeg << "," << wallAdjacentMixedCells << ","
      << maxAngleErrorDeg << "," << minTangentAlignment << "," << projectionVerified
      << ",0,UTILITY_READY_PRODUCTION_CURVATURE_UNWIRED\n";
  check(projectionVerified == 1, "Candido contact-angle projection utility verified");
}

static void writeContactAngleCurvatureDiagnostic(
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    double targetCaE) {
  const double ly = setup.collectorDistance / setup.outerDiameter;
  fvm::Mesh3D mesh = fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, opt.radialWindowOuterDiameters, ly,
                                          opt.radialWindowOuterDiameters, opt.skew);
  fvm::ScalarField alpha = electrospray::candidoInitialAlpha3D(mesh, setup, opt, targetCaE);
  const double yTol = ly / std::max(opt.ny, 1) * 1.5;
  const fvm::Vec3 wallNormal = fvm::Vec3::UnitY();
  const auto baseline = fvm::curvatureFromLocalPlicQuadricReport3D(mesh, alpha, 28);
  const auto adjusted = fvm::curvatureFromLocalPlicQuadricReport3D(
      mesh, alpha, 28, &wallNormal, setup.contactAngleDeg, yTol);

  int wallMixedCells = 0;
  int changedWallCells = 0;
  double maxWallDelta = 0.0;
  double meanWallDelta = 0.0;
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    const double a = alpha[ci];
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    if (mesh.cells[ci].centroid.y() > yTol) continue;
    const double delta = std::abs(adjusted.kappa[ci] - baseline.kappa[ci]);
    ++wallMixedCells;
    meanWallDelta += delta;
    maxWallDelta = std::max(maxWallDelta, delta);
    if (delta > 1e-12) ++changedWallCells;
  }
  if (wallMixedCells > 0) meanWallDelta /= static_cast<double>(wallMixedCells);

  std::ofstream csv("benchmark_logs/candido_contact_angle_curvature_diagnostic3d.csv");
  csv << "target_ca_e,contact_angle_deg,wall_mixed_cells,changed_wall_cells,"
         "baseline_fitted_cells,adjusted_fitted_cells,baseline_fallback_fraction,"
         "adjusted_fallback_fraction,baseline_p95_condition,adjusted_p95_condition,"
         "baseline_max_condition,adjusted_max_condition,baseline_max_abs_curvature,"
         "adjusted_max_abs_curvature,mean_wall_curvature_delta,max_wall_curvature_delta,"
         "production_curvature_enforced,status\n";
  csv << targetCaE << "," << setup.contactAngleDeg << "," << wallMixedCells << ","
      << changedWallCells << "," << baseline.fittedCells << "," << adjusted.fittedCells << ","
      << baseline.fallbackFraction << "," << adjusted.fallbackFraction << ","
      << baseline.p95StencilCondition << "," << adjusted.p95StencilCondition << ","
      << baseline.maxStencilCondition << "," << adjusted.maxStencilCondition << ","
      << baseline.maxAbsCurvature << "," << adjusted.maxAbsCurvature << ","
      << meanWallDelta << "," << maxWallDelta
      << ",0,DIAGNOSTIC_CONTACT_ANGLE_CURVATURE_PATH_UNWIRED\n";

  const auto baselineSurface =
      fvm::buildBalancedForceSurfaceTensionState3D(mesh, alpha, 1.0, &baseline.kappa);
  const auto adjustedSurface =
      fvm::buildBalancedForceSurfaceTensionState3D(mesh, alpha, 1.0, &adjusted.kappa);
  auto forceStats = [&](const fvm::VectorField3& force, bool wallOnly) {
    double maxForce = 0.0;
    double meanForce = 0.0;
    int count = 0;
    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
      if (wallOnly) {
        const double a = alpha[ci];
        if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
        if (mesh.cells[ci].centroid.y() > yTol) continue;
      }
      const double mag = force[ci].norm();
      maxForce = std::max(maxForce, mag);
      meanForce += mag;
      ++count;
    }
    if (count > 0) meanForce /= static_cast<double>(count);
    return std::pair<double, double>{maxForce, meanForce};
  };
  auto kappaStats = [&](const fvm::ScalarField& kappa, bool wallOnly) {
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    int count = 0;
    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
      if (wallOnly) {
        const double a = alpha[ci];
        if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
        if (mesh.cells[ci].centroid.y() > yTol) continue;
      }
      const double mag = std::abs(kappa[ci]);
      maxAbs = std::max(maxAbs, mag);
      meanAbs += mag;
      ++count;
    }
    if (count > 0) meanAbs /= static_cast<double>(count);
    return std::pair<double, double>{maxAbs, meanAbs};
  };
  const auto baselineForceAll = forceStats(baselineSurface.csfForce, false);
  const auto adjustedForceAll = forceStats(adjustedSurface.csfForce, false);
  const auto baselineForceWall = forceStats(baselineSurface.csfForce, true);
  const auto adjustedForceWall = forceStats(adjustedSurface.csfForce, true);
  const auto baselineKappaAll = kappaStats(baseline.kappa, false);
  const auto adjustedKappaAll = kappaStats(adjusted.kappa, false);
  const auto baselineKappaWall = kappaStats(baseline.kappa, true);
  const auto adjustedKappaWall = kappaStats(adjusted.kappa, true);
  std::ofstream forceCsv("benchmark_logs/candido_contact_angle_force_decomposition3d.csv");
  forceCsv << "target_ca_e,contact_angle_deg,wall_mixed_cells,"
              "baseline_max_abs_kappa_all,adjusted_max_abs_kappa_all,"
              "baseline_mean_abs_kappa_all,adjusted_mean_abs_kappa_all,"
              "baseline_max_abs_kappa_wall,adjusted_max_abs_kappa_wall,"
              "baseline_mean_abs_kappa_wall,adjusted_mean_abs_kappa_wall,"
              "baseline_max_csf_all,adjusted_max_csf_all,"
              "baseline_mean_csf_all,adjusted_mean_csf_all,"
              "baseline_max_csf_wall,adjusted_max_csf_wall,"
              "baseline_mean_csf_wall,adjusted_mean_csf_wall,status\n";
  forceCsv << targetCaE << "," << setup.contactAngleDeg << "," << wallMixedCells << ","
           << baselineKappaAll.first << "," << adjustedKappaAll.first << ","
           << baselineKappaAll.second << "," << adjustedKappaAll.second << ","
           << baselineKappaWall.first << "," << adjustedKappaWall.first << ","
           << baselineKappaWall.second << "," << adjustedKappaWall.second << ","
           << baselineForceAll.first << "," << adjustedForceAll.first << ","
           << baselineForceAll.second << "," << adjustedForceAll.second << ","
           << baselineForceWall.first << "," << adjustedForceWall.first << ","
           << baselineForceWall.second << "," << adjustedForceWall.second
           << ",DIAGNOSTIC_FORCE_SCALE_DECOMPOSITION\n";
  check(wallMixedCells > 0, "Candido contact-angle curvature diagnostic has wall mixed cells");
  check(adjusted.fallbackFraction >= 0.0 && adjusted.fallbackFraction <= 1.0,
        "Candido contact-angle curvature fallback fraction bounded");
}

static void writeContactAngleCurvatureSwitchDiagnostic(
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    double targetCaE) {
  electrospray::CandidoConeJetSmokeOptions3D baseOpt = opt;
  baseOpt.steps = 1;
  baseOpt.useContactAngleCurvature = false;
  electrospray::CandidoConeJetSmokeOptions3D contactOpt = baseOpt;
  contactOpt.useContactAngleCurvature = true;
  const auto baseline = electrospray::runCandidoConeJetSmoke3D(targetCaE, setup, baseOpt);
  const auto contact = electrospray::runCandidoConeJetSmoke3D(targetCaE, setup, contactOpt);

  std::ofstream csv("benchmark_logs/candido_contact_angle_curvature_switch3d.csv");
  csv << "target_ca_e,baseline_max_curvature,contact_max_curvature,"
         "baseline_max_csf_force,contact_max_csf_force,baseline_fallback_fraction,"
         "contact_fallback_fraction,baseline_alpha_mass_drift,contact_alpha_mass_drift,"
         "baseline_max_div,contact_max_div,production_curvature_enforced,status\n";
  csv << targetCaE << "," << baseline.maxCurvature << "," << contact.maxCurvature << ","
      << baseline.maxCsfForce << "," << contact.maxCsfForce << ","
      << baseline.curvatureFallbackFraction << "," << contact.curvatureFallbackFraction << ","
      << baseline.alphaMassDrift << "," << contact.alphaMassDrift << ","
      << baseline.maxDiv << "," << contact.maxDiv
      << ",1,DIAGNOSTIC_SWITCH_EXERCISED_NOT_DEFAULT\n";
  check(std::isfinite(contact.maxCurvature) && contact.maxCurvature > 0.0,
        "Candido contact-angle curvature switch produces finite curvature");
  check(std::isfinite(contact.maxCsfForce) && contact.maxCsfForce > 0.0,
        "Candido contact-angle curvature switch produces finite CSF force");
  check(contact.curvatureFallbackFraction >= 0.0 && contact.curvatureFallbackFraction <= 1.0,
        "Candido contact-angle curvature switch fallback fraction bounded");
  check(contact.alphaMassDrift <= 1e-3,
        "Candido contact-angle curvature switch keeps VoF mass drift bounded");
  check(contact.maxDiv <= 1e-7,
        "Candido contact-angle curvature switch keeps projection continuity bounded");
}

static void writeMaxwellTangentialClosureDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    double targetCaE) {
  electrospray::CandidoConeJetSmokeOptions3D cellOpt = opt;
  cellOpt.steps = 1;
  cellOpt.usePoissonFaceConductiveCurrent = true;
  cellOpt.usePoissonFaceMaxwellForce = false;
  cellOpt.usePoissonHybridMaxwellForce = false;

  electrospray::CandidoConeJetSmokeOptions3D faceOpt = cellOpt;
  faceOpt.usePoissonFaceMaxwellForce = true;

  electrospray::CandidoConeJetSmokeOptions3D hybridOpt = faceOpt;
  hybridOpt.usePoissonHybridMaxwellForce = true;

  const auto cell = electrospray::runCandidoConeJetSmoke3D(targetCaE, setup, cellOpt);
  const auto face = electrospray::runCandidoConeJetSmoke3D(targetCaE, setup, faceOpt);
  const auto hybrid = electrospray::runCandidoConeJetSmoke3D(targetCaE, setup, hybridOpt);

  const double faceToCellForce =
      face.maxElectricForce / std::max(cell.maxElectricForce, 1e-300);
  const double hybridToFaceForce =
      hybrid.maxElectricForce / std::max(face.maxElectricForce, 1e-300);
  const double hybridToCellForce =
      hybrid.maxElectricForce / std::max(cell.maxElectricForce, 1e-300);
  const double hybridToFaceVelocity =
      hybrid.maxVelocity / std::max(face.maxVelocity, 1e-300);
  const bool finite =
      std::isfinite(faceToCellForce) && std::isfinite(hybridToFaceForce) &&
      std::isfinite(hybridToCellForce) && std::isfinite(hybrid.maxElectricForce) &&
      std::isfinite(hybrid.maxVelocity);
  const std::string status =
      !finite ? "DOWNGRADED_NONFINITE_HYBRID_FORCE"
              : (std::abs(hybridToFaceForce - 1.0) > 0.25
                     ? "TANGENTIAL_COMPONENT_SIGNIFICANT_DIAGNOSTIC_ONLY"
                     : "FACE_NORMAL_FORCE_CLOSE_TO_HYBRID_DIAGNOSTIC_ONLY");

  csv << targetCaE << "," << cell.maxElectricForce << "," << face.maxElectricForce
      << "," << hybrid.maxElectricForce << "," << faceToCellForce << ","
      << hybridToFaceForce << "," << hybridToCellForce << ","
      << cell.maxVelocity << "," << face.maxVelocity << "," << hybrid.maxVelocity
      << "," << hybridToFaceVelocity << "," << cell.alphaMassDrift << ","
      << face.alphaMassDrift << "," << hybrid.alphaMassDrift << ","
      << cell.maxDiv << "," << face.maxDiv << "," << hybrid.maxDiv << ","
      << status << "\n";
  csv.flush();

  check(finite, "Candido hybrid Maxwell force diagnostic stays finite");
  check(hybrid.alphaMassDrift <= 1e-3,
        "Candido hybrid Maxwell force diagnostic keeps VoF mass drift bounded");
  check(hybrid.maxDiv <= 1e-7,
        "Candido hybrid Maxwell force diagnostic keeps projection continuity bounded");
}

static void writeFaceElectricReconstructionDiagnosticRow(
    std::ofstream& csv,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    double targetCaE) {
  const auto d =
      electrospray::candidoFaceElectricReconstructionDiagnostic3D(targetCaE, setup, opt);
  const bool finite =
      std::isfinite(d.meanRelativeNormalMismatch) &&
      std::isfinite(d.maxRelativeNormalMismatch) &&
      std::isfinite(d.meanTangentialFraction) &&
      std::isfinite(d.maxTangentialFraction) &&
      std::isfinite(d.meanHybridToNormalTractionRatio) &&
      std::isfinite(d.p95HybridToNormalTractionRatio) &&
      std::isfinite(d.maxHybridToNormalTractionRatio) &&
      std::isfinite(d.potentialResidual);
  const std::string status =
      !finite ? "DOWNGRADED_NONFINITE_FACE_VECTOR_RECONSTRUCTION"
              : (d.meanTangentialFraction > 0.25 ||
                         std::abs(d.meanHybridToNormalTractionRatio - 1.0) > 0.25
                     ? "DOWNGRADED_FACE_NORMAL_ONLY_DROPS_SIGNIFICANT_TANGENTIAL_FIELD"
                     : "APPROXIMATE_FACE_VECTOR_RECONSTRUCTION_BOUNDED");

  csv << targetCaE << "," << d.sampledFaces << "," << d.internalFaces << ","
      << d.dirichletBoundaryFaces << "," << d.tractionRatioFaces << ","
      << d.normalTractionDegenerateFaces << "," << d.maxPoissonNormalE << ","
      << d.maxCellTangentialE << "," << d.meanRelativeNormalMismatch << ","
      << d.maxRelativeNormalMismatch << "," << d.meanTangentialFraction << ","
      << d.maxTangentialFraction << "," << d.meanHybridToNormalTractionRatio << ","
      << d.p95HybridToNormalTractionRatio << ","
      << d.maxHybridToNormalTractionRatio << "," << d.potentialResidual << ","
      << status << "\n";
  csv.flush();

  check(d.sampledFaces > 0, "Candido face-vector electric diagnostic samples faces");
  check(d.internalFaces > 0, "Candido face-vector electric diagnostic samples internal faces");
  check(d.dirichletBoundaryFaces > 0,
        "Candido face-vector electric diagnostic samples Dirichlet electrode faces");
  check(d.tractionRatioFaces > 0,
        "Candido face-vector electric diagnostic samples active traction faces");
  check(finite, "Candido face-vector electric reconstruction diagnostic stays finite");
}

static void writeBoundedVectorMaxwellDiagnosticRows(
    std::ofstream& csv,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    double targetCaE) {
  constexpr double floorFraction = 0.05;
  const std::vector<double> factors = {0.0, 0.25, 0.5, 1.0, 2.0};
  const auto normal =
      electrospray::candidoBoundedVectorMaxwellDiagnostic3D(targetCaE, setup, opt,
                                                            0.0, floorFraction);
  for (double factor : factors) {
    const auto d = electrospray::candidoBoundedVectorMaxwellDiagnostic3D(
        targetCaE, setup, opt, factor, floorFraction);
    const double clippedFraction =
        static_cast<double>(d.tangentialClippedFaces) /
        std::max(1.0, static_cast<double>(d.sampledFaces));
    const double forceRatio =
        d.force.maxForce / std::max(normal.force.maxForce, 1e-300);
    const bool finite = std::isfinite(d.force.maxForce) &&
                        std::isfinite(d.meanTangentialClipRatio) &&
                        std::isfinite(forceRatio) &&
                        std::isfinite(d.potentialResidual);
    const std::string status =
        !finite ? "DOWNGRADED_NONFINITE_BOUNDED_VECTOR_FORCE"
                : (factor == 0.0
                       ? "BASELINE_FACE_NORMAL_ONLY"
                       : (forceRatio > 10.0
                              ? "DOWNGRADED_BOUNDED_VECTOR_FORCE_STILL_LARGE"
                              : "APPROXIMATE_BOUNDED_VECTOR_FORCE_FINITE"));
    csv << targetCaE << "," << factor << "," << floorFraction << ","
        << d.sampledFaces << "," << d.tangentialClippedFaces << ","
        << clippedFraction << "," << d.meanTangentialClipRatio << ","
        << d.minTangentialClipRatio << "," << d.maxPoissonNormalE << ","
        << d.maxRawTangentialE << "," << d.maxLimitedTangentialE << ","
        << normal.force.maxForce << "," << d.force.maxForce << ","
        << forceRatio << "," << d.force.maxStressDivergence << ","
        << d.potentialResidual << "," << status << "\n";
    csv.flush();
    check(d.sampledFaces > 0, "Candido bounded vector Maxwell diagnostic samples faces");
    check(finite, "Candido bounded vector Maxwell diagnostic stays finite");
  }
}

static void writeTomarConductingSurfaceForceDiagnosticRows(
    std::ofstream& csv,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    double targetCaE) {
  const auto d =
      electrospray::candidoTomarConductingSurfaceForceDiagnostic3D(targetCaE,
                                                                   setup, opt);
  const double forceRatio =
      d.force.maxForce / std::max(d.defaultMaxForce, 1e-300);
  const double tangentialShare =
      d.maxTangentialTerm /
      std::max(d.maxTangentialTerm + d.maxNormalTerm, 1e-300);
  const bool finite = std::isfinite(d.force.maxForce) &&
                      std::isfinite(d.defaultMaxForce) &&
                      std::isfinite(forceRatio) &&
                      std::isfinite(tangentialShare) &&
                      std::isfinite(d.potentialResidual);
  const std::string status =
      !finite ? "DOWNGRADED_NONFINITE_TOMAR_CONDUCTING_FORCE"
              : (forceRatio > 1.0e6
                     ? "DOWNGRADED_TOMAR_FORCE_STIFF_FOR_GAS_CONDUCTIVITY"
                     : "APPROXIMATE_TOMAR_FORCE_FINITE_DIAGNOSTIC");
  csv << targetCaE << "," << d.sampledCells << "," << d.mixedCells << ","
      << d.activeInterfaceCells << "," << d.maxGradAlpha << ","
      << d.maxNormalCurrent << "," << d.maxTangentialE << ","
      << d.maxNormalTerm << "," << d.maxTangentialTerm << ","
      << tangentialShare << "," << d.defaultMaxForce << ","
      << d.force.maxForce << "," << forceRatio << ","
      << d.potentialResidual << "," << status << "\n";
  csv.flush();
  check(d.sampledCells > 0, "Candido Tomar conducting force samples cells");
  check(d.activeInterfaceCells > 0,
        "Candido Tomar conducting force samples interface cells");
  check(finite, "Candido Tomar conducting force diagnostic stays finite");
}

int main() {
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/candido_cone_jet_smoke3d.csv");
  csv << "case,cells,faces,steps,target_ca_e,voltage,computed_ca_e,we_e,"
         "hydrodynamic_time_scale,inlet_velocity,dt,initial_mass,final_mass,"
         "alpha_mass_drift,cumulative_boundary_liquid_flux,"
         "cumulative_boundary_liquid_inflow,cumulative_boundary_liquid_outflow,"
         "mass_budget_expected_final,mass_budget_residual,"
         "relative_mass_budget_residual,min_alpha,max_alpha,initial_tip_y,final_tip_y,"
         "tip_displacement,final_centroid_y,max_div,max_potential_residual,"
         "max_electric_force,max_csf_force,max_curvature,curvature_fallback_fraction,"
         "min_charge,max_charge,max_conductive_current,max_convective_current,"
         "final_radial_asymmetry,final_midplane_jet_radius,max_velocity\n";
  std::ofstream history("benchmark_logs/candido_morphology_timeseries3d.csv");
  history << "case,step,time,mass,min_alpha,max_alpha,tip_y,centroid_y,"
             "radial_asymmetry,max_div,potential_residual,electric_force,csf_force,"
             "curvature,conductive_current,convective_current,"
             "liquid_convective_current,alpha05_convective_current,"
             "midplane_liquid_area_di2,midplane_alpha05_area_di2,"
             "developed_jet_y_over_Di,developed_jet_alpha05_area_di2,"
             "developed_jet_convective_current,"
             "developed_jet_liquid_convective_current,"
             "developed_jet_alpha05_convective_current,"
             "developed_jet_total_current,"
             "developed_jet_liquid_total_current,"
             "developed_jet_alpha05_total_current,"
             "developed_jet_alpha05_conductive_current,"
             "developed_jet_mean_alpha05_charge,"
             "developed_jet_mean_alpha05_abs_charge,"
             "developed_jet_mean_alpha05_uy,"
             "developed_jet_mean_alpha05_abs_uy,"
             "developed_jet_mean_alpha05_abs_electric_momentum_source_y,"
             "developed_jet_mean_alpha05_abs_surface_momentum_source_y,"
             "developed_jet_mean_alpha05_abs_momentum_source_y,"
             "developed_jet_mean_alpha05_abs_momentum_acceleration_y,"
             "developed_jet_alpha05_current_shape_factor,"
             "total_current,max_velocity,"
             "wave_y_over_Di,wave_asymmetry,"
             "morphology_volume_di3,connected_morphology_volume_di3,"
             "alpha05_silhouette_volume_di3,ray_alpha05_silhouette_volume_di3,"
             "all_liquid_ray_alpha05_silhouette_volume_di3,"
             "ray_alpha05_cell_boundary_silhouette_volume_di3,"
             "linear_ray_alpha05_silhouette_volume_di3,"
             "plic_contour_silhouette_volume_di3,plic_polygon_silhouette_volume_di3,"
             "plic_sector_median_silhouette_volume_di3,"
             "plic_ray_plane_silhouette_volume_di3,"
             "plic_ray_plane_q25_silhouette_volume_di3,"
             "plic_first_exit_silhouette_volume_di3,"
             "poisson_face_convective_current,"
             "poisson_face_conductive_current,"
             "poisson_face_total_current,"
             "poisson_face_alpha05_convective_current,"
             "poisson_face_alpha05_conductive_current,"
             "poisson_face_alpha05_total_current,"
             "poisson_face_developed_y_over_Di,"
             "poisson_face_developed_alpha05_area_di2,"
             "poisson_face_developed_alpha05_convective_current,"
             "poisson_face_developed_alpha05_conductive_current,"
             "poisson_face_developed_alpha05_total_current,"
             "poisson_face_developed_alpha05_mean_abs_upwind_charge,"
             "poisson_face_developed_alpha05_mean_abs_face_flux,"
             "poisson_face_developed_alpha05_mean_abs_convective_flux,"
             "poisson_face_developed_alpha05_max_abs_upwind_charge,"
             "poisson_face_developed_alpha05_max_abs_face_flux,"
             "raw_velocity_face_developed_alpha05_convective_current,"
             "raw_velocity_face_developed_alpha05_mean_abs_upwind_charge,"
             "raw_velocity_face_developed_alpha05_mean_abs_face_flux,"
             "raw_velocity_face_developed_alpha05_mean_abs_convective_flux,"
             "raw_velocity_face_developed_alpha05_max_abs_upwind_charge,"
             "raw_velocity_face_developed_alpha05_max_abs_face_flux\n";
  std::ofstream morphologyAudit("benchmark_logs/candido_morphology_observable_audit3d.csv");
  morphologyAudit << "case,step,physical_time_ms,alpha_integral_volume_di3,"
                     "connected_alpha_integral_volume_di3,alpha05_silhouette_volume_di3,"
                     "ray_alpha05_silhouette_volume_di3,"
                     "all_liquid_ray_alpha05_silhouette_volume_di3,"
                     "ray_alpha05_cell_boundary_silhouette_volume_di3,"
                     "linear_ray_alpha05_silhouette_volume_di3,"
                     "connected_minus_alpha_integral_di3,"
                     "alpha05_silhouette_minus_alpha_integral_di3,"
                     "ray_alpha05_silhouette_minus_alpha_integral_di3,"
                     "all_liquid_ray_alpha05_silhouette_minus_alpha_integral_di3,"
                     "ray_alpha05_cell_boundary_minus_alpha_integral_di3,"
                     "linear_ray_alpha05_silhouette_minus_alpha_integral_di3,"
                     "plic_contour_silhouette_volume_di3,"
                     "plic_contour_minus_alpha_integral_di3,"
                     "plic_polygon_silhouette_volume_di3,"
                     "plic_polygon_minus_alpha_integral_di3,"
                     "plic_sector_median_silhouette_volume_di3,"
                     "plic_sector_median_minus_alpha_integral_di3,"
                     "plic_ray_plane_silhouette_volume_di3,"
                     "plic_ray_plane_minus_alpha_integral_di3,"
                     "plic_ray_plane_q25_silhouette_volume_di3,"
                     "plic_ray_plane_q25_minus_alpha_integral_di3,"
                     "plic_first_exit_silhouette_volume_di3,"
                     "plic_first_exit_minus_alpha_integral_di3,status\n";
  std::ofstream morphologyBracket("benchmark_logs/candido_morphology_silhouette_bracket3d.csv");
  morphologyBracket << "case,reference_time_ms,nearest_sim_time_ms,"
                       "ray_alpha05_silhouette_volume_di3,"
                       "all_liquid_ray_alpha05_silhouette_volume_di3,"
                       "ray_alpha05_cell_boundary_silhouette_volume_di3,"
                       "linear_ray_alpha05_silhouette_volume_di3,"
                       "connected_alpha_integral_volume_di3,"
                       "plic_contour_silhouette_volume_di3,"
                       "plic_polygon_silhouette_volume_di3,"
                       "plic_sector_median_silhouette_volume_di3,"
                       "plic_ray_plane_silhouette_volume_di3,"
                       "plic_ray_plane_q25_silhouette_volume_di3,"
                       "plic_first_exit_silhouette_volume_di3,"
                       "digitized_experimental_volume_di3,bracket_lower_di3,"
                       "bracket_upper_di3,lower_error_percent,upper_error_percent,status\n";
  std::ofstream physicalTime("benchmark_logs/candido_physical_time_progress3d.csv");
  physicalTime << "case,step,dimensionless_time,physical_time_s,physical_time_ms,"
                  "hydrodynamic_time_scale_s,tip_y,centroid_y,radial_asymmetry,max_velocity\n";
  std::ofstream jetCurrent("benchmark_logs/candido_jet_current_metrics3d.csv");
  jetCurrent << "case,target_ca_e,voltage,computed_ca_e,final_midplane_jet_radius,"
                "max_conductive_current,max_convective_current,final_radial_asymmetry,"
                "max_velocity,alpha_mass_drift,max_div\n";
  std::ofstream currentScaling("benchmark_logs/candido_current_scaling3d.csv");
  currentScaling << "case,target_ca_e,voltage,computed_ca_e,max_conductive_current,"
                    "max_convective_current,max_electric_force,max_velocity,"
                    "final_midplane_jet_radius,alpha_mass_drift,max_div\n";
  std::ofstream currentScalingValidation("benchmark_logs/candido_current_scaling_validation3d.csv");
  currentScalingValidation << "case,target_ca_e,voltage,flow_rate,surface_tension,"
                              "liquid_conductivity,ganan_calvo_current_scale,"
                              "max_conductive_current,conductive_ratio,"
                              "max_convective_current,convective_ratio,interpretation,status\n";
  std::ofstream currentVoltageSensitivity("benchmark_logs/candido_current_voltage_sensitivity3d.csv");
  currentVoltageSensitivity << "low_ca_e,high_ca_e,low_peak_convective_current,"
                               "high_peak_convective_current,peak_current_ratio,"
                               "low_mean_all_convective_current,high_mean_all_convective_current,"
                               "low_mean_tail_convective_current,high_mean_tail_convective_current,"
                               "tail_mean_current_ratio,external_source,status\n";
  std::ofstream combinedCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_combined_charge3d.csv");
  combinedCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream rayleighCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_rayleigh_charge3d.csv");
  rayleighCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream collectorOnlyCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_collector_only_charge3d.csv");
  collectorOnlyCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream poissonFaceCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_poisson_face_current3d.csv");
  poissonFaceCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream implicitOhmicCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_implicit_ohmic_charge3d.csv");
  implicitOhmicCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream faceConsistentElectricCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_face_consistent_electric3d.csv");
  faceConsistentElectricCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream faceImplicitElectricCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_face_implicit_electric3d.csv");
  faceImplicitElectricCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream totalCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_total_current3d.csv");
  totalCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_total_current,"
         "high_peak_total_current,peak_current_ratio,"
         "low_mean_all_total_current,high_mean_all_total_current,"
         "low_mean_tail_total_current,high_mean_tail_total_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream poissonFaceTotalCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_poisson_face_total3d.csv");
  poissonFaceTotalCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_poisson_face_total_current,"
         "high_peak_poisson_face_total_current,peak_current_ratio,"
         "low_mean_all_poisson_face_total_current,"
         "high_mean_all_poisson_face_total_current,"
         "low_mean_tail_poisson_face_total_current,"
         "high_mean_tail_poisson_face_total_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream poissonFaceAlpha05TotalCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_poisson_face_alpha05_total3d.csv");
  poissonFaceAlpha05TotalCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_poisson_face_alpha05_total_current,"
         "high_peak_poisson_face_alpha05_total_current,peak_current_ratio,"
         "low_mean_all_poisson_face_alpha05_total_current,"
         "high_mean_all_poisson_face_alpha05_total_current,"
         "low_mean_tail_poisson_face_alpha05_total_current,"
         "high_mean_tail_poisson_face_alpha05_total_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream liquidJetCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_liquid_jet3d.csv");
  liquidJetCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream alpha05JetCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_alpha05_jet3d.csv");
  alpha05JetCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream caIndependentBoundaryCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_ca_independent_boundary3d.csv");
  caIndependentBoundaryCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream unitMaxwellBoundaryCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_unit_maxwell_boundary3d.csv");
  unitMaxwellBoundaryCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream paperChargeBoundaryCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_paper_charge_boundary3d.csv");
  paperChargeBoundaryCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream paperInletVelocityCurrentVoltageSensitivity(
      "benchmark_logs/candido_current_voltage_sensitivity_paper_inlet_velocity3d.csv");
  paperInletVelocityCurrentVoltageSensitivity
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream developedJetCurrentWindow(
      "benchmark_logs/candido_developed_jet_current_window3d.csv");
  developedJetCurrentWindow
      << "case,low_ca_e,high_ca_e,observable,min_midplane_alpha05_area_di2,"
         "low_tail_samples,high_tail_samples,low_developed_samples,"
         "high_developed_samples,low_mean_tail_area_di2,high_mean_tail_area_di2,"
         "low_max_area_di2,high_max_area_di2,low_mean_developed_area_di2,"
         "high_mean_developed_area_di2,low_mean_developed_current,"
         "high_mean_developed_current,developed_current_ratio,status\n";
  std::ofstream axialDevelopedJetCurrentWindow(
      "benchmark_logs/candido_axial_developed_jet_current_window3d.csv");
  axialDevelopedJetCurrentWindow
      << "case,low_ca_e,high_ca_e,observable,min_alpha05_area_di2,"
         "low_tail_samples,high_tail_samples,low_developed_samples,"
         "high_developed_samples,low_max_area_di2,high_max_area_di2,"
         "low_mean_developed_area_di2,high_mean_developed_area_di2,"
         "low_mean_developed_y_over_Di,high_mean_developed_y_over_Di,"
         "low_mean_developed_current,high_mean_developed_current,"
         "developed_current_ratio,status\n";
  std::ofstream midplaneCurrentReach(
      "benchmark_logs/candido_midplane_current_reach_diagnostic3d.csv");
  midplaneCurrentReach
      << "case,low_ca_e,high_ca_e,low_steps,high_steps,min_alpha05_area_di2,"
         "low_tail_samples,high_tail_samples,low_developed_samples,"
         "high_developed_samples,low_max_alpha05_area_di2,"
         "high_max_alpha05_area_di2,low_tail_max_tip_y,high_tail_max_tip_y,"
         "low_mean_developed_alpha05_current,"
         "high_mean_developed_alpha05_current,mean_current_ratio,"
         "low_peak_developed_alpha05_current,"
         "high_peak_developed_alpha05_current,peak_current_ratio,"
         "low_alpha_mass_drift,high_alpha_mass_drift,low_max_div,high_max_div,"
         "status\n";
  std::ofstream reducedCollectorCurrentFixture(
      "benchmark_logs/candido_reduced_collector_current_fixture3d.csv");
  reducedCollectorCurrentFixture
      << "case,collector_distance_m,collector_distance_over_inner_diameter,"
         "midplane_y_over_Di,low_ca_e,high_ca_e,low_steps,high_steps,"
         "min_alpha05_area_di2,low_tail_samples,high_tail_samples,"
         "low_developed_samples,high_developed_samples,"
         "low_max_alpha05_area_di2,high_max_alpha05_area_di2,"
         "low_tail_max_tip_y,high_tail_max_tip_y,"
         "low_mean_developed_alpha05_current,"
         "high_mean_developed_alpha05_current,mean_current_ratio,"
         "low_peak_developed_alpha05_current,"
         "high_peak_developed_alpha05_current,peak_current_ratio,"
         "low_alpha_mass_drift,high_alpha_mass_drift,low_max_div,high_max_div,"
         "status\n";
  std::ofstream fig8bCurrentBlocker(
      "benchmark_logs/candido_fig8b_current_blocker3d.csv");
  fig8bCurrentBlocker
      << "case,nx,ny,nz,min_alpha05_area_di2,paper_midplane_y_over_Di,"
         "reduced_midplane_y_over_Di,paper_low_midplane_developed_samples,"
         "paper_high_midplane_developed_samples,"
         "reduced_low_midplane_developed_samples,"
         "reduced_high_midplane_developed_samples,paper_low_tail_max_tip_y,"
         "paper_high_tail_max_tip_y,reduced_low_tail_max_tip_y,"
         "reduced_high_tail_max_tip_y,paper_charge_tail_ratio,"
         "paper_inlet_velocity_tail_ratio,best_tail_ratio,"
         "paper_charge_peak_ratio,paper_inlet_velocity_peak_ratio,"
         "best_peak_ratio,status\n";
  std::ofstream currentPareto(
      "benchmark_logs/candido_current_morphology_whip_pareto3d.csv");
  currentPareto
      << "case,low_ca_e,high_ca_e,low_steps,high_steps,min_alpha05_area_di2,"
         "fixed_midplane_y_over_Di,low_alpha_mass_drift,high_alpha_mass_drift,"
         "low_max_div,high_max_div,low_max_morphology_error_0_4_0_7_percent,"
         "high_max_radial_asymmetry,all_phase_tail_current_ratio,"
         "all_phase_peak_current_ratio,low_fixed_midplane_developed_samples,"
         "high_fixed_midplane_developed_samples,low_axial_developed_samples,"
         "high_axial_developed_samples,low_axial_mean_area_di2,"
         "high_axial_mean_area_di2,low_axial_alpha05_convective_current,"
         "high_axial_alpha05_convective_current,axial_alpha05_convective_ratio,"
         "low_axial_alpha05_total_current,high_axial_alpha05_total_current,"
         "axial_alpha05_total_ratio,low_axial_mean_abs_charge,"
         "high_axial_mean_abs_charge,axial_charge_ratio,low_axial_mean_abs_uy,"
         "high_axial_mean_abs_uy,axial_velocity_ratio,status\n";
  std::ofstream openBoundaryCurrent(
      "benchmark_logs/candido_open_boundary_current_diagnostic3d.csv");
  openBoundaryCurrent
      << "case,low_ca_e,high_ca_e,low_steps,high_steps,min_alpha05_area_di2,"
         "fixed_midplane_y_over_Di,low_alpha_mass_drift,high_alpha_mass_drift,"
         "low_max_div,high_max_div,low_boundary_liquid_inflow,"
         "high_boundary_liquid_inflow,low_boundary_liquid_outflow,"
         "high_boundary_liquid_outflow,low_boundary_liquid_flux,"
         "high_boundary_liquid_flux,low_mass_budget_residual,"
         "high_mass_budget_residual,low_max_morphology_error_0_4_0_7_percent,"
         "high_max_radial_asymmetry,all_phase_tail_current_ratio,"
         "all_phase_peak_current_ratio,low_fixed_midplane_developed_samples,"
         "high_fixed_midplane_developed_samples,low_axial_developed_samples,"
         "high_axial_developed_samples,low_axial_mean_area_di2,"
         "high_axial_mean_area_di2,low_axial_alpha05_convective_current,"
         "high_axial_alpha05_convective_current,axial_alpha05_convective_ratio,"
         "low_axial_alpha05_total_current,high_axial_alpha05_total_current,"
         "axial_alpha05_total_ratio,status\n";
  std::ofstream movingCollectorBoundary(
      "benchmark_logs/candido_moving_collector_boundary_diagnostic3d.csv");
  movingCollectorBoundary
      << "case,low_ca_e,high_ca_e,collector_speed_m_per_s,"
         "collector_speed_dimensionless,low_steps,high_steps,"
         "min_alpha05_area_di2,fixed_midplane_y_over_Di,"
         "low_alpha_mass_drift,high_alpha_mass_drift,low_max_div,high_max_div,"
         "low_max_morphology_error_0_4_0_7_percent,"
         "high_max_radial_asymmetry,all_phase_tail_current_ratio,"
         "all_phase_peak_current_ratio,low_fixed_midplane_developed_samples,"
         "high_fixed_midplane_developed_samples,low_axial_developed_samples,"
         "high_axial_developed_samples,low_axial_mean_area_di2,"
         "high_axial_mean_area_di2,low_axial_alpha05_convective_current,"
         "high_axial_alpha05_convective_current,axial_alpha05_convective_ratio,"
         "low_axial_alpha05_total_current,high_axial_alpha05_total_current,"
         "axial_alpha05_total_ratio,status\n";
  std::ofstream paperCurrentDevelopment(
      "benchmark_logs/candido_paper_current_development_tradeoff3d.csv");
  paperCurrentDevelopment
      << "case,min_alpha05_area_di2,stable_low_steps,stable_high_steps,"
         "extended_low_steps,extended_high_steps,stable_low_alpha_mass_drift,"
         "stable_high_alpha_mass_drift,extended_low_alpha_mass_drift,"
         "extended_high_alpha_mass_drift,stable_low_max_div,stable_high_max_div,"
         "extended_low_max_div,extended_high_max_div,"
         "stable_low_midplane_developed_samples,"
         "stable_high_midplane_developed_samples,"
         "extended_low_midplane_developed_samples,"
         "extended_high_midplane_developed_samples,"
         "stable_low_max_midplane_alpha05_area_di2,"
         "stable_high_max_midplane_alpha05_area_di2,"
         "extended_low_max_midplane_alpha05_area_di2,"
         "extended_high_max_midplane_alpha05_area_di2,"
         "stable_low_max_tip_y,stable_high_max_tip_y,"
         "extended_low_max_tip_y,extended_high_max_tip_y,"
         "extended_low_first_developed_step,"
         "extended_high_first_developed_step,"
         "extended_low_first_developed_time_ms,"
         "extended_high_first_developed_time_ms,"
         "extended_low_mean_developed_alpha05_current,"
         "extended_high_mean_developed_alpha05_current,"
         "extended_mean_current_ratio,"
         "extended_low_peak_developed_alpha05_current,"
         "extended_high_peak_developed_alpha05_current,"
         "extended_peak_current_ratio,status\n";
  std::ofstream preconditionedCurrentPlane(
      "benchmark_logs/candido_preconditioned_current_plane_diagnostic3d.csv");
  preconditionedCurrentPlane
      << "case,low_ca_e,high_ca_e,low_steps,high_steps,"
         "min_alpha05_area_di2,fixed_midplane_y_over_Di,"
         "preconditioned_tip_y_over_Di,preconditioned_radius_Di,"
         "preconditioned_width_Di,preconditioned_velocity_scale,"
         "low_alpha_mass_drift,high_alpha_mass_drift,low_max_div,high_max_div,"
         "low_fixed_midplane_developed_samples,"
         "high_fixed_midplane_developed_samples,"
         "low_max_fixed_midplane_alpha05_area_di2,"
         "high_max_fixed_midplane_alpha05_area_di2,"
         "low_mean_fixed_alpha05_current,high_mean_fixed_alpha05_current,"
         "fixed_mean_current_ratio,low_peak_fixed_alpha05_current,"
         "high_peak_fixed_alpha05_current,fixed_peak_current_ratio,"
         "all_phase_tail_current_ratio,all_phase_peak_current_ratio,"
         "low_axial_developed_samples,high_axial_developed_samples,"
         "low_axial_alpha05_convective_current,"
         "high_axial_alpha05_convective_current,"
         "axial_alpha05_convective_ratio,low_axial_alpha05_total_current,"
         "high_axial_alpha05_total_current,axial_alpha05_total_ratio,status\n";
  std::ofstream axialCurrentFactorization(
      "benchmark_logs/candido_axial_current_factorization3d.csv");
  axialCurrentFactorization
      << "case,low_ca_e,high_ca_e,min_alpha05_area_di2,low_tail_samples,"
         "high_tail_samples,low_developed_samples,high_developed_samples,"
         "low_mean_area_di2,high_mean_area_di2,area_ratio,"
         "low_mean_abs_charge,high_mean_abs_charge,charge_ratio,"
         "low_mean_abs_uy,high_mean_abs_uy,velocity_ratio,"
         "low_mean_current_shape_factor,high_mean_current_shape_factor,"
         "shape_factor_ratio,low_mean_alpha05_current,"
         "high_mean_alpha05_current,current_ratio,product_ratio,"
         "dominant_factor,status\n";
  std::ofstream poissonFaceConvectiveFactorization(
      "benchmark_logs/candido_poisson_face_convective_factorization3d.csv");
  poissonFaceConvectiveFactorization
      << "case,low_ca_e,high_ca_e,min_alpha05_area_di2,low_tail_samples,"
         "high_tail_samples,low_developed_samples,high_developed_samples,"
         "low_mean_area_di2,high_mean_area_di2,area_ratio,"
         "low_mean_signed_current,high_mean_signed_current,current_ratio,"
         "low_mean_abs_upwind_charge,high_mean_abs_upwind_charge,charge_ratio,"
         "low_mean_abs_face_flux,high_mean_abs_face_flux,face_flux_ratio,"
         "low_mean_abs_convective_flux,high_mean_abs_convective_flux,"
         "abs_convective_flux_ratio,low_max_abs_upwind_charge,"
         "high_max_abs_upwind_charge,low_max_abs_face_flux,"
         "high_max_abs_face_flux,status\n";
  std::ofstream poissonFaceVelocityProjectionFactorization(
      "benchmark_logs/candido_poisson_face_velocity_projection_factorization3d.csv");
  poissonFaceVelocityProjectionFactorization
      << "case,low_ca_e,high_ca_e,min_alpha05_area_di2,low_tail_samples,"
         "high_tail_samples,low_developed_samples,high_developed_samples,"
         "low_mean_area_di2,high_mean_area_di2,area_ratio,"
         "low_projected_current,high_projected_current,projected_current_ratio,"
         "low_raw_velocity_current,high_raw_velocity_current,raw_velocity_current_ratio,"
         "low_projected_abs_face_flux,high_projected_abs_face_flux,"
         "projected_face_flux_ratio,low_raw_velocity_abs_face_flux,"
         "high_raw_velocity_abs_face_flux,raw_velocity_face_flux_ratio,"
         "low_projected_abs_upwind_charge,high_projected_abs_upwind_charge,"
         "low_raw_velocity_abs_upwind_charge,high_raw_velocity_abs_upwind_charge,"
         "low_projected_abs_convective_flux,high_projected_abs_convective_flux,"
         "low_raw_velocity_abs_convective_flux,high_raw_velocity_abs_convective_flux,"
         "low_projected_to_raw_current,high_projected_to_raw_current,"
         "low_projected_to_raw_face_flux,high_projected_to_raw_face_flux,status\n";
  std::ofstream momentumSourceFactorization(
      "benchmark_logs/candido_momentum_source_factorization3d.csv");
  momentumSourceFactorization
      << "case,low_ca_e,high_ca_e,min_alpha05_area_di2,low_tail_samples,"
         "high_tail_samples,low_developed_samples,high_developed_samples,"
         "low_mean_area_di2,high_mean_area_di2,area_ratio,"
         "low_mean_abs_uy,high_mean_abs_uy,velocity_ratio,"
         "low_mean_abs_electric_source,high_mean_abs_electric_source,"
         "electric_source_ratio,low_mean_abs_surface_source,"
         "high_mean_abs_surface_source,surface_source_ratio,"
         "low_mean_abs_source,high_mean_abs_source,source_ratio,"
         "low_mean_abs_acceleration,high_mean_abs_acceleration,"
         "acceleration_ratio,dominant_factor,status\n";
  std::ofstream axialTotalCurrentClosure(
      "benchmark_logs/candido_axial_total_current_closure3d.csv");
  axialTotalCurrentClosure
      << "case,low_ca_e,high_ca_e,min_alpha05_area_di2,"
         "low_tail_samples,high_tail_samples,low_developed_samples,"
         "high_developed_samples,low_mean_area_di2,high_mean_area_di2,"
         "low_mean_alpha05_convective_current,"
         "high_mean_alpha05_convective_current,convective_ratio,"
         "low_mean_alpha05_conductive_current,"
         "high_mean_alpha05_conductive_current,conductive_ratio,"
         "low_mean_alpha05_total_current,high_mean_alpha05_total_current,"
         "alpha05_total_ratio,low_mean_all_phase_total_current,"
         "high_mean_all_phase_total_current,all_phase_total_ratio,"
         "low_conductive_share,high_conductive_share,dominant_factor,status\n";
  std::ofstream axialCurrentThresholdSweep(
      "benchmark_logs/candido_axial_current_threshold_sweep3d.csv");
  axialCurrentThresholdSweep
      << "case,low_ca_e,high_ca_e,min_alpha05_area_di2,low_tail_samples,"
         "high_tail_samples,low_developed_samples,high_developed_samples,"
         "low_developed_fraction,high_developed_fraction,low_mean_area_di2,"
         "high_mean_area_di2,low_mean_alpha05_current,"
         "high_mean_alpha05_current,current_ratio,status\n";
  std::ofstream currentBlowup("benchmark_logs/candido_current_blowup_diagnostic3d.csv");
  currentBlowup << "case,target_ca_e,ganan_calvo_current_scale,peak_step,peak_time_ms,"
                   "peak_convective_current,peak_reference_ratio,peak_max_velocity,"
                   "peak_mass,peak_mass_drift,peak_min_alpha,peak_max_alpha,"
                   "peak_radial_asymmetry,peak_tip_y,first_out_of_scale_step,"
                   "first_out_of_scale_time_ms,first_out_of_scale_reference_ratio,"
                   "run_max_charge,run_min_charge,status\n";
  std::ofstream massBudget("benchmark_logs/candido_long_window_mass_budget3d.csv");
  massBudget << "case,target_ca_e,steps,initial_mass,final_mass,"
                "cumulative_boundary_liquid_flux,cumulative_boundary_liquid_inflow,"
                "cumulative_boundary_liquid_outflow,mass_budget_expected_final,"
                "mass_budget_residual,relative_mass_budget_residual,"
                "open_domain_growth,signed_boundary_growth,alpha_mass_drift,"
                "max_convective_current,max_velocity,status\n";
  std::ofstream chargeBudget("benchmark_logs/candido_long_window_charge_budget3d.csv");
  chargeBudget << "case,target_ca_e,steps,initial_integrated_charge,"
                  "final_integrated_charge,cumulative_boundary_charge_flux,"
                  "cumulative_conductive_boundary_charge_flux,"
                  "cumulative_charge_relaxation_sink,"
                  "charge_budget_expected_final,charge_budget_residual,"
                  "relative_charge_budget_residual,cumulative_charge_clamp_correction_l1,"
                  "max_charge_redistribution_residual,max_charge_clamped_cells,"
                  "max_unclamped_abs_charge,max_charge,min_charge,max_conductive_current,"
                  "max_convective_current,max_velocity,status\n";
  std::ofstream chargeSubcycling("benchmark_logs/candido_charge_subcycling_diagnostic3d.csv");
  chargeSubcycling << "target_ca_e,subcycles,baseline_relative_charge_budget_residual,"
                      "subcycled_relative_charge_budget_residual,residual_ratio,"
                      "baseline_clamp_correction_l1,subcycled_clamp_correction_l1,"
                      "clamp_correction_ratio,baseline_max_clamped_cells,"
                      "subcycled_max_clamped_cells,baseline_max_unclamped_abs_charge,"
                      "subcycled_max_unclamped_abs_charge,baseline_max_convective_current,"
                      "subcycled_max_convective_current,current_ratio,"
                      "baseline_max_velocity,subcycled_max_velocity,status\n";
  std::ofstream conservativeChargeBounding(
      "benchmark_logs/candido_charge_conservative_bounding_diagnostic3d.csv");
  conservativeChargeBounding << "target_ca_e,baseline_relative_charge_budget_residual,"
                                "bounded_relative_charge_budget_residual,residual_ratio,"
                                "baseline_clamp_correction_l1,bounded_clamp_correction_l1,"
                                "clamp_correction_ratio,bounded_max_redistribution_residual,"
                                "baseline_max_clamped_cells,bounded_max_clamped_cells,"
                                "baseline_max_unclamped_abs_charge,bounded_max_unclamped_abs_charge,"
                                "baseline_max_convective_current,bounded_max_convective_current,"
                                "current_ratio,baseline_max_velocity,bounded_max_velocity,status\n";
  std::ofstream combinedChargeBoundingSubcycling(
      "benchmark_logs/candido_charge_combined_bounding_subcycling3d.csv");
  combinedChargeBoundingSubcycling
      << "target_ca_e,baseline_relative_charge_budget_residual,"
         "bounded_relative_charge_budget_residual,residual_ratio,"
         "baseline_clamp_correction_l1,bounded_clamp_correction_l1,"
         "clamp_correction_ratio,bounded_max_redistribution_residual,"
         "baseline_max_clamped_cells,bounded_max_clamped_cells,"
         "baseline_max_unclamped_abs_charge,bounded_max_unclamped_abs_charge,"
         "baseline_max_convective_current,bounded_max_convective_current,"
         "current_ratio,baseline_max_velocity,bounded_max_velocity,status\n";
  std::ofstream chargeLimitSensitivity(
      "benchmark_logs/candido_charge_limit_sensitivity3d.csv");
  chargeLimitSensitivity << "target_ca_e,charge_limit_base,relative_charge_budget_residual,"
                            "cumulative_charge_clamp_correction_l1,max_charge_clamped_cells,"
                            "max_unclamped_abs_charge,max_charge,min_charge,"
                            "max_convective_current,max_velocity,alpha_mass_drift,max_div,status\n";
  std::ofstream chargeScaleAudit(
      "benchmark_logs/candido_charge_scale_audit3d.csv");
  chargeScaleAudit << "case,target_ca_e,charge_limit_base,effective_q_limit,"
                      "rayleigh_charge_scale_coulomb,ganan_calvo_current_scale_ampere,"
                      "rayleigh_over_current_time_scale_s,initial_integrated_charge,"
                      "final_integrated_charge,max_abs_integrated_charge,"
                      "cumulative_charge_clamp_correction_l1,"
                      "clamp_correction_over_rayleigh_charge,"
                      "max_unclamped_abs_charge,max_charge,min_charge,"
                      "max_convective_current,current_over_ganan_calvo,status\n";
  std::ofstream chargeUnitConsistency(
      "benchmark_logs/candido_charge_unit_consistency3d.csv");
  chargeUnitConsistency << "case,target_ca_e,rayleigh_charge_scale_coulomb,"
                           "ganan_calvo_current_scale_ampere,hydrodynamic_time_scale_s,"
                           "max_abs_integrated_charge,max_convective_current,"
                           "charge_unit_from_integrated_charge_coulomb,"
                           "charge_unit_from_current_coulomb,unit_consistency_ratio,status\n";
  std::ofstream nondimChargeScale(
      "benchmark_logs/candido_nondim_charge_scale_audit3d.csv");
  nondimChargeScale << "case,target_ca_e,charge_limit_base,effective_q_limit,"
                       "poisson_charge_scale_coulomb,poisson_current_scale_ampere,"
                       "rayleigh_charge_scale_coulomb,ganan_calvo_current_scale_ampere,"
                       "q_limit_physical_coulomb,q_limit_over_rayleigh,"
                       "max_integrated_charge_physical_coulomb,"
                       "max_integrated_charge_over_rayleigh,"
                       "clamp_correction_physical_coulomb,"
                       "clamp_correction_over_rayleigh,"
                       "max_convective_current_physical_ampere,"
                       "max_convective_current_over_ganan_calvo,status\n";
  std::ofstream chargeFieldConsistency(
      "benchmark_logs/candido_charge_field_consistency3d.csv");
  chargeFieldConsistency
      << "case,target_ca_e,max_potential_residual,"
         "max_gauss_law_cell_gradient_residual,"
         "max_relative_gauss_law_cell_gradient_residual,max_charge,min_charge,"
         "max_convective_current,max_conductive_current,relative_charge_budget_residual,"
         "cumulative_charge_clamp_correction_l1,status\n";
  std::ofstream electricPropertyScaling(
      "benchmark_logs/candido_electric_property_scaling_audit3d.csv");
  electricPropertyScaling
      << "case,target_ca_e,liquid_relative_permittivity,liquid_conductivity_S_per_m,"
         "physical_liquid_tau_s,physical_liquid_tau_us,physical_liquid_tau_over_hydro,"
         "gas_relative_permittivity,gas_conductivity_S_per_m,physical_gas_tau_s,"
         "physical_gas_tau_over_hydro,normalized_liquid_conductivity,"
         "normalized_gas_conductivity,normalized_liquid_tau,normalized_gas_tau,"
         "dimensionless_dt,physical_dt_s,dt_over_normalized_liquid_tau,"
         "physical_dt_over_liquid_tau,status\n";
  std::ofstream dimensionalElectricalScaling(
      "benchmark_logs/candido_dimensional_electrical_scaling_diagnostic3d.csv");
  dimensionalElectricalScaling
      << "target_ca_e,baseline_relative_charge_budget_residual,"
         "scaled_relative_charge_budget_residual,residual_ratio,"
         "baseline_clamp_correction_l1,scaled_clamp_correction_l1,clamp_ratio,"
         "baseline_max_unclamped_abs_charge,scaled_max_unclamped_abs_charge,"
         "baseline_max_convective_current,scaled_max_convective_current,current_ratio,"
         "baseline_max_velocity,scaled_max_velocity,velocity_ratio,"
         "scaled_alpha_mass_drift,scaled_max_div,status\n";
  std::ofstream electricRelaxationTimestep(
      "benchmark_logs/candido_electric_relaxation_timestep_limit3d.csv");
  electricRelaxationTimestep
      << "low_ca_e,high_ca_e,baseline_low_dt,baseline_high_dt,"
         "limited_low_unrestricted_dt,limited_low_dt,"
         "limited_low_electric_relaxation_dt_limit,"
         "limited_low_dt_over_electric_relaxation_tau,"
         "limited_low_timestep_limited,"
         "limited_high_unrestricted_dt,limited_high_dt,"
         "limited_high_electric_relaxation_dt_limit,"
         "limited_high_dt_over_electric_relaxation_tau,"
         "limited_high_timestep_limited,low_dt_reduction,high_dt_reduction,"
         "baseline_low_tail_current,baseline_high_tail_current,"
         "baseline_tail_current_ratio,limited_low_tail_current,"
         "limited_high_tail_current,limited_tail_current_ratio,"
         "limited_low_relative_charge_budget_residual,"
         "limited_high_relative_charge_budget_residual,"
         "limited_low_charge_budget_expected_final,"
         "limited_low_charge_budget_residual,"
         "limited_low_cumulative_charge_clamp_correction_l1,"
         "limited_low_max_charge_redistribution_residual,"
         "limited_high_charge_budget_expected_final,"
         "limited_high_charge_budget_residual,"
         "limited_high_cumulative_charge_clamp_correction_l1,"
         "limited_high_max_charge_redistribution_residual,"
         "limited_low_alpha_mass_drift,limited_high_alpha_mass_drift,"
         "limited_low_max_div,limited_high_max_div,status\n";
  std::ofstream boundaryChargeAdvection(
      "benchmark_logs/candido_boundary_charge_advection_diagnostic3d.csv");
  boundaryChargeAdvection
      << "low_ca_e,high_ca_e,baseline_low_tail_current,"
         "baseline_high_tail_current,baseline_tail_current_ratio,"
         "advected_low_tail_current,advected_high_tail_current,"
         "advected_tail_current_ratio,baseline_low_relative_charge_budget_residual,"
         "baseline_high_relative_charge_budget_residual,"
         "advected_low_relative_charge_budget_residual,"
         "advected_high_relative_charge_budget_residual,"
         "low_residual_ratio,high_residual_ratio,"
         "baseline_low_cumulative_boundary_charge_flux,"
         "baseline_high_cumulative_boundary_charge_flux,"
         "advected_low_cumulative_boundary_charge_flux,"
         "advected_high_cumulative_boundary_charge_flux,"
         "advected_low_cumulative_conductive_boundary_charge_flux,"
         "advected_high_cumulative_conductive_boundary_charge_flux,"
         "advected_low_alpha_mass_drift,advected_high_alpha_mass_drift,"
         "advected_low_max_div,advected_high_max_div,external_source,status\n";
  std::ofstream interfaceChargeTransport(
      "benchmark_logs/candido_interface_charge_transport_diagnostic3d.csv");
  interfaceChargeTransport
      << "case,low_ca_e,high_ca_e,min_area_di2,"
         "baseline_low_developed_samples,baseline_high_developed_samples,"
         "candidate_low_developed_samples,candidate_high_developed_samples,"
         "baseline_axial_alpha05_current_ratio,"
         "candidate_axial_alpha05_current_ratio,candidate_charge_ratio,"
         "candidate_velocity_ratio,candidate_low_alpha05_current,"
         "candidate_high_alpha05_current,candidate_low_mean_abs_charge,"
         "candidate_high_mean_abs_charge,candidate_low_mean_abs_uy,"
         "candidate_high_mean_abs_uy,candidate_low_clamp_l1,"
         "candidate_high_clamp_l1,candidate_low_redistribution_deficit_l1,"
         "candidate_high_redistribution_deficit_l1,"
         "candidate_low_weighted_cells,candidate_high_weighted_cells,"
         "candidate_low_weighted_capacity,candidate_high_weighted_capacity,"
         "candidate_low_relative_charge_budget_residual,"
         "candidate_high_relative_charge_budget_residual,"
         "candidate_low_alpha_mass_drift,candidate_high_alpha_mass_drift,"
         "candidate_low_max_div,candidate_high_max_div,"
         "candidate_low_morphology_error_percent,"
         "candidate_high_max_radial_asymmetry,status\n";
  std::ofstream postChargePotentialRefresh(
      "benchmark_logs/candido_post_charge_potential_refresh_diagnostic3d.csv");
  postChargePotentialRefresh
      << "case,low_ca_e,high_ca_e,min_area_di2,"
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
  std::ofstream conductivityPotentialChargeClosure(
      "benchmark_logs/candido_conductivity_potential_charge_closure3d.csv");
  conductivityPotentialChargeClosure
      << "case,low_ca_e,high_ca_e,min_area_di2,"
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
         "candidate_low_conductivity_potential_residual,"
         "candidate_high_conductivity_potential_residual,"
         "candidate_low_closure_clamp_l1,candidate_high_closure_clamp_l1,"
         "candidate_low_relative_charge_budget_residual,"
         "candidate_high_relative_charge_budget_residual,"
         "candidate_low_alpha_mass_drift,candidate_high_alpha_mass_drift,"
         "candidate_low_max_div,candidate_high_max_div,"
         "candidate_low_morphology_error_percent,"
         "candidate_high_max_radial_asymmetry,status\n";
  std::ofstream conservativeSurfaceChargeClosure(
      "benchmark_logs/candido_conservative_surface_charge_closure3d.csv");
  conservativeSurfaceChargeClosure
      << "case,low_ca_e,high_ca_e,min_area_di2,"
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
  std::ofstream electricDriveScaling(
      "benchmark_logs/candido_electric_drive_scaling_diagnostic3d.csv");
  electricDriveScaling
      << "case,low_ca_e,high_ca_e,baseline_ca_exponent,tested_ca_exponent,"
         "baseline_tail_current_ratio,tested_tail_current_ratio,"
         "baseline_low_developed_samples,baseline_high_developed_samples,"
         "tested_low_developed_samples,tested_high_developed_samples,"
         "baseline_axial_alpha05_current_ratio,tested_axial_alpha05_current_ratio,"
         "baseline_velocity_ratio,tested_velocity_ratio,"
         "baseline_charge_ratio,tested_charge_ratio,"
         "tested_low_alpha_mass_drift,tested_high_alpha_mass_drift,"
         "tested_low_max_div,tested_high_max_div,status\n";
  std::ofstream maxwellTangentialClosure(
      "benchmark_logs/candido_maxwell_tangential_closure3d.csv");
  maxwellTangentialClosure
      << "target_ca_e,cell_gradient_max_electric_force,"
         "face_normal_max_electric_force,hybrid_max_electric_force,"
         "face_normal_to_cell_force_ratio,hybrid_to_face_force_ratio,"
         "hybrid_to_cell_force_ratio,cell_gradient_max_velocity,"
         "face_normal_max_velocity,hybrid_max_velocity,"
         "hybrid_to_face_velocity_ratio,cell_gradient_alpha_mass_drift,"
         "face_normal_alpha_mass_drift,hybrid_alpha_mass_drift,"
         "cell_gradient_max_div,face_normal_max_div,hybrid_max_div,status\n";
  std::ofstream faceElectricReconstruction(
      "benchmark_logs/candido_face_electric_reconstruction3d.csv");
  faceElectricReconstruction
      << "target_ca_e,sampled_faces,internal_faces,dirichlet_boundary_faces,"
         "traction_ratio_faces,normal_traction_degenerate_faces,"
         "max_poisson_normal_e,max_cell_tangential_e,"
         "mean_relative_normal_mismatch,max_relative_normal_mismatch,"
         "mean_tangential_fraction,max_tangential_fraction,"
         "mean_hybrid_to_normal_traction_ratio,"
         "p95_hybrid_to_normal_traction_ratio,"
         "max_hybrid_to_normal_traction_ratio,potential_residual,status\n";
  std::ofstream boundedVectorMaxwell(
      "benchmark_logs/candido_bounded_vector_maxwell_diagnostic3d.csv");
  boundedVectorMaxwell
      << "target_ca_e,tangential_limit_factor,tangential_limit_floor_fraction,"
         "sampled_faces,tangential_clipped_faces,clipped_fraction,"
         "mean_tangential_clip_ratio,min_tangential_clip_ratio,"
         "max_poisson_normal_e,max_raw_tangential_e,max_limited_tangential_e,"
         "face_normal_max_force,bounded_vector_max_force,"
         "bounded_to_face_normal_force_ratio,max_stress_divergence,"
         "potential_residual,status\n";
  std::ofstream tomarConductingSurfaceForce(
      "benchmark_logs/candido_tomar_conducting_surface_force3d.csv");
  tomarConductingSurfaceForce
      << "target_ca_e,sampled_cells,mixed_cells,active_interface_cells,"
         "max_grad_alpha,max_normal_current,max_tangential_e,"
         "max_normal_term,max_tangential_term,tangential_term_share,"
         "default_max_force,tomar_max_force,tomar_to_default_force_ratio,"
         "potential_residual,status\n";
  std::ofstream hybridMaxwellLongWindow(
      "benchmark_logs/candido_hybrid_maxwell_long_window3d.csv");
  hybridMaxwellLongWindow
      << "low_ca_e,high_ca_e,baseline_tail_current_ratio,"
         "hybrid_tail_current_ratio,baseline_low_developed_samples,"
         "baseline_high_developed_samples,hybrid_low_developed_samples,"
         "hybrid_high_developed_samples,baseline_axial_alpha05_current_ratio,"
         "hybrid_axial_alpha05_current_ratio,baseline_velocity_ratio,"
         "hybrid_velocity_ratio,baseline_charge_ratio,hybrid_charge_ratio,"
         "baseline_low_max_electric_force,baseline_high_max_electric_force,"
         "hybrid_low_max_electric_force,hybrid_high_max_electric_force,"
         "hybrid_low_alpha_mass_drift,hybrid_high_alpha_mass_drift,"
         "hybrid_low_max_div,hybrid_high_max_div,status\n";
  std::ofstream boundedVectorMaxwellLongWindow(
      "benchmark_logs/candido_bounded_vector_maxwell_long_window3d.csv");
  boundedVectorMaxwellLongWindow
      << "low_ca_e,high_ca_e,baseline_tail_current_ratio,"
         "bounded_tail_current_ratio,baseline_low_developed_samples,"
         "baseline_high_developed_samples,bounded_low_developed_samples,"
         "bounded_high_developed_samples,baseline_axial_alpha05_current_ratio,"
         "bounded_axial_alpha05_current_ratio,baseline_velocity_ratio,"
         "bounded_velocity_ratio,baseline_charge_ratio,bounded_charge_ratio,"
         "baseline_low_max_electric_force,baseline_high_max_electric_force,"
         "bounded_low_max_electric_force,bounded_high_max_electric_force,"
         "bounded_low_alpha_mass_drift,bounded_high_alpha_mass_drift,"
         "bounded_low_max_div,bounded_high_max_div,status\n";
  std::ofstream tomarConductingLongWindow(
      "benchmark_logs/candido_tomar_conducting_long_window3d.csv");
  tomarConductingLongWindow
      << "low_ca_e,high_ca_e,baseline_tail_current_ratio,"
         "tomar_tail_current_ratio,baseline_low_developed_samples,"
         "baseline_high_developed_samples,tomar_low_developed_samples,"
         "tomar_high_developed_samples,baseline_axial_alpha05_current_ratio,"
         "tomar_axial_alpha05_current_ratio,tomar_velocity_ratio,"
         "tomar_charge_ratio,tomar_low_max_electric_force,"
         "tomar_high_max_electric_force,tomar_low_alpha_mass_drift,"
         "tomar_high_alpha_mass_drift,tomar_low_max_div,tomar_high_max_div,"
         "status\n";
  std::ofstream caIndependentCurrentResolution(
      "benchmark_logs/candido_ca_independent_current_resolution_sweep3d.csv");
  caIndependentCurrentResolution
      << "case,nx,ny,nz,low_cells,high_cells,low_dt,high_dt,"
         "low_developed_samples,high_developed_samples,"
         "low_mean_area_di2,high_mean_area_di2,tail_current_ratio,"
         "axial_alpha05_current_ratio,velocity_ratio,charge_ratio,"
         "low_alpha_mass_drift,high_alpha_mass_drift,low_max_div,high_max_div,"
         "status\n";
  std::ofstream chargeRelaxation(
      "benchmark_logs/candido_charge_relaxation_diagnostic3d.csv");
  chargeRelaxation << "target_ca_e,baseline_relative_charge_budget_residual,"
                      "relaxed_relative_charge_budget_residual,"
                      "baseline_clamp_correction_l1,relaxed_clamp_correction_l1,"
                      "relaxation_sink,baseline_max_unclamped_abs_charge,"
                      "relaxed_max_unclamped_abs_charge,baseline_max_convective_current,"
                      "relaxed_max_convective_current,current_ratio,baseline_max_velocity,"
                      "relaxed_max_velocity,velocity_ratio,relaxed_alpha_mass_drift,"
                      "relaxed_max_div,status\n";
  std::ofstream boundaryCurrent(
      "benchmark_logs/candido_boundary_current_decomposition3d.csv");
  boundaryCurrent << "case,target_ca_e,patch,cumulative_conductive_charge_flux,"
                     "max_abs_conductive_boundary_current,"
                     "fraction_of_total_cumulative_conductive_flux,"
                     "total_cumulative_conductive_boundary_flux,"
                     "max_face_conductive_current,status\n";
  std::ofstream boundaryCurrentSensitivity(
      "benchmark_logs/candido_boundary_current_sensitivity3d.csv");
  boundaryCurrentSensitivity
      << "case,low_ca_e,high_ca_e,low_total_cumulative_conductive_flux,"
         "high_total_cumulative_conductive_flux,total_cumulative_ratio,"
         "low_nozzle_cumulative_flux,high_nozzle_cumulative_flux,nozzle_ratio,"
         "low_collector_cumulative_flux,high_collector_cumulative_flux,"
         "collector_ratio,low_lateral_cumulative_flux,"
         "high_lateral_cumulative_flux,lateral_ratio,low_nozzle_fraction,"
         "high_nozzle_fraction,low_collector_fraction,high_collector_fraction,"
         "low_lateral_fraction,high_lateral_fraction,"
         "low_nozzle_peak_current,high_nozzle_peak_current,nozzle_peak_ratio,"
         "low_collector_peak_current,high_collector_peak_current,"
         "collector_peak_ratio,low_lateral_peak_current,high_lateral_peak_current,"
         "lateral_peak_ratio,dominant_high_patch,status\n";
  std::ofstream nozzleBoundary(
      "benchmark_logs/candido_nozzle_charge_boundary_diagnostic3d.csv");
  nozzleBoundary << "target_ca_e,baseline_relative_charge_budget_residual,"
                    "suppressed_relative_charge_budget_residual,"
                    "baseline_ymin_nozzle_cumulative_flux,"
                    "suppressed_ymin_nozzle_cumulative_flux,nozzle_flux_ratio,"
                    "baseline_total_cumulative_conductive_flux,"
                    "suppressed_total_cumulative_conductive_flux,"
                    "baseline_max_convective_current,suppressed_max_convective_current,"
                    "current_ratio,baseline_max_velocity,suppressed_max_velocity,"
                    "velocity_ratio,suppressed_alpha_mass_drift,suppressed_max_div,status\n";
  std::ofstream chargeReferenceGap(
      "benchmark_logs/candido_charge_model_reference_gap3d.csv");
  chargeReferenceGap
      << "item,local_evidence,reference_evidence,missing_requirement,status\n";
  chargeReferenceGap
      << "bulk_charge_conservation,"
      << "candidoAdvanceCharge3D_tracks_conduction_plus_convection_but_uses_qLimit,"
      << "Candido_eq6_bulk_free_charge_conservation_and_LopezHerrera_charge_conservative_VOF,"
      << "replace_arbitrary_qLimit_with_dimensional_charge_scaling_and_conservative_bounded_transport,"
      << "BLOCKED_PHYSICAL_CHARGE_SCALING_UNSPECIFIED\n";
  chargeReferenceGap
      << "current_boundary_treatment,"
      << "Candido_smoke_uses_internal_cross_section_rhoE_u_current_and_diagnostic_boundary_fluxes,"
      << "Candido_boundary_text_uses_charge_Neumann_or_zero_gradient_and_static_potential_assumption,"
      << "implement_paper_faithful_electrode_nozzle_outlet_charge_current_boundary_conditions,"
      << "BLOCKED_BOUNDARY_CURRENT_MODEL_UNSPECIFIED\n";
  chargeReferenceGap
      << "voltage_sensitivity,"
      << "default_and_combined_paths_both_fail_weak_average_current_voltage_sensitivity,"
      << "Candido_Fig8b_text_average_current_weakly_influenced_by_electric_potential,"
      << "calibrate_current_observable_against_physical_units_or_external_current_dataset,"
      << "DOWNGRADED_CURRENT_NOT_CALIBRATED\n";
  std::ofstream refinement("benchmark_logs/candido_refinement_sweep3d.csv");
  refinement << "case,nx,ny,nz,cells,faces,steps,target_ca_e,dt,alpha_mass_drift,"
                "max_div,max_electric_force,max_csf_force,final_midplane_jet_radius,"
                "final_radial_asymmetry,max_velocity\n";
  std::ofstream refinementQuality("benchmark_logs/candido_refinement_quality3d.csv");
  refinementQuality << "observable,coarse_n,mid_n,fine_n,coarse_value,mid_value,fine_value,"
                       "coarse_to_mid_relative_change,mid_to_fine_relative_change,status\n";
  std::ofstream morphologyError("benchmark_logs/candido_guo_morphology_error3d.csv");
  morphologyError << "case,reference_time_ms,nearest_sim_time_ms,paper_reported_error_percent,"
                     "sim_tip_y,sim_centroid_y,sim_radial_asymmetry,sim_volume_di3,"
                     "connected_proxy_volume_di3,disconnected_proxy_volume_di3,"
                     "alpha05_silhouette_volume_di3,ray_alpha05_silhouette_volume_di3,"
                     "all_liquid_ray_alpha05_silhouette_volume_di3,"
                     "ray_alpha05_cell_boundary_silhouette_volume_di3,"
                     "linear_ray_alpha05_silhouette_volume_di3,"
                     "plic_contour_silhouette_volume_di3,"
                     "plic_polygon_silhouette_volume_di3,"
                     "plic_sector_median_silhouette_volume_di3,"
                     "plic_ray_plane_silhouette_volume_di3,"
                     "plic_ray_plane_q25_silhouette_volume_di3,"
                     "plic_first_exit_silhouette_volume_di3,"
                     "outer_envelope_alpha05_silhouette_volume_di3,"
                     "digitized_experimental_volume_di3,computed_relative_error_percent,"
                     "connected_proxy_error_percent,alpha05_silhouette_error_percent,"
                     "ray_alpha05_silhouette_error_percent,"
                     "all_liquid_ray_alpha05_silhouette_error_percent,"
                     "ray_alpha05_cell_boundary_silhouette_error_percent,"
                     "linear_ray_alpha05_silhouette_error_percent,"
                     "plic_contour_error_percent,"
                     "plic_polygon_error_percent,"
                     "plic_sector_median_error_percent,"
                     "plic_ray_plane_error_percent,"
                     "plic_ray_plane_q25_error_percent,"
                     "plic_first_exit_error_percent,"
                     "outer_envelope_alpha05_silhouette_error_percent,"
                     "disconnected_proxy_percent_of_reference,"
                     "external_source,status\n";
  std::ofstream morphologyAlignment("benchmark_logs/candido_morphology_time_alignment3d.csv");
  morphologyAlignment << "case,reference_time_ms,digitized_experimental_volume_di3,"
                         "fixed_sim_time_ms,fixed_sim_volume_di3,fixed_error_percent,"
                         "best_volume_sim_time_ms,best_volume_sim_volume_di3,"
                         "best_volume_error_percent,time_lag_ms\n";
  std::ofstream morphologyPhaseLag("benchmark_logs/candido_morphology_phase_lag_diagnostic3d.csv");
  morphologyPhaseLag << "case,reference_time_ms,digitized_experimental_volume_di3,"
                        "fixed_sim_time_ms,fixed_sim_volume_di3,fixed_error_percent,"
                        "best_volume_sim_time_ms,best_volume_sim_volume_di3,"
                        "best_volume_error_percent,time_lag_ms,local_slope_di3_per_ms,"
                        "lag_explained_volume_di3,phase_explained_fraction,status\n";
  std::ofstream morphologyTipSync("benchmark_logs/candido_morphology_tip_sync_diagnostic3d.csv");
  morphologyTipSync << "case,history_samples,unique_tip_levels,initial_tip_y,max_tip_y,"
                       "min_nonzero_tip_step,max_tip_time_ms,first_tip_jump_time_ms,"
                       "paper_sync_offset_ms,sync_0_4_sim_time_ms,sync_0_4_volume_di3,"
                       "sync_0_4_error_percent,sync_0_4_connected_ray_alpha05_di3,"
                       "sync_0_4_connected_ray_alpha05_error_percent,"
                       "sync_0_4_all_liquid_ray_alpha05_di3,"
                       "sync_0_4_all_liquid_ray_alpha05_error_percent,"
                       "sync_0_7_sim_time_ms,sync_0_7_volume_di3,"
                       "sync_0_7_error_percent,sync_0_7_connected_ray_alpha05_di3,"
                       "sync_0_7_connected_ray_alpha05_error_percent,"
                       "sync_0_7_all_liquid_ray_alpha05_di3,"
                       "sync_0_7_all_liquid_ray_alpha05_error_percent,"
                       "external_source,alpha05_status,status\n";
  std::ofstream interfacePreservation(
      "benchmark_logs/candido_interface_preservation_candidate3d.csv");
  interfacePreservation
      << "case,vof_compression,vof_post_sharpening,vof_post_sharpening_sweeps,"
         "vof_inlet_boundary_alpha,alpha_mass_drift,max_div,max_alpha,"
         "unique_tip_levels,max_tip_time_ms,"
         "paper_sync_offset_ms,sync_0_4_sim_time_ms,sync_0_4_volume_di3,"
         "sync_0_4_volume_error_percent,sync_0_4_all_liquid_ray_alpha05_di3,"
         "sync_0_4_all_liquid_ray_alpha05_error_percent,sync_0_7_sim_time_ms,"
         "sync_0_7_volume_di3,sync_0_7_volume_error_percent,"
         "sync_0_7_all_liquid_ray_alpha05_di3,"
         "sync_0_7_all_liquid_ray_alpha05_error_percent,status\n";
  std::ofstream lateBlocker("benchmark_logs/candido_late_morphology_blocker3d.csv");
  writeLateMorphologyBlockerRows(lateBlocker);
  std::ofstream whippingDiagnostic("benchmark_logs/candido_whipping_diagnostic3d.csv");
  whippingDiagnostic << "case,target_ca_e,voltage,asymmetry_threshold,"
                        "max_radial_asymmetry,onset_time_ms,onset_tip_y_over_Di,"
                        "onset_centroid_y_over_Di,max_asymmetry_time_ms,"
                        "max_asymmetry_tip_y_over_Di,max_asymmetry_centroid_y_over_Di,"
                        "wave_peak_time_ms,wave_peak_y_over_Di,wave_peak_asymmetry,"
                        "wave_speed_Di_per_sh,early_to_peak_wave_speed_Di_per_sh,"
                        "wave_speed_fit_count,"
                        "paper_initiation_y_over_Di,onset_location_error_percent,"
                        "max_asymmetry_location_error_percent,"
                        "wave_peak_location_error_percent,external_source,status\n";

  electrospray::CandidoTaylorConeJetSetup setup;
  electrospray::CandidoConeJetSmokeOptions3D opt;
  opt.nx = 10;
  opt.ny = 14;
  opt.nz = 10;
  opt.steps = 3;
  opt.skew = 0.04;
  writeBoundaryConditionDiagnostics(setup, opt, 0.25);
  writeContactAngleDiagnostic(setup, opt, 0.25);
  writeContactAngleCurvatureDiagnostic(setup, opt, 0.25);
  writeContactAngleCurvatureSwitchDiagnostic(setup, opt, 0.25);
  writeMaxwellTangentialClosureDiagnosticRow(maxwellTangentialClosure, setup, opt, 0.25);
  writeMaxwellTangentialClosureDiagnosticRow(maxwellTangentialClosure, setup, opt, 0.42);
  writeFaceElectricReconstructionDiagnosticRow(faceElectricReconstruction, setup, opt, 0.25);
  writeFaceElectricReconstructionDiagnosticRow(faceElectricReconstruction, setup, opt, 0.42);
  writeBoundedVectorMaxwellDiagnosticRows(boundedVectorMaxwell, setup, opt, 0.25);
  writeBoundedVectorMaxwellDiagnosticRows(boundedVectorMaxwell, setup, opt, 0.42);
  writeTomarConductingSurfaceForceDiagnosticRows(tomarConductingSurfaceForce,
                                                 setup, opt, 0.25);
  writeTomarConductingSurfaceForceDiagnosticRows(tomarConductingSurfaceForce,
                                                 setup, opt, 0.42);

  std::vector<double> cases = {0.25, 0.42};
  electrospray::CandidoConeJetSmokeReport3D low;
  electrospray::CandidoConeJetSmokeReport3D high;
  for (double caE : cases) {
    auto r = electrospray::runCandidoConeJetSmoke3D(caE, setup, opt);
    const std::string name = caE < 0.3 ? "validation_ca025" : "whipping_ca042";
    writeRow(csv, name, r);
    writeHistoryRows(history, name, r);
    writeMorphologyObservableAuditRows(morphologyAudit, name, r);
    writePhysicalTimeRows(physicalTime, name, r);
    writeMorphologyReferenceGapRows(morphologyError, name, r);
    jetCurrent << name << "," << r.targetCaE << "," << r.voltage << ","
               << r.computedCaE << "," << r.finalMidplaneJetRadius << ","
               << r.maxConductiveCurrent << "," << r.maxConvectiveCurrent << ","
               << r.finalRadialAsymmetry << "," << r.maxVelocity << ","
               << r.alphaMassDrift << "," << r.maxDiv << "\n";
    checkSmokeReport(r, caE);
    if (caE < 0.3) low = r;
    else high = r;
  }

  check(high.voltage > low.voltage, "Candido cone-jet voltage increases with CaE");
  check(high.maxElectricForce >= 0.5 * low.maxElectricForce,
        "Candido cone-jet high-CaE electric force remains comparable or stronger");
  check(high.finalRadialAsymmetry >= 0.0 && low.finalRadialAsymmetry >= 0.0,
        "Candido cone-jet radial asymmetry metric finite");

  electrospray::CandidoConeJetSmokeReport3D previous;
  bool havePrevious = false;
  for (double caE : {0.12, 0.25, 0.42}) {
    auto r = electrospray::runCandidoConeJetSmoke3D(caE, setup, opt);
    const std::string name = caE < 0.2 ? "low_ca012" : (caE < 0.3 ? "validation_ca025" : "whipping_ca042");
    currentScaling << name << "," << r.targetCaE << "," << r.voltage << ","
                   << r.computedCaE << "," << r.maxConductiveCurrent << ","
                   << r.maxConvectiveCurrent << "," << r.maxElectricForce << ","
                   << r.maxVelocity << "," << r.finalMidplaneJetRadius << ","
                   << r.alphaMassDrift << "," << r.maxDiv << "\n";
    writeCurrentScalingValidationRow(currentScalingValidation, name, setup, r);
    checkSmokeReport(r, caE);
    if (havePrevious) {
      check(r.voltage > previous.voltage, "Candido current-scaling voltage increases with CaE");
      check(r.maxElectricForce >= 0.5 * previous.maxElectricForce,
            "Candido current-scaling electric force remains comparable under CaE increase");
    }
    previous = r;
    havePrevious = true;
  }

  struct RefinementRow {
    int n;
    double radius;
    double electricForce;
    double csfForce;
  };
  std::vector<RefinementRow> refinementRows;
  for (int n : {10, 12, 14}) {
    electrospray::CandidoConeJetSmokeOptions3D ropt = opt;
    ropt.nx = n;
    ropt.nz = n;
    ropt.ny = std::max(8, static_cast<int>(std::round(1.4 * n)));
    ropt.steps = 2;
    auto r = electrospray::runCandidoConeJetSmoke3D(0.25, setup, ropt);
    refinement << "refine_" << n << "," << ropt.nx << "," << ropt.ny << ","
               << ropt.nz << "," << r.cells << "," << r.faces << "," << r.steps << ","
               << r.targetCaE << "," << r.dt << "," << r.alphaMassDrift << ","
               << r.maxDiv << "," << r.maxElectricForce << "," << r.maxCsfForce << ","
               << r.finalMidplaneJetRadius << "," << r.finalRadialAsymmetry << ","
               << r.maxVelocity << "\n";
    check(r.cells > 0 && r.faces > 0, "Candido refinement sweep mesh is non-empty");
    check(r.alphaMassDrift <= 1e-3, "Candido refinement sweep mass drift bounded");
    check(r.maxDiv <= 1e-7, "Candido refinement sweep continuity bounded");
    check(std::isfinite(r.maxElectricForce) && std::isfinite(r.finalMidplaneJetRadius),
          "Candido refinement sweep metrics finite");
    refinementRows.push_back({n, r.finalMidplaneJetRadius, r.maxElectricForce, r.maxCsfForce});
  }
  const auto writeRefinementQualityRow = [&](const std::string& observable,
                                             double RefinementRow::*member) {
    check(refinementRows.size() == 3, "Candido refinement quality has three grid levels");
    const double coarse = refinementRows[0].*member;
    const double mid = refinementRows[1].*member;
    const double fine = refinementRows[2].*member;
    const double denom = std::max(std::abs(fine), 1e-30);
    const double coarseToMid = std::abs(mid - coarse) / denom;
    const double midToFine = std::abs(fine - mid) / denom;
    const bool decreasing = midToFine < coarseToMid;
    const bool boundedFineChange = midToFine <= 0.35;
    const std::string status =
        (decreasing && boundedFineChange) ? "PASS_CONVERGING" : "DOWNGRADED_NONCONVERGENT";
    refinementQuality << observable << "," << refinementRows[0].n << ","
                      << refinementRows[1].n << "," << refinementRows[2].n << ","
                      << coarse << "," << mid << "," << fine << ","
                      << coarseToMid << "," << midToFine << "," << status << "\n";
  };
  writeRefinementQualityRow("final_midplane_jet_radius", &RefinementRow::radius);
  writeRefinementQualityRow("max_electric_force", &RefinementRow::electricForce);
  writeRefinementQualityRow("max_csf_force", &RefinementRow::csfForce);

  electrospray::CandidoConeJetSmokeOptions3D longOpt = opt;
  longOpt.nx = 12;
  longOpt.nz = 12;
  longOpt.ny = 17;
  longOpt.steps = 52;
  longOpt.cfl = 1.0;
  // The morphology long-window diagnostic operates on the hydrodynamic timescale and must
  // reach the paper reference window (~0.9 ms) within a small fixed step budget. The faithful
  // production electric-relaxation timestep (dt <= 0.1*tau_e) makes that ~20x more steps, so
  // this fast smoke diagnostic uses the hydrodynamic dt (electric-relaxation limit off,
  // normalized conductivity) to stay affordable; production defaults remain faithful.
  longOpt.useElectricRelaxationTimeStepLimit = false;
  longOpt.useDimensionalElectricalScaling = false;
  auto longWindow = electrospray::runCandidoConeJetSmoke3D(0.25, setup, longOpt);
  writeHistoryRows(history, "long_window_ca025", longWindow);
  writeMorphologyObservableAuditRows(morphologyAudit, "long_window_ca025", longWindow);
  writeMorphologySilhouetteBracketRows(morphologyBracket, "long_window_ca025", longWindow);
  writePhysicalTimeRows(physicalTime, "long_window_ca025", longWindow);
  writeMorphologyReferenceGapRows(morphologyError, "long_window_ca025", longWindow);
  writeMorphologyTimeAlignmentRows(morphologyAlignment, "long_window_ca025", longWindow);
  writeMorphologyPhaseLagDiagnosticRows(morphologyPhaseLag, "long_window_ca025", longWindow);
  writeMorphologyTipSyncDiagnosticRow(morphologyTipSync, "long_window_ca025", longWindow);
  writeCurrentScalingValidationRow(currentScalingValidation, "long_window_ca025", setup, longWindow);
  writeCurrentBlowupDiagnosticRow(currentBlowup, "long_window_ca025", setup, longWindow);
  writeLongWindowMassBudgetRow(massBudget, "long_window_ca025", longWindow);
  writeLongWindowChargeBudgetRow(chargeBudget, "long_window_ca025", longWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "long_window_ca025", longWindow);
  writeChargeScaleAuditRow(chargeScaleAudit, "long_window_ca025", longOpt.chargeLimitBase,
                           setup, longWindow);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "long_window_ca025", setup,
                                longWindow);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling, "long_window_ca025",
                                       setup, longOpt, longWindow);
  check(longWindow.history.back().time * longWindow.hydrodynamicTimeScale * 1.0e3 >= 0.9,
        "Candido long-window morphology run reaches the paper reference time window");
  check(longWindow.alphaMassDrift <= 1e-3, "Candido long-window morphology mass drift bounded");
  check(longWindow.maxDiv <= 1e-7, "Candido long-window morphology continuity bounded");
  writeInterfacePreservationCandidateRow(interfacePreservation, "baseline_long_window_ca025",
                                         longOpt, longWindow);

  electrospray::CandidoConeJetSmokeOptions3D sharpenedLongOpt = longOpt;
  sharpenedLongOpt.vofPostSharpening = 0.6;
  sharpenedLongOpt.vofPostSharpeningSweeps = 1;
  auto sharpenedLongWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, sharpenedLongOpt);
  writeInterfacePreservationCandidateRow(interfacePreservation,
                                         "post_sharpen_0_6_long_window_ca025",
                                         sharpenedLongOpt, sharpenedLongWindow);
  check(sharpenedLongWindow.alphaMassDrift <= 1e-3,
        "Candido post-sharpen interface-preservation candidate mass drift bounded");
  check(sharpenedLongWindow.maxDiv <= 1e-7,
        "Candido post-sharpen interface-preservation candidate continuity bounded");

  electrospray::CandidoConeJetSmokeOptions3D inletAlphaLongOpt = longOpt;
  inletAlphaLongOpt.useVofInletBoundaryAlpha = true;
  auto inletAlphaLongWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, inletAlphaLongOpt);
  writeInterfacePreservationCandidateRow(interfacePreservation,
                                         "inlet_boundary_alpha_long_window_ca025",
                                         inletAlphaLongOpt, inletAlphaLongWindow);
  check(inletAlphaLongWindow.alphaMassDrift <= 1e-3,
        "Candido inlet-boundary-alpha interface-preservation candidate mass drift bounded");
  check(inletAlphaLongWindow.maxDiv <= 1e-7,
        "Candido inlet-boundary-alpha interface-preservation candidate continuity bounded");

  electrospray::CandidoConeJetSmokeOptions3D inletAlphaCompressedLongOpt = longOpt;
  inletAlphaCompressedLongOpt.useVofInletBoundaryAlpha = true;
  inletAlphaCompressedLongOpt.vofCompression = 0.2;
  auto inletAlphaCompressedLongWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, inletAlphaCompressedLongOpt);
  writeInterfacePreservationCandidateRow(
      interfacePreservation, "inlet_boundary_alpha_compression_0_2_long_window_ca025",
      inletAlphaCompressedLongOpt, inletAlphaCompressedLongWindow);
  check(inletAlphaCompressedLongWindow.alphaMassDrift <= 1e-3,
        "Candido inlet-boundary-alpha compressed candidate mass drift bounded");
  check(inletAlphaCompressedLongWindow.maxDiv <= 1e-7,
        "Candido inlet-boundary-alpha compressed candidate continuity bounded");

  electrospray::CandidoConeJetSmokeOptions3D whipOpt = longOpt;
  auto longWhip = electrospray::runCandidoConeJetSmoke3D(0.42, setup, whipOpt);
  writeHistoryRows(history, "long_window_ca042", longWhip);
  writeMorphologyObservableAuditRows(morphologyAudit, "long_window_ca042", longWhip);
  writePhysicalTimeRows(physicalTime, "long_window_ca042", longWhip);
  writeCurrentScalingValidationRow(currentScalingValidation, "long_window_ca042", setup, longWhip);
  writeWhippingDiagnosticRow(whippingDiagnostic, "long_window_ca042", setup, longWhip);
  writeCurrentVoltageSensitivityRow(currentVoltageSensitivity, longWindow, longWhip);
  writeCurrentVoltageSensitivityRow(totalCurrentVoltageSensitivity, longWindow, longWhip, true);
  writeCurrentVoltageSensitivityRowForObservable(
      liquidJetCurrentVoltageSensitivity, longWindow, longWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective);
  writeCurrentVoltageSensitivityRowForObservable(
      alpha05JetCurrentVoltageSensitivity, longWindow, longWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "long_window_convective", longWindow, longWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "long_window_liquid_convective", longWindow, longWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "long_window_alpha05_convective", longWindow, longWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "long_window_convective", longWindow, longWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "long_window_liquid_convective", longWindow, longWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "long_window_alpha05_convective", longWindow, longWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "long_window_alpha05", longWindow, longWhip, 1e-4);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "long_window_alpha05", longWindow, longWhip, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep, "long_window_alpha05", longWindow, longWhip);
  writeMidplaneCurrentReachDiagnosticRow(
      midplaneCurrentReach, "baseline_long_window_fixed_midplane",
      longWindow, longWhip, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D extendedMidplaneOpt = longOpt;
  extendedMidplaneOpt.steps = 90;
  auto extendedMidplaneLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, extendedMidplaneOpt);
  auto extendedMidplaneHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, extendedMidplaneOpt);
  writeMidplaneCurrentReachDiagnosticRow(
      midplaneCurrentReach, "extended_90_step_fixed_midplane",
      extendedMidplaneLow, extendedMidplaneHigh, 1e-4);
  writeCurrentBlowupDiagnosticRow(currentBlowup, "long_window_ca042", setup, longWhip);
  writeLongWindowMassBudgetRow(massBudget, "long_window_ca042", longWhip);
  writeLongWindowChargeBudgetRow(chargeBudget, "long_window_ca042", longWhip);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "long_window_ca042", longWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity, "long_window",
                                     longWindow, longWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "long_window_ca042", whipOpt.chargeLimitBase,
                           setup, longWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "long_window_ca042", setup,
                                longWhip);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling, "long_window_ca042",
                                       setup, whipOpt, longWhip);
  electrospray::CandidoConeJetSmokeOptions3D subcycleOpt = whipOpt;
  subcycleOpt.chargeSubcycles = 8;
  auto subcycledWhip = electrospray::runCandidoConeJetSmoke3D(0.42, setup, subcycleOpt);
  writeChargeSubcyclingDiagnosticRow(chargeSubcycling, longWhip, subcycledWhip,
                                     subcycleOpt.chargeSubcycles);
  electrospray::CandidoConeJetSmokeOptions3D boundedChargeOpt = whipOpt;
  boundedChargeOpt.conservativeChargeBounding = true;
  auto boundedChargeWhip = electrospray::runCandidoConeJetSmoke3D(0.42, setup, boundedChargeOpt);
  writeChargeConservativeBoundingDiagnosticRow(conservativeChargeBounding, longWhip,
                                               boundedChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D combinedChargeOpt = subcycleOpt;
  combinedChargeOpt.conservativeChargeBounding = true;
  auto combinedLowWindow = electrospray::runCandidoConeJetSmoke3D(0.25, setup, combinedChargeOpt);
  auto combinedChargeWhip = electrospray::runCandidoConeJetSmoke3D(0.42, setup, combinedChargeOpt);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "combined_low_ca025",
                                        combinedLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "combined_high_ca042",
                                        combinedChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "combined_charge_bounding_subcycled",
                                     combinedLowWindow, combinedChargeWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "combined_low_ca025",
                           combinedChargeOpt.chargeLimitBase, setup, combinedLowWindow);
  writeChargeScaleAuditRow(chargeScaleAudit, "combined_high_ca042",
                           combinedChargeOpt.chargeLimitBase, setup, combinedChargeWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "combined_low_ca025", setup,
                                combinedLowWindow);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "combined_high_ca042", setup,
                                combinedChargeWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "combined_low_ca025",
                                 combinedChargeOpt.chargeLimitBase, setup,
                                 combinedLowWindow);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "combined_high_ca042",
                                 combinedChargeOpt.chargeLimitBase, setup,
                                 combinedChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency, "combined_low_ca025",
                                 combinedLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency, "combined_high_ca042",
                                 combinedChargeWhip);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling, "combined_low_ca025",
                                       setup, combinedChargeOpt, combinedLowWindow);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling, "combined_high_ca042",
                                       setup, combinedChargeOpt, combinedChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D dimensionalElectricalOpt = combinedChargeOpt;
  dimensionalElectricalOpt.useDimensionalElectricalScaling = true;
  auto dimensionalElectricalWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, dimensionalElectricalOpt);
  writeDimensionalElectricalScalingDiagnosticRow(dimensionalElectricalScaling,
                                                combinedChargeWhip,
                                                dimensionalElectricalWhip);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling,
                                       "dimensional_electrical_high_ca042", setup,
                                       dimensionalElectricalOpt, dimensionalElectricalWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "dimensional_electrical_high_ca042",
                           dimensionalElectricalOpt.chargeLimitBase, setup,
                           dimensionalElectricalWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency,
                                "dimensional_electrical_high_ca042", setup,
                                dimensionalElectricalWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale,
                                 "dimensional_electrical_high_ca042",
                                 dimensionalElectricalOpt.chargeLimitBase, setup,
                                 dimensionalElectricalWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "dimensional_electrical_high_ca042",
                                 dimensionalElectricalWhip);
  electrospray::CandidoConeJetSmokeOptions3D dimensionalImplicitBulkOpt =
      dimensionalElectricalOpt;
  dimensionalImplicitBulkOpt.quasiImplicitBulkConduction = true;
  auto dimensionalImplicitBulkWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, dimensionalImplicitBulkOpt);
  writeDimensionalElectricalScalingDiagnosticRow(dimensionalElectricalScaling,
                                                combinedChargeWhip,
                                                dimensionalImplicitBulkWhip);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling,
                                       "dimensional_implicit_bulk_high_ca042", setup,
                                       dimensionalImplicitBulkOpt,
                                       dimensionalImplicitBulkWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "dimensional_implicit_bulk_high_ca042",
                           dimensionalImplicitBulkOpt.chargeLimitBase, setup,
                           dimensionalImplicitBulkWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency,
                                "dimensional_implicit_bulk_high_ca042", setup,
                                dimensionalImplicitBulkWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale,
                                 "dimensional_implicit_bulk_high_ca042",
                                 dimensionalImplicitBulkOpt.chargeLimitBase, setup,
                                 dimensionalImplicitBulkWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "dimensional_implicit_bulk_high_ca042",
                                 dimensionalImplicitBulkWhip);
  electrospray::CandidoConeJetSmokeOptions3D nozzleSuppressedOpt = combinedChargeOpt;
  nozzleSuppressedOpt.suppressNozzleConductiveChargeFlux = true;
  auto nozzleSuppressedWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, nozzleSuppressedOpt);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "nozzle_suppressed_high_ca042",
                                        nozzleSuppressedWhip);
  writeNozzleChargeBoundaryDiagnosticRow(nozzleBoundary, combinedChargeWhip,
                                         nozzleSuppressedWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "nozzle_suppressed_high_ca042",
                           nozzleSuppressedOpt.chargeLimitBase, setup, nozzleSuppressedWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "nozzle_suppressed_high_ca042",
                                setup, nozzleSuppressedWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale,
                                 "nozzle_suppressed_high_ca042",
                                 nozzleSuppressedOpt.chargeLimitBase, setup,
                                 nozzleSuppressedWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "nozzle_suppressed_high_ca042",
                                 nozzleSuppressedWhip);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling,
                                       "nozzle_suppressed_high_ca042", setup,
                                       nozzleSuppressedOpt, nozzleSuppressedWhip);
  writeChargeConservativeBoundingDiagnosticRow(combinedChargeBoundingSubcycling, longWhip,
                                               combinedChargeWhip);
  writeCurrentVoltageSensitivityRow(combinedCurrentVoltageSensitivity, combinedLowWindow,
                                    combinedChargeWhip);
  writeCurrentVoltageSensitivityRowForObservable(
      liquidJetCurrentVoltageSensitivity, combinedLowWindow, combinedChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective);
  writeCurrentVoltageSensitivityRowForObservable(
      alpha05JetCurrentVoltageSensitivity, combinedLowWindow, combinedChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective);
  electrospray::CandidoConeJetSmokeOptions3D rayleighLimitOpt = combinedChargeOpt;
  rayleighLimitOpt.useRayleighChargeLimit = true;
  auto rayleighLowWindow = electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                                                  rayleighLimitOpt);
  auto rayleighChargeWhip = electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                                                   rayleighLimitOpt);
  const double rayleighLowEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(setup,
                                                            rayleighLowWindow.voltage) /
      std::max(1.0, rayleighLowWindow.targetCaE / 0.25);
  const double rayleighHighEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(setup,
                                                            rayleighChargeWhip.voltage) /
      std::max(1.0, rayleighChargeWhip.targetCaE / 0.25);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "rayleigh_limited_low_ca025",
                                        rayleighLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "rayleigh_limited_high_ca042",
                                        rayleighChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity, "rayleigh_limited",
                                     rayleighLowWindow, rayleighChargeWhip);
  writeCurrentVoltageSensitivityRow(rayleighCurrentVoltageSensitivity,
                                    rayleighLowWindow, rayleighChargeWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "rayleigh_limited_low_ca025",
                           rayleighLowEquivalentBase, setup, rayleighLowWindow);
  writeChargeScaleAuditRow(chargeScaleAudit, "rayleigh_limited_high_ca042",
                           rayleighHighEquivalentBase, setup, rayleighChargeWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "rayleigh_limited_low_ca025",
                                setup, rayleighLowWindow);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "rayleigh_limited_high_ca042",
                                setup, rayleighChargeWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "rayleigh_limited_low_ca025",
                                 rayleighLowEquivalentBase, setup,
                                 rayleighLowWindow);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "rayleigh_limited_high_ca042",
                                 rayleighHighEquivalentBase, setup,
                                 rayleighChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "rayleigh_limited_low_ca025",
                                 rayleighLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "rayleigh_limited_high_ca042",
                                 rayleighChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D collectorOnlyOpt = rayleighLimitOpt;
  collectorOnlyOpt.collectorOnlyConductiveChargeFlux = true;
  auto collectorOnlyLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, collectorOnlyOpt);
  auto collectorOnlyChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, collectorOnlyOpt);
  const double collectorOnlyLowEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, collectorOnlyLowWindow.voltage) /
      std::max(1.0, collectorOnlyLowWindow.targetCaE / 0.25);
  const double collectorOnlyHighEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, collectorOnlyChargeWhip.voltage) /
      std::max(1.0, collectorOnlyChargeWhip.targetCaE / 0.25);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "collector_only_low_ca025",
                                        collectorOnlyLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "collector_only_high_ca042",
                                        collectorOnlyChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity, "collector_only",
                                     collectorOnlyLowWindow, collectorOnlyChargeWhip);
  writeCurrentVoltageSensitivityRow(collectorOnlyCurrentVoltageSensitivity,
                                    collectorOnlyLowWindow, collectorOnlyChargeWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "collector_only_low_ca025",
                           collectorOnlyLowEquivalentBase, setup,
                           collectorOnlyLowWindow);
  writeChargeScaleAuditRow(chargeScaleAudit, "collector_only_high_ca042",
                           collectorOnlyHighEquivalentBase, setup,
                           collectorOnlyChargeWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "collector_only_low_ca025",
                                setup, collectorOnlyLowWindow);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "collector_only_high_ca042",
                                setup, collectorOnlyChargeWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "collector_only_low_ca025",
                                 collectorOnlyLowEquivalentBase, setup,
                                 collectorOnlyLowWindow);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "collector_only_high_ca042",
                                 collectorOnlyHighEquivalentBase, setup,
                                 collectorOnlyChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "collector_only_low_ca025",
                                 collectorOnlyLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "collector_only_high_ca042",
                                 collectorOnlyChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D poissonFaceCurrentOpt = rayleighLimitOpt;
  poissonFaceCurrentOpt.usePoissonFaceConductiveCurrent = true;
  auto poissonFaceLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, poissonFaceCurrentOpt);
  auto poissonFaceChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, poissonFaceCurrentOpt);
  const double poissonFaceLowEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, poissonFaceLowWindow.voltage) /
      std::max(1.0, poissonFaceLowWindow.targetCaE / 0.25);
  const double poissonFaceHighEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, poissonFaceChargeWhip.voltage) /
      std::max(1.0, poissonFaceChargeWhip.targetCaE / 0.25);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "poisson_face_low_ca025",
                                        poissonFaceLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "poisson_face_high_ca042",
                                        poissonFaceChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "poisson_face_conductive_current",
                                     poissonFaceLowWindow, poissonFaceChargeWhip);
  writeCurrentVoltageSensitivityRow(poissonFaceCurrentVoltageSensitivity,
                                    poissonFaceLowWindow, poissonFaceChargeWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "poisson_face_low_ca025",
                           poissonFaceLowEquivalentBase, setup,
                           poissonFaceLowWindow);
  writeChargeScaleAuditRow(chargeScaleAudit, "poisson_face_high_ca042",
                           poissonFaceHighEquivalentBase, setup,
                           poissonFaceChargeWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "poisson_face_low_ca025",
                                setup, poissonFaceLowWindow);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "poisson_face_high_ca042",
                                setup, poissonFaceChargeWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "poisson_face_low_ca025",
                                 poissonFaceLowEquivalentBase, setup,
                                 poissonFaceLowWindow);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "poisson_face_high_ca042",
                                 poissonFaceHighEquivalentBase, setup,
                                 poissonFaceChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "poisson_face_low_ca025",
                                 poissonFaceLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "poisson_face_high_ca042",
                                 poissonFaceChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D implicitOhmicOpt = rayleighLimitOpt;
  implicitOhmicOpt.implicitOhmicChargeProjection = true;
  auto implicitOhmicLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, implicitOhmicOpt);
  auto implicitOhmicChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, implicitOhmicOpt);
  const double implicitOhmicLowEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, implicitOhmicLowWindow.voltage) /
      std::max(1.0, implicitOhmicLowWindow.targetCaE / 0.25);
  const double implicitOhmicHighEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, implicitOhmicChargeWhip.voltage) /
      std::max(1.0, implicitOhmicChargeWhip.targetCaE / 0.25);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "implicit_ohmic_low_ca025",
                                        implicitOhmicLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "implicit_ohmic_high_ca042",
                                        implicitOhmicChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity, "implicit_ohmic",
                                     implicitOhmicLowWindow, implicitOhmicChargeWhip);
  writeCurrentVoltageSensitivityRow(implicitOhmicCurrentVoltageSensitivity,
                                    implicitOhmicLowWindow,
                                    implicitOhmicChargeWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "implicit_ohmic_low_ca025",
                           implicitOhmicLowEquivalentBase, setup,
                           implicitOhmicLowWindow);
  writeChargeScaleAuditRow(chargeScaleAudit, "implicit_ohmic_high_ca042",
                           implicitOhmicHighEquivalentBase, setup,
                           implicitOhmicChargeWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "implicit_ohmic_low_ca025",
                                setup, implicitOhmicLowWindow);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "implicit_ohmic_high_ca042",
                                setup, implicitOhmicChargeWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "implicit_ohmic_low_ca025",
                                 implicitOhmicLowEquivalentBase, setup,
                                 implicitOhmicLowWindow);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "implicit_ohmic_high_ca042",
                                 implicitOhmicHighEquivalentBase, setup,
                                 implicitOhmicChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "implicit_ohmic_low_ca025",
                                 implicitOhmicLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "implicit_ohmic_high_ca042",
                                 implicitOhmicChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D faceConsistentElectricOpt =
      rayleighLimitOpt;
  faceConsistentElectricOpt.usePoissonFaceConductiveCurrent = true;
  faceConsistentElectricOpt.usePoissonFaceMaxwellForce = true;
  auto faceConsistentLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             faceConsistentElectricOpt);
  auto faceConsistentChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             faceConsistentElectricOpt);
  const double faceConsistentLowEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, faceConsistentLowWindow.voltage) /
      std::max(1.0, faceConsistentLowWindow.targetCaE / 0.25);
  const double faceConsistentHighEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, faceConsistentChargeWhip.voltage) /
      std::max(1.0, faceConsistentChargeWhip.targetCaE / 0.25);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent,
                                        "face_consistent_electric_low_ca025",
                                        faceConsistentLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent,
                                        "face_consistent_electric_high_ca042",
                                        faceConsistentChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "face_consistent_electric",
                                     faceConsistentLowWindow,
                                     faceConsistentChargeWhip);
  writeCurrentVoltageSensitivityRow(faceConsistentElectricCurrentVoltageSensitivity,
                                    faceConsistentLowWindow,
                                    faceConsistentChargeWhip);
  writeCurrentVoltageSensitivityRowForObservable(
      poissonFaceTotalCurrentVoltageSensitivity, faceConsistentLowWindow,
      faceConsistentChargeWhip, CandidoCurrentSensitivityObservable::PoissonFaceTotal);
  writeCurrentVoltageSensitivityRowForObservable(
      poissonFaceAlpha05TotalCurrentVoltageSensitivity, faceConsistentLowWindow,
      faceConsistentChargeWhip,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total);
  writeChargeScaleAuditRow(chargeScaleAudit,
                           "face_consistent_electric_low_ca025",
                           faceConsistentLowEquivalentBase, setup,
                           faceConsistentLowWindow);
  writeChargeScaleAuditRow(chargeScaleAudit,
                           "face_consistent_electric_high_ca042",
                           faceConsistentHighEquivalentBase, setup,
                           faceConsistentChargeWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency,
                                "face_consistent_electric_low_ca025",
                                setup, faceConsistentLowWindow);
  writeChargeUnitConsistencyRow(chargeUnitConsistency,
                                "face_consistent_electric_high_ca042",
                                setup, faceConsistentChargeWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale,
                                 "face_consistent_electric_low_ca025",
                                 faceConsistentLowEquivalentBase, setup,
                                 faceConsistentLowWindow);
  writeNondimChargeScaleAuditRow(nondimChargeScale,
                                 "face_consistent_electric_high_ca042",
                                 faceConsistentHighEquivalentBase, setup,
                                 faceConsistentChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "face_consistent_electric_low_ca025",
                                 faceConsistentLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "face_consistent_electric_high_ca042",
                                 faceConsistentChargeWhip);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "face_consistent_electric_convective",
      faceConsistentLowWindow, faceConsistentChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "face_consistent_electric_liquid_convective",
      faceConsistentLowWindow, faceConsistentChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "face_consistent_electric_alpha05_convective",
      faceConsistentLowWindow, faceConsistentChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "face_consistent_electric_convective",
      faceConsistentLowWindow, faceConsistentChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "face_consistent_electric_liquid_convective",
      faceConsistentLowWindow, faceConsistentChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "face_consistent_electric_alpha05_convective",
      faceConsistentLowWindow, faceConsistentChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "face_consistent_electric_poisson_face_alpha05_total",
      faceConsistentLowWindow, faceConsistentChargeWhip,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "face_consistent_electric_alpha05",
      faceConsistentLowWindow, faceConsistentChargeWhip, 1e-4);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "face_consistent_electric_alpha05",
      faceConsistentLowWindow, faceConsistentChargeWhip, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep, "face_consistent_electric_alpha05",
      faceConsistentLowWindow, faceConsistentChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D faceImplicitElectricOpt =
      rayleighLimitOpt;
  faceImplicitElectricOpt.usePoissonFaceConductiveCurrent = true;
  faceImplicitElectricOpt.usePoissonFaceMaxwellForce = true;
  faceImplicitElectricOpt.implicitOhmicChargeProjection = true;
  auto faceImplicitLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             faceImplicitElectricOpt);
  auto faceImplicitChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             faceImplicitElectricOpt);
  const double faceImplicitLowEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, faceImplicitLowWindow.voltage) /
      std::max(1.0, faceImplicitLowWindow.targetCaE / 0.25);
  const double faceImplicitHighEquivalentBase =
      electrospray::candidoDimensionlessRayleighChargeLimit(
          setup, faceImplicitChargeWhip.voltage) /
      std::max(1.0, faceImplicitChargeWhip.targetCaE / 0.25);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent,
                                        "face_implicit_electric_low_ca025",
                                        faceImplicitLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent,
                                        "face_implicit_electric_high_ca042",
                                        faceImplicitChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "face_implicit_electric",
                                     faceImplicitLowWindow, faceImplicitChargeWhip);
  writeCurrentVoltageSensitivityRow(faceImplicitElectricCurrentVoltageSensitivity,
                                    faceImplicitLowWindow,
                                    faceImplicitChargeWhip);
  writeChargeScaleAuditRow(chargeScaleAudit,
                           "face_implicit_electric_low_ca025",
                           faceImplicitLowEquivalentBase, setup,
                           faceImplicitLowWindow);
  writeChargeScaleAuditRow(chargeScaleAudit,
                           "face_implicit_electric_high_ca042",
                           faceImplicitHighEquivalentBase, setup,
                           faceImplicitChargeWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency,
                                "face_implicit_electric_low_ca025",
                                setup, faceImplicitLowWindow);
  writeChargeUnitConsistencyRow(chargeUnitConsistency,
                                "face_implicit_electric_high_ca042",
                                setup, faceImplicitChargeWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale,
                                 "face_implicit_electric_low_ca025",
                                 faceImplicitLowEquivalentBase, setup,
                                 faceImplicitLowWindow);
  writeNondimChargeScaleAuditRow(nondimChargeScale,
                                 "face_implicit_electric_high_ca042",
                                 faceImplicitHighEquivalentBase, setup,
                                 faceImplicitChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "face_implicit_electric_low_ca025",
                                 faceImplicitLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "face_implicit_electric_high_ca042",
                                 faceImplicitChargeWhip);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "face_implicit_electric_convective",
      faceImplicitLowWindow, faceImplicitChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "face_implicit_electric_liquid_convective",
      faceImplicitLowWindow, faceImplicitChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "face_implicit_electric_alpha05_convective",
      faceImplicitLowWindow, faceImplicitChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "face_implicit_electric_convective",
      faceImplicitLowWindow, faceImplicitChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "face_implicit_electric_liquid_convective",
      faceImplicitLowWindow, faceImplicitChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "face_implicit_electric_alpha05_convective",
      faceImplicitLowWindow, faceImplicitChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "face_implicit_electric_alpha05",
      faceImplicitLowWindow, faceImplicitChargeWhip, 1e-4);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "face_implicit_electric_alpha05",
      faceImplicitLowWindow, faceImplicitChargeWhip, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep, "face_implicit_electric_alpha05",
      faceImplicitLowWindow, faceImplicitChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D relaxationLimitedElectricOpt =
      faceImplicitElectricOpt;
  relaxationLimitedElectricOpt.useDimensionalElectricalScaling = true;
  relaxationLimitedElectricOpt.useElectricRelaxationTimeStepLimit = true;
  relaxationLimitedElectricOpt.electricRelaxationTimeStepSafety = 1.0;
  auto relaxationLimitedLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             relaxationLimitedElectricOpt);
  auto relaxationLimitedChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             relaxationLimitedElectricOpt);
  writeElectricRelaxationTimestepDiagnosticRow(
      electricRelaxationTimestep, faceImplicitLowWindow, faceImplicitChargeWhip,
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent,
                                        "relaxation_limited_electric_low_ca025",
                                        relaxationLimitedLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent,
                                        "relaxation_limited_electric_high_ca042",
                                        relaxationLimitedChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "relaxation_limited_electric",
                                     relaxationLimitedLowWindow,
                                     relaxationLimitedChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "relaxation_limited_electric_low_ca025",
                                 relaxationLimitedLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "relaxation_limited_electric_high_ca042",
                                 relaxationLimitedChargeWhip);
  writeLongWindowChargeBudgetRow(chargeBudget,
                                 "relaxation_limited_electric_low_ca025",
                                 relaxationLimitedLowWindow);
  writeLongWindowChargeBudgetRow(chargeBudget,
                                 "relaxation_limited_electric_high_ca042",
                                 relaxationLimitedChargeWhip);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling,
                                       "relaxation_limited_electric_low_ca025",
                                       setup, relaxationLimitedElectricOpt,
                                       relaxationLimitedLowWindow);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling,
                                       "relaxation_limited_electric_high_ca042",
                                       setup, relaxationLimitedElectricOpt,
                                       relaxationLimitedChargeWhip);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "relaxation_limited_electric_convective",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "relaxation_limited_electric_liquid_convective",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "relaxation_limited_electric_alpha05_convective",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "relaxation_limited_electric_convective",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "relaxation_limited_electric_liquid_convective",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "relaxation_limited_electric_alpha05_convective",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "relaxation_limited_electric_alpha05",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip, 1e-4);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "relaxation_limited_electric_alpha05",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep, "relaxation_limited_electric_alpha05",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D hybridMaxwellLongOpt =
      relaxationLimitedElectricOpt;
  hybridMaxwellLongOpt.usePoissonHybridMaxwellForce = true;
  auto hybridMaxwellLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             hybridMaxwellLongOpt);
  auto hybridMaxwellChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             hybridMaxwellLongOpt);
  writeHybridMaxwellLongWindowDiagnosticRow(
      hybridMaxwellLongWindow, relaxationLimitedLowWindow,
      relaxationLimitedChargeWhip, hybridMaxwellLowWindow,
      hybridMaxwellChargeWhip, 1e-4);
  writeHistoryRows(history, "hybrid_maxwell_ca025", hybridMaxwellLowWindow);
  writeHistoryRows(history, "hybrid_maxwell_ca042", hybridMaxwellChargeWhip);
  writeMorphologyObservableAuditRows(morphologyAudit, "hybrid_maxwell_ca025",
                                     hybridMaxwellLowWindow);
  writeMorphologyObservableAuditRows(morphologyAudit, "hybrid_maxwell_ca042",
                                     hybridMaxwellChargeWhip);
  writePhysicalTimeRows(physicalTime, "hybrid_maxwell_ca025",
                        hybridMaxwellLowWindow);
  writePhysicalTimeRows(physicalTime, "hybrid_maxwell_ca042",
                        hybridMaxwellChargeWhip);
  writeMorphologyReferenceGapRows(morphologyError, "hybrid_maxwell_ca025",
                                  hybridMaxwellLowWindow);
  writeMorphologyReferenceGapRows(morphologyError, "hybrid_maxwell_ca042",
                                  hybridMaxwellChargeWhip);
  writeMorphologyTimeAlignmentRows(morphologyAlignment, "hybrid_maxwell_ca025",
                                   hybridMaxwellLowWindow);
  writeMorphologyPhaseLagDiagnosticRows(morphologyPhaseLag,
                                        "hybrid_maxwell_ca025",
                                        hybridMaxwellLowWindow);
  writeMorphologyTipSyncDiagnosticRow(morphologyTipSync,
                                      "hybrid_maxwell_ca025",
                                      hybridMaxwellLowWindow);
  writeWhippingDiagnosticRow(whippingDiagnostic, "hybrid_maxwell_ca042", setup,
                             hybridMaxwellChargeWhip);
  writeLongWindowMassBudgetRow(massBudget, "hybrid_maxwell_ca025",
                               hybridMaxwellLowWindow);
  writeLongWindowMassBudgetRow(massBudget, "hybrid_maxwell_ca042",
                               hybridMaxwellChargeWhip);
  writeLongWindowChargeBudgetRow(chargeBudget, "hybrid_maxwell_ca025",
                                 hybridMaxwellLowWindow);
  writeLongWindowChargeBudgetRow(chargeBudget, "hybrid_maxwell_ca042",
                                 hybridMaxwellChargeWhip);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "hybrid_maxwell_convective",
      hybridMaxwellLowWindow, hybridMaxwellChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "hybrid_maxwell_liquid_convective",
      hybridMaxwellLowWindow, hybridMaxwellChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "hybrid_maxwell_alpha05_convective",
      hybridMaxwellLowWindow, hybridMaxwellChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "hybrid_maxwell_alpha05_convective",
      hybridMaxwellLowWindow, hybridMaxwellChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "hybrid_maxwell_alpha05",
      hybridMaxwellLowWindow, hybridMaxwellChargeWhip, 1e-4);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "hybrid_maxwell_alpha05",
      hybridMaxwellLowWindow, hybridMaxwellChargeWhip, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep, "hybrid_maxwell_alpha05",
      hybridMaxwellLowWindow, hybridMaxwellChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D boundedVectorMaxwellLongOpt =
      relaxationLimitedElectricOpt;
  boundedVectorMaxwellLongOpt.usePoissonBoundedVectorMaxwellForce = true;
  boundedVectorMaxwellLongOpt.poissonTangentialLimitFactor = 2.0;
  boundedVectorMaxwellLongOpt.poissonTangentialLimitFloorFraction = 0.05;
  auto boundedVectorMaxwellLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             boundedVectorMaxwellLongOpt);
  auto boundedVectorMaxwellChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             boundedVectorMaxwellLongOpt);
  writeBoundedVectorMaxwellLongWindowDiagnosticRow(
      boundedVectorMaxwellLongWindow, relaxationLimitedLowWindow,
      relaxationLimitedChargeWhip, boundedVectorMaxwellLowWindow,
      boundedVectorMaxwellChargeWhip, 1e-4);
  writeHistoryRows(history, "bounded_vector_maxwell_ca025",
                   boundedVectorMaxwellLowWindow);
  writeHistoryRows(history, "bounded_vector_maxwell_ca042",
                   boundedVectorMaxwellChargeWhip);
  writeMorphologyObservableAuditRows(morphologyAudit,
                                     "bounded_vector_maxwell_ca025",
                                     boundedVectorMaxwellLowWindow);
  writeMorphologyObservableAuditRows(morphologyAudit,
                                     "bounded_vector_maxwell_ca042",
                                     boundedVectorMaxwellChargeWhip);
  writePhysicalTimeRows(physicalTime, "bounded_vector_maxwell_ca025",
                        boundedVectorMaxwellLowWindow);
  writePhysicalTimeRows(physicalTime, "bounded_vector_maxwell_ca042",
                        boundedVectorMaxwellChargeWhip);
  writeMorphologyReferenceGapRows(morphologyError,
                                  "bounded_vector_maxwell_ca025",
                                  boundedVectorMaxwellLowWindow);
  writeMorphologyReferenceGapRows(morphologyError,
                                  "bounded_vector_maxwell_ca042",
                                  boundedVectorMaxwellChargeWhip);
  writeMorphologyTimeAlignmentRows(morphologyAlignment,
                                   "bounded_vector_maxwell_ca025",
                                   boundedVectorMaxwellLowWindow);
  writeMorphologyPhaseLagDiagnosticRows(morphologyPhaseLag,
                                        "bounded_vector_maxwell_ca025",
                                        boundedVectorMaxwellLowWindow);
  writeMorphologyTipSyncDiagnosticRow(morphologyTipSync,
                                      "bounded_vector_maxwell_ca025",
                                      boundedVectorMaxwellLowWindow);
  writeWhippingDiagnosticRow(whippingDiagnostic, "bounded_vector_maxwell_ca042",
                             setup, boundedVectorMaxwellChargeWhip);
  writeLongWindowMassBudgetRow(massBudget, "bounded_vector_maxwell_ca025",
                               boundedVectorMaxwellLowWindow);
  writeLongWindowMassBudgetRow(massBudget, "bounded_vector_maxwell_ca042",
                               boundedVectorMaxwellChargeWhip);
  writeLongWindowChargeBudgetRow(chargeBudget, "bounded_vector_maxwell_ca025",
                                 boundedVectorMaxwellLowWindow);
  writeLongWindowChargeBudgetRow(chargeBudget, "bounded_vector_maxwell_ca042",
                                 boundedVectorMaxwellChargeWhip);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "bounded_vector_maxwell_convective",
      boundedVectorMaxwellLowWindow, boundedVectorMaxwellChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "bounded_vector_maxwell_liquid_convective",
      boundedVectorMaxwellLowWindow, boundedVectorMaxwellChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "bounded_vector_maxwell_alpha05_convective",
      boundedVectorMaxwellLowWindow, boundedVectorMaxwellChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "bounded_vector_maxwell_alpha05_convective",
      boundedVectorMaxwellLowWindow, boundedVectorMaxwellChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "bounded_vector_maxwell_alpha05",
      boundedVectorMaxwellLowWindow, boundedVectorMaxwellChargeWhip, 1e-4);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "bounded_vector_maxwell_alpha05",
      boundedVectorMaxwellLowWindow, boundedVectorMaxwellChargeWhip, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep, "bounded_vector_maxwell_alpha05",
      boundedVectorMaxwellLowWindow, boundedVectorMaxwellChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D tomarConductingLongOpt =
      relaxationLimitedElectricOpt;
  tomarConductingLongOpt.useTomarConductingSurfaceForce = true;
  auto tomarConductingLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             tomarConductingLongOpt);
  auto tomarConductingChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             tomarConductingLongOpt);
  writeTomarConductingLongWindowDiagnosticRow(
      tomarConductingLongWindow, relaxationLimitedLowWindow,
      relaxationLimitedChargeWhip, tomarConductingLowWindow,
      tomarConductingChargeWhip, 1e-4);
  writeHistoryRows(history, "tomar_conducting_force_ca025",
                   tomarConductingLowWindow);
  writeHistoryRows(history, "tomar_conducting_force_ca042",
                   tomarConductingChargeWhip);
  writeMorphologyObservableAuditRows(morphologyAudit,
                                     "tomar_conducting_force_ca025",
                                     tomarConductingLowWindow);
  writeMorphologyObservableAuditRows(morphologyAudit,
                                     "tomar_conducting_force_ca042",
                                     tomarConductingChargeWhip);
  writeLongWindowMassBudgetRow(massBudget, "tomar_conducting_force_ca025",
                               tomarConductingLowWindow);
  writeLongWindowMassBudgetRow(massBudget, "tomar_conducting_force_ca042",
                               tomarConductingChargeWhip);
  writeLongWindowChargeBudgetRow(chargeBudget, "tomar_conducting_force_ca025",
                                 tomarConductingLowWindow);
  writeLongWindowChargeBudgetRow(chargeBudget, "tomar_conducting_force_ca042",
                                 tomarConductingChargeWhip);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "tomar_conducting_force_alpha05",
      tomarConductingLowWindow, tomarConductingChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "tomar_conducting_force_alpha05",
      tomarConductingLowWindow, tomarConductingChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "tomar_conducting_force_alpha05",
      tomarConductingLowWindow, tomarConductingChargeWhip, 1e-4);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "tomar_conducting_force_alpha05",
      tomarConductingLowWindow, tomarConductingChargeWhip, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep, "tomar_conducting_force_alpha05",
      tomarConductingLowWindow, tomarConductingChargeWhip);
  electrospray::CandidoConeJetSmokeOptions3D caIndependentDriveOpt =
      relaxationLimitedElectricOpt;
  caIndependentDriveOpt.electricDriveCaExponent = 0.0;
  auto caIndependentDriveLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             caIndependentDriveOpt);
  auto caIndependentDriveHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             caIndependentDriveOpt);
  writeElectricDriveScalingDiagnosticRow(
      electricDriveScaling, "ca_independent_drive",
      relaxationLimitedLowWindow,
      relaxationLimitedChargeWhip, caIndependentDriveLow, caIndependentDriveHigh,
      relaxationLimitedElectricOpt.electricDriveCaExponent,
      caIndependentDriveOpt.electricDriveCaExponent, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "ca_independent_drive_relaxation_limited_alpha05_convective",
      caIndependentDriveLow, caIndependentDriveHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization,
      "ca_independent_drive_relaxation_limited_alpha05",
      caIndependentDriveLow, caIndependentDriveHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization,
      "ca_independent_drive_relaxation_limited_alpha05",
      caIndependentDriveLow, caIndependentDriveHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "ca_independent_drive_relaxation_limited",
                                     caIndependentDriveLow,
                                     caIndependentDriveHigh);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure,
      "ca_independent_drive_relaxation_limited_alpha05",
      caIndependentDriveLow, caIndependentDriveHigh, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep,
      "ca_independent_drive_relaxation_limited_alpha05",
      caIndependentDriveLow, caIndependentDriveHigh);
  electrospray::CandidoConeJetSmokeOptions3D caIndependentBoundaryOpt =
      caIndependentDriveOpt;
  caIndependentBoundaryOpt.useBoundaryChargeAdvection = true;
  auto caIndependentBoundaryLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             caIndependentBoundaryOpt);
  auto caIndependentBoundaryHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             caIndependentBoundaryOpt);
  writeElectricDriveScalingDiagnosticRow(
      electricDriveScaling, "ca_independent_drive_boundary_advected",
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip,
      caIndependentBoundaryLow, caIndependentBoundaryHigh,
      relaxationLimitedElectricOpt.electricDriveCaExponent,
      caIndependentBoundaryOpt.electricDriveCaExponent, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "ca_independent_drive_boundary_advected_alpha05_convective",
      caIndependentBoundaryLow, caIndependentBoundaryHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization,
      "ca_independent_drive_boundary_advected_alpha05",
      caIndependentBoundaryLow, caIndependentBoundaryHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization,
      "ca_independent_drive_boundary_advected_alpha05",
      caIndependentBoundaryLow, caIndependentBoundaryHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "ca_independent_drive_boundary_advected",
                                     caIndependentBoundaryLow,
                                     caIndependentBoundaryHigh);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure,
      "ca_independent_drive_boundary_advected_alpha05",
      caIndependentBoundaryLow, caIndependentBoundaryHigh, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep,
      "ca_independent_drive_boundary_advected_alpha05",
      caIndependentBoundaryLow, caIndependentBoundaryHigh);
  writeCurrentVoltageSensitivityRow(caIndependentBoundaryCurrentVoltageSensitivity,
                                    caIndependentBoundaryLow,
                                    caIndependentBoundaryHigh);
  writeCurrentVoltageSensitivityRowForObservable(
      caIndependentBoundaryCurrentVoltageSensitivity, caIndependentBoundaryLow,
      caIndependentBoundaryHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective);
  electrospray::CandidoTaylorConeJetSetup reducedCollectorSetup = setup;
  reducedCollectorSetup.collectorDistance = 0.75e-3;
  auto reducedCollectorLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, reducedCollectorSetup,
                                             caIndependentBoundaryOpt);
  auto reducedCollectorHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, reducedCollectorSetup,
                                             caIndependentBoundaryOpt);
  writeReducedCollectorCurrentFixtureRow(
      reducedCollectorCurrentFixture,
      "ca_independent_boundary_reduced_collector_0_75mm",
      reducedCollectorSetup, reducedCollectorLow, reducedCollectorHigh, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D reducedCollectorInletAlphaOpt =
      caIndependentBoundaryOpt;
  reducedCollectorInletAlphaOpt.useVofInletBoundaryAlpha = true;
  auto reducedCollectorInletAlphaLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, reducedCollectorSetup,
                                             reducedCollectorInletAlphaOpt);
  auto reducedCollectorInletAlphaHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, reducedCollectorSetup,
                                             reducedCollectorInletAlphaOpt);
  writeReducedCollectorCurrentFixtureRow(
      reducedCollectorCurrentFixture,
      "ca_independent_boundary_inlet_alpha_reduced_collector_0_75mm",
      reducedCollectorSetup, reducedCollectorInletAlphaLow,
      reducedCollectorInletAlphaHigh, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D paperChargeBoundaryOpt =
      caIndependentBoundaryOpt;
  paperChargeBoundaryOpt.useVofInletBoundaryAlpha = true;
  paperChargeBoundaryOpt.suppressNozzleConductiveChargeFlux = true;
  auto paperChargeBoundaryLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperChargeBoundaryOpt);
  auto paperChargeBoundaryHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperChargeBoundaryOpt);
  writeHistoryRows(history, "paper_charge_boundary_ca025",
                   paperChargeBoundaryLow);
  writeHistoryRows(history, "paper_charge_boundary_ca042",
                   paperChargeBoundaryHigh);
  writeCurrentVoltageSensitivityRow(paperChargeBoundaryCurrentVoltageSensitivity,
                                    paperChargeBoundaryLow,
                                    paperChargeBoundaryHigh);
  writeCurrentVoltageSensitivityRowForObservable(
      paperChargeBoundaryCurrentVoltageSensitivity, paperChargeBoundaryLow,
      paperChargeBoundaryHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective);
  writeMidplaneCurrentReachDiagnosticRow(
      midplaneCurrentReach, "paper_charge_boundary_fixed_midplane",
      paperChargeBoundaryLow, paperChargeBoundaryHigh, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_charge_boundary_alpha05_convective",
      paperChargeBoundaryLow, paperChargeBoundaryHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_charge_boundary_poisson_face_alpha05_total",
      paperChargeBoundaryLow, paperChargeBoundaryHigh,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_charge_boundary_poisson_face_alpha05_convective",
      paperChargeBoundaryLow, paperChargeBoundaryHigh,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective, 1e-4);
  writePoissonFaceConvectiveFactorizationDiagnosticRow(
      poissonFaceConvectiveFactorization,
      "paper_charge_boundary_poisson_face_alpha05",
      paperChargeBoundaryLow, paperChargeBoundaryHigh, 1e-4);
  writePoissonFaceVelocityProjectionFactorizationDiagnosticRow(
      poissonFaceVelocityProjectionFactorization,
      "paper_charge_boundary_poisson_face_alpha05",
      paperChargeBoundaryLow, paperChargeBoundaryHigh, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "paper_charge_boundary_alpha05",
      paperChargeBoundaryLow, paperChargeBoundaryHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization, "paper_charge_boundary_alpha05",
      paperChargeBoundaryLow, paperChargeBoundaryHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "paper_charge_boundary",
                                     paperChargeBoundaryLow,
                                     paperChargeBoundaryHigh);
  electrospray::CandidoConeJetSmokeOptions3D postChargePotentialRefreshOpt =
      paperChargeBoundaryOpt;
  postChargePotentialRefreshOpt.refreshPotentialAfterChargeAdvance = true;
  auto postChargePotentialRefreshLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             postChargePotentialRefreshOpt);
  auto postChargePotentialRefreshHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             postChargePotentialRefreshOpt);
  writeHistoryRows(history, "post_charge_potential_refresh_ca025",
                   postChargePotentialRefreshLow);
  writeHistoryRows(history, "post_charge_potential_refresh_ca042",
                   postChargePotentialRefreshHigh);
  writePostChargePotentialRefreshDiagnosticRow(
      postChargePotentialRefresh, "post_charge_potential_refresh",
      paperChargeBoundaryLow, paperChargeBoundaryHigh,
      postChargePotentialRefreshLow, postChargePotentialRefreshHigh, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "post_charge_potential_refresh_alpha05_convective",
      postChargePotentialRefreshLow, postChargePotentialRefreshHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "post_charge_potential_refresh_alpha05",
      postChargePotentialRefreshLow, postChargePotentialRefreshHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization, "post_charge_potential_refresh_alpha05",
      postChargePotentialRefreshLow, postChargePotentialRefreshHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "post_charge_potential_refresh", setup,
      postChargePotentialRefreshLow, postChargePotentialRefreshHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(
      boundaryCurrentSensitivity, "post_charge_potential_refresh",
      postChargePotentialRefreshLow, postChargePotentialRefreshHigh);
  electrospray::CandidoConeJetSmokeOptions3D conductivityPotentialClosureOpt =
      paperChargeBoundaryOpt;
  conductivityPotentialClosureOpt.useConductivityPotentialChargeClosure = true;
  conductivityPotentialClosureOpt.usePoissonFaceConductiveCurrent = true;
  auto conductivityPotentialClosureLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             conductivityPotentialClosureOpt);
  auto conductivityPotentialClosureHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             conductivityPotentialClosureOpt);
  writeHistoryRows(history, "conductivity_potential_charge_closure_ca025",
                   conductivityPotentialClosureLow);
  writeHistoryRows(history, "conductivity_potential_charge_closure_ca042",
                   conductivityPotentialClosureHigh);
  writeConductivityPotentialChargeClosureDiagnosticRow(
      conductivityPotentialChargeClosure,
      "conductivity_potential_charge_closure",
      paperChargeBoundaryLow, paperChargeBoundaryHigh,
      conductivityPotentialClosureLow, conductivityPotentialClosureHigh, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "conductivity_potential_charge_closure_alpha05_convective",
      conductivityPotentialClosureLow, conductivityPotentialClosureHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization,
      "conductivity_potential_charge_closure_alpha05",
      conductivityPotentialClosureLow, conductivityPotentialClosureHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization,
      "conductivity_potential_charge_closure_alpha05",
      conductivityPotentialClosureLow, conductivityPotentialClosureHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "conductivity_potential_charge_closure", setup,
      conductivityPotentialClosureLow, conductivityPotentialClosureHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(
      boundaryCurrentSensitivity, "conductivity_potential_charge_closure",
      conductivityPotentialClosureLow, conductivityPotentialClosureHigh);
  electrospray::CandidoConeJetSmokeOptions3D conservativeSurfaceChargeOpt =
      paperChargeBoundaryOpt;
  conservativeSurfaceChargeOpt.implicitOhmicChargeProjection = true;
  conservativeSurfaceChargeOpt.conservativeChargeBounding = true;
  conservativeSurfaceChargeOpt.useInterfaceLocalizedChargeRedistribution = true;
  conservativeSurfaceChargeOpt.interfaceChargeRedistributionLiquidFloor = 0.02;
  conservativeSurfaceChargeOpt.usePoissonFaceConductiveCurrent = true;
  conservativeSurfaceChargeOpt.usePoissonFaceMaxwellForce = true;
  conservativeSurfaceChargeOpt.refreshPotentialAfterChargeAdvance = true;
  auto conservativeSurfaceChargeLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             conservativeSurfaceChargeOpt);
  auto conservativeSurfaceChargeHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             conservativeSurfaceChargeOpt);
  writeHistoryRows(history, "conservative_surface_charge_closure_ca025",
                   conservativeSurfaceChargeLow);
  writeHistoryRows(history, "conservative_surface_charge_closure_ca042",
                   conservativeSurfaceChargeHigh);
  writeConservativeSurfaceChargeClosureDiagnosticRow(
      conservativeSurfaceChargeClosure,
      "conservative_surface_charge_closure",
      paperChargeBoundaryLow, paperChargeBoundaryHigh,
      conservativeSurfaceChargeLow, conservativeSurfaceChargeHigh, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "conservative_surface_charge_closure_alpha05_convective",
      conservativeSurfaceChargeLow, conservativeSurfaceChargeHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization,
      "conservative_surface_charge_closure_alpha05",
      conservativeSurfaceChargeLow, conservativeSurfaceChargeHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization,
      "conservative_surface_charge_closure_alpha05",
      conservativeSurfaceChargeLow, conservativeSurfaceChargeHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "conservative_surface_charge_closure", setup,
      conservativeSurfaceChargeLow, conservativeSurfaceChargeHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(
      boundaryCurrentSensitivity, "conservative_surface_charge_closure",
      conservativeSurfaceChargeLow, conservativeSurfaceChargeHigh);
  electrospray::CandidoConeJetSmokeOptions3D interfaceChargeTransportOpt =
      paperChargeBoundaryOpt;
  interfaceChargeTransportOpt.useInterfaceLocalizedChargeRedistribution = true;
  interfaceChargeTransportOpt.interfaceChargeRedistributionLiquidFloor = 0.02;
  auto interfaceChargeTransportLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             interfaceChargeTransportOpt);
  auto interfaceChargeTransportHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             interfaceChargeTransportOpt);
  writeHistoryRows(history, "interface_localized_charge_transport_ca025",
                   interfaceChargeTransportLow);
  writeHistoryRows(history, "interface_localized_charge_transport_ca042",
                   interfaceChargeTransportHigh);
  writeInterfaceChargeTransportDiagnosticRow(
      interfaceChargeTransport, "interface_localized_charge_transport",
      paperChargeBoundaryLow, paperChargeBoundaryHigh,
      interfaceChargeTransportLow, interfaceChargeTransportHigh, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "interface_localized_charge_transport_alpha05_convective",
      interfaceChargeTransportLow, interfaceChargeTransportHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "interface_localized_charge_transport_alpha05",
      interfaceChargeTransportLow, interfaceChargeTransportHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization,
      "interface_localized_charge_transport_alpha05",
      interfaceChargeTransportLow, interfaceChargeTransportHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "interface_localized_charge_transport", setup,
      interfaceChargeTransportLow, interfaceChargeTransportHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(
      boundaryCurrentSensitivity, "interface_localized_charge_transport",
      interfaceChargeTransportLow, interfaceChargeTransportHigh);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "paper_charge_boundary_alpha05",
      paperChargeBoundaryLow, paperChargeBoundaryHigh, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D paperInletVelocityOpt =
      paperChargeBoundaryOpt;
  paperInletVelocityOpt.useFullyDevelopedInletVelocityBoundary = true;
  auto paperInletVelocityLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperInletVelocityOpt);
  auto paperInletVelocityHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperInletVelocityOpt);
  writeHistoryRows(history, "paper_inlet_velocity_ca025",
                   paperInletVelocityLow);
  writeHistoryRows(history, "paper_inlet_velocity_ca042",
                   paperInletVelocityHigh);
  writeCurrentVoltageSensitivityRow(
      paperInletVelocityCurrentVoltageSensitivity, paperInletVelocityLow,
      paperInletVelocityHigh);
  writeCurrentVoltageSensitivityRowForObservable(
      paperInletVelocityCurrentVoltageSensitivity, paperInletVelocityLow,
      paperInletVelocityHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective);
  writeMidplaneCurrentReachDiagnosticRow(
      midplaneCurrentReach, "paper_inlet_velocity_fixed_midplane",
      paperInletVelocityLow, paperInletVelocityHigh, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_inlet_velocity_alpha05_convective",
      paperInletVelocityLow, paperInletVelocityHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_inlet_velocity_poisson_face_alpha05_total",
      paperInletVelocityLow, paperInletVelocityHigh,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_inlet_velocity_poisson_face_alpha05_convective",
      paperInletVelocityLow, paperInletVelocityHigh,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective, 1e-4);
  writePoissonFaceConvectiveFactorizationDiagnosticRow(
      poissonFaceConvectiveFactorization,
      "paper_inlet_velocity_poisson_face_alpha05",
      paperInletVelocityLow, paperInletVelocityHigh, 1e-4);
  writePoissonFaceVelocityProjectionFactorizationDiagnosticRow(
      poissonFaceVelocityProjectionFactorization,
      "paper_inlet_velocity_poisson_face_alpha05",
      paperInletVelocityLow, paperInletVelocityHigh, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "paper_inlet_velocity_alpha05",
      paperInletVelocityLow, paperInletVelocityHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization, "paper_inlet_velocity_alpha05",
      paperInletVelocityLow, paperInletVelocityHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "paper_inlet_velocity",
                                     paperInletVelocityLow,
                                     paperInletVelocityHigh);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "paper_inlet_velocity_alpha05",
      paperInletVelocityLow, paperInletVelocityHigh, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D paperOpenBoundaryOpt =
      paperInletVelocityOpt;
  paperOpenBoundaryOpt.useOpenAtmosphericBoundaryFlux = true;
  auto paperOpenBoundaryLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperOpenBoundaryOpt);
  auto paperOpenBoundaryHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperOpenBoundaryOpt);
  writeHistoryRows(history, "paper_inlet_velocity_open_atmosphere_ca025",
                   paperOpenBoundaryLow);
  writeHistoryRows(history, "paper_inlet_velocity_open_atmosphere_ca042",
                   paperOpenBoundaryHigh);
  writeOpenBoundaryCurrentDiagnosticRow(
      openBoundaryCurrent, "paper_inlet_velocity_open_atmosphere", setup,
      paperOpenBoundaryLow, paperOpenBoundaryHigh, 1e-4);
  writeMidplaneCurrentReachDiagnosticRow(
      midplaneCurrentReach,
      "paper_inlet_velocity_open_atmosphere_fixed_midplane",
      paperOpenBoundaryLow, paperOpenBoundaryHigh, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_inlet_velocity_open_atmosphere_alpha05_convective",
      paperOpenBoundaryLow, paperOpenBoundaryHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05_total",
      paperOpenBoundaryLow, paperOpenBoundaryHigh,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05_convective",
      paperOpenBoundaryLow, paperOpenBoundaryHigh,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective, 1e-4);
  writePoissonFaceConvectiveFactorizationDiagnosticRow(
      poissonFaceConvectiveFactorization,
      "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05",
      paperOpenBoundaryLow, paperOpenBoundaryHigh, 1e-4);
  writePoissonFaceVelocityProjectionFactorizationDiagnosticRow(
      poissonFaceVelocityProjectionFactorization,
      "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05",
      paperOpenBoundaryLow, paperOpenBoundaryHigh, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "paper_inlet_velocity_open_atmosphere_alpha05",
      paperOpenBoundaryLow, paperOpenBoundaryHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization, "paper_inlet_velocity_open_atmosphere_alpha05",
      paperOpenBoundaryLow, paperOpenBoundaryHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "paper_inlet_velocity_open_atmosphere",
                                     paperOpenBoundaryLow,
                                     paperOpenBoundaryHigh);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "paper_inlet_velocity_open_atmosphere_alpha05",
      paperOpenBoundaryLow, paperOpenBoundaryHigh, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D movingCollectorOpt =
      paperOpenBoundaryOpt;
  movingCollectorOpt.useMovingCollectorWall = true;
  auto movingCollectorLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             movingCollectorOpt);
  auto movingCollectorHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             movingCollectorOpt);
  writeHistoryRows(history,
                   "paper_inlet_velocity_open_atmosphere_moving_collector_ca025",
                   movingCollectorLow);
  writeHistoryRows(history,
                   "paper_inlet_velocity_open_atmosphere_moving_collector_ca042",
                   movingCollectorHigh);
  writeMovingCollectorBoundaryDiagnosticRow(
      movingCollectorBoundary,
      "paper_inlet_velocity_open_atmosphere_moving_collector", setup,
      movingCollectorLow, movingCollectorHigh, 1e-4);
  writeMidplaneCurrentReachDiagnosticRow(
      midplaneCurrentReach,
      "paper_inlet_velocity_open_atmosphere_moving_collector_fixed_midplane",
      movingCollectorLow, movingCollectorHigh, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05_total",
      movingCollectorLow, movingCollectorHigh,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Total, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow,
      "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05_convective",
      movingCollectorLow, movingCollectorHigh,
      CandidoCurrentSensitivityObservable::PoissonFaceAlpha05Convective, 1e-4);
  writePoissonFaceConvectiveFactorizationDiagnosticRow(
      poissonFaceConvectiveFactorization,
      "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05",
      movingCollectorLow, movingCollectorHigh, 1e-4);
  writePoissonFaceVelocityProjectionFactorizationDiagnosticRow(
      poissonFaceVelocityProjectionFactorization,
      "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05",
      movingCollectorLow, movingCollectorHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto,
      "paper_inlet_velocity_open_atmosphere_moving_collector", setup,
      movingCollectorLow, movingCollectorHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization,
      "paper_inlet_velocity_open_atmosphere_moving_collector_alpha05",
      movingCollectorLow, movingCollectorHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(
      boundaryCurrentSensitivity,
      "paper_inlet_velocity_open_atmosphere_moving_collector",
      movingCollectorLow, movingCollectorHigh);
  electrospray::CandidoConeJetSmokeOptions3D paperOpenExtendedOpt =
      paperOpenBoundaryOpt;
  paperOpenExtendedOpt.steps = 90;
  auto paperOpenExtendedLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperOpenExtendedOpt);
  auto paperOpenExtendedHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperOpenExtendedOpt);
  writeHistoryRows(history,
                   "paper_inlet_velocity_open_atmosphere_extended90_ca025",
                   paperOpenExtendedLow);
  writeHistoryRows(history,
                   "paper_inlet_velocity_open_atmosphere_extended90_ca042",
                   paperOpenExtendedHigh);
  writeMidplaneCurrentReachDiagnosticRow(
      midplaneCurrentReach,
      "paper_inlet_velocity_open_atmosphere_extended90_fixed_midplane",
      paperOpenExtendedLow, paperOpenExtendedHigh, 1e-4);
  writePaperCurrentDevelopmentTradeoffRow(
      paperCurrentDevelopment,
      "paper_inlet_velocity_open_atmosphere_extended90", paperOpenBoundaryLow,
      paperOpenBoundaryHigh, paperOpenExtendedLow, paperOpenExtendedHigh, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D preconditionedCurrentOpt =
      paperOpenBoundaryOpt;
  preconditionedCurrentOpt.usePreconditionedPaperCurrentJet = true;
  preconditionedCurrentOpt.preconditionedJetRadiusInnerDiameters = 0.65;
  preconditionedCurrentOpt.preconditionedJetInterfaceWidthInnerDiameters = 0.20;
  preconditionedCurrentOpt.preconditionedJetVelocityScale = 1.0;
  auto preconditionedCurrentLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             preconditionedCurrentOpt);
  auto preconditionedCurrentHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             preconditionedCurrentOpt);
  writeHistoryRows(history,
                   "paper_preconditioned_current_plane_ca025",
                   preconditionedCurrentLow);
  writeHistoryRows(history,
                   "paper_preconditioned_current_plane_ca042",
                   preconditionedCurrentHigh);
  writePreconditionedCurrentPlaneDiagnosticRow(
      preconditionedCurrentPlane, "paper_preconditioned_current_plane",
      setup, preconditionedCurrentOpt, preconditionedCurrentLow,
      preconditionedCurrentHigh, 1e-4);
  writeMidplaneCurrentReachDiagnosticRow(
      midplaneCurrentReach,
      "paper_preconditioned_current_plane_fixed_midplane",
      preconditionedCurrentLow, preconditionedCurrentHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "paper_preconditioned_current_plane", setup,
      preconditionedCurrentLow, preconditionedCurrentHigh, 1e-4);
  writeFig8bCurrentBlockerRow(
      fig8bCurrentBlocker, setup, reducedCollectorSetup, paperInletVelocityOpt,
      paperChargeBoundaryLow, paperChargeBoundaryHigh, paperInletVelocityLow,
      paperInletVelocityHigh, reducedCollectorInletAlphaLow,
      reducedCollectorInletAlphaHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "baseline_long_window", setup, longWindow, longWhip, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "relaxation_limited_electric", setup,
      relaxationLimitedLowWindow, relaxationLimitedChargeWhip, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "hybrid_maxwell", setup, hybridMaxwellLowWindow,
      hybridMaxwellChargeWhip, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "bounded_vector_maxwell", setup,
      boundedVectorMaxwellLowWindow, boundedVectorMaxwellChargeWhip, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "ca_independent_drive", setup, caIndependentDriveLow,
      caIndependentDriveHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "ca_independent_drive_boundary_advected", setup,
      caIndependentBoundaryLow, caIndependentBoundaryHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "paper_charge_boundary", setup, paperChargeBoundaryLow,
      paperChargeBoundaryHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "paper_inlet_velocity", setup, paperInletVelocityLow,
      paperInletVelocityHigh, 1e-4);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "paper_inlet_velocity_open_atmosphere", setup,
      paperOpenBoundaryLow, paperOpenBoundaryHigh, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D unitMaxwellBoundaryOpt =
      caIndependentBoundaryOpt;
  unitMaxwellBoundaryOpt.electricDriveReferenceScale = 1.0;
  auto unitMaxwellBoundaryLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             unitMaxwellBoundaryOpt);
  auto unitMaxwellBoundaryHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             unitMaxwellBoundaryOpt);
  writeElectricDriveScalingDiagnosticRow(
      electricDriveScaling, "unit_maxwell_drive_boundary_advected",
      caIndependentBoundaryLow, caIndependentBoundaryHigh,
      unitMaxwellBoundaryLow, unitMaxwellBoundaryHigh,
      caIndependentBoundaryOpt.electricDriveReferenceScale,
      unitMaxwellBoundaryOpt.electricDriveReferenceScale, 1e-4);
  writeCurrentVoltageSensitivityRow(unitMaxwellBoundaryCurrentVoltageSensitivity,
                                    unitMaxwellBoundaryLow,
                                    unitMaxwellBoundaryHigh);
  writeCurrentVoltageSensitivityRowForObservable(
      unitMaxwellBoundaryCurrentVoltageSensitivity, unitMaxwellBoundaryLow,
      unitMaxwellBoundaryHigh,
      CandidoCurrentSensitivityObservable::Alpha05Convective);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "unit_maxwell_drive_boundary_advected", setup,
      unitMaxwellBoundaryLow, unitMaxwellBoundaryHigh, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization,
      "unit_maxwell_drive_boundary_advected_alpha05",
      unitMaxwellBoundaryLow, unitMaxwellBoundaryHigh, 1e-4);
  writeMomentumSourceFactorizationDiagnosticRow(
      momentumSourceFactorization,
      "unit_maxwell_drive_boundary_advected_alpha05",
      unitMaxwellBoundaryLow, unitMaxwellBoundaryHigh, 1e-4);
  writePoissonFaceConvectiveFactorizationDiagnosticRow(
      poissonFaceConvectiveFactorization,
      "unit_maxwell_drive_boundary_advected_poisson_face_alpha05",
      unitMaxwellBoundaryLow, unitMaxwellBoundaryHigh, 1e-4);
  writePoissonFaceVelocityProjectionFactorizationDiagnosticRow(
      poissonFaceVelocityProjectionFactorization,
      "unit_maxwell_drive_boundary_advected_poisson_face_alpha05",
      unitMaxwellBoundaryLow, unitMaxwellBoundaryHigh, 1e-4);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "unit_maxwell_drive_boundary_advected",
                                     unitMaxwellBoundaryLow,
                                     unitMaxwellBoundaryHigh);
  writeHistoryRows(history, "ca_independent_drive_boundary_ca025",
                   caIndependentBoundaryLow);
  writeHistoryRows(history, "ca_independent_drive_boundary_ca042",
                   caIndependentBoundaryHigh);
  writeMorphologyObservableAuditRows(morphologyAudit,
                                     "ca_independent_drive_boundary_ca025",
                                     caIndependentBoundaryLow);
  writeMorphologyObservableAuditRows(morphologyAudit,
                                     "ca_independent_drive_boundary_ca042",
                                     caIndependentBoundaryHigh);
  writeMorphologySilhouetteBracketRows(morphologyBracket,
                                       "ca_independent_drive_boundary_ca025",
                                       caIndependentBoundaryLow);
  writePhysicalTimeRows(physicalTime, "ca_independent_drive_boundary_ca025",
                        caIndependentBoundaryLow);
  writePhysicalTimeRows(physicalTime, "ca_independent_drive_boundary_ca042",
                        caIndependentBoundaryHigh);
  writeMorphologyReferenceGapRows(morphologyError,
                                  "ca_independent_drive_boundary_ca025",
                                  caIndependentBoundaryLow);
  writeMorphologyReferenceGapRows(morphologyError,
                                  "ca_independent_drive_boundary_ca042",
                                  caIndependentBoundaryHigh);
  writeMorphologyTimeAlignmentRows(morphologyAlignment,
                                   "ca_independent_drive_boundary_ca025",
                                   caIndependentBoundaryLow);
  writeMorphologyPhaseLagDiagnosticRows(morphologyPhaseLag,
                                        "ca_independent_drive_boundary_ca025",
                                        caIndependentBoundaryLow);
  writeMorphologyTipSyncDiagnosticRow(morphologyTipSync,
                                      "ca_independent_drive_boundary_ca025",
                                      caIndependentBoundaryLow);
  writeWhippingDiagnosticRow(whippingDiagnostic,
                             "ca_independent_drive_boundary_ca042", setup,
                             caIndependentBoundaryHigh);
  writeLongWindowMassBudgetRow(massBudget,
                               "ca_independent_drive_boundary_ca025",
                               caIndependentBoundaryLow);
  writeLongWindowMassBudgetRow(massBudget,
                               "ca_independent_drive_boundary_ca042",
                               caIndependentBoundaryHigh);
  writeLongWindowChargeBudgetRow(chargeBudget,
                                 "ca_independent_drive_boundary_ca025",
                                 caIndependentBoundaryLow);
  writeLongWindowChargeBudgetRow(chargeBudget,
                                 "ca_independent_drive_boundary_ca042",
                                 caIndependentBoundaryHigh);
  writeCaIndependentCurrentResolutionRow(
      caIndependentCurrentResolution, "n12_ca_independent_boundary_advected",
      caIndependentBoundaryOpt, caIndependentBoundaryLow, caIndependentBoundaryHigh,
      1e-4);
  for (int n : {10, 14}) {
    electrospray::CandidoConeJetSmokeOptions3D sweepOpt =
        caIndependentBoundaryOpt;
    sweepOpt.nx = n;
    sweepOpt.nz = n;
    sweepOpt.ny = std::max(8, static_cast<int>(std::round(1.4 * n)));
    auto sweepLow = electrospray::runCandidoConeJetSmoke3D(0.25, setup, sweepOpt);
    auto sweepHigh = electrospray::runCandidoConeJetSmoke3D(0.42, setup, sweepOpt);
    writeCaIndependentCurrentResolutionRow(
        caIndependentCurrentResolution,
        "n" + std::to_string(n) + "_ca_independent_boundary_advected",
        sweepOpt, sweepLow, sweepHigh, 1e-4);
  }
  electrospray::CandidoConeJetSmokeOptions3D boundaryChargeAdvectionOpt =
      relaxationLimitedElectricOpt;
  boundaryChargeAdvectionOpt.useBoundaryChargeAdvection = true;
  auto boundaryAdvectedLowWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             boundaryChargeAdvectionOpt);
  auto boundaryAdvectedChargeWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             boundaryChargeAdvectionOpt);
  writeBoundaryChargeAdvectionDiagnosticRow(
      boundaryChargeAdvection, relaxationLimitedLowWindow,
      relaxationLimitedChargeWhip, boundaryAdvectedLowWindow,
      boundaryAdvectedChargeWhip);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent,
                                        "boundary_advected_electric_low_ca025",
                                        boundaryAdvectedLowWindow);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent,
                                        "boundary_advected_electric_high_ca042",
                                        boundaryAdvectedChargeWhip);
  writeBoundaryCurrentSensitivityRow(boundaryCurrentSensitivity,
                                     "boundary_advected_electric",
                                     boundaryAdvectedLowWindow,
                                     boundaryAdvectedChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "boundary_advected_electric_low_ca025",
                                 boundaryAdvectedLowWindow);
  writeChargeFieldConsistencyRow(chargeFieldConsistency,
                                 "boundary_advected_electric_high_ca042",
                                 boundaryAdvectedChargeWhip);
  writeLongWindowChargeBudgetRow(chargeBudget,
                                 "boundary_advected_electric_low_ca025",
                                 boundaryAdvectedLowWindow);
  writeLongWindowChargeBudgetRow(chargeBudget,
                                 "boundary_advected_electric_high_ca042",
                                 boundaryAdvectedChargeWhip);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling,
                                       "boundary_advected_electric_low_ca025",
                                       setup, boundaryChargeAdvectionOpt,
                                       boundaryAdvectedLowWindow);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling,
                                       "boundary_advected_electric_high_ca042",
                                       setup, boundaryChargeAdvectionOpt,
                                       boundaryAdvectedChargeWhip);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "boundary_advected_electric_convective",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "boundary_advected_electric_liquid_convective",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeDevelopedJetCurrentWindowDiagnosticRow(
      developedJetCurrentWindow, "boundary_advected_electric_alpha05_convective",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "boundary_advected_electric_convective",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip,
      CandidoCurrentSensitivityObservable::Convective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "boundary_advected_electric_liquid_convective",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip,
      CandidoCurrentSensitivityObservable::LiquidConvective, 1e-4);
  writeAxialDevelopedJetCurrentWindowDiagnosticRow(
      axialDevelopedJetCurrentWindow, "boundary_advected_electric_alpha05_convective",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip,
      CandidoCurrentSensitivityObservable::Alpha05Convective, 1e-4);
  writeAxialCurrentFactorizationDiagnosticRow(
      axialCurrentFactorization, "boundary_advected_electric_alpha05",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip, 1e-4);
  writeAxialTotalCurrentClosureDiagnosticRow(
      axialTotalCurrentClosure, "boundary_advected_electric_alpha05",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip, 1e-4);
  writeAxialCurrentThresholdSweepRows(
      axialCurrentThresholdSweep, "boundary_advected_electric_alpha05",
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip);
  writePaperCurrentParetoTradeoffRow(
      currentPareto, "boundary_advected_electric", setup,
      boundaryAdvectedLowWindow, boundaryAdvectedChargeWhip, 1e-4);
  electrospray::CandidoConeJetSmokeOptions3D relaxedChargeOpt = combinedChargeOpt;
  relaxedChargeOpt.quasiImplicitChargeRelaxation = true;
  auto relaxedChargeWhip = electrospray::runCandidoConeJetSmoke3D(0.42, setup, relaxedChargeOpt);
  writeBoundaryCurrentDecompositionRows(boundaryCurrent, "relaxed_high_ca042",
                                        relaxedChargeWhip);
  writeChargeRelaxationDiagnosticRow(chargeRelaxation, combinedChargeWhip, relaxedChargeWhip);
  writeChargeScaleAuditRow(chargeScaleAudit, "relaxed_high_ca042",
                           relaxedChargeOpt.chargeLimitBase, setup, relaxedChargeWhip);
  writeChargeUnitConsistencyRow(chargeUnitConsistency, "relaxed_high_ca042", setup,
                                relaxedChargeWhip);
  writeNondimChargeScaleAuditRow(nondimChargeScale, "relaxed_high_ca042",
                                 relaxedChargeOpt.chargeLimitBase, setup,
                                 relaxedChargeWhip);
  writeChargeFieldConsistencyRow(chargeFieldConsistency, "relaxed_high_ca042",
                                 relaxedChargeWhip);
  writeElectricPropertyScalingAuditRow(electricPropertyScaling, "relaxed_high_ca042",
                                       setup, relaxedChargeOpt, relaxedChargeWhip);
  for (double chargeLimitBase : {5.0, 50.0, 500.0}) {
    electrospray::CandidoConeJetSmokeOptions3D limitOpt = combinedChargeOpt;
    limitOpt.chargeLimitBase = chargeLimitBase;
    auto limitRun = electrospray::runCandidoConeJetSmoke3D(0.42, setup, limitOpt);
    writeChargeLimitSensitivityRow(chargeLimitSensitivity, chargeLimitBase, limitRun);
    writeChargeScaleAuditRow(chargeScaleAudit, "limit_high_ca042", chargeLimitBase,
                             setup, limitRun);
    writeChargeUnitConsistencyRow(chargeUnitConsistency, "limit_high_ca042", setup,
                                  limitRun);
    writeNondimChargeScaleAuditRow(nondimChargeScale, "limit_high_ca042",
                                   chargeLimitBase, setup, limitRun);
    writeChargeFieldConsistencyRow(chargeFieldConsistency, "limit_high_ca042",
                                   limitRun);
  }
  check(longWhip.history.back().time * longWhip.hydrodynamicTimeScale * 1.0e3 >= 0.9,
        "Candido long-window whipping run reaches the paper reference time window");
  check(longWhip.alphaMassDrift <= 1e-3, "Candido long-window whipping mass drift bounded");
  check(longWhip.maxDiv <= 1e-7, "Candido long-window whipping continuity bounded");

  std::cout << "candido_cone_jet_smoke3d_rows=2"
            << " ca025_voltage=" << low.voltage
            << " ca042_voltage=" << high.voltage
            << " ca025_mass_drift=" << low.alphaMassDrift
            << " ca042_mass_drift=" << high.alphaMassDrift
            << " ca025_max_div=" << low.maxDiv
            << " ca042_max_div=" << high.maxDiv
            << " ca025_force=" << low.maxElectricForce
            << " ca042_force=" << high.maxElectricForce << "\n";
  return 0;
}
