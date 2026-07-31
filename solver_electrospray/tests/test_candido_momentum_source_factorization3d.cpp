#include "TestUtil.hpp"
#include "electrospray/CandidoTaylorConeJet3D.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

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

struct PoissonFaceDevelopedStats {
  int tailSamples = 0;
  int developedSamples = 0;
  double maxArea = 0.0;
  double meanArea = 0.0;
  double meanYOverDi = 0.0;
  double meanCurrent = 0.0;
};

struct PoissonFaceFactorStats {
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

struct PoissonFaceProjectionStats {
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

struct AxialCurrentStats {
  int developedSamples = 0;
  double meanArea = 0.0;
  double meanAlpha05Convective = 0.0;
  double meanAlpha05Total = 0.0;
};

struct CurrentSensitivityStats {
  double peak = 0.0;
  double meanAll = 0.0;
  double meanTail = 0.0;
};

struct MidplaneDevelopmentStats {
  int developedSamples = 0;
  double maxAlpha05AreaDi2 = 0.0;
  double meanDevelopedAlpha05Current = 0.0;
  double peakDevelopedAlpha05Current = 0.0;
};

double ratio(double highValue, double lowValue) {
  return highValue / std::max(lowValue, 1e-30);
}

SourceStats sourceStats(const electrospray::CandidoConeJetSmokeReport3D& r,
                        double minAreaDi2) {
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
}

AxialCurrentStats axialCurrentStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  AxialCurrentStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    if (h.developedJetAlpha05AreaDi2 < minAreaDi2) continue;
    ++s.developedSamples;
    s.meanArea += h.developedJetAlpha05AreaDi2;
    s.meanAlpha05Convective += std::abs(h.developedJetAlpha05ConvectiveCurrent);
    s.meanAlpha05Total += std::abs(h.developedJetAlpha05TotalCurrent);
  }
  if (s.developedSamples > 0) {
    const double inv = 1.0 / static_cast<double>(s.developedSamples);
    s.meanArea *= inv;
    s.meanAlpha05Convective *= inv;
    s.meanAlpha05Total *= inv;
  }
  return s;
}

int tailDevelopedSamplesAtMidplane(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  int samples = 0;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    if (r.history[i].midplaneAlpha05AreaDi2 >= minAreaDi2) ++samples;
  }
  return samples;
}

double meanTailConvectiveCurrent(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  double sum = 0.0;
  int samples = 0;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    sum += std::abs(r.history[i].convectiveCurrent);
    ++samples;
  }
  return samples > 0 ? sum / static_cast<double>(samples) : 0.0;
}

double peakConvectiveCurrent(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  double value = 0.0;
  for (const auto& h : r.history) value = std::max(value, std::abs(h.convectiveCurrent));
  return value;
}

double maxTailTipY(const electrospray::CandidoConeJetSmokeReport3D& r) {
  double value = 0.0;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    value = std::max(value, r.history[i].tipY);
  }
  return value;
}

CurrentSensitivityStats currentSensitivityStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    bool alpha05JetCurrent) {
  CurrentSensitivityStats s;
  int allCount = 0;
  int tailCount = 0;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = 0; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    const double value =
        std::abs(alpha05JetCurrent ? h.alpha05ConvectiveCurrent
                                   : h.convectiveCurrent);
    s.peak = std::max(s.peak, value);
    s.meanAll += value;
    ++allCount;
    if (i >= tailStart) {
      s.meanTail += value;
      ++tailCount;
    }
  }
  s.meanAll /= std::max(allCount, 1);
  s.meanTail /= std::max(tailCount, 1);
  return s;
}

MidplaneDevelopmentStats midplaneDevelopmentStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  MidplaneDevelopmentStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = 0; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    s.maxAlpha05AreaDi2 = std::max(s.maxAlpha05AreaDi2, h.midplaneAlpha05AreaDi2);
    if (i < tailStart || h.midplaneAlpha05AreaDi2 < minAreaDi2) continue;
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

PoissonFaceDevelopedStats poissonFaceDevelopedStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2,
    bool totalCurrent) {
  PoissonFaceDevelopedStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    ++s.tailSamples;
    s.maxArea = std::max(s.maxArea, h.poissonFaceDevelopedAlpha05AreaDi2);
    if (h.poissonFaceDevelopedAlpha05AreaDi2 < minAreaDi2) continue;
    ++s.developedSamples;
    s.meanArea += h.poissonFaceDevelopedAlpha05AreaDi2;
    s.meanYOverDi += h.poissonFaceDevelopedYOverDi;
    s.meanCurrent += std::abs(totalCurrent
                                  ? h.poissonFaceDevelopedAlpha05TotalCurrent
                                  : h.poissonFaceDevelopedAlpha05ConvectiveCurrent);
  }
  if (s.developedSamples > 0) {
    const double inv = 1.0 / static_cast<double>(s.developedSamples);
    s.meanArea *= inv;
    s.meanYOverDi *= inv;
    s.meanCurrent *= inv;
  }
  return s;
}

PoissonFaceFactorStats poissonFaceFactorStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  PoissonFaceFactorStats s;
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
}

PoissonFaceProjectionStats poissonFaceProjectionStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  PoissonFaceProjectionStats s;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    ++s.tailSamples;
    if (h.poissonFaceDevelopedAlpha05AreaDi2 < minAreaDi2) continue;
    ++s.developedSamples;
    s.meanArea += h.poissonFaceDevelopedAlpha05AreaDi2;
    s.projectedCurrent +=
        std::abs(h.poissonFaceDevelopedAlpha05ConvectiveCurrent);
    s.projectedAbsUpwindCharge +=
        h.poissonFaceDevelopedAlpha05MeanAbsUpwindCharge;
    s.projectedAbsFaceFlux += h.poissonFaceDevelopedAlpha05MeanAbsFaceFlux;
    s.projectedAbsConvectiveFlux +=
        h.poissonFaceDevelopedAlpha05MeanAbsConvectiveFlux;
    s.rawCurrent += std::abs(h.rawVelocityFaceDevelopedAlpha05ConvectiveCurrent);
    s.rawAbsUpwindCharge += h.rawVelocityFaceDevelopedAlpha05MeanAbsUpwindCharge;
    s.rawAbsFaceFlux += h.rawVelocityFaceDevelopedAlpha05MeanAbsFaceFlux;
    s.rawAbsConvectiveFlux +=
        h.rawVelocityFaceDevelopedAlpha05MeanAbsConvectiveFlux;
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

void writePoissonFaceCurrentVoltageRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    bool alpha05Total) {
  auto stats = [alpha05Total](const electrospray::CandidoConeJetSmokeReport3D& r) {
    double maxCurrent = 0.0;
    double meanAll = 0.0;
    double meanTail = 0.0;
    int allCount = 0;
    int tailCount = 0;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = 0; i < r.history.size(); ++i) {
      const auto& h = r.history[i];
      const double value = std::abs(alpha05Total ? h.poissonFaceAlpha05TotalCurrent
                                                 : h.poissonFaceTotalCurrent);
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
    return std::array<double, 3>{maxCurrent, meanAll, meanTail};
  };
  const auto lowStats = stats(low);
  const auto highStats = stats(high);
  const double peakRatio = ratio(highStats[0], lowStats[0]);
  const double tailRatio = ratio(highStats[2], lowStats[2]);
  const std::string source =
      alpha05Total
          ? "face_consistent_alpha05_total_current=rho_e_phiFlux_plus_sigma_gradphi_flux;"
            "uses_Poisson_snGrad_flux"
          : "face_consistent_total_current=rho_e_phiFlux_plus_sigma_gradphi_flux;"
            "uses_Poisson_snGrad_flux";
  const double currentScale = std::max(lowStats[2], highStats[2]);
  const std::string status =
      currentScale <= 1e-30
          ? "BLOCKED_ZERO_CURRENT_OBSERVABLE"
          : (tailRatio <= 2.0
                 ? "APPROXIMATE_WEAK_AVERAGE_VOLTAGE_SENSITIVITY"
                 : "DOWNGRADED_AVERAGE_CURRENT_TOO_VOLTAGE_SENSITIVE");
  csv << low.targetCaE << "," << high.targetCaE << "," << lowStats[0] << ","
      << highStats[0] << "," << peakRatio << "," << lowStats[1] << ","
      << highStats[1] << "," << lowStats[2] << "," << highStats[2] << ","
      << tailRatio << "," << source << "," << status << "\n";
}

void writePoissonFaceAxialWindowRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    bool totalCurrent,
    double minAreaDi2) {
  const PoissonFaceDevelopedStats lowStats =
      poissonFaceDevelopedStats(low, minAreaDi2, totalCurrent);
  const PoissonFaceDevelopedStats highStats =
      poissonFaceDevelopedStats(high, minAreaDi2, totalCurrent);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double currentRatio =
      comparable ? ratio(highStats.meanCurrent, lowStats.meanCurrent)
                 : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    const double currentScale =
        std::max(lowStats.meanCurrent, highStats.meanCurrent);
    status = currentScale <= 1e-30
                 ? "BLOCKED_ZERO_DEVELOPED_CURRENT_OBSERVABLE"
                 : (currentRatio <= 2.0
                        ? "APPROXIMATE_AXIAL_DEVELOPED_WINDOW_WEAK_SENSITIVITY"
                        : "DOWNGRADED_AXIAL_DEVELOPED_WINDOW_TOO_VOLTAGE_SENSITIVE");
  }
  const std::string observable =
      totalCurrent
          ? "face_consistent_alpha05_total_current=rho_e_phiFlux_plus_sigma_gradphi_flux;"
            "uses_Poisson_snGrad_flux"
          : "face_consistent_alpha05_convective_current=rho_e_phiFlux;"
            "uses_projected_Rhie_Chow_Poisson_face_flux";
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << observable << "," << minAreaDi2 << "," << lowStats.tailSamples << ","
      << highStats.tailSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.maxArea << ","
      << highStats.maxArea << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << lowStats.meanYOverDi << ","
      << highStats.meanYOverDi << "," << lowStats.meanCurrent << ","
      << highStats.meanCurrent << "," << currentRatio << "," << status << "\n";
}

void writePoissonFaceConvectiveFactorizationRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const PoissonFaceFactorStats lowStats = poissonFaceFactorStats(low, minAreaDi2);
  const PoissonFaceFactorStats highStats = poissonFaceFactorStats(high, minAreaDi2);
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
  const double convectiveRatio =
      comparable ? ratio(highStats.meanAbsConvectiveFlux,
                         lowStats.meanAbsConvectiveFlux)
                 : std::numeric_limits<double>::infinity();
  std::string status = "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW";
  if (comparable) {
    const double chargeScale =
        std::max(lowStats.meanAbsUpwindCharge, highStats.meanAbsUpwindCharge);
    const double faceFluxScale =
        std::max(lowStats.meanAbsFaceFlux, highStats.meanAbsFaceFlux);
    const double productScale =
        std::max(lowStats.meanAbsConvectiveFlux, highStats.meanAbsConvectiveFlux);
    const double currentScale = std::max(lowStats.meanCurrent, highStats.meanCurrent);
    if (chargeScale <= 1e-30) {
      status = "BLOCKED_ZERO_FACE_UPWIND_CHARGE";
    } else if (faceFluxScale <= 1e-30) {
      status = "BLOCKED_ZERO_FACE_FLUX";
    } else if (productScale <= 1e-30) {
      status = "BLOCKED_ZERO_FACE_CONVECTIVE_PRODUCT";
    } else if (currentScale <= 1e-30) {
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
      << highStats.meanArea << "," << areaRatio << "," << lowStats.meanCurrent
      << "," << highStats.meanCurrent << "," << currentRatio << ","
      << lowStats.meanAbsUpwindCharge << "," << highStats.meanAbsUpwindCharge
      << "," << chargeRatio << "," << lowStats.meanAbsFaceFlux << ","
      << highStats.meanAbsFaceFlux << "," << faceFluxRatio << ","
      << lowStats.meanAbsConvectiveFlux << ","
      << highStats.meanAbsConvectiveFlux << "," << convectiveRatio << ","
      << lowStats.maxAbsUpwindCharge << "," << highStats.maxAbsUpwindCharge
      << "," << lowStats.maxAbsFaceFlux << "," << highStats.maxAbsFaceFlux
      << "," << status << "\n";
}

void writePoissonFaceVelocityProjectionRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const PoissonFaceProjectionStats lowStats =
      poissonFaceProjectionStats(low, minAreaDi2);
  const PoissonFaceProjectionStats highStats =
      poissonFaceProjectionStats(high, minAreaDi2);
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
      comparable ? ratio(highStats.projectedAbsFaceFlux,
                         lowStats.projectedAbsFaceFlux)
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
    const double projectedScale =
        std::max(lowStats.projectedCurrent, highStats.projectedCurrent);
    const double rawCurrentScale = std::max(lowStats.rawCurrent, highStats.rawCurrent);
    const double rawFaceFluxScale =
        std::max(lowStats.rawAbsFaceFlux, highStats.rawAbsFaceFlux);
    if (projectedScale <= 1e-30) {
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

void writePaperCurrentCandidateRows(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high) {
  auto writeOne = [&](bool alpha05JetCurrent, const std::string& source) {
    const CurrentSensitivityStats lowStats =
        currentSensitivityStats(low, alpha05JetCurrent);
    const CurrentSensitivityStats highStats =
        currentSensitivityStats(high, alpha05JetCurrent);
    const double peakRatio = ratio(highStats.peak, lowStats.peak);
    const double tailRatio = ratio(highStats.meanTail, lowStats.meanTail);
    const double currentScale = std::max(lowStats.meanTail, highStats.meanTail);
    const std::string status =
        currentScale <= 1e-30
            ? "BLOCKED_ZERO_CURRENT_OBSERVABLE"
            : (tailRatio <= 2.0
                   ? "APPROXIMATE_WEAK_AVERAGE_VOLTAGE_SENSITIVITY"
                   : "DOWNGRADED_AVERAGE_CURRENT_TOO_VOLTAGE_SENSITIVE");
    csv << low.targetCaE << "," << high.targetCaE << "," << lowStats.peak
        << "," << highStats.peak << "," << peakRatio << ","
        << lowStats.meanAll << "," << highStats.meanAll << ","
        << lowStats.meanTail << "," << highStats.meanTail << ","
        << tailRatio << "," << source << "," << status << "\n";
  };
  writeOne(false, "Candido_Fig8b_text_average_current_not_influenced_by_voltage");
  writeOne(true,
           "Candido_Fig8b_current_ie=int_S_qe_U_dot_n_dS;"
           "alpha05_liquid_jet_cross_section");
}

void writePreconditionedCurrentPlaneRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const MidplaneDevelopmentStats lowFixed =
      midplaneDevelopmentStats(low, minAreaDi2);
  const MidplaneDevelopmentStats highFixed =
      midplaneDevelopmentStats(high, minAreaDi2);
  const AxialCurrentStats lowAxial = axialCurrentStats(low, minAreaDi2);
  const AxialCurrentStats highAxial = axialCurrentStats(high, minAreaDi2);
  const CurrentSensitivityStats lowCurrent = currentSensitivityStats(low, false);
  const CurrentSensitivityStats highCurrent = currentSensitivityStats(high, false);
  const bool fixedComparable =
      lowFixed.developedSamples > 0 && highFixed.developedSamples > 0;
  const bool axialComparable =
      lowAxial.developedSamples > 0 && highAxial.developedSamples > 0;
  const double fixedMeanRatio =
      fixedComparable ? ratio(highFixed.meanDevelopedAlpha05Current,
                              lowFixed.meanDevelopedAlpha05Current)
                      : 0.0;
  const double fixedPeakRatio =
      fixedComparable ? ratio(highFixed.peakDevelopedAlpha05Current,
                              lowFixed.peakDevelopedAlpha05Current)
                      : 0.0;
  const double axialConvectiveRatio =
      axialComparable ? ratio(highAxial.meanAlpha05Convective,
                              lowAxial.meanAlpha05Convective)
                      : 0.0;
  const double axialTotalRatio =
      axialComparable ? ratio(highAxial.meanAlpha05Total, lowAxial.meanAlpha05Total)
                      : 0.0;
  const double midplaneYOverDi =
      0.5 * setup.collectorDistance / std::max(setup.innerDiameter, 1e-30);
  const double tipYOverDi =
      opt.preconditionedJetTipYOverInnerDiameter > 0.0
          ? opt.preconditionedJetTipYOverInnerDiameter
          : midplaneYOverDi + 0.75;
  std::string status = "DOWNGRADED_NONFINITE_PRECONDITIONED_CURRENT_DIAGNOSTIC";
  const bool finite =
      std::isfinite(low.alphaMassDrift) && std::isfinite(high.alphaMassDrift) &&
      std::isfinite(low.maxDiv) && std::isfinite(high.maxDiv);
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
      status =
          "APPROXIMATE_PRECONDITIONED_FIXED_PLANE_WEAK_SENSITIVITY_DIAGNOSTIC_ONLY";
    }
  }
  csv << caseName << "," << low.targetCaE << "," << high.targetCaE << ","
      << low.steps << "," << high.steps << "," << minAreaDi2 << ","
      << midplaneYOverDi << "," << tipYOverDi << ","
      << opt.preconditionedJetRadiusInnerDiameters << ","
      << opt.preconditionedJetInterfaceWidthInnerDiameters << ","
      << opt.preconditionedJetVelocityScale << "," << low.alphaMassDrift
      << "," << high.alphaMassDrift << "," << low.maxDiv << ","
      << high.maxDiv << "," << lowFixed.developedSamples << ","
      << highFixed.developedSamples << "," << lowFixed.maxAlpha05AreaDi2
      << "," << highFixed.maxAlpha05AreaDi2 << ","
      << lowFixed.meanDevelopedAlpha05Current << ","
      << highFixed.meanDevelopedAlpha05Current << "," << fixedMeanRatio
      << "," << lowFixed.peakDevelopedAlpha05Current << ","
      << highFixed.peakDevelopedAlpha05Current << "," << fixedPeakRatio << ","
      << ratio(highCurrent.meanTail, lowCurrent.meanTail) << ","
      << ratio(highCurrent.peak, lowCurrent.peak) << ","
      << lowAxial.developedSamples << "," << highAxial.developedSamples << ","
      << lowAxial.meanAlpha05Convective << ","
      << highAxial.meanAlpha05Convective << "," << axialConvectiveRatio << ","
      << lowAxial.meanAlpha05Total << "," << highAxial.meanAlpha05Total << ","
      << axialTotalRatio << "," << status << "\n";
}

void writeReducedCollectorCurrentFixtureRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const MidplaneDevelopmentStats lowStats =
      midplaneDevelopmentStats(low, minAreaDi2);
  const MidplaneDevelopmentStats highStats =
      midplaneDevelopmentStats(high, minAreaDi2);
  const int lowTailSamples = static_cast<int>(low.history.size() / 2);
  const int highTailSamples = static_cast<int>(high.history.size() / 2);
  const bool comparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double meanRatio =
      comparable ? ratio(highStats.meanDevelopedAlpha05Current,
                         lowStats.meanDevelopedAlpha05Current)
                 : std::numeric_limits<double>::infinity();
  const double peakRatio =
      comparable ? ratio(highStats.peakDevelopedAlpha05Current,
                         lowStats.peakDevelopedAlpha05Current)
                 : std::numeric_limits<double>::infinity();
  std::string status = "DOWNGRADED_REDUCED_DISTANCE_NONFINITE_DIAGNOSTIC";
  const bool finiteQuality =
      std::isfinite(low.alphaMassDrift) && std::isfinite(high.alphaMassDrift) &&
      std::isfinite(low.maxDiv) && std::isfinite(high.maxDiv);
  if (finiteQuality) {
    if (low.alphaMassDrift > 1e-3 || high.alphaMassDrift > 1e-3 ||
        low.maxDiv > 1e-7 || high.maxDiv > 1e-7) {
      status = "DOWNGRADED_REDUCED_DISTANCE_NUMERICAL_FAILURE";
    } else if (!comparable) {
      status = "BLOCKED_REDUCED_DISTANCE_UNDEVELOPED_FIXED_PLANE";
    } else if (meanRatio <= 2.0) {
      status =
          "APPROXIMATE_REDUCED_DISTANCE_WEAK_SENSITIVITY_NOT_PAPER_GEOMETRY";
    } else {
      status = "DOWNGRADED_REDUCED_DISTANCE_TOO_VOLTAGE_SENSITIVE";
    }
  }
  const double collectorOverDi =
      setup.collectorDistance / std::max(setup.innerDiameter, 1e-30);
  csv << caseName << "," << setup.collectorDistance << ","
      << collectorOverDi << "," << 0.5 * collectorOverDi << ","
      << low.targetCaE << "," << high.targetCaE << "," << low.steps << ","
      << high.steps << "," << minAreaDi2 << "," << lowTailSamples << ","
      << highTailSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.maxAlpha05AreaDi2
      << "," << highStats.maxAlpha05AreaDi2 << "," << maxTailTipY(low)
      << "," << maxTailTipY(high) << ","
      << lowStats.meanDevelopedAlpha05Current << ","
      << highStats.meanDevelopedAlpha05Current << "," << meanRatio << ","
      << lowStats.peakDevelopedAlpha05Current << ","
      << highStats.peakDevelopedAlpha05Current << "," << peakRatio << ","
      << low.alphaMassDrift << "," << high.alphaMassDrift << ","
      << low.maxDiv << "," << high.maxDiv << "," << status << "\n";
}

void writeFig8bCurrentBlockerRow(
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
  const double chargeTailRatio =
      ratio(meanTailConvectiveCurrent(paperChargeHigh),
            meanTailConvectiveCurrent(paperChargeLow));
  const double inletTailRatio =
      ratio(meanTailConvectiveCurrent(inletVelocityHigh),
            meanTailConvectiveCurrent(inletVelocityLow));
  const double bestTailRatio = std::min(chargeTailRatio, inletTailRatio);
  const double chargePeakRatio =
      ratio(peakConvectiveCurrent(paperChargeHigh),
            peakConvectiveCurrent(paperChargeLow));
  const double inletPeakRatio =
      ratio(peakConvectiveCurrent(inletVelocityHigh),
            peakConvectiveCurrent(inletVelocityLow));
  const double bestPeakRatio = std::min(chargePeakRatio, inletPeakRatio);
  const int paperLowMidplane =
      tailDevelopedSamplesAtMidplane(inletVelocityLow, minAreaDi2);
  const int paperHighMidplane =
      tailDevelopedSamplesAtMidplane(inletVelocityHigh, minAreaDi2);
  const int reducedLowMidplane =
      tailDevelopedSamplesAtMidplane(reducedLow, minAreaDi2);
  const int reducedHighMidplane =
      tailDevelopedSamplesAtMidplane(reducedHigh, minAreaDi2);
  const double paperMidplaneYOverDi =
      0.5 * setup.collectorDistance / std::max(setup.innerDiameter, 1e-30);
  const double reducedMidplaneYOverDi =
      0.5 * reducedSetup.collectorDistance /
      std::max(reducedSetup.innerDiameter, 1e-30);
  const bool paperFixedPlaneDeveloped =
      paperLowMidplane > 0 && paperHighMidplane > 0;
  const bool reducedFixedPlaneDeveloped =
      reducedLowMidplane > 0 && reducedHighMidplane > 0;
  std::string status = "DOWNGRADED_FIG8B_CURRENT_NONFINITE_DIAGNOSTIC";
  if (std::isfinite(bestTailRatio) && std::isfinite(bestPeakRatio)) {
    if (!paperFixedPlaneDeveloped && !reducedFixedPlaneDeveloped) {
      status = "BLOCKED_COARSE_FIXTURE_FIG8B_CURRENT_UNDEVELOPED_FIXED_PLANE";
    } else if (bestTailRatio > 2.0) {
      status = "DOWNGRADED_FIG8B_CURRENT_TOO_VOLTAGE_SENSITIVE";
    } else {
      status = "APPROXIMATE_FIG8B_CURRENT_WEAK_SENSITIVITY";
    }
  }
  csv << "coarse_smoke_fig8b_current" << "," << opt.nx << "," << opt.ny
      << "," << opt.nz << "," << minAreaDi2 << ","
      << paperMidplaneYOverDi << "," << reducedMidplaneYOverDi << ","
      << paperLowMidplane << "," << paperHighMidplane << ","
      << reducedLowMidplane << "," << reducedHighMidplane << ","
      << maxTailTipY(inletVelocityLow) << ","
      << maxTailTipY(inletVelocityHigh) << "," << maxTailTipY(reducedLow)
      << "," << maxTailTipY(reducedHigh) << "," << chargeTailRatio << ","
      << inletTailRatio << "," << bestTailRatio << "," << chargePeakRatio
      << "," << inletPeakRatio << "," << bestPeakRatio << "," << status
      << "\n";
}

void writeJetCurrentMetricsRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  csv << caseName << "," << r.targetCaE << "," << r.voltage << ","
      << r.computedCaE << "," << r.finalMidplaneJetRadius << ","
      << r.maxConductiveCurrent << "," << r.maxConvectiveCurrent << ","
      << r.finalRadialAsymmetry << "," << r.maxVelocity << ","
      << r.alphaMassDrift << "," << r.maxDiv << "\n";
}

void writeCurrentScalingRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  csv << caseName << "," << r.targetCaE << "," << r.voltage << ","
      << r.computedCaE << "," << r.maxConductiveCurrent << ","
      << r.maxConvectiveCurrent << "," << r.maxElectricForce << ","
      << r.maxVelocity << "," << r.finalMidplaneJetRadius << ","
      << r.alphaMassDrift << "," << r.maxDiv << "\n";
}

void writeRefinementDiagnostics(
    std::ofstream& refinement,
    std::ofstream& refinementQuality,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& baseOpt) {
  struct RefinementRow {
    int n = 0;
    double radius = 0.0;
    double electricForce = 0.0;
    double csfForce = 0.0;
  };

  std::vector<RefinementRow> rows;
  for (int n : {10, 12, 14}) {
    electrospray::CandidoConeJetSmokeOptions3D opt = baseOpt;
    opt.nx = n;
    opt.nz = n;
    opt.ny = std::max(8, static_cast<int>(std::round(1.4 * n)));
    opt.steps = 2;
    const auto r = electrospray::runCandidoConeJetSmoke3D(0.25, setup, opt);
    refinement << "refine_" << n << "," << opt.nx << "," << opt.ny << ","
               << opt.nz << "," << r.cells << "," << r.faces << ","
               << r.steps << "," << r.targetCaE << "," << r.dt << ","
               << r.alphaMassDrift << "," << r.maxDiv << ","
               << r.maxElectricForce << "," << r.maxCsfForce << ","
               << r.finalMidplaneJetRadius << "," << r.finalRadialAsymmetry
               << "," << r.maxVelocity << "\n";
    check(r.cells > 0 && r.faces > 0, "refinement sweep mesh is non-empty");
    check(r.alphaMassDrift <= 1e-3, "refinement sweep mass drift bounded");
    check(r.maxDiv <= 1e-7, "refinement sweep continuity bounded");
    check(std::isfinite(r.maxElectricForce) &&
              std::isfinite(r.finalMidplaneJetRadius),
          "refinement sweep metrics finite");
    rows.push_back({n, r.finalMidplaneJetRadius, r.maxElectricForce,
                    r.maxCsfForce});
  }

  const auto writeQualityRow = [&](const std::string& observable,
                                   double RefinementRow::*member) {
    const double coarse = rows[0].*member;
    const double mid = rows[1].*member;
    const double fine = rows[2].*member;
    const double denom = std::max(std::abs(fine), 1e-30);
    const double coarseToMid = std::abs(mid - coarse) / denom;
    const double midToFine = std::abs(fine - mid) / denom;
    const bool decreasing = midToFine < coarseToMid;
    const bool boundedFineChange = midToFine <= 0.35;
    const std::string status =
        (decreasing && boundedFineChange) ? "PASS_CONVERGING"
                                          : "DOWNGRADED_NONCONVERGENT";
    refinementQuality << observable << "," << rows[0].n << "," << rows[1].n
                      << "," << rows[2].n << "," << coarse << "," << mid
                      << "," << fine << "," << coarseToMid << ","
                      << midToFine << "," << status << "\n";
  };
  writeQualityRow("final_midplane_jet_radius", &RefinementRow::radius);
  writeQualityRow("max_electric_force", &RefinementRow::electricForce);
  writeQualityRow("max_csf_force", &RefinementRow::csfForce);
}

void writeOpenBoundaryCurrentRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const AxialCurrentStats lowStats = axialCurrentStats(low, minAreaDi2);
  const AxialCurrentStats highStats = axialCurrentStats(high, minAreaDi2);
  const double tailRatio =
      ratio(meanTailConvectiveCurrent(high), meanTailConvectiveCurrent(low));
  const double peakRatio = ratio(peakConvectiveCurrent(high),
                                 peakConvectiveCurrent(low));
  const int lowMidplane = tailDevelopedSamplesAtMidplane(low, minAreaDi2);
  const int highMidplane = tailDevelopedSamplesAtMidplane(high, minAreaDi2);
  const bool axialComparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double axialConvectiveRatio =
      axialComparable ? ratio(highStats.meanAlpha05Convective,
                              lowStats.meanAlpha05Convective)
                      : std::numeric_limits<double>::infinity();
  const double axialTotalRatio =
      axialComparable ? ratio(highStats.meanAlpha05Total, lowStats.meanAlpha05Total)
                      : std::numeric_limits<double>::infinity();
  const double lowMorphologyError = maxMorphologyError04_07(low);
  const double highAsymmetry = maxRadialAsymmetry(high);
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
      std::isfinite(lowMorphologyError) && std::isfinite(highAsymmetry) &&
      std::isfinite(lowBoundaryActivity) && std::isfinite(highBoundaryActivity);
  if (finite) {
    const bool openActive = lowBoundaryActivity > 0.0 || highBoundaryActivity > 0.0;
    const bool numericalQuality =
        low.alphaMassDrift <= 1e-3 && high.alphaMassDrift <= 1e-3 &&
        low.maxDiv <= 1e-7 && high.maxDiv <= 1e-7;
    const bool fixedPlaneDeveloped = lowMidplane > 0 && highMidplane > 0;
    const bool weakAllPhase = tailRatio <= 2.0 && peakRatio <= 2.0;
    const bool weakAxial = axialComparable && axialConvectiveRatio <= 2.0;
    if (!openActive) {
      status = "BLOCKED_OPEN_BOUNDARY_NO_MEASURABLE_FLUX";
    } else if (!numericalQuality) {
      status = "DOWNGRADED_OPEN_BOUNDARY_NUMERICAL_QUALITY";
    } else if (!fixedPlaneDeveloped) {
      status = "BLOCKED_OPEN_BOUNDARY_FIXED_PLANE_UNDEVELOPED";
    } else if (!weakAllPhase || !weakAxial) {
      status = "DOWNGRADED_OPEN_BOUNDARY_CURRENT_RATIO_ABOVE_WEAK_BAR";
    } else if (lowMorphologyError > 10.0 || highAsymmetry < 0.05) {
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
      << high.alphaMassDrift << "," << low.maxDiv << "," << high.maxDiv
      << "," << low.cumulativeBoundaryLiquidInflow << ","
      << high.cumulativeBoundaryLiquidInflow << ","
      << low.cumulativeBoundaryLiquidOutflow << ","
      << high.cumulativeBoundaryLiquidOutflow << ","
      << low.cumulativeBoundaryLiquidFlux << ","
      << high.cumulativeBoundaryLiquidFlux << "," << low.massBudgetResidual
      << "," << high.massBudgetResidual << "," << lowMorphologyError << ","
      << highAsymmetry << "," << tailRatio << "," << peakRatio << ","
      << lowMidplane << "," << highMidplane << ","
      << lowStats.developedSamples << "," << highStats.developedSamples << ","
      << lowStats.meanArea << "," << highStats.meanArea << ","
      << lowStats.meanAlpha05Convective << ","
      << highStats.meanAlpha05Convective << "," << axialConvectiveRatio << ","
      << lowStats.meanAlpha05Total << "," << highStats.meanAlpha05Total << ","
      << axialTotalRatio << "," << status << "\n";
}

void writeMovingCollectorBoundaryRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const AxialCurrentStats lowStats = axialCurrentStats(low, minAreaDi2);
  const AxialCurrentStats highStats = axialCurrentStats(high, minAreaDi2);
  const double tailRatio =
      ratio(meanTailConvectiveCurrent(high), meanTailConvectiveCurrent(low));
  const double peakRatio = ratio(peakConvectiveCurrent(high),
                                 peakConvectiveCurrent(low));
  const int lowMidplane = tailDevelopedSamplesAtMidplane(low, minAreaDi2);
  const int highMidplane = tailDevelopedSamplesAtMidplane(high, minAreaDi2);
  const bool axialComparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double axialConvectiveRatio =
      axialComparable ? ratio(highStats.meanAlpha05Convective,
                              lowStats.meanAlpha05Convective)
                      : std::numeric_limits<double>::infinity();
  const double axialTotalRatio =
      axialComparable ? ratio(highStats.meanAlpha05Total, lowStats.meanAlpha05Total)
                      : std::numeric_limits<double>::infinity();
  const double lowMorphologyError = maxMorphologyError04_07(low);
  const double highAsymmetry = maxRadialAsymmetry(high);
  const double collectorDimensionlessSpeed =
      electrospray::candidoDimensionlessCollectorVelocityScale(setup);
  std::string status = "DOWNGRADED_NONFINITE_MOVING_COLLECTOR_DIAGNOSTIC";
  const bool finite =
      std::isfinite(tailRatio) && std::isfinite(peakRatio) &&
      std::isfinite(axialConvectiveRatio) && std::isfinite(axialTotalRatio) &&
      std::isfinite(lowMorphologyError) && std::isfinite(highAsymmetry) &&
      std::isfinite(collectorDimensionlessSpeed);
  if (finite) {
    const bool numericalQuality =
        low.alphaMassDrift <= 1e-3 && high.alphaMassDrift <= 1e-3 &&
        low.maxDiv <= 1e-7 && high.maxDiv <= 1e-7;
    const bool fixedPlaneDeveloped = lowMidplane > 0 && highMidplane > 0;
    const bool weakAllPhase = tailRatio <= 2.0 && peakRatio <= 2.0;
    const bool weakAxial = axialComparable && axialConvectiveRatio <= 2.0;
    if (!numericalQuality) {
      status = "DOWNGRADED_MOVING_COLLECTOR_NUMERICAL_QUALITY";
    } else if (!fixedPlaneDeveloped) {
      status = "BLOCKED_MOVING_COLLECTOR_FIXED_PLANE_UNDEVELOPED";
    } else if (!weakAllPhase || !weakAxial) {
      status = "DOWNGRADED_MOVING_COLLECTOR_CURRENT_RATIO_ABOVE_WEAK_BAR";
    } else if (lowMorphologyError > 10.0 || highAsymmetry < 0.05) {
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
      << "," << lowMorphologyError << "," << highAsymmetry << ","
      << tailRatio << "," << peakRatio << "," << lowMidplane << ","
      << highMidplane << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << lowStats.meanAlpha05Convective << ","
      << highStats.meanAlpha05Convective << "," << axialConvectiveRatio << ","
      << lowStats.meanAlpha05Total << "," << highStats.meanAlpha05Total << ","
      << axialTotalRatio << "," << status << "\n";
}

void writeMomentumSourceFactorizationDiagnosticRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const SourceStats lowStats = sourceStats(low, minAreaDi2);
  const SourceStats highStats = sourceStats(high, minAreaDi2);
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
    const double velocityScale =
        std::max(lowStats.meanAbsUy, highStats.meanAbsUy);
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
      << lowStats.meanAbsSurfaceSource << ","
      << highStats.meanAbsSurfaceSource << "," << surfaceSourceRatio << ","
      << lowStats.meanAbsSource << "," << highStats.meanAbsSource << ","
      << sourceRatio << "," << lowStats.meanAbsAcceleration << ","
      << highStats.meanAbsAcceleration << "," << accelerationRatio << ","
      << dominant << "," << status << "\n";
}

electrospray::CandidoConeJetSmokeOptions3D rayleighLimitedBaseOptions() {
  electrospray::CandidoConeJetSmokeOptions3D opt;
  opt.conservativeChargeBounding = true;
  opt.useRayleighChargeLimit = true;
  return opt;
}

}  // namespace

int main() {
  std::filesystem::create_directories("benchmark_logs");

  electrospray::CandidoTaylorConeJetSetup setup;
  constexpr double minAreaDi2 = 1e-4;

  auto faceImplicitElectricOpt = rayleighLimitedBaseOptions();
  faceImplicitElectricOpt.usePoissonFaceConductiveCurrent = true;
  faceImplicitElectricOpt.usePoissonFaceMaxwellForce = true;
  faceImplicitElectricOpt.implicitOhmicChargeProjection = true;

  electrospray::CandidoConeJetSmokeOptions3D relaxationLimitedElectricOpt =
      faceImplicitElectricOpt;
  relaxationLimitedElectricOpt.useDimensionalElectricalScaling = true;
  relaxationLimitedElectricOpt.useElectricRelaxationTimeStepLimit = true;
  relaxationLimitedElectricOpt.electricRelaxationTimeStepSafety = 1.0;

  electrospray::CandidoConeJetSmokeOptions3D caIndependentDriveOpt =
      relaxationLimitedElectricOpt;
  caIndependentDriveOpt.electricDriveCaExponent = 0.0;

  electrospray::CandidoConeJetSmokeOptions3D caIndependentBoundaryOpt =
      caIndependentDriveOpt;
  caIndependentBoundaryOpt.useBoundaryChargeAdvection = true;

  electrospray::CandidoConeJetSmokeOptions3D paperChargeBoundaryOpt =
      caIndependentBoundaryOpt;
  paperChargeBoundaryOpt.useVofInletBoundaryAlpha = true;
  paperChargeBoundaryOpt.suppressNozzleConductiveChargeFlux = true;

  electrospray::CandidoConeJetSmokeOptions3D paperInletVelocityOpt =
      paperChargeBoundaryOpt;
  paperInletVelocityOpt.useFullyDevelopedInletVelocityBoundary = true;

  electrospray::CandidoConeJetSmokeOptions3D paperOpenBoundaryOpt =
      paperInletVelocityOpt;
  paperOpenBoundaryOpt.useOpenAtmosphericBoundaryFlux = true;

  electrospray::CandidoConeJetSmokeOptions3D movingCollectorOpt =
      paperOpenBoundaryOpt;
  movingCollectorOpt.useMovingCollectorWall = true;

  electrospray::CandidoTaylorConeJetSetup reducedCollectorSetup = setup;
  reducedCollectorSetup.collectorDistance = 0.75e-3;
  electrospray::CandidoConeJetSmokeOptions3D reducedCollectorInletAlphaOpt =
      caIndependentBoundaryOpt;
  reducedCollectorInletAlphaOpt.useVofInletBoundaryAlpha = true;

  electrospray::CandidoConeJetSmokeOptions3D preconditionedCurrentOpt =
      paperOpenBoundaryOpt;
  preconditionedCurrentOpt.usePreconditionedPaperCurrentJet = true;
  preconditionedCurrentOpt.preconditionedJetRadiusInnerDiameters = 0.65;
  preconditionedCurrentOpt.preconditionedJetInterfaceWidthInnerDiameters = 0.20;
  preconditionedCurrentOpt.preconditionedJetVelocityScale = 1.0;

  electrospray::CandidoConeJetSmokeOptions3D unitMaxwellBoundaryOpt =
      caIndependentBoundaryOpt;
  unitMaxwellBoundaryOpt.electricDriveReferenceScale = 1.0;

  auto faceConsistentElectricOpt = rayleighLimitedBaseOptions();
  faceConsistentElectricOpt.usePoissonFaceConductiveCurrent = true;
  faceConsistentElectricOpt.usePoissonFaceMaxwellForce = true;

  const auto caDriveLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             caIndependentDriveOpt);
  const auto caDriveHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             caIndependentDriveOpt);
  const auto caBoundaryLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             caIndependentBoundaryOpt);
  const auto caBoundaryHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             caIndependentBoundaryOpt);
  const auto unitMaxwellLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             unitMaxwellBoundaryOpt);
  const auto unitMaxwellHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             unitMaxwellBoundaryOpt);
  const auto faceConsistentLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             faceConsistentElectricOpt);
  const auto faceConsistentHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             faceConsistentElectricOpt);
  const auto paperBoundaryLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperChargeBoundaryOpt);
  const auto paperBoundaryHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperChargeBoundaryOpt);
  const auto paperInletLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperInletVelocityOpt);
  const auto paperInletHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperInletVelocityOpt);
  const auto paperOpenLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperOpenBoundaryOpt);
  const auto paperOpenHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperOpenBoundaryOpt);
  const auto movingCollectorLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             movingCollectorOpt);
  const auto movingCollectorHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             movingCollectorOpt);
  const auto reducedCollectorLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, reducedCollectorSetup,
                                             caIndependentBoundaryOpt);
  const auto reducedCollectorHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, reducedCollectorSetup,
                                             caIndependentBoundaryOpt);
  const auto reducedCollectorInletAlphaLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, reducedCollectorSetup,
                                             reducedCollectorInletAlphaOpt);
  const auto reducedCollectorInletAlphaHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, reducedCollectorSetup,
                                             reducedCollectorInletAlphaOpt);
  const auto preconditionedCurrentLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             preconditionedCurrentOpt);
  const auto preconditionedCurrentHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             preconditionedCurrentOpt);

  checkSmokeNumerics(caDriveLow, "Ca-independent drive low-CaE");
  checkSmokeNumerics(caDriveHigh, "Ca-independent drive high-CaE");
  checkSmokeNumerics(caBoundaryLow, "boundary-advected drive low-CaE");
  checkSmokeNumerics(caBoundaryHigh, "boundary-advected drive high-CaE");
  checkSmokeNumerics(unitMaxwellLow, "unit-Maxwell drive low-CaE");
  checkSmokeNumerics(unitMaxwellHigh, "unit-Maxwell drive high-CaE");
  checkSmokeNumerics(faceConsistentLow, "face-consistent electric low-CaE");
  checkSmokeNumerics(faceConsistentHigh, "face-consistent electric high-CaE");
  checkSmokeNumerics(paperBoundaryLow, "paper charge-boundary low-CaE");
  checkSmokeNumerics(paperBoundaryHigh, "paper charge-boundary high-CaE");
  checkSmokeNumerics(paperInletLow, "paper inlet-velocity low-CaE");
  checkSmokeNumerics(paperInletHigh, "paper inlet-velocity high-CaE");
  checkSmokeNumerics(paperOpenLow, "paper open-boundary low-CaE");
  checkSmokeNumerics(paperOpenHigh, "paper open-boundary high-CaE");
  checkSmokeNumerics(movingCollectorLow, "moving-collector low-CaE");
  checkSmokeNumerics(movingCollectorHigh, "moving-collector high-CaE");
  checkSmokeNumerics(reducedCollectorLow,
                     "reduced-collector boundary low-CaE");
  checkSmokeNumerics(reducedCollectorHigh,
                     "reduced-collector boundary high-CaE");
  checkSmokeNumerics(reducedCollectorInletAlphaLow,
                     "reduced-collector inlet-alpha low-CaE");
  checkSmokeNumerics(reducedCollectorInletAlphaHigh,
                     "reduced-collector inlet-alpha high-CaE");
  checkSmokeNumerics(preconditionedCurrentLow,
                     "preconditioned current-plane low-CaE");
  checkSmokeNumerics(preconditionedCurrentHigh,
                     "preconditioned current-plane high-CaE");

  std::ofstream csv(
      "benchmark_logs/candido_momentum_source_factorization3d.csv");
  csv << "case,low_ca_e,high_ca_e,min_alpha05_area_di2,"
         "low_tail_samples,high_tail_samples,"
         "low_developed_samples,high_developed_samples,"
         "low_mean_area_di2,high_mean_area_di2,area_ratio,"
         "low_mean_abs_uy,high_mean_abs_uy,velocity_ratio,"
         "low_mean_abs_electric_source,high_mean_abs_electric_source,"
         "electric_source_ratio,"
         "low_mean_abs_surface_source,high_mean_abs_surface_source,"
         "surface_source_ratio,"
         "low_mean_abs_source,high_mean_abs_source,source_ratio,"
         "low_mean_abs_acceleration,high_mean_abs_acceleration,"
         "acceleration_ratio,dominant_factor,status\n";
  writeMomentumSourceFactorizationDiagnosticRow(
      csv, "ca_independent_drive_relaxation_limited_alpha05", caDriveLow,
      caDriveHigh, minAreaDi2);
  writeMomentumSourceFactorizationDiagnosticRow(
      csv, "ca_independent_drive_boundary_advected_alpha05", caBoundaryLow,
      caBoundaryHigh, minAreaDi2);
  writeMomentumSourceFactorizationDiagnosticRow(
      csv, "unit_maxwell_drive_boundary_advected_alpha05", unitMaxwellLow,
      unitMaxwellHigh, minAreaDi2);
  writeMomentumSourceFactorizationDiagnosticRow(
      csv, "paper_charge_boundary_alpha05", paperBoundaryLow,
      paperBoundaryHigh, minAreaDi2);
  writeMomentumSourceFactorizationDiagnosticRow(
      csv, "paper_inlet_velocity_alpha05", paperInletLow, paperInletHigh,
      minAreaDi2);
  writeMomentumSourceFactorizationDiagnosticRow(
      csv, "paper_inlet_velocity_open_atmosphere_alpha05", paperOpenLow,
      paperOpenHigh, minAreaDi2);
  writeMomentumSourceFactorizationDiagnosticRow(
      csv, "paper_inlet_velocity_open_atmosphere_moving_collector_alpha05",
      movingCollectorLow, movingCollectorHigh, minAreaDi2);
  csv.flush();
  check(csv.good(), "momentum source factorization diagnostic CSV written");

  std::ofstream paperChargeCurrent(
      "benchmark_logs/"
      "candido_current_voltage_sensitivity_paper_charge_boundary3d.csv");
  paperChargeCurrent
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  writePaperCurrentCandidateRows(paperChargeCurrent, paperBoundaryLow,
                                 paperBoundaryHigh);
  paperChargeCurrent.flush();

  std::ofstream paperInletCurrent(
      "benchmark_logs/"
      "candido_current_voltage_sensitivity_paper_inlet_velocity3d.csv");
  paperInletCurrent
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  writePaperCurrentCandidateRows(paperInletCurrent, paperInletLow,
                                 paperInletHigh);
  paperInletCurrent.flush();

  check(paperChargeCurrent.good(),
        "paper charge-boundary current candidate CSV written");
  check(paperInletCurrent.good(),
        "paper inlet-velocity current candidate CSV written");

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
  writePreconditionedCurrentPlaneRow(
      preconditionedCurrentPlane, "paper_preconditioned_current_plane", setup,
      preconditionedCurrentOpt, preconditionedCurrentLow,
      preconditionedCurrentHigh, minAreaDi2);
  preconditionedCurrentPlane.flush();
  check(preconditionedCurrentPlane.good(),
        "preconditioned current-plane diagnostic CSV written");

  std::ofstream reducedCollectorCurrent(
      "benchmark_logs/candido_reduced_collector_current_fixture3d.csv");
  reducedCollectorCurrent
      << "case,collector_distance_m,collector_distance_over_Di,"
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
  writeReducedCollectorCurrentFixtureRow(
      reducedCollectorCurrent,
      "ca_independent_boundary_reduced_collector_0_75mm",
      reducedCollectorSetup, reducedCollectorLow, reducedCollectorHigh,
      minAreaDi2);
  writeReducedCollectorCurrentFixtureRow(
      reducedCollectorCurrent,
      "ca_independent_boundary_inlet_alpha_reduced_collector_0_75mm",
      reducedCollectorSetup, reducedCollectorInletAlphaLow,
      reducedCollectorInletAlphaHigh, minAreaDi2);
  reducedCollectorCurrent.flush();
  check(reducedCollectorCurrent.good(),
        "reduced collector current fixture CSV written");

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
  writeFig8bCurrentBlockerRow(
      fig8bCurrentBlocker, setup, reducedCollectorSetup, paperInletVelocityOpt,
      paperBoundaryLow, paperBoundaryHigh, paperInletLow, paperInletHigh,
      reducedCollectorInletAlphaLow, reducedCollectorInletAlphaHigh,
      minAreaDi2);
  fig8bCurrentBlocker.flush();
  check(fig8bCurrentBlocker.good(), "Fig. 8b current blocker CSV written");

  std::ofstream jetCurrent(
      "benchmark_logs/candido_jet_current_metrics3d.csv");
  jetCurrent
      << "case,target_ca_e,voltage,computed_ca_e,final_midplane_jet_radius,"
         "max_conductive_current,max_convective_current,"
         "final_radial_asymmetry,max_velocity,alpha_mass_drift,max_div\n";
  std::ofstream currentScaling(
      "benchmark_logs/candido_current_scaling3d.csv");
  currentScaling
      << "case,target_ca_e,voltage,computed_ca_e,max_conductive_current,"
         "max_convective_current,max_electric_force,max_velocity,"
         "final_midplane_jet_radius,alpha_mass_drift,max_div\n";
  auto writeCurrentMetricPair =
      [&](const std::string& prefix,
          const electrospray::CandidoConeJetSmokeReport3D& low,
          const electrospray::CandidoConeJetSmokeReport3D& high) {
        writeJetCurrentMetricsRow(jetCurrent, prefix + "_ca025", low);
        writeJetCurrentMetricsRow(jetCurrent, prefix + "_ca042", high);
        writeCurrentScalingRow(currentScaling, prefix + "_ca025", low);
        writeCurrentScalingRow(currentScaling, prefix + "_ca042", high);
      };
  writeCurrentMetricPair("ca_independent_drive", caDriveLow, caDriveHigh);
  writeCurrentMetricPair("ca_independent_boundary", caBoundaryLow,
                         caBoundaryHigh);
  writeCurrentMetricPair("paper_charge_boundary", paperBoundaryLow,
                         paperBoundaryHigh);
  writeCurrentMetricPair("paper_inlet_velocity", paperInletLow,
                         paperInletHigh);
  writeCurrentMetricPair("paper_inlet_velocity_open_atmosphere", paperOpenLow,
                         paperOpenHigh);
  writeCurrentMetricPair(
      "paper_inlet_velocity_open_atmosphere_moving_collector",
      movingCollectorLow, movingCollectorHigh);
  writeCurrentMetricPair("paper_preconditioned_current_plane",
                         preconditionedCurrentLow,
                         preconditionedCurrentHigh);
  jetCurrent.flush();
  currentScaling.flush();
  check(jetCurrent.good(), "jet/current metrics CSV written");
  check(currentScaling.good(), "current scaling CSV written");

  std::ofstream refinement(
      "benchmark_logs/candido_refinement_sweep3d.csv");
  refinement
      << "case,nx,ny,nz,cells,faces,steps,target_ca_e,dt,"
         "alpha_mass_drift,max_div,max_electric_force,max_csf_force,"
         "final_midplane_jet_radius,final_radial_asymmetry,max_velocity\n";
  std::ofstream refinementQuality(
      "benchmark_logs/candido_refinement_quality3d.csv");
  refinementQuality
      << "observable,coarse_n,mid_n,fine_n,coarse_value,mid_value,"
         "fine_value,coarse_to_mid_relative_change,"
         "mid_to_fine_relative_change,status\n";
  writeRefinementDiagnostics(refinement, refinementQuality, setup,
                             rayleighLimitedBaseOptions());
  refinement.flush();
  refinementQuality.flush();
  check(refinement.good(), "refinement sweep CSV written");
  check(refinementQuality.good(), "refinement quality CSV written");

  std::ofstream poissonTotalCurrent(
      "benchmark_logs/candido_current_voltage_sensitivity_poisson_face_total3d.csv");
  poissonTotalCurrent
      << "low_ca_e,high_ca_e,low_peak_poisson_face_total_current,"
         "high_peak_poisson_face_total_current,peak_current_ratio,"
         "low_mean_all_poisson_face_total_current,"
         "high_mean_all_poisson_face_total_current,"
         "low_mean_tail_poisson_face_total_current,"
         "high_mean_tail_poisson_face_total_current,"
         "tail_mean_current_ratio,external_source,status\n";
  writePoissonFaceCurrentVoltageRow(poissonTotalCurrent, faceConsistentLow,
                                    faceConsistentHigh, false);
  poissonTotalCurrent.flush();

  std::ofstream poissonAlpha05TotalCurrent(
      "benchmark_logs/"
      "candido_current_voltage_sensitivity_poisson_face_alpha05_total3d.csv");
  poissonAlpha05TotalCurrent
      << "low_ca_e,high_ca_e,low_peak_poisson_face_alpha05_total_current,"
         "high_peak_poisson_face_alpha05_total_current,peak_current_ratio,"
         "low_mean_all_poisson_face_alpha05_total_current,"
         "high_mean_all_poisson_face_alpha05_total_current,"
         "low_mean_tail_poisson_face_alpha05_total_current,"
         "high_mean_tail_poisson_face_alpha05_total_current,"
         "tail_mean_current_ratio,external_source,status\n";
  writePoissonFaceCurrentVoltageRow(poissonAlpha05TotalCurrent,
                                    faceConsistentLow, faceConsistentHigh, true);
  poissonAlpha05TotalCurrent.flush();

  std::ofstream axialWindow(
      "benchmark_logs/candido_axial_developed_jet_current_window3d.csv");
  axialWindow << "case,low_ca_e,high_ca_e,observable,min_alpha05_area_di2,"
                 "low_tail_samples,high_tail_samples,low_developed_samples,"
                 "high_developed_samples,low_max_area_di2,high_max_area_di2,"
                 "low_mean_developed_area_di2,high_mean_developed_area_di2,"
                 "low_mean_developed_y_over_Di,high_mean_developed_y_over_Di,"
                 "low_mean_developed_current,high_mean_developed_current,"
                 "developed_current_ratio,status\n";
  writePoissonFaceAxialWindowRow(
      axialWindow, "face_consistent_electric_poisson_face_alpha05_total",
      faceConsistentLow, faceConsistentHigh, true, minAreaDi2);
  writePoissonFaceAxialWindowRow(
      axialWindow, "paper_charge_boundary_poisson_face_alpha05_total",
      paperBoundaryLow, paperBoundaryHigh, true, minAreaDi2);
  writePoissonFaceAxialWindowRow(
      axialWindow, "paper_charge_boundary_poisson_face_alpha05_convective",
      paperBoundaryLow, paperBoundaryHigh, false, minAreaDi2);
  writePoissonFaceAxialWindowRow(
      axialWindow, "paper_inlet_velocity_poisson_face_alpha05_total",
      paperInletLow, paperInletHigh, true, minAreaDi2);
  writePoissonFaceAxialWindowRow(
      axialWindow, "paper_inlet_velocity_poisson_face_alpha05_convective",
      paperInletLow, paperInletHigh, false, minAreaDi2);
  writePoissonFaceAxialWindowRow(
      axialWindow,
      "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05_total",
      paperOpenLow, paperOpenHigh, true, minAreaDi2);
  writePoissonFaceAxialWindowRow(
      axialWindow,
      "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05_convective",
      paperOpenLow, paperOpenHigh, false, minAreaDi2);
  writePoissonFaceAxialWindowRow(
      axialWindow,
      "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05_total",
      movingCollectorLow, movingCollectorHigh, true, minAreaDi2);
  writePoissonFaceAxialWindowRow(
      axialWindow,
      "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05_convective",
      movingCollectorLow, movingCollectorHigh, false, minAreaDi2);
  axialWindow.flush();

  std::ofstream poissonConvectiveFactorization(
      "benchmark_logs/candido_poisson_face_convective_factorization3d.csv");
  poissonConvectiveFactorization
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
  writePoissonFaceConvectiveFactorizationRow(
      poissonConvectiveFactorization, "paper_charge_boundary_poisson_face_alpha05",
      paperBoundaryLow, paperBoundaryHigh, minAreaDi2);
  writePoissonFaceConvectiveFactorizationRow(
      poissonConvectiveFactorization, "paper_inlet_velocity_poisson_face_alpha05",
      paperInletLow, paperInletHigh, minAreaDi2);
  writePoissonFaceConvectiveFactorizationRow(
      poissonConvectiveFactorization,
      "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05",
      paperOpenLow, paperOpenHigh, minAreaDi2);
  writePoissonFaceConvectiveFactorizationRow(
      poissonConvectiveFactorization,
      "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05",
      movingCollectorLow, movingCollectorHigh, minAreaDi2);
  poissonConvectiveFactorization.flush();

  std::ofstream poissonVelocityProjection(
      "benchmark_logs/"
      "candido_poisson_face_velocity_projection_factorization3d.csv");
  poissonVelocityProjection
      << "case,low_ca_e,high_ca_e,min_alpha05_area_di2,low_tail_samples,"
         "high_tail_samples,low_developed_samples,high_developed_samples,"
         "low_mean_area_di2,high_mean_area_di2,area_ratio,"
         "low_projected_current,high_projected_current,projected_current_ratio,"
         "low_raw_velocity_current,high_raw_velocity_current,"
         "raw_velocity_current_ratio,low_projected_abs_face_flux,"
         "high_projected_abs_face_flux,projected_face_flux_ratio,"
         "low_raw_velocity_abs_face_flux,high_raw_velocity_abs_face_flux,"
         "raw_velocity_face_flux_ratio,low_projected_abs_upwind_charge,"
         "high_projected_abs_upwind_charge,low_raw_velocity_abs_upwind_charge,"
         "high_raw_velocity_abs_upwind_charge,"
         "low_projected_abs_convective_flux,"
         "high_projected_abs_convective_flux,"
         "low_raw_velocity_abs_convective_flux,"
         "high_raw_velocity_abs_convective_flux,"
         "low_projected_to_raw_current,high_projected_to_raw_current,"
         "low_projected_to_raw_face_flux,high_projected_to_raw_face_flux,status\n";
  writePoissonFaceVelocityProjectionRow(
      poissonVelocityProjection, "paper_charge_boundary_poisson_face_alpha05",
      paperBoundaryLow, paperBoundaryHigh, minAreaDi2);
  writePoissonFaceVelocityProjectionRow(
      poissonVelocityProjection, "paper_inlet_velocity_poisson_face_alpha05",
      paperInletLow, paperInletHigh, minAreaDi2);
  writePoissonFaceVelocityProjectionRow(
      poissonVelocityProjection,
      "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05",
      paperOpenLow, paperOpenHigh, minAreaDi2);
  writePoissonFaceVelocityProjectionRow(
      poissonVelocityProjection,
      "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05",
      movingCollectorLow, movingCollectorHigh, minAreaDi2);
  poissonVelocityProjection.flush();

  check(poissonTotalCurrent.good(), "Poisson-face total current CSV written");
  check(poissonAlpha05TotalCurrent.good(),
        "Poisson-face alpha05 total current CSV written");
  check(axialWindow.good(), "Poisson-face axial window CSV written");
  check(poissonConvectiveFactorization.good(),
        "Poisson-face convective factorization CSV written");
  check(poissonVelocityProjection.good(),
        "Poisson-face velocity projection factorization CSV written");

  std::ofstream openBoundaryCsv(
      "benchmark_logs/candido_open_boundary_current_diagnostic3d.csv");
  openBoundaryCsv
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
  writeOpenBoundaryCurrentRow(openBoundaryCsv,
                              "paper_inlet_velocity_open_atmosphere", setup,
                              paperOpenLow, paperOpenHigh, minAreaDi2);
  openBoundaryCsv.flush();

  std::ofstream movingCollectorCsv(
      "benchmark_logs/candido_moving_collector_boundary_diagnostic3d.csv");
  movingCollectorCsv
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
  writeMovingCollectorBoundaryRow(
      movingCollectorCsv,
      "paper_inlet_velocity_open_atmosphere_moving_collector", setup,
      movingCollectorLow, movingCollectorHigh, minAreaDi2);
  movingCollectorCsv.flush();

  check(openBoundaryCsv.good(), "open-boundary current diagnostic CSV written");
  check(movingCollectorCsv.good(),
        "moving-collector boundary diagnostic CSV written");

  std::cout << "candido_momentum_source_factorization_rows=7\n";
  std::cout << "paper_charge_boundary_status="
            << sourceStats(paperBoundaryLow, minAreaDi2).developedSamples
            << "," << sourceStats(paperBoundaryHigh, minAreaDi2).developedSamples
            << "\n";
  return 0;
}
