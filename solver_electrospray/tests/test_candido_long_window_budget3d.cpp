#include "TestUtil.hpp"
#include "electrospray/CandidoTaylorConeJet3D.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr double kMinAreaDi2 = 1e-4;

double ratio(double highValue, double lowValue) {
  return highValue / std::max(lowValue, 1e-30);
}

double absRatio(double highValue, double lowValue) {
  return std::abs(highValue) / std::max(std::abs(lowValue), 1e-30);
}

struct AxialStats {
  int developedSamples = 0;
  double meanArea = 0.0;
  double meanAlpha05Convective = 0.0;
  double meanAlpha05Total = 0.0;
  double meanAbsCharge = 0.0;
  double meanAbsUy = 0.0;
};

struct MidplaneStats {
  int developedSamples = 0;
  double maxAlpha05AreaDi2 = 0.0;
  double maxTipY = 0.0;
  double firstDevelopedStep = -1.0;
  double firstDevelopedTimeMs = -1.0;
  double meanDevelopedAlpha05Current = 0.0;
  double peakDevelopedAlpha05Current = 0.0;
};

AxialStats axialStats(const electrospray::CandidoConeJetSmokeReport3D& r,
                      double minAreaDi2) {
  AxialStats s;
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

MidplaneStats midplaneStats(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  MidplaneStats s;
  const double timeScaleMs = r.hydrodynamicTimeScale * 1.0e3;
  const size_t tailStart = r.history.size() / 2;
  for (size_t i = tailStart; i < r.history.size(); ++i) {
    const auto& h = r.history[i];
    s.maxAlpha05AreaDi2 =
        std::max(s.maxAlpha05AreaDi2, h.midplaneAlpha05AreaDi2);
    s.maxTipY = std::max(s.maxTipY, h.tipY);
    if (h.midplaneAlpha05AreaDi2 < minAreaDi2) continue;
    if (s.developedSamples == 0) {
      s.firstDevelopedStep = static_cast<double>(h.step);
      s.firstDevelopedTimeMs = h.time * timeScaleMs;
    }
    ++s.developedSamples;
    s.meanDevelopedAlpha05Current += std::abs(h.alpha05ConvectiveCurrent);
    s.peakDevelopedAlpha05Current =
        std::max(s.peakDevelopedAlpha05Current,
                 std::abs(h.alpha05ConvectiveCurrent));
  }
  if (s.developedSamples > 0) {
    s.meanDevelopedAlpha05Current /=
        static_cast<double>(s.developedSamples);
  }
  return s;
}

const electrospray::CandidoConeJetHistorySample3D& nearestHistoryAtMs(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double timeMs) {
  check(!r.history.empty(), "Candido budget fixture history is non-empty");
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
  return *best;
}

double morphologyErrorAtMs(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double timeMs,
    double referenceVolumeDi3) {
  const auto& h = nearestHistoryAtMs(r, timeMs);
  return 100.0 * (h.morphologyVolumeDi3 - referenceVolumeDi3) /
         std::max(std::abs(referenceVolumeDi3), 1e-30);
}

double maxMorphologyError04_07(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const double e04 = morphologyErrorAtMs(r, 0.4, 1.2826510303495016);
  const double e07 = morphologyErrorAtMs(r, 0.7, 1.2550259882802302);
  return std::max(std::abs(e04), std::abs(e07));
}

double maxRadialAsymmetry(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  double value = 0.0;
  for (const auto& h : r.history) value = std::max(value, h.radialAsymmetry);
  return value;
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
  return sum / std::max(samples, 1);
}

double peakConvectiveCurrent(
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  double peak = 0.0;
  for (const auto& h : r.history) {
    peak = std::max(peak, std::abs(h.convectiveCurrent));
  }
  return peak;
}

int tailDevelopedSamplesAtMidplane(
    const electrospray::CandidoConeJetSmokeReport3D& r,
    double minAreaDi2) {
  return midplaneStats(r, minAreaDi2).developedSamples;
}

void writeLongWindowMassBudgetRow(
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

void writeLongWindowChargeBudgetRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  const bool finiteBudget =
      std::isfinite(r.initialIntegratedCharge) &&
      std::isfinite(r.finalIntegratedCharge) &&
      std::isfinite(r.cumulativeBoundaryChargeFlux) &&
      std::isfinite(r.chargeBudgetResidual) &&
      std::isfinite(r.cumulativeChargeClampCorrectionL1);
  const std::string status =
      !finiteBudget ? "DOWNGRADED_NONFINITE_CHARGE_BUDGET"
                    : (r.maxChargeClampedCells > 0 ||
                       r.cumulativeChargeClampCorrectionL1 > 0.0)
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

void writeBoundaryCurrentSensitivityRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high) {
  auto lateralCumulative =
      [](const electrospray::CandidoConeJetSmokeReport3D& r) {
        return r.cumulativeConductiveBoundaryChargeFluxByPatch[0] +
               r.cumulativeConductiveBoundaryChargeFluxByPatch[1] +
               r.cumulativeConductiveBoundaryChargeFluxByPatch[4] +
               r.cumulativeConductiveBoundaryChargeFluxByPatch[5];
      };
  auto lateralPeak =
      [](const electrospray::CandidoConeJetSmokeReport3D& r) {
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
  const double lowCollector =
      low.cumulativeConductiveBoundaryChargeFluxByPatch[3];
  const double highCollector =
      high.cumulativeConductiveBoundaryChargeFluxByPatch[3];
  const double lowLateral = lateralCumulative(low);
  const double highLateral = lateralCumulative(high);
  const double lowTotal = low.cumulativeConductiveBoundaryChargeFlux;
  const double highTotal = high.cumulativeConductiveBoundaryChargeFlux;
  std::string dominant = "nozzle";
  double dominantAbs = std::abs(highNozzle);
  if (std::abs(highCollector) > dominantAbs) {
    dominant = "collector";
    dominantAbs = std::abs(highCollector);
  }
  if (std::abs(highLateral) > dominantAbs) dominant = "lateral";
  const double totalRatio = absRatio(highTotal, lowTotal);
  const double nozzleRatio = absRatio(highNozzle, lowNozzle);
  const double collectorRatio = absRatio(highCollector, lowCollector);
  const double lateralRatio = absRatio(highLateral, lowLateral);
  std::string status = "PATCH_CURRENT_DIAGNOSTIC_ONLY";
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
      << fraction(lowNozzle, lowTotal) << ","
      << fraction(highNozzle, highTotal) << ","
      << fraction(lowCollector, lowTotal) << ","
      << fraction(highCollector, highTotal) << ","
      << fraction(lowLateral, lowTotal) << ","
      << fraction(highLateral, highTotal) << ","
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

void writePaperCurrentParetoTradeoffRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    double minAreaDi2) {
  const AxialStats lowStats = axialStats(low, minAreaDi2);
  const AxialStats highStats = axialStats(high, minAreaDi2);
  const double tailRatio =
      ratio(meanTailConvectiveCurrent(high), meanTailConvectiveCurrent(low));
  const double peakRatio =
      ratio(peakConvectiveCurrent(high), peakConvectiveCurrent(low));
  const bool axialComparable =
      lowStats.developedSamples > 0 && highStats.developedSamples > 0;
  const double axialConvectiveRatio =
      axialComparable ? ratio(highStats.meanAlpha05Convective,
                              lowStats.meanAlpha05Convective)
                      : std::numeric_limits<double>::infinity();
  const double axialTotalRatio =
      axialComparable ? ratio(highStats.meanAlpha05Total,
                              lowStats.meanAlpha05Total)
                      : std::numeric_limits<double>::infinity();
  const double chargeRatio =
      axialComparable ? ratio(highStats.meanAbsCharge, lowStats.meanAbsCharge)
                      : std::numeric_limits<double>::infinity();
  const double velocityRatio =
      axialComparable ? ratio(highStats.meanAbsUy, lowStats.meanAbsUy)
                      : std::numeric_limits<double>::infinity();
  const int lowMidplaneSamples =
      tailDevelopedSamplesAtMidplane(low, minAreaDi2);
  const int highMidplaneSamples =
      tailDevelopedSamplesAtMidplane(high, minAreaDi2);
  const double lowMorphologyError = maxMorphologyError04_07(low);
  const double highMaxAsymmetry = maxRadialAsymmetry(high);
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
      << high.alphaMassDrift << "," << low.maxDiv << "," << high.maxDiv
      << "," << lowMorphologyError << "," << highMaxAsymmetry << ","
      << tailRatio << "," << peakRatio << "," << lowMidplaneSamples << ","
      << highMidplaneSamples << "," << lowStats.developedSamples << ","
      << highStats.developedSamples << "," << lowStats.meanArea << ","
      << highStats.meanArea << "," << lowStats.meanAlpha05Convective << ","
      << highStats.meanAlpha05Convective << "," << axialConvectiveRatio
      << "," << lowStats.meanAlpha05Total << ","
      << highStats.meanAlpha05Total << "," << axialTotalRatio << ","
      << lowStats.meanAbsCharge << "," << highStats.meanAbsCharge << ","
      << chargeRatio << "," << lowStats.meanAbsUy << ","
      << highStats.meanAbsUy << "," << velocityRatio << "," << status
      << "\n";
}

void writeCurrentVoltageSensitivityRow(
    std::ofstream& csv,
    const electrospray::CandidoConeJetSmokeReport3D& low,
    const electrospray::CandidoConeJetSmokeReport3D& high,
    const std::string& source) {
  struct Stats {
    double peak = 0.0;
    double meanAll = 0.0;
    double meanTail = 0.0;
    int allCount = 0;
    int tailCount = 0;
  };
  auto stats = [](const electrospray::CandidoConeJetSmokeReport3D& r) {
    Stats s;
    const size_t tailStart = r.history.size() / 2;
    for (size_t i = 0; i < r.history.size(); ++i) {
      const double current = std::abs(r.history[i].convectiveCurrent);
      s.peak = std::max(s.peak, current);
      s.meanAll += current;
      ++s.allCount;
      if (i >= tailStart) {
        s.meanTail += current;
        ++s.tailCount;
      }
    }
    s.meanAll /= std::max(s.allCount, 1);
    s.meanTail /= std::max(s.tailCount, 1);
    return s;
  };
  const Stats lowStats = stats(low);
  const Stats highStats = stats(high);
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
}

void writeLateMorphologyBlockerRows(std::ofstream& csv) {
  csv << "reference_time_ms,paper_reported_error_percent,status,"
         "visible_fig3b_times_ms,blocker,required_input\n";
  csv << "0.8,-0.395,BLOCKED_DIGITIZED_GEOMETRY_REQUIRED,0.0;0.4;0.7,"
      << "Candido text reports only relative error and public figure assets do "
         "not expose extractable experimental contour coordinates,"
      << "external digitized Fig3b contour or numerical/experimental morphology "
         "volume at 0.8 ms\n";
  csv << "0.9,-0.948,BLOCKED_DIGITIZED_GEOMETRY_REQUIRED,0.0;0.4;0.7,"
      << "Candido text reports only relative error and public figure assets do "
         "not expose extractable experimental contour coordinates,"
      << "external digitized Fig3b contour or numerical/experimental morphology "
         "volume at 0.9 ms\n";
}

void writeLateMorphologySourceAuditRows(std::ofstream& csv) {
  csv << "source_id,source_type,artifact,checked_evidence,visible_times_ms,"
         "reported_late_error_only_times_ms,has_0_8_contour,has_0_9_contour,"
         "status,required_input\n";
  csv << "candido_public_aip_fig3_image,public_article_figure,"
      << "https://aipp.silverchair-cdn.com/aipp/content_public/journal/pof/"
         "35/5/10.1063_5.0151109/3/052110_1_5.0151109.figures.online.f3.jpeg,"
      << "downloaded_asset_is_700x310_three_panel_fig3,0.0;0.4;0.7,"
      << "0.8;0.9,0,0,NO_LATE_CONTOURS_IN_PUBLIC_FIGURE,"
      << "independent 0.8 and 0.9 ms contour coordinates or morphology volumes\n";
  csv << "candido_local_pdf_fig3_render,local_pdf_render,"
      << "papers/library/pdf/2024_candido_dynamic-3d-ehd-instabilities-"
         "taylor-cone-jets.pdf,"
      << "page_7_fig3_render_matches_public_three_panel_asset,0.0;0.4;0.7,"
      << "0.8;0.9,0,0,NO_LATE_CONTOURS_IN_LOCAL_PDF_RENDER,"
      << "independent 0.8 and 0.9 ms contour coordinates or morphology volumes\n";
  csv << "candido_paper_text_error_row,paper_text,"
      << "papers/library/md/2024_candido_dynamic-3d-ehd-instabilities-"
         "taylor-cone-jets.md,"
      << "text_lists_relative_errors_for_0.0_0.4_0.7_0.8_0.9_without_late_"
         "contour_geometry,0.0;0.4;0.7,0.8;0.9,0,0,"
      << "ERROR_ONLY_NOT_DIGITIZATION,"
      << "do not infer reference volumes from reported errors\n";
  csv << "guo2018_source_paper_figures,source_experiment_paper,"
      << "papers/library/pdf/2018_guo_ehd_jet_printing_experiment.pdf,"
      << "figures_show_experiment_frames_but_not_candido_synced_0.8_0.9_"
         "fig3b_contours,not_applicable,not_applicable,0,0,"
      << "NO_CANDIDO_SYNCED_LATE_CONTOURS,"
      << "external contour digitization tied to Candido Fig3b synchronization\n";
  csv << "candido_data_availability_statement,paper_text,"
      << "papers/library/md/2024_candido_dynamic-3d-ehd-instabilities-"
         "taylor-cone-jets.md,"
      << "data_available_from_corresponding_author_on_reasonable_request,"
      << "not_applicable,0.8;0.9,0,0,"
      << "NO_PUBLIC_LATE_DATASET_INCLUDED,"
      << "corresponding-author data file or independent late contour digitization\n";
  csv << "candido_external_dataset_schema,local_input_contract,"
      << "docs/electrospray/candido_late_morphology_external_schema.csv,"
      << "requires_positive_external_0.8_0.9_ms_volumes_not_backsolved_from_"
         "reported_errors,not_applicable,0.8;0.9,0,0,"
      << "INPUT_SCHEMA_READY_BUT_DATA_MISSING,"
      << "populate docs/electrospray/candido_late_morphology_external_dataset.csv"
         " with independent 0.8 and 0.9 ms volume rows\n";
  csv << "candido_author_github_interisofoamehd,public_author_code_repository,"
      << "https://github.com/silviomcandido/interIsoFoamEHD,"
      << "repository_has_openfoam_solver_cases_and_images_but_no_postprocessed_"
         "candido_fig3b_late_contour_or_volume_csv,not_applicable,0.8;0.9,"
      << "0,0,NO_PUBLIC_LATE_REFERENCE_DATA_IN_AUTHOR_REPOSITORY,"
      << "independent 0.8 and 0.9 ms contour coordinates or morphology volumes\n";
}

void writePaperCurrentDevelopmentTradeoffRow(
    std::ofstream& csv,
    const std::string& caseName,
    const electrospray::CandidoConeJetSmokeReport3D& stableLow,
    const electrospray::CandidoConeJetSmokeReport3D& stableHigh,
    const electrospray::CandidoConeJetSmokeReport3D& extendedLow,
    const electrospray::CandidoConeJetSmokeReport3D& extendedHigh,
    double minAreaDi2) {
  const MidplaneStats stableLowStats = midplaneStats(stableLow, minAreaDi2);
  const MidplaneStats stableHighStats = midplaneStats(stableHigh, minAreaDi2);
  const MidplaneStats extendedLowStats = midplaneStats(extendedLow, minAreaDi2);
  const MidplaneStats extendedHighStats =
      midplaneStats(extendedHigh, minAreaDi2);
  const bool extendedComparable =
      extendedLowStats.developedSamples > 0 &&
      extendedHighStats.developedSamples > 0;
  const double extendedMeanRatio =
      extendedComparable ? ratio(extendedHighStats.meanDevelopedAlpha05Current,
                                 extendedLowStats.meanDevelopedAlpha05Current)
                         : -1.0;
  const double extendedPeakRatio =
      extendedComparable ? ratio(extendedHighStats.peakDevelopedAlpha05Current,
                                 extendedLowStats.peakDevelopedAlpha05Current)
                         : -1.0;
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
      << stableHigh.alphaMassDrift << "," << extendedLow.alphaMassDrift
      << "," << extendedHigh.alphaMassDrift << "," << stableLow.maxDiv
      << "," << stableHigh.maxDiv << "," << extendedLow.maxDiv << ","
      << extendedHigh.maxDiv << "," << stableLowStats.developedSamples
      << "," << stableHighStats.developedSamples << ","
      << extendedLowStats.developedSamples << ","
      << extendedHighStats.developedSamples << ","
      << stableLowStats.maxAlpha05AreaDi2 << ","
      << stableHighStats.maxAlpha05AreaDi2 << ","
      << extendedLowStats.maxAlpha05AreaDi2 << ","
      << extendedHighStats.maxAlpha05AreaDi2 << ","
      << stableLowStats.maxTipY << "," << stableHighStats.maxTipY << ","
      << extendedLowStats.maxTipY << "," << extendedHighStats.maxTipY
      << "," << extendedLowStats.firstDevelopedStep << ","
      << extendedHighStats.firstDevelopedStep << ","
      << extendedLowStats.firstDevelopedTimeMs << ","
      << extendedHighStats.firstDevelopedTimeMs << ","
      << extendedLowStats.meanDevelopedAlpha05Current << ","
      << extendedHighStats.meanDevelopedAlpha05Current << ","
      << extendedMeanRatio << ","
      << extendedLowStats.peakDevelopedAlpha05Current << ","
      << extendedHighStats.peakDevelopedAlpha05Current << ","
      << extendedPeakRatio << "," << status << "\n";
}

void writeMorphologyTipSyncDiagnosticRow(
    std::ofstream& csv,
    const std::string& name,
    const electrospray::CandidoConeJetSmokeReport3D& r) {
  check(!r.history.empty(), "Candido tip-sync history is non-empty");
  const auto* maxTip = &r.history.front();
  const auto* firstTipJump =
      static_cast<const electrospray::CandidoConeJetHistorySample3D*>(nullptr);
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
                   : -1.0;
  const double paperSyncOffsetMs = 0.4 - maxTipMs;
  auto nearestShiftedTime = [&](double paperTimeMs) {
    const double targetSimMs = paperTimeMs - paperSyncOffsetMs;
    const auto* best = &r.history.front();
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

  constexpr double reference04 = 1.2826510303495016;
  constexpr double reference07 = 1.2550259882802302;
  const auto* sync04 = nearestShiftedTime(0.4);
  const auto* sync07 = nearestShiftedTime(0.7);
  const double sync04Ms = sync04->time * r.hydrodynamicTimeScale * 1.0e3;
  const double sync07Ms = sync07->time * r.hydrodynamicTimeScale * 1.0e3;
  const double sync04Error =
      100.0 * (sync04->morphologyVolumeDi3 - reference04) /
      std::abs(reference04);
  const double sync07Error =
      100.0 * (sync07->morphologyVolumeDi3 - reference07) /
      std::abs(reference07);
  const double sync04AllLiquidRayAlpha05Error =
      100.0 * (sync04->allLiquidRayAlpha05SilhouetteVolumeDi3 - reference04) /
      std::abs(reference04);
  const double sync07AllLiquidRayAlpha05Error =
      100.0 * (sync07->allLiquidRayAlpha05SilhouetteVolumeDi3 - reference07) /
      std::abs(reference07);
  const double sync04ConnectedRayAlpha05Error =
      100.0 * (sync04->rayAlpha05SilhouetteVolumeDi3 - reference04) /
      std::abs(reference04);
  const double sync07ConnectedRayAlpha05Error =
      100.0 * (sync07->rayAlpha05SilhouetteVolumeDi3 - reference07) /
      std::abs(reference07);
  const bool enoughTipLevels = uniqueTipLevels.size() >= 5;
  const std::string status =
      !enoughTipLevels
          ? "DOWNGRADED_TIP_QUANTIZED_COARSE_GRID"
          : ((std::abs(sync04Error) <= 10.0 &&
              std::abs(sync07Error) <= 10.0)
                 ? "TIP_SYNC_MORPHOLOGY_WITHIN_10_PERCENT"
                 : "DOWNGRADED_TIP_SYNC_MORPHOLOGY_ERROR");
  const std::string alpha05Status =
      (std::abs(sync04AllLiquidRayAlpha05Error) <= 10.0 &&
       std::abs(sync07AllLiquidRayAlpha05Error) <= 10.0)
          ? "TIP_SYNC_ALL_LIQUID_ALPHA05_WITHIN_10_PERCENT"
          : "DOWNGRADED_TIP_SYNC_ALPHA05_INTERFACE_LOST_OR_MISMATCHED";

  csv << name << "," << r.history.size() << "," << uniqueTipLevels.size()
      << "," << r.history.front().tipY << "," << maxTip->tipY << ","
      << minNonzeroTipStep << "," << maxTipMs << "," << firstJumpMs << ","
      << paperSyncOffsetMs << "," << sync04Ms << ","
      << sync04->morphologyVolumeDi3 << "," << sync04Error << ","
      << sync04->rayAlpha05SilhouetteVolumeDi3 << ","
      << sync04ConnectedRayAlpha05Error << ","
      << sync04->allLiquidRayAlpha05SilhouetteVolumeDi3 << ","
      << sync04AllLiquidRayAlpha05Error << "," << sync07Ms << ","
      << sync07->morphologyVolumeDi3 << "," << sync07Error << ","
      << sync07->rayAlpha05SilhouetteVolumeDi3 << ","
      << sync07ConnectedRayAlpha05Error << ","
      << sync07->allLiquidRayAlpha05SilhouetteVolumeDi3 << ","
      << sync07AllLiquidRayAlpha05Error << ","
      << "Candido_text_sets_0.4ms_to_maximum_cone_length_without_jet_emission,"
      << alpha05Status << "," << status << "\n";
}

electrospray::CandidoConeJetSmokeOptions3D paperCurrentBaseOptions() {
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
  return opt;
}

void checkFiniteReport(const electrospray::CandidoConeJetSmokeReport3D& r,
                       const std::string& name) {
  check(!r.history.empty(), name + " history is non-empty");
  check(r.cells > 0 && r.faces > 0, name + " mesh is non-empty");
  check(std::isfinite(r.alphaMassDrift) && r.alphaMassDrift <= 1e-3,
        name + " mass drift bounded");
  check(std::isfinite(r.maxDiv) && r.maxDiv <= 1e-7,
        name + " continuity bounded");
  check(std::isfinite(r.maxConvectiveCurrent),
        name + " convective current finite");
}

}  // namespace

int main() {
  std::filesystem::create_directories("benchmark_logs");

  electrospray::CandidoTaylorConeJetSetup setup;

  std::ofstream massBudget("benchmark_logs/candido_long_window_mass_budget3d.csv");
  massBudget << "case,target_ca_e,steps,initial_mass,final_mass,"
                "cumulative_boundary_liquid_flux,"
                "cumulative_boundary_liquid_inflow,"
                "cumulative_boundary_liquid_outflow,"
                "mass_budget_expected_final,mass_budget_residual,"
                "relative_mass_budget_residual,open_domain_growth,"
                "signed_boundary_growth,alpha_mass_drift,max_convective_current,"
                "max_velocity,status\n";
  std::ofstream chargeBudget(
      "benchmark_logs/candido_long_window_charge_budget3d.csv");
  chargeBudget << "case,target_ca_e,steps,initial_integrated_charge,"
                  "final_integrated_charge,cumulative_boundary_charge_flux,"
                  "cumulative_conductive_boundary_charge_flux,"
                  "cumulative_charge_relaxation_sink,"
                  "charge_budget_expected_final,charge_budget_residual,"
                  "relative_charge_budget_residual,"
                  "cumulative_charge_clamp_correction_l1,"
                  "max_charge_redistribution_residual,max_charge_clamped_cells,"
                  "max_unclamped_abs_charge,max_charge,min_charge,"
                  "max_conductive_current,max_convective_current,max_velocity,"
                  "status\n";
  std::ofstream currentPareto(
      "benchmark_logs/candido_current_morphology_whip_pareto3d.csv");
  currentPareto
      << "case,low_ca_e,high_ca_e,low_steps,high_steps,min_alpha05_area_di2,"
         "fixed_midplane_y_over_Di,low_alpha_mass_drift,high_alpha_mass_drift,"
         "low_max_div,high_max_div,"
         "low_max_morphology_error_0_4_0_7_percent,high_max_radial_asymmetry,"
         "all_phase_tail_current_ratio,all_phase_peak_current_ratio,"
         "low_fixed_midplane_developed_samples,"
         "high_fixed_midplane_developed_samples,low_axial_developed_samples,"
         "high_axial_developed_samples,low_axial_mean_area_di2,"
         "high_axial_mean_area_di2,low_axial_alpha05_convective_current,"
         "high_axial_alpha05_convective_current,axial_alpha05_convective_ratio,"
         "low_axial_alpha05_total_current,high_axial_alpha05_total_current,"
         "axial_alpha05_total_ratio,low_axial_mean_abs_charge,"
         "high_axial_mean_abs_charge,charge_ratio,low_axial_mean_abs_uy,"
         "high_axial_mean_abs_uy,velocity_ratio,status\n";
  std::ofstream development(
      "benchmark_logs/candido_paper_current_development_tradeoff3d.csv");
  development
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
  std::ofstream boundaryCurrent(
      "benchmark_logs/candido_boundary_current_sensitivity3d.csv");
  boundaryCurrent
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
  std::ofstream defaultCurrent(
      "benchmark_logs/candido_current_voltage_sensitivity3d.csv");
  defaultCurrent
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream combinedCurrent(
      "benchmark_logs/candido_current_voltage_sensitivity_combined_charge3d.csv");
  combinedCurrent
      << "low_ca_e,high_ca_e,low_peak_convective_current,"
         "high_peak_convective_current,peak_current_ratio,"
         "low_mean_all_convective_current,high_mean_all_convective_current,"
         "low_mean_tail_convective_current,high_mean_tail_convective_current,"
         "tail_mean_current_ratio,external_source,status\n";
  std::ofstream tipSync(
      "benchmark_logs/candido_morphology_tip_sync_diagnostic3d.csv");
  tipSync << "case,history_samples,unique_tip_levels,initial_tip_y,max_tip_y,"
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
  std::ofstream lateBlocker(
      "benchmark_logs/candido_late_morphology_blocker3d.csv");
  writeLateMorphologyBlockerRows(lateBlocker);
  std::ofstream lateSourceAudit(
      "benchmark_logs/candido_late_morphology_source_audit3d.csv");
  writeLateMorphologySourceAuditRows(lateSourceAudit);

  electrospray::CandidoConeJetSmokeOptions3D longOpt;
  longOpt.nx = 12;
  longOpt.nz = 12;
  longOpt.ny = 17;
  longOpt.steps = 52;
  longOpt.cfl = 1.0;
  longOpt.skew = 0.04;

  const auto longWindow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, longOpt);
  const auto longWhip =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, longOpt);
  checkFiniteReport(longWindow, "long_window_ca025");
  checkFiniteReport(longWhip, "long_window_ca042");
  writeLongWindowMassBudgetRow(massBudget, "long_window_ca025", longWindow);
  writeLongWindowMassBudgetRow(massBudget, "long_window_ca042", longWhip);
  writeLongWindowChargeBudgetRow(chargeBudget, "long_window_ca025", longWindow);
  writeLongWindowChargeBudgetRow(chargeBudget, "long_window_ca042", longWhip);
  writeMorphologyTipSyncDiagnosticRow(tipSync, "long_window_ca025",
                                      longWindow);
  writeBoundaryCurrentSensitivityRow(boundaryCurrent, "long_window",
                                     longWindow, longWhip);
  writeCurrentVoltageSensitivityRow(
      defaultCurrent, longWindow, longWhip,
      "Candido_Fig8b_text_average_current_not_influenced_by_voltage;"
      "standalone_long_window_primary_convective_current");
  writePaperCurrentParetoTradeoffRow(currentPareto, "baseline_long_window",
                                     setup, longWindow, longWhip,
                                     kMinAreaDi2);

  electrospray::CandidoConeJetSmokeOptions3D caIndependentBoundaryOpt =
      paperCurrentBaseOptions();
  const auto caIndependentBoundaryLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             caIndependentBoundaryOpt);
  const auto caIndependentBoundaryHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             caIndependentBoundaryOpt);
  checkFiniteReport(caIndependentBoundaryLow,
                    "ca_independent_drive_boundary_advected_ca025");
  checkFiniteReport(caIndependentBoundaryHigh,
                    "ca_independent_drive_boundary_advected_ca042");
  writeBoundaryCurrentSensitivityRow(
      boundaryCurrent, "ca_independent_drive_boundary_advected",
      caIndependentBoundaryLow, caIndependentBoundaryHigh);
  writeCurrentVoltageSensitivityRow(
      defaultCurrent, caIndependentBoundaryLow, caIndependentBoundaryHigh,
      "Candido_Fig8b_text_average_current_not_influenced_by_voltage;"
      "opt_in_short_ca_independent_drive_boundary_charge_advection_candidate");

  electrospray::CandidoConeJetSmokeOptions3D caIndependentBoundaryLongOpt =
      caIndependentBoundaryOpt;
  caIndependentBoundaryLongOpt.nx = longOpt.nx;
  caIndependentBoundaryLongOpt.nz = longOpt.nz;
  caIndependentBoundaryLongOpt.ny = longOpt.ny;
  caIndependentBoundaryLongOpt.steps = longOpt.steps;
  caIndependentBoundaryLongOpt.cfl = longOpt.cfl;
  caIndependentBoundaryLongOpt.skew = longOpt.skew;
  const auto caIndependentBoundaryLongLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             caIndependentBoundaryLongOpt);
  const auto caIndependentBoundaryLongHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             caIndependentBoundaryLongOpt);
  checkFiniteReport(caIndependentBoundaryLongLow,
                    "long_ca_independent_boundary_ca025");
  checkFiniteReport(caIndependentBoundaryLongHigh,
                    "long_ca_independent_boundary_ca042");
  writeCurrentVoltageSensitivityRow(
      defaultCurrent, caIndependentBoundaryLongLow,
      caIndependentBoundaryLongHigh,
      "Candido_Fig8b_text_average_current_not_influenced_by_voltage;"
      "opt_in_long_ca_independent_drive_boundary_charge_advection_candidate");

  electrospray::CandidoConeJetSmokeOptions3D combinedChargeOpt =
      caIndependentBoundaryOpt;
  combinedChargeOpt.chargeSubcycles = 8;
  combinedChargeOpt.conservativeChargeBounding = true;
  const auto combinedLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, combinedChargeOpt);
  const auto combinedHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, combinedChargeOpt);
  checkFiniteReport(combinedLow, "combined_charge_bounding_subcycled_ca025");
  checkFiniteReport(combinedHigh, "combined_charge_bounding_subcycled_ca042");
  writeBoundaryCurrentSensitivityRow(
      boundaryCurrent, "combined_charge_bounding_subcycled", combinedLow,
      combinedHigh);
  writeCurrentVoltageSensitivityRow(
      combinedCurrent, combinedLow, combinedHigh,
      "Candido_Fig8b_text_average_current_not_influenced_by_voltage;"
      "combined_charge_bounding_subcycled_primary_convective_current");

  electrospray::CandidoConeJetSmokeOptions3D combinedLongOpt =
      caIndependentBoundaryLongOpt;
  combinedLongOpt.chargeSubcycles = combinedChargeOpt.chargeSubcycles;
  combinedLongOpt.conservativeChargeBounding =
      combinedChargeOpt.conservativeChargeBounding;
  const auto combinedLongLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, combinedLongOpt);
  const auto combinedLongHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, combinedLongOpt);
  checkFiniteReport(combinedLongLow, "long_combined_charge_ca025");
  checkFiniteReport(combinedLongHigh, "long_combined_charge_ca042");
  writeCurrentVoltageSensitivityRow(
      combinedCurrent, combinedLongLow, combinedLongHigh,
      "Candido_Fig8b_text_average_current_not_influenced_by_voltage;"
      "long_combined_charge_bounding_subcycled_convective_current");

  electrospray::CandidoConeJetSmokeOptions3D paperBoundaryCombinedLongOpt =
      combinedLongOpt;
  paperBoundaryCombinedLongOpt.useVofInletBoundaryAlpha = true;
  paperBoundaryCombinedLongOpt.suppressNozzleConductiveChargeFlux = true;
  const auto paperBoundaryCombinedLongLow =
      electrospray::runCandidoConeJetSmoke3D(
          0.25, setup, paperBoundaryCombinedLongOpt);
  const auto paperBoundaryCombinedLongHigh =
      electrospray::runCandidoConeJetSmoke3D(
          0.42, setup, paperBoundaryCombinedLongOpt);
  checkFiniteReport(paperBoundaryCombinedLongLow,
                    "long_paper_boundary_combined_charge_ca025");
  checkFiniteReport(paperBoundaryCombinedLongHigh,
                    "long_paper_boundary_combined_charge_ca042");
  writeCurrentVoltageSensitivityRow(
      defaultCurrent, paperBoundaryCombinedLongLow,
      paperBoundaryCombinedLongHigh,
      "Candido_Fig8b_text_average_current_not_influenced_by_voltage;"
      "long_paper_boundary_combined_charge_candidate");

  electrospray::CandidoConeJetSmokeOptions3D paperChargeBoundaryOpt =
      caIndependentBoundaryOpt;
  paperChargeBoundaryOpt.useVofInletBoundaryAlpha = true;
  paperChargeBoundaryOpt.suppressNozzleConductiveChargeFlux = true;
  const auto paperChargeBoundaryLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperChargeBoundaryOpt);
  const auto paperChargeBoundaryHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperChargeBoundaryOpt);
  checkFiniteReport(paperChargeBoundaryLow, "paper_charge_boundary_ca025");
  checkFiniteReport(paperChargeBoundaryHigh, "paper_charge_boundary_ca042");
  writeBoundaryCurrentSensitivityRow(boundaryCurrent, "paper_charge_boundary",
                                     paperChargeBoundaryLow,
                                     paperChargeBoundaryHigh);
  writePaperCurrentParetoTradeoffRow(currentPareto, "paper_charge_boundary",
                                     setup, paperChargeBoundaryLow,
                                     paperChargeBoundaryHigh, kMinAreaDi2);

  electrospray::CandidoConeJetSmokeOptions3D paperInletVelocityOpt =
      paperChargeBoundaryOpt;
  paperInletVelocityOpt.useFullyDevelopedInletVelocityBoundary = true;
  const auto paperInletLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperInletVelocityOpt);
  const auto paperInletHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperInletVelocityOpt);
  checkFiniteReport(paperInletLow, "paper_inlet_velocity_ca025");
  checkFiniteReport(paperInletHigh, "paper_inlet_velocity_ca042");
  writePaperCurrentParetoTradeoffRow(currentPareto, "paper_inlet_velocity",
                                     setup, paperInletLow, paperInletHigh,
                                     kMinAreaDi2);

  electrospray::CandidoConeJetSmokeOptions3D paperOpenBoundaryOpt =
      paperInletVelocityOpt;
  paperOpenBoundaryOpt.useOpenAtmosphericBoundaryFlux = true;
  const auto paperOpenLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperOpenBoundaryOpt);
  const auto paperOpenHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperOpenBoundaryOpt);
  checkFiniteReport(paperOpenLow, "paper_inlet_velocity_open_ca025");
  checkFiniteReport(paperOpenHigh, "paper_inlet_velocity_open_ca042");

  electrospray::CandidoConeJetSmokeOptions3D movingCollectorOpt =
      paperOpenBoundaryOpt;
  movingCollectorOpt.useMovingCollectorWall = true;
  const auto movingCollectorLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup, movingCollectorOpt);
  const auto movingCollectorHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup, movingCollectorOpt);
  checkFiniteReport(movingCollectorLow, "moving_collector_ca025");
  checkFiniteReport(movingCollectorHigh, "moving_collector_ca042");
  writeBoundaryCurrentSensitivityRow(
      boundaryCurrent, "paper_inlet_velocity_open_atmosphere_moving_collector",
      movingCollectorLow, movingCollectorHigh);

  electrospray::CandidoConeJetSmokeOptions3D unitMaxwellBoundaryOpt =
      caIndependentBoundaryOpt;
  unitMaxwellBoundaryOpt.electricDriveReferenceScale = 1.0;
  const auto unitMaxwellLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             unitMaxwellBoundaryOpt);
  const auto unitMaxwellHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             unitMaxwellBoundaryOpt);
  checkFiniteReport(unitMaxwellLow, "unit_maxwell_ca025");
  checkFiniteReport(unitMaxwellHigh, "unit_maxwell_ca042");
  writeBoundaryCurrentSensitivityRow(boundaryCurrent,
                                     "unit_maxwell_drive_boundary_advected",
                                     unitMaxwellLow, unitMaxwellHigh);

  electrospray::CandidoConeJetSmokeOptions3D paperOpenExtendedOpt =
      paperOpenBoundaryOpt;
  paperOpenExtendedOpt.steps = 90;
  const auto paperOpenExtendedLow =
      electrospray::runCandidoConeJetSmoke3D(0.25, setup,
                                             paperOpenExtendedOpt);
  const auto paperOpenExtendedHigh =
      electrospray::runCandidoConeJetSmoke3D(0.42, setup,
                                             paperOpenExtendedOpt);
  checkFiniteReport(paperOpenExtendedLow, "paper_open_extended90_ca025");
  checkFiniteReport(paperOpenExtendedHigh, "paper_open_extended90_ca042");
  writePaperCurrentDevelopmentTradeoffRow(
      development, "paper_inlet_velocity_open_atmosphere_extended90",
      paperOpenLow, paperOpenHigh, paperOpenExtendedLow,
      paperOpenExtendedHigh, kMinAreaDi2);

  std::cout << "candido_long_window_budget3d "
            << "long_mass_residuals="
            << longWindow.relativeMassBudgetResidual << "/"
            << longWhip.relativeMassBudgetResidual
            << " charge_residuals="
            << longWindow.relativeChargeBudgetResidual << "/"
            << longWhip.relativeChargeBudgetResidual
            << " pareto_tail_ratio="
            << ratio(meanTailConvectiveCurrent(longWhip),
                     meanTailConvectiveCurrent(longWindow))
            << " extended_steps=" << paperOpenExtendedLow.steps << "/"
            << paperOpenExtendedHigh.steps << "\n";
  return 0;
}
