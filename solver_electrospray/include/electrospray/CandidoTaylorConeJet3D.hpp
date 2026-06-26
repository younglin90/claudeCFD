#pragma once

#include "fvm/EHDCoupling3D.hpp"
#include "fvm/SurfaceTension3D.hpp"
#include "electrospray/MaterialProperties.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace electrospray {

struct CandidoTaylorConeJetSetup {
  double innerDiameter = 160e-6;
  double outerDiameter = 260e-6;
  double nozzleLength = 300e-6;
  double collectorDistance = 1.5e-3;
  double collectorDiameter = 5.0e-3;
  double validationVoltage = 2180.0;
  double validationFlowRate = 16.1e-12;
  double collectorSpeed = 20.0e-3;
  double contactAngleDeg = 51.0;
  double liquidDensity = 1208.4;
  double gasDensity = 1.225;
  double liquidViscosity = 60.0e-3;
  double gasViscosity = 0.0120e-3;
  double liquidRelativePermittivity = 55.6;
  double gasRelativePermittivity = 1.0;
  double liquidConductivity = 60.0e-6;
  double gasConductivity = 1.0e-15;
  double surfaceTension = 64.5e-3;
};

struct CandidoConeJetSmokeOptions3D {
  int nx = 12;
  int ny = 18;
  int nz = 12;
  int steps = 4;
  double radialWindowOuterDiameters = 6.0;
  double skew = 0.06;
  double cfl = 0.1;
  double pseudoViscosity = 0.03;
  double vofCompression = 0.06;
  double vofPostSharpening = 0.0;
  int vofPostSharpeningSweeps = 0;
  bool useVofInletBoundaryAlpha = false;
  bool useOpenAtmosphericBoundaryFlux = false;
  double alphaInterfaceWidthOuterDiameters = 0.22;
  double normalizedLiquidConductivity = 1.0;
  double normalizedGasConductivity = 1e-6;
  bool useDimensionalElectricalScaling = false;
  double chargeLimitBase = 50.0;
  int chargeSubcycles = 1;
  bool conservativeChargeBounding = false;
  bool quasiImplicitChargeRelaxation = false;
  bool quasiImplicitBulkConduction = false;
  bool useRayleighChargeLimit = false;
  bool useInterfaceLocalizedChargeRedistribution = false;
  double interfaceChargeRedistributionLiquidFloor = 0.02;
  bool useInterfacialOhmicChargeSource = false;
  double interfacialOhmicChargeSourceScale = 1.0;
  bool useConductivityPotentialChargeClosure = false;
  bool suppressNozzleConductiveChargeFlux = false;
  bool collectorOnlyConductiveChargeFlux = false;
  bool applyConductiveBoundaryFiltersInImplicitOhmic = false;
  bool usePoissonFaceConductiveCurrent = false;
  bool implicitOhmicChargeProjection = false;
  bool refreshPotentialAfterChargeAdvance = false;
  bool usePoissonFaceMaxwellForce = false;
  bool usePoissonHybridMaxwellForce = false;
  bool usePoissonBoundedVectorMaxwellForce = false;
  bool useTomarConductingSurfaceForce = false;
  bool useElectricRelaxationTimeStepLimit = false;
  double electricRelaxationTimeStepSafety = 1.0;
  bool useBoundaryChargeAdvection = false;
  bool useFullyDevelopedInletVelocityBoundary = false;
  bool useMovingCollectorWall = false;
  bool usePreconditionedPaperCurrentJet = false;
  double preconditionedJetTipYOverInnerDiameter = -1.0;
  double preconditionedJetRadiusInnerDiameters = 0.65;
  double preconditionedJetInterfaceWidthInnerDiameters = 0.20;
  double preconditionedJetVelocityScale = 1.0;
  bool useContactAngleCurvature = false;
  double contactAngleCurvatureWallBandCells = 1.5;
  double electricDriveReferenceScale = 15.2 * 0.25;
  double electricDriveCaExponent = 1.25;
  double poissonTangentialLimitFactor = 1.0;
  double poissonTangentialLimitFloorFraction = 0.05;
  double surfaceTensionDriveScale = 0.20;
};

struct CandidoConeJetHistorySample3D {
  int step = 0;
  double time = 0.0;
  double mass = 0.0;
  double minAlpha = 0.0;
  double maxAlpha = 0.0;
  double tipY = 0.0;
  double centroidY = 0.0;
  double radialAsymmetry = 0.0;
  double maxDiv = 0.0;
  double potentialResidual = 0.0;
  double electricForce = 0.0;
  double csfForce = 0.0;
  double curvature = 0.0;
  double conductiveCurrent = 0.0;
  double convectiveCurrent = 0.0;
  double liquidConvectiveCurrent = 0.0;
  double alpha05ConvectiveCurrent = 0.0;
  double midplaneLiquidAreaDi2 = 0.0;
  double midplaneAlpha05AreaDi2 = 0.0;
  double developedJetYOverDi = 0.0;
  double developedJetAlpha05AreaDi2 = 0.0;
  double developedJetConvectiveCurrent = 0.0;
  double developedJetLiquidConvectiveCurrent = 0.0;
  double developedJetAlpha05ConvectiveCurrent = 0.0;
  double developedJetTotalCurrent = 0.0;
  double developedJetLiquidTotalCurrent = 0.0;
  double developedJetAlpha05TotalCurrent = 0.0;
  double developedJetAlpha05ConductiveCurrent = 0.0;
  double developedJetMeanAlpha05Charge = 0.0;
  double developedJetMeanAlpha05AbsCharge = 0.0;
  double developedJetMeanAlpha05Uy = 0.0;
  double developedJetMeanAlpha05AbsUy = 0.0;
  double developedJetMeanAlpha05AbsElectricMomentumSourceY = 0.0;
  double developedJetMeanAlpha05AbsSurfaceMomentumSourceY = 0.0;
  double developedJetMeanAlpha05AbsMomentumSourceY = 0.0;
  double developedJetMeanAlpha05AbsMomentumAccelerationY = 0.0;
  double developedJetAlpha05CurrentShapeFactor = 0.0;
  double totalCurrent = 0.0;
  double maxVelocity = 0.0;
  double waveYOverDi = 0.0;
  double waveAsymmetry = 0.0;
  double morphologyVolumeDi3 = 0.0;
  double connectedMorphologyVolumeDi3 = 0.0;
  double alpha05SilhouetteVolumeDi3 = 0.0;
  double rayAlpha05SilhouetteVolumeDi3 = 0.0;
  double allLiquidRayAlpha05SilhouetteVolumeDi3 = 0.0;
  double rayAlpha05CellBoundarySilhouetteVolumeDi3 = 0.0;
  double linearRayAlpha05SilhouetteVolumeDi3 = 0.0;
  double plicContourSilhouetteVolumeDi3 = 0.0;
  double plicPolygonSilhouetteVolumeDi3 = 0.0;
  double plicSectorMedianSilhouetteVolumeDi3 = 0.0;
  double plicRayPlaneSilhouetteVolumeDi3 = 0.0;
  double plicRayPlaneQ25SilhouetteVolumeDi3 = 0.0;
  double plicFirstExitSilhouetteVolumeDi3 = 0.0;
  double poissonFaceConvectiveCurrent = 0.0;
  double poissonFaceConductiveCurrent = 0.0;
  double poissonFaceTotalCurrent = 0.0;
  double poissonFaceAlpha05ConvectiveCurrent = 0.0;
  double poissonFaceAlpha05ConductiveCurrent = 0.0;
  double poissonFaceAlpha05TotalCurrent = 0.0;
  double poissonFaceDevelopedYOverDi = 0.0;
  double poissonFaceDevelopedAlpha05AreaDi2 = 0.0;
  double poissonFaceDevelopedAlpha05ConvectiveCurrent = 0.0;
  double poissonFaceDevelopedAlpha05ConductiveCurrent = 0.0;
  double poissonFaceDevelopedAlpha05TotalCurrent = 0.0;
  double poissonFaceDevelopedAlpha05MeanAbsUpwindCharge = 0.0;
  double poissonFaceDevelopedAlpha05MeanAbsFaceFlux = 0.0;
  double poissonFaceDevelopedAlpha05MeanAbsConvectiveFlux = 0.0;
  double poissonFaceDevelopedAlpha05MaxAbsUpwindCharge = 0.0;
  double poissonFaceDevelopedAlpha05MaxAbsFaceFlux = 0.0;
  double rawVelocityFaceDevelopedAlpha05ConvectiveCurrent = 0.0;
  double rawVelocityFaceDevelopedAlpha05MeanAbsUpwindCharge = 0.0;
  double rawVelocityFaceDevelopedAlpha05MeanAbsFaceFlux = 0.0;
  double rawVelocityFaceDevelopedAlpha05MeanAbsConvectiveFlux = 0.0;
  double rawVelocityFaceDevelopedAlpha05MaxAbsUpwindCharge = 0.0;
  double rawVelocityFaceDevelopedAlpha05MaxAbsFaceFlux = 0.0;
};

struct CandidoFaceElectricReconstructionDiagnostic3D {
  int sampledFaces = 0;
  int internalFaces = 0;
  int dirichletBoundaryFaces = 0;
  int tractionRatioFaces = 0;
  int normalTractionDegenerateFaces = 0;
  double maxPoissonNormalE = 0.0;
  double maxCellTangentialE = 0.0;
  double meanRelativeNormalMismatch = 0.0;
  double maxRelativeNormalMismatch = 0.0;
  double meanTangentialFraction = 0.0;
  double maxTangentialFraction = 0.0;
  double meanHybridToNormalTractionRatio = 0.0;
  double p95HybridToNormalTractionRatio = 0.0;
  double maxHybridToNormalTractionRatio = 0.0;
  double potentialResidual = 0.0;
};

struct CandidoBoundedVectorMaxwellReport3D {
  fvm::EHDBodyForceReport3D force;
  int sampledFaces = 0;
  int tangentialClippedFaces = 0;
  double meanTangentialClipRatio = 1.0;
  double minTangentialClipRatio = 1.0;
  double maxRawTangentialE = 0.0;
  double maxLimitedTangentialE = 0.0;
  double maxPoissonNormalE = 0.0;
  double tangentialLimitFactor = 0.0;
  double tangentialLimitFloorFraction = 0.0;
  double potentialResidual = 0.0;
};

struct CandidoTomarConductingForceReport3D {
  fvm::EHDBodyForceReport3D force;
  int sampledCells = 0;
  int mixedCells = 0;
  int activeInterfaceCells = 0;
  double maxGradAlpha = 0.0;
  double maxNormalCurrent = 0.0;
  double maxTangentialE = 0.0;
  double maxNormalTerm = 0.0;
  double maxTangentialTerm = 0.0;
  double maxRawCellForce = 0.0;
  double defaultMaxForce = 0.0;
  double potentialResidual = 0.0;
};

struct CandidoConeJetSmokeReport3D {
  double targetCaE = 0.0;
  double voltage = 0.0;
  double computedCaE = 0.0;
  double electricWeber = 0.0;
  double hydrodynamicTimeScale = 0.0;
  double inletVelocity = 0.0;
  double dt = 0.0;
  double unrestrictedDt = 0.0;
  double electricRelaxationDtLimit = std::numeric_limits<double>::infinity();
  double dtOverElectricRelaxationLimit = 0.0;
  int electricRelaxationTimestepLimited = 0;
  int cells = 0;
  int faces = 0;
  int steps = 0;
  double initialMass = 0.0;
  double finalMass = 0.0;
  double alphaMassDrift = 0.0;
  double cumulativeBoundaryLiquidFlux = 0.0;
  double cumulativeBoundaryLiquidInflow = 0.0;
  double cumulativeBoundaryLiquidOutflow = 0.0;
  double massBudgetExpectedFinal = 0.0;
  double massBudgetResidual = 0.0;
  double relativeMassBudgetResidual = 0.0;
  double minAlpha = 0.0;
  double maxAlpha = 0.0;
  double initialTipY = 0.0;
  double finalTipY = 0.0;
  double tipDisplacement = 0.0;
  double finalCentroidY = 0.0;
  double maxDiv = 0.0;
  double maxPotentialResidual = 0.0;
  double maxPostChargePotentialResidual = 0.0;
  double maxPostChargeRelativeGaussLawResidual = 0.0;
  double maxConductivityPotentialResidual = 0.0;
  double cumulativeConductivityClosureClampCorrectionL1 = 0.0;
  double maxElectricForce = 0.0;
  double maxCsfForce = 0.0;
  double maxCurvature = 0.0;
  double curvatureFallbackFraction = 0.0;
  double maxCharge = 0.0;
  double minCharge = 0.0;
  double initialIntegratedCharge = 0.0;
  double finalIntegratedCharge = 0.0;
  double cumulativeBoundaryChargeFlux = 0.0;
  double cumulativeConductiveBoundaryChargeFlux = 0.0;
  std::array<double, 6> cumulativeConductiveBoundaryChargeFluxByPatch = {0.0, 0.0, 0.0,
                                                                         0.0, 0.0, 0.0};
  std::array<double, 6> maxAbsConductiveBoundaryCurrentByPatch = {0.0, 0.0, 0.0,
                                                                  0.0, 0.0, 0.0};
  double cumulativeChargeRelaxationSink = 0.0;
  double cumulativeInterfacialOhmicChargeSource = 0.0;
  double cumulativeInterfacialOhmicChargeClampL1 = 0.0;
  double maxInterfacialOhmicChargeSourceDensity = 0.0;
  int maxInterfacialOhmicChargeSourceCells = 0;
  double chargeBudgetExpectedFinal = 0.0;
  double chargeBudgetResidual = 0.0;
  double relativeChargeBudgetResidual = 0.0;
  double cumulativeChargeClampCorrectionL1 = 0.0;
  double maxChargeRedistributionResidual = 0.0;
  double cumulativeChargeRedistributionDeficitL1 = 0.0;
  double maxChargeRedistributionWeightedCapacity = 0.0;
  int maxChargeRedistributionWeightedCells = 0;
  int maxChargeClampedCells = 0;
  double maxUnclampedAbsCharge = 0.0;
  double maxGaussLawCellGradientResidual = 0.0;
  double maxRelativeGaussLawCellGradientResidual = 0.0;
  double maxImplicitOhmicChargeResidual = 0.0;
  double maxConductiveCurrent = 0.0;
  double maxConvectiveCurrent = 0.0;
  double finalRadialAsymmetry = 0.0;
  double finalMidplaneJetRadius = 0.0;
  double maxVelocity = 0.0;
  std::vector<CandidoConeJetHistorySample3D> history;
};

inline double candidoVacuumPermittivity() {
  return 8.8541878128e-12;
}

inline double candidoElectricFieldScale(const CandidoTaylorConeJetSetup& s, double voltage) {
  return voltage / (s.outerDiameter * std::log(4.0 * s.collectorDistance / s.outerDiameter));
}

inline double candidoElectricCapillaryNumber(const CandidoTaylorConeJetSetup& s,
                                             double voltage) {
  const double e0 = candidoElectricFieldScale(s, voltage);
  return candidoVacuumPermittivity() * s.outerDiameter * e0 * e0 / s.surfaceTension;
}

inline double candidoVoltageForElectricCapillary(const CandidoTaylorConeJetSetup& s,
                                                 double caE) {
  const double denom = candidoVacuumPermittivity() * s.outerDiameter;
  const double e0 = std::sqrt(caE * s.surfaceTension / std::max(denom, 1e-300));
  return e0 * s.outerDiameter * std::log(4.0 * s.collectorDistance / s.outerDiameter);
}

inline double candidoInletVelocity(const CandidoTaylorConeJetSetup& s) {
  const double area = M_PI * s.innerDiameter * s.innerDiameter / 4.0;
  return s.validationFlowRate / area;
}

inline double candidoHydrodynamicTimeScale(const CandidoTaylorConeJetSetup& s) {
  return std::sqrt(s.liquidDensity * std::pow(s.innerDiameter, 3.0) / s.surfaceTension);
}

inline double candidoDimensionlessInletVelocityScale(
    const CandidoTaylorConeJetSetup& s) {
  return candidoInletVelocity(s) * candidoHydrodynamicTimeScale(s) / s.outerDiameter;
}

inline double candidoDimensionlessCollectorVelocityScale(
    const CandidoTaylorConeJetSetup& s) {
  return s.collectorSpeed * candidoHydrodynamicTimeScale(s) / s.outerDiameter;
}

inline double candidoDimensionlessConductivityFromPhysical(
    const CandidoTaylorConeJetSetup& s,
    double physicalConductivity) {
  return physicalConductivity * candidoHydrodynamicTimeScale(s) /
         std::max(candidoVacuumPermittivity(), 1e-300);
}

inline double candidoRayleighChargeScale(const CandidoTaylorConeJetSetup& s) {
  const double radius = 0.5 * s.innerDiameter;
  return std::sqrt(64.0 * M_PI * M_PI * candidoVacuumPermittivity() *
                   s.surfaceTension * radius * radius * radius);
}

inline double candidoPoissonChargeScale(const CandidoTaylorConeJetSetup& s,
                                        double voltage) {
  const double e0 = candidoElectricFieldScale(s, voltage);
  return candidoVacuumPermittivity() * e0 * s.outerDiameter * s.outerDiameter;
}

inline double candidoDimensionlessRayleighChargeLimit(
    const CandidoTaylorConeJetSetup& s,
    double voltage) {
  return candidoRayleighChargeScale(s) /
         std::max(candidoPoissonChargeScale(s, voltage), 1e-300);
}

inline double candidoValidationElectricWeber() {
  return 20.4;
}

inline double candidoHarmonicMixture(double alpha, double liquid, double gas) {
  alpha = std::clamp(alpha, 0.0, 1.0);
  return 1.0 / std::max(alpha / liquid + (1.0 - alpha) / gas, 1e-300);
}

inline double candidoSmoothIndicator(double signedDistance, double width) {
  return std::clamp(0.5 * (1.0 - std::tanh(signedDistance / std::max(width, 1e-30))),
                    0.0, 1.0);
}

inline fvm::ScalarField candidoInitialAlpha3D(const fvm::Mesh3D& mesh,
                                             const CandidoTaylorConeJetSetup& s,
                                             const CandidoConeJetSmokeOptions3D& opt,
                                             double caE) {
  const double d0 = s.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  const double ri = 0.5 * s.innerDiameter / d0;
  const double capRadius = 0.95 * 0.5;
  const double inletLength = std::min(s.nozzleLength / d0, 0.8);
  const double width = opt.alphaInterfaceWidthOuterDiameters;
  const fvm::Vec3 center{0.5 * lx, 0.0, 0.5 * lz};
  fvm::ScalarField alpha(mesh.cells.size(), 0.0);
  const double perturb = 0.002 + 0.006 * std::max(0.0, caE - 0.25) / 0.17;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const fvm::Vec3 x = mesh.cells[ci].centroid;
    const double dx = x.x() - center.x();
    const double dz = x.z() - center.z();
    const double r = std::sqrt(dx * dx + dz * dz);
    const double sphere = candidoSmoothIndicator((x - center).norm() - capRadius, width);
    const double cylRadial = candidoSmoothIndicator(r - ri, 0.6 * width);
    const double cylAxial = candidoSmoothIndicator(x.y() - inletLength, 0.6 * width);
    double a = std::max(sphere, cylRadial * cylAxial);
    if (opt.usePreconditionedPaperCurrentJet) {
      const double innerToOuter = s.innerDiameter / std::max(d0, 1e-30);
      const double paperMidplaneYOverDi =
          0.5 * s.collectorDistance / std::max(s.innerDiameter, 1e-30);
      const double tipYOverDi =
          opt.preconditionedJetTipYOverInnerDiameter > 0.0
              ? opt.preconditionedJetTipYOverInnerDiameter
              : paperMidplaneYOverDi + 0.75;
      const double jetRadius =
          std::max(opt.preconditionedJetRadiusInnerDiameters, 1e-6) *
          innerToOuter;
      const double jetWidth =
          std::max(opt.preconditionedJetInterfaceWidthInnerDiameters * innerToOuter,
                   0.25 * width);
      const double jetRadial = candidoSmoothIndicator(r - jetRadius, jetWidth);
      const double jetAxial =
          candidoSmoothIndicator(x.y() - tipYOverDi * innerToOuter, jetWidth);
      a = std::max(a, jetRadial * jetAxial);
    }
    if (a > 1e-6 && a < 1.0 - 1e-6) {
      const double theta = std::atan2(dz, dx);
      const double radialWindow = std::exp(-std::pow((r - ri) / std::max(0.8 * width, 1e-30), 2.0));
      a += perturb * std::sin(3.0 * theta) * radialWindow * a * (1.0 - a);
    }
    alpha[ci] = std::clamp(a, 0.0, 1.0);
  }
  return alpha;
}

inline void candidoMixtureFields3D(const fvm::Mesh3D& mesh,
                                   const fvm::ScalarField& alpha,
                                   const CandidoTaylorConeJetSetup& s,
                                   const CandidoConeJetSmokeOptions3D& opt,
                                   fvm::ScalarField& rho,
                                   fvm::ScalarField& eps,
                                   fvm::ScalarField& sigmaE) {
  rho.assign(mesh.cells.size(), 0.0);
  eps.assign(mesh.cells.size(), 0.0);
  sigmaE.assign(mesh.cells.size(), 0.0);
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    const double liquidSigma =
        opt.useDimensionalElectricalScaling
            ? candidoDimensionlessConductivityFromPhysical(s, s.liquidConductivity)
            : opt.normalizedLiquidConductivity;
    const double gasSigma =
        opt.useDimensionalElectricalScaling
            ? candidoDimensionlessConductivityFromPhysical(s, s.gasConductivity)
            : opt.normalizedGasConductivity;
    rho[ci] = a * s.liquidDensity + (1.0 - a) * s.gasDensity;
    eps[ci] = candidoHarmonicMixture(a, s.liquidRelativePermittivity,
                                     s.gasRelativePermittivity);
    sigmaE[ci] = candidoHarmonicMixture(a, liquidSigma, gasSigma);
  }
}

inline fvm::PotentialBoundary3D candidoPotentialBoundary3D(
    const fvm::Mesh3D& mesh, const CandidoTaylorConeJetSetup& s,
    const CandidoConeJetSmokeOptions3D& opt, double voltage) {
  fvm::PotentialBoundary3D bc;
  bc.faceDirichlet.assign(mesh.faces.size(), 0);
  bc.faceValue.assign(mesh.faces.size(), 0.0);
  const double lx = opt.radialWindowOuterDiameters;
  const double ly = s.collectorDistance / s.outerDiameter;
  const double lz = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * lx, 0.0, 0.5 * lz};
  const double outerRadius = 0.5;
  const double yTol = ly / std::max(opt.ny, 1) * 0.55;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (f.internal()) continue;
    const double r = std::hypot(f.centroid.x() - axis.x(), f.centroid.z() - axis.z());
    if (f.centroid.y() <= yTol && r <= outerRadius) {
      bc.faceDirichlet[fi] = 1;
      bc.faceValue[fi] = voltage;
    } else if (f.centroid.y() >= ly - yTol) {
      bc.faceDirichlet[fi] = 1;
      bc.faceValue[fi] = 0.0;
    }
  }
  return bc;
}

inline bool candidoIsNozzleBoundaryFace3D(
    const fvm::Face3D& f,
    const CandidoTaylorConeJetSetup& s,
    const CandidoConeJetSmokeOptions3D& opt) {
  const double lx = opt.radialWindowOuterDiameters;
  const double ly = s.collectorDistance / s.outerDiameter;
  const double lz = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * lx, 0.0, 0.5 * lz};
  const double outerRadius = 0.5;
  const double yTol = ly / std::max(opt.ny, 1) * 0.55;
  const double r = std::hypot(f.centroid.x() - axis.x(), f.centroid.z() - axis.z());
  return !f.internal() && f.centroid.y() <= yTol && r <= outerRadius;
}

inline void candidoSuppressNozzleConductiveFlux3D(
    const fvm::Mesh3D& mesh,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt,
    fvm::ScalarField& conductiveFlux) {
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    if (candidoIsNozzleBoundaryFace3D(mesh.faces[fi], setup, opt)) conductiveFlux[fi] = 0.0;
  }
}

inline void candidoKeepOnlyCollectorBoundaryConductiveFlux3D(
    const fvm::Mesh3D& mesh,
    fvm::ScalarField& conductiveFlux) {
  constexpr int collectorPatch = 3;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (!f.internal() && f.patch != collectorPatch) conductiveFlux[fi] = 0.0;
  }
}

inline fvm::ScalarField candidoImplicitOhmicConductiveFaceScale3D(
    const fvm::Mesh3D& mesh,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  fvm::ScalarField scale(mesh.faces.size(), 1.0);
  if (!opt.applyConductiveBoundaryFiltersInImplicitOhmic) return scale;

  constexpr int collectorPatch = 3;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (f.internal()) continue;
    if (opt.collectorOnlyConductiveChargeFlux && f.patch != collectorPatch) {
      scale[fi] = 0.0;
    }
    if (opt.suppressNozzleConductiveChargeFlux &&
        candidoIsNozzleBoundaryFace3D(f, setup, opt)) {
      scale[fi] = 0.0;
    }
  }
  return scale;
}

inline fvm::VectorField3 candidoInitialVelocity3D(const fvm::Mesh3D& mesh,
                                                  const CandidoTaylorConeJetSetup& s,
                                                  const CandidoConeJetSmokeOptions3D& opt) {
  const double d0 = s.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  const double ri = 0.5 * s.innerDiameter / d0;
  const double uScale = candidoDimensionlessInletVelocityScale(s);
  const fvm::Vec3 axis{0.5 * lx, 0.0, 0.5 * lz};
  fvm::VectorField3 u(mesh.cells.size(), fvm::Vec3::Zero());
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const fvm::Vec3 x = mesh.cells[ci].centroid;
    const double r = std::hypot(x.x() - axis.x(), x.z() - axis.z());
    if (x.y() < 0.8 && r < ri) {
      const double profile = std::max(0.0, 1.0 - (r * r) / std::max(ri * ri, 1e-30));
      u[ci].y() = 2.0 * uScale * profile;
    }
  }
  return u;
}

inline void candidoApplyPreconditionedPaperCurrentJetVelocityCells3D(
    fvm::VectorField3& u,
    const fvm::Mesh3D& mesh,
    const CandidoTaylorConeJetSetup& s,
    const CandidoConeJetSmokeOptions3D& opt) {
  if (!opt.usePreconditionedPaperCurrentJet) return;
  const double d0 = s.outerDiameter;
  const double innerToOuter = s.innerDiameter / std::max(d0, 1e-30);
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  const double paperMidplaneYOverDi =
      0.5 * s.collectorDistance / std::max(s.innerDiameter, 1e-30);
  const double tipYOverDi =
      opt.preconditionedJetTipYOverInnerDiameter > 0.0
          ? opt.preconditionedJetTipYOverInnerDiameter
          : paperMidplaneYOverDi + 0.75;
  const double tipY = tipYOverDi * innerToOuter;
  const double radius =
      std::max(opt.preconditionedJetRadiusInnerDiameters, 1e-6) * innerToOuter;
  const double width =
      std::max(opt.preconditionedJetInterfaceWidthInnerDiameters * innerToOuter,
               1e-6);
  const double uScale = candidoDimensionlessInletVelocityScale(s);
  const fvm::Vec3 axis{0.5 * lx, 0.0, 0.5 * lz};
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const fvm::Vec3 x = mesh.cells[ci].centroid;
    const double r = std::hypot(x.x() - axis.x(), x.z() - axis.z());
    const double radialBlend = candidoSmoothIndicator(r - radius, width);
    const double axialBlend = candidoSmoothIndicator(x.y() - tipY, width);
    const double blend = radialBlend * axialBlend;
    if (blend <= 1e-4) continue;
    const double parabolic =
        std::max(0.0, 1.0 - (r * r) / std::max(radius * radius, 1e-30));
    const double velocity =
        opt.preconditionedJetVelocityScale * uScale *
        (0.25 + 1.75 * parabolic) * blend;
    u[ci].y() = std::max(u[ci].y(), velocity);
  }
}

inline void candidoApplyFullyDevelopedInletVelocityCells3D(
    fvm::VectorField3& u,
    const fvm::Mesh3D& mesh,
    const CandidoTaylorConeJetSetup& s,
    const CandidoConeJetSmokeOptions3D& opt) {
  const double d0 = s.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  const double ri = 0.5 * s.innerDiameter / d0;
  const double inletLength = std::min(s.nozzleLength / d0, 0.8);
  const double uScale = candidoDimensionlessInletVelocityScale(s);
  const fvm::Vec3 axis{0.5 * lx, 0.0, 0.5 * lz};
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const fvm::Vec3 x = mesh.cells[ci].centroid;
    const double r = std::hypot(x.x() - axis.x(), x.z() - axis.z());
    if (x.y() <= inletLength && r <= ri) {
      const double profile = std::max(0.0, 1.0 - (r * r) / std::max(ri * ri, 1e-30));
      u[ci] = fvm::Vec3{0.0, 2.0 * uScale * profile, 0.0};
    }
  }
}

inline void candidoApplyMovingCollectorWallCells3D(
    fvm::VectorField3& u,
    const fvm::Mesh3D& mesh,
    const CandidoTaylorConeJetSetup& s,
    const CandidoConeJetSmokeOptions3D& opt) {
  if (!opt.useMovingCollectorWall) return;
  const double ly = s.collectorDistance / s.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  const double collectorRadius = 0.5 * s.collectorDiameter / s.outerDiameter;
  const double yTol = ly / std::max(opt.ny, 1) * 0.55;
  const double ux = -candidoDimensionlessCollectorVelocityScale(s);
  const fvm::Vec3 axis{0.5 * lx, 0.0, 0.5 * lz};
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const fvm::Vec3 x = mesh.cells[ci].centroid;
    if (x.y() < ly - yTol) continue;
    const double r = std::hypot(x.x() - axis.x(), x.z() - axis.z());
    if (r > collectorRadius) continue;
    u[ci].x() = ux;
  }
}

inline fvm::ScalarField candidoFaceFlux3D(const fvm::Mesh3D& mesh,
                                          const fvm::VectorField3& u) {
  fvm::ScalarField flux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const fvm::Vec3 uf = f.internal() ? 0.5 * (u[f.owner] + u[f.neighbour]) : u[f.owner];
    flux[fi] = f.internal() ? uf.dot(f.Sf) : 0.0;
  }
  return flux;
}

inline bool candidoIsInletBoundaryFace3D(const fvm::Mesh3D& mesh,
                                         const fvm::Face3D& f,
                                         const CandidoTaylorConeJetSetup& setup,
                                         const CandidoConeJetSmokeOptions3D& opt) {
  if (f.internal()) return false;
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double axisX = 0.5 * opt.radialWindowOuterDiameters;
  const double axisZ = 0.5 * opt.radialWindowOuterDiameters;
  const double inletRadius = 0.5 * setup.innerDiameter / setup.outerDiameter;
  const double yTol = ly / std::max(opt.ny, 1) * 0.55;
  if (f.centroid.y() > yTol) return false;
  double minVertexRadius =
      std::hypot(f.centroid.x() - axisX, f.centroid.z() - axisZ);
  for (int pi : f.points) {
    minVertexRadius = std::min(
        minVertexRadius,
        std::hypot(mesh.points[pi].x() - axisX, mesh.points[pi].z() - axisZ));
  }
  return minVertexRadius <= inletRadius;
}

inline void candidoApplyOpenAtmosphericBoundaryFlux3D(
    fvm::ScalarField& faceFlux,
    const fvm::Mesh3D& mesh,
    const fvm::VectorField3& u,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  if (!opt.useOpenAtmosphericBoundaryFlux) return;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (f.internal()) continue;
    if (candidoIsInletBoundaryFace3D(mesh, f, setup, opt)) continue;
    if (f.patch == 2) continue;
    faceFlux[fi] = u[f.owner].dot(f.Sf);
  }
}

inline void candidoApplyInletBoundaryFlux3D(fvm::ScalarField& faceFlux,
                                            const fvm::Mesh3D& mesh,
                                            const fvm::VectorField3& u,
                                            const CandidoTaylorConeJetSetup& setup,
                                            const CandidoConeJetSmokeOptions3D& opt) {
  const double axisX = 0.5 * opt.radialWindowOuterDiameters;
  const double axisZ = 0.5 * opt.radialWindowOuterDiameters;
  const double inletRadius = 0.5 * setup.innerDiameter / setup.outerDiameter;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (candidoIsInletBoundaryFace3D(mesh, f, setup, opt)) {
      if (opt.useFullyDevelopedInletVelocityBoundary) {
        const double r =
            std::hypot(f.centroid.x() - axisX, f.centroid.z() - axisZ);
        const double profile =
            std::max(0.0, 1.0 - (r * r) / std::max(inletRadius * inletRadius, 1e-30));
        const fvm::Vec3 inletU{0.0,
                               2.0 * candidoDimensionlessInletVelocityScale(setup) *
                                   profile,
                               0.0};
        faceFlux[fi] = inletU.dot(f.Sf);
      } else {
        faceFlux[fi] = u[f.owner].dot(f.Sf);
      }
    }
  }
}

inline fvm::ScalarField candidoInletBoundaryAlpha3D(
    const fvm::Mesh3D& mesh, const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  fvm::ScalarField boundaryAlpha(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (candidoIsInletBoundaryFace3D(mesh, f, setup, opt)) boundaryAlpha[fi] = 1.0;
  }
  return boundaryAlpha;
}

inline fvm::ScalarField candidoConductiveCurrentFlux3D(const fvm::Mesh3D& mesh,
                                                       const fvm::ScalarField& sigmaE,
                                                       const fvm::VectorField3& E) {
  fvm::ScalarField flux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double sf = f.internal() ? fvm::harmonicMean(sigmaE[f.owner], sigmaE[f.neighbour])
                                   : sigmaE[f.owner];
    const fvm::Vec3 ef = f.internal() ? 0.5 * (E[f.owner] + E[f.neighbour]) : E[f.owner];
    flux[fi] = sf * ef.dot(f.Sf);
  }
  return flux;
}

inline fvm::ScalarField candidoPoissonFaceNormalFlux3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& coeffCell,
    const fvm::ScalarField& phi,
    const fvm::PotentialBoundary3D& bc) {
  fvm::ScalarField flux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double geom = std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
    if (f.internal()) {
      const double coeffF = fvm::harmonicMean(coeffCell[f.owner], coeffCell[f.neighbour]);
      flux[fi] = coeffF * geom * (phi[f.owner] - phi[f.neighbour]);
    } else if (!bc.faceDirichlet.empty() && bc.faceDirichlet[fi]) {
      const double facePhi = bc.faceValue.empty() ? 0.0 : bc.faceValue[fi];
      flux[fi] = coeffCell[f.owner] * geom * (phi[f.owner] - facePhi);
    }
  }
  return flux;
}

inline fvm::ScalarField candidoPoissonFaceConductiveCurrentFlux3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& sigmaE,
    const fvm::ScalarField& phi,
    const fvm::PotentialBoundary3D& bc) {
  return candidoPoissonFaceNormalFlux3D(mesh, sigmaE, phi, bc);
}

inline fvm::EHDBodyForceReport3D candidoPoissonFaceMaxwellBodyForce3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& rhoE,
    const fvm::ScalarField& eps,
    const fvm::ScalarField& phi,
    const fvm::PotentialBoundary3D& bc) {
  fvm::require(rhoE.size() == mesh.cells.size(),
               "Candido Poisson-face Maxwell force charge size mismatch");
  fvm::require(eps.size() == mesh.cells.size(),
               "Candido Poisson-face Maxwell force eps size mismatch");
  fvm::require(phi.size() == mesh.cells.size(),
               "Candido Poisson-face Maxwell force phi size mismatch");

  const fvm::ScalarField epsFlux = candidoPoissonFaceNormalFlux3D(mesh, eps, phi, bc);
  const fvm::ScalarField epsF = fvm::facePermittivityHarmonic3D(mesh, eps);
  fvm::EHDBodyForceReport3D report;
  report.force.assign(mesh.cells.size(), fvm::Vec3::Zero());
  report.stressDivergence.assign(mesh.cells.size(), fvm::Vec3::Zero());
  report.faceCoupledForce.assign(mesh.cells.size(), fvm::Vec3::Zero());
  const fvm::VectorField3 gradEps = fvm::gradFromFaceSnGrad3D(mesh, eps);

  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double area = f.Sf.norm();
    if (area <= 1e-30 || epsF[fi] <= 1e-30) continue;
    const fvm::Vec3 n = f.Sf / area;
    const double eNormal = epsFlux[fi] / std::max(epsF[fi] * area, 1e-30);
    const fvm::Vec3 ef = eNormal * n;
    const Eigen::Matrix3d stress =
        epsF[fi] * (ef * ef.transpose() -
                    0.5 * ef.squaredNorm() * Eigen::Matrix3d::Identity());
    const fvm::Vec3 traction = stress * f.Sf;
    report.stressDivergence[f.owner] -= traction;
    if (f.internal()) report.stressDivergence[f.neighbour] += traction;
  }

  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    report.stressDivergence[ci] /= std::max(mesh.cells[ci].V, 1e-30);
    report.force[ci] = report.stressDivergence[ci];
    report.faceCoupledForce[ci] = report.stressDivergence[ci];
    report.maxForce = std::max(report.maxForce, report.force[ci].norm());
    report.maxGradEps = std::max(report.maxGradEps, gradEps[ci].norm());
    report.maxStressDivergence =
        std::max(report.maxStressDivergence, report.stressDivergence[ci].norm());
    report.maxFaceCoupledForce =
        std::max(report.maxFaceCoupledForce, report.faceCoupledForce[ci].norm());
  }
  return report;
}

inline fvm::EHDBodyForceReport3D candidoPoissonHybridMaxwellBodyForce3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& rhoE,
    const fvm::ScalarField& eps,
    const fvm::ScalarField& phi,
    const fvm::VectorField3& cellE,
    const fvm::PotentialBoundary3D& bc) {
  fvm::require(rhoE.size() == mesh.cells.size(),
               "Candido hybrid Maxwell force charge size mismatch");
  fvm::require(eps.size() == mesh.cells.size(),
               "Candido hybrid Maxwell force eps size mismatch");
  fvm::require(phi.size() == mesh.cells.size(),
               "Candido hybrid Maxwell force phi size mismatch");
  fvm::require(cellE.size() == mesh.cells.size(),
               "Candido hybrid Maxwell force E size mismatch");

  const fvm::ScalarField epsFlux = candidoPoissonFaceNormalFlux3D(mesh, eps, phi, bc);
  const fvm::ScalarField epsF = fvm::facePermittivityHarmonic3D(mesh, eps);
  const fvm::VectorField3 gradEps = fvm::gradFromFaceSnGrad3D(mesh, eps);
  fvm::EHDBodyForceReport3D report;
  report.force.assign(mesh.cells.size(), fvm::Vec3::Zero());
  report.stressDivergence.assign(mesh.cells.size(), fvm::Vec3::Zero());
  report.faceCoupledForce.assign(mesh.cells.size(), fvm::Vec3::Zero());

  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double area = f.Sf.norm();
    if (area <= 1e-30 || epsF[fi] <= 1e-30) continue;
    const fvm::Vec3 n = f.Sf / area;
    const double eNormal = epsFlux[fi] / std::max(epsF[fi] * area, 1e-30);
    const fvm::Vec3 efCell = f.internal() ? 0.5 * (cellE[f.owner] + cellE[f.neighbour])
                                          : cellE[f.owner];
    const fvm::Vec3 eTangential = efCell - efCell.dot(n) * n;
    const fvm::Vec3 ef = eNormal * n + eTangential;
    const Eigen::Matrix3d stress =
        epsF[fi] * (ef * ef.transpose() -
                    0.5 * ef.squaredNorm() * Eigen::Matrix3d::Identity());
    const fvm::Vec3 traction = stress * f.Sf;
    report.stressDivergence[f.owner] -= traction;
    if (f.internal()) report.stressDivergence[f.neighbour] += traction;
  }

  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    report.stressDivergence[ci] /= std::max(mesh.cells[ci].V, 1e-30);
    report.force[ci] = report.stressDivergence[ci];
    report.faceCoupledForce[ci] = report.stressDivergence[ci];
    report.maxForce = std::max(report.maxForce, report.force[ci].norm());
    report.maxGradEps = std::max(report.maxGradEps, gradEps[ci].norm());
    report.maxStressDivergence =
        std::max(report.maxStressDivergence, report.stressDivergence[ci].norm());
    report.maxFaceCoupledForce =
        std::max(report.maxFaceCoupledForce, report.faceCoupledForce[ci].norm());
  }
  return report;
}

inline CandidoBoundedVectorMaxwellReport3D
candidoPoissonBoundedVectorMaxwellBodyForceReport3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& rhoE,
    const fvm::ScalarField& eps,
    const fvm::ScalarField& phi,
    const fvm::VectorField3& cellE,
    const fvm::PotentialBoundary3D& bc,
    double tangentialLimitFactor,
    double tangentialLimitFloorFraction) {
  fvm::require(rhoE.size() == mesh.cells.size(),
               "Candido bounded vector Maxwell force charge size mismatch");
  fvm::require(eps.size() == mesh.cells.size(),
               "Candido bounded vector Maxwell force eps size mismatch");
  fvm::require(phi.size() == mesh.cells.size(),
               "Candido bounded vector Maxwell force phi size mismatch");
  fvm::require(cellE.size() == mesh.cells.size(),
               "Candido bounded vector Maxwell force E size mismatch");

  const fvm::ScalarField epsFlux = candidoPoissonFaceNormalFlux3D(mesh, eps, phi, bc);
  const fvm::ScalarField epsF = fvm::facePermittivityHarmonic3D(mesh, eps);
  const fvm::VectorField3 gradEps = fvm::gradFromFaceSnGrad3D(mesh, eps);
  const double limitFactor = std::max(0.0, tangentialLimitFactor);
  const double floorFraction = std::max(0.0, tangentialLimitFloorFraction);

  CandidoBoundedVectorMaxwellReport3D bounded;
  bounded.tangentialLimitFactor = limitFactor;
  bounded.tangentialLimitFloorFraction = floorFraction;
  bounded.force.force.assign(mesh.cells.size(), fvm::Vec3::Zero());
  bounded.force.stressDivergence.assign(mesh.cells.size(), fvm::Vec3::Zero());
  bounded.force.faceCoupledForce.assign(mesh.cells.size(), fvm::Vec3::Zero());

  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double area = f.Sf.norm();
    if (area <= 1e-30 || epsF[fi] <= 1e-30) continue;
    const double eNormal = epsFlux[fi] / std::max(epsF[fi] * area, 1e-30);
    bounded.maxPoissonNormalE = std::max(bounded.maxPoissonNormalE, std::abs(eNormal));
  }

  double clipRatioSum = 0.0;
  bounded.minTangentialClipRatio = 1.0;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double area = f.Sf.norm();
    if (area <= 1e-30 || epsF[fi] <= 1e-30) continue;
    ++bounded.sampledFaces;
    const fvm::Vec3 n = f.Sf / area;
    const double eNormal = epsFlux[fi] / std::max(epsF[fi] * area, 1e-30);
    const fvm::Vec3 efCell = f.internal() ? 0.5 * (cellE[f.owner] + cellE[f.neighbour])
                                          : cellE[f.owner];
    fvm::Vec3 eTangential = efCell - efCell.dot(n) * n;
    const double rawTangential = eTangential.norm();
    bounded.maxRawTangentialE = std::max(bounded.maxRawTangentialE, rawTangential);
    const double tangentialLimit =
        limitFactor *
        std::max(std::abs(eNormal), floorFraction * bounded.maxPoissonNormalE);
    double clipRatio = 1.0;
    if (rawTangential > tangentialLimit && rawTangential > 1e-30) {
      clipRatio = tangentialLimit / rawTangential;
      eTangential *= clipRatio;
      ++bounded.tangentialClippedFaces;
    }
    bounded.minTangentialClipRatio =
        std::min(bounded.minTangentialClipRatio, clipRatio);
    clipRatioSum += clipRatio;
    bounded.maxLimitedTangentialE =
        std::max(bounded.maxLimitedTangentialE, eTangential.norm());

    const fvm::Vec3 ef = eNormal * n + eTangential;
    const Eigen::Matrix3d stress =
        epsF[fi] * (ef * ef.transpose() -
                    0.5 * ef.squaredNorm() * Eigen::Matrix3d::Identity());
    const fvm::Vec3 traction = stress * f.Sf;
    bounded.force.stressDivergence[f.owner] -= traction;
    if (f.internal()) bounded.force.stressDivergence[f.neighbour] += traction;
  }

  if (bounded.sampledFaces > 0) {
    bounded.meanTangentialClipRatio =
        clipRatioSum / static_cast<double>(bounded.sampledFaces);
  }
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    bounded.force.stressDivergence[ci] /= std::max(mesh.cells[ci].V, 1e-30);
    bounded.force.force[ci] = bounded.force.stressDivergence[ci];
    bounded.force.faceCoupledForce[ci] = bounded.force.stressDivergence[ci];
    bounded.force.maxForce =
        std::max(bounded.force.maxForce, bounded.force.force[ci].norm());
    bounded.force.maxGradEps =
        std::max(bounded.force.maxGradEps, gradEps[ci].norm());
    bounded.force.maxStressDivergence =
        std::max(bounded.force.maxStressDivergence,
                 bounded.force.stressDivergence[ci].norm());
    bounded.force.maxFaceCoupledForce =
        std::max(bounded.force.maxFaceCoupledForce,
                 bounded.force.faceCoupledForce[ci].norm());
  }
  return bounded;
}

inline fvm::EHDBodyForceReport3D candidoPoissonBoundedVectorMaxwellBodyForce3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& rhoE,
    const fvm::ScalarField& eps,
    const fvm::ScalarField& phi,
    const fvm::VectorField3& cellE,
    const fvm::PotentialBoundary3D& bc,
    double tangentialLimitFactor,
    double tangentialLimitFloorFraction) {
  return candidoPoissonBoundedVectorMaxwellBodyForceReport3D(
             mesh, rhoE, eps, phi, cellE, bc, tangentialLimitFactor,
             tangentialLimitFloorFraction)
      .force;
}

inline CandidoBoundedVectorMaxwellReport3D candidoBoundedVectorMaxwellDiagnostic3D(
    double targetCaE,
    const CandidoTaylorConeJetSetup& setup = {},
    const CandidoConeJetSmokeOptions3D& opt = {},
    double tangentialLimitFactor = 1.0,
    double tangentialLimitFloorFraction = 0.05) {
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  fvm::Mesh3D mesh = fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, lx, ly, lz, opt.skew);
  const double voltage = candidoVoltageForElectricCapillary(setup, targetCaE);
  const double dimensionlessVoltage =
      voltage / (candidoElectricFieldScale(setup, setup.validationVoltage) *
                 setup.outerDiameter);
  const fvm::ScalarField alpha = candidoInitialAlpha3D(mesh, setup, opt, targetCaE);
  fvm::ScalarField rho, eps, sigmaE;
  candidoMixtureFields3D(mesh, alpha, setup, opt, rho, eps, sigmaE);
  const fvm::ScalarField rhoE(mesh.cells.size(), 0.0);
  const fvm::PotentialBoundary3D bc =
      candidoPotentialBoundary3D(mesh, setup, opt, dimensionlessVoltage);
  const fvm::PotentialSolveReport3D potential =
      fvm::solvePotential3D(mesh, eps, rhoE, bc, 1e-11, 4000);
  CandidoBoundedVectorMaxwellReport3D report =
      candidoPoissonBoundedVectorMaxwellBodyForceReport3D(
          mesh, rhoE, eps, potential.phi, potential.E, bc, tangentialLimitFactor,
          tangentialLimitFloorFraction);
  report.potentialResidual = potential.residual;
  return report;
}

inline CandidoTomarConductingForceReport3D
candidoTomarConductingSurfaceForceReport3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const fvm::ScalarField& rhoE,
    const fvm::ScalarField& eps,
    const fvm::ScalarField& sigmaE,
    const fvm::VectorField3& electricField,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  fvm::require(alpha.size() == mesh.cells.size(),
               "Candido Tomar conducting force alpha size mismatch");
  fvm::require(rhoE.size() == mesh.cells.size(),
               "Candido Tomar conducting force charge size mismatch");
  fvm::require(eps.size() == mesh.cells.size(),
               "Candido Tomar conducting force eps size mismatch");
  fvm::require(sigmaE.size() == mesh.cells.size(),
               "Candido Tomar conducting force sigma size mismatch");
  fvm::require(electricField.size() == mesh.cells.size(),
               "Candido Tomar conducting force E size mismatch");

  const double liquidSigma =
      opt.useDimensionalElectricalScaling
          ? candidoDimensionlessConductivityFromPhysical(setup, setup.liquidConductivity)
          : opt.normalizedLiquidConductivity;
  const double gasSigma =
      opt.useDimensionalElectricalScaling
          ? candidoDimensionlessConductivityFromPhysical(setup, setup.gasConductivity)
          : opt.normalizedGasConductivity;
  const double eps1 = std::max(setup.liquidRelativePermittivity, 1e-30);
  const double eps2 = std::max(setup.gasRelativePermittivity, 1e-30);
  const double sigma1 = std::max(liquidSigma, 1e-30);
  const double sigma2 = std::max(gasSigma, 1e-30);
  const double normalJumpCoeff =
      eps1 / (sigma1 * sigma1) - eps2 / (sigma2 * sigma2);
  const double tangentialJumpCoeff = eps1 - eps2;
  const double chargeTangentialCoeff = eps1 / sigma1 - eps2 / sigma2;

  const fvm::VectorField3 gradAlpha = fvm::gradFromFaceSnGrad3D(mesh, alpha);
  const fvm::VectorField3 gradEps = fvm::gradFromFaceSnGrad3D(mesh, eps);
  CandidoTomarConductingForceReport3D report;
  report.force.force.assign(mesh.cells.size(), fvm::Vec3::Zero());
  report.force.stressDivergence.assign(mesh.cells.size(), fvm::Vec3::Zero());
  report.force.faceCoupledForce.assign(mesh.cells.size(), fvm::Vec3::Zero());

  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    ++report.sampledCells;
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a > 1e-8 && a < 1.0 - 1e-8) ++report.mixedCells;
    const fvm::Vec3 gAlpha = gradAlpha[ci];
    const double gradMag = gAlpha.norm();
    report.maxGradAlpha = std::max(report.maxGradAlpha, gradMag);
    if (gradMag <= 1e-14) continue;
    ++report.activeInterfaceCells;
    const fvm::Vec3 normal = gAlpha / gradMag;
    const fvm::Vec3 E = electricField[ci];
    const double jNormal = sigmaE[ci] * E.dot(normal);
    const fvm::Vec3 eTangential = E - E.dot(normal) * normal;
    const double eTangential2 = eTangential.squaredNorm();
    const double normalTerm =
        0.5 * (jNormal * jNormal * normalJumpCoeff -
               eTangential2 * tangentialJumpCoeff);
    const double jDotGradAlpha = (sigmaE[ci] * E).dot(gAlpha);
    const fvm::Vec3 tangentialTerm =
        jDotGradAlpha * chargeTangentialCoeff * eTangential;
    const fvm::Vec3 cellForce = normalTerm * gAlpha + tangentialTerm;
    report.force.force[ci] = cellForce;
    report.force.stressDivergence[ci] = cellForce;
    report.force.faceCoupledForce[ci] = cellForce;
    report.force.maxForce = std::max(report.force.maxForce, cellForce.norm());
    report.force.maxGradEps =
        std::max(report.force.maxGradEps, gradEps[ci].norm());
    report.force.maxStressDivergence =
        std::max(report.force.maxStressDivergence, cellForce.norm());
    report.force.maxFaceCoupledForce =
        std::max(report.force.maxFaceCoupledForce, cellForce.norm());
    report.maxNormalCurrent =
        std::max(report.maxNormalCurrent, std::abs(jNormal));
    report.maxTangentialE =
        std::max(report.maxTangentialE, std::sqrt(eTangential2));
    report.maxNormalTerm =
        std::max(report.maxNormalTerm, std::abs(normalTerm) * gradMag);
    report.maxTangentialTerm =
        std::max(report.maxTangentialTerm, tangentialTerm.norm());
    report.maxRawCellForce =
        std::max(report.maxRawCellForce, cellForce.norm());
  }

  fvm::EHDBodyForceReport3D defaultForce =
      fvm::maxwellBodyForce3D(mesh, rhoE, electricField, eps);
  report.defaultMaxForce = defaultForce.maxForce;
  return report;
}

inline fvm::EHDBodyForceReport3D candidoTomarConductingSurfaceForce3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const fvm::ScalarField& rhoE,
    const fvm::ScalarField& eps,
    const fvm::ScalarField& sigmaE,
    const fvm::VectorField3& electricField,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  return candidoTomarConductingSurfaceForceReport3D(
             mesh, alpha, rhoE, eps, sigmaE, electricField, setup, opt)
      .force;
}

inline CandidoTomarConductingForceReport3D
candidoTomarConductingSurfaceForceDiagnostic3D(
    double targetCaE,
    const CandidoTaylorConeJetSetup& setup = {},
    const CandidoConeJetSmokeOptions3D& opt = {}) {
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  fvm::Mesh3D mesh = fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, lx, ly, lz, opt.skew);
  const double voltage = candidoVoltageForElectricCapillary(setup, targetCaE);
  const double dimensionlessVoltage =
      voltage / (candidoElectricFieldScale(setup, setup.validationVoltage) *
                 setup.outerDiameter);
  const fvm::ScalarField alpha = candidoInitialAlpha3D(mesh, setup, opt, targetCaE);
  fvm::ScalarField rho, eps, sigmaE;
  candidoMixtureFields3D(mesh, alpha, setup, opt, rho, eps, sigmaE);
  const fvm::ScalarField rhoE(mesh.cells.size(), 0.0);
  const fvm::PotentialBoundary3D bc =
      candidoPotentialBoundary3D(mesh, setup, opt, dimensionlessVoltage);
  const fvm::PotentialSolveReport3D potential =
      fvm::solvePotential3D(mesh, eps, rhoE, bc, 1e-11, 4000);
  CandidoTomarConductingForceReport3D report =
      candidoTomarConductingSurfaceForceReport3D(mesh, alpha, rhoE, eps, sigmaE,
                                                 potential.E, setup, opt);
  report.potentialResidual = potential.residual;
  return report;
}

inline CandidoFaceElectricReconstructionDiagnostic3D
candidoFaceElectricReconstructionDiagnostic3D(
    double targetCaE,
    const CandidoTaylorConeJetSetup& setup = {},
    const CandidoConeJetSmokeOptions3D& opt = {}) {
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  fvm::Mesh3D mesh = fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, lx, ly, lz, opt.skew);
  const double voltage = candidoVoltageForElectricCapillary(setup, targetCaE);
  const double dimensionlessVoltage =
      voltage / (candidoElectricFieldScale(setup, setup.validationVoltage) *
                 setup.outerDiameter);
  const fvm::ScalarField alpha = candidoInitialAlpha3D(mesh, setup, opt, targetCaE);
  fvm::ScalarField rho, eps, sigmaE;
  candidoMixtureFields3D(mesh, alpha, setup, opt, rho, eps, sigmaE);
  const fvm::ScalarField rhoE(mesh.cells.size(), 0.0);
  const fvm::PotentialBoundary3D bc =
      candidoPotentialBoundary3D(mesh, setup, opt, dimensionlessVoltage);
  const fvm::PotentialSolveReport3D potential =
      fvm::solvePotential3D(mesh, eps, rhoE, bc, 1e-11, 4000);
  const fvm::ScalarField epsFlux =
      candidoPoissonFaceNormalFlux3D(mesh, eps, potential.phi, bc);
  const fvm::ScalarField epsF = fvm::facePermittivityHarmonic3D(mesh, eps);

  CandidoFaceElectricReconstructionDiagnostic3D report;
  report.potentialResidual = potential.residual;
  std::vector<double> normalTractions;
  std::vector<double> hybridTractions;
  normalTractions.reserve(mesh.faces.size());
  hybridTractions.reserve(mesh.faces.size());
  double maxNormalTraction = 0.0;

  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double area = f.Sf.norm();
    if (area <= 1e-30 || epsF[fi] <= 1e-30) continue;
    ++report.sampledFaces;
    if (f.internal()) {
      ++report.internalFaces;
    } else if (!bc.faceDirichlet.empty() && bc.faceDirichlet[fi]) {
      ++report.dirichletBoundaryFaces;
    }
    const fvm::Vec3 n = f.Sf / area;
    const double eNormalPoisson = epsFlux[fi] / std::max(epsF[fi] * area, 1e-30);
    const fvm::Vec3 eCell =
        f.internal() ? 0.5 * (potential.E[f.owner] + potential.E[f.neighbour])
                     : potential.E[f.owner];
    const double eNormalCell = eCell.dot(n);
    const fvm::Vec3 eTangential = eCell - eNormalCell * n;
    const double tangentialMag = eTangential.norm();
    const double hybridMag = std::hypot(eNormalPoisson, tangentialMag);
    const double normalDenom =
        std::max(std::max(std::abs(eNormalPoisson), std::abs(eNormalCell)), 1e-30);
    const double relNormalMismatch =
        std::abs(eNormalPoisson - eNormalCell) / normalDenom;
    const double tangentialFraction = tangentialMag / std::max(hybridMag, 1e-30);

    auto tractionNorm = [&](const fvm::Vec3& ef) {
      const Eigen::Matrix3d stress =
          epsF[fi] * (ef * ef.transpose() -
                      0.5 * ef.squaredNorm() * Eigen::Matrix3d::Identity());
      return (stress * f.Sf).norm();
    };
    const double normalTraction = tractionNorm(eNormalPoisson * n);
    const double hybridTraction = tractionNorm(eNormalPoisson * n + eTangential);

    report.maxPoissonNormalE =
        std::max(report.maxPoissonNormalE, std::abs(eNormalPoisson));
    report.maxCellTangentialE =
        std::max(report.maxCellTangentialE, tangentialMag);
    report.meanRelativeNormalMismatch += relNormalMismatch;
    report.maxRelativeNormalMismatch =
        std::max(report.maxRelativeNormalMismatch, relNormalMismatch);
    report.meanTangentialFraction += tangentialFraction;
    report.maxTangentialFraction =
        std::max(report.maxTangentialFraction, tangentialFraction);
    maxNormalTraction = std::max(maxNormalTraction, normalTraction);
    normalTractions.push_back(normalTraction);
    hybridTractions.push_back(hybridTraction);
  }

  if (report.sampledFaces > 0) {
    const double inv = 1.0 / static_cast<double>(report.sampledFaces);
    report.meanRelativeNormalMismatch *= inv;
    report.meanTangentialFraction *= inv;
  }
  std::vector<double> tractionRatios;
  tractionRatios.reserve(normalTractions.size());
  const double activeNormalTraction = 1e-8 * std::max(maxNormalTraction, 1e-300);
  for (size_t i = 0; i < normalTractions.size(); ++i) {
    if (normalTractions[i] <= activeNormalTraction) {
      ++report.normalTractionDegenerateFaces;
      continue;
    }
    const double tractionRatio = hybridTractions[i] / normalTractions[i];
    report.meanHybridToNormalTractionRatio += tractionRatio;
    report.maxHybridToNormalTractionRatio =
        std::max(report.maxHybridToNormalTractionRatio, tractionRatio);
    tractionRatios.push_back(tractionRatio);
  }
  report.tractionRatioFaces = static_cast<int>(tractionRatios.size());
  if (report.tractionRatioFaces > 0) {
    report.meanHybridToNormalTractionRatio /=
        static_cast<double>(report.tractionRatioFaces);
  }
  if (!tractionRatios.empty()) {
    std::sort(tractionRatios.begin(), tractionRatios.end());
    const size_t p95Index = static_cast<size_t>(
        std::min<double>(tractionRatios.size() - 1,
                         std::floor(0.95 * static_cast<double>(tractionRatios.size() - 1))));
    report.p95HybridToNormalTractionRatio = tractionRatios[p95Index];
  }
  return report;
}

inline fvm::ScalarField candidoConvectiveChargeFlux3D(const fvm::Mesh3D& mesh,
                                                      const fvm::ScalarField& rhoE,
                                                      const fvm::ScalarField& faceFlux,
                                                      bool useBoundaryChargeAdvection = false,
                                                      double inflowCharge = 0.0) {
  fvm::ScalarField flux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (f.internal()) {
      const int up = faceFlux[fi] >= 0.0 ? f.owner : f.neighbour;
      flux[fi] = faceFlux[fi] * rhoE[up];
    } else if (useBoundaryChargeAdvection) {
      const double qFace = faceFlux[fi] >= 0.0 ? rhoE[f.owner] : inflowCharge;
      flux[fi] = faceFlux[fi] * qFace;
    }
  }
  return flux;
}

inline fvm::ScalarField candidoBoundaryOnlyConductiveFlux3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& conductiveFlux) {
  fvm::ScalarField boundaryFlux(conductiveFlux.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    if (!mesh.faces[fi].internal()) boundaryFlux[fi] = conductiveFlux[fi];
  }
  return boundaryFlux;
}

inline double candidoIntegratedCharge3D(const fvm::Mesh3D& mesh,
                                        const fvm::ScalarField& rhoE) {
  double q = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) q += rhoE[ci] * mesh.cells[ci].V;
  return q;
}

struct CandidoChargeAdvanceReport3D {
  double initialCharge = 0.0;
  double unclampedCharge = 0.0;
  double finalCharge = 0.0;
  double maxUnclampedAbsCharge = 0.0;
  double clampCorrectionL1 = 0.0;
  double redistributionResidual = 0.0;
  double redistributionDeficitL1 = 0.0;
  double maxRedistributionWeightedCapacity = 0.0;
  int maxRedistributionWeightedCells = 0;
  double implicitPotentialResidual = 0.0;
  double boundaryConvectiveChargeFlux = 0.0;
  double boundaryConductiveChargeFlux = 0.0;
  double maxAbsConductiveCurrent = 0.0;
  std::array<double, 6> boundaryConductiveByPatch = {0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0};
  int clampedCells = 0;
};

inline fvm::ScalarField candidoInterfaceChargeRedistributionWeights3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoConeJetSmokeOptions3D& opt) {
  fvm::require(alpha.size() == mesh.cells.size(),
               "Candido interface charge redistribution alpha size mismatch");
  fvm::ScalarField weights(mesh.cells.size(), 0.0);
  const double liquidFloor =
      std::max(0.0, opt.interfaceChargeRedistributionLiquidFloor);
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    weights[ci] = a * (1.0 - a) + liquidFloor * a;
  }
  return weights;
}

struct CandidoInterfacialOhmicChargeSourceReport3D {
  int sourceCells = 0;
  double appliedCharge = 0.0;
  double clampCorrectionL1 = 0.0;
  double maxAbsSourceDensity = 0.0;
};

inline CandidoInterfacialOhmicChargeSourceReport3D
candidoApplyInterfacialOhmicChargeSource3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const fvm::VectorField3& electricField,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt,
    double dt,
    double qLimit,
    fvm::ScalarField& rhoE) {
  fvm::require(alpha.size() == mesh.cells.size(),
               "Candido interfacial Ohmic source alpha size mismatch");
  fvm::require(electricField.size() == mesh.cells.size(),
               "Candido interfacial Ohmic source electric-field size mismatch");
  fvm::require(rhoE.size() == mesh.cells.size(),
               "Candido interfacial Ohmic source charge size mismatch");

  CandidoInterfacialOhmicChargeSourceReport3D report;
  if (!opt.useInterfacialOhmicChargeSource) return report;

  const double sigmaLiquid =
      opt.useDimensionalElectricalScaling
          ? candidoDimensionlessConductivityFromPhysical(setup, setup.liquidConductivity)
          : opt.normalizedLiquidConductivity;
  const double sigmaGas =
      opt.useDimensionalElectricalScaling
          ? candidoDimensionlessConductivityFromPhysical(setup, setup.gasConductivity)
          : opt.normalizedGasConductivity;
  const double sigmaJump = sigmaLiquid - sigmaGas;
  const fvm::VectorField3 gradAlpha = fvm::gradLeastSquares3D(mesh, alpha);
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    if (gradAlpha[ci].norm() <= 1e-14) continue;
    const double sourceDensityRate =
        -opt.interfacialOhmicChargeSourceScale * sigmaJump *
        gradAlpha[ci].dot(electricField[ci]);
    const double oldCharge = rhoE[ci];
    const double unclamped = oldCharge + dt * sourceDensityRate;
    const double clamped = std::clamp(unclamped, -qLimit, qLimit);
    rhoE[ci] = clamped;
    report.appliedCharge += (clamped - oldCharge) * mesh.cells[ci].V;
    report.clampCorrectionL1 += std::abs(clamped - unclamped) * mesh.cells[ci].V;
    report.maxAbsSourceDensity =
        std::max(report.maxAbsSourceDensity, std::abs(sourceDensityRate));
    ++report.sourceCells;
  }
  return report;
}

inline void candidoRedistributeChargeDeficit3D(
    const fvm::Mesh3D& mesh,
    fvm::ScalarField& rhoE,
    double qLimit,
    double chargeConservationTarget,
    const fvm::ScalarField* redistributionWeights,
    CandidoChargeAdvanceReport3D& report) {
  bool useWeights = redistributionWeights != nullptr;
  if (useWeights) {
    fvm::require(redistributionWeights->size() == rhoE.size(),
                 "Candido charge redistribution weights size mismatch");
  }
  for (int iter = 0; iter < 8; ++iter) {
    const double current = candidoIntegratedCharge3D(mesh, rhoE);
    const double deficit = chargeConservationTarget - current;
    if (std::abs(deficit) <=
        1e-12 * std::max(std::abs(chargeConservationTarget), 1.0)) {
      break;
    }
    report.redistributionDeficitL1 += std::abs(deficit);
    double capacity = 0.0;
    int weightedCells = 0;
    for (size_t ci = 0; ci < rhoE.size(); ++ci) {
      const double capDensity =
          deficit > 0.0 ? (qLimit - rhoE[ci]) : (rhoE[ci] + qLimit);
      if (capDensity <= 1e-30) continue;
      const double w =
          useWeights ? std::max((*redistributionWeights)[ci], 0.0) : 1.0;
      if (w <= 1e-30) continue;
      capacity += capDensity * mesh.cells[ci].V * w;
      ++weightedCells;
    }
    if (capacity <= 1e-30 && useWeights) {
      useWeights = false;
      --iter;
      continue;
    }
    if (capacity <= 1e-30) break;
    report.maxRedistributionWeightedCapacity =
        std::max(report.maxRedistributionWeightedCapacity, capacity);
    report.maxRedistributionWeightedCells =
        std::max(report.maxRedistributionWeightedCells, weightedCells);
    for (size_t ci = 0; ci < rhoE.size(); ++ci) {
      const double capDensity =
          deficit > 0.0 ? (qLimit - rhoE[ci]) : (rhoE[ci] + qLimit);
      if (capDensity <= 1e-30) continue;
      const double w =
          useWeights ? std::max((*redistributionWeights)[ci], 0.0) : 1.0;
      if (w <= 1e-30) continue;
      const double deltaDensity = deficit * capDensity * w / capacity;
      rhoE[ci] = std::clamp(rhoE[ci] + deltaDensity, -qLimit, qLimit);
    }
  }
}

inline CandidoChargeAdvanceReport3D candidoAdvanceCharge3D(
    const fvm::Mesh3D& mesh,
    fvm::ScalarField& rhoE,
    const fvm::ScalarField& faceFlux,
    const fvm::ScalarField& conductiveFlux,
    double dt,
    double qLimit,
    bool conservativeBounding = false,
    bool useBoundaryChargeAdvection = false,
    const fvm::ScalarField* redistributionWeights = nullptr) {
  CandidoChargeAdvanceReport3D report;
  report.initialCharge = candidoIntegratedCharge3D(mesh, rhoE);
  fvm::ScalarField conv =
      candidoConvectiveChargeFlux3D(mesh, rhoE, faceFlux, useBoundaryChargeAdvection);
  for (int fi = 0; fi < static_cast<int>(conv.size()); ++fi) {
    if (!mesh.faces[fi].internal()) {
      report.boundaryConvectiveChargeFlux += conv[fi];
      report.boundaryConductiveChargeFlux += conductiveFlux[fi];
      report.maxAbsConductiveCurrent =
          std::max(report.maxAbsConductiveCurrent, std::abs(conductiveFlux[fi]));
      const int patch = mesh.faces[fi].patch;
      if (patch >= 0 && patch < static_cast<int>(report.boundaryConductiveByPatch.size())) {
        report.boundaryConductiveByPatch[patch] += conductiveFlux[fi];
      }
    }
    conv[fi] += conductiveFlux[fi];
  }
  fvm::ScalarField div = fvm::explicitDivFaceFlux3D(mesh, conv);
  for (size_t ci = 0; ci < rhoE.size(); ++ci) {
    const double unclamped = rhoE[ci] - dt * div[ci];
    const double clamped = std::clamp(unclamped, -qLimit, qLimit);
    report.unclampedCharge += unclamped * mesh.cells[ci].V;
    report.maxUnclampedAbsCharge = std::max(report.maxUnclampedAbsCharge, std::abs(unclamped));
    if (clamped != unclamped) {
      ++report.clampedCells;
      report.clampCorrectionL1 += std::abs(clamped - unclamped) * mesh.cells[ci].V;
    }
    rhoE[ci] = clamped;
  }
  const double chargeConservationTarget =
      report.initialCharge -
      dt * (report.boundaryConvectiveChargeFlux + report.boundaryConductiveChargeFlux);
  if (conservativeBounding) {
    candidoRedistributeChargeDeficit3D(mesh, rhoE, qLimit,
                                       chargeConservationTarget,
                                       redistributionWeights, report);
  }
  report.finalCharge = candidoIntegratedCharge3D(mesh, rhoE);
  report.redistributionResidual = report.finalCharge - chargeConservationTarget;
  return report;
}

inline void candidoAddPotentialOperator3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& coeffCell,
    const fvm::PotentialBoundary3D& bc,
    double factor,
    std::vector<fvm::Triplet>& trips,
    Eigen::VectorXd& boundaryRhs,
    const fvm::ScalarField* faceScale = nullptr) {
  if (faceScale != nullptr) {
    fvm::require(faceScale->size() == mesh.faces.size(),
                 "Candido potential operator face-scale size mismatch");
  }
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double scale = faceScale == nullptr ? 1.0 : (*faceScale)[fi];
    if (scale == 0.0) continue;
    const double coeffF =
        f.internal() ? fvm::harmonicMean(coeffCell[f.owner], coeffCell[f.neighbour])
                     : coeffCell[f.owner];
    const double coeff = scale * factor * coeffF * std::abs(f.Sf.dot(f.d)) /
                         std::max(f.d.squaredNorm(), 1e-30);
    if (coeff == 0.0) continue;
    if (f.internal()) {
      trips.emplace_back(f.owner, f.owner, coeff);
      trips.emplace_back(f.owner, f.neighbour, -coeff);
      trips.emplace_back(f.neighbour, f.neighbour, coeff);
      trips.emplace_back(f.neighbour, f.owner, -coeff);
    } else if (!bc.faceDirichlet.empty() && bc.faceDirichlet[fi]) {
      trips.emplace_back(f.owner, f.owner, coeff);
      boundaryRhs[f.owner] += coeff * (bc.faceValue.empty() ? 0.0 : bc.faceValue[fi]);
    }
  }
}

inline CandidoChargeAdvanceReport3D candidoAdvanceChargeImplicitOhmic3D(
    const fvm::Mesh3D& mesh,
    fvm::ScalarField& rhoE,
    const fvm::ScalarField& faceFlux,
    const fvm::ScalarField& eps,
    const fvm::ScalarField& sigmaE,
    const fvm::PotentialBoundary3D& bc,
    double dt,
    double qLimit,
    bool conservativeBounding = false,
    bool useBoundaryChargeAdvection = false,
    const fvm::ScalarField* redistributionWeights = nullptr,
    const fvm::ScalarField* conductiveFaceScale = nullptr) {
  if (conductiveFaceScale != nullptr) {
    fvm::require(conductiveFaceScale->size() == mesh.faces.size(),
                 "Candido implicit Ohmic conductive face-scale size mismatch");
  }
  CandidoChargeAdvanceReport3D report;
  report.initialCharge = candidoIntegratedCharge3D(mesh, rhoE);

  fvm::ScalarField conv =
      candidoConvectiveChargeFlux3D(mesh, rhoE, faceFlux, useBoundaryChargeAdvection);
  for (int fi = 0; fi < static_cast<int>(conv.size()); ++fi) {
    if (mesh.faces[fi].internal()) continue;
    report.boundaryConvectiveChargeFlux += conv[fi];
  }
  const fvm::ScalarField convDiv = fvm::explicitDivFaceFlux3D(mesh, conv);
  fvm::ScalarField qStar(rhoE.size(), 0.0);
  for (size_t ci = 0; ci < rhoE.size(); ++ci) qStar[ci] = rhoE[ci] - dt * convDiv[ci];

  const int n = static_cast<int>(mesh.cells.size());
  std::vector<fvm::Triplet> trips;
  trips.reserve(mesh.faces.size() * 8);
  Eigen::VectorXd epsBoundary = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
  candidoAddPotentialOperator3D(mesh, eps, bc, 1.0, trips, epsBoundary);
  rhs += epsBoundary;
  Eigen::VectorXd sigmaBoundary = Eigen::VectorXd::Zero(n);
  candidoAddPotentialOperator3D(mesh, sigmaE, bc, dt, trips, sigmaBoundary,
                                conductiveFaceScale);
  rhs += sigmaBoundary;
  for (int ci = 0; ci < n; ++ci) rhs[ci] += qStar[ci] * mesh.cells[ci].V;

  fvm::SpMat A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  Eigen::ConjugateGradient<fvm::SpMat, Eigen::Lower | Eigen::Upper,
                           Eigen::IncompleteCholesky<double>> solver;
  solver.setTolerance(1e-11);
  solver.setMaxIterations(4000);
  solver.compute(A);
  fvm::require(solver.info() == Eigen::Success,
               "Candido implicit Ohmic charge projection factorization failed");
  const Eigen::VectorXd phi = solver.solve(rhs);
  fvm::require(solver.info() == Eigen::Success,
               "Candido implicit Ohmic charge projection solve failed");
  report.implicitPotentialResidual =
      (A * phi - rhs).norm() / std::max(rhs.norm(), 1e-30);

  fvm::ScalarField phiField(mesh.cells.size(), 0.0);
  for (int ci = 0; ci < n; ++ci) phiField[ci] = phi[ci];
  const fvm::ScalarField epsFlux =
      candidoPoissonFaceNormalFlux3D(mesh, eps, phiField, bc);
  fvm::ScalarField sigmaFlux =
      candidoPoissonFaceNormalFlux3D(mesh, sigmaE, phiField, bc);
  if (conductiveFaceScale != nullptr) {
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      sigmaFlux[fi] *= (*conductiveFaceScale)[fi];
    }
  }
  const fvm::ScalarField qNew = fvm::explicitDivFaceFlux3D(mesh, epsFlux);

  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    report.maxAbsConductiveCurrent =
        std::max(report.maxAbsConductiveCurrent, std::abs(sigmaFlux[fi]));
    if (mesh.faces[fi].internal()) continue;
    report.boundaryConductiveChargeFlux += sigmaFlux[fi];
    const int patch = mesh.faces[fi].patch;
    if (patch >= 0 && patch < static_cast<int>(report.boundaryConductiveByPatch.size())) {
      report.boundaryConductiveByPatch[patch] += sigmaFlux[fi];
    }
  }

  for (size_t ci = 0; ci < rhoE.size(); ++ci) {
    const double unclamped = qNew[ci];
    const double clamped = std::clamp(unclamped, -qLimit, qLimit);
    report.unclampedCharge += unclamped * mesh.cells[ci].V;
    report.maxUnclampedAbsCharge =
        std::max(report.maxUnclampedAbsCharge, std::abs(unclamped));
    if (clamped != unclamped) {
      ++report.clampedCells;
      report.clampCorrectionL1 += std::abs(clamped - unclamped) * mesh.cells[ci].V;
    }
    rhoE[ci] = clamped;
  }
  const double chargeConservationTarget =
      report.initialCharge -
      dt * (report.boundaryConvectiveChargeFlux + report.boundaryConductiveChargeFlux);
  if (conservativeBounding) {
    candidoRedistributeChargeDeficit3D(mesh, rhoE, qLimit,
                                       chargeConservationTarget,
                                       redistributionWeights, report);
  }
  report.finalCharge = candidoIntegratedCharge3D(mesh, rhoE);
  report.redistributionResidual = report.finalCharge - chargeConservationTarget;
  return report;
}

inline double candidoLiquidTipY3D(const fvm::Mesh3D& mesh,
                                  const fvm::ScalarField& alpha) {
  std::vector<std::pair<double, double>> yMass;
  yMass.reserve(mesh.cells.size());
  double total = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double w = std::clamp(alpha[ci], 0.0, 1.0) * mesh.cells[ci].V;
    if (w <= 1e-30) continue;
    yMass.push_back({mesh.cells[ci].centroid.y(), w});
    total += w;
  }
  if (yMass.empty()) return 0.0;
  std::sort(yMass.begin(), yMass.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  const double target = 0.995 * total;
  double accum = 0.0;
  for (const auto& [y, w] : yMass) {
    accum += w;
    if (accum >= target) return y;
  }
  return yMass.back().first;
}

inline double candidoLiquidCentroidY3D(const fvm::Mesh3D& mesh,
                                       const fvm::ScalarField& alpha) {
  double m = 0.0;
  double y = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double w = std::clamp(alpha[ci], 0.0, 1.0) * mesh.cells[ci].V;
    m += w;
    y += w * mesh.cells[ci].centroid.y();
  }
  return y / std::max(m, 1e-30);
}

inline double candidoInterfaceRadialAsymmetry3D(const fvm::Mesh3D& mesh,
                                                const fvm::ScalarField& alpha,
                                                const CandidoConeJetSmokeOptions3D& opt) {
  const double cx0 = 0.5 * opt.radialWindowOuterDiameters;
  const double cz0 = 0.5 * opt.radialWindowOuterDiameters;
  double wsum = 0.0;
  double cx = 0.0;
  double cz = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    const double w = a * (1.0 - a) * mesh.cells[ci].V;
    wsum += w;
    cx += w * mesh.cells[ci].centroid.x();
    cz += w * mesh.cells[ci].centroid.z();
  }
  if (wsum <= 1e-30) return 0.0;
  cx /= wsum;
  cz /= wsum;
  return std::hypot(cx - cx0, cz - cz0);
}

inline std::pair<double, double> candidoAxialWaveAsymmetryPeak3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const int bins = std::max(3, opt.ny);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double cx0 = 0.5 * opt.radialWindowOuterDiameters;
  const double cz0 = 0.5 * opt.radialWindowOuterDiameters;
  std::vector<double> wsum(static_cast<size_t>(bins), 0.0);
  std::vector<double> xsum(static_cast<size_t>(bins), 0.0);
  std::vector<double> zsum(static_cast<size_t>(bins), 0.0);
  std::vector<double> ysum(static_cast<size_t>(bins), 0.0);
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    const double w = a * (1.0 - a) * mesh.cells[ci].V;
    if (w <= 1e-30) continue;
    int b = static_cast<int>(std::floor(mesh.cells[ci].centroid.y() / std::max(ly, 1e-30) * bins));
    b = std::clamp(b, 0, bins - 1);
    wsum[static_cast<size_t>(b)] += w;
    xsum[static_cast<size_t>(b)] += w * mesh.cells[ci].centroid.x();
    zsum[static_cast<size_t>(b)] += w * mesh.cells[ci].centroid.z();
    ysum[static_cast<size_t>(b)] += w * mesh.cells[ci].centroid.y();
  }
  double bestAsym = 0.0;
  double bestY = 0.0;
  for (int b = 0; b < bins; ++b) {
    const double w = wsum[static_cast<size_t>(b)];
    if (w <= 1e-30) continue;
    const double cx = xsum[static_cast<size_t>(b)] / w;
    const double cz = zsum[static_cast<size_t>(b)] / w;
    const double asym = std::hypot(cx - cx0, cz - cz0);
    if (asym > bestAsym) {
      bestAsym = asym;
      bestY = ysum[static_cast<size_t>(b)] / w;
    }
  }
  return {bestY / std::max(setup.innerDiameter / setup.outerDiameter, 1e-30), bestAsym};
}

inline double candidoCrossSectionConvectiveCurrent3D(const fvm::Mesh3D& mesh,
                                                     const fvm::ScalarField& rhoE,
                                                     const fvm::VectorField3& u,
                                                     double planeY,
                                                     double slabWidth) {
  double current = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    if (std::abs(mesh.cells[ci].centroid.y() - planeY) <= 0.5 * slabWidth) {
      current += rhoE[ci] * u[ci].y() * mesh.cells[ci].V / std::max(slabWidth, 1e-30);
    }
  }
  return current;
}

inline double candidoCrossSectionLiquidJetConvectiveCurrent3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const fvm::ScalarField& rhoE,
    const fvm::VectorField3& u,
    double planeY,
    double slabWidth,
    bool alpha05Mask = false) {
  double current = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    if (std::abs(mesh.cells[ci].centroid.y() - planeY) > 0.5 * slabWidth) continue;
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    const double weight = alpha05Mask ? (a >= 0.5 ? 1.0 : 0.0) : a;
    current += weight * rhoE[ci] * u[ci].y() * mesh.cells[ci].V /
               std::max(slabWidth, 1e-30);
  }
  return current;
}

inline double candidoCrossSectionLiquidAreaInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    double planeY,
    double slabWidth,
    bool alpha05Mask = false) {
  double areaOuterDiameterUnits = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    if (std::abs(mesh.cells[ci].centroid.y() - planeY) > 0.5 * slabWidth) continue;
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    const double weight = alpha05Mask ? (a >= 0.5 ? 1.0 : 0.0) : a;
    areaOuterDiameterUnits += weight * mesh.cells[ci].V / std::max(slabWidth, 1e-30);
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return areaOuterDiameterUnits * outerToInner * outerToInner;
}

struct CandidoAxialCurrentScan3D {
  double yOverDi = 0.0;
  double alpha05AreaDi2 = 0.0;
  double convectiveCurrent = 0.0;
  double liquidConvectiveCurrent = 0.0;
  double alpha05ConvectiveCurrent = 0.0;
  double totalCurrent = 0.0;
  double liquidTotalCurrent = 0.0;
  double alpha05TotalCurrent = 0.0;
  double alpha05ConductiveCurrent = 0.0;
  double meanAlpha05Charge = 0.0;
  double meanAlpha05AbsCharge = 0.0;
  double meanAlpha05Uy = 0.0;
  double meanAlpha05AbsUy = 0.0;
  double alpha05CurrentShapeFactor = 0.0;
};

inline CandidoAxialCurrentScan3D candidoAxialDevelopedJetCurrentScan3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const fvm::ScalarField& rhoE,
    const fvm::VectorField3& u,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt,
    const fvm::ScalarField* sigmaE = nullptr,
    const fvm::VectorField3* E = nullptr) {
  const int bins = std::max(1, opt.ny);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double dy = ly / static_cast<double>(bins);
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  std::vector<double> area(static_cast<size_t>(bins), 0.0);
  std::vector<double> alpha05AreaOuter(static_cast<size_t>(bins), 0.0);
  std::vector<double> conv(static_cast<size_t>(bins), 0.0);
  std::vector<double> liquid(static_cast<size_t>(bins), 0.0);
  std::vector<double> alpha05Current(static_cast<size_t>(bins), 0.0);
  std::vector<double> total(static_cast<size_t>(bins), 0.0);
  std::vector<double> liquidTotal(static_cast<size_t>(bins), 0.0);
  std::vector<double> alpha05Total(static_cast<size_t>(bins), 0.0);
  std::vector<double> alpha05Conductive(static_cast<size_t>(bins), 0.0);
  std::vector<double> alpha05Charge(static_cast<size_t>(bins), 0.0);
  std::vector<double> alpha05AbsCharge(static_cast<size_t>(bins), 0.0);
  std::vector<double> alpha05Uy(static_cast<size_t>(bins), 0.0);
  std::vector<double> alpha05AbsUy(static_cast<size_t>(bins), 0.0);
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    int b = static_cast<int>(std::floor(mesh.cells[ci].centroid.y() /
                                        std::max(ly, 1e-30) * bins));
    b = std::clamp(b, 0, bins - 1);
    const size_t bi = static_cast<size_t>(b);
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    const double alpha05 = a >= 0.5 ? 1.0 : 0.0;
    const double cellArea = mesh.cells[ci].V / std::max(dy, 1e-30);
    const double convectiveCurrent = rhoE[ci] * u[ci].y();
    double conductiveCurrent = 0.0;
    if (sigmaE != nullptr && E != nullptr && ci < sigmaE->size() && ci < E->size()) {
      conductiveCurrent = (*sigmaE)[ci] * (*E)[ci].y();
    }
    const double current = convectiveCurrent * cellArea;
    const double conductive = conductiveCurrent * cellArea;
    const double totalCurrent = current + conductive;
    area[bi] += alpha05 * cellArea * outerToInner * outerToInner;
    alpha05AreaOuter[bi] += alpha05 * cellArea;
    conv[bi] += current;
    liquid[bi] += a * current;
    alpha05Current[bi] += alpha05 * current;
    total[bi] += totalCurrent;
    liquidTotal[bi] += a * totalCurrent;
    alpha05Total[bi] += alpha05 * totalCurrent;
    alpha05Conductive[bi] += alpha05 * conductive;
    alpha05Charge[bi] += alpha05 * rhoE[ci] * cellArea;
    alpha05AbsCharge[bi] += alpha05 * std::abs(rhoE[ci]) * cellArea;
    alpha05Uy[bi] += alpha05 * u[ci].y() * cellArea;
    alpha05AbsUy[bi] += alpha05 * std::abs(u[ci].y()) * cellArea;
  }
  int best = 0;
  for (int b = 1; b < bins; ++b) {
    if (area[static_cast<size_t>(b)] > area[static_cast<size_t>(best)]) best = b;
  }
  const size_t bi = static_cast<size_t>(best);
  CandidoAxialCurrentScan3D scan;
  scan.yOverDi = (static_cast<double>(best) + 0.5) * dy * outerToInner;
  scan.alpha05AreaDi2 = area[bi];
  scan.convectiveCurrent = std::abs(conv[bi]);
  scan.liquidConvectiveCurrent = std::abs(liquid[bi]);
  scan.alpha05ConvectiveCurrent = std::abs(alpha05Current[bi]);
  scan.totalCurrent = std::abs(total[bi]);
  scan.liquidTotalCurrent = std::abs(liquidTotal[bi]);
  scan.alpha05TotalCurrent = std::abs(alpha05Total[bi]);
  scan.alpha05ConductiveCurrent = std::abs(alpha05Conductive[bi]);
  const double areaOuter = std::max(alpha05AreaOuter[bi], 0.0);
  if (areaOuter > 1e-30) {
    scan.meanAlpha05Charge = alpha05Charge[bi] / areaOuter;
    scan.meanAlpha05AbsCharge = alpha05AbsCharge[bi] / areaOuter;
    scan.meanAlpha05Uy = alpha05Uy[bi] / areaOuter;
    scan.meanAlpha05AbsUy = alpha05AbsUy[bi] / areaOuter;
    const double productScale =
        areaOuter * scan.meanAlpha05AbsCharge * scan.meanAlpha05AbsUy;
    scan.alpha05CurrentShapeFactor =
        std::abs(alpha05Current[bi]) / std::max(productScale, 1e-300);
  }
  return scan;
}

struct CandidoAxialMomentumSourceScan3D {
  double yOverDi = 0.0;
  double alpha05AreaDi2 = 0.0;
  double meanAbsUy = 0.0;
  double meanAbsElectricMomentumSourceY = 0.0;
  double meanAbsSurfaceMomentumSourceY = 0.0;
  double meanAbsMomentumSourceY = 0.0;
  double meanAbsMomentumAccelerationY = 0.0;
};

inline CandidoAxialMomentumSourceScan3D candidoAxialMomentumSourceScan3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const fvm::ScalarField& rho,
    const fvm::VectorField3& u,
    const fvm::VectorField3& electricForce,
    double electricDriveScale,
    const fvm::VectorField3& surfaceForce,
    double surfaceDriveScale,
    const fvm::VectorField3& source,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const int bins = std::max(1, opt.ny);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double dy = ly / static_cast<double>(bins);
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  std::vector<double> area(static_cast<size_t>(bins), 0.0);
  std::vector<double> areaOuter(static_cast<size_t>(bins), 0.0);
  std::vector<double> absUy(static_cast<size_t>(bins), 0.0);
  std::vector<double> absElectricSource(static_cast<size_t>(bins), 0.0);
  std::vector<double> absSurfaceSource(static_cast<size_t>(bins), 0.0);
  std::vector<double> absSource(static_cast<size_t>(bins), 0.0);
  std::vector<double> absAcceleration(static_cast<size_t>(bins), 0.0);
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    int b = static_cast<int>(std::floor(mesh.cells[ci].centroid.y() /
                                        std::max(ly, 1e-30) * bins));
    b = std::clamp(b, 0, bins - 1);
    const size_t bi = static_cast<size_t>(b);
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a < 0.5) continue;
    const double cellArea = mesh.cells[ci].V / std::max(dy, 1e-30);
    const double electricY = -electricDriveScale * electricForce[ci].y();
    const double surfaceY = surfaceDriveScale * surfaceForce[ci].y();
    const double sourceY = source[ci].y();
    const double rhoCell = ci < rho.size() ? std::max(rho[ci], 1e-30) : 1.0;
    area[bi] += cellArea * outerToInner * outerToInner;
    areaOuter[bi] += cellArea;
    absUy[bi] += std::abs(u[ci].y()) * cellArea;
    absElectricSource[bi] += std::abs(electricY) * cellArea;
    absSurfaceSource[bi] += std::abs(surfaceY) * cellArea;
    absSource[bi] += std::abs(sourceY) * cellArea;
    absAcceleration[bi] += std::abs(sourceY / rhoCell) * cellArea;
  }
  int best = 0;
  for (int b = 1; b < bins; ++b) {
    if (area[static_cast<size_t>(b)] > area[static_cast<size_t>(best)]) best = b;
  }
  const size_t bi = static_cast<size_t>(best);
  CandidoAxialMomentumSourceScan3D scan;
  scan.yOverDi = (static_cast<double>(best) + 0.5) * dy * outerToInner;
  scan.alpha05AreaDi2 = area[bi];
  const double denom = areaOuter[bi];
  if (denom > 1e-30) {
    scan.meanAbsUy = absUy[bi] / denom;
    scan.meanAbsElectricMomentumSourceY = absElectricSource[bi] / denom;
    scan.meanAbsSurfaceMomentumSourceY = absSurfaceSource[bi] / denom;
    scan.meanAbsMomentumSourceY = absSource[bi] / denom;
    scan.meanAbsMomentumAccelerationY = absAcceleration[bi] / denom;
  }
  return scan;
}

inline double candidoCrossSectionTotalCurrent3D(const fvm::Mesh3D& mesh,
                                                const fvm::ScalarField& rhoE,
                                                const fvm::VectorField3& u,
                                                const fvm::ScalarField& sigmaE,
                                                const fvm::VectorField3& E,
                                                double planeY,
                                                double slabWidth) {
  double current = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    if (std::abs(mesh.cells[ci].centroid.y() - planeY) <= 0.5 * slabWidth) {
      const double jy = rhoE[ci] * u[ci].y() + sigmaE[ci] * E[ci].y();
      current += jy * mesh.cells[ci].V / std::max(slabWidth, 1e-30);
    }
  }
  return current;
}

struct CandidoPoissonFaceCrossSectionCurrent3D {
  double convectiveCurrent = 0.0;
  double conductiveCurrent = 0.0;
  double totalCurrent = 0.0;
  int sampledFaces = 0;
};

inline double candidoFaceAlpha3D(const fvm::Mesh3D& mesh,
                                 const fvm::ScalarField& alpha,
                                 int faceIndex) {
  const fvm::Face3D& f = mesh.faces[faceIndex];
  if (f.internal()) {
    return 0.5 * (std::clamp(alpha[f.owner], 0.0, 1.0) +
                  std::clamp(alpha[f.neighbour], 0.0, 1.0));
  }
  return std::clamp(alpha[f.owner], 0.0, 1.0);
}

inline CandidoPoissonFaceCrossSectionCurrent3D
candidoPoissonFaceCrossSectionCurrent3D(const fvm::Mesh3D& mesh,
                                        const fvm::ScalarField& alpha,
                                        const fvm::ScalarField& rhoE,
                                        const fvm::ScalarField& faceFlux,
                                        const fvm::ScalarField& sigmaE,
                                        const fvm::ScalarField& phi,
                                        const fvm::PotentialBoundary3D& bc,
                                        double planeY,
                                        double slabWidth,
                                        bool alpha05Mask = false,
                                        bool useBoundaryChargeAdvection = false) {
  const fvm::ScalarField convectiveFlux =
      candidoConvectiveChargeFlux3D(mesh, rhoE, faceFlux,
                                    useBoundaryChargeAdvection);
  const fvm::ScalarField conductiveFlux =
      candidoPoissonFaceConductiveCurrentFlux3D(mesh, sigmaE, phi, bc);
  CandidoPoissonFaceCrossSectionCurrent3D current;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (std::abs(f.centroid.y() - planeY) > 0.5 * slabWidth) continue;
    const double area = f.Sf.norm();
    if (area <= 1e-30) continue;
    const double yFraction = std::abs(f.Sf.y()) / area;
    if (yFraction < 0.5) continue;
    const double aFace = candidoFaceAlpha3D(mesh, alpha, fi);
    const double mask = alpha05Mask ? (aFace >= 0.5 ? 1.0 : 0.0) : 1.0;
    if (mask == 0.0) continue;
    const double orientToPositiveY = f.Sf.y() >= 0.0 ? 1.0 : -1.0;
    current.convectiveCurrent += mask * orientToPositiveY * convectiveFlux[fi];
    current.conductiveCurrent += mask * orientToPositiveY * conductiveFlux[fi];
    ++current.sampledFaces;
  }
  current.totalCurrent = current.convectiveCurrent + current.conductiveCurrent;
  current.convectiveCurrent = std::abs(current.convectiveCurrent);
  current.conductiveCurrent = std::abs(current.conductiveCurrent);
  current.totalCurrent = std::abs(current.totalCurrent);
  return current;
}

struct CandidoPoissonFaceAxialCurrentScan3D {
  double yOverDi = 0.0;
  double alpha05AreaDi2 = 0.0;
  double alpha05ConvectiveCurrent = 0.0;
  double alpha05ConductiveCurrent = 0.0;
  double alpha05TotalCurrent = 0.0;
  double meanAbsUpwindCharge = 0.0;
  double meanAbsFaceFlux = 0.0;
  double meanAbsConvectiveFlux = 0.0;
  double maxAbsUpwindCharge = 0.0;
  double maxAbsFaceFlux = 0.0;
  int sampledFaces = 0;
};

inline CandidoPoissonFaceAxialCurrentScan3D
candidoPoissonFaceAxialDevelopedCurrentScan3D(const fvm::Mesh3D& mesh,
                                              const fvm::ScalarField& alpha,
                                              const fvm::ScalarField& rhoE,
                                              const fvm::ScalarField& faceFlux,
                                              const fvm::ScalarField& sigmaE,
                                              const fvm::ScalarField& phi,
                                              const fvm::PotentialBoundary3D& bc,
                                              const CandidoTaylorConeJetSetup& setup,
                                              const CandidoConeJetSmokeOptions3D& opt,
                                              bool useBoundaryChargeAdvection = false) {
  const int bins = std::max(1, opt.ny);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double dy = ly / static_cast<double>(bins);
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  std::vector<double> area(static_cast<size_t>(bins), 0.0);
  std::vector<double> conv(static_cast<size_t>(bins), 0.0);
  std::vector<double> cond(static_cast<size_t>(bins), 0.0);
  std::vector<double> absUpwindCharge(static_cast<size_t>(bins), 0.0);
  std::vector<double> absFaceFlux(static_cast<size_t>(bins), 0.0);
  std::vector<double> absConvectiveFlux(static_cast<size_t>(bins), 0.0);
  std::vector<double> maxAbsUpwindCharge(static_cast<size_t>(bins), 0.0);
  std::vector<double> maxAbsFaceFlux(static_cast<size_t>(bins), 0.0);
  std::vector<int> sampled(static_cast<size_t>(bins), 0);
  const fvm::ScalarField convectiveFlux =
      candidoConvectiveChargeFlux3D(mesh, rhoE, faceFlux,
                                    useBoundaryChargeAdvection);
  const fvm::ScalarField conductiveFlux =
      candidoPoissonFaceConductiveCurrentFlux3D(mesh, sigmaE, phi, bc);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double faceArea = f.Sf.norm();
    if (faceArea <= 1e-30) continue;
    const double yFraction = std::abs(f.Sf.y()) / faceArea;
    if (yFraction < 0.5) continue;
    const double aFace = candidoFaceAlpha3D(mesh, alpha, fi);
    if (aFace < 0.5) continue;
    int b = static_cast<int>(std::floor(f.centroid.y() / std::max(ly, 1e-30) * bins));
    b = std::clamp(b, 0, bins - 1);
    const size_t bi = static_cast<size_t>(b);
    const double orientToPositiveY = f.Sf.y() >= 0.0 ? 1.0 : -1.0;
    double qFace = 0.0;
    if (f.internal()) {
      const int upwindCell = faceFlux[fi] >= 0.0 ? f.owner : f.neighbour;
      qFace = rhoE[upwindCell];
    } else if (useBoundaryChargeAdvection) {
      qFace = faceFlux[fi] >= 0.0 ? rhoE[f.owner] : 0.0;
    }
    area[bi] += std::abs(f.Sf.y()) * outerToInner * outerToInner;
    conv[bi] += orientToPositiveY * convectiveFlux[fi];
    cond[bi] += orientToPositiveY * conductiveFlux[fi];
    absUpwindCharge[bi] += std::abs(qFace);
    absFaceFlux[bi] += std::abs(faceFlux[fi]);
    absConvectiveFlux[bi] += std::abs(convectiveFlux[fi]);
    maxAbsUpwindCharge[bi] =
        std::max(maxAbsUpwindCharge[bi], std::abs(qFace));
    maxAbsFaceFlux[bi] = std::max(maxAbsFaceFlux[bi], std::abs(faceFlux[fi]));
    ++sampled[bi];
  }
  int best = 0;
  for (int b = 1; b < bins; ++b) {
    if (area[static_cast<size_t>(b)] > area[static_cast<size_t>(best)]) best = b;
  }
  const size_t bi = static_cast<size_t>(best);
  CandidoPoissonFaceAxialCurrentScan3D scan;
  scan.yOverDi = (static_cast<double>(best) + 0.5) * dy * outerToInner;
  scan.alpha05AreaDi2 = area[bi];
  scan.alpha05ConvectiveCurrent = std::abs(conv[bi]);
  scan.alpha05ConductiveCurrent = std::abs(cond[bi]);
  scan.alpha05TotalCurrent = std::abs(conv[bi] + cond[bi]);
  scan.sampledFaces = sampled[bi];
  if (scan.sampledFaces > 0) {
    const double invSampled = 1.0 / static_cast<double>(scan.sampledFaces);
    scan.meanAbsUpwindCharge = absUpwindCharge[bi] * invSampled;
    scan.meanAbsFaceFlux = absFaceFlux[bi] * invSampled;
    scan.meanAbsConvectiveFlux = absConvectiveFlux[bi] * invSampled;
    scan.maxAbsUpwindCharge = maxAbsUpwindCharge[bi];
    scan.maxAbsFaceFlux = maxAbsFaceFlux[bi];
  }
  return scan;
}

struct CandidoGaussLawCellGradientAudit3D {
  double maxAbsResidual = 0.0;
  double relativeL2Residual = 0.0;
};

inline CandidoGaussLawCellGradientAudit3D candidoGaussLawCellGradientAudit3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& eps,
    const fvm::VectorField3& E,
    const fvm::ScalarField& rhoE) {
  fvm::ScalarField epsEFlux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    const double epsF = f.internal() ? fvm::harmonicMean(eps[f.owner], eps[f.neighbour])
                                     : eps[f.owner];
    const fvm::Vec3 ef = f.internal() ? 0.5 * (E[f.owner] + E[f.neighbour]) : E[f.owner];
    epsEFlux[fi] = epsF * ef.dot(f.Sf);
  }
  const fvm::ScalarField gaussCharge = fvm::explicitDivFaceFlux3D(mesh, epsEFlux);
  double maxAbs = 0.0;
  double num = 0.0;
  double rhoNorm = 0.0;
  double gaussNorm = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    const double residual = gaussCharge[ci] - rhoE[ci];
    maxAbs = std::max(maxAbs, std::abs(residual));
    num += residual * residual * mesh.cells[ci].V;
    rhoNorm += rhoE[ci] * rhoE[ci] * mesh.cells[ci].V;
    gaussNorm += gaussCharge[ci] * gaussCharge[ci] * mesh.cells[ci].V;
  }
  return {maxAbs, std::sqrt(num / std::max(std::max(rhoNorm, gaussNorm), 1e-300))};
}

inline double candidoEquivalentLiquidRadiusAtY3D(const fvm::Mesh3D& mesh,
                                                 const fvm::ScalarField& alpha,
                                                 double planeY,
                                                 double slabWidth) {
  double liquidVolumeInSlab = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    if (std::abs(mesh.cells[ci].centroid.y() - planeY) <= 0.5 * slabWidth) {
      liquidVolumeInSlab += std::clamp(alpha[ci], 0.0, 1.0) * mesh.cells[ci].V;
    }
  }
  const double area = liquidVolumeInSlab / std::max(slabWidth, 1e-30);
  return std::sqrt(std::max(area, 0.0) / M_PI);
}

inline double candidoMorphologyVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup) {
  double volumeOuterDiameterUnits = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    volumeOuterDiameterUnits += std::clamp(alpha[ci], 0.0, 1.0) * mesh.cells[ci].V;
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline std::vector<int> candidoInletConnectedLiquidMask3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt,
    double alphaThreshold = 0.05) {
  std::vector<int> connected(mesh.cells.size(), 0);
  std::vector<int> stack;
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double yTol = ly / std::max(opt.ny, 1) * 0.75;
  const double inletRadius = 0.5 * setup.innerDiameter / setup.outerDiameter;
  const fvm::Vec3 axis{0.5 * opt.radialWindowOuterDiameters, 0.0,
                       0.5 * opt.radialWindowOuterDiameters};
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    const fvm::Vec3& c = mesh.cells[ci].centroid;
    const double r = std::hypot(c.x() - axis.x(), c.z() - axis.z());
    if (c.y() <= yTol && r <= 2.0 * inletRadius && alpha[ci] > alphaThreshold) {
      connected[ci] = 1;
      stack.push_back(ci);
    }
  }
  while (!stack.empty()) {
    const int ci = stack.back();
    stack.pop_back();
    for (int fi : mesh.cells[ci].faces) {
      const fvm::Face3D& f = mesh.faces[fi];
      if (!f.internal()) continue;
      const int nb = (f.owner == ci) ? f.neighbour : f.owner;
      if (nb < 0 || connected[nb] || alpha[nb] <= alphaThreshold) continue;
      connected[nb] = 1;
      stack.push_back(nb);
    }
  }
  return connected;
}

inline double candidoConnectedMorphologyVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt);
  double volumeOuterDiameterUnits = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    if (connected[ci]) volumeOuterDiameterUnits += std::clamp(alpha[ci], 0.0, 1.0) * mesh.cells[ci].V;
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoAlpha05AxialSilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const double dy = ly / static_cast<double>(bins);
  std::vector<double> maxRadius(bins, 0.0);
  const fvm::Vec3 axis{0.5 * opt.radialWindowOuterDiameters, 0.0,
                       0.5 * opt.radialWindowOuterDiameters};
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    if (!connected[ci] || alpha[ci] < 0.5) continue;
    const fvm::Vec3& c = mesh.cells[ci].centroid;
    const int bi = std::clamp(static_cast<int>(std::floor(c.y() / std::max(dy, 1e-30))),
                              0, bins - 1);
    const double r = std::hypot(c.x() - axis.x(), c.z() - axis.z());
    maxRadius[bi] = std::max(maxRadius[bi], r);
  }
  double volumeOuterDiameterUnits = 0.0;
  for (double r : maxRadius) {
    volumeOuterDiameterUnits += M_PI * r * r * dy;
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const int rays = 16;
  const double dy = ly / static_cast<double>(bins);
  const double radialWindow = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * radialWindow, 0.0, 0.5 * radialWindow};
  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    const double y0 = (static_cast<double>(bi) + 0.5) * dy;
    double sumRadiusSquared = 0.0;
    int contributingRays = 0;
    for (int ri = 0; ri < rays; ++ri) {
      const double theta = 2.0 * M_PI * (static_cast<double>(ri) + 0.5) /
                           static_cast<double>(rays);
      std::vector<std::pair<double, double>> samples;
      samples.reserve(mesh.cells.size() / std::max(1, bins));
      for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
        if (!connected[ci]) continue;
        const fvm::Vec3& c = mesh.cells[ci].centroid;
        if (std::abs(c.y() - y0) > 0.75 * dy) continue;
        const double dx = c.x() - axis.x();
        const double dz = c.z() - axis.z();
        const double r = std::hypot(dx, dz);
        if (r < 1e-14) {
          samples.push_back({r, std::clamp(alpha[ci], 0.0, 1.0)});
          continue;
        }
        const double a = std::atan2(dz, dx);
        const double angleDistance = std::abs(std::atan2(std::sin(a - theta),
                                                         std::cos(a - theta)));
        if (angleDistance <= M_PI / static_cast<double>(rays)) {
          samples.push_back({r, std::clamp(alpha[ci], 0.0, 1.0)});
        }
      }
      if (samples.empty()) continue;
      std::sort(samples.begin(), samples.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      double radius = 0.0;
      bool foundInside = false;
      double prevR = samples.front().first;
      double prevA = samples.front().second;
      for (const auto& [r, a] : samples) {
        if (a >= 0.5) {
          foundInside = true;
          radius = std::max(radius, r);
        } else if (foundInside && prevA >= 0.5) {
          const double denom = std::max(prevA - a, 1e-30);
          const double t = std::clamp((prevA - 0.5) / denom, 0.0, 1.0);
          radius = prevR + t * (r - prevR);
          break;
        }
        prevR = r;
        prevA = a;
      }
      if (radius > 0.0) {
        sumRadiusSquared += radius * radius;
        ++contributingRays;
      }
    }
    if (contributingRays > 0) {
      volumeOuterDiameterUnits +=
          M_PI * (sumRadiusSquared / static_cast<double>(contributingRays)) * dy;
    }
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoAllLiquidRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const int rays = 16;
  const double dy = ly / static_cast<double>(bins);
  const double radialWindow = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * radialWindow, 0.0, 0.5 * radialWindow};
  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    const double y0 = (static_cast<double>(bi) + 0.5) * dy;
    double sumRadiusSquared = 0.0;
    int contributingRays = 0;
    for (int ri = 0; ri < rays; ++ri) {
      const double theta = 2.0 * M_PI * (static_cast<double>(ri) + 0.5) /
                           static_cast<double>(rays);
      std::vector<std::pair<double, double>> samples;
      samples.reserve(mesh.cells.size() / std::max(1, bins));
      for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
        const double aClamped = std::clamp(alpha[ci], 0.0, 1.0);
        if (aClamped <= 1e-8) continue;
        const fvm::Vec3& c = mesh.cells[ci].centroid;
        if (std::abs(c.y() - y0) > 0.75 * dy) continue;
        const double dx = c.x() - axis.x();
        const double dz = c.z() - axis.z();
        const double r = std::hypot(dx, dz);
        if (r < 1e-14) {
          samples.push_back({r, aClamped});
          continue;
        }
        const double a = std::atan2(dz, dx);
        const double angleDistance = std::abs(std::atan2(std::sin(a - theta),
                                                         std::cos(a - theta)));
        if (angleDistance <= M_PI / static_cast<double>(rays)) {
          samples.push_back({r, aClamped});
        }
      }
      if (samples.empty()) continue;
      std::sort(samples.begin(), samples.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      double radius = 0.0;
      bool foundInside = false;
      double prevR = samples.front().first;
      double prevA = samples.front().second;
      for (const auto& [r, a] : samples) {
        if (a >= 0.5) {
          foundInside = true;
          radius = std::max(radius, r);
        } else if (foundInside && prevA >= 0.5) {
          const double denom = std::max(prevA - a, 1e-30);
          const double t = std::clamp((prevA - 0.5) / denom, 0.0, 1.0);
          radius = prevR + t * (r - prevR);
          break;
        }
        prevR = r;
        prevA = a;
      }
      if (radius > 0.0) {
        sumRadiusSquared += radius * radius;
        ++contributingRays;
      }
    }
    if (contributingRays > 0) {
      volumeOuterDiameterUnits +=
          M_PI * (sumRadiusSquared / static_cast<double>(contributingRays)) * dy;
    }
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoRayAlpha05CellBoundarySilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const int rays = 16;
  const double dy = ly / static_cast<double>(bins);
  const double radialWindow = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * radialWindow, 0.0, 0.5 * radialWindow};
  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    const double y0 = (static_cast<double>(bi) + 0.5) * dy;
    double sumRadiusSquared = 0.0;
    int contributingRays = 0;
    for (int ri = 0; ri < rays; ++ri) {
      const double theta = 2.0 * M_PI * (static_cast<double>(ri) + 0.5) /
                           static_cast<double>(rays);
      const fvm::Vec3 dir{std::cos(theta), 0.0, std::sin(theta)};
      const fvm::Vec3 perp{-std::sin(theta), 0.0, std::cos(theta)};
      double radius = 0.0;
      for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        if (!connected[static_cast<size_t>(ci)] || alpha[static_cast<size_t>(ci)] < 0.5) {
          continue;
        }
        const fvm::Cell3D& cell = mesh.cells[static_cast<size_t>(ci)];
        double minY = std::numeric_limits<double>::infinity();
        double maxY = -std::numeric_limits<double>::infinity();
        double minS = std::numeric_limits<double>::infinity();
        double maxS = -std::numeric_limits<double>::infinity();
        double minT = std::numeric_limits<double>::infinity();
        double maxT = -std::numeric_limits<double>::infinity();
        for (int pi : cell.points) {
          const fvm::Vec3 rel = mesh.points[static_cast<size_t>(pi)] - axis;
          minY = std::min(minY, mesh.points[static_cast<size_t>(pi)].y());
          maxY = std::max(maxY, mesh.points[static_cast<size_t>(pi)].y());
          const double s = rel.dot(dir);
          const double t = rel.dot(perp);
          minS = std::min(minS, s);
          maxS = std::max(maxS, s);
          minT = std::min(minT, t);
          maxT = std::max(maxT, t);
        }
        const double h = std::cbrt(std::max(cell.V, 1e-30));
        if (y0 < minY - 0.25 * h || y0 > maxY + 0.25 * h) continue;
        if (minT > 0.25 * h || maxT < -0.25 * h) continue;
        if (maxS <= 0.0) continue;
        radius = std::max(radius, std::max(0.0, maxS));
      }
      if (radius > 0.0 && std::isfinite(radius)) {
        sumRadiusSquared += radius * radius;
        ++contributingRays;
      }
    }
    if (contributingRays > 0) {
      volumeOuterDiameterUnits +=
          M_PI * (sumRadiusSquared / static_cast<double>(contributingRays)) * dy;
    }
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoLinearRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const fvm::VectorField3 gradAlpha = fvm::gradLeastSquares3D(mesh, alpha);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const int rays = 32;
  const double dy = ly / static_cast<double>(bins);
  const double radialWindow = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * radialWindow, 0.0, 0.5 * radialWindow};
  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    const double y0 = (static_cast<double>(bi) + 0.5) * dy;
    double sumRadiusSquared = 0.0;
    int contributingRays = 0;
    for (int ri = 0; ri < rays; ++ri) {
      const double theta = 2.0 * M_PI * (static_cast<double>(ri) + 0.5) /
                           static_cast<double>(rays);
      const fvm::Vec3 dir{std::cos(theta), 0.0, std::sin(theta)};
      const fvm::Vec3 perp{-std::sin(theta), 0.0, std::cos(theta)};
      std::vector<double> crossings;
      double maxInsideSupport = 0.0;
      for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        if (!connected[static_cast<size_t>(ci)]) continue;
        const fvm::Cell3D& cell = mesh.cells[static_cast<size_t>(ci)];
        double minY = std::numeric_limits<double>::infinity();
        double maxY = -std::numeric_limits<double>::infinity();
        double minS = std::numeric_limits<double>::infinity();
        double maxS = -std::numeric_limits<double>::infinity();
        double minT = std::numeric_limits<double>::infinity();
        double maxT = -std::numeric_limits<double>::infinity();
        for (int pi : cell.points) {
          const fvm::Vec3 rel = mesh.points[static_cast<size_t>(pi)] - axis;
          minY = std::min(minY, mesh.points[static_cast<size_t>(pi)].y());
          maxY = std::max(maxY, mesh.points[static_cast<size_t>(pi)].y());
          const double s = rel.dot(dir);
          const double t = rel.dot(perp);
          minS = std::min(minS, s);
          maxS = std::max(maxS, s);
          minT = std::min(minT, t);
          maxT = std::max(maxT, t);
        }
        const double h = std::cbrt(std::max(cell.V, 1e-30));
        if (y0 < minY - 0.25 * h || y0 > maxY + 0.25 * h) continue;
        if (minT > 0.25 * h || maxT < -0.25 * h) continue;
        if (maxS <= 0.0) continue;
        const double r0 = std::max(0.0, minS);
        const double r1 = std::max(r0, maxS);
        const auto alphaAtRadius = [&](double r) {
          const fvm::Vec3 x = axis + fvm::Vec3{r * dir.x(), y0, r * dir.z()};
          return alpha[static_cast<size_t>(ci)] +
                 gradAlpha[static_cast<size_t>(ci)].dot(x - cell.centroid);
        };
        const double a0 = alphaAtRadius(r0);
        const double a1 = alphaAtRadius(r1);
        if (std::max(a0, a1) >= 0.5) maxInsideSupport = std::max(maxInsideSupport, r1);
        const double f0 = a0 - 0.5;
        const double f1 = a1 - 0.5;
        if (!std::isfinite(f0) || !std::isfinite(f1)) continue;
        if (f0 == 0.0) {
          crossings.push_back(r0);
        } else if (f1 == 0.0) {
          crossings.push_back(r1);
        } else if (f0 * f1 < 0.0) {
          const double t = std::clamp(f0 / (f0 - f1), 0.0, 1.0);
          crossings.push_back(r0 + t * (r1 - r0));
        }
      }
      double radius = 0.0;
      if (!crossings.empty()) {
        radius = *std::max_element(crossings.begin(), crossings.end());
      } else {
        radius = maxInsideSupport;
      }
      if (radius > 0.0 && std::isfinite(radius)) {
        sumRadiusSquared += radius * radius;
        ++contributingRays;
      }
    }
    if (contributingRays > 0) {
      volumeOuterDiameterUnits +=
          M_PI * (sumRadiusSquared / static_cast<double>(contributingRays)) * dy;
    }
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoPlicContourSilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const std::vector<fvm::IsoSurfaceReconstruction3D> iso =
      fvm::reconstructIsoInterface3D(mesh, alpha);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const double dy = ly / static_cast<double>(bins);
  const fvm::Vec3 axis{0.5 * opt.radialWindowOuterDiameters, 0.0,
                       0.5 * opt.radialWindowOuterDiameters};
  std::vector<std::vector<double>> radii(bins);
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    if (!connected[ci] || !iso[ci].mixed) continue;
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    const fvm::Vec3& x = iso[ci].interfaceCentroid;
    const int bi = std::clamp(static_cast<int>(std::floor(x.y() / std::max(dy, 1e-30))),
                              0, bins - 1);
    const double r = std::hypot(x.x() - axis.x(), x.z() - axis.z());
    if (std::isfinite(r) && r > 0.0) radii[bi].push_back(r);
  }

  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    if (radii[bi].empty()) continue;
    std::sort(radii[bi].begin(), radii[bi].end());
    const size_t qi = std::min(radii[bi].size() - 1,
                               static_cast<size_t>(std::floor(0.90 * (radii[bi].size() - 1))));
    const double r = radii[bi][qi];
    volumeOuterDiameterUnits += M_PI * r * r * dy;
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline std::vector<fvm::Vec3> candidoPlicCutPolygonPoints3D(
    const fvm::Mesh3D& mesh,
    int celli,
    const fvm::IsoSurfaceReconstruction3D& iso) {
  std::vector<fvm::Vec3> points;
  const double tol = 1e-10 * std::max(1.0, std::cbrt(std::max(mesh.cells[celli].V, 1e-30)));
  auto addUnique = [&](const fvm::Vec3& x) {
    if (!std::isfinite(x.x()) || !std::isfinite(x.y()) || !std::isfinite(x.z())) return;
    for (const fvm::Vec3& p : points) {
      if ((p - x).norm() <= tol) return;
    }
    points.push_back(x);
  };
  auto signedDistance = [&](const fvm::Vec3& x) {
    return iso.normal.dot(x - mesh.cells[celli].centroid) - iso.cut;
  };
  for (int fi : mesh.cells[celli].faces) {
    const fvm::Face3D& face = mesh.faces[fi];
    for (size_t i = 0; i < face.points.size(); ++i) {
      const fvm::Vec3& a = mesh.points[face.points[i]];
      const fvm::Vec3& b = mesh.points[face.points[(i + 1) % face.points.size()]];
      const double sa = signedDistance(a);
      const double sb = signedDistance(b);
      if (std::abs(sa) <= tol) addUnique(a);
      if (std::abs(sb) <= tol) addUnique(b);
      if (sa * sb < 0.0) {
        const double t = std::clamp(sa / (sa - sb), 0.0, 1.0);
        addUnique(a + t * (b - a));
      }
    }
  }
  return points;
}

inline double candidoPlicPolygonSilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const std::vector<fvm::IsoSurfaceReconstruction3D> iso =
      fvm::reconstructIsoInterface3D(mesh, alpha);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const double dy = ly / static_cast<double>(bins);
  const fvm::Vec3 axis{0.5 * opt.radialWindowOuterDiameters, 0.0,
                       0.5 * opt.radialWindowOuterDiameters};
  std::vector<std::vector<double>> radii(bins);
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    if (!connected[ci] || !iso[ci].mixed) continue;
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    const std::vector<fvm::Vec3> cutPoints = candidoPlicCutPolygonPoints3D(mesh, ci, iso[ci]);
    for (const fvm::Vec3& x : cutPoints) {
      const int bi = std::clamp(static_cast<int>(std::floor(x.y() / std::max(dy, 1e-30))),
                                0, bins - 1);
      const double r = std::hypot(x.x() - axis.x(), x.z() - axis.z());
      if (std::isfinite(r) && r > 0.0) radii[bi].push_back(r);
    }
  }

  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    if (radii[bi].empty()) continue;
    std::sort(radii[bi].begin(), radii[bi].end());
    const size_t qi = std::min(radii[bi].size() - 1,
                               static_cast<size_t>(std::floor(0.90 * (radii[bi].size() - 1))));
    const double r = radii[bi][qi];
    volumeOuterDiameterUnits += M_PI * r * r * dy;
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoPlicSectorMedianSilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const std::vector<fvm::IsoSurfaceReconstruction3D> iso =
      fvm::reconstructIsoInterface3D(mesh, alpha);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const int sectors = 16;
  const double dy = ly / static_cast<double>(bins);
  const fvm::Vec3 axis{0.5 * opt.radialWindowOuterDiameters, 0.0,
                       0.5 * opt.radialWindowOuterDiameters};
  std::vector<std::vector<std::vector<double>>> radii(
      bins, std::vector<std::vector<double>>(sectors));
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    if (!connected[ci] || !iso[ci].mixed) continue;
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    const std::vector<fvm::Vec3> cutPoints = candidoPlicCutPolygonPoints3D(mesh, ci, iso[ci]);
    for (const fvm::Vec3& x : cutPoints) {
      const int bi = std::clamp(static_cast<int>(std::floor(x.y() / std::max(dy, 1e-30))),
                                0, bins - 1);
      const double dx = x.x() - axis.x();
      const double dz = x.z() - axis.z();
      const double r = std::hypot(dx, dz);
      if (!std::isfinite(r) || r <= 0.0) continue;
      double theta = std::atan2(dz, dx);
      if (theta < 0.0) theta += 2.0 * M_PI;
      const int si = std::clamp(static_cast<int>(std::floor(theta / (2.0 * M_PI) *
                                                            static_cast<double>(sectors))),
                                0, sectors - 1);
      radii[bi][si].push_back(r);
    }
  }

  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    double sumRadiusSquared = 0.0;
    int contributingSectors = 0;
    for (int si = 0; si < sectors; ++si) {
      auto& rs = radii[bi][si];
      if (rs.empty()) continue;
      std::sort(rs.begin(), rs.end());
      const double r = rs[rs.size() / 2];
      sumRadiusSquared += r * r;
      ++contributingSectors;
    }
    if (contributingSectors > 0) {
      volumeOuterDiameterUnits +=
          M_PI * (sumRadiusSquared / static_cast<double>(contributingSectors)) * dy;
    }
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoPlicRayPlaneSilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const std::vector<fvm::IsoSurfaceReconstruction3D> iso =
      fvm::reconstructIsoInterface3D(mesh, alpha);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const int rays = 16;
  const double dy = ly / static_cast<double>(bins);
  const double radialWindow = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * radialWindow, 0.0, 0.5 * radialWindow};
  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    const double y0 = (static_cast<double>(bi) + 0.5) * dy;
    double sumRadiusSquared = 0.0;
    int contributingRays = 0;
    for (int ri = 0; ri < rays; ++ri) {
      const double theta = 2.0 * M_PI * (static_cast<double>(ri) + 0.5) /
                           static_cast<double>(rays);
      const fvm::Vec3 er{std::cos(theta), 0.0, std::sin(theta)};
      const fvm::Vec3 base{axis.x(), y0, axis.z()};
      std::vector<double> candidates;
      for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        if (!connected[ci] || !iso[ci].mixed) continue;
        const double a = std::clamp(alpha[ci], 0.0, 1.0);
        if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
        if (std::abs(mesh.cells[ci].centroid.y() - y0) > dy) continue;
        const double denom = iso[ci].normal.dot(er);
        if (std::abs(denom) <= 1e-10) continue;
        const double radius =
            (iso[ci].cut - iso[ci].normal.dot(base - mesh.cells[ci].centroid)) / denom;
        if (!std::isfinite(radius) || radius <= 0.0 || radius > 0.5 * radialWindow) continue;
        const fvm::Vec3 x = base + radius * er;
        fvm::Vec3 lo = mesh.points[mesh.cells[ci].points.front()];
        fvm::Vec3 hi = lo;
        for (int pi : mesh.cells[ci].points) {
          lo = lo.cwiseMin(mesh.points[pi]);
          hi = hi.cwiseMax(mesh.points[pi]);
        }
        const double tol = 0.25 * std::cbrt(std::max(mesh.cells[ci].V, 1e-30));
        if (x.x() < lo.x() - tol || x.x() > hi.x() + tol ||
            x.y() < lo.y() - tol || x.y() > hi.y() + tol ||
            x.z() < lo.z() - tol || x.z() > hi.z() + tol) {
          continue;
        }
        candidates.push_back(radius);
      }
      if (candidates.empty()) continue;
      std::sort(candidates.begin(), candidates.end());
      const double radius = candidates[candidates.size() / 2];
      sumRadiusSquared += radius * radius;
      ++contributingRays;
    }
    if (contributingRays > 0) {
      volumeOuterDiameterUnits +=
          M_PI * (sumRadiusSquared / static_cast<double>(contributingRays)) * dy;
    }
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoPlicRayPlaneQuantileSilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt,
    double quantile) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const std::vector<fvm::IsoSurfaceReconstruction3D> iso =
      fvm::reconstructIsoInterface3D(mesh, alpha);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const int rays = 32;
  const double dy = ly / static_cast<double>(bins);
  const double radialWindow = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * radialWindow, 0.0, 0.5 * radialWindow};
  double volumeOuterDiameterUnits = 0.0;
  const double q = std::clamp(quantile, 0.0, 1.0);
  for (int bi = 0; bi < bins; ++bi) {
    const double y0 = (static_cast<double>(bi) + 0.5) * dy;
    double sumRadiusSquared = 0.0;
    int contributingRays = 0;
    for (int ri = 0; ri < rays; ++ri) {
      const double theta = 2.0 * M_PI * (static_cast<double>(ri) + 0.5) /
                           static_cast<double>(rays);
      const fvm::Vec3 er{std::cos(theta), 0.0, std::sin(theta)};
      const fvm::Vec3 base{axis.x(), y0, axis.z()};
      std::vector<double> candidates;
      for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        if (!connected[ci] || !iso[ci].mixed) continue;
        const double a = std::clamp(alpha[ci], 0.0, 1.0);
        if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
        if (std::abs(mesh.cells[ci].centroid.y() - y0) > dy) continue;
        const double denom = iso[ci].normal.dot(er);
        if (std::abs(denom) <= 1e-10) continue;
        const double radius =
            (iso[ci].cut - iso[ci].normal.dot(base - mesh.cells[ci].centroid)) / denom;
        if (!std::isfinite(radius) || radius <= 0.0 || radius > 0.5 * radialWindow) continue;
        const fvm::Vec3 x = base + radius * er;
        fvm::Vec3 lo = mesh.points[mesh.cells[ci].points.front()];
        fvm::Vec3 hi = lo;
        for (int pi : mesh.cells[ci].points) {
          lo = lo.cwiseMin(mesh.points[pi]);
          hi = hi.cwiseMax(mesh.points[pi]);
        }
        const double tol = 0.25 * std::cbrt(std::max(mesh.cells[ci].V, 1e-30));
        if (x.x() < lo.x() - tol || x.x() > hi.x() + tol ||
            x.y() < lo.y() - tol || x.y() > hi.y() + tol ||
            x.z() < lo.z() - tol || x.z() > hi.z() + tol) {
          continue;
        }
        candidates.push_back(radius);
      }
      if (candidates.empty()) continue;
      std::sort(candidates.begin(), candidates.end());
      const double pos = q * static_cast<double>(candidates.size() - 1);
      const size_t loIdx = static_cast<size_t>(std::floor(pos));
      const size_t hiIdx = std::min(candidates.size() - 1, loIdx + 1);
      const double t = std::clamp(pos - static_cast<double>(loIdx), 0.0, 1.0);
      const double radius = (1.0 - t) * candidates[loIdx] + t * candidates[hiIdx];
      sumRadiusSquared += radius * radius;
      ++contributingRays;
    }
    if (contributingRays > 0) {
      volumeOuterDiameterUnits +=
          M_PI * (sumRadiusSquared / static_cast<double>(contributingRays)) * dy;
    }
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline double candidoPlicFirstExitSilhouetteVolumeInnerDiameterUnits3D(
    const fvm::Mesh3D& mesh,
    const fvm::ScalarField& alpha,
    const CandidoTaylorConeJetSetup& setup,
    const CandidoConeJetSmokeOptions3D& opt) {
  const std::vector<int> connected =
      candidoInletConnectedLiquidMask3D(mesh, alpha, setup, opt, 0.02);
  const std::vector<fvm::IsoSurfaceReconstruction3D> iso =
      fvm::reconstructIsoInterface3D(mesh, alpha);
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const int bins = std::max(1, opt.ny);
  const int rays = 32;
  const double dy = ly / static_cast<double>(bins);
  const double radialWindow = opt.radialWindowOuterDiameters;
  const fvm::Vec3 axis{0.5 * radialWindow, 0.0, 0.5 * radialWindow};
  double volumeOuterDiameterUnits = 0.0;
  for (int bi = 0; bi < bins; ++bi) {
    const double y0 = (static_cast<double>(bi) + 0.5) * dy;
    double sumRadiusSquared = 0.0;
    int contributingRays = 0;
    for (int ri = 0; ri < rays; ++ri) {
      const double theta = 2.0 * M_PI * (static_cast<double>(ri) + 0.5) /
                           static_cast<double>(rays);
      const fvm::Vec3 er{std::cos(theta), 0.0, std::sin(theta)};
      const fvm::Vec3 base{axis.x(), y0, axis.z()};
      std::vector<double> candidates;
      for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        if (!connected[ci] || !iso[ci].mixed) continue;
        const double a = std::clamp(alpha[ci], 0.0, 1.0);
        if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
        if (std::abs(mesh.cells[ci].centroid.y() - y0) > dy) continue;
        const double denom = iso[ci].normal.dot(er);
        if (std::abs(denom) <= 1e-10) continue;
        const double radius =
            (iso[ci].cut - iso[ci].normal.dot(base - mesh.cells[ci].centroid)) / denom;
        if (!std::isfinite(radius) || radius <= 0.0 || radius > 0.5 * radialWindow) continue;
        const fvm::Vec3 x = base + radius * er;
        fvm::Vec3 lo = mesh.points[mesh.cells[ci].points.front()];
        fvm::Vec3 hi = lo;
        for (int pi : mesh.cells[ci].points) {
          lo = lo.cwiseMin(mesh.points[pi]);
          hi = hi.cwiseMax(mesh.points[pi]);
        }
        const double tol = 0.25 * std::cbrt(std::max(mesh.cells[ci].V, 1e-30));
        if (x.x() < lo.x() - tol || x.x() > hi.x() + tol ||
            x.y() < lo.y() - tol || x.y() > hi.y() + tol ||
            x.z() < lo.z() - tol || x.z() > hi.z() + tol) {
          continue;
        }
        candidates.push_back(radius);
      }
      if (candidates.empty()) continue;
      std::sort(candidates.begin(), candidates.end());
      const double radius = candidates.front();
      sumRadiusSquared += radius * radius;
      ++contributingRays;
    }
    if (contributingRays > 0) {
      volumeOuterDiameterUnits +=
          M_PI * (sumRadiusSquared / static_cast<double>(contributingRays)) * dy;
    }
  }
  const double outerToInner = setup.outerDiameter / setup.innerDiameter;
  return volumeOuterDiameterUnits * outerToInner * outerToInner * outerToInner;
}

inline CandidoConeJetSmokeReport3D runCandidoConeJetSmoke3D(
    double targetCaE,
    const CandidoTaylorConeJetSetup& setup = {},
    const CandidoConeJetSmokeOptions3D& opt = {},
    const fvm::Mesh3D* externalMesh = nullptr) {
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  // Optional external mesh injection: a caller-supplied mesh (e.g. an OpenFOAM
  // polyMesh normalized into this solver's nondimensional box convention) is used
  // when provided; otherwise the built-in structured hex box is generated.
  fvm::Mesh3D mesh = externalMesh
                         ? *externalMesh
                         : fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, lx, ly, lz, opt.skew);
  const double voltage = candidoVoltageForElectricCapillary(setup, targetCaE);
  const double dimensionlessVoltage = voltage / (candidoElectricFieldScale(setup, setup.validationVoltage) *
                                                setup.outerDiameter);
  fvm::ScalarField alpha = candidoInitialAlpha3D(mesh, setup, opt, targetCaE);
  fvm::VectorField3 u = candidoInitialVelocity3D(mesh, setup, opt);
  candidoApplyPreconditionedPaperCurrentJetVelocityCells3D(u, mesh, setup, opt);
  if (opt.useFullyDevelopedInletVelocityBoundary) {
    candidoApplyFullyDevelopedInletVelocityCells3D(u, mesh, setup, opt);
  }
  candidoApplyMovingCollectorWallCells3D(u, mesh, setup, opt);
  fvm::ScalarField p(mesh.cells.size(), 0.0);
  fvm::ScalarField rho, eps, sigmaE;
  candidoMixtureFields3D(mesh, alpha, setup, opt, rho, eps, sigmaE);
  fvm::ScalarField rhoE(mesh.cells.size(), 0.0);
  fvm::PotentialBoundary3D bc = candidoPotentialBoundary3D(mesh, setup, opt, dimensionlessVoltage);
  const double dx = std::cbrt((lx * ly * lz) / static_cast<double>(mesh.cells.size()));
  const double dtAdv = opt.cfl * dx / std::max(1.0, dimensionlessVoltage / std::max(ly, 1e-30));
  const double dtCap = std::sqrt(std::max(dx * dx * dx, 1e-30) / (4.0 * M_PI));
  const double unrestrictedDt = std::min(dtAdv, dtCap);
  const double liquidSigmaForTau =
      opt.useDimensionalElectricalScaling
          ? candidoDimensionlessConductivityFromPhysical(setup, setup.liquidConductivity)
          : opt.normalizedLiquidConductivity;
  const double gasSigmaForTau =
      opt.useDimensionalElectricalScaling
          ? candidoDimensionlessConductivityFromPhysical(setup, setup.gasConductivity)
          : opt.normalizedGasConductivity;
  const double minElectricRelaxationTau =
      std::min(setup.liquidRelativePermittivity / std::max(liquidSigmaForTau, 1e-300),
               setup.gasRelativePermittivity / std::max(gasSigmaForTau, 1e-300));
  const double dtElectric =
      opt.useElectricRelaxationTimeStepLimit
          ? std::max(opt.electricRelaxationTimeStepSafety, 1e-12) *
                minElectricRelaxationTau
          : std::numeric_limits<double>::infinity();
  const double dt = std::min(unrestrictedDt, dtElectric);
  fvm::ScalarField rAU(mesh.cells.size(), dt);

  CandidoConeJetSmokeReport3D report;
  report.targetCaE = targetCaE;
  report.voltage = voltage;
  report.computedCaE = candidoElectricCapillaryNumber(setup, voltage);
  report.electricWeber = candidoValidationElectricWeber();
  report.hydrodynamicTimeScale = candidoHydrodynamicTimeScale(setup);
  report.inletVelocity = candidoInletVelocity(setup);
  report.dt = dt;
  report.unrestrictedDt = unrestrictedDt;
  report.electricRelaxationDtLimit = dtElectric;
  report.dtOverElectricRelaxationLimit =
      dt / std::max(minElectricRelaxationTau, 1e-300);
  report.electricRelaxationTimestepLimited = dt < 0.999999 * unrestrictedDt ? 1 : 0;
  report.cells = static_cast<int>(mesh.cells.size());
  report.faces = static_cast<int>(mesh.faces.size());
  report.steps = opt.steps;
  report.initialMass = fvm::vofMass3D(mesh, alpha);
  report.initialIntegratedCharge = candidoIntegratedCharge3D(mesh, rhoE);
  report.initialTipY = candidoLiquidTipY3D(mesh, alpha);
  report.minAlpha = 1.0;
  report.maxAlpha = 0.0;
  const double qLimit =
      opt.useRayleighChargeLimit
          ? candidoDimensionlessRayleighChargeLimit(setup, voltage)
          : opt.chargeLimitBase * std::max(1.0, targetCaE / 0.25);
  const CandidoAxialCurrentScan3D initialAxialScan =
      candidoAxialDevelopedJetCurrentScan3D(mesh, alpha, rhoE, u, setup, opt);
  report.history.push_back({0,
                            0.0,
                            report.initialMass,
                            *std::min_element(alpha.begin(), alpha.end()),
                            *std::max_element(alpha.begin(), alpha.end()),
                            report.initialTipY,
                            candidoLiquidCentroidY3D(mesh, alpha),
                            candidoInterfaceRadialAsymmetry3D(mesh, alpha, opt),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            candidoCrossSectionLiquidAreaInnerDiameterUnits3D(
                                mesh, alpha, setup, 0.5 * ly, ly / opt.ny, false),
                            candidoCrossSectionLiquidAreaInnerDiameterUnits3D(
                                mesh, alpha, setup, 0.5 * ly, ly / opt.ny, true),
                            initialAxialScan.yOverDi,
                            initialAxialScan.alpha05AreaDi2,
                            initialAxialScan.convectiveCurrent,
                            initialAxialScan.liquidConvectiveCurrent,
                            initialAxialScan.alpha05ConvectiveCurrent,
                            initialAxialScan.totalCurrent,
                            initialAxialScan.liquidTotalCurrent,
                            initialAxialScan.alpha05TotalCurrent,
                            initialAxialScan.alpha05ConductiveCurrent,
                            initialAxialScan.meanAlpha05Charge,
                            initialAxialScan.meanAlpha05AbsCharge,
                            initialAxialScan.meanAlpha05Uy,
                            initialAxialScan.meanAlpha05AbsUy,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            initialAxialScan.alpha05CurrentShapeFactor,
                            0.0,
                            0.0,
                            candidoAxialWaveAsymmetryPeak3D(mesh, alpha, setup, opt).first,
                            candidoAxialWaveAsymmetryPeak3D(mesh, alpha, setup, opt).second,
                            candidoMorphologyVolumeInnerDiameterUnits3D(mesh, alpha, setup),
                            candidoConnectedMorphologyVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                setup, opt),
                            candidoAlpha05AxialSilhouetteVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                   setup, opt),
                            candidoRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                 setup, opt),
                            candidoAllLiquidRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(
                                mesh, alpha, setup, opt),
                            candidoRayAlpha05CellBoundarySilhouetteVolumeInnerDiameterUnits3D(
                                mesh, alpha, setup, opt),
                            candidoLinearRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(
                                mesh, alpha, setup, opt),
                            candidoPlicContourSilhouetteVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                  setup, opt),
                            candidoPlicPolygonSilhouetteVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                  setup, opt),
                            candidoPlicSectorMedianSilhouetteVolumeInnerDiameterUnits3D(
                                mesh, alpha, setup, opt),
                            candidoPlicRayPlaneSilhouetteVolumeInnerDiameterUnits3D(
                                mesh, alpha, setup, opt),
                            candidoPlicRayPlaneQuantileSilhouetteVolumeInnerDiameterUnits3D(
                                mesh, alpha, setup, opt, 0.25),
                            candidoPlicFirstExitSilhouetteVolumeInnerDiameterUnits3D(
                                mesh, alpha, setup, opt)});

  for (int step = 0; step < opt.steps; ++step) {
    candidoMixtureFields3D(mesh, alpha, setup, opt, rho, eps, sigmaE);
    for (size_t ci = 0; ci < mesh.cells.size(); ++ci) rAU[ci] = dt / std::max(rho[ci], 1e-30);
    fvm::PotentialSolveReport3D potential;
    if (opt.useConductivityPotentialChargeClosure) {
      const fvm::ScalarField zeroCharge(mesh.cells.size(), 0.0);
      potential = fvm::solvePotential3D(mesh, sigmaE, zeroCharge, bc, 1e-11, 4000);
      report.maxConductivityPotentialResidual =
          std::max(report.maxConductivityPotentialResidual, potential.residual);
      const fvm::ScalarField epsFlux =
          candidoPoissonFaceNormalFlux3D(mesh, eps, potential.phi, bc);
      const fvm::ScalarField gaussCharge = fvm::explicitDivFaceFlux3D(mesh, epsFlux);
      for (size_t ci = 0; ci < rhoE.size(); ++ci) {
        const double unclamped = gaussCharge[ci];
        const double clamped = std::clamp(unclamped, -qLimit, qLimit);
        report.cumulativeConductivityClosureClampCorrectionL1 +=
            std::abs(clamped - unclamped) * mesh.cells[ci].V;
        rhoE[ci] = clamped;
      }
    } else {
      potential = fvm::solvePotential3D(mesh, eps, rhoE, bc, 1e-11, 4000);
    }
    report.maxPotentialResidual = std::max(report.maxPotentialResidual, potential.residual);
    const CandidoGaussLawCellGradientAudit3D gaussAudit =
        candidoGaussLawCellGradientAudit3D(mesh, eps, potential.E, rhoE);
    report.maxGaussLawCellGradientResidual =
        std::max(report.maxGaussLawCellGradientResidual, gaussAudit.maxAbsResidual);
    report.maxRelativeGaussLawCellGradientResidual =
        std::max(report.maxRelativeGaussLawCellGradientResidual,
                 gaussAudit.relativeL2Residual);
    if (opt.useInterfacialOhmicChargeSource) {
      const CandidoInterfacialOhmicChargeSourceReport3D sourceReport =
          candidoApplyInterfacialOhmicChargeSource3D(
              mesh, alpha, potential.E, setup, opt, dt, qLimit, rhoE);
      report.cumulativeInterfacialOhmicChargeSource += sourceReport.appliedCharge;
      report.cumulativeInterfacialOhmicChargeClampL1 +=
          sourceReport.clampCorrectionL1;
      report.maxInterfacialOhmicChargeSourceDensity =
          std::max(report.maxInterfacialOhmicChargeSourceDensity,
                   sourceReport.maxAbsSourceDensity);
      report.maxInterfacialOhmicChargeSourceCells =
          std::max(report.maxInterfacialOhmicChargeSourceCells,
                   sourceReport.sourceCells);
      potential = fvm::solvePotential3D(mesh, eps, rhoE, bc, 1e-11, 4000);
      report.maxPostChargePotentialResidual =
          std::max(report.maxPostChargePotentialResidual, potential.residual);
      const CandidoGaussLawCellGradientAudit3D sourceGaussAudit =
          candidoGaussLawCellGradientAudit3D(mesh, eps, potential.E, rhoE);
      report.maxPostChargeRelativeGaussLawResidual =
          std::max(report.maxPostChargeRelativeGaussLawResidual,
                   sourceGaussAudit.relativeL2Residual);
    }
    if (opt.useFullyDevelopedInletVelocityBoundary) {
      candidoApplyFullyDevelopedInletVelocityCells3D(u, mesh, setup, opt);
    }
    candidoApplyMovingCollectorWallCells3D(u, mesh, setup, opt);
    fvm::ScalarField faceFlux = candidoFaceFlux3D(mesh, u);
    candidoApplyOpenAtmosphericBoundaryFlux3D(faceFlux, mesh, u, setup, opt);
    candidoApplyInletBoundaryFlux3D(faceFlux, mesh, u, setup, opt);
    fvm::ScalarField conductiveFlux =
        opt.usePoissonFaceConductiveCurrent
            ? candidoPoissonFaceConductiveCurrentFlux3D(mesh, sigmaE, potential.phi, bc)
            : candidoConductiveCurrentFlux3D(mesh, sigmaE, potential.E);
    if (opt.suppressNozzleConductiveChargeFlux) {
      candidoSuppressNozzleConductiveFlux3D(mesh, setup, opt, conductiveFlux);
    }
    if (opt.collectorOnlyConductiveChargeFlux) {
      candidoKeepOnlyCollectorBoundaryConductiveFlux3D(mesh, conductiveFlux);
    }
    double conductiveCurrent = 0.0;
    for (double j : conductiveFlux) conductiveCurrent = std::max(conductiveCurrent, std::abs(j));
    report.maxConductiveCurrent = std::max(report.maxConductiveCurrent, conductiveCurrent);
    double boundaryChargeFlux = 0.0;
    double boundaryConductiveChargeFlux = 0.0;
    std::array<double, 6> boundaryConductiveByPatch = {0.0, 0.0, 0.0,
                                                       0.0, 0.0, 0.0};
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      if (mesh.faces[fi].internal()) continue;
      boundaryChargeFlux += conductiveFlux[fi];
      boundaryConductiveChargeFlux += conductiveFlux[fi];
      const int patch = mesh.faces[fi].patch;
      if (patch >= 0 && patch < static_cast<int>(boundaryConductiveByPatch.size())) {
        boundaryConductiveByPatch[patch] += conductiveFlux[fi];
      }
    }
    for (size_t pi = 0; pi < boundaryConductiveByPatch.size(); ++pi) {
      report.maxAbsConductiveBoundaryCurrentByPatch[pi] =
          std::max(report.maxAbsConductiveBoundaryCurrentByPatch[pi],
                   std::abs(boundaryConductiveByPatch[pi]));
    }
    const int chargeSubcycles = std::max(1, opt.chargeSubcycles);
    const double chargeDt = dt / static_cast<double>(chargeSubcycles);
    fvm::ScalarField chargeRedistributionWeights;
    const fvm::ScalarField* chargeRedistributionWeightsPtr = nullptr;
    if (opt.useInterfaceLocalizedChargeRedistribution) {
      chargeRedistributionWeights =
          candidoInterfaceChargeRedistributionWeights3D(mesh, alpha, opt);
      chargeRedistributionWeightsPtr = &chargeRedistributionWeights;
    }
    fvm::ScalarField implicitOhmicConductiveFaceScale;
    const fvm::ScalarField* implicitOhmicConductiveFaceScalePtr = nullptr;
    if (opt.applyConductiveBoundaryFiltersInImplicitOhmic) {
      implicitOhmicConductiveFaceScale =
          candidoImplicitOhmicConductiveFaceScale3D(mesh, setup, opt);
      implicitOhmicConductiveFaceScalePtr = &implicitOhmicConductiveFaceScale;
    }
    if (opt.useConductivityPotentialChargeClosure) {
      report.cumulativeBoundaryChargeFlux += dt * boundaryConductiveChargeFlux;
      report.cumulativeConductiveBoundaryChargeFlux +=
          dt * boundaryConductiveChargeFlux;
      for (size_t pi = 0; pi < boundaryConductiveByPatch.size(); ++pi) {
        report.cumulativeConductiveBoundaryChargeFluxByPatch[pi] +=
            dt * boundaryConductiveByPatch[pi];
      }
    } else {
    for (int sub = 0; sub < chargeSubcycles; ++sub) {
      CandidoChargeAdvanceReport3D chargeAdvance;
      if (opt.implicitOhmicChargeProjection) {
        chargeAdvance = candidoAdvanceChargeImplicitOhmic3D(
            mesh, rhoE, faceFlux, eps, sigmaE, bc, chargeDt, qLimit,
            opt.conservativeChargeBounding, opt.useBoundaryChargeAdvection,
            chargeRedistributionWeightsPtr, implicitOhmicConductiveFaceScalePtr);
        report.cumulativeBoundaryChargeFlux +=
            chargeDt * (chargeAdvance.boundaryConvectiveChargeFlux +
                        chargeAdvance.boundaryConductiveChargeFlux);
        report.cumulativeConductiveBoundaryChargeFlux +=
            chargeDt * chargeAdvance.boundaryConductiveChargeFlux;
        for (size_t pi = 0; pi < chargeAdvance.boundaryConductiveByPatch.size(); ++pi) {
          report.cumulativeConductiveBoundaryChargeFluxByPatch[pi] +=
              chargeDt * chargeAdvance.boundaryConductiveByPatch[pi];
          report.maxAbsConductiveBoundaryCurrentByPatch[pi] =
              std::max(report.maxAbsConductiveBoundaryCurrentByPatch[pi],
                       std::abs(chargeAdvance.boundaryConductiveByPatch[pi]));
        }
        report.maxConductiveCurrent =
            std::max(report.maxConductiveCurrent, chargeAdvance.maxAbsConductiveCurrent);
        report.maxImplicitOhmicChargeResidual =
            std::max(report.maxImplicitOhmicChargeResidual,
                     chargeAdvance.implicitPotentialResidual);
      } else {
        chargeAdvance = candidoAdvanceCharge3D(
            mesh, rhoE, faceFlux,
            opt.quasiImplicitBulkConduction
                ? candidoBoundaryOnlyConductiveFlux3D(mesh, conductiveFlux)
                : conductiveFlux,
            chargeDt, qLimit, opt.conservativeChargeBounding,
            opt.useBoundaryChargeAdvection, chargeRedistributionWeightsPtr);
        report.cumulativeBoundaryChargeFlux +=
            chargeDt * (chargeAdvance.boundaryConvectiveChargeFlux +
                        chargeAdvance.boundaryConductiveChargeFlux);
        report.cumulativeConductiveBoundaryChargeFlux +=
            chargeDt * chargeAdvance.boundaryConductiveChargeFlux;
        for (size_t pi = 0; pi < chargeAdvance.boundaryConductiveByPatch.size(); ++pi) {
          report.cumulativeConductiveBoundaryChargeFluxByPatch[pi] +=
              chargeDt * chargeAdvance.boundaryConductiveByPatch[pi];
        }
      }
      report.cumulativeChargeClampCorrectionL1 += chargeAdvance.clampCorrectionL1;
      report.cumulativeChargeRedistributionDeficitL1 +=
          chargeAdvance.redistributionDeficitL1;
      report.maxChargeRedistributionResidual =
          std::max(report.maxChargeRedistributionResidual,
                   std::abs(chargeAdvance.redistributionResidual));
      report.maxChargeRedistributionWeightedCapacity =
          std::max(report.maxChargeRedistributionWeightedCapacity,
                   chargeAdvance.maxRedistributionWeightedCapacity);
      report.maxChargeRedistributionWeightedCells =
          std::max(report.maxChargeRedistributionWeightedCells,
                   chargeAdvance.maxRedistributionWeightedCells);
      report.maxChargeClampedCells =
          std::max(report.maxChargeClampedCells, chargeAdvance.clampedCells);
      report.maxUnclampedAbsCharge =
          std::max(report.maxUnclampedAbsCharge, chargeAdvance.maxUnclampedAbsCharge);
      if (!opt.implicitOhmicChargeProjection &&
          (opt.quasiImplicitChargeRelaxation || opt.quasiImplicitBulkConduction)) {
        double relaxationSink = 0.0;
        for (size_t ci = 0; ci < rhoE.size(); ++ci) {
          const double oldCharge = rhoE[ci];
          const double rate = sigmaE[ci] / std::max(eps[ci], 1e-30);
          rhoE[ci] = oldCharge / (1.0 + chargeDt * std::max(rate, 0.0));
          relaxationSink += (oldCharge - rhoE[ci]) * mesh.cells[ci].V;
        }
        report.cumulativeChargeRelaxationSink += relaxationSink;
      }
    }
    }
    if (opt.refreshPotentialAfterChargeAdvance) {
      potential = fvm::solvePotential3D(mesh, eps, rhoE, bc, 1e-11, 4000);
      report.maxPotentialResidual =
          std::max(report.maxPotentialResidual, potential.residual);
      report.maxPostChargePotentialResidual =
          std::max(report.maxPostChargePotentialResidual, potential.residual);
      const CandidoGaussLawCellGradientAudit3D postChargeGaussAudit =
          candidoGaussLawCellGradientAudit3D(mesh, eps, potential.E, rhoE);
      report.maxGaussLawCellGradientResidual =
          std::max(report.maxGaussLawCellGradientResidual,
                   postChargeGaussAudit.maxAbsResidual);
      report.maxRelativeGaussLawCellGradientResidual =
          std::max(report.maxRelativeGaussLawCellGradientResidual,
                   postChargeGaussAudit.relativeL2Residual);
      report.maxPostChargeRelativeGaussLawResidual =
          std::max(report.maxPostChargeRelativeGaussLawResidual,
                   postChargeGaussAudit.relativeL2Residual);
    }
    auto chargeBounds = std::minmax_element(rhoE.begin(), rhoE.end());
    report.minCharge = std::min(report.minCharge, *chargeBounds.first);
    report.maxCharge = std::max(report.maxCharge, *chargeBounds.second);

    fvm::EHDBodyForceReport3D electric =
        opt.useTomarConductingSurfaceForce
            ? candidoTomarConductingSurfaceForce3D(mesh, alpha, rhoE, eps, sigmaE,
                                                   potential.E, setup, opt)
            : (opt.usePoissonBoundedVectorMaxwellForce
                   ? candidoPoissonBoundedVectorMaxwellBodyForce3D(
                         mesh, rhoE, eps, potential.phi, potential.E, bc,
                         opt.poissonTangentialLimitFactor,
                         opt.poissonTangentialLimitFloorFraction)
                   : (opt.usePoissonHybridMaxwellForce
                          ? candidoPoissonHybridMaxwellBodyForce3D(mesh, rhoE, eps,
                                                                   potential.phi,
                                                                   potential.E, bc)
                          : (opt.usePoissonFaceMaxwellForce
                                 ? candidoPoissonFaceMaxwellBodyForce3D(
                                       mesh, rhoE, eps, potential.phi, bc)
                                 : fvm::maxwellBodyForce3D(mesh, rhoE,
                                                          potential.E, eps))));
    const fvm::Vec3 contactWallNormal = fvm::Vec3::UnitY();
    const double contactWallYMax =
        ly / std::max(opt.ny, 1) * std::max(opt.contactAngleCurvatureWallBandCells, 0.0);
    fvm::LocalPlicQuadricCurvatureReport3D kappa =
        opt.useContactAngleCurvature
            ? fvm::curvatureFromLocalPlicQuadricReport3D(mesh, alpha, 28,
                                                         &contactWallNormal,
                                                         setup.contactAngleDeg,
                                                         contactWallYMax)
            : fvm::curvatureFromLocalPlicQuadricReport3D(mesh, alpha, 28);
    fvm::BalancedForceSurfaceTensionState3D surface =
        fvm::buildBalancedForceSurfaceTensionState3D(mesh, alpha, 1.0, &kappa.kappa);
    report.maxElectricForce = std::max(report.maxElectricForce, electric.maxForce);
    report.maxCurvature = std::max(report.maxCurvature, kappa.maxAbsCurvature);
    report.curvatureFallbackFraction = kappa.fallbackFraction;
    double stepMaxCsfForce = 0.0;
    for (const fvm::Vec3& f : surface.csfForce) report.maxCsfForce = std::max(report.maxCsfForce, f.norm());
    for (const fvm::Vec3& f : surface.csfForce) stepMaxCsfForce = std::max(stepMaxCsfForce, f.norm());

    fvm::VectorField3 source(mesh.cells.size(), fvm::Vec3::Zero());
    const double electricDriveScale =
        opt.electricDriveReferenceScale *
        std::pow(std::max(targetCaE, 1e-30) / 0.25, opt.electricDriveCaExponent);
    for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
      source[ci] = -electricDriveScale * electric.faceCoupledForce[ci] +
                   opt.surfaceTensionDriveScale * surface.csfForce[ci];
    }
    fvm::MomentumPredictorReport3D mom =
        fvm::solveMomentumPredictorBiCGSTABILUT3D(mesh, u, source, rho, dt, opt.pseudoViscosity);
    u = mom.velocity;
    if (opt.useFullyDevelopedInletVelocityBoundary) {
      candidoApplyFullyDevelopedInletVelocityCells3D(u, mesh, setup, opt);
    }
    candidoApplyMovingCollectorWallCells3D(u, mesh, setup, opt);
    fvm::RhieChowProjector3D projector(mesh, rAU);
    fvm::CouplingReport3D projection = projector.project(u, p, 0.85);
    for (int corr = 0; corr < 3 && projection.maxDiv > 1e-8; ++corr) {
      projection = projector.project(u, p, 0.85);
    }
    candidoApplyOpenAtmosphericBoundaryFlux3D(projection.faceFlux, mesh, u, setup, opt);
    candidoApplyInletBoundaryFlux3D(projection.faceFlux, mesh, u, setup, opt);
    fvm::ScalarField rawVelocityFaceFlux = candidoFaceFlux3D(mesh, u);
    candidoApplyOpenAtmosphericBoundaryFlux3D(rawVelocityFaceFlux, mesh, u, setup, opt);
    candidoApplyInletBoundaryFlux3D(rawVelocityFaceFlux, mesh, u, setup, opt);
    const CandidoAxialMomentumSourceScan3D momentumSourceScan =
        candidoAxialMomentumSourceScan3D(
            mesh, alpha, rho, u, electric.faceCoupledForce, electricDriveScale,
            surface.csfForce, opt.surfaceTensionDriveScale, source, setup, opt);
    report.maxDiv = std::max(report.maxDiv, projection.maxDiv);
    fvm::VofTransportOptions3D vofOpt;
    vofOpt.scheme = fvm::VofAdvectionScheme3D::IsoAdvector;
    vofOpt.compression = opt.vofCompression;
    vofOpt.postSharpening = opt.vofPostSharpening;
    vofOpt.postSharpeningSweeps = opt.vofPostSharpeningSweeps;
    vofOpt.correctionSweeps = 5;
    fvm::ScalarField inletBoundaryAlpha;
    if (opt.useVofInletBoundaryAlpha || opt.useOpenAtmosphericBoundaryFlux) {
      inletBoundaryAlpha.assign(mesh.faces.size(), 0.0);
      if (opt.useVofInletBoundaryAlpha) {
        inletBoundaryAlpha = candidoInletBoundaryAlpha3D(mesh, setup, opt);
      }
      vofOpt.boundaryAlpha = &inletBoundaryAlpha;
    }
    fvm::ScalarField liquidFlux =
        fvm::isoAdvectorFaceFlux3D(mesh, alpha, projection.faceFlux, dt,
                                   vofOpt.boundaryAlpha);
    fvm::ScalarField compressionFlux = fvm::vofCompressionFlux3D(mesh, alpha, vofOpt.compression);
    double stepBoundaryFlux = 0.0;
    double stepBoundaryInflow = 0.0;
    double stepBoundaryOutflow = 0.0;
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      if (mesh.faces[fi].internal()) continue;
      const double signedFlux = liquidFlux[fi] + compressionFlux[fi];
      stepBoundaryFlux += signedFlux;
      stepBoundaryInflow += std::max(-signedFlux, 0.0);
      stepBoundaryOutflow += std::max(signedFlux, 0.0);
    }
    report.cumulativeBoundaryLiquidFlux += dt * stepBoundaryFlux;
    report.cumulativeBoundaryLiquidInflow += dt * stepBoundaryInflow;
    report.cumulativeBoundaryLiquidOutflow += dt * stepBoundaryOutflow;
    fvm::VofTransportReport3D vof = fvm::advectVof3D(mesh, alpha, projection.faceFlux, dt, vofOpt);
    report.alphaMassDrift = std::max(report.alphaMassDrift, vof.relativeMassDrift);
    report.minAlpha = std::min(report.minAlpha, vof.minAlpha);
    report.maxAlpha = std::max(report.maxAlpha, vof.maxAlpha);
    const double current =
        candidoCrossSectionConvectiveCurrent3D(mesh, rhoE, u, 0.5 * ly, ly / opt.ny);
    const double liquidCurrent =
        candidoCrossSectionLiquidJetConvectiveCurrent3D(mesh, alpha, rhoE, u, 0.5 * ly,
                                                        ly / opt.ny, false);
    const double alpha05Current =
        candidoCrossSectionLiquidJetConvectiveCurrent3D(mesh, alpha, rhoE, u, 0.5 * ly,
                                                        ly / opt.ny, true);
    const double midplaneLiquidArea =
        candidoCrossSectionLiquidAreaInnerDiameterUnits3D(mesh, alpha, setup, 0.5 * ly,
                                                         ly / opt.ny, false);
    const double midplaneAlpha05Area =
        candidoCrossSectionLiquidAreaInnerDiameterUnits3D(mesh, alpha, setup, 0.5 * ly,
                                                         ly / opt.ny, true);
    const CandidoAxialCurrentScan3D axialScan =
        candidoAxialDevelopedJetCurrentScan3D(mesh, alpha, rhoE, u, setup, opt,
                                              &sigmaE, &potential.E);
    const double totalCurrent =
        candidoCrossSectionTotalCurrent3D(mesh, rhoE, u, sigmaE, potential.E, 0.5 * ly,
                                          ly / opt.ny);
    const CandidoPoissonFaceCrossSectionCurrent3D poissonFaceCurrent =
        candidoPoissonFaceCrossSectionCurrent3D(mesh, alpha, rhoE, projection.faceFlux,
                                                sigmaE, potential.phi, bc, 0.5 * ly,
                                                ly / opt.ny, false,
                                                opt.useBoundaryChargeAdvection);
    const CandidoPoissonFaceCrossSectionCurrent3D poissonFaceAlpha05Current =
        candidoPoissonFaceCrossSectionCurrent3D(mesh, alpha, rhoE, projection.faceFlux,
                                                sigmaE, potential.phi, bc, 0.5 * ly,
                                                ly / opt.ny, true,
                                                opt.useBoundaryChargeAdvection);
    const CandidoPoissonFaceAxialCurrentScan3D poissonFaceAxialScan =
        candidoPoissonFaceAxialDevelopedCurrentScan3D(mesh, alpha, rhoE,
                                                      projection.faceFlux, sigmaE,
                                                      potential.phi, bc, setup, opt,
                                                      opt.useBoundaryChargeAdvection);
    const CandidoPoissonFaceAxialCurrentScan3D rawVelocityPoissonFaceAxialScan =
        candidoPoissonFaceAxialDevelopedCurrentScan3D(mesh, alpha, rhoE,
                                                      rawVelocityFaceFlux, sigmaE,
                                                      potential.phi, bc, setup, opt,
                                                      opt.useBoundaryChargeAdvection);
    report.maxConvectiveCurrent = std::max(report.maxConvectiveCurrent, std::abs(current));
    for (const fvm::Vec3& uc : u) report.maxVelocity = std::max(report.maxVelocity, uc.norm());
    double stepMaxVelocity = 0.0;
    for (const fvm::Vec3& uc : u) stepMaxVelocity = std::max(stepMaxVelocity, uc.norm());
    const auto wavePeak = candidoAxialWaveAsymmetryPeak3D(mesh, alpha, setup, opt);
    report.history.push_back({step + 1,
                              (step + 1) * dt,
                              fvm::vofMass3D(mesh, alpha),
                              vof.minAlpha,
                              vof.maxAlpha,
                              candidoLiquidTipY3D(mesh, alpha),
                              candidoLiquidCentroidY3D(mesh, alpha),
                              candidoInterfaceRadialAsymmetry3D(mesh, alpha, opt),
                              projection.maxDiv,
                              potential.residual,
                              electric.maxForce,
                              stepMaxCsfForce,
                              kappa.maxAbsCurvature,
                              conductiveCurrent,
                              std::abs(current),
                              std::abs(liquidCurrent),
                              std::abs(alpha05Current),
                              midplaneLiquidArea,
                              midplaneAlpha05Area,
                              axialScan.yOverDi,
                              axialScan.alpha05AreaDi2,
                              axialScan.convectiveCurrent,
                              axialScan.liquidConvectiveCurrent,
                              axialScan.alpha05ConvectiveCurrent,
                              axialScan.totalCurrent,
                              axialScan.liquidTotalCurrent,
                              axialScan.alpha05TotalCurrent,
                              axialScan.alpha05ConductiveCurrent,
                              axialScan.meanAlpha05Charge,
                              axialScan.meanAlpha05AbsCharge,
                              axialScan.meanAlpha05Uy,
                              axialScan.meanAlpha05AbsUy,
                              momentumSourceScan.meanAbsElectricMomentumSourceY,
                              momentumSourceScan.meanAbsSurfaceMomentumSourceY,
                              momentumSourceScan.meanAbsMomentumSourceY,
                              momentumSourceScan.meanAbsMomentumAccelerationY,
                              axialScan.alpha05CurrentShapeFactor,
                              std::abs(totalCurrent),
                              stepMaxVelocity,
                              wavePeak.first,
                              wavePeak.second,
                              candidoMorphologyVolumeInnerDiameterUnits3D(mesh, alpha, setup),
                              candidoConnectedMorphologyVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                  setup, opt),
                              candidoAlpha05AxialSilhouetteVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                     setup, opt),
                              candidoRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                   setup, opt),
                              candidoAllLiquidRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(
                                  mesh, alpha, setup, opt),
                              candidoRayAlpha05CellBoundarySilhouetteVolumeInnerDiameterUnits3D(
                                  mesh, alpha, setup, opt),
                              candidoLinearRayAlpha05SilhouetteVolumeInnerDiameterUnits3D(
                                  mesh, alpha, setup, opt),
                              candidoPlicContourSilhouetteVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                    setup, opt),
                              candidoPlicPolygonSilhouetteVolumeInnerDiameterUnits3D(mesh, alpha,
                                                                                    setup, opt),
                              candidoPlicSectorMedianSilhouetteVolumeInnerDiameterUnits3D(
                                  mesh, alpha, setup, opt),
                              candidoPlicRayPlaneSilhouetteVolumeInnerDiameterUnits3D(
                                  mesh, alpha, setup, opt),
                              candidoPlicRayPlaneQuantileSilhouetteVolumeInnerDiameterUnits3D(
                                  mesh, alpha, setup, opt, 0.25),
                              candidoPlicFirstExitSilhouetteVolumeInnerDiameterUnits3D(
                              mesh, alpha, setup, opt)});
    report.history.back().poissonFaceConvectiveCurrent =
        poissonFaceCurrent.convectiveCurrent;
    report.history.back().poissonFaceConductiveCurrent =
        poissonFaceCurrent.conductiveCurrent;
    report.history.back().poissonFaceTotalCurrent = poissonFaceCurrent.totalCurrent;
    report.history.back().poissonFaceAlpha05ConvectiveCurrent =
        poissonFaceAlpha05Current.convectiveCurrent;
    report.history.back().poissonFaceAlpha05ConductiveCurrent =
        poissonFaceAlpha05Current.conductiveCurrent;
    report.history.back().poissonFaceAlpha05TotalCurrent =
        poissonFaceAlpha05Current.totalCurrent;
    report.history.back().poissonFaceDevelopedYOverDi = poissonFaceAxialScan.yOverDi;
    report.history.back().poissonFaceDevelopedAlpha05AreaDi2 =
        poissonFaceAxialScan.alpha05AreaDi2;
    report.history.back().poissonFaceDevelopedAlpha05ConvectiveCurrent =
        poissonFaceAxialScan.alpha05ConvectiveCurrent;
    report.history.back().poissonFaceDevelopedAlpha05ConductiveCurrent =
        poissonFaceAxialScan.alpha05ConductiveCurrent;
    report.history.back().poissonFaceDevelopedAlpha05TotalCurrent =
        poissonFaceAxialScan.alpha05TotalCurrent;
    report.history.back().poissonFaceDevelopedAlpha05MeanAbsUpwindCharge =
        poissonFaceAxialScan.meanAbsUpwindCharge;
    report.history.back().poissonFaceDevelopedAlpha05MeanAbsFaceFlux =
        poissonFaceAxialScan.meanAbsFaceFlux;
    report.history.back().poissonFaceDevelopedAlpha05MeanAbsConvectiveFlux =
        poissonFaceAxialScan.meanAbsConvectiveFlux;
    report.history.back().poissonFaceDevelopedAlpha05MaxAbsUpwindCharge =
        poissonFaceAxialScan.maxAbsUpwindCharge;
    report.history.back().poissonFaceDevelopedAlpha05MaxAbsFaceFlux =
        poissonFaceAxialScan.maxAbsFaceFlux;
    report.history.back().rawVelocityFaceDevelopedAlpha05ConvectiveCurrent =
        rawVelocityPoissonFaceAxialScan.alpha05ConvectiveCurrent;
    report.history.back().rawVelocityFaceDevelopedAlpha05MeanAbsUpwindCharge =
        rawVelocityPoissonFaceAxialScan.meanAbsUpwindCharge;
    report.history.back().rawVelocityFaceDevelopedAlpha05MeanAbsFaceFlux =
        rawVelocityPoissonFaceAxialScan.meanAbsFaceFlux;
    report.history.back().rawVelocityFaceDevelopedAlpha05MeanAbsConvectiveFlux =
        rawVelocityPoissonFaceAxialScan.meanAbsConvectiveFlux;
    report.history.back().rawVelocityFaceDevelopedAlpha05MaxAbsUpwindCharge =
        rawVelocityPoissonFaceAxialScan.maxAbsUpwindCharge;
    report.history.back().rawVelocityFaceDevelopedAlpha05MaxAbsFaceFlux =
        rawVelocityPoissonFaceAxialScan.maxAbsFaceFlux;
  }

  report.finalMass = fvm::vofMass3D(mesh, alpha);
  report.finalIntegratedCharge = candidoIntegratedCharge3D(mesh, rhoE);
  report.chargeBudgetExpectedFinal =
      report.initialIntegratedCharge - report.cumulativeBoundaryChargeFlux -
      report.cumulativeChargeRelaxationSink +
      report.cumulativeInterfacialOhmicChargeSource;
  report.chargeBudgetResidual = report.finalIntegratedCharge - report.chargeBudgetExpectedFinal;
  report.relativeChargeBudgetResidual =
      std::abs(report.chargeBudgetResidual) /
      std::max(std::abs(report.chargeBudgetExpectedFinal), 1e-30);
  report.massBudgetExpectedFinal = report.initialMass - report.cumulativeBoundaryLiquidFlux;
  report.massBudgetResidual = report.finalMass - report.massBudgetExpectedFinal;
  report.relativeMassBudgetResidual =
      std::abs(report.massBudgetResidual) /
      std::max(std::abs(report.massBudgetExpectedFinal), 1e-30);
  report.finalTipY = candidoLiquidTipY3D(mesh, alpha);
  report.tipDisplacement = report.finalTipY - report.initialTipY;
  report.finalCentroidY = candidoLiquidCentroidY3D(mesh, alpha);
  report.finalRadialAsymmetry = candidoInterfaceRadialAsymmetry3D(mesh, alpha, opt);
  report.finalMidplaneJetRadius =
      candidoEquivalentLiquidRadiusAtY3D(mesh, alpha, 0.5 * ly, 5.0 * ly / opt.ny);
  return report;
}

}  // namespace electrospray
