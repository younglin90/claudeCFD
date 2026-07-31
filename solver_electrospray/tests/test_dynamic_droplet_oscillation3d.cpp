#include "TestUtil.hpp"
#include "fvm/EHDCoupling3D.hpp"
#include "fvm/PressureVelocityCoupling3D.hpp"
#include "fvm/SurfaceTension3D.hpp"
#include "fvm/VofTransport3D.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

enum class ValidationStatus { UPHELD, APPROXIMATE, DOWNGRADED, BLOCKED };

static const char* statusName(ValidationStatus s) {
  switch (s) {
    case ValidationStatus::UPHELD: return "UPHELD";
    case ValidationStatus::APPROXIMATE: return "APPROXIMATE";
    case ValidationStatus::DOWNGRADED: return "DOWNGRADED";
    case ValidationStatus::BLOCKED: return "BLOCKED";
  }
  return "BLOCKED";
}

struct Sample {
  double t = 0.0;
  double coeff = 0.0;
  double isoCoeff = 0.0;
  double gradCoeff = 0.0;
  double moment = 0.0;
};

struct DynamicSnapshot {
  int step = 0;
  double t = 0.0;
  double coeff = 0.0;
  double isoCoeff = 0.0;
  double gradCoeff = 0.0;
  double moment = 0.0;
  double massDrift = 0.0;
  double minAlpha = 0.0;
  double maxAlpha = 0.0;
  int mixedCells = 0;
  double interfaceThickness = 0.0;
  double curvatureConditionP95 = 0.0;
  double curvatureConditionMax = 0.0;
  double curvatureFallbackFraction = 0.0;
  int curvatureFittedCells = 0;
  int curvatureClampCells = 0;
  int curvatureIllConditionedFallbackCells = 0;
  double maxDiv = 0.0;
};

struct OscillationReport {
  std::string diagnosticPath = "full_path";
  std::string curvatureMethod = "iso_rdf";
  int n = 0;
  int cellsPerDim = 0;
  int steps = 0;
  double dt = 0.0;
  double capillaryDt = 0.0;
  double amplitudeRatio = 0.0;
  double lambOmega = 0.0;
  double measuredOmega = std::numeric_limits<double>::quiet_NaN();
  double fftOmega = std::numeric_limits<double>::quiet_NaN();
  double momentOmega = std::numeric_limits<double>::quiet_NaN();
  double isoOmega = std::numeric_limits<double>::quiet_NaN();
  double omegaError = std::numeric_limits<double>::quiet_NaN();
  double prosperettiDamping = 0.0;
  double measuredDamping = std::numeric_limits<double>::quiet_NaN();
  double dampingError = std::numeric_limits<double>::quiet_NaN();
  double maxDiv = 0.0;
  double maxMomentumResidual = 0.0;
  double maxSnGradDifference = 0.0;
  double maxIsoRdfAlphaKappaDifference = 0.0;
  double maxCurvatureFallbackFraction = 0.0;
  int maxCurvatureFittedCells = 0;
  int maxCurvatureClampCells = 0;
  double maxCurvatureStencilCondition = 0.0;
  double maxCurvatureStencilConditionP95 = 0.0;
  int maxCurvatureIllConditionedFallbackCells = 0;
  double massDrift = 0.0;
  double minAlpha = 0.0;
  double maxAlpha = 0.0;
  int zeroCrossings = 0;
  int peaks = 0;
  ValidationStatus frequencyStatus = ValidationStatus::BLOCKED;
  ValidationStatus dampingStatus = ValidationStatus::BLOCKED;
};

static double legendreMode(int n, double mu) {
  if (n == 2) return 0.5 * (3.0 * mu * mu - 1.0);
  if (n == 3) return 0.5 * (5.0 * mu * mu * mu - 3.0 * mu);
  return 1.0;
}

static fvm::ScalarField perturbedDropletAlpha(const fvm::Mesh3D& mesh, int n,
                                              double radius, double amplitudeRatio,
                                              double interfaceWidth) {
  fvm::ScalarField alpha(mesh.cells.size(), 0.0);
  const fvm::Vec3 center{0.5, 0.5, 0.5};
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    fvm::Vec3 rel = mesh.cells[c].centroid - center;
    double r = rel.norm();
    double mu = r > 1e-30 ? rel.x() / r : 1.0;
    double rb = radius * (1.0 + amplitudeRatio * legendreMode(n, mu));
    alpha[c] = std::clamp(0.5 * (1.0 - std::tanh((r - rb) / interfaceWidth)), 0.0, 1.0);
  }
  return alpha;
}

static double interfaceModeCoeff(const fvm::Mesh3D& mesh, const fvm::ScalarField& alpha,
                                 int n, double radius) {
  const fvm::Vec3 center{0.5, 0.5, 0.5};
  double num = 0.0;
  double den = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    fvm::Vec3 rel = mesh.cells[c].centroid - center;
    double r = rel.norm();
    double mu = r > 1e-30 ? rel.x() / r : 1.0;
    double p = legendreMode(n, mu);
    double w = std::max(alpha[c] * (1.0 - alpha[c]), 0.0) * mesh.cells[c].V;
    num += w * ((r - radius) / std::max(radius, 1e-30)) * p;
    den += w * p * p;
  }
  return num / std::max(den, 1e-30);
}

static double gradWeightedModeCoeff(const fvm::Mesh3D& mesh, const fvm::ScalarField& alpha,
                                    int n, double radius) {
  const fvm::Vec3 center{0.5, 0.5, 0.5};
  fvm::VectorField3 grad = fvm::gradLeastSquares3D(mesh, alpha);
  double num = 0.0;
  double den = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    fvm::Vec3 rel = mesh.cells[c].centroid - center;
    double r = rel.norm();
    double mu = r > 1e-30 ? rel.x() / r : 1.0;
    double p = legendreMode(n, mu);
    double w = grad[c].norm() * mesh.cells[c].V;
    num += w * ((r - radius) / std::max(radius, 1e-30)) * p;
    den += w * p * p;
  }
  return num / std::max(den, 1e-30);
}

static double reconstructedInterfaceModeCoeff(const fvm::Mesh3D& mesh,
                                              const fvm::ScalarField& alpha,
                                              int n, double radius) {
  const fvm::Vec3 center{0.5, 0.5, 0.5};
  std::vector<fvm::IsoSurfaceReconstruction3D> iso = fvm::reconstructIsoInterface3D(mesh, alpha);
  double num = 0.0;
  double den = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    if (!iso[c].mixed) continue;
    fvm::Vec3 rel = iso[c].interfaceCentroid - center;
    double r = rel.norm();
    double mu = r > 1e-30 ? rel.x() / r : 1.0;
    double p = legendreMode(n, mu);
    double w = std::max(iso[c].areaDensity, 1e-12) * mesh.cells[c].V;
    num += w * ((r - radius) / std::max(radius, 1e-30)) * p;
    den += w * p * p;
  }
  return num / std::max(den, 1e-30);
}

static double momentMetric(const fvm::Mesh3D& mesh, const fvm::ScalarField& alpha) {
  return fvm::deformationFromAlphaMoments3D(mesh, alpha, {0.5, 0.5, 0.5});
}

static DynamicSnapshot dynamicSnapshot(const fvm::Mesh3D& mesh, const fvm::ScalarField& alpha,
                                       int mode, double radius, int step, double dt,
                                       double initialMass, double maxDiv,
                                       const std::string& curvatureMethod) {
  DynamicSnapshot s;
  s.step = step;
  s.t = step * dt;
  s.coeff = interfaceModeCoeff(mesh, alpha, mode, radius);
  s.isoCoeff = reconstructedInterfaceModeCoeff(mesh, alpha, mode, radius);
  s.gradCoeff = gradWeightedModeCoeff(mesh, alpha, mode, radius);
  s.moment = momentMetric(mesh, alpha);
  const double mass = fvm::vofMass3D(mesh, alpha);
  s.massDrift = std::abs(mass - initialMass) / std::max(std::abs(initialMass), 1e-30);
  auto [amin, amax] = fvm::vofBounds3D(alpha);
  s.minAlpha = amin;
  s.maxAlpha = amax;
  s.maxDiv = maxDiv;

  const fvm::Vec3 center{0.5, 0.5, 0.5};
  double rMin = std::numeric_limits<double>::infinity();
  double rMax = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    if (alpha[c] <= 1e-6 || alpha[c] >= 1.0 - 1e-6) continue;
    ++s.mixedCells;
    const double r = (mesh.cells[c].centroid - center).norm();
    rMin = std::min(rMin, r);
    rMax = std::max(rMax, r);
  }
  s.interfaceThickness = s.mixedCells > 0 ? rMax - rMin : 0.0;

  if (curvatureMethod == "local_plic_quadric") {
    auto local = fvm::curvatureFromLocalPlicQuadricReport3D(mesh, alpha, 36);
    s.curvatureConditionP95 = local.p95StencilCondition;
    s.curvatureConditionMax = local.maxStencilCondition;
    s.curvatureFallbackFraction = local.fallbackFraction;
    s.curvatureFittedCells = local.fittedCells;
    s.curvatureClampCells = local.curvatureClampCells;
    s.curvatureIllConditionedFallbackCells = local.illConditionedFallbackCells;
  }
  return s;
}

static void writeHistoryRow(std::ofstream& csv, const std::string& diagnosticPath,
                            const std::string& curvatureMethod, int cellsPerDim,
                            int mode, const DynamicSnapshot& s) {
  csv << diagnosticPath << "," << curvatureMethod << "," << cellsPerDim << "," << mode << ","
      << s.step << "," << s.t << "," << s.coeff << "," << s.isoCoeff << ","
      << s.gradCoeff << "," << s.moment << "," << s.massDrift << ","
      << s.minAlpha << "," << s.maxAlpha << "," << s.mixedCells << ","
      << s.interfaceThickness << "," << s.curvatureConditionP95 << ","
      << s.curvatureConditionMax << "," << s.curvatureFallbackFraction << ","
      << s.curvatureFittedCells << "," << s.curvatureClampCells << ","
      << s.curvatureIllConditionedFallbackCells << "," << s.maxDiv << "\n";
}

static double radialAccelerationModeCoeff(const fvm::Mesh3D& mesh, const fvm::ScalarField& alpha,
                                          const fvm::VectorField3& acceleration, int mode,
                                          double radius) {
  (void)radius;
  const fvm::Vec3 center{0.5, 0.5, 0.5};
  double num = 0.0;
  double den = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    fvm::Vec3 rel = mesh.cells[c].centroid - center;
    double r = rel.norm();
    if (r <= 1e-30) continue;
    double mu = rel.x() / r;
    double p = legendreMode(mode, mu);
    double w = std::max(alpha[c] * (1.0 - alpha[c]), 0.0) * mesh.cells[c].V;
    num += w * acceleration[c].dot(rel / r) * p;
    den += w * p * p;
  }
  return num / std::max(den, 1e-30);
}

struct ForceIsolationReport {
  std::string curvatureMethod = "local_plic_quadric";
  std::string forcePath = "balanced_face_csf";
  int cellsPerDim = 0;
  int mode = 0;
  double shapeCoeff = 0.0;
  double lambOmega = 0.0;
  double measuredAccelCoeff = 0.0;
  double lambAccelCoeff = 0.0;
  double accelRelativeError = 0.0;
  double maxBalanceResidual = 0.0;
  double fallbackFraction = 0.0;
  double conditionP95 = 0.0;
  double conditionMax = 0.0;
  int illConditionedFallbackCells = 0;
  bool restoringSign = false;
  ValidationStatus status = ValidationStatus::BLOCKED;
};

struct CurvatureModeReport {
  std::string curvatureMethod = "local_plic_quadric";
  int cellsPerDim = 0;
  int mode = 0;
  double shapeCoeff = 0.0;
  double measuredKappaCoeff = 0.0;
  double analyticKappaCoeff = 0.0;
  double relativeError = 0.0;
  bool curvatureSign = false;
  double fallbackFraction = 0.0;
  double conditionP95 = 0.0;
  double conditionMax = 0.0;
  ValidationStatus status = ValidationStatus::BLOCKED;
};

static double curvatureModeCoeff(const fvm::Mesh3D& mesh, const fvm::ScalarField& alpha,
                                 const fvm::ScalarField& kappa, int mode) {
  const fvm::Vec3 center{0.5, 0.5, 0.5};
  double mean = 0.0;
  double weight = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const double w = std::max(alpha[c] * (1.0 - alpha[c]), 0.0) * mesh.cells[c].V;
    mean += w * kappa[c];
    weight += w;
  }
  mean /= std::max(weight, 1e-30);
  double num = 0.0;
  double den = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    fvm::Vec3 rel = mesh.cells[c].centroid - center;
    const double r = rel.norm();
    if (r <= 1e-30) continue;
    const double mu = rel.x() / r;
    const double p = legendreMode(mode, mu);
    const double w = std::max(alpha[c] * (1.0 - alpha[c]), 0.0) * mesh.cells[c].V;
    num += w * (kappa[c] - mean) * p;
    den += w * p * p;
  }
  return num / std::max(den, 1e-30);
}

static CurvatureModeReport runCurvatureModeCase(int cellsPerDim, int mode,
                                                double kappaMultiplier = 1.0,
                                                const std::string& methodLabel =
                                                    "local_plic_quadric",
                                                bool useHeightQuadric = false,
                                                bool reflectPerturbation = false,
                                                int maxSamples = 36) {
  constexpr double radius = 0.25;
  constexpr double amplitudeRatio = 0.04;
  auto mesh = fvm::Mesh3D::hexGrid(cellsPerDim, cellsPerDim, cellsPerDim, 1.0, 1.0, 1.0, 0.03);
  double dx = std::cbrt(1.0 / static_cast<double>(mesh.cells.size()));
  double interfaceWidth = std::max(1.15 * dx, 0.025);
  fvm::ScalarField alpha = perturbedDropletAlpha(mesh, mode, radius, amplitudeRatio, interfaceWidth);
  auto local = useHeightQuadric
                   ? fvm::curvatureFromLocalPlicHeightQuadricReport3D(mesh, alpha, maxSamples)
                   : fvm::curvatureFromLocalPlicQuadricReport3D(mesh, alpha, maxSamples);
  if (reflectPerturbation) {
    local.kappa = fvm::reflectedInterfaceKappaPerturbation3D(mesh, alpha, local.kappa);
  }
  if (kappaMultiplier != 1.0) {
    for (double& k : local.kappa) k *= kappaMultiplier;
  }

  CurvatureModeReport r;
  r.curvatureMethod = methodLabel;
  r.cellsPerDim = cellsPerDim;
  r.mode = mode;
  r.shapeCoeff = interfaceModeCoeff(mesh, alpha, mode, radius);
  r.measuredKappaCoeff = curvatureModeCoeff(mesh, alpha, local.kappa, mode);
  r.analyticKappaCoeff =
      -static_cast<double>(mode * (mode + 1) - 2) * r.shapeCoeff / std::max(radius, 1e-30);
  r.relativeError = std::abs(r.measuredKappaCoeff - r.analyticKappaCoeff) /
                    std::max(std::abs(r.analyticKappaCoeff), 1e-30);
  r.curvatureSign = r.measuredKappaCoeff * r.analyticKappaCoeff > 0.0;
  r.fallbackFraction = local.fallbackFraction;
  r.conditionP95 = local.p95StencilCondition;
  r.conditionMax = local.maxStencilCondition;
  if (!std::isfinite(r.relativeError)) {
    r.status = ValidationStatus::BLOCKED;
  } else if (r.curvatureSign && r.relativeError <= 0.20) {
    r.status = ValidationStatus::UPHELD;
  } else if (r.curvatureSign && r.relativeError <= 1.0) {
    r.status = ValidationStatus::APPROXIMATE;
  } else {
    r.status = ValidationStatus::DOWNGRADED;
  }
  return r;
}

static std::vector<double> centeredValues(const std::vector<Sample>& samples,
                                          double Sample::*member) {
  std::vector<double> y(samples.size(), 0.0);
  double mean = 0.0;
  for (const Sample& s : samples) mean += s.*member;
  mean /= std::max<size_t>(samples.size(), 1);
  for (size_t i = 0; i < samples.size(); ++i) y[i] = samples[i].*member - mean;
  return y;
}

static std::vector<Sample> smoothedSamples(const std::vector<Sample>& samples,
                                           double Sample::*member,
                                           int halfWindow = 2) {
  std::vector<Sample> out = samples;
  if (samples.empty()) return out;
  const int hw = std::max(0, halfWindow);
  for (size_t i = 0; i < samples.size(); ++i) {
    const int lo = std::max<int>(0, static_cast<int>(i) - hw);
    const int hi = std::min<int>(static_cast<int>(samples.size()) - 1,
                                 static_cast<int>(i) + hw);
    double sum = 0.0;
    int count = 0;
    for (int j = lo; j <= hi; ++j) {
      sum += samples[static_cast<size_t>(j)].*member;
      ++count;
    }
    out[i].*member = sum / std::max(count, 1);
  }
  return out;
}

static double zeroCrossingOmega(const std::vector<Sample>& samples, double Sample::*member,
                                int& crossingsOut) {
  crossingsOut = 0;
  if (samples.size() < 4) return std::numeric_limits<double>::quiet_NaN();
  std::vector<double> y = centeredValues(samples, member);
  std::vector<double> crossings;
  for (size_t i = 1; i < y.size(); ++i) {
    if (y[i - 1] == 0.0 || y[i] == 0.0 || (y[i - 1] < 0.0) == (y[i] < 0.0)) continue;
    double frac = std::abs(y[i - 1]) / std::max(std::abs(y[i - 1]) + std::abs(y[i]), 1e-30);
    crossings.push_back(samples[i - 1].t + frac * (samples[i].t - samples[i - 1].t));
  }
  crossingsOut = static_cast<int>(crossings.size());
  if (crossings.size() < 3) return std::numeric_limits<double>::quiet_NaN();
  double sumHalfPeriod = 0.0;
  int count = 0;
  for (size_t i = 1; i < crossings.size(); ++i) {
    double halfPeriod = crossings[i] - crossings[i - 1];
    if (halfPeriod > 0.0) {
      sumHalfPeriod += halfPeriod;
      ++count;
    }
  }
  if (count <= 0) return std::numeric_limits<double>::quiet_NaN();
  const double omega = M_PI / (sumHalfPeriod / static_cast<double>(count));
  const double totalTime = samples.back().t - samples.front().t;
  const double estimatedCrossings =
      totalTime > 0.0 ? 2.0 * totalTime * omega / (2.0 * M_PI) : 0.0;
  if (static_cast<double>(crossingsOut) > std::max(8.0, 3.0 * estimatedCrossings + 2.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return omega;
}

static double dftOmega(const std::vector<Sample>& samples, double Sample::*member) {
  if (samples.size() < 8) return std::numeric_limits<double>::quiet_NaN();
  std::vector<double> y = centeredValues(samples, member);
  double dt = samples[1].t - samples[0].t;
  if (!(dt > 0.0)) return std::numeric_limits<double>::quiet_NaN();
  int bestK = -1;
  double bestPower = 0.0;
  const int N = static_cast<int>(samples.size());
  for (int k = 1; k <= N / 2; ++k) {
    double re = 0.0;
    double im = 0.0;
    for (int j = 0; j < N; ++j) {
      double a = -2.0 * M_PI * static_cast<double>(k * j) / static_cast<double>(N);
      re += y[j] * std::cos(a);
      im += y[j] * std::sin(a);
    }
    double power = re * re + im * im;
    if (power > bestPower) {
      bestPower = power;
      bestK = k;
    }
  }
  if (bestK < 0) return std::numeric_limits<double>::quiet_NaN();
  return 2.0 * M_PI * static_cast<double>(bestK) / (static_cast<double>(N) * dt);
}

static double dampingRateFromEnvelope(const std::vector<Sample>& samples,
                                      double Sample::*member, int& peaksOut) {
  peaksOut = 0;
  if (samples.size() < 5) return std::numeric_limits<double>::quiet_NaN();
  std::vector<double> y = centeredValues(samples, member);
  std::vector<double> tPeak;
  std::vector<double> aPeak;
  for (size_t i = 1; i + 1 < y.size(); ++i) {
    double a0 = std::abs(y[i - 1]);
    double a1 = std::abs(y[i]);
    double a2 = std::abs(y[i + 1]);
    if (a1 > a0 && a1 >= a2 && a1 > 1e-12) {
      tPeak.push_back(samples[i].t);
      aPeak.push_back(a1);
    }
  }
  peaksOut = static_cast<int>(aPeak.size());
  if (aPeak.size() < 3) return std::numeric_limits<double>::quiet_NaN();
  double mt = 0.0;
  double ml = 0.0;
  for (size_t i = 0; i < aPeak.size(); ++i) {
    mt += tPeak[i];
    ml += std::log(aPeak[i]);
  }
  mt /= static_cast<double>(aPeak.size());
  ml /= static_cast<double>(aPeak.size());
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < aPeak.size(); ++i) {
    num += (tPeak[i] - mt) * (std::log(aPeak[i]) - ml);
    den += (tPeak[i] - mt) * (tPeak[i] - mt);
  }
  if (den <= 1e-30) return std::numeric_limits<double>::quiet_NaN();
  return -num / den;
}

static double lambOmega(int n, double radius, double sigma, double rhoIn, double rhoOut) {
  double factor = static_cast<double>(n * (n - 1) * (n + 1) * (n + 2));
  double denom = radius * radius * radius *
                 (static_cast<double>(n + 1) * rhoIn + static_cast<double>(n) * rhoOut);
  return std::sqrt(factor * sigma / std::max(denom, 1e-30));
}

static double prosperettiLowOhDamping(int n, double radius, double muIn, double muOut,
                                      double rhoIn, double rhoOut) {
  double rhoEff = (static_cast<double>(n + 1) * rhoIn + static_cast<double>(n) * rhoOut) /
                  static_cast<double>(2 * n + 1);
  double nuEff = (muIn + muOut) / std::max(rhoEff, 1e-30);
  return static_cast<double>((n - 1) * (2 * n + 1)) * nuEff / (radius * radius);
}

enum class ForceIsolationPath {
  BalancedFaceCsf,
  BalancedPressureGradient,
  FaceGaussAlpha,
  FaceGaussSnPressureGradient,
  HybridMeanBalancedDeltaGauss,
  DeltaGaussAlpha,
  CellCenteredGrad
};

static const char* forceIsolationPathName(ForceIsolationPath path) {
  switch (path) {
    case ForceIsolationPath::BalancedFaceCsf:
      return "balanced_face_csf";
    case ForceIsolationPath::BalancedPressureGradient:
      return "balanced_pressure_gradient";
    case ForceIsolationPath::FaceGaussAlpha:
      return "face_gauss_alpha";
    case ForceIsolationPath::FaceGaussSnPressureGradient:
      return "face_gauss_sn_pressure_gradient";
    case ForceIsolationPath::HybridMeanBalancedDeltaGauss:
      return "hybrid_mean_balanced_delta_gauss";
    case ForceIsolationPath::DeltaGaussAlpha:
      return "delta_gauss_alpha";
    case ForceIsolationPath::CellCenteredGrad:
      return "cell_centered_grad_alpha";
  }
  return "unknown";
}

static ForceIsolationReport runForceIsolationCase(int cellsPerDim, int mode,
                                                  double kappaMultiplier = 1.0,
                                                  const std::string& methodLabel =
                                                      "local_plic_quadric",
                                                  ForceIsolationPath forcePath =
                                                      ForceIsolationPath::BalancedFaceCsf,
                                                  bool reflectPerturbation = false,
                                                  int curvatureSamples = 36) {
  constexpr double radius = 0.25;
  constexpr double sigma = 0.05;
  constexpr double rhoIn = 1.0;
  constexpr double rhoOut = 1.0;
  constexpr double amplitudeRatio = 0.04;
  auto mesh = fvm::Mesh3D::hexGrid(cellsPerDim, cellsPerDim, cellsPerDim, 1.0, 1.0, 1.0, 0.03);
  double dx = std::cbrt(1.0 / static_cast<double>(mesh.cells.size()));
  double interfaceWidth = std::max(1.15 * dx, 0.025);
  fvm::ScalarField alpha = perturbedDropletAlpha(mesh, mode, radius, amplitudeRatio, interfaceWidth);

  auto local = fvm::curvatureFromLocalPlicQuadricReport3D(mesh, alpha, curvatureSamples);
  if (reflectPerturbation) {
    local.kappa = fvm::reflectedInterfaceKappaPerturbation3D(mesh, alpha, local.kappa);
  }
  if (kappaMultiplier != 1.0) {
    for (double& k : local.kappa) k *= kappaMultiplier;
  }
  fvm::VectorField3 acceleration(mesh.cells.size(), fvm::Vec3::Zero());
  double maxBalanceResidual = 0.0;
  if (forcePath == ForceIsolationPath::CellCenteredGrad) {
    fvm::VectorField3 gradAlpha = fvm::gradLeastSquares3D(mesh, alpha);
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const double rho = rhoOut + (rhoIn - rhoOut) * std::clamp(alpha[c], 0.0, 1.0);
      acceleration[c] = sigma * local.kappa[c] * gradAlpha[c] / std::max(rho, 1e-30);
    }
  } else if (forcePath == ForceIsolationPath::FaceGaussAlpha) {
    fvm::VectorField3 force = fvm::gaussAlphaCsfForce3D(mesh, alpha, sigma, &local.kappa);
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const double rho = rhoOut + (rhoIn - rhoOut) * std::clamp(alpha[c], 0.0, 1.0);
      acceleration[c] = force[c] / std::max(rho, 1e-30);
    }
  } else if (forcePath == ForceIsolationPath::FaceGaussSnPressureGradient) {
    fvm::ScalarField kappaF = fvm::faceInterpolate3D(mesh, local.kappa);
    fvm::ScalarField faceSource(mesh.faces.size(), 0.0);
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      const fvm::Face3D& f = mesh.faces[fi];
      const double alphaF = f.internal() ? 0.5 * (alpha[f.owner] + alpha[f.neighbour])
                                         : alpha[f.owner];
      faceSource[fi] = sigma * kappaF[fi] * alphaF;
    }
    fvm::ScalarField p = fvm::pressureFromSnGradFaceSource3D(mesh, faceSource);
    fvm::VectorField3 force = fvm::pressureGradientFromSnGrad3D(mesh, p);
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const double rho = rhoOut + (rhoIn - rhoOut) * std::clamp(alpha[c], 0.0, 1.0);
      acceleration[c] = force[c] / std::max(rho, 1e-30);
    }
  } else if (forcePath == ForceIsolationPath::HybridMeanBalancedDeltaGauss) {
    fvm::VectorField3 force =
        fvm::hybridMeanBalancedDeltaGaussCsfForce3D(mesh, alpha, sigma, &local.kappa);
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const double rho = rhoOut + (rhoIn - rhoOut) * std::clamp(alpha[c], 0.0, 1.0);
      acceleration[c] = force[c] / std::max(rho, 1e-30);
    }
  } else if (forcePath == ForceIsolationPath::DeltaGaussAlpha) {
    fvm::VectorField3 force = fvm::deltaGaussAlphaCsfForce3D(mesh, alpha, sigma, &local.kappa);
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const double rho = rhoOut + (rhoIn - rhoOut) * std::clamp(alpha[c], 0.0, 1.0);
      acceleration[c] = force[c] / std::max(rho, 1e-30);
    }
  } else {
    fvm::BalancedForceSurfaceTensionState3D state =
        fvm::buildBalancedForceSurfaceTensionState3D(mesh, alpha, sigma, &local.kappa);
    maxBalanceResidual = state.maxBalanceResidual;
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const double rho = rhoOut + (rhoIn - rhoOut) * std::clamp(alpha[c], 0.0, 1.0);
      const fvm::Vec3 source = forcePath == ForceIsolationPath::BalancedPressureGradient
                                   ? state.pressureGradient[c]
                                   : state.csfForce[c];
      acceleration[c] = source / std::max(rho, 1e-30);
    }
  }

  ForceIsolationReport r;
  r.curvatureMethod = methodLabel;
  r.forcePath = forceIsolationPathName(forcePath);
  r.cellsPerDim = cellsPerDim;
  r.mode = mode;
  r.shapeCoeff = interfaceModeCoeff(mesh, alpha, mode, radius);
  r.lambOmega = lambOmega(mode, radius, sigma, rhoIn, rhoOut);
  r.measuredAccelCoeff = radialAccelerationModeCoeff(mesh, alpha, acceleration, mode, radius);
  r.lambAccelCoeff = -radius * r.lambOmega * r.lambOmega * r.shapeCoeff;
  r.accelRelativeError = std::abs(r.measuredAccelCoeff - r.lambAccelCoeff) /
                         std::max(std::abs(r.lambAccelCoeff), 1e-30);
  r.maxBalanceResidual = maxBalanceResidual;
  r.fallbackFraction = local.fallbackFraction;
  r.conditionP95 = local.p95StencilCondition;
  r.conditionMax = local.maxStencilCondition;
  r.illConditionedFallbackCells = local.illConditionedFallbackCells;
  r.restoringSign = r.shapeCoeff * r.measuredAccelCoeff < 0.0;
  if (!std::isfinite(r.accelRelativeError)) {
    r.status = ValidationStatus::BLOCKED;
  } else if (r.restoringSign && r.accelRelativeError <= 0.20) {
    r.status = ValidationStatus::UPHELD;
  } else if (r.restoringSign && r.accelRelativeError <= 1.0) {
    r.status = ValidationStatus::APPROXIMATE;
  } else {
    r.status = ValidationStatus::DOWNGRADED;
  }
  return r;
}

static ValidationStatus classifyFrequency(double err, bool converging, int crossings) {
  (void)crossings;
  if (!std::isfinite(err)) return ValidationStatus::BLOCKED;
  if (err <= 0.05 && converging) return ValidationStatus::UPHELD;
  if (err <= 0.20) return ValidationStatus::APPROXIMATE;
  return ValidationStatus::DOWNGRADED;
}

static ValidationStatus classifyDamping(double err, int peaks) {
  if (!std::isfinite(err) || peaks < 3) return ValidationStatus::DOWNGRADED;
  if (err <= 0.20) return ValidationStatus::UPHELD;
  if (err <= 0.50) return ValidationStatus::APPROXIMATE;
  return ValidationStatus::DOWNGRADED;
}

static OscillationReport runOscillationCase(int cellsPerDim, int mode, int sampleStride = 1,
                                            const std::string& curvatureMethod = "iso_rdf",
                                            const std::string& diagnosticPath = "full_path",
                                            std::ofstream* historyCsv = nullptr) {
  constexpr double radius = 0.25;
  constexpr double sigma = 0.05;
  constexpr double rhoIn = 1.0;
  constexpr double rhoOut = 1.0;
  constexpr double muIn = 1.0e-2;
  constexpr double muOut = 1.0e-2;
  constexpr double amplitudeRatio = 0.04;

  auto mesh = fvm::Mesh3D::hexGrid(cellsPerDim, cellsPerDim, cellsPerDim, 1.0, 1.0, 1.0, 0.03);
  double dx = std::cbrt(1.0 / static_cast<double>(mesh.cells.size()));
  double interfaceWidth = std::max(1.15 * dx, 0.025);
  fvm::ScalarField alpha = perturbedDropletAlpha(mesh, mode, radius, amplitudeRatio, interfaceWidth);
  double initialMass = fvm::vofMass3D(mesh, alpha);

  fvm::ScalarField rho(mesh.cells.size(), 1.0);
  fvm::ScalarField rAU(mesh.cells.size(), 0.0);
  fvm::ScalarField p(mesh.cells.size(), 0.0);
  fvm::VectorField3 u(mesh.cells.size(), fvm::Vec3::Zero());
  if (diagnosticPath == "projection_only") {
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const auto& x = mesh.cells[c].centroid;
      u[c] = {std::sin(2.0 * M_PI * x.x()),
              0.25 * std::sin(2.0 * M_PI * x.y()),
              -0.15 * std::cos(2.0 * M_PI * x.z())};
    }
  }
  double capDt = fvm::capillaryTimeStepLimit3D(mesh, rhoOut, sigma);
  double dt = 0.35 * capDt;
  int steps = std::max(90, static_cast<int>(std::ceil(2.4 * 2.0 * M_PI / (lambOmega(mode, radius, sigma, rhoIn, rhoOut) * dt))));
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    rho[c] = rhoOut + (rhoIn - rhoOut) * std::clamp(alpha[c], 0.0, 1.0);
    rAU[c] = dt / rho[c];
  }

  OscillationReport report;
  report.diagnosticPath = diagnosticPath;
  report.curvatureMethod = curvatureMethod;
  report.n = mode;
  report.cellsPerDim = cellsPerDim;
  report.steps = steps;
  report.dt = dt;
  report.capillaryDt = capDt;
  report.amplitudeRatio = amplitudeRatio;
  report.lambOmega = lambOmega(mode, radius, sigma, rhoIn, rhoOut);
  report.prosperettiDamping = prosperettiLowOhDamping(mode, radius, muIn, muOut, rhoIn, rhoOut);

  std::vector<Sample> samples;
  samples.reserve(static_cast<size_t>(steps / std::max(sampleStride, 1)) + 1);
  fvm::RhieChowProjector3D projector(mesh, rAU);
  const bool runCurvature = diagnosticPath == "full_path" || diagnosticPath == "curvature_only";
  const bool runMomentum = diagnosticPath == "full_path";
  const bool runProjection = diagnosticPath == "full_path" || diagnosticPath == "projection_only";
  const bool runVof = diagnosticPath == "full_path" || diagnosticPath == "vof_only";
  const fvm::ScalarField prescribedVofFlux = fvm::divergenceFreeBoxFlux3D(mesh, 0.02);
  for (int step = 0; step <= steps; ++step) {
    if (step % sampleStride == 0) {
      DynamicSnapshot snapshot =
          dynamicSnapshot(mesh, alpha, mode, radius, step, dt, initialMass, report.maxDiv,
                          curvatureMethod);
      samples.push_back({snapshot.t, snapshot.coeff, snapshot.isoCoeff,
                         snapshot.gradCoeff, snapshot.moment});
      if (historyCsv != nullptr) {
        writeHistoryRow(*historyCsv, diagnosticPath, curvatureMethod, cellsPerDim, mode, snapshot);
      }
    }
    if (step == steps) break;

    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      rho[c] = rhoOut + (rhoIn - rhoOut) * std::clamp(alpha[c], 0.0, 1.0);
      rAU[c] = dt / rho[c];
    }
    fvm::VectorField3 source(mesh.cells.size(), fvm::Vec3::Zero());
    if (runCurvature) {
      fvm::ScalarField alphaKappa = fvm::curvatureFromAlpha3D(mesh, alpha);
      fvm::ScalarField kappa;
      if (curvatureMethod == "local_plic_quadric") {
        auto local = fvm::curvatureFromLocalPlicQuadricReport3D(mesh, alpha, 36);
        kappa = local.kappa;
        report.maxCurvatureFallbackFraction =
            std::max(report.maxCurvatureFallbackFraction, local.fallbackFraction);
        report.maxCurvatureFittedCells =
            std::max(report.maxCurvatureFittedCells, local.fittedCells);
        report.maxCurvatureClampCells =
            std::max(report.maxCurvatureClampCells, local.curvatureClampCells);
        report.maxCurvatureStencilCondition =
            std::max(report.maxCurvatureStencilCondition, local.maxStencilCondition);
        report.maxCurvatureStencilConditionP95 =
            std::max(report.maxCurvatureStencilConditionP95, local.p95StencilCondition);
        report.maxCurvatureIllConditionedFallbackCells =
            std::max(report.maxCurvatureIllConditionedFallbackCells,
                     local.illConditionedFallbackCells);
      } else {
        kappa = fvm::curvatureFromIsoRDF3D(mesh, alpha, 2);
      }
      for (size_t c = 0; c < kappa.size(); ++c) {
        report.maxIsoRdfAlphaKappaDifference =
            std::max(report.maxIsoRdfAlphaKappaDifference, std::abs(kappa[c] - alphaKappa[c]));
      }
      fvm::BalancedForceSurfaceTensionState3D state =
          fvm::buildBalancedForceSurfaceTensionState3D(mesh, alpha, sigma, &kappa);
      report.maxSnGradDifference = std::max(report.maxSnGradDifference, state.maxSnGradDifference);
      for (size_t c = 0; c < mesh.cells.size(); ++c) {
        source[c] = state.csfForce[c];
      }
    }

    if (runMomentum) {
      fvm::MomentumPredictorReport3D momentum =
          fvm::solveMomentumPredictorBiCGSTABILUT3D(mesh, u, source, rho, dt,
                                                    0.5 * (muIn + muOut));
      u = momentum.velocity;
      report.maxMomentumResidual = std::max(report.maxMomentumResidual, momentum.maxResidual);
    }

    fvm::ScalarField transportFlux = prescribedVofFlux;
    if (runProjection) {
      fvm::CouplingReport3D projection = projector.project(u, p, 0.85);
      report.maxDiv = std::max(report.maxDiv, projection.maxDiv);
      transportFlux = projection.faceFlux;
    } else if (runVof) {
      fvm::ScalarField div = fvm::explicitDivFaceFlux3D(mesh, transportFlux);
      for (double d : div) report.maxDiv = std::max(report.maxDiv, std::abs(d));
    }

    if (runVof) {
      fvm::VofTransportOptions3D opt;
      opt.scheme = fvm::VofAdvectionScheme3D::IsoAdvector;
      opt.tvdBlend = 1.0;
      opt.compression = 0.0;
      opt.correctionSweeps = 4;
      fvm::VofTransportReport3D vof = fvm::advectVof3D(mesh, alpha, transportFlux, dt, opt);
      report.massDrift = std::max(report.massDrift, vof.relativeMassDrift);
      report.minAlpha = vof.minAlpha;
      report.maxAlpha = vof.maxAlpha;
    } else {
      auto [amin, amax] = fvm::vofBounds3D(alpha);
      report.minAlpha = amin;
      report.maxAlpha = amax;
    }
  }

  std::vector<Sample> measuredSamples = smoothedSamples(samples, &Sample::coeff, 2);
  report.measuredOmega =
      zeroCrossingOmega(measuredSamples, &Sample::coeff, report.zeroCrossings);
  report.fftOmega = dftOmega(measuredSamples, &Sample::coeff);
  if (!std::isfinite(report.measuredOmega) && std::isfinite(report.fftOmega)) {
    report.measuredOmega = report.fftOmega;
  }
  int isoCrossings = 0;
  std::vector<Sample> isoSamples = smoothedSamples(samples, &Sample::isoCoeff, 2);
  report.isoOmega = zeroCrossingOmega(isoSamples, &Sample::isoCoeff, isoCrossings);
  if (!std::isfinite(report.isoOmega)) report.isoOmega = dftOmega(isoSamples, &Sample::isoCoeff);
  int momentCrossings = 0;
  report.momentOmega = zeroCrossingOmega(samples, &Sample::moment, momentCrossings);
  if (!std::isfinite(report.momentOmega)) report.momentOmega = dftOmega(samples, &Sample::moment);
  if (!std::isfinite(report.momentOmega)) {
    int gradCrossings = 0;
    report.momentOmega = zeroCrossingOmega(samples, &Sample::gradCoeff, gradCrossings);
  }
  if (!std::isfinite(report.momentOmega)) report.momentOmega = dftOmega(samples, &Sample::gradCoeff);
  report.measuredDamping =
      dampingRateFromEnvelope(measuredSamples, &Sample::coeff, report.peaks);
  report.omegaError = std::abs(report.measuredOmega - report.lambOmega) /
                      std::max(std::abs(report.lambOmega), 1e-30);
  report.dampingError = std::abs(report.measuredDamping - report.prosperettiDamping) /
                        std::max(std::abs(report.prosperettiDamping), 1e-30);
  (void)initialMass;
  return report;
}

static void writeCase(std::ofstream& csv, const OscillationReport& r) {
  csv << r.diagnosticPath << "," << r.curvatureMethod << "," << r.cellsPerDim << "," << r.n << "," << r.steps << "," << r.dt << ","
      << r.capillaryDt << "," << r.amplitudeRatio << "," << r.lambOmega << ","
      << r.measuredOmega << "," << r.fftOmega << "," << r.momentOmega << ","
      << r.isoOmega << ","
      << r.omegaError << "," << r.prosperettiDamping << "," << r.measuredDamping
      << "," << r.dampingError << "," << r.zeroCrossings << "," << r.peaks << ","
      << r.maxDiv << "," << r.maxMomentumResidual << "," << r.maxSnGradDifference << ","
      << r.maxIsoRdfAlphaKappaDifference << "," << r.maxCurvatureFallbackFraction << ","
      << r.maxCurvatureFittedCells << "," << r.maxCurvatureClampCells << ","
      << r.maxCurvatureStencilCondition << "," << r.maxCurvatureStencilConditionP95 << ","
      << r.maxCurvatureIllConditionedFallbackCells << ","
      << r.massDrift << "," << r.minAlpha << ","
      << r.maxAlpha << ","
      << statusName(r.frequencyStatus) << "," << statusName(r.dampingStatus) << "\n";
}

int main() {
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/dynamic_droplet_oscillation3d.csv");
  csv << "diagnostic_path,curvature_method,grid_n,mode,steps,dt,capillary_dt,amplitude_ratio,lamb_omega,measured_omega,"
         "fft_omega,moment_omega,iso_omega,omega_rel_error,prosperetti_damping,measured_damping,"
         "damping_rel_error,zero_crossings,peaks,max_div,max_momentum_residual,"
         "max_snGrad_difference,max_iso_rdf_alpha_kappa_difference,max_curvature_fallback_fraction,"
         "max_curvature_fitted_cells,max_curvature_clamp_cells,max_curvature_stencil_condition,"
         "max_curvature_stencil_condition_p95,max_curvature_ill_conditioned_fallback_cells,"
         "mass_drift,min_alpha,"
         "max_alpha,frequency_status,damping_status\n";
  std::ofstream historyCsv("benchmark_logs/dynamic_droplet_oscillation3d_history.csv");
  historyCsv << "diagnostic_path,curvature_method,grid_n,mode,step,t,mode_coeff,iso_mode_coeff,"
                "grad_mode_coeff,moment_metric,mass_drift,min_alpha,max_alpha,mixed_cells,"
                "interface_thickness,curvature_condition_p95,curvature_condition_max,"
                "curvature_fallback_fraction,curvature_fitted_cells,curvature_clamp_cells,"
                "curvature_ill_conditioned_fallback_cells,max_div\n";

  double rayleighLimit = 8.0 * 0.05 / (1.0 * 0.25 * 0.25 * 0.25);
  double lambVacuumSquared = std::pow(lambOmega(2, 0.25, 0.05, 1.0, 0.0), 2.0);
  check(std::abs(lambVacuumSquared - rayleighLimit) / rayleighLimit < 1e-14,
        "Lamb mode-2 formula recovers Rayleigh rho_out->0 limit");

  std::vector<OscillationReport> mode2;
  for (int n : {8, 10, 12}) mode2.push_back(runOscillationCase(n, 2, 1, "iso_rdf", "full_path", &historyCsv));
  bool mode2Converging = std::isfinite(mode2[0].omegaError) &&
                         std::isfinite(mode2[1].omegaError) &&
                         std::isfinite(mode2[2].omegaError) &&
                         mode2[2].omegaError < mode2[1].omegaError &&
                         mode2[1].omegaError < mode2[0].omegaError;
  for (OscillationReport& r : mode2) {
    r.frequencyStatus = classifyFrequency(r.omegaError, mode2Converging, r.zeroCrossings);
    r.dampingStatus = classifyDamping(r.dampingError, r.peaks);
    writeCase(csv, r);
  }

  std::vector<OscillationReport> mode3;
  for (int n : {8, 10}) mode3.push_back(runOscillationCase(n, 3, 1, "iso_rdf", "full_path", &historyCsv));
  bool mode3Comparable = std::isfinite(mode3.back().omegaError);
  for (OscillationReport& r : mode3) {
    r.frequencyStatus = classifyFrequency(r.omegaError, mode3Comparable, r.zeroCrossings);
    r.dampingStatus = classifyDamping(r.dampingError, r.peaks);
    writeCase(csv, r);
  }

  OscillationReport denseOutput = runOscillationCase(10, 2, 1, "iso_rdf", "full_path", &historyCsv);
  OscillationReport sparseOutput = runOscillationCase(10, 2, 2, "iso_rdf", "full_path", &historyCsv);
  denseOutput.frequencyStatus =
      classifyFrequency(denseOutput.omegaError, true, denseOutput.zeroCrossings);
  denseOutput.dampingStatus = classifyDamping(denseOutput.dampingError, denseOutput.peaks);
  sparseOutput.frequencyStatus =
      classifyFrequency(sparseOutput.omegaError, true, sparseOutput.zeroCrossings);
  sparseOutput.dampingStatus = classifyDamping(sparseOutput.dampingError, sparseOutput.peaks);
  writeCase(csv, denseOutput);
  writeCase(csv, sparseOutput);

  std::vector<OscillationReport> localQuadricCases;
  localQuadricCases.push_back(runOscillationCase(10, 2, 2, "local_plic_quadric", "full_path", &historyCsv));
  localQuadricCases.push_back(runOscillationCase(10, 3, 2, "local_plic_quadric", "full_path", &historyCsv));
  for (OscillationReport& r : localQuadricCases) {
    r.frequencyStatus = classifyFrequency(r.omegaError, false, r.zeroCrossings);
    r.dampingStatus = classifyDamping(r.dampingError, r.peaks);
    writeCase(csv, r);
  }

  std::vector<OscillationReport> decompositionCases;
  decompositionCases.push_back(runOscillationCase(8, 2, 2, "iso_rdf", "vof_only", &historyCsv));
  decompositionCases.push_back(runOscillationCase(8, 2, 2, "local_plic_quadric", "curvature_only", &historyCsv));
  decompositionCases.push_back(runOscillationCase(8, 2, 2, "none", "projection_only", &historyCsv));
  for (OscillationReport& r : decompositionCases) {
    r.frequencyStatus = classifyFrequency(r.omegaError, false, r.zeroCrossings);
    r.dampingStatus = classifyDamping(r.dampingError, r.peaks);
    writeCase(csv, r);
  }

  std::vector<ForceIsolationReport> forceCases;
  for (int mode : {2, 3}) {
    for (int n : {8, 10, 12}) {
      forceCases.push_back(runForceIsolationCase(n, mode));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_reflected_perturbation",
                                                 ForceIsolationPath::BalancedFaceCsf,
                                                 true));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_reflected_samples_12",
                                                 ForceIsolationPath::BalancedFaceCsf,
                                                 true, 12));
      forceCases.push_back(runForceIsolationCase(n, mode, -1.0,
                                                 "local_plic_quadric_negated_kappa"));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_pressure_gradient",
                                                 ForceIsolationPath::BalancedPressureGradient));
      forceCases.push_back(runForceIsolationCase(n, mode, -1.0,
                                                 "local_plic_quadric_negated_pressure_gradient",
                                                 ForceIsolationPath::BalancedPressureGradient));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_face_gauss_alpha",
                                                 ForceIsolationPath::FaceGaussAlpha));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_face_gauss_sn_pressure",
                                                 ForceIsolationPath::FaceGaussSnPressureGradient));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_reflected_face_gauss",
                                                 ForceIsolationPath::FaceGaussAlpha,
                                                 true));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_reflected_face_gauss_sn_pressure",
                                                 ForceIsolationPath::FaceGaussSnPressureGradient,
                                                 true));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_reflected_samples_12_face_gauss",
                                                 ForceIsolationPath::FaceGaussAlpha,
                                                 true, 12));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_reflected_samples_12_face_gauss_sn_pressure",
                                                 ForceIsolationPath::FaceGaussSnPressureGradient,
                                                 true, 12));
      forceCases.push_back(runForceIsolationCase(n, mode, -1.0,
                                                 "local_plic_quadric_negated_face_gauss_alpha",
                                                 ForceIsolationPath::FaceGaussAlpha));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_hybrid_mean_delta",
                                                 ForceIsolationPath::HybridMeanBalancedDeltaGauss));
      forceCases.push_back(runForceIsolationCase(n, mode, -1.0,
                                                 "local_plic_quadric_negated_hybrid_mean_delta",
                                                 ForceIsolationPath::HybridMeanBalancedDeltaGauss));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_delta_gauss",
                                                 ForceIsolationPath::DeltaGaussAlpha));
      forceCases.push_back(runForceIsolationCase(n, mode, -1.0,
                                                 "local_plic_quadric_negated_delta_gauss",
                                                 ForceIsolationPath::DeltaGaussAlpha));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_cell_grad_force",
                                                 ForceIsolationPath::CellCenteredGrad));
      forceCases.push_back(runForceIsolationCase(n, mode, 1.0,
                                                 "local_plic_quadric_reflected_samples_12_cell_grad",
                                                 ForceIsolationPath::CellCenteredGrad,
                                                 true, 12));
      forceCases.push_back(runForceIsolationCase(n, mode, -1.0,
                                                 "local_plic_quadric_negated_cell_grad_force",
                                                 ForceIsolationPath::CellCenteredGrad));
    }
  }
  std::vector<CurvatureModeReport> curvatureModeCases;
  for (int mode : {2, 3}) {
    for (int n : {8, 10, 12}) {
      curvatureModeCases.push_back(runCurvatureModeCase(n, mode));
      curvatureModeCases.push_back(runCurvatureModeCase(n, mode, 1.0,
                                                        "local_plic_quadric_reflected_perturbation",
                                                        false, true));
      curvatureModeCases.push_back(runCurvatureModeCase(n, mode, -1.0,
                                                        "local_plic_quadric_negated_kappa"));
      for (int samples : {12, 18, 24, 48}) {
        curvatureModeCases.push_back(runCurvatureModeCase(
            n, mode, 1.0, "local_plic_quadric_samples_" + std::to_string(samples),
            false, false, samples));
        curvatureModeCases.push_back(runCurvatureModeCase(
            n, mode, 1.0,
            "local_plic_quadric_reflected_samples_" + std::to_string(samples),
            false, true, samples));
      }
      curvatureModeCases.push_back(runCurvatureModeCase(n, mode, 1.0,
                                                        "local_plic_height_quadric",
                                                        true));
      curvatureModeCases.push_back(runCurvatureModeCase(n, mode, -1.0,
                                                        "local_plic_height_quadric_negated_kappa",
                                                        true));
    }
  }
  std::ofstream curvatureModeCsv("benchmark_logs/dynamic_droplet_curvature_mode3d.csv");
  curvatureModeCsv << "curvature_method,grid_n,mode,shape_coeff,measured_kappa_coeff,"
                      "analytic_kappa_coeff,kappa_rel_error,curvature_sign,"
                      "fallback_fraction,condition_p95,condition_max,status\n";
  for (const auto& k : curvatureModeCases) {
    curvatureModeCsv << k.curvatureMethod << "," << k.cellsPerDim << "," << k.mode << ","
                     << k.shapeCoeff << "," << k.measuredKappaCoeff << ","
                     << k.analyticKappaCoeff << "," << k.relativeError << ","
                     << (k.curvatureSign ? 1 : 0) << "," << k.fallbackFraction << ","
                     << k.conditionP95 << "," << k.conditionMax << ","
                     << statusName(k.status) << "\n";
  }
  std::ofstream forceCsv("benchmark_logs/dynamic_droplet_force_isolation3d.csv");
  forceCsv << "curvature_method,force_path,grid_n,mode,shape_coeff,lamb_omega,measured_accel_coeff,"
              "lamb_accel_coeff,accel_rel_error,restoring_sign,fallback_fraction,"
              "condition_p95,condition_max,ill_conditioned_fallback_cells,"
              "max_balance_residual,status\n";
  for (const auto& f : forceCases) {
    forceCsv << f.curvatureMethod << "," << f.forcePath << "," << f.cellsPerDim << ","
             << f.mode << ","
             << f.shapeCoeff << "," << f.lambOmega << "," << f.measuredAccelCoeff << ","
             << f.lambAccelCoeff << "," << f.accelRelativeError << ","
             << (f.restoringSign ? 1 : 0) << "," << f.fallbackFraction << ","
             << f.conditionP95 << "," << f.conditionMax << ","
             << f.illConditionedFallbackCells << "," << f.maxBalanceResidual << ","
             << statusName(f.status) << "\n";
  }

  double dispersionMeasured = mode3.back().measuredOmega / std::max(mode2.back().measuredOmega, 1e-30);
  double dispersionLamb = mode3.back().lambOmega / std::max(mode2.back().lambOmega, 1e-30);
  double dispersionError = std::abs(dispersionMeasured - dispersionLamb) /
                           std::max(std::abs(dispersionLamb), 1e-30);
  bool artifactReviewFinite = std::isfinite(denseOutput.measuredOmega) &&
                              std::isfinite(sparseOutput.measuredOmega) &&
                              std::isfinite(denseOutput.momentOmega) &&
                              std::isfinite(denseOutput.isoOmega);
  double outputCadenceDelta = artifactReviewFinite
                                  ? std::abs(denseOutput.measuredOmega - sparseOutput.measuredOmega) /
                                        std::max(std::abs(denseOutput.measuredOmega), 1e-30)
                                  : std::numeric_limits<double>::quiet_NaN();
  double secondMetricDelta = artifactReviewFinite
                                 ? std::abs(denseOutput.measuredOmega - denseOutput.momentOmega) /
                                       std::max(std::abs(denseOutput.measuredOmega), 1e-30)
                                 : std::numeric_limits<double>::quiet_NaN();
  double isoMetricDelta = artifactReviewFinite
                              ? std::abs(denseOutput.measuredOmega - denseOutput.isoOmega) /
                                    std::max(std::abs(denseOutput.measuredOmega), 1e-30)
                              : std::numeric_limits<double>::quiet_NaN();
  double strictSecondMetricDelta =
      std::max(std::isfinite(secondMetricDelta) ? secondMetricDelta : 0.0,
               std::isfinite(isoMetricDelta) ? isoMetricDelta : 0.0);

  ValidationStatus dispersionStatus =
      (!std::isfinite(dispersionError)) ? ValidationStatus::BLOCKED
      : (dispersionError <= 0.05)       ? ValidationStatus::UPHELD
      : (dispersionError <= 0.20)       ? ValidationStatus::APPROXIMATE
                                        : ValidationStatus::DOWNGRADED;
  ValidationStatus artifactStatus =
      (!artifactReviewFinite) ? ValidationStatus::BLOCKED
      : (outputCadenceDelta <= 0.05 && strictSecondMetricDelta <= 0.15)
          ? ValidationStatus::UPHELD
      : (outputCadenceDelta <= 0.15 && strictSecondMetricDelta <= 0.30)
          ? ValidationStatus::APPROXIMATE
          : ValidationStatus::DOWNGRADED;

  std::ofstream md("benchmark_logs/dynamic_droplet_oscillation3d_report.md");
  md << "# Dynamic 3D Oscillating-Droplet Validation\n\n";
  md << "Path: curvatureFromIsoRDF3D/local_plic_quadric -> buildBalancedForceSurfaceTensionState3D -> "
        "solveMomentumPredictorBiCGSTABILUT3D -> RhieChowProjector3D -> exact-swept-PLIC advectVof3D.\n";
  md << "Curvature paths: RDF kappa=-div(grad(psi)/|grad(psi)|), plus an additive "
        "local PLIC point/normal weighted shape-operator fit. The local path reports fallback_fraction.\n\n";
  md << "Frequency/damping metric: Legendre coefficient measured from the diffuse-alpha "
        "interface band; reconstructed iso-interface centroids and moments are retained for "
        "adversarial checks.\n\n";
  md << "Amplitude ratio eps/R = 0.04; gravity off; surface tension is the only restoring force. "
        "The momentum predictor viscosity and Prosperetti reference viscosity are both 0.01.\n\n";
  md << "| path | method | grid | mode | omega_measured | omega_Lamb | omega_error | damping_measured | damping_ref | damping_error | fallback | fitted | clamp | cond_p95 | cond_max | ill_cond_fallback | freq_status | damp_status |\n";
  md << "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n";
  auto writeMd = [&](const OscillationReport& r) {
    md << "|" << r.diagnosticPath << "|" << r.curvatureMethod << "|" << r.cellsPerDim << "|" << r.n << "|" << r.measuredOmega << "|"
       << r.lambOmega << "|" << r.omegaError << "|" << r.measuredDamping << "|"
       << r.prosperettiDamping << "|" << r.dampingError << "|"
       << r.maxCurvatureFallbackFraction << "|"
       << r.maxCurvatureFittedCells << "|" << r.maxCurvatureClampCells << "|"
       << r.maxCurvatureStencilConditionP95 << "|" << r.maxCurvatureStencilCondition << "|"
       << r.maxCurvatureIllConditionedFallbackCells << "|"
       << statusName(r.frequencyStatus) << "|" << statusName(r.dampingStatus) << "|\n";
  };
  for (const auto& r : mode2) writeMd(r);
  for (const auto& r : mode3) writeMd(r);
  for (const auto& r : localQuadricCases) writeMd(r);
  for (const auto& r : decompositionCases) writeMd(r);
  md << "\nDispersion mode3/mode2 measured=" << dispersionMeasured
     << " Lamb=" << dispersionLamb << " error=" << dispersionError
     << " status=" << statusName(dispersionStatus) << "\n\n";
  md << "Adversarial review: output_cadence_delta=" << outputCadenceDelta
     << ", moment_metric_delta=" << secondMetricDelta
     << ", iso_metric_delta=" << isoMetricDelta
     << ", status=" << statusName(artifactStatus) << "\n\n";
  md << "Cause decomposition rows: `vof_only` isolates exact swept PLIC transport under a "
        "prescribed divergence-free flux; `curvature_only` computes the local reconstructed "
        "CSF source without moving alpha; `projection_only` projects a synthetic velocity "
        "without curvature or VoF advection.\n\n";
  md << "Force-isolation rows project the frozen-alpha CSF acceleration onto the same "
        "Legendre mode and compare it against `-R*omega_Lamb^2*mode_coeff`; this separates "
        "curvature-force sign/magnitude from VoF transport and pressure projection.\n\n";
  md << "Curvature-mode rows compare the local reconstructed kappa Legendre coefficient "
        "against the small-perturbation sphere coefficient before force assembly.\n\n";
  md << "| kappa_method | grid | mode | kappa_measured | kappa_ref | kappa_error | sign | fallback | cond_p95 | cond_max | status |\n";
  md << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
  for (const auto& k : curvatureModeCases) {
    md << "|" << k.curvatureMethod << "|" << k.cellsPerDim << "|" << k.mode << "|"
       << k.measuredKappaCoeff << "|" << k.analyticKappaCoeff << "|"
       << k.relativeError << "|" << (k.curvatureSign ? 1 : 0) << "|"
       << k.fallbackFraction << "|" << k.conditionP95 << "|" << k.conditionMax << "|"
       << statusName(k.status) << "|\n";
  }
  md << "\n";
  md << "| force_method | grid | mode | accel_measured | accel_Lamb | accel_error | restoring | fallback | cond_p95 | cond_max | status |\n";
  md << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
  for (const auto& f : forceCases) {
    md << "|" << f.curvatureMethod << "|" << f.cellsPerDim << "|" << f.mode << "|"
       << f.measuredAccelCoeff << "|" << f.lambAccelCoeff << "|"
       << f.accelRelativeError << "|" << (f.restoringSign ? 1 : 0) << "|"
       << f.fallbackFraction << "|" << f.conditionP95 << "|" << f.conditionMax << "|"
       << statusName(f.status) << "|\n";
  }
  md << "\n";
  md << "Static Ca~1e-11 context: this dynamic fixture exercises alpha-derived curvature, "
        "momentum, projection, and VoF advection; it should supersede the static proxy when "
        "judging curvature dynamics.\n";

  auto statusAssigned = [](ValidationStatus s) {
    return s == ValidationStatus::UPHELD || s == ValidationStatus::APPROXIMATE ||
           s == ValidationStatus::DOWNGRADED || s == ValidationStatus::BLOCKED;
  };
  for (const auto& r : mode2) {
    check(statusAssigned(r.frequencyStatus), "mode-2 frequency status assigned");
    check(statusAssigned(r.dampingStatus), "mode-2 damping status assigned");
    check(r.amplitudeRatio <= 0.05, "mode-2 amplitude remains in requested linear range");
    check(r.dt <= r.capillaryDt * (1.0 + 1e-14), "mode-2 capillary timestep limit obeyed");
    check(r.maxSnGradDifference == 0.0, "mode-2 identical snGrad invariant holds");
    check(r.minAlpha >= -1e-14 && r.maxAlpha <= 1.0 + 1e-14, "mode-2 VoF boundedness");
  }
  for (const auto& r : mode3) {
    check(statusAssigned(r.frequencyStatus), "mode-3 frequency status assigned");
    check(statusAssigned(r.dampingStatus), "mode-3 damping status assigned");
    check(r.amplitudeRatio <= 0.05, "mode-3 amplitude remains in requested linear range");
    check(r.dt <= r.capillaryDt * (1.0 + 1e-14), "mode-3 capillary timestep limit obeyed");
    check(r.maxSnGradDifference == 0.0, "mode-3 identical snGrad invariant holds");
    check(r.minAlpha >= -1e-14 && r.maxAlpha <= 1.0 + 1e-14, "mode-3 VoF boundedness");
  }
  for (const auto& r : localQuadricCases) {
    check(statusAssigned(r.frequencyStatus), "local PLIC quadric frequency status assigned");
    check(statusAssigned(r.dampingStatus), "local PLIC quadric damping status assigned");
    check(r.maxCurvatureFallbackFraction < 0.25, "local PLIC quadric dynamic fallback bounded");
    check(r.maxSnGradDifference == 0.0, "local PLIC quadric identical snGrad invariant holds");
    check(r.minAlpha >= -1e-14 && r.maxAlpha <= 1.0 + 1e-14, "local PLIC quadric VoF boundedness");
  }
  for (const auto& r : decompositionCases) {
    check(statusAssigned(r.frequencyStatus), "decomposition frequency status assigned");
    check(statusAssigned(r.dampingStatus), "decomposition damping status assigned");
    check(r.dt <= r.capillaryDt * (1.0 + 1e-14), "decomposition capillary timestep limit obeyed");
    check(r.minAlpha >= -1e-14 && r.maxAlpha <= 1.0 + 1e-14, "decomposition VoF boundedness");
  }
  for (const auto& f : forceCases) {
    check(statusAssigned(f.status), "force-isolation status assigned");
    check(std::isfinite(f.measuredAccelCoeff), "force-isolation measured acceleration finite");
    check(std::isfinite(f.lambAccelCoeff), "force-isolation Lamb acceleration finite");
    check(f.fallbackFraction >= 0.0 && f.fallbackFraction <= 1.0,
          "force-isolation fallback fraction bounded");
  }
  for (const auto& k : curvatureModeCases) {
    check(statusAssigned(k.status), "curvature-mode status assigned");
    check(std::isfinite(k.measuredKappaCoeff), "curvature-mode measured kappa finite");
    check(std::isfinite(k.analyticKappaCoeff), "curvature-mode analytic kappa finite");
    check(k.fallbackFraction >= 0.0 && k.fallbackFraction <= 1.0,
          "curvature-mode fallback fraction bounded");
  }
  check(statusAssigned(dispersionStatus), "mode dispersion status assigned");
  check(statusAssigned(artifactStatus), "adversarial artifact-review status assigned");

  std::cout << "dynamic_droplet_mode2_fine_omega_error=" << mode2.back().omegaError
            << " mode2_frequency_status=" << statusName(mode2.back().frequencyStatus)
            << " mode3_omega_error=" << mode3.back().omegaError
            << " mode3_frequency_status=" << statusName(mode3.back().frequencyStatus)
            << " damping_error=" << mode2.back().dampingError
            << " damping_status=" << statusName(mode2.back().dampingStatus)
            << " dispersion_error=" << dispersionError
            << " dispersion_status=" << statusName(dispersionStatus)
            << " artifact_status=" << statusName(artifactStatus) << "\n";
}
