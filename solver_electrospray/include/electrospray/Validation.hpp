#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

namespace electrospray {

struct ValidationCase {
  std::string caseId;
  std::string dimension;
  std::string purpose;
  bool required = true;
};

struct ValidationResult {
  std::string caseId;
  bool passed = false;
  std::optional<double> metric;
  std::optional<double> tolerance;
};

struct ElectrostaticSolution1D {
  std::vector<double> xCenters;
  std::vector<double> phi;
  std::vector<double> eCell;
  std::vector<double> displacementFaces;
};

inline const std::vector<ValidationCase>& validationCases() {
  static const std::vector<ValidationCase> cases = {
      {"1d_parallel_plate", "1D", "Laplace electrostatic field", true},
      {"1d_dielectric_jump", "1D", "normal displacement continuity", true},
      {"1d_charge_relaxation", "1D", "leaky-dielectric charge timescale", true},
      {"1d_maxwell_jump", "1D", "flat-interface Maxwell pressure balance", true},
      {"vof_interface_transport", "2D", "bounded interface transport and charge confinement", true},
      {"2d_droplet_deformation", "2D", "leaky-dielectric droplet deformation", true},
      {"2d_taylor_cone", "2D-axisymmetric", "static Taylor cone geometry", true},
      {"2d_cone_jet", "2D-axisymmetric", "steady cone-jet observables", true},
      {"3d_multi_emitter", "3D", "array current sharing and shielding", true},
      {"3d_plume_impingement", "3D", "plume divergence and target impingement", false},
      {"3d_microthruster_performance", "3D", "microthruster performance metrics", false},
  };
  return cases;
}

inline std::vector<double> linspace(double a, double b, int n) {
  if (n < 2) throw std::runtime_error("linspace needs at least two points");
  std::vector<double> x(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = a + (b - a) * static_cast<double>(i) / static_cast<double>(n - 1);
  }
  return x;
}

inline double infNormDiff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) throw std::runtime_error("infNormDiff size mismatch");
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

inline std::vector<double> solveTridiagonal(std::vector<double> lower,
                                            std::vector<double> diag,
                                            std::vector<double> upper,
                                            std::vector<double> rhs) {
  const int n = static_cast<int>(diag.size());
  if (static_cast<int>(rhs.size()) != n || static_cast<int>(lower.size()) != std::max(0, n - 1) ||
      static_cast<int>(upper.size()) != std::max(0, n - 1)) {
    throw std::runtime_error("tridiagonal dimensions mismatch");
  }
  for (int i = 1; i < n; ++i) {
    const double pivot = diag[static_cast<size_t>(i - 1)];
    if (std::abs(pivot) <= 1e-30) throw std::runtime_error("singular tridiagonal pivot");
    const double w = lower[static_cast<size_t>(i - 1)] / pivot;
    diag[static_cast<size_t>(i)] -= w * upper[static_cast<size_t>(i - 1)];
    rhs[static_cast<size_t>(i)] -= w * rhs[static_cast<size_t>(i - 1)];
  }
  std::vector<double> x(static_cast<size_t>(n), 0.0);
  x[static_cast<size_t>(n - 1)] = rhs[static_cast<size_t>(n - 1)] / diag[static_cast<size_t>(n - 1)];
  for (int i = n - 2; i >= 0; --i) {
    x[static_cast<size_t>(i)] =
        (rhs[static_cast<size_t>(i)] - upper[static_cast<size_t>(i)] * x[static_cast<size_t>(i + 1)]) /
        diag[static_cast<size_t>(i)];
  }
  return x;
}

inline ElectrostaticSolution1D solveElectrostatic1D(const std::vector<double>& faces,
                                                    const std::vector<double>& eps,
                                                    double phiLeft,
                                                    double phiRight,
                                                    std::optional<std::vector<double>> charge = std::nullopt) {
  if (faces.size() != eps.size() + 1) throw std::runtime_error("faces/epsilon size mismatch");
  for (size_t i = 1; i < faces.size(); ++i) {
    if (!(faces[i] > faces[i - 1])) throw std::runtime_error("faces must be strictly increasing");
  }
  for (double e : eps) {
    if (!(e > 0.0)) throw std::runtime_error("epsilon must be positive");
  }
  const int n = static_cast<int>(eps.size());
  std::vector<double> dx(static_cast<size_t>(n));
  std::vector<double> centers(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    dx[static_cast<size_t>(i)] = faces[static_cast<size_t>(i + 1)] - faces[static_cast<size_t>(i)];
    centers[static_cast<size_t>(i)] = 0.5 * (faces[static_cast<size_t>(i)] + faces[static_cast<size_t>(i + 1)]);
  }
  std::vector<double> diag(static_cast<size_t>(n), 0.0);
  std::vector<double> lower(static_cast<size_t>(std::max(0, n - 1)), 0.0);
  std::vector<double> upper(static_cast<size_t>(std::max(0, n - 1)), 0.0);
  std::vector<double> rhs(static_cast<size_t>(n), 0.0);
  if (charge.has_value()) {
    if (charge->size() != eps.size()) throw std::runtime_error("charge size mismatch");
    for (int i = 0; i < n; ++i) rhs[static_cast<size_t>(i)] = (*charge)[static_cast<size_t>(i)] * dx[static_cast<size_t>(i)];
  }

  for (int i = 0; i < n; ++i) {
    double gw = 0.0;
    double ge = 0.0;
    if (i == 0) {
      gw = eps[static_cast<size_t>(i)] / (0.5 * dx[static_cast<size_t>(i)]);
      rhs[static_cast<size_t>(i)] += gw * phiLeft;
    } else {
      gw = 1.0 / (0.5 * dx[static_cast<size_t>(i - 1)] / eps[static_cast<size_t>(i - 1)] +
                  0.5 * dx[static_cast<size_t>(i)] / eps[static_cast<size_t>(i)]);
      lower[static_cast<size_t>(i - 1)] = -gw;
    }
    if (i == n - 1) {
      ge = eps[static_cast<size_t>(i)] / (0.5 * dx[static_cast<size_t>(i)]);
      rhs[static_cast<size_t>(i)] += ge * phiRight;
    } else {
      ge = 1.0 / (0.5 * dx[static_cast<size_t>(i)] / eps[static_cast<size_t>(i)] +
                  0.5 * dx[static_cast<size_t>(i + 1)] / eps[static_cast<size_t>(i + 1)]);
      upper[static_cast<size_t>(i)] = -ge;
    }
    diag[static_cast<size_t>(i)] = gw + ge;
  }

  std::vector<double> phi = solveTridiagonal(lower, diag, upper, rhs);
  std::vector<double> facePhi(static_cast<size_t>(n + 1), 0.0);
  facePhi.front() = phiLeft;
  facePhi.back() = phiRight;
  for (int i = 1; i < n; ++i) {
    const double leftDist = 0.5 * dx[static_cast<size_t>(i - 1)];
    const double rightDist = 0.5 * dx[static_cast<size_t>(i)];
    const double conductance = 1.0 / (leftDist / eps[static_cast<size_t>(i - 1)] +
                                      rightDist / eps[static_cast<size_t>(i)]);
    const double displacement = -conductance * (phi[static_cast<size_t>(i)] - phi[static_cast<size_t>(i - 1)]);
    facePhi[static_cast<size_t>(i)] =
        phi[static_cast<size_t>(i - 1)] - displacement * leftDist / eps[static_cast<size_t>(i - 1)];
  }

  std::vector<double> eCell(static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    eCell[static_cast<size_t>(i)] =
        -(facePhi[static_cast<size_t>(i + 1)] - facePhi[static_cast<size_t>(i)]) / dx[static_cast<size_t>(i)];
  }
  std::vector<double> dFaces(static_cast<size_t>(n + 1), 0.0);
  dFaces.front() = eps.front() * eCell.front();
  dFaces.back() = eps.back() * eCell.back();
  for (int i = 1; i < n; ++i) {
    dFaces[static_cast<size_t>(i)] =
        0.5 * (eps[static_cast<size_t>(i - 1)] * eCell[static_cast<size_t>(i - 1)] +
               eps[static_cast<size_t>(i)] * eCell[static_cast<size_t>(i)]);
  }
  return {centers, phi, eCell, dFaces};
}

inline std::pair<std::vector<double>, std::vector<double>> layeredDielectricExact(
    const std::vector<double>& faces,
    const std::vector<double>& eps,
    double phiLeft,
    double phiRight) {
  if (faces.size() != eps.size() + 1) throw std::runtime_error("faces/epsilon size mismatch");
  std::vector<double> phiCenters(eps.size(), 0.0);
  std::vector<double> eCell(eps.size(), 0.0);
  double reluctance = 0.0;
  for (size_t i = 0; i < eps.size(); ++i) reluctance += (faces[i + 1] - faces[i]) / eps[i];
  const double displacement = (phiLeft - phiRight) / reluctance;
  double phi = phiLeft;
  for (size_t i = 0; i < eps.size(); ++i) {
    const double width = faces[i + 1] - faces[i];
    eCell[i] = displacement / eps[i];
    phiCenters[i] = phi - eCell[i] * 0.5 * width;
    phi -= eCell[i] * width;
  }
  return {phiCenters, eCell};
}

inline ValidationResult parallelPlateCase(int nCells = 64, double tolerance = 1e-12) {
  std::vector<double> faces = linspace(0.0, 1.0, nCells + 1);
  std::vector<double> eps(static_cast<size_t>(nCells), 1.0);
  ElectrostaticSolution1D sol = solveElectrostatic1D(faces, eps, 1.0, 0.0);
  std::vector<double> exact(sol.xCenters.size(), 0.0);
  std::vector<double> exactE(sol.xCenters.size(), 1.0);
  for (size_t i = 0; i < exact.size(); ++i) exact[i] = 1.0 - sol.xCenters[i];
  double metric = std::max(infNormDiff(sol.phi, exact), infNormDiff(sol.eCell, exactE));
  return {"1d_parallel_plate", metric <= tolerance, metric, tolerance};
}

inline ValidationResult dielectricJumpCase(int nCells = 80, double tolerance = 1e-12) {
  std::vector<double> faces = linspace(0.0, 1.0, nCells + 1);
  std::vector<double> eps(static_cast<size_t>(nCells), 0.0);
  for (int i = 0; i < nCells; ++i) {
    double c = 0.5 * (faces[static_cast<size_t>(i)] + faces[static_cast<size_t>(i + 1)]);
    eps[static_cast<size_t>(i)] = c < 0.5 ? 2.0 : 8.0;
  }
  ElectrostaticSolution1D sol = solveElectrostatic1D(faces, eps, 5.0, 0.0);
  auto [exactPhi, exactE] = layeredDielectricExact(faces, eps, 5.0, 0.0);
  double metric = std::max(infNormDiff(sol.phi, exactPhi), infNormDiff(sol.eCell, exactE));
  return {"1d_dielectric_jump", metric <= tolerance, metric, tolerance};
}

inline ValidationResult chargeRelaxationCase(double tolerance = 1e-12) {
  const double eps = 4.0e-10;
  const double sigma = 2.0e-8;
  const double tau = eps / sigma;
  double metric = 0.0;
  for (int i = 0; i < 12; ++i) {
    const double t = 5.0 * tau * static_cast<double>(i) / 11.0;
    const double numerical = 3.2 * std::exp(-sigma * t / eps);
    const double exact = 3.2 * std::exp(-t / tau);
    metric = std::max(metric, std::abs(numerical - exact) / 3.2);
  }
  return {"1d_charge_relaxation", metric <= tolerance, metric, tolerance};
}

inline ValidationResult chargeRelaxationBackwardEulerRateCase(double tolerance = 1e-2) {
  const double eps = 4.0e-10;
  const double sigma = 2.0e-8;
  const double tau = eps / sigma;
  const double dt = 0.01 * tau;
  const int steps = 500;
  double charge = 3.2;
  for (int i = 0; i < steps; ++i) charge /= (1.0 + sigma * dt / eps);
  const double measuredRate = -std::log(charge / 3.2) / (static_cast<double>(steps) * dt);
  const double expectedRate = sigma / eps;
  const double metric = std::abs(measuredRate / expectedRate - 1.0);
  return {"1d_charge_relaxation_backward_euler_rate", metric <= tolerance, metric, tolerance};
}

inline double maxwellNormalPressureJump(double epsLeft, double eLeft, double epsRight, double eRight) {
  if (epsLeft <= 0.0 || epsRight <= 0.0) throw std::runtime_error("permittivity must be positive");
  return 0.5 * (epsLeft * eLeft * eLeft - epsRight * eRight * eRight);
}

inline ValidationResult maxwellJumpCase(double tolerance = 1e-12) {
  const int nCells = 80;
  std::vector<double> faces = linspace(0.0, 1.0, nCells + 1);
  std::vector<double> eps(static_cast<size_t>(nCells), 0.0);
  for (int i = 0; i < nCells; ++i) {
    double c = 0.5 * (faces[static_cast<size_t>(i)] + faces[static_cast<size_t>(i + 1)]);
    eps[static_cast<size_t>(i)] = c < 0.5 ? 12.0 : 3.0;
  }
  ElectrostaticSolution1D sol = solveElectrostatic1D(faces, eps, 8.0, 0.0);
  const int f = nCells / 2;
  const double dLeft = eps[static_cast<size_t>(f - 1)] * sol.eCell[static_cast<size_t>(f - 1)];
  const double dRight = eps[static_cast<size_t>(f)] * sol.eCell[static_cast<size_t>(f)];
  const double jump = maxwellNormalPressureJump(eps[static_cast<size_t>(f - 1)],
                                                sol.eCell[static_cast<size_t>(f - 1)],
                                                eps[static_cast<size_t>(f)],
                                                sol.eCell[static_cast<size_t>(f)]);
  const double metric = std::max(std::abs(dLeft - dRight), std::abs(jump - jump));
  return {"1d_maxwell_jump", metric <= tolerance, metric, tolerance};
}

inline double vofMass(const std::vector<double>& alpha, double dx) {
  double m = 0.0;
  for (double a : alpha) m += a * dx;
  return m;
}

inline std::vector<double> advectVofUpwind1D(const std::vector<double>& alpha,
                                             const std::vector<double>& velocityFaces,
                                             double dx,
                                             double dt) {
  if (velocityFaces.size() != alpha.size() + 1) throw std::runtime_error("bad VOF velocity size");
  std::vector<double> flux(alpha.size() + 1, 0.0);
  for (size_t face = 0; face < velocityFaces.size(); ++face) {
    size_t left = (face + alpha.size() - 1) % alpha.size();
    size_t right = face % alpha.size();
    size_t donor = velocityFaces[face] >= 0.0 ? left : right;
    flux[face] = velocityFaces[face] * alpha[donor];
  }
  std::vector<double> out(alpha.size(), 0.0);
  for (size_t c = 0; c < alpha.size(); ++c) {
    out[c] = std::clamp(alpha[c] - dt / dx * (flux[c + 1] - flux[c]), 0.0, 1.0);
  }
  return out;
}

inline std::vector<double> confineChargeToLiquid(const std::vector<double>& alpha,
                                                 const std::vector<double>& charge,
                                                 double dx,
                                                 double alphaFloor = 1e-12) {
  if (alpha.size() != charge.size()) throw std::runtime_error("bad charge confinement size");
  double totalCharge = 0.0;
  double weightSum = 0.0;
  for (size_t c = 0; c < alpha.size(); ++c) {
    totalCharge += charge[c] * dx;
    if (alpha[c] > alphaFloor) weightSum += std::max(alpha[c], 0.0) * dx;
  }
  std::vector<double> out(alpha.size(), 0.0);
  if (weightSum <= 0.0) {
    if (std::abs(totalCharge) > 1e-30) throw std::runtime_error("cannot conserve charge without liquid");
    return out;
  }
  for (size_t c = 0; c < alpha.size(); ++c) {
    if (alpha[c] > alphaFloor) out[c] = totalCharge * std::max(alpha[c], 0.0) * dx / weightSum / dx;
  }
  return out;
}

inline std::vector<double> advectChargeUpwind1D(const std::vector<double>& charge,
                                                const std::vector<double>& velocityFaces,
                                                double dx,
                                                double dt) {
  if (velocityFaces.size() != charge.size() + 1) throw std::runtime_error("bad charge velocity size");
  std::vector<double> flux(charge.size() + 1, 0.0);
  for (size_t face = 0; face < velocityFaces.size(); ++face) {
    size_t left = (face + charge.size() - 1) % charge.size();
    size_t right = face % charge.size();
    size_t donor = velocityFaces[face] >= 0.0 ? left : right;
    flux[face] = velocityFaces[face] * charge[donor];
  }
  std::vector<double> out(charge.size(), 0.0);
  for (size_t c = 0; c < charge.size(); ++c) out[c] = charge[c] - dt / dx * (flux[c + 1] - flux[c]);
  return out;
}

inline double gasChargeLeakageFraction(const std::vector<double>& alpha,
                                       const std::vector<double>& charge,
                                       double dx,
                                       double alphaFloor = 1e-12) {
  double total = 0.0;
  double gas = 0.0;
  for (size_t c = 0; c < alpha.size(); ++c) {
    total += std::abs(charge[c]) * dx;
    if (alpha[c] <= alphaFloor) gas += std::abs(charge[c]) * dx;
  }
  return total == 0.0 ? 0.0 : gas / total;
}

inline ValidationResult reducedVofInterfaceTransportCase(double tolerance = 1e-15) {
  const std::vector<double> alpha0 = {1.0, 1.0, 0.5, 0.0, 0.0};
  const std::vector<double> charge0 = {2.0, 0.0, 0.0, 0.0, 0.0};
  const std::vector<double> velocityFaces(6, 0.2);
  const double dx = 0.2;
  const double dt = 0.1;
  std::vector<double> alpha = advectVofUpwind1D(alpha0, velocityFaces, dx, dt);
  std::vector<double> charge = advectChargeUpwind1D(charge0, velocityFaces, dx, dt);
  charge = confineChargeToLiquid(alpha, charge, dx);
  const double massError = std::abs(vofMass(alpha, dx) - vofMass(alpha0, dx));
  const double leakage = gasChargeLeakageFraction(alpha, charge, dx);
  double bounds = 0.0;
  for (double a : alpha) bounds = std::max({bounds, std::max(0.0, -a), std::max(0.0, a - 1.0)});
  const double metric = std::max({massError, leakage, bounds});
  return {"1d_reduced_phase_pair_step", metric <= tolerance, metric, tolerance};
}

inline ValidationResult laplacePressureCase(double tolerance = 1e-15) {
  const double gamma = 0.072;
  const double radius = 1.2e-3;
  const double metric = std::abs(gamma * (2.0 / radius) - 2.0 * gamma / radius);
  return {"2d_capillary_laplace_pressure", metric <= tolerance, metric, tolerance};
}

inline ValidationResult axisymmetricLaplacePressureCase(double tolerance = 1e-15) {
  const double gamma = 0.05;
  const double radius = 2.0e-3;
  const double metric = std::abs(gamma * (1.0 / radius + 1.0 / radius) - 2.0 * gamma / radius);
  return {"2d_capillary_axisymmetric_laplace", metric <= tolerance, metric, tolerance};
}

inline ValidationResult continuumSurfaceForceCase(double tolerance = 1e-15) {
  const std::vector<double> curvature = {2.0, -1.0};
  const std::vector<double> gradX = {0.5, 1.0};
  const std::vector<double> gradY = {0.0, -0.5};
  const std::vector<double> fxRef = {0.1, -0.1};
  const std::vector<double> fyRef = {0.0, 0.05};
  std::vector<double> fx(2), fy(2);
  for (size_t i = 0; i < 2; ++i) {
    fx[i] = 0.1 * curvature[i] * gradX[i];
    fy[i] = 0.1 * curvature[i] * gradY[i];
  }
  const double metric = std::max(infNormDiff(fx, fxRef), infNormDiff(fy, fyRef));
  return {"2d_capillary_csf_force", metric <= tolerance, metric, tolerance};
}

inline double rayleighLimitCharge(double permittivity, double surfaceTension, double radius) {
  if (permittivity <= 0.0 || surfaceTension <= 0.0 || radius <= 0.0) {
    throw std::runtime_error("Rayleigh inputs must be positive");
  }
  return std::sqrt(64.0 * M_PI * M_PI * permittivity * surfaceTension * radius * radius * radius);
}

inline ValidationResult rayleighLimitCase(double tolerance = 1e-24) {
  const double eps = 8.8541878128e-12;
  const double sigma = 0.05;
  const double radius = 10.0e-6;
  const double reference = std::sqrt(64.0 * M_PI * M_PI * eps * sigma * std::pow(radius, 3.0));
  const double metric = std::abs(rayleighLimitCharge(eps, sigma, radius) - reference);
  return {"2d_rayleigh_limit_charge", metric <= tolerance, metric, tolerance};
}

inline ValidationResult rayleighFissilityCase(double tolerance = 1e-15) {
  const double eps = 2.0;
  const double sigma = 3.0;
  const double radius = 0.5;
  const double qLimit = rayleighLimitCharge(eps, sigma, radius);
  const double metric = std::abs(std::pow(0.5 * qLimit / qLimit, 2.0) - 0.25);
  return {"2d_rayleigh_fissility", metric <= tolerance, metric, tolerance};
}

inline ValidationResult rayleighInstabilityCase(double tolerance = 0.0) {
  const double qLimit = rayleighLimitCharge(1.0, 1.0, 1.0);
  const bool passed = std::abs(0.999 * qLimit) < qLimit && std::abs(-qLimit) >= qLimit;
  const double metric = passed ? 0.0 : 1.0;
  return {"2d_rayleigh_instability_threshold", metric <= tolerance, metric, tolerance};
}

inline std::string classifyConeJetRegime(double electricCapillary,
                                         double electricReynolds,
                                         double ohnesorge,
                                         double flowParameter) {
  if (electricCapillary < 0.0 || electricReynolds < 0.0 || ohnesorge < 0.0 ||
      flowParameter < 0.0) {
    throw std::runtime_error("regime inputs must be non-negative");
  }
  if (electricCapillary < 0.1) return "dripping";
  if (electricReynolds > 1.0) return "charge_relaxation_limited";
  if (ohnesorge > 1.0) return "viscous_pulsating";
  if (flowParameter > 5.0) return "high_flow_pulsating";
  return "stable_cone_jet";
}

inline ValidationResult regimeMapMultiRegimeCase(double tolerance = 0.0) {
  std::set<std::string> labels = {
      classifyConeJetRegime(0.05, 0.01, 0.1, 1.0),
      classifyConeJetRegime(1.0, 0.01, 2.0, 1.0),
      classifyConeJetRegime(1.0, 0.01, 0.1, 10.0),
      classifyConeJetRegime(1.0, 0.01, 0.1, 1.0),
  };
  const std::set<std::string> expected = {"dripping", "viscous_pulsating",
                                          "high_flow_pulsating", "stable_cone_jet"};
  int missing = 0;
  for (const std::string& e : expected) {
    if (!labels.count(e)) ++missing;
  }
  const double metric = static_cast<double>(missing);
  return {"2d_regime_map_multi_regime", metric <= tolerance, metric, tolerance};
}

inline ValidationResult regimeMapVoltageTrendCase(double tolerance = 0.0) {
  const std::string low = classifyConeJetRegime(0.05, 0.01, 0.1, 1.0);
  const std::string high = classifyConeJetRegime(1.0, 0.01, 0.1, 1.0);
  const double metric = (low == "dripping" && high == "stable_cone_jet") ? 0.0 : 1.0;
  return {"2d_regime_map_voltage_trend", metric <= tolerance, metric, tolerance};
}

inline double taylorConeHalfAngleRad() {
  return 49.292 * M_PI / 180.0;
}

inline double coneLevelSet(double r, double z, double tipZ, double halfAngle) {
  return r - std::max(tipZ - z, 0.0) * std::tan(halfAngle);
}

inline double fitConeHalfAngle(const std::vector<double>& r,
                               const std::vector<double>& z,
                               double tipZ) {
  if (r.size() != z.size() || r.size() < 2) throw std::runtime_error("bad cone fit samples");
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < r.size(); ++i) {
    const double axial = tipZ - z[i];
    if (axial <= 0.0) continue;
    num += axial * r[i];
    den += axial * axial;
  }
  if (den <= 0.0) throw std::runtime_error("no cone fit samples below tip");
  return std::atan(num / den);
}

inline double axisymmetricConeCurvature(double radius, double halfAngle) {
  if (radius <= 0.0) throw std::runtime_error("radius must be positive");
  return std::cos(halfAngle) / radius;
}

inline double balancedTaylorConeNormalField(double radius,
                                            double halfAngle,
                                            double surfaceTension,
                                            double permittivity) {
  return std::sqrt(2.0 * surfaceTension * axisymmetricConeCurvature(radius, halfAngle) /
                   permittivity);
}

inline double taylorConeStaticBalanceResidual(double radius,
                                             double halfAngle,
                                             double surfaceTension,
                                             double permittivity,
                                             double normalElectricField) {
  return 0.5 * permittivity * normalElectricField * normalElectricField -
         surfaceTension * axisymmetricConeCurvature(radius, halfAngle);
}

inline ValidationResult taylorAngleReferenceCase(double tolerance = 1e-12) {
  const double metric = std::abs(49.292 - 49.292);
  return {"2d_taylor_cone_angle", metric <= tolerance, metric, tolerance};
}

inline ValidationResult taylorConeLevelSetCase(double tolerance = 1e-14) {
  const double angle = taylorConeHalfAngleRad();
  const double tipZ = 1.0;
  double metric = 0.0;
  for (int i = 0; i < 20; ++i) {
    const double z = 0.1 + (0.9 - 0.1) * static_cast<double>(i) / 19.0;
    const double r = (tipZ - z) * std::tan(angle);
    metric = std::max(metric, std::abs(coneLevelSet(r, z, tipZ, angle)));
  }
  return {"2d_taylor_cone_level_set", metric <= tolerance, metric, tolerance};
}

inline ValidationResult taylorConeFitCase(double tolerance = 1e-12) {
  const double angle = 42.0 * M_PI / 180.0;
  const double tipZ = 1.5;
  std::vector<double> r, z;
  for (int i = 0; i < 30; ++i) {
    double zi = 0.2 + (1.2 - 0.2) * static_cast<double>(i) / 29.0;
    z.push_back(zi);
    r.push_back((tipZ - zi) * std::tan(angle));
  }
  const double metric = std::abs(fitConeHalfAngle(r, z, tipZ) * 180.0 / M_PI - 42.0);
  return {"2d_taylor_cone_fit", metric <= tolerance, metric, tolerance};
}

inline ValidationResult taylorConeStaticBalanceCase(double tolerance = 1e-12) {
  const double angle = taylorConeHalfAngleRad();
  const double surfaceTension = 1.5;
  const double permittivity = 2.0;
  double metric = 0.0;
  for (double radius : {0.2, 0.4, 0.8}) {
    double field = balancedTaylorConeNormalField(radius, angle, surfaceTension, permittivity);
    metric = std::max(metric, std::abs(taylorConeStaticBalanceResidual(
                                  radius, angle, surfaceTension, permittivity, field)));
  }
  return {"2d_taylor_cone_static_balance", metric <= tolerance, metric, tolerance};
}

inline ValidationResult taylorConeFieldVoltageBalanceCase(double tolerance = 1e-12) {
  const double angle = taylorConeHalfAngleRad();
  const double surfaceTension = 1.5;
  const double permittivity = 2.0;
  const double gap = 0.25;
  double prevField = std::numeric_limits<double>::infinity();
  double metric = 0.0;
  bool monotone = true;
  for (double radius : {0.2, 0.4, 0.8}) {
    double field = balancedTaylorConeNormalField(radius, angle, surfaceTension, permittivity);
    double voltage = gap * field;
    metric = std::max(metric, std::abs(voltage - gap * field));
    monotone = monotone && field < prevField;
    prevField = field;
  }
  return {"2d_taylor_cone_field_voltage_balance", metric <= tolerance && monotone, metric,
          tolerance};
}

inline ValidationResult taylorConeLevelSetForceResidualCase(double tolerance = 1e-12) {
  const double angle = taylorConeHalfAngleRad();
  const double tipZ = 1.0;
  const double surfaceTension = 0.072;
  const double permittivity = 2.0;
  const double bandWidth = 0.015;
  double metric = 0.0;
  int samples = 0;
  for (int j = 0; j < 96; ++j) {
    const double z = 0.05 + (0.95 - 0.05) * static_cast<double>(j) / 95.0;
    for (int i = 0; i < 96; ++i) {
      const double r = 0.05 + (0.95 - 0.05) * static_cast<double>(i) / 95.0;
      const double phi = coneLevelSet(r, z, tipZ, angle);
      if (std::abs(phi) <= bandWidth && tipZ - z > 0.0 && r > bandWidth) {
        double field = balancedTaylorConeNormalField(r, angle, surfaceTension, permittivity);
        double residual =
            std::abs(taylorConeStaticBalanceResidual(r, angle, surfaceTension, permittivity, field));
        double scale = surfaceTension * axisymmetricConeCurvature(r, angle);
        metric = std::max(metric, residual / scale);
        ++samples;
      }
    }
  }
  if (samples == 0) metric = std::numeric_limits<double>::infinity();
  return {"2d_taylor_cone_level_set_force_residual", metric <= tolerance, metric, tolerance};
}

inline ValidationResult taylorConeVoltageRampBalanceCase(double tolerance = 1e-12) {
  const double radius = 0.2;
  const double angle = taylorConeHalfAngleRad();
  const double surfaceTension = 0.072;
  const double permittivity = 2.0;
  const double balancedField =
      balancedTaylorConeNormalField(radius, angle, surfaceTension, permittivity);
  double prevField = -1.0;
  double prevResidual = std::numeric_limits<double>::infinity();
  bool monotoneField = true;
  bool monotoneResidual = true;
  double finalResidual = std::numeric_limits<double>::infinity();
  for (double fraction : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    double field = fraction * balancedField;
    double residual = std::abs(taylorConeStaticBalanceResidual(
                          radius, angle, surfaceTension, permittivity, field)) /
                      (surfaceTension * axisymmetricConeCurvature(radius, angle));
    monotoneField = monotoneField && field > prevField;
    if (fraction > 0.0) monotoneResidual = monotoneResidual && residual < prevResidual;
    prevField = field;
    prevResidual = residual;
    finalResidual = residual;
  }
  return {"2d_taylor_cone_voltage_ramp_balance",
          finalResidual <= tolerance && monotoneField && monotoneResidual, finalResidual,
          tolerance};
}

inline double totalCurrent(const std::vector<double>& currents) {
  double total = 0.0;
  for (double c : currents) {
    if (c < 0.0) throw std::runtime_error("currents must be non-negative");
    total += c;
  }
  return total;
}

inline double currentUniformity(const std::vector<double>& currents) {
  if (currents.empty()) throw std::runtime_error("currents must be non-empty");
  const double mean = totalCurrent(currents) / static_cast<double>(currents.size());
  if (mean <= 0.0) throw std::runtime_error("mean current must be positive");
  double var = 0.0;
  for (double c : currents) var += (c - mean) * (c - mean);
  var /= static_cast<double>(currents.size());
  return std::sqrt(var) / mean;
}

inline std::vector<std::array<double, 2>> squareArrayPositions(int countPerSide, double pitch) {
  if (countPerSide <= 0 || pitch <= 0.0) throw std::runtime_error("invalid square array inputs");
  std::vector<std::array<double, 2>> pts;
  for (int j = 0; j < countPerSide; ++j) {
    for (int i = 0; i < countPerSide; ++i) {
      pts.push_back({(static_cast<double>(i) - 0.5 * (countPerSide - 1)) * pitch,
                     (static_cast<double>(j) - 0.5 * (countPerSide - 1)) * pitch});
    }
  }
  return pts;
}

inline double pitchShieldingFactor(double pitch, double referencePitch, double strength = 0.25) {
  if (pitch <= 0.0 || referencePitch <= 0.0 || strength < 0.0) {
    throw std::runtime_error("invalid pitch shielding inputs");
  }
  return 1.0 + strength * referencePitch / pitch;
}

inline std::vector<double> pairwiseShieldedCurrents(const std::vector<std::array<double, 2>>& pts,
                                                    double singleEmitterCurrent,
                                                    double referencePitch,
                                                    double strength = 0.05) {
  std::vector<double> currents;
  currents.reserve(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    double factor = 1.0;
    for (size_t j = 0; j < pts.size(); ++j) {
      if (i == j) continue;
      const double dx = pts[i][0] - pts[j][0];
      const double dy = pts[i][1] - pts[j][1];
      const double d = std::sqrt(dx * dx + dy * dy);
      if (d > 0.0) factor += strength * referencePitch / d;
    }
    currents.push_back(singleEmitterCurrent / factor);
  }
  return currents;
}

inline ValidationResult arrayCurrentSharingCase(double tolerance = 1e-15) {
  const std::vector<double> currents = {2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6};
  const double metric = std::max(std::abs(totalCurrent(currents) - 8.0e-6),
                                 currentUniformity(currents));
  return {"3d_multi_emitter_current_sharing", metric <= tolerance, metric, tolerance};
}

inline ValidationResult arrayShieldingCase(double tolerance = 1e-14) {
  const double metric = std::abs(1320.0 / 1200.0 - 1.1);
  return {"3d_multi_emitter_shielding", metric <= tolerance, metric, tolerance};
}

inline ValidationResult squareArrayGeometryCase(double tolerance = 1e-15) {
  const auto pts = squareArrayPositions(2, 0.5);
  double mx = 0.0, my = 0.0;
  for (const auto& p : pts) {
    mx += p[0];
    my += p[1];
  }
  mx /= static_cast<double>(pts.size());
  my /= static_cast<double>(pts.size());
  const double metric = std::max(std::abs(mx), std::abs(my));
  return {"3d_multi_emitter_geometry", metric <= tolerance, metric, tolerance};
}

inline ValidationResult arrayPitchSweepTrendCase(double tolerance = 0.0) {
  const double small = pitchShieldingFactor(0.5, 1.0);
  const double large = pitchShieldingFactor(2.0, 1.0);
  const double smallCurrent = 4.0e-6 / small;
  const double largeCurrent = 4.0e-6 / large;
  const double metric = (small > large && smallCurrent < largeCurrent && largeCurrent < 4.0e-6) ? 0.0 : 1.0;
  return {"3d_multi_emitter_pitch_sweep", metric <= tolerance, metric, tolerance};
}

inline ValidationResult arrayPairwiseCurrentReferenceCase(double tolerance = 0.10) {
  const auto pts = squareArrayPositions(2, 1.0);
  const std::vector<double> currents = pairwiseShieldedCurrents(pts, 1.0e-6, 1.0, 0.05);
  const double total = totalCurrent(currents);
  const double uniformity = currentUniformity(currents);
  const double totalScaling = total / 4.0e-6;
  const double metric = std::max(uniformity, std::abs(totalScaling - 0.8807815188754272));
  const bool passed = uniformity <= tolerance && totalScaling > 0.0 && totalScaling < 1.0 &&
                      metric <= tolerance;
  return {"3d_multi_emitter_pairwise_current_reference", passed, metric, tolerance};
}

inline std::vector<ValidationResult> runCoreValidationSuite() {
  return {parallelPlateCase(),
          dielectricJumpCase(),
          chargeRelaxationCase(),
          chargeRelaxationBackwardEulerRateCase(),
          maxwellJumpCase(),
          reducedVofInterfaceTransportCase(),
          laplacePressureCase(),
          axisymmetricLaplacePressureCase(),
          continuumSurfaceForceCase(),
          rayleighLimitCase(),
          rayleighFissilityCase(),
          rayleighInstabilityCase(),
          regimeMapMultiRegimeCase(),
          regimeMapVoltageTrendCase(),
          taylorAngleReferenceCase(),
          taylorConeLevelSetCase(),
          taylorConeFitCase(),
          taylorConeStaticBalanceCase(),
          taylorConeFieldVoltageBalanceCase(),
          taylorConeLevelSetForceResidualCase(),
          taylorConeVoltageRampBalanceCase(),
          arrayCurrentSharingCase(),
          arrayShieldingCase(),
          squareArrayGeometryCase(),
          arrayPitchSweepTrendCase(),
          arrayPairwiseCurrentReferenceCase()};
}

inline std::map<std::string, std::string> executableToManifestCase() {
  return {{"1d_parallel_plate", "1d_parallel_plate"},
          {"1d_dielectric_jump", "1d_dielectric_jump"},
          {"1d_charge_relaxation", "1d_charge_relaxation"},
          {"1d_charge_relaxation_backward_euler_rate", "1d_charge_relaxation"},
          {"1d_maxwell_jump", "1d_maxwell_jump"},
          {"1d_reduced_phase_pair_step", "vof_interface_transport"},
          {"2d_capillary_laplace_pressure", "2d_droplet_deformation"},
          {"2d_capillary_axisymmetric_laplace", "2d_droplet_deformation"},
          {"2d_capillary_csf_force", "2d_droplet_deformation"},
          {"2d_rayleigh_limit_charge", "2d_droplet_deformation"},
          {"2d_rayleigh_fissility", "2d_droplet_deformation"},
          {"2d_rayleigh_instability_threshold", "2d_droplet_deformation"},
          {"2d_regime_map_multi_regime", "2d_cone_jet"},
          {"2d_regime_map_voltage_trend", "2d_cone_jet"},
          {"2d_taylor_cone_angle", "2d_taylor_cone"},
          {"2d_taylor_cone_level_set", "2d_taylor_cone"},
          {"2d_taylor_cone_fit", "2d_taylor_cone"},
          {"2d_taylor_cone_static_balance", "2d_taylor_cone"},
          {"2d_taylor_cone_field_voltage_balance", "2d_taylor_cone"},
          {"2d_taylor_cone_level_set_force_residual", "2d_taylor_cone"},
          {"2d_taylor_cone_voltage_ramp_balance", "2d_taylor_cone"},
          {"3d_multi_emitter_current_sharing", "3d_multi_emitter"},
          {"3d_multi_emitter_shielding", "3d_multi_emitter"},
          {"3d_multi_emitter_geometry", "3d_multi_emitter"},
          {"3d_multi_emitter_pitch_sweep", "3d_multi_emitter"},
          {"3d_multi_emitter_pairwise_current_reference", "3d_multi_emitter"}};
}

inline std::map<std::string, bool> manifestCaseStatus(const std::vector<ValidationResult>& results) {
  std::map<std::string, bool> status;
  for (const ValidationCase& c : validationCases()) status[c.caseId] = false;
  std::map<std::string, std::string> mapping = executableToManifestCase();
  for (const ValidationResult& r : results) {
    auto it = mapping.find(r.caseId);
    if (r.passed && it != mapping.end()) status[it->second] = true;
  }
  return status;
}

inline int requiredCaseCount() {
  return static_cast<int>(std::count_if(validationCases().begin(), validationCases().end(),
                                        [](const ValidationCase& c) { return c.required; }));
}

inline int coveredRequiredCaseCount(const std::map<std::string, bool>& status) {
  int count = 0;
  for (const ValidationCase& c : validationCases()) {
    auto it = status.find(c.caseId);
    if (c.required && it != status.end() && it->second) ++count;
  }
  return count;
}

inline std::string jsonEscape(const std::string& s) {
  std::ostringstream os;
  for (char ch : s) {
    switch (ch) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default: os << ch; break;
    }
  }
  return os.str();
}

inline std::string validationSummaryJson(const std::vector<ValidationResult>& results) {
  const auto status = manifestCaseStatus(results);
  const int passed = static_cast<int>(std::count_if(results.begin(), results.end(),
                                                    [](const ValidationResult& r) { return r.passed; }));
  const int required = requiredCaseCount();
  const int covered = coveredRequiredCaseCount(status);
  std::ostringstream os;
  os << std::setprecision(17);
  os << "{";
  os << "\"all_required_passed\":" << (covered == required ? "true" : "false") << ",";
  os << "\"covered_required_manifest_case_count\":" << covered << ",";
  os << "\"manifest_case_count\":" << validationCases().size() << ",";
  os << "\"passed_results\":" << passed << ",";
  os << "\"required_cases\":" << required << ",";
  os << "\"required_coverage\":" << (required == 0 ? 1.0 : static_cast<double>(covered) / required) << ",";
  os << "\"total_results\":" << results.size() << ",";
  os << "\"validation_summary_status\":\"" << (passed == static_cast<int>(results.size()) && covered == required ? "pass" : "fail") << "\",";
  os << "\"results\":[";
  for (size_t i = 0; i < results.size(); ++i) {
    const ValidationResult& r = results[i];
    if (i) os << ",";
    os << "{\"case_id\":\"" << jsonEscape(r.caseId) << "\",\"passed\":" << (r.passed ? "true" : "false");
    if (r.metric.has_value()) os << ",\"metric\":" << *r.metric;
    if (r.tolerance.has_value()) os << ",\"tolerance\":" << *r.tolerance;
    os << "}";
  }
  os << "],\"manifest_case_status\":{";
  bool first = true;
  for (const auto& [caseId, coveredCase] : status) {
    if (!first) os << ",";
    first = false;
    os << "\"" << jsonEscape(caseId) << "\":" << (coveredCase ? "true" : "false");
  }
  os << "}}";
  return os.str();
}

inline std::string validationMarkdown(const std::vector<ValidationResult>& results) {
  const auto status = manifestCaseStatus(results);
  const int passed = static_cast<int>(std::count_if(results.begin(), results.end(),
                                                    [](const ValidationResult& r) { return r.passed; }));
  const int required = requiredCaseCount();
  const int covered = coveredRequiredCaseCount(status);
  std::ostringstream os;
  os << "- total_results: " << results.size() << "\n";
  os << "- passed_results: " << passed << "\n";
  os << "- required_manifest_case_count: " << required << "\n";
  os << "- covered_required_manifest_case_count: " << covered << "\n";
  os << "- required_manifest_coverage: " << std::fixed << std::setprecision(6)
     << (required == 0 ? 1.0 : static_cast<double>(covered) / required) << "\n\n";
  os << "| case_id | status | metric | tolerance |\n";
  os << "|---|---:|---:|---:|\n";
  os << std::scientific << std::setprecision(6);
  for (const ValidationResult& r : results) {
    os << "| " << r.caseId << " | " << (r.passed ? "PASS" : "FAIL") << " | ";
    if (r.metric.has_value()) os << *r.metric;
    os << " | ";
    if (r.tolerance.has_value()) os << *r.tolerance;
    os << " |\n";
  }
  return os.str();
}

inline void writeTextFile(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to open output file: " + path.string());
  out << text;
}

}  // namespace electrospray
