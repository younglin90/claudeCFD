#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace electrospray {

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

inline double laplacePressureJump(double surfaceTension, double curvature) {
  if (surfaceTension < 0.0) throw std::runtime_error("surface_tension must be non-negative");
  return surfaceTension * curvature;
}

inline double normalTraction2D(double epsilon, Vec2 electricField, Vec2 normal) {
  if (!(epsilon > 0.0)) throw std::runtime_error("epsilon must be positive");
  const double norm = std::hypot(normal.x, normal.y);
  if (!(norm > 0.0)) throw std::runtime_error("normal vector must be nonzero");
  const double nx = normal.x / norm;
  const double ny = normal.y / norm;
  const double e2 = electricField.x * electricField.x + electricField.y * electricField.y;
  const double txx = epsilon * (electricField.x * electricField.x - 0.5 * e2);
  const double txy = epsilon * electricField.x * electricField.y;
  const double tyy = epsilon * (electricField.y * electricField.y - 0.5 * e2);
  return nx * (txx * nx + txy * ny) + ny * (txy * nx + tyy * ny);
}

inline double normalTractionJump2D(double epsilonLeft,
                                   Vec2 eLeft,
                                   double epsilonRight,
                                   Vec2 eRight,
                                   Vec2 normal) {
  return normalTraction2D(epsilonLeft, eLeft, normal) - normalTraction2D(epsilonRight, eRight, normal);
}

inline Vec2 ohmicCurrentDensity(double conductivity, Vec2 electricField) {
  if (conductivity < 0.0) throw std::runtime_error("conductivity must be non-negative");
  return {conductivity * electricField.x, conductivity * electricField.y};
}

inline double electrostaticEnergyDensity(double permittivity, Vec2 electricField) {
  if (!(permittivity > 0.0)) throw std::runtime_error("permittivity must be positive");
  return 0.5 * permittivity * (electricField.x * electricField.x + electricField.y * electricField.y);
}

inline double totalElectrostaticEnergy(const std::vector<double>& permittivity,
                                       const std::vector<Vec2>& electricField,
                                       const std::vector<double>& cellVolume) {
  if (permittivity.size() != electricField.size() || cellVolume.size() != electricField.size()) {
    throw std::runtime_error("permittivity, electric field, and volume arrays must match");
  }
  double total = 0.0;
  for (size_t i = 0; i < electricField.size(); ++i) {
    if (!(cellVolume[i] > 0.0)) throw std::runtime_error("cell volumes must be positive");
    total += electrostaticEnergyDensity(permittivity[i], electricField[i]) * cellVolume[i];
  }
  return total;
}

inline double chargeConservationResidual(double chargeOld, double chargeNew, double currentDivergence, double dt) {
  if (!(dt > 0.0)) throw std::runtime_error("dt must be positive");
  return (chargeNew - chargeOld) / dt + currentDivergence;
}

inline Vec2 unitNormal(Vec2 normal) {
  const double mag = std::hypot(normal.x, normal.y);
  if (!(mag > 0.0)) throw std::runtime_error("normal must be nonzero");
  return {normal.x / mag, normal.y / mag};
}

inline Vec2 unitTangent(Vec2 normal) {
  const Vec2 n = unitNormal(normal);
  return {-n.y, n.x};
}

inline double dot(Vec2 a, Vec2 b) {
  return a.x * b.x + a.y * b.y;
}

inline double normalComponent(Vec2 vector, Vec2 normal) {
  return dot(vector, unitNormal(normal));
}

inline double tangentialComponent(Vec2 vector, Vec2 normal) {
  return dot(vector, unitTangent(normal));
}

inline double surfaceChargeDensity(double epsilonLeft, Vec2 eLeft, double epsilonRight, Vec2 eRight, Vec2 normal) {
  if (!(epsilonLeft > 0.0) || !(epsilonRight > 0.0)) throw std::runtime_error("permittivities must be positive");
  return epsilonRight * normalComponent(eRight, normal) - epsilonLeft * normalComponent(eLeft, normal);
}

inline double normalOhmicCurrentJump(double sigmaLeft, Vec2 eLeft, double sigmaRight, Vec2 eRight, Vec2 normal) {
  if (sigmaLeft < 0.0 || sigmaRight < 0.0) throw std::runtime_error("conductivities must be non-negative");
  return sigmaRight * normalComponent(eRight, normal) - sigmaLeft * normalComponent(eLeft, normal);
}

inline double tangentialFieldJump(Vec2 eLeft, Vec2 eRight, Vec2 normal) {
  return tangentialComponent(eRight, normal) - tangentialComponent(eLeft, normal);
}

inline double electricShearTractionJump(double epsilonLeft, Vec2 eLeft, double epsilonRight, Vec2 eRight, Vec2 normal) {
  if (!(epsilonLeft > 0.0) || !(epsilonRight > 0.0)) throw std::runtime_error("permittivities must be positive");
  const double left = epsilonLeft * normalComponent(eLeft, normal) * tangentialComponent(eLeft, normal);
  const double right = epsilonRight * normalComponent(eRight, normal) * tangentialComponent(eRight, normal);
  return right - left;
}

inline double staticNormalStressResidual(double pressureLeft,
                                         double pressureRight,
                                         double surfaceTension,
                                         double curvature,
                                         double epsilonLeft,
                                         Vec2 eLeft,
                                         double epsilonRight,
                                         Vec2 eRight,
                                         Vec2 normal) {
  const double pressureJump = pressureLeft - pressureRight;
  const double capillaryJump = laplacePressureJump(surfaceTension, curvature);
  const double electricJump = normalTractionJump2D(epsilonLeft, eLeft, epsilonRight, eRight, normal);
  return pressureJump - capillaryJump - electricJump;
}

inline double maxAbsResidual(const std::vector<double>& residual) {
  if (residual.empty()) throw std::runtime_error("residual must be non-empty");
  double maxValue = 0.0;
  for (double value : residual) maxValue = std::max(maxValue, std::abs(value));
  return maxValue;
}

inline double l2Error(const std::vector<double>& numerical,
                      const std::vector<double>& exact,
                      const std::vector<double>& weights) {
  if (numerical.size() != exact.size() || weights.size() != numerical.size()) {
    throw std::runtime_error("numerical, exact, and weights must have matching shapes");
  }
  double weightedError = 0.0;
  double weightSum = 0.0;
  for (size_t i = 0; i < numerical.size(); ++i) {
    if (!(weights[i] > 0.0)) throw std::runtime_error("weights must be positive");
    const double diff = numerical[i] - exact[i];
    weightedError += weights[i] * diff * diff;
    weightSum += weights[i];
  }
  return std::sqrt(weightedError / weightSum);
}

inline double l2Error(const std::vector<double>& numerical, const std::vector<double>& exact, double weight = 1.0) {
  if (!(weight > 0.0)) throw std::runtime_error("weights must be positive");
  return l2Error(numerical, exact, std::vector<double>(numerical.size(), weight));
}

inline double linfError(const std::vector<double>& numerical, const std::vector<double>& exact) {
  if (numerical.size() != exact.size()) throw std::runtime_error("numerical and exact must have matching shapes");
  double value = 0.0;
  for (size_t i = 0; i < numerical.size(); ++i) value = std::max(value, std::abs(numerical[i] - exact[i]));
  return value;
}

inline double convergenceRate(double errorCoarse, double errorFine, double refinement = 2.0) {
  if (!(errorCoarse > 0.0) || !(errorFine > 0.0) || !(refinement > 1.0)) {
    throw std::runtime_error("errors must be positive and refinement must exceed one");
  }
  return std::log(errorCoarse / errorFine) / std::log(refinement);
}

inline bool passesThreshold(double value, double threshold, bool inclusive = true) {
  if (threshold < 0.0) throw std::runtime_error("threshold must be non-negative");
  return inclusive ? value <= threshold : value < threshold;
}

inline std::vector<double> sinusoidalPotential1D(const std::vector<double>& x,
                                                 double amplitude = 1.0,
                                                 double length = 1.0) {
  if (!(length > 0.0)) throw std::runtime_error("length must be positive");
  std::vector<double> phi(x.size(), 0.0);
  for (size_t i = 0; i < x.size(); ++i) phi[i] = amplitude * std::sin(M_PI * x[i] / length);
  return phi;
}

inline std::vector<double> sinusoidalCharge1D(const std::vector<double>& x,
                                              double epsilon,
                                              double amplitude = 1.0,
                                              double length = 1.0) {
  if (!(epsilon > 0.0)) throw std::runtime_error("epsilon must be positive");
  const double k = M_PI / length;
  std::vector<double> phi = sinusoidalPotential1D(x, amplitude, length);
  for (double& value : phi) value *= epsilon * k * k;
  return phi;
}

inline std::vector<double> separablePotential2D(const std::vector<double>& x,
                                                const std::vector<double>& y,
                                                double amplitude = 1.0,
                                                double lx = 1.0,
                                                double ly = 1.0) {
  if (x.size() != y.size()) throw std::runtime_error("x and y must have matching shapes");
  if (!(lx > 0.0) || !(ly > 0.0)) throw std::runtime_error("lengths must be positive");
  std::vector<double> phi(x.size(), 0.0);
  for (size_t i = 0; i < x.size(); ++i) {
    phi[i] = amplitude * std::sin(M_PI * x[i] / lx) * std::cos(M_PI * y[i] / ly);
  }
  return phi;
}

inline std::vector<double> separableCharge2D(const std::vector<double>& x,
                                             const std::vector<double>& y,
                                             double epsilon,
                                             double amplitude = 1.0,
                                             double lx = 1.0,
                                             double ly = 1.0) {
  if (!(epsilon > 0.0)) throw std::runtime_error("epsilon must be positive");
  const double k2 = std::pow(M_PI / lx, 2) + std::pow(M_PI / ly, 2);
  std::vector<double> phi = separablePotential2D(x, y, amplitude, lx, ly);
  for (double& value : phi) value *= epsilon * k2;
  return phi;
}

}  // namespace electrospray
