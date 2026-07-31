#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace electrospray {

struct LeakyDielectricMaterial {
  double permittivity = 0.0;
  double conductivity = 0.0;
  double density = std::numeric_limits<double>::quiet_NaN();
  double dynamicViscosity = std::numeric_limits<double>::quiet_NaN();

  LeakyDielectricMaterial(double eps, double sigma,
                          double rho = std::numeric_limits<double>::quiet_NaN(),
                          double mu = std::numeric_limits<double>::quiet_NaN())
      : permittivity(eps), conductivity(sigma), density(rho), dynamicViscosity(mu) {
    validatePositive("permittivity", permittivity);
    validatePositive("conductivity", conductivity);
    if (hasDensity()) validatePositive("density", density);
    if (hasDynamicViscosity()) validatePositive("dynamic_viscosity", dynamicViscosity);
  }

  static void validatePositive(const std::string& name, double value) {
    if (!(value > 0.0)) throw std::runtime_error(name + " must be positive");
  }

  bool hasDensity() const { return !std::isnan(density); }
  bool hasDynamicViscosity() const { return !std::isnan(dynamicViscosity); }

  double relaxationTime() const { return permittivity / conductivity; }

  double relaxationFactor(double dt) const {
    if (dt < 0.0) throw std::runtime_error("dt must be non-negative");
    return std::exp(-dt / relaxationTime());
  }
};

inline void validateAlpha(double alpha) {
  if (alpha < 0.0 || alpha > 1.0) throw std::runtime_error("alpha_liquid must stay in [0, 1]");
}

inline double mixtureProperty(double alphaLiquid, double liquidValue, double gasValue) {
  LeakyDielectricMaterial::validatePositive("liquid_value", liquidValue);
  LeakyDielectricMaterial::validatePositive("gas_value", gasValue);
  validateAlpha(alphaLiquid);
  return alphaLiquid * liquidValue + (1.0 - alphaLiquid) * gasValue;
}

inline std::vector<double> mixtureProperty(const std::vector<double>& alphaLiquid,
                                           double liquidValue,
                                           double gasValue) {
  std::vector<double> out(alphaLiquid.size(), 0.0);
  for (size_t i = 0; i < alphaLiquid.size(); ++i) {
    out[i] = mixtureProperty(alphaLiquid[i], liquidValue, gasValue);
  }
  return out;
}

inline double harmonicFaceProperty(double leftValue, double rightValue) {
  if (!(leftValue > 0.0) || !(rightValue > 0.0)) {
    throw std::runtime_error("face-neighbor properties must be positive");
  }
  return 2.0 * leftValue * rightValue / (leftValue + rightValue);
}

inline std::vector<double> harmonicFaceProperty(const std::vector<double>& leftValue,
                                                const std::vector<double>& rightValue) {
  if (leftValue.size() != rightValue.size()) throw std::runtime_error("face field size mismatch");
  std::vector<double> out(leftValue.size(), 0.0);
  for (size_t i = 0; i < leftValue.size(); ++i) out[i] = harmonicFaceProperty(leftValue[i], rightValue[i]);
  return out;
}

inline std::pair<std::vector<double>, std::vector<double>> leakyDielectricProperties(
    const std::vector<double>& alphaLiquid,
    double epsilonLiquid,
    double epsilonGas,
    double sigmaLiquid,
    double sigmaGas) {
  return {mixtureProperty(alphaLiquid, epsilonLiquid, epsilonGas),
          mixtureProperty(alphaLiquid, sigmaLiquid, sigmaGas)};
}

struct LeakyDielectricPhasePair {
  LeakyDielectricMaterial liquid;
  LeakyDielectricMaterial gas;

  std::pair<std::vector<double>, std::vector<double>> mixtureFields(const std::vector<double>& alphaLiquid) const {
    return leakyDielectricProperties(alphaLiquid, liquid.permittivity, gas.permittivity, liquid.conductivity,
                                     gas.conductivity);
  }

  std::pair<std::vector<double>, std::vector<double>> harmonicFaceFields(
      const std::vector<double>& alphaLeft,
      const std::vector<double>& alphaRight) const {
    auto [epsLeft, sigmaLeft] = mixtureFields(alphaLeft);
    auto [epsRight, sigmaRight] = mixtureFields(alphaRight);
    return {harmonicFaceProperty(epsLeft, epsRight), harmonicFaceProperty(sigmaLeft, sigmaRight)};
  }

  std::vector<double> relaxationTimeField(const std::vector<double>& alphaLiquid) const {
    auto [eps, sigma] = mixtureFields(alphaLiquid);
    std::vector<double> out(eps.size(), 0.0);
    for (size_t i = 0; i < eps.size(); ++i) out[i] = eps[i] / sigma[i];
    return out;
  }

  std::vector<double> densityField(const std::vector<double>& alphaLiquid, double fallbackDensity) const {
    LeakyDielectricMaterial::validatePositive("fallback_density", fallbackDensity);
    if (!liquid.hasDensity() || !gas.hasDensity()) return std::vector<double>(alphaLiquid.size(), fallbackDensity);
    return mixtureProperty(alphaLiquid, liquid.density, gas.density);
  }

  std::vector<double> dynamicViscosityField(const std::vector<double>& alphaLiquid,
                                            double fallbackDynamicViscosity) const {
    if (fallbackDynamicViscosity < 0.0) throw std::runtime_error("fallback_dynamic_viscosity must be non-negative");
    if (!liquid.hasDynamicViscosity() || !gas.hasDynamicViscosity()) {
      return std::vector<double>(alphaLiquid.size(), fallbackDynamicViscosity);
    }
    return mixtureProperty(alphaLiquid, liquid.dynamicViscosity, gas.dynamicViscosity);
  }

  std::vector<double> kinematicViscosityField(const std::vector<double>& alphaLiquid,
                                              double fallbackDensity,
                                              double fallbackKinematicViscosity) const {
    if (fallbackKinematicViscosity < 0.0) throw std::runtime_error("fallback_kinematic_viscosity must be non-negative");
    const std::vector<double> rho = densityField(alphaLiquid, fallbackDensity);
    if (!liquid.hasDynamicViscosity() || !gas.hasDynamicViscosity()) {
      return std::vector<double>(alphaLiquid.size(), fallbackKinematicViscosity);
    }
    const std::vector<double> mu = dynamicViscosityField(alphaLiquid, fallbackDensity * fallbackKinematicViscosity);
    std::vector<double> nu(alphaLiquid.size(), 0.0);
    for (size_t i = 0; i < alphaLiquid.size(); ++i) nu[i] = mu[i] / rho[i];
    return nu;
  }

  double permittivityRatio() const { return liquid.permittivity / gas.permittivity; }
  double conductivityRatio() const { return liquid.conductivity / gas.conductivity; }
};

inline double electricCapillaryNumber(double permittivity, double electricField, double radius, double surfaceTension) {
  if (!(permittivity > 0.0) || !(radius > 0.0) || !(surfaceTension > 0.0)) {
    throw std::runtime_error("permittivity, radius, and surface_tension must be positive");
  }
  return permittivity * electricField * electricField * radius / surfaceTension;
}

inline double chargeRelaxationTime(double permittivity, double conductivity) {
  if (!(permittivity > 0.0)) throw std::runtime_error("permittivity must be positive");
  if (!(conductivity > 0.0)) throw std::runtime_error("conductivity must be positive");
  return permittivity / conductivity;
}

inline double electricReynoldsNumber(double permittivity, double velocity, double length, double conductivity) {
  if (!(length > 0.0)) throw std::runtime_error("length must be positive");
  return chargeRelaxationTime(permittivity, conductivity) * std::abs(velocity) / length;
}

inline double ohnesorgeNumber(double viscosity, double density, double surfaceTension, double radius) {
  if (viscosity < 0.0) throw std::runtime_error("viscosity must be non-negative");
  if (!(density > 0.0) || !(surfaceTension > 0.0) || !(radius > 0.0)) {
    throw std::runtime_error("density, surface_tension, and radius must be positive");
  }
  return viscosity / std::sqrt(density * surfaceTension * radius);
}

inline double flowRateParameter(double flowRate, double radius, double surfaceTension, double density) {
  if (flowRate < 0.0) throw std::runtime_error("flow_rate must be non-negative");
  if (!(radius > 0.0) || !(surfaceTension > 0.0) || !(density > 0.0)) {
    throw std::runtime_error("radius, surface_tension, and density must be positive");
  }
  return flowRate / std::sqrt(surfaceTension * std::pow(radius, 5) / density);
}

inline double advectiveDt(double dx, double maxVelocity, double cfl) {
  if (!(dx > 0.0) || !(cfl > 0.0)) throw std::runtime_error("dx and cfl must be positive");
  const double speed = std::abs(maxVelocity);
  if (speed == 0.0) return std::numeric_limits<double>::infinity();
  return cfl * dx / speed;
}

inline double diffusiveDt(double dx, double diffusivity, double safety = 0.5) {
  if (!(dx > 0.0) || !(safety > 0.0)) throw std::runtime_error("dx and safety must be positive");
  if (diffusivity < 0.0) throw std::runtime_error("diffusivity must be non-negative");
  if (diffusivity == 0.0) return std::numeric_limits<double>::infinity();
  return safety * dx * dx / diffusivity;
}

inline double electricRelaxationDt(double permittivity, double conductivity, double safety = 1.0) {
  if (!(permittivity > 0.0) || !(safety > 0.0)) throw std::runtime_error("permittivity and safety must be positive");
  if (conductivity < 0.0) throw std::runtime_error("conductivity must be non-negative");
  if (conductivity == 0.0) return std::numeric_limits<double>::infinity();
  return safety * permittivity / conductivity;
}

inline double electricRelaxationDt(const LeakyDielectricMaterial& material, double safety = 1.0) {
  if (!(safety > 0.0)) throw std::runtime_error("safety must be positive");
  return safety * material.relaxationTime();
}

inline double electricRelaxationDt(const LeakyDielectricPhasePair& phasePair,
                                   const std::vector<double>& alphaLiquid,
                                   double safety = 1.0) {
  if (!(safety > 0.0)) throw std::runtime_error("safety must be positive");
  std::vector<double> tau = phasePair.relaxationTimeField(alphaLiquid);
  return safety * *std::min_element(tau.begin(), tau.end());
}

inline double capillaryDt(double dx, double density, double surfaceTension, double safety = 1.0) {
  if (!(dx > 0.0) || !(density > 0.0) || !(safety > 0.0)) {
    throw std::runtime_error("dx, density, and safety must be positive");
  }
  if (surfaceTension < 0.0) throw std::runtime_error("surface_tension must be non-negative");
  if (surfaceTension == 0.0) return std::numeric_limits<double>::infinity();
  return safety * std::sqrt(density * dx * dx * dx / surfaceTension);
}

inline double combinedExplicitDt(const std::vector<double>& limits) {
  if (limits.empty()) throw std::runtime_error("at least one limit is required");
  for (double limit : limits) {
    if (!(limit > 0.0)) throw std::runtime_error("limits must be positive");
  }
  return *std::min_element(limits.begin(), limits.end());
}

}  // namespace electrospray
