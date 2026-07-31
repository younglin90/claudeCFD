#pragma once

#include "electrospray/Diagnostics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace electrospray {

constexpr double standardGravity = 9.80665;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Propellant {
  std::string name;
  double density = 0.0;
  double viscosity = 0.0;
  double surfaceTension = 0.0;
  double conductivity = 0.0;
  double permittivity = 0.0;
};

inline void validatePropellant(const Propellant& propellant) {
  if (propellant.name.empty()) throw std::runtime_error("name must be non-empty");
  if (!(propellant.density > 0.0)) throw std::runtime_error("density must be positive");
  if (!(propellant.viscosity > 0.0)) throw std::runtime_error("viscosity must be positive");
  if (!(propellant.surfaceTension > 0.0)) throw std::runtime_error("surface_tension must be positive");
  if (!(propellant.conductivity > 0.0)) throw std::runtime_error("conductivity must be positive");
  if (!(propellant.permittivity > 0.0)) throw std::runtime_error("permittivity must be positive");
}

inline Propellant emiBf4Reference() {
  return {"EMI-BF4-reference", 1280.0, 0.036, 0.052, 1.4, 12.0};
}

inline double massFlowFromVolumeFlow(const Propellant& propellant, double volumeFlowRate) {
  validatePropellant(propellant);
  if (volumeFlowRate < 0.0) throw std::runtime_error("volume_flow_rate must be non-negative");
  return propellant.density * volumeFlowRate;
}

inline double currentPerEmitter(double totalCurrent, int emitterCount) {
  if (totalCurrent < 0.0) throw std::runtime_error("total_current must be non-negative");
  if (emitterCount <= 0) throw std::runtime_error("emitter_count must be positive");
  return totalCurrent / static_cast<double>(emitterCount);
}

inline double specificCharge(double current, double massFlowRate) {
  if (!(massFlowRate > 0.0)) throw std::runtime_error("mass_flow_rate must be positive");
  return current / massFlowRate;
}

inline double idealExhaustVelocity(double chargeToMass, double accelerationVoltage) {
  if (chargeToMass < 0.0) throw std::runtime_error("charge_to_mass must be non-negative");
  if (accelerationVoltage < 0.0) throw std::runtime_error("acceleration_voltage must be non-negative");
  return std::sqrt(2.0 * chargeToMass * accelerationVoltage);
}

inline double thrustFromMassFlow(double massFlowRate, double exhaustVelocity) {
  if (massFlowRate < 0.0) throw std::runtime_error("mass_flow_rate must be non-negative");
  if (exhaustVelocity < 0.0) throw std::runtime_error("exhaust_velocity must be non-negative");
  return massFlowRate * exhaustVelocity;
}

inline double electricalPower(double current, double voltage) {
  if (current < 0.0) throw std::runtime_error("current must be non-negative");
  if (voltage < 0.0) throw std::runtime_error("voltage must be non-negative");
  return current * voltage;
}

inline double kineticPower(double massFlowRate, double exhaustVelocity) {
  if (massFlowRate < 0.0) throw std::runtime_error("mass_flow_rate must be non-negative");
  if (exhaustVelocity < 0.0) throw std::runtime_error("exhaust_velocity must be non-negative");
  return 0.5 * massFlowRate * exhaustVelocity * exhaustVelocity;
}

inline double idealPowerEfficiency(double massFlowRate, double exhaustVelocity, double current, double voltage) {
  const double power = electricalPower(current, voltage);
  if (!(power > 0.0)) throw std::runtime_error("electrical power must be positive");
  return kineticPower(massFlowRate, exhaustVelocity) / power;
}

inline double thrustToPower(double thrust, double power) {
  if (thrust < 0.0) throw std::runtime_error("thrust must be non-negative");
  if (!(power > 0.0)) throw std::runtime_error("power must be positive");
  return thrust / power;
}

inline double specificImpulse(double exhaustVelocity, double gravity = standardGravity) {
  if (!(gravity > 0.0)) throw std::runtime_error("gravity must be positive");
  if (exhaustVelocity < 0.0) throw std::runtime_error("exhaust_velocity must be non-negative");
  return exhaustVelocity / gravity;
}

inline double specificImpulseFromThrust(double thrust, double massFlowRate, double gravity = standardGravity) {
  if (thrust < 0.0) throw std::runtime_error("thrust must be non-negative");
  if (!(massFlowRate > 0.0)) throw std::runtime_error("mass_flow_rate must be positive");
  if (!(gravity > 0.0)) throw std::runtime_error("gravity must be positive");
  return thrust / (massFlowRate * gravity);
}

struct OperatingPoint {
  double massFlowRate = 0.0;
  double chargeToMass = 0.0;
  double exhaustVelocity = 0.0;
  double thrust = 0.0;
  double specificImpulseValue = 0.0;
  double electricalPowerValue = 0.0;
  double kineticPowerValue = 0.0;
  double idealEfficiency = 0.0;
  double thrustToPowerValue = 0.0;
};

inline double extractorField(double voltage, double gap) {
  if (!(gap > 0.0)) throw std::runtime_error("gap must be positive");
  return std::abs(voltage) / gap;
}

inline double extractorOpenAreaFraction(double apertureRadius, double pitch) {
  if (!(apertureRadius > 0.0) || !(pitch > 0.0)) {
    throw std::runtime_error("aperture_radius and pitch must be positive");
  }
  const double fraction = M_PI * apertureRadius * apertureRadius / (pitch * pitch);
  if (fraction > 1.0) throw std::runtime_error("aperture area cannot exceed pitch cell area");
  return fraction;
}

inline double arrayCurrentDensity(double totalCurrent, double apertureRadius, int emitterCount) {
  if (totalCurrent < 0.0) throw std::runtime_error("total_current must be non-negative");
  if (!(apertureRadius > 0.0)) throw std::runtime_error("aperture_radius must be positive");
  if (emitterCount <= 0) throw std::runtime_error("emitter_count must be positive");
  return totalCurrent / (static_cast<double>(emitterCount) * M_PI * apertureRadius * apertureRadius);
}

inline OperatingPoint operatingPoint(const Propellant& propellant,
                                     double volumeFlowRate,
                                     double current,
                                     double accelerationVoltage) {
  if (current < 0.0) throw std::runtime_error("current must be non-negative");
  const double mdot = massFlowFromVolumeFlow(propellant, volumeFlowRate);
  const double qOverM = specificCharge(current, mdot);
  const double vex = idealExhaustVelocity(qOverM, accelerationVoltage);
  const double thrust = thrustFromMassFlow(mdot, vex);
  const double beamPower = electricalPower(current, accelerationVoltage);
  const double jetPower = kineticPower(mdot, vex);
  return {mdot,
          qOverM,
          vex,
          thrust,
          specificImpulse(vex),
          beamPower,
          jetPower,
          idealPowerEfficiency(mdot, vex, current, accelerationVoltage),
          thrustToPower(thrust, beamPower)};
}

inline Vec3 ballisticPosition(Vec3 initialPosition, Vec3 velocity, double time, Vec3 acceleration = {}) {
  if (time < 0.0) throw std::runtime_error("time must be non-negative");
  return {initialPosition.x + velocity.x * time + 0.5 * acceleration.x * time * time,
          initialPosition.y + velocity.y * time + 0.5 * acceleration.y * time * time,
          initialPosition.z + velocity.z * time + 0.5 * acceleration.z * time * time};
}

inline double plumeHalfAngle(const std::vector<Vec3>& velocities) {
  if (velocities.empty()) throw std::runtime_error("velocity_vectors must be non-empty");
  double angle = 0.0;
  for (const Vec3& v : velocities) {
    if (!(v.z > 0.0)) throw std::runtime_error("all particles must have positive axial velocity");
    angle = std::max(angle, std::atan2(std::hypot(v.x, v.y), v.z));
  }
  return angle;
}

inline double circularPlaneImpingementFraction(const std::vector<Vec3>& positions,
                                               double planeZ,
                                               double radius,
                                               Vec2 origin = {}) {
  if (positions.empty()) throw std::runtime_error("positions must be non-empty");
  if (radius < 0.0) throw std::runtime_error("radius must be non-negative");
  int hits = 0;
  for (const Vec3& p : positions) {
    const double dx = p.x - origin.x;
    const double dy = p.y - origin.y;
    if (std::abs(p.z - planeZ) <= 1e-12 && dx * dx + dy * dy <= radius * radius) ++hits;
  }
  return static_cast<double>(hits) / static_cast<double>(positions.size());
}

struct WeightedTracks {
  std::vector<Vec3> positions;
  std::vector<double> weights;
};

inline WeightedTracks deterministicConicalTracksToPlane(const std::vector<Vec2>& emitters,
                                                        double planeZ,
                                                        double halfAngle,
                                                        int particlesPerEmitter,
                                                        std::vector<double> currentWeights = {}) {
  if (emitters.empty()) throw std::runtime_error("emitter_positions_xy must be non-empty");
  if (!(planeZ > 0.0)) throw std::runtime_error("plane_z must be positive");
  if (halfAngle < 0.0 || halfAngle >= 0.5 * M_PI) throw std::runtime_error("half_angle must be in [0, pi/2)");
  if (particlesPerEmitter <= 0) throw std::runtime_error("particles_per_emitter must be positive");
  if (currentWeights.empty()) currentWeights.assign(emitters.size(), 1.0);
  if (currentWeights.size() != emitters.size()) throw std::runtime_error("current_weights must match emitter count");
  for (double w : currentWeights) {
    if (w < 0.0) throw std::runtime_error("current_weights must be non-negative");
  }
  const double plumeRadius = planeZ * std::tan(halfAngle);
  const double goldenAngle = M_PI * (3.0 - std::sqrt(5.0));
  WeightedTracks tracks;
  tracks.positions.reserve(emitters.size() * static_cast<size_t>(particlesPerEmitter));
  tracks.weights.reserve(tracks.positions.capacity());
  for (size_t e = 0; e < emitters.size(); ++e) {
    for (int k = 0; k < particlesPerEmitter; ++k) {
      const double local = static_cast<double>(k) + 0.5;
      const double radial = plumeRadius * std::sqrt(local / static_cast<double>(particlesPerEmitter));
      const double theta = goldenAngle * local + static_cast<double>(e) * 0.5 * M_PI;
      tracks.positions.push_back({emitters[e].x + radial * std::cos(theta),
                                  emitters[e].y + radial * std::sin(theta),
                                  planeZ});
      tracks.weights.push_back(currentWeights[e] / static_cast<double>(particlesPerEmitter));
    }
  }
  return tracks;
}

struct RectangularPanelTrackingResult {
  std::vector<Vec3> particlePositions;
  std::vector<double> particleWeights;
  std::vector<bool> hitMask;
  double impingementFraction = 0.0;
  double depositedWeight = 0.0;
  double retainedWeight = 0.0;
  double weightBalanceError = 0.0;
};

inline RectangularPanelTrackingResult weightedRectangularPanelTracking(const std::vector<Vec3>& positions,
                                                                       const std::vector<double>& weights,
                                                                       double planeZ,
                                                                       double width,
                                                                       double height,
                                                                       Vec2 center = {}) {
  if (positions.empty()) throw std::runtime_error("particle_positions must be non-empty");
  if (weights.size() != positions.size()) throw std::runtime_error("particle_weights must match particle count");
  if (width < 0.0 || height < 0.0) throw std::runtime_error("width and height must be non-negative");
  double total = 0.0;
  for (double w : weights) {
    if (w < 0.0) throw std::runtime_error("particle_weights must be non-negative");
    total += w;
  }
  if (!(total > 0.0)) throw std::runtime_error("total particle weight must be positive");
  RectangularPanelTrackingResult result;
  result.particlePositions = positions;
  result.particleWeights = weights;
  result.hitMask.assign(positions.size(), false);
  const double halfWidth = 0.5 * width;
  const double halfHeight = 0.5 * height;
  for (size_t i = 0; i < positions.size(); ++i) {
    const double rx = positions[i].x - center.x;
    const double ry = positions[i].y - center.y;
    const bool hit = std::abs(positions[i].z - planeZ) <= 1e-12 && std::abs(rx) <= halfWidth &&
                     std::abs(ry) <= halfHeight;
    result.hitMask[i] = hit;
    if (hit) result.depositedWeight += weights[i];
  }
  result.retainedWeight = total - result.depositedWeight;
  result.impingementFraction = result.depositedWeight / total;
  result.weightBalanceError = std::abs((result.depositedWeight + result.retainedWeight) / total - 1.0);
  return result;
}

inline double plumeHalfAngleFromJetAndDroplet(double jetDiameter, double dropletDiameter) {
  if (!(jetDiameter > 0.0)) throw std::runtime_error("jet_diameter must be positive");
  if (!(dropletDiameter > 0.0)) throw std::runtime_error("droplet_diameter must be positive");
  return std::atan(0.5 * jetDiameter / dropletDiameter);
}

inline double rectangularPanelImpingementFraction(const std::vector<Vec3>& positions,
                                                  double planeZ,
                                                  double width,
                                                  double height,
                                                  Vec2 center = {}) {
  std::vector<double> weights(positions.size(), 1.0);
  return weightedRectangularPanelTracking(positions, weights, planeZ, width, height, center).impingementFraction;
}

inline double depositedCurrent(double totalCurrent, double impingementFraction) {
  if (totalCurrent < 0.0) throw std::runtime_error("total_current must be non-negative");
  if (impingementFraction < 0.0 || impingementFraction > 1.0) {
    throw std::runtime_error("impingement_fraction must be in [0, 1]");
  }
  return totalCurrent * impingementFraction;
}

inline double retainedCurrent(double totalCurrent, double impingementFraction) {
  return totalCurrent - depositedCurrent(totalCurrent, impingementFraction);
}

inline double panelCurrentDensity(double totalCurrent, double impingementFraction, double panelArea) {
  if (!(panelArea > 0.0)) throw std::runtime_error("panel_area must be positive");
  return depositedCurrent(totalCurrent, impingementFraction) / panelArea;
}

inline double depositedMassFlow(double totalMassFlow, double impingementFraction) {
  if (totalMassFlow < 0.0) throw std::runtime_error("total_mass_flow must be non-negative");
  if (impingementFraction < 0.0 || impingementFraction > 1.0) {
    throw std::runtime_error("impingement_fraction must be in [0, 1]");
  }
  return totalMassFlow * impingementFraction;
}

inline double retainedMassFlow(double totalMassFlow, double impingementFraction) {
  return totalMassFlow - depositedMassFlow(totalMassFlow, impingementFraction);
}

inline double panelMassFlux(double totalMassFlow, double impingementFraction, double panelArea) {
  if (!(panelArea > 0.0)) throw std::runtime_error("panel_area must be positive");
  return depositedMassFlow(totalMassFlow, impingementFraction) / panelArea;
}

inline double accumulatedPanelMassLoading(double massFlux, double exposureTime) {
  if (massFlux < 0.0) throw std::runtime_error("mass_flux must be non-negative");
  if (exposureTime < 0.0) throw std::runtime_error("exposure_time must be non-negative");
  return massFlux * exposureTime;
}

inline double timeToPanelMassLoading(double limit, double massFlux) {
  if (limit < 0.0) throw std::runtime_error("limit must be non-negative");
  if (!(massFlux > 0.0)) throw std::runtime_error("mass_flux must be positive");
  return limit / massFlux;
}

inline double exposureMargin(double timeToLimit, double exposureTime) {
  if (timeToLimit < 0.0) throw std::runtime_error("time_to_limit must be non-negative");
  if (!(exposureTime > 0.0)) throw std::runtime_error("exposure_time must be positive");
  return timeToLimit / exposureTime;
}

inline std::string exposureMarginStatus(double margin) {
  if (margin < 0.0) throw std::runtime_error("margin must be non-negative");
  return margin >= 1.0 ? "pass" : "fail";
}

inline double thrustLossFraction(double impingementFraction, double axialMomentumFraction = 1.0) {
  if (impingementFraction < 0.0 || impingementFraction > 1.0) {
    throw std::runtime_error("impingement_fraction must be in [0, 1]");
  }
  if (axialMomentumFraction < 0.0 || axialMomentumFraction > 1.0) {
    throw std::runtime_error("axial_momentum_fraction must be in [0, 1]");
  }
  return impingementFraction * axialMomentumFraction;
}

inline double retainedThrustFraction(double impingementFraction, double axialMomentumFraction = 1.0) {
  return 1.0 - thrustLossFraction(impingementFraction, axialMomentumFraction);
}

inline double effectiveThrustAfterImpingement(double thrust,
                                              double impingementFraction,
                                              double axialMomentumFraction = 1.0) {
  if (thrust < 0.0) throw std::runtime_error("thrust must be non-negative");
  return thrust * retainedThrustFraction(impingementFraction, axialMomentumFraction);
}

}  // namespace electrospray
