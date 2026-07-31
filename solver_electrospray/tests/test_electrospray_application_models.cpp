#include "TestUtil.hpp"
#include "electrospray/ApplicationModels.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checkClose(double got, double expected, double tol, const std::string& msg) {
  const double scale = std::max(1.0, std::abs(expected));
  check(std::abs(got - expected) <= tol * scale, msg + " got=" + std::to_string(got) +
                                                    " expected=" + std::to_string(expected));
}

void checkThrows(void (*fn)(), const std::string& msg) {
  bool threw = false;
  try {
    fn();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, msg);
}

}  // namespace

int main() {
  using namespace electrospray;

  const Propellant propellant = emiBf4Reference();
  check(propellant.name == "EMI-BF4-reference", "reference propellant name");
  checkClose(massFlowFromVolumeFlow(propellant, 2.0e-12), 2.56e-9, 1e-14, "mass flow");
  checkClose(currentPerEmitter(8.0e-6, 4), 2.0e-6, 1e-14, "current per emitter");

  const double mdot = 2.0e-9;
  const double current = 4.0e-6;
  const double voltage = 1200.0;
  const double qOverM = specificCharge(current, mdot);
  const double vex = idealExhaustVelocity(qOverM, voltage);
  checkClose(qOverM, 2000.0, 1e-14, "specific charge");
  checkClose(vex, std::sqrt(2.0 * 2000.0 * 1200.0), 1e-14, "ideal exhaust velocity");
  checkClose(thrustFromMassFlow(mdot, vex), mdot * vex, 1e-14, "thrust from mass flow");
  checkClose(electricalPower(current, voltage), 4.8e-3, 1e-14, "electrical power");
  checkClose(kineticPower(mdot, vex), 0.5 * mdot * vex * vex, 1e-14, "kinetic power");
  checkClose(idealPowerEfficiency(mdot, vex, current, voltage), kineticPower(mdot, vex) / 4.8e-3, 1e-14,
             "ideal power efficiency");
  checkClose(thrustToPower(thrustFromMassFlow(mdot, vex), 4.8e-3), thrustFromMassFlow(mdot, vex) / 4.8e-3,
             1e-14, "thrust to power");
  checkClose(specificImpulse(vex), vex / standardGravity, 1e-14, "specific impulse");
  checkClose(specificImpulseFromThrust(mdot * vex, mdot), vex / standardGravity, 1e-14,
             "specific impulse from thrust");

  const OperatingPoint op = operatingPoint(propellant, 2.0e-12, current, voltage);
  checkClose(op.massFlowRate, 2.56e-9, 1e-14, "operating point mass flow");
  checkClose(op.chargeToMass, current / 2.56e-9, 1e-14, "operating point q/m");
  checkClose(op.thrust, op.massFlowRate * op.exhaustVelocity, 1e-14, "operating point thrust");
  checkClose(extractorField(-1500.0, 0.003), 5.0e5, 1e-14, "extractor field uses voltage magnitude");
  checkClose(extractorOpenAreaFraction(1.0e-4, 5.0e-4), M_PI * 1.0e-8 / 2.5e-7, 1e-14,
             "extractor open area fraction");
  checkClose(arrayCurrentDensity(4.0e-6, 1.0e-4, 4), 4.0e-6 / (4.0 * M_PI * 1.0e-8), 1e-14,
             "array current density");

  const Vec3 ballistic = ballisticPosition(Vec3{1.0, 2.0, 3.0}, Vec3{4.0, 5.0, 6.0}, 0.5, Vec3{2.0, 0.0, -2.0});
  checkClose(ballistic.x, 1.0 + 2.0 + 0.25, 1e-14, "ballistic x");
  checkClose(ballistic.y, 4.5, 1e-14, "ballistic y");
  checkClose(ballistic.z, 3.0 + 3.0 - 0.25, 1e-14, "ballistic z");
  checkClose(plumeHalfAngle({Vec3{1.0, 0.0, 1.0}, Vec3{0.0, 2.0, 2.0}}), M_PI / 4.0, 1e-14,
             "plume half angle");
  checkClose(circularPlaneImpingementFraction({Vec3{0.0, 0.0, 2.0}, Vec3{2.0, 0.0, 2.0}}, 2.0, 1.0),
             0.5, 1e-14, "circular impingement fraction");

  const std::vector<Vec2> emitters{Vec2{-0.1, 0.0}, Vec2{0.1, 0.0}};
  const WeightedTracks tracks = deterministicConicalTracksToPlane(emitters, 2.0, 0.1, 8, {2.0, 4.0});
  check(tracks.positions.size() == 16, "deterministic conical track count");
  check(tracks.weights.size() == 16, "deterministic conical weight count");
  double weightSum = 0.0;
  for (double w : tracks.weights) weightSum += w;
  checkClose(weightSum, 6.0, 1e-14, "track weights conserve emitter weights");
  for (const Vec3& p : tracks.positions) checkClose(p.z, 2.0, 1e-14, "track endpoint plane");

  const RectangularPanelTrackingResult panel =
      weightedRectangularPanelTracking({Vec3{0.0, 0.0, 1.0}, Vec3{2.0, 0.0, 1.0}},
                                       {3.0, 1.0}, 1.0, 1.0, 1.0);
  checkClose(panel.impingementFraction, 0.75, 1e-14, "weighted rectangular panel fraction");
  checkClose(panel.depositedWeight, 3.0, 1e-14, "weighted deposited weight");
  checkClose(panel.retainedWeight, 1.0, 1e-14, "weighted retained weight");
  checkClose(panel.weightBalanceError, 0.0, 1e-14, "weighted balance error");
  checkClose(rectangularPanelImpingementFraction({Vec3{0.0, 0.0, 1.0}, Vec3{2.0, 0.0, 1.0}},
                                                 1.0, 1.0, 1.0),
             0.5, 1e-14, "unweighted rectangular panel fraction");
  checkClose(plumeHalfAngleFromJetAndDroplet(2.0e-6, 4.0e-6), std::atan(0.25), 1e-14,
             "plume angle from jet and droplet");

  checkClose(depositedCurrent(10.0, 0.3), 3.0, 1e-14, "deposited current");
  checkClose(retainedCurrent(10.0, 0.3), 7.0, 1e-14, "retained current");
  checkClose(panelCurrentDensity(10.0, 0.3, 2.0), 1.5, 1e-14, "panel current density");
  checkClose(depositedMassFlow(8.0, 0.25), 2.0, 1e-14, "deposited mass flow");
  checkClose(retainedMassFlow(8.0, 0.25), 6.0, 1e-14, "retained mass flow");
  checkClose(panelMassFlux(8.0, 0.25, 4.0), 0.5, 1e-14, "panel mass flux");
  checkClose(accumulatedPanelMassLoading(0.5, 10.0), 5.0, 1e-14, "mass loading");
  checkClose(timeToPanelMassLoading(5.0, 0.5), 10.0, 1e-14, "time to mass loading");
  checkClose(exposureMargin(10.0, 5.0), 2.0, 1e-14, "exposure margin");
  check(exposureMarginStatus(1.0) == "pass", "margin pass");
  check(exposureMarginStatus(0.99) == "fail", "margin fail");
  checkClose(thrustLossFraction(0.2, 0.5), 0.1, 1e-14, "thrust loss fraction");
  checkClose(retainedThrustFraction(0.2, 0.5), 0.9, 1e-14, "retained thrust fraction");
  checkClose(effectiveThrustAfterImpingement(4.0, 0.2, 0.5), 3.6, 1e-14, "effective thrust");

  checkThrows([]() { massFlowFromVolumeFlow(Propellant{"bad", 0.0, 1.0, 1.0, 1.0, 1.0}, 1.0); },
              "invalid propellant rejected");
  checkThrows([]() { idealExhaustVelocity(-1.0, 1.0); }, "negative q/m rejected");
  checkThrows([]() { extractorOpenAreaFraction(2.0, 1.0); }, "overfilled aperture rejected");
  checkThrows([]() { plumeHalfAngle({Vec3{1.0, 0.0, 0.0}}); }, "nonpositive axial velocity rejected");
  checkThrows([]() { deterministicConicalTracksToPlane({Vec2{0.0, 0.0}}, 1.0, M_PI, 1); },
              "invalid plume half angle rejected");
  checkThrows([]() { weightedRectangularPanelTracking({Vec3{}}, {0.0}, 1.0, 1.0, 1.0); },
              "zero total weight rejected");
  checkThrows([]() { depositedCurrent(1.0, 1.2); }, "invalid impingement fraction rejected");

  std::cout << "electrospray application model C++ checks passed\n";
  return 0;
}
