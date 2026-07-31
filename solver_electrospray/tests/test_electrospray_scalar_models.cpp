#include "TestUtil.hpp"
#include "electrospray/MaterialProperties.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
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

  const LeakyDielectricMaterial liquid(4.0e-10, 2.0e-8, 1000.0, 1.5e-3);
  const LeakyDielectricMaterial gas(1.0e-11, 5.0e-10, 1.2, 1.8e-5);
  const LeakyDielectricPhasePair pair{liquid, gas};
  const std::vector<double> alpha{0.0, 0.25, 0.5, 1.0};

  checkClose(liquid.relaxationTime(), 0.02, 1e-14, "material relaxation time matches Python epsilon/sigma");
  checkClose(liquid.relaxationFactor(0.01), std::exp(-0.5), 1e-14, "exact relaxation factor matches Python exp");

  auto [eps, sigma] = pair.mixtureFields(alpha);
  checkClose(eps[0], 1.0e-11, 1e-14, "gas-side permittivity mixture");
  checkClose(eps[1], 1.075e-10, 1e-14, "quarter-liquid permittivity mixture");
  checkClose(eps[2], 2.05e-10, 1e-14, "half-liquid permittivity mixture");
  checkClose(sigma[3], 2.0e-8, 1e-14, "liquid-side conductivity mixture");

  auto [faceEps, faceSigma] = pair.harmonicFaceFields(std::vector<double>{0.0, 0.5}, std::vector<double>{0.5, 1.0});
  checkClose(faceEps[0], 2.0 * 1.0e-11 * 2.05e-10 / (1.0e-11 + 2.05e-10), 1e-14,
             "harmonic face permittivity matches Python formula");
  checkClose(faceSigma[1], 2.0 * 1.025e-8 * 2.0e-8 / (1.025e-8 + 2.0e-8), 1e-14,
             "harmonic face conductivity matches Python formula");

  const auto rho = pair.densityField(alpha, 42.0);
  const auto mu = pair.dynamicViscosityField(alpha, 7.0e-4);
  const auto nu = pair.kinematicViscosityField(alpha, 1000.0, 1.0e-6);
  checkClose(rho[2], 500.6, 1e-14, "mixed density matches Python linear blend");
  checkClose(mu[2], 0.000759, 1e-14, "mixed dynamic viscosity matches Python linear blend");
  checkClose(nu[2], mu[2] / rho[2], 1e-14, "mixed kinematic viscosity matches Python dynamic/rho");
  checkClose(pair.permittivityRatio(), 40.0, 1e-14, "permittivity ratio");
  checkClose(pair.conductivityRatio(), 40.0, 1e-14, "conductivity ratio");

  checkClose(electricCapillaryNumber(2.0e-10, 3.0e5, 2.0e-6, 0.05), 7.2e-4, 1e-14,
             "electric capillary number");
  checkClose(chargeRelaxationTime(4.0e-10, 2.0e-8), 0.02, 1e-14, "charge relaxation time");
  checkClose(electricReynoldsNumber(4.0e-10, -0.5, 0.01, 2.0e-8), 1.0, 1e-14,
             "electric Reynolds number uses abs velocity");
  checkClose(ohnesorgeNumber(0.001, 1000.0, 0.072, 1.0e-4),
             0.001 / std::sqrt(1000.0 * 0.072 * 1.0e-4), 1e-14, "Ohnesorge number");
  checkClose(flowRateParameter(1.0e-12, 2.0e-5, 0.05, 1000.0),
             1.0e-12 / std::sqrt(0.05 * std::pow(2.0e-5, 5) / 1000.0), 1e-14,
             "flow-rate parameter");

  checkClose(advectiveDt(0.01, -2.0, 0.4), 0.002, 1e-14, "advective dt");
  check(std::isinf(advectiveDt(0.01, 0.0, 0.4)), "zero-speed advective dt is infinity");
  checkClose(diffusiveDt(0.02, 1.0e-4), 2.0, 1e-14, "default diffusive dt");
  check(std::isinf(diffusiveDt(0.02, 0.0)), "zero-diffusivity dt is infinity");
  checkClose(electricRelaxationDt(4.0e-10, 2.0e-8, 0.5), 0.01, 1e-14, "electric relaxation dt");
  checkClose(electricRelaxationDt(pair, alpha, 0.25), 0.005, 1e-14, "phase-pair conservative relaxation dt");
  checkClose(capillaryDt(0.01, 1000.0, 0.05), std::sqrt(1000.0 * 1.0e-6 / 0.05), 1e-14,
             "capillary dt");
  checkClose(combinedExplicitDt({0.1, 0.02, std::numeric_limits<double>::infinity()}), 0.02, 1e-14,
             "combined explicit dt");

  checkThrows([]() { LeakyDielectricMaterial(-1.0, 1.0); }, "negative permittivity rejected");
  checkThrows([]() { mixtureProperty(1.1, 2.0, 1.0); }, "out-of-bounds alpha rejected");
  checkThrows([]() { harmonicFaceProperty(0.0, 1.0); }, "non-positive face property rejected");
  checkThrows([]() { advectiveDt(0.0, 1.0, 0.1); }, "invalid advective dt inputs rejected");
  checkThrows([]() { combinedExplicitDt({}); }, "empty combined dt rejected");

  std::cout << "electrospray scalar model C++ equivalence checks passed\n";
  return 0;
}
