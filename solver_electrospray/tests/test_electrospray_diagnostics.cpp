#include "TestUtil.hpp"
#include "electrospray/Diagnostics.hpp"

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

  checkClose(laplacePressureJump(0.072, 20.0), 1.44, 1e-14, "Laplace pressure jump");
  const Vec2 normal{2.0, 0.0};
  const double leftTraction = normalTraction2D(2.0, Vec2{3.0, 4.0}, normal);
  const double rightTraction = normalTraction2D(1.0, Vec2{1.0, 2.0}, normal);
  checkClose(leftTraction, 2.0 * (9.0 - 0.5 * 25.0), 1e-14, "left Maxwell normal traction");
  checkClose(normalTractionJump2D(2.0, Vec2{3.0, 4.0}, 1.0, Vec2{1.0, 2.0}, normal),
             leftTraction - rightTraction, 1e-14, "Maxwell traction jump");
  const Vec2 j = ohmicCurrentDensity(3.0, Vec2{2.0, -4.0});
  checkClose(j.x, 6.0, 1e-14, "Ohmic current x");
  checkClose(j.y, -12.0, 1e-14, "Ohmic current y");
  checkClose(electrostaticEnergyDensity(2.0, Vec2{3.0, 4.0}), 25.0, 1e-14,
             "electrostatic energy density");
  checkClose(totalElectrostaticEnergy({2.0, 4.0}, {Vec2{3.0, 4.0}, Vec2{1.0, 2.0}}, {0.5, 2.0}),
             25.0 * 0.5 + 10.0 * 2.0, 1e-14, "total electrostatic energy");
  checkClose(chargeConservationResidual(1.0, 1.5, -2.0, 0.25), 0.0, 1e-14,
             "charge conservation residual");
  checkClose(normalComponent(Vec2{2.0, 2.0}, Vec2{2.0, 0.0}), 2.0, 1e-14, "normal component");
  checkClose(tangentialComponent(Vec2{2.0, 2.0}, Vec2{2.0, 0.0}), 2.0, 1e-14,
             "tangential component");
  checkClose(surfaceChargeDensity(2.0, Vec2{3.0, 4.0}, 5.0, Vec2{1.0, 2.0}, Vec2{2.0, 0.0}),
             5.0 * 1.0 - 2.0 * 3.0, 1e-14, "surface charge density");
  checkClose(normalOhmicCurrentJump(2.0, Vec2{3.0, 4.0}, 5.0, Vec2{1.0, 2.0}, Vec2{2.0, 0.0}),
             5.0 * 1.0 - 2.0 * 3.0, 1e-14, "normal Ohmic current jump");
  checkClose(tangentialFieldJump(Vec2{3.0, 4.0}, Vec2{1.0, 2.0}, Vec2{2.0, 0.0}),
             2.0 - 4.0, 1e-14, "tangential field jump");
  checkClose(electricShearTractionJump(2.0, Vec2{3.0, 4.0}, 5.0, Vec2{1.0, 2.0}, Vec2{2.0, 0.0}),
             5.0 * 1.0 * 2.0 - 2.0 * 3.0 * 4.0, 1e-14, "electric shear traction jump");
  checkClose(staticNormalStressResidual(5.0, 1.0, 0.5, 2.0, 2.0, Vec2{3.0, 4.0}, 1.0, Vec2{1.0, 2.0}, normal),
             4.0 - 1.0 - (leftTraction - rightTraction), 1e-14, "static force balance residual");
  checkClose(maxAbsResidual({-2.0, 1.0, 3.5}), 3.5, 1e-14, "max absolute residual");

  const std::vector<double> numerical{1.0, 2.0, 4.0};
  const std::vector<double> exact{1.0, 1.0, 1.0};
  checkClose(l2Error(numerical, exact), std::sqrt((0.0 + 1.0 + 9.0) / 3.0), 1e-14, "unweighted L2");
  checkClose(l2Error(numerical, exact, std::vector<double>{1.0, 2.0, 3.0}),
             std::sqrt((0.0 + 2.0 + 27.0) / 6.0), 1e-14, "weighted L2");
  checkClose(linfError(numerical, exact), 3.0, 1e-14, "Linf");
  checkClose(convergenceRate(0.25, 0.0625), 2.0, 1e-14, "convergence rate");
  check(passesThreshold(0.1, 0.1), "inclusive threshold");
  check(!passesThreshold(0.1, 0.1, false), "strict threshold");

  const std::vector<double> x{0.0, 0.25, 0.5};
  const auto phi1 = sinusoidalPotential1D(x, 2.0, 1.0);
  checkClose(phi1[0], 0.0, 1e-14, "sinusoidal potential left boundary");
  checkClose(phi1[2], 2.0, 1e-14, "sinusoidal potential midpoint");
  const auto rho1 = sinusoidalCharge1D(x, 3.0, 2.0, 1.0);
  checkClose(rho1[2], 3.0 * M_PI * M_PI * 2.0, 1e-14, "sinusoidal charge source");

  const std::vector<double> y{0.0, 0.25, 0.5};
  const auto phi2 = separablePotential2D(x, y, 1.5, 1.0, 2.0);
  checkClose(phi2[1], 1.5 * std::sin(M_PI * 0.25) * std::cos(M_PI * 0.25 / 2.0), 1e-14,
             "separable 2D potential");
  const auto rho2 = separableCharge2D(x, y, 4.0, 1.5, 1.0, 2.0);
  checkClose(rho2[1], 4.0 * (M_PI * M_PI + M_PI * M_PI / 4.0) * phi2[1], 1e-14,
             "separable 2D charge source");

  checkThrows([]() { laplacePressureJump(-1.0, 1.0); }, "negative surface tension rejected");
  checkThrows([]() { ohmicCurrentDensity(-1.0, Vec2{1.0, 0.0}); }, "negative conductivity rejected");
  checkThrows([]() { electrostaticEnergyDensity(0.0, Vec2{1.0, 0.0}); }, "non-positive permittivity rejected");
  checkThrows([]() { totalElectrostaticEnergy({1.0}, {Vec2{1.0, 0.0}}, {0.0}); },
              "non-positive cell volume rejected");
  checkThrows([]() { chargeConservationResidual(1.0, 1.0, 0.0, 0.0); }, "non-positive dt rejected");
  checkThrows([]() { unitNormal(Vec2{0.0, 0.0}); }, "zero normal rejected");
  checkThrows([]() { maxAbsResidual({}); }, "empty residual rejected");
  checkThrows([]() { l2Error({1.0}, {1.0}, {0.0}); }, "non-positive weight rejected");
  checkThrows([]() { convergenceRate(1.0, 0.5, 1.0); }, "invalid refinement rejected");
  checkThrows([]() { sinusoidalPotential1D({0.0}, 1.0, 0.0); }, "invalid manufactured length rejected");

  std::cout << "electrospray diagnostics/manufactured C++ checks passed\n";
  return 0;
}
