#include "TestUtil.hpp"
#include "electrospray/AxisymmetricElectrostatics.hpp"

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

  const int n = 64;
  const double radius = 2.0;
  std::vector<double> faces(static_cast<size_t>(n + 1), 0.0);
  for (int i = 0; i <= n; ++i) faces[static_cast<size_t>(i)] = radius * static_cast<double>(i) / n;
  std::vector<double> eps(static_cast<size_t>(n), 3.0);

  const auto laplace = solveRadialPoissonAxisymmetric(faces, eps, 7.0);
  for (int i = 0; i < n; ++i) {
    checkClose(laplace.phi[static_cast<size_t>(i)], 7.0, 1e-11, "axisymmetric zero-charge constant potential");
    checkClose(laplace.eRadial[static_cast<size_t>(i)], 0.0, 1e-11, "axisymmetric zero-charge radial field");
  }

  const double rho = 12.0;
  const double phiOuter = 0.5;
  std::vector<double> charge(static_cast<size_t>(n), rho);
  const auto charged = solveRadialPoissonAxisymmetric(faces, eps, phiOuter, charge);
  double maxPhiError = 0.0;
  double maxEError = 0.0;
  for (int i = 0; i < n; ++i) {
    const double r = charged.rCenters[static_cast<size_t>(i)];
    const double exactPhi = phiOuter + rho * (radius * radius - r * r) / (4.0 * eps[0]);
    const double exactE = rho * r / (2.0 * eps[0]);
    maxPhiError = std::max(maxPhiError, std::abs(charged.phi[static_cast<size_t>(i)] - exactPhi));
    maxEError = std::max(maxEError, std::abs(charged.eRadial[static_cast<size_t>(i)] - exactE));
  }
  check(maxPhiError < 2.0e-3, "charged radial Poisson potential matches analytic cylinder solution");
  check(maxEError < 4.0e-2, "charged radial Poisson field matches analytic cylinder solution");

  checkThrows([]() { solveRadialPoissonAxisymmetric({0.1, 1.0}, {1.0}, 0.0); },
              "r faces must start at zero");
  checkThrows([]() { solveRadialPoissonAxisymmetric({0.0, 1.0}, {-1.0}, 0.0); },
              "negative epsilon rejected");
  checkThrows([]() { solveRadialPoissonAxisymmetric({0.0, 1.0}, {1.0}, 0.0, {1.0, 2.0}); },
              "charge size mismatch rejected");

  std::cout << "axisymmetric electrostatic C++ checks passed max_phi_error=" << maxPhiError
            << " max_e_error=" << maxEError << "\n";
  return 0;
}
