#include "TestUtil.hpp"

int main() {
  double err = fvm::runTaylorGreen(0.01, 1.0);
  std::cout << "taylor_green_decay_error=" << err << "\n";
  check(err < 0.02, "Taylor-Green kinetic-energy decay within 2%");
}
