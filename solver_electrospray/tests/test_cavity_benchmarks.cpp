#include "TestUtil.hpp"

int main() {
  double e100 = fvm::runCavityBenchmark(100);
  double e1000 = fvm::runCavityBenchmark(1000);
  std::cout << "cavity_re100_l2=" << e100 << "\n";
  std::cout << "cavity_re1000_l2=" << e1000 << "\n";
  check(e100 < 0.02, "Ghia cavity Re=100 centerline L2");
  check(e1000 < 0.02, "Ghia cavity Re=1000 centerline L2");
}
