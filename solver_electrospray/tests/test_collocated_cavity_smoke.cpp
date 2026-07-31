#include "TestUtil.hpp"

int main() {
  constexpr int n = 48;
  auto mesh = fvm::Mesh::quadGrid(n, n, 1.0, 1.0, 0.0);
  auto sol = fvm::solveCavityProjection(n, 100, 8000, 0.001, 1.0, true, false, -1.0);
  fvm::VelocityBC bc{[](const fvm::Vec& x) { return fvm::cavityVelocityBC(x); }};
  double n2 = 0.0;
  double e2 = 0.0;
  const auto& ghia = fvm::ghiaData();
  for (size_t row = 0; row < ghia.size(); ++row) {
    const auto& p = ghia[row];
    if (row < 17) {
      double got = fvm::interpolateStructuredCellComponent(mesh, sol.u, {0.5, p.y}, true, &bc);
      n2 += p.u100 * p.u100;
      e2 += fvm::sqr(got - p.u100);
    } else {
      double got = fvm::interpolateStructuredCellComponent(mesh, sol.u, {p.x, 0.5}, false, &bc);
      n2 += p.v100 * p.v100;
      e2 += fvm::sqr(got - p.v100);
    }
  }
  double err = std::sqrt(e2 / std::max(n2, 1e-30));
  check(std::isfinite(err), "collocated cavity smoke error finite");
  check(sol.maxDiv < 1e-8, "collocated cavity Rhie-Chow face continuity at solver tolerance");
  check(err < 0.02, "collocated cavity production path matches Ghia Re=100 within 2% L2");
  std::cout << "collocated_cavity_re100_l2=" << err
            << " max_continuity_residual=" << sol.maxDiv << "\n";
}
