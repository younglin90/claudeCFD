#include "TestUtil.hpp"
#include <fstream>

int main() {
  constexpr int n = 40;
  constexpr double stretch = 1.8;
  auto mesh = fvm::Mesh::stretchedQuadGrid(n, n, stretch);
  std::vector<fvm::CavityStage> stages = {
    {100, 4000, 0.0010},
    {400, 4000, 0.0007},
    {1000, 24000, 0.0010},
  };
  auto sol = fvm::solveCavityProjectionContinuation(n, stages, 1.0, true, false, -1.0,
                                                    stretch);
  fvm::VelocityBC bc{[](const fvm::Vec& x) { return fvm::cavityVelocityBC(x); }};
  double n2 = 0.0;
  double e2 = 0.0;
  const auto& ghia = fvm::ghiaData();
  fvm::ensureLogDir();
  std::ofstream csv("benchmark_logs/collocated_re1000_diagnostic.csv");
  csv << "# mesh=stretched_quad n=40 stretch=1.8 continuation_stages=Re100:4000@0.001,Re400:4000@0.0007,Re1000:24000@0.001\n";
  csv << "coord,component,computed,ghia,error\n";
  for (size_t row = 0; row < ghia.size(); ++row) {
    const auto& p = ghia[row];
    if (row < 17) {
      double got = fvm::interpolateStructuredCellComponent(mesh, sol.u, {0.5, p.y}, true, &bc);
      csv << p.y << ",u," << got << "," << p.u1000 << "," << got - p.u1000 << "\n";
      n2 += p.u1000 * p.u1000;
      e2 += fvm::sqr(got - p.u1000);
    } else {
      double got = fvm::interpolateStructuredCellComponent(mesh, sol.u, {p.x, 0.5}, false, &bc);
      csv << p.x << ",v," << got << "," << p.v1000 << "," << got - p.v1000 << "\n";
      n2 += p.v1000 * p.v1000;
      e2 += fvm::sqr(got - p.v1000);
    }
  }
  double err = std::sqrt(e2 / std::max(n2, 1e-30));
  check(std::isfinite(err), "collocated Re=1000 diagnostic error finite");
  check(err < 0.02, "collocated Re=1000 production path matches Ghia within 2% L2");
  check(sol.maxDiv < 1e-8, "collocated Re=1000 diagnostic continuity bounded");
  std::cout << "collocated_cavity_re1000_diagnostic_l2=" << err
            << " max_continuity_residual=" << sol.maxDiv << "\n";
}
