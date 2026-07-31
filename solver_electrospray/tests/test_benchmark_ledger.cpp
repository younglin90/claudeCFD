#include "TestUtil.hpp"
#include <fstream>
#include <sstream>

static double mmsSlopeForLedger() {
  auto err = [](int n) {
    auto mesh = fvm::Mesh::quadGrid(n, n, 1.0, 1.0, 0.22);
    fvm::ScalarField phi(mesh.cells.size()), exact(mesh.cells.size());
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      double x = mesh.cells[c].centroid.x();
      double y = mesh.cells[c].centroid.y();
      phi[c] = std::sin(M_PI * x) * std::sin(M_PI * y);
      exact[c] = -2.0 * M_PI * M_PI * phi[c];
    }
    auto lap = fvm::laplacianExplicit(mesh, phi);
    double e = 0.0, v = 0.0;
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      const auto& cc = mesh.cells[c].centroid;
      if (cc.x() < 1.0 / n || cc.x() > 1.0 - 1.0 / n ||
          cc.y() < 1.0 / n || cc.y() > 1.0 - 1.0 / n) continue;
      e += fvm::sqr(lap[c] - exact[c]) * mesh.cells[c].V;
      v += mesh.cells[c].V;
    }
    return std::sqrt(e / std::max(v, 1e-30));
  };
  return std::log(err(16) / err(64)) / std::log(4.0);
}

static double continuityForLedger() {
  auto mesh = fvm::Mesh::quadGrid(24, 24, 1.0, 1.0, 0.08);
  fvm::VectorField u(mesh.cells.size(), fvm::Vec::Zero());
  fvm::ScalarField p(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double x = mesh.cells[c].centroid.x();
    double y = mesh.cells[c].centroid.y();
    u[c] = {0.08 * std::sin(M_PI * x) * std::cos(M_PI * y),
            -0.05 * std::cos(M_PI * x) * std::sin(M_PI * y)};
    p[c] = ((static_cast<int>(c) / mesh.nx + static_cast<int>(c) % mesh.nx) % 2 == 0) ? 1.0 : -1.0;
  }
  return fvm::projectVelocityRhieChow(mesh, u, p, 0.01).maxDiv;
}

static double cavityErrorFromCsv(const std::string& path) {
  std::ifstream in(path);
  if (!in) return 1e300;
  std::string line;
  std::getline(in, line);
  double n2 = 0.0;
  double e2 = 0.0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#' || line.rfind("coord,", 0) == 0) continue;
    std::stringstream ss(line);
    std::string coord, component, computed, ref, error;
    std::getline(ss, coord, ',');
    std::getline(ss, component, ',');
    std::getline(ss, computed, ',');
    std::getline(ss, ref, ',');
    std::getline(ss, error, ',');
    double r = std::stod(ref);
    double e = std::stod(error);
    n2 += fvm::sqr(r);
    e2 += fvm::sqr(e);
  }
  return std::sqrt(e2 / std::max(n2, 1e-30));
}

int main() {
  double cav = std::max(cavityErrorFromCsv("benchmark_logs/cavity_re100.csv"),
                        cavityErrorFromCsv("benchmark_logs/cavity_re1000.csv"));
  cav = std::max(cav, cavityErrorFromCsv("benchmark_logs/collocated_re1000_diagnostic.csv"));
  double tg = fvm::runTaylorGreen(0.01, 1.0);
  double mms = mmsSlopeForLedger();
  double div = continuityForLedger();
  fvm::appendLedger("Cached momentum/pressure solves plus stretched collocated Re=1000 cavity gate",
                    cav, tg, mms, div,
                    "complete final audit and preserve regression guards");
  check(cav < 0.02, "ledger cavity guard");
  check(tg < 0.02, "ledger Taylor-Green guard");
  check(mms >= 1.85, "ledger MMS guard");
  check(div < 1e-9, "ledger continuity guard");
  std::cout << "ledger cavity_l2=" << cav << " taylor_green_decay_error=" << tg
            << " mms_slope=" << mms << " max_continuity_residual=" << div << "\n";
}
