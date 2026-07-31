#pragma once

#include "fvm/IncompressibleSolver.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace fvm {

inline void ensureLogDir() { std::filesystem::create_directories("benchmark_logs"); }

inline double taylorGreenEnergy(double nu, double t) {
  return 0.25 * std::exp(-16.0 * M_PI * M_PI * nu * t);
}

inline double runTaylorGreen(double nu, double tEnd) {
  ensureLogDir();
  std::ofstream csv("benchmark_logs/taylor_green.csv");
  csv << "t,computed_energy,analytic_energy,relative_error\n";
  Mesh mesh = Mesh::quadGrid(32, 32, 1.0, 1.0, 0.0);
  double worst = 0.0;
  for (int k = 1; k <= 10; ++k) {
    double t = tEnd * k / 10.0;
    IncompressibleResult sol = solveTaylorGreenPeriodic(32, nu, t, 0.01);
    double exact = taylorGreenEnergy(nu, t);
    double computed = kineticEnergy(mesh, sol.u);
    double err = std::abs(computed - exact) / exact;
    worst = std::max(worst, err);
    csv << t << "," << computed << "," << exact << "," << err << "\n";
  }
  return worst;
}

struct CavityPoint { double x; double y; double u100; double v100; double u1000; double v1000; };

inline const std::vector<CavityPoint>& ghiaData() {
  static const std::vector<CavityPoint> data = {
    {0.5,1.0000,1.00000,0.00000,1.00000,0.00000}, {0.5,0.9766,0.84123,0.00000,0.65928,0.00000},
    {0.5,0.9688,0.78871,0.00000,0.57492,0.00000}, {0.5,0.9609,0.73722,0.00000,0.51117,0.00000},
    {0.5,0.9531,0.68717,0.00000,0.46604,0.00000}, {0.5,0.8516,0.23151,0.00000,0.33304,0.00000},
    {0.5,0.7344,0.00332,0.00000,0.18719,0.00000}, {0.5,0.6172,-0.13641,0.00000,0.05702,0.00000},
    {0.5,0.5000,-0.20581,0.00000,-0.06080,0.00000}, {0.5,0.4531,-0.21090,0.00000,-0.10648,0.00000},
    {0.5,0.2813,-0.15662,0.00000,-0.27805,0.00000}, {0.5,0.1719,-0.10150,0.00000,-0.38289,0.00000},
    {0.5,0.1016,-0.06434,0.00000,-0.29730,0.00000}, {0.5,0.0703,-0.04775,0.00000,-0.22220,0.00000},
    {0.5,0.0625,-0.04192,0.00000,-0.20196,0.00000}, {0.5,0.0547,-0.03717,0.00000,-0.18109,0.00000},
    {0.5,0.0000,0.00000,0.00000,0.00000,0.00000},
    {1.0000,0.5,0.00000,0.00000,0.00000,0.00000}, {0.9688,0.5,0.00000,-0.05906,0.00000,-0.21388},
    {0.9609,0.5,0.00000,-0.07391,0.00000,-0.27669}, {0.9531,0.5,0.00000,-0.08864,0.00000,-0.33714},
    {0.9453,0.5,0.00000,-0.10313,0.00000,-0.39188}, {0.9063,0.5,0.00000,-0.16914,0.00000,-0.51550},
    {0.8594,0.5,0.00000,-0.22445,0.00000,-0.42665}, {0.8047,0.5,0.00000,-0.24533,0.00000,-0.31966},
    {0.5000,0.5,0.00000,0.05454,0.00000,0.02526}, {0.2344,0.5,0.00000,0.17527,0.00000,0.32235},
    {0.2266,0.5,0.00000,0.17507,0.00000,0.33075}, {0.1563,0.5,0.00000,0.16077,0.00000,0.37095},
    {0.0938,0.5,0.00000,0.12317,0.00000,0.32627}, {0.0781,0.5,0.00000,0.10890,0.00000,0.30353},
    {0.0703,0.5,0.00000,0.10091,0.00000,0.29012}, {0.0625,0.5,0.00000,0.09233,0.00000,0.27485},
    {0.0000,0.5,0.00000,0.00000,0.00000,0.00000}
  };
  return data;
}

inline double runCavityBenchmark(int Re) {
  ensureLogDir();
  std::ofstream csv("benchmark_logs/cavity_re" + std::to_string(Re) + ".csv");
  csv << "coord,component,computed,ghia,error\n";
  const int n = (Re == 100) ? 65 : 129;
  const int iterations = (Re == 100) ? 12000 : 30000;
  CavitySampledSolution sol = solveCavityVorticityStream(n, Re, iterations);
  double n2 = 0.0, e2 = 0.0;
  const auto& ghia = ghiaData();
  for (size_t row = 0; row < ghia.size(); ++row) {
    const auto& p = ghia[row];
    if (row < 17) {
      double ref = (Re == 100) ? p.u100 : p.u1000;
      double got = sampleCavityGrid(sol, 0.5, p.y, true);
      csv << p.y << ",u," << got << "," << ref << "," << got - ref << "\n";
      n2 += ref * ref; e2 += sqr(got - ref);
    }
    if (row >= 17) {
      double ref = (Re == 100) ? p.v100 : p.v1000;
      double got = sampleCavityGrid(sol, p.x, 0.5, false);
      csv << p.x << ",v," << got << "," << ref << "," << got - ref << "\n";
      n2 += ref * ref; e2 += sqr(got - ref);
    }
  }
  return std::sqrt(e2 / std::max(n2, 1e-30));
}

inline void appendLedger(const std::string& change, double cav, double tg, double mms,
                         double div,
                         const std::string& nextAction = "complete final audit and preserve regression guards") {
  ensureLogDir();
  std::ofstream led("benchmark_logs/ledger.csv", std::ios::app);
  if (std::filesystem::file_size("benchmark_logs/ledger.csv") == 0) {
    led << "change,cavity_l2,taylor_green_decay_error,mms_slope,max_continuity_residual,next_action\n";
  }
  led << std::quoted(change) << "," << cav << "," << tg << "," << mms << "," << div
      << "," << std::quoted(nextAction) << "\n";
}

}
