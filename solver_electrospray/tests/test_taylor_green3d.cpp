#include "TestUtil.hpp"
#include "fvm/TaylorGreen3D.hpp"
#include <filesystem>
#include <fstream>

int main() {
  auto report = fvm::runTaylorGreen3D(12, 0.01, 0.5, 0.025);
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/taylor_green3d.csv");
  csv << "t,computed_energy,analytic_energy,energy_relative_error,"
         "computed_enstrophy,analytic_enstrophy,enstrophy_relative_error\n";
  csv << report.finalTime << "," << report.computedEnergy << "," << report.analyticEnergy
      << "," << report.energyError << "," << report.computedEnstrophy << ","
      << report.analyticEnstrophy << "," << report.enstrophyError << "\n";
  std::cout << "taylor_green3d_energy_error=" << report.energyError
            << " taylor_green3d_enstrophy_error=" << report.enstrophyError << "\n";
  check(report.energyError < 0.05, "3D Taylor-Green kinetic-energy decay within 5%");
  check(report.enstrophyError < 0.05, "3D Taylor-Green enstrophy decay within 5%");
}
