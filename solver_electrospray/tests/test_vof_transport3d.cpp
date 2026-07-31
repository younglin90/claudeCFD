#include "TestUtil.hpp"
#include "fvm/VofTransport3D.hpp"
#include <filesystem>
#include <fstream>

static fvm::ScalarField sphereAlpha(const fvm::Mesh3D& mesh, const fvm::Vec3& center, double radius) {
  fvm::ScalarField a(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    a[c] = (mesh.cells[c].centroid - center).norm() <= radius ? 1.0 : 0.0;
  }
  return a;
}

static double mixedness(const fvm::Mesh3D& mesh, const fvm::ScalarField& alpha) {
  double mix = 0.0, vol = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    mix += alpha[c] * (1.0 - alpha[c]) * mesh.cells[c].V;
    vol += mesh.cells[c].V;
  }
  return mix / std::max(vol, 1e-30);
}

int main() {
  auto mesh = fvm::Mesh3D::hexGrid(14, 14, 10, 1.0, 1.0, 1.0, 0.04);
  auto faceFlux = fvm::divergenceFreeBoxFlux3D(mesh, 0.08);

  fvm::ScalarField constant(mesh.cells.size(), 0.37);
  double constantMass0 = fvm::vofMass3D(mesh, constant);
  fvm::ScalarField zeroFlux(mesh.faces.size(), 0.0);
  auto constantReport = fvm::advectVof3D(mesh, constant, zeroFlux, 0.02);
  double constantLinf = 0.0;
  for (double a : constant) constantLinf = std::max(constantLinf, std::abs(a - 0.37));

  fvm::ScalarField alpha = sphereAlpha(mesh, {0.5, 0.5, 0.5}, 0.22);
  double initialMass = fvm::vofMass3D(mesh, alpha);
  fvm::VofTransportOptions3D opt;
  opt.scheme = fvm::VofAdvectionScheme3D::AlgebraicTVD;
  opt.tvdBlend = 1.0;
  opt.compression = 0.0;
  opt.correctionSweeps = 4;
  fvm::VofTransportReport3D report;
  for (int step = 0; step < 120; ++step) {
    report = fvm::advectVof3D(mesh, alpha, faceFlux, 0.01, opt);
  }
  double finalMass = fvm::vofMass3D(mesh, alpha);
  auto [amin, amax] = fvm::vofBounds3D(alpha);
  double drift = std::abs(finalMass - initialMass) / std::max(std::abs(initialMass), 1e-30);

  auto compressionMesh = fvm::Mesh3D::hexGrid(16, 16, 12, 1.0, 1.0, 1.0, 0.03);
  auto compressionFlux = fvm::divergenceFreeBoxFlux3D(compressionMesh, 0.08);
  fvm::ScalarField noCompression = sphereAlpha(compressionMesh, {0.5, 0.5, 0.5}, 0.22);
  fvm::ScalarField withCompression = noCompression;
  double compressionMass0 = fvm::vofMass3D(compressionMesh, withCompression);
  fvm::VofTransportOptions3D noCompressionOpt;
  noCompressionOpt.scheme = fvm::VofAdvectionScheme3D::AlgebraicTVD;
  noCompressionOpt.tvdBlend = 1.0;
  noCompressionOpt.compression = 0.0;
  noCompressionOpt.correctionSweeps = 4;
  fvm::VofTransportOptions3D compressionOpt = noCompressionOpt;
  compressionOpt.compression = 0.05;
  fvm::VofTransportReport3D compressionReport;
  for (int step = 0; step < 80; ++step) {
    fvm::advectVof3D(compressionMesh, noCompression, compressionFlux, 0.01, noCompressionOpt);
    compressionReport = fvm::advectVof3D(compressionMesh, withCompression, compressionFlux, 0.01, compressionOpt);
  }
  auto [cmin, cmax] = fvm::vofBounds3D(withCompression);
  double noCompressionMix = mixedness(compressionMesh, noCompression);
  double compressionMix = mixedness(compressionMesh, withCompression);
  double compressionDrift = std::abs(fvm::vofMass3D(compressionMesh, withCompression) - compressionMass0) /
                            std::max(std::abs(compressionMass0), 1e-30);

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/vof_transport3d.csv");
  csv << "case,cells,steps,dt,initial_mass,final_mass,relative_mass_drift,min_alpha,max_alpha,constant_linf\n";
  csv << "constant," << mesh.cells.size() << ",1,0.02," << constantMass0 << ","
      << constantReport.finalMass << "," << constantReport.relativeMassDrift << ","
      << constantReport.minAlpha << "," << constantReport.maxAlpha << "," << constantLinf << "\n";
  csv << "sphere_swirl," << mesh.cells.size() << ",120,0.01," << initialMass << ","
      << finalMass << "," << drift << "," << amin << "," << amax << ",nan\n";
  csv << "compression_sphere," << compressionMesh.cells.size() << ",80,0.01," << compressionMass0 << ","
      << fvm::vofMass3D(compressionMesh, withCompression) << "," << compressionDrift << ","
      << cmin << "," << cmax << "," << compressionMix << "\n";

  std::ofstream compCsv("benchmark_logs/vof_compression3d.csv");
  compCsv << "case,cells,steps,dt,compression,mixedness,mass_drift,min_alpha,max_alpha\n";
  compCsv << "no_compression," << compressionMesh.cells.size() << ",80,0.01,0,"
          << noCompressionMix << ",nan,nan,nan\n";
  compCsv << "compression_0p05," << compressionMesh.cells.size() << ",80,0.01,0.05,"
          << compressionMix << "," << compressionDrift << "," << cmin << "," << cmax << "\n";

  check(constantLinf < 1e-12, "3D VoF constant field remains invariant");
  check(constantReport.relativeMassDrift <= 1e-12, "3D VoF constant mass conserved");
  check(amin >= -1e-14, "3D VoF alpha lower bound enforced");
  check(amax <= 1.0 + 1e-14, "3D VoF alpha upper bound enforced");
  check(drift <= 1e-3, "3D VoF relative mass drift within Leg 2 target");
  check(compressionDrift <= 1e-3, "3D VoF compression mass drift within Leg 2 target");
  check(cmin >= -1e-14 && cmax <= 1.0 + 1e-14, "3D VoF compression bounds enforced");
  check(compressionMix < 0.25 * noCompressionMix, "3D VoF compression sharpens the interface");

  std::cout << "vof3d_mass_drift=" << drift
            << " vof3d_min_alpha=" << amin
            << " vof3d_max_alpha=" << amax
            << " vof3d_compression_mixedness=" << compressionMix
            << " vof3d_no_compression_mixedness=" << noCompressionMix
            << " vof3d_constant_linf=" << constantLinf << "\n";
}
