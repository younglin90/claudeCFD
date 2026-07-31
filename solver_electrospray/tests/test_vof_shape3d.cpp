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

static fvm::ScalarField slottedSphereAlpha(const fvm::Mesh3D& mesh) {
  fvm::ScalarField a = sphereAlpha(mesh, {0.5, 0.5, 0.5}, 0.25);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    if (x.z() > 0.5 && std::abs(x.x() - 0.5) < 0.055 && x.y() < 0.66) a[c] = 0.0;
  }
  return a;
}

static double volumeL1(const fvm::Mesh3D& mesh, const fvm::ScalarField& a, const fvm::ScalarField& b) {
  double e = 0.0, vol = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    e += std::abs(a[c] - b[c]) * mesh.cells[c].V;
    vol += mesh.cells[c].V;
  }
  return e / std::max(vol, 1e-30);
}

static fvm::VofTransportReport3D runForwardReverse(const fvm::Mesh3D& mesh,
                                                   fvm::ScalarField& alpha,
                                                   const fvm::ScalarField& flux,
                                                   int steps,
                                                   double dt) {
  fvm::VofTransportOptions3D opt;
  opt.tvdBlend = 1.0;
  opt.compression = 0.0;
  opt.correctionSweeps = 4;
  fvm::VofTransportReport3D report;
  for (int step = 0; step < steps; ++step) report = fvm::advectVof3D(mesh, alpha, flux, dt, opt);
  fvm::ScalarField reverseFlux = flux;
  for (double& f : reverseFlux) f = -f;
  for (int step = 0; step < steps; ++step) report = fvm::advectVof3D(mesh, alpha, reverseFlux, dt, opt);
  return report;
}

int main() {
  auto mesh = fvm::Mesh3D::hexGrid(18, 18, 18);
  auto flux = fvm::divergenceFreeBoxFlux3D(mesh, 0.08);
  constexpr int steps = 40;
  constexpr double dt = 0.01;

  struct Case {
    const char* label;
    fvm::ScalarField initial;
    double shapeTarget;
  };
  std::vector<Case> cases = {{"rider_kothe_sphere", sphereAlpha(mesh, {0.5, 0.5, 0.5}, 0.23), 0.02},
                             {"zalesak_slotted_sphere", slottedSphereAlpha(mesh), 0.02}};

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/vof_shape3d.csv");
  csv << "case,cells,forward_steps,reverse_steps,dt,initial_mass,final_mass,"
         "relative_mass_drift,min_alpha,max_alpha,shape_l1,target_shape_l1\n";

  for (const auto& c : cases) {
    fvm::ScalarField alpha = c.initial;
    double m0 = fvm::vofMass3D(mesh, alpha);
    auto report = runForwardReverse(mesh, alpha, flux, steps, dt);
    double mf = fvm::vofMass3D(mesh, alpha);
    auto [amin, amax] = fvm::vofBounds3D(alpha);
    double drift = std::abs(mf - m0) / std::max(std::abs(m0), 1e-30);
    double shape = volumeL1(mesh, alpha, c.initial);
    csv << c.label << "," << mesh.cells.size() << "," << steps << "," << steps << "," << dt
        << "," << m0 << "," << mf << "," << drift << "," << amin << "," << amax
        << "," << shape << "," << c.shapeTarget << "\n";

    check(drift <= 1e-3, std::string(c.label) + " 3D VoF mass drift within target");
    check(amin >= -1e-14 && amax <= 1.0 + 1e-14, std::string(c.label) + " 3D VoF bounds enforced");
    check(shape <= c.shapeTarget, std::string(c.label) + " 3D VoF shape error within target");
    (void)report;
  }

  std::cout << "vof_shape3d_cases=2 target_shape_l1=0.02\n";
}
