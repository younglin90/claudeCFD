#include "fvm/MeshQuality3D.hpp"
#include "fvm/SurfaceTension3D.hpp"
#include "fvm/VofTransport3D.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct ProfileRow {
  std::string name;
  int cells = 0;
  int faces = 0;
  int iterations = 0;
  double seconds = 0.0;
  double metric = 0.0;
};

template <class Fn>
double timeSeconds(Fn&& fn) {
  const auto t0 = Clock::now();
  fn();
  const auto t1 = Clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

std::filesystem::path outputPath(int argc, char** argv) {
  std::filesystem::path out = "benchmark_logs/electrospray_profile.csv";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--output") {
      if (i + 1 >= argc) throw std::runtime_error("--output requires a path");
      out = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: electrospray_profile [--output path]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return out;
}

fvm::ScalarField sphereAlpha(const fvm::Mesh3D& mesh, double radius, double width) {
  fvm::ScalarField alpha(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const double r = (mesh.cells[c].centroid - fvm::Vec3{0.5, 0.5, 0.5}).norm();
    alpha[c] = std::clamp(0.5 * (1.0 - std::tanh((r - radius) / width)), 0.0, 1.0);
  }
  return alpha;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path out = outputPath(argc, argv);
    std::vector<ProfileRow> rows;
    fvm::Mesh3D mesh;
    rows.push_back({"mesh_build_quality", 0, 0, 1, timeSeconds([&] {
                      mesh = fvm::Mesh3D::hexGrid(14, 12, 10, 1.0, 1.0, 1.0, 0.08);
                      volatile auto q = fvm::meshQualityReport3D(mesh);
                      (void)q;
                    }), 0.0});
    rows.back().cells = static_cast<int>(mesh.cells.size());
    rows.back().faces = static_cast<int>(mesh.faces.size());
    rows.back().metric = fvm::meshQualityReport3D(mesh).maxNonOrthogonalityDeg;

    auto alpha = sphereAlpha(mesh, 0.24, 0.03);
    auto faceFlux = fvm::divergenceFreeBoxFlux3D(mesh, 0.04);
    fvm::VofTransportOptions3D opt;
    opt.scheme = fvm::VofAdvectionScheme3D::IsoAdvector;
    opt.correctionSweeps = 5;
    fvm::VofTransportReport3D vofReport;
    rows.push_back({"vof_isoadvector_10_steps", static_cast<int>(mesh.cells.size()),
                    static_cast<int>(mesh.faces.size()), 10, timeSeconds([&] {
                      for (int i = 0; i < 10; ++i) vofReport = fvm::advectVof3D(mesh, alpha, faceFlux, 0.004, opt);
                    }), vofReport.relativeMassDrift});

    double maxKappa = 0.0;
    rows.push_back({"curvature_balanced_csf", static_cast<int>(mesh.cells.size()),
                    static_cast<int>(mesh.faces.size()), 1, timeSeconds([&] {
                      auto state = fvm::buildBalancedForceSurfaceTensionState3D(mesh, alpha, 0.072);
                      for (double k : state.kappa) maxKappa = std::max(maxKappa, std::abs(k));
                    }), maxKappa});

    std::filesystem::create_directories(out.parent_path());
    std::ofstream csv(out);
    if (!csv) throw std::runtime_error("failed to write profile CSV");
    csv << "name,cells,faces,iterations,seconds,metric\n";
    for (const auto& r : rows) {
      csv << r.name << "," << r.cells << "," << r.faces << "," << r.iterations
          << "," << r.seconds << "," << r.metric << "\n";
    }
    std::cout << "electrospray_profile_rows=" << rows.size()
              << " output=" << out
              << " vof_mass_drift=" << vofReport.relativeMassDrift
              << " max_kappa=" << maxKappa << "\n";
    return rows.size() >= 3 && vofReport.relativeMassDrift <= 1e-3 && maxKappa > 0.0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "electrospray_profile_error=" << e.what() << "\n";
    return 2;
  }
}
