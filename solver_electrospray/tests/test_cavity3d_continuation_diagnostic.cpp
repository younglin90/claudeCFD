#include "TestUtil.hpp"
#include "fvm/IncompressibleSolver3D.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

struct ContinuationRefPoint {
  std::string axis;
  double coord = 0.0;
  std::string component;
  double value = 0.0;
};

struct ContinuationRefError {
  double relativeAll = 0.0;
  double relativeInterior = 0.0;
  double rmsAll = 0.0;
  double rmsInterior = 0.0;
  int count = 0;
  int interiorCount = 0;
};

static std::vector<ContinuationRefPoint> readContinuationReference() {
  std::string path = std::string(FVM_SOURCE_DIR) +
                     "/reference/3d_cavity/albensoeder_kuhlmann_fig20_digitized.csv";
  std::ifstream in(path);
  check(in.good(), "3D cavity continuation reference CSV is readable");
  std::vector<ContinuationRefPoint> refs;
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string source, axis, coord, component, value;
    std::getline(ss, source, ',');
    std::getline(ss, axis, ',');
    std::getline(ss, coord, ',');
    std::getline(ss, component, ',');
    std::getline(ss, value, ',');
    refs.push_back({axis, std::stod(coord), component, std::stod(value)});
  }
  return refs;
}

static ContinuationRefError referenceError(const fvm::Mesh3D& mesh,
                                          const fvm::VectorField3& u,
                                          const std::vector<ContinuationRefPoint>& refs,
                                          const fvm::Cavity3DLid& lid) {
  fvm::VelocityBC3D bc = fvm::makeCavityVelocityBC3D(lid);
  double e2 = 0.0, n2 = 0.0, ie2 = 0.0, in2 = 0.0;
  int count = 0, interiorCount = 0;
  for (const auto& r : refs) {
    fvm::Vec3 v = fvm::Vec3::Zero();
    if (r.axis == "y_center") v = fvm::interpolateStructuredCellVector3D(mesh, u, {0.5, r.coord, 0.5}, &bc);
    else if (r.axis == "x_center") v = fvm::interpolateStructuredCellVector3D(mesh, u, {r.coord, 0.5, 0.5}, &bc);
    else continue;
    double computed = r.component == "ux" ? v.x() : v.y();
    double err = computed - r.value;
    e2 += fvm::sqr(err);
    n2 += fvm::sqr(r.value);
    ++count;
    if (r.coord > 0.05 && r.coord < 0.95) {
      ie2 += fvm::sqr(err);
      in2 += fvm::sqr(r.value);
      ++interiorCount;
    }
  }
  return {std::sqrt(e2 / std::max(n2, 1e-30)),
          std::sqrt(ie2 / std::max(in2, 1e-30)),
          std::sqrt(e2 / std::max(count, 1)),
          std::sqrt(ie2 / std::max(interiorCount, 1)),
          count,
          interiorCount};
}

int main() {
  constexpr int n = 8;
  fvm::Cavity3DLid referenceLid{0, 0.0, 1, 1.0};
  std::vector<fvm::Cavity3DStage> stages = {{1000, 120, 0.02}, {1000, 180, 0.01}, {1000, 500, 0.08}};
  auto sol = fvm::solveCavityProjection3DContinuation(n, stages, 1.0, true, -1.0, 2,
                                                      referenceLid, 24, 24, 16, true);
  fvm::Cavity3DCase meshCfg;
  meshCfg.n = n;
  meshCfg.nx = 24;
  meshCfg.ny = 24;
  meshCfg.nz = 16;
  meshCfg.cosineMesh = true;
  auto mesh = fvm::makeCavityMesh3D(meshCfg);
  auto refs = readContinuationReference();
  auto err = referenceError(mesh, sol.u, refs, referenceLid);
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/cavity3d_continuation_diagnostic.csv");
  csv << "nx,ny,nz,cosine_mesh,total_steps,stages,pressure_correctors,Re,lid_normal_axis,lid_side,lid_velocity_component,relative_all_l2,relative_interior_l2,rms_all,rms_interior,reference_count,interior_reference_count,max_div,max_courant\n";
  csv << mesh.nx << "," << mesh.ny << "," << mesh.nz << ",1," << sol.steps << ",3,2,1000,"
      << referenceLid.normalAxis << "," << referenceLid.side << "," << referenceLid.velocityComponent
      << "," << err.relativeAll << "," << err.relativeInterior
      << "," << err.rmsAll << "," << err.rmsInterior
      << "," << err.count << "," << err.interiorCount << "," << sol.maxDiv
      << "," << sol.maxCourant << "\n";
  check(std::isfinite(err.relativeAll), "3D cavity continuation all L2 finite");
  check(std::isfinite(err.relativeInterior), "3D cavity continuation interior L2 finite");
  check(std::isfinite(err.rmsAll), "3D cavity continuation all RMS finite");
  check(std::isfinite(err.rmsInterior), "3D cavity continuation interior RMS finite");
  check(std::isfinite(sol.maxCourant), "3D cavity continuation Courant finite");
  check(sol.maxDiv <= 1e-10, "3D cavity continuation continuity at tolerance");
  std::cout << "cavity3d_continuation_all_l2=" << err.relativeAll
            << " cavity3d_continuation_interior_l2=" << err.relativeInterior
            << " cavity3d_continuation_rms_all=" << err.rmsAll
            << " cavity3d_continuation_rms_interior=" << err.rmsInterior
            << " cavity3d_continuation_max_div=" << sol.maxDiv
            << " cavity3d_continuation_max_courant=" << sol.maxCourant
            << " total_steps=" << sol.steps << "\n";
}
