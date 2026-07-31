#include "TestUtil.hpp"
#include "fvm/IncompressibleSolver3D.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

struct CourantSweepRefPoint {
  std::string axis;
  double coord = 0.0;
  std::string component;
  double value = 0.0;
};

struct CourantSweepError {
  double relativeAll = 0.0;
  double relativeInterior = 0.0;
  double rmsAll = 0.0;
  double rmsInterior = 0.0;
};

static std::vector<CourantSweepRefPoint> readCourantSweepReference() {
  std::string path = std::string(FVM_SOURCE_DIR) +
                     "/reference/3d_cavity/albensoeder_kuhlmann_fig20_digitized.csv";
  std::ifstream in(path);
  check(in.good(), "3D cavity Courant sweep reference CSV is readable");
  std::vector<CourantSweepRefPoint> refs;
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

static CourantSweepError referenceError(const fvm::Mesh3D& mesh,
                                        const fvm::VectorField3& u,
                                        const std::vector<CourantSweepRefPoint>& refs,
                                        const fvm::Cavity3DLid& lid) {
  fvm::VelocityBC3D bc = fvm::makeCavityVelocityBC3D(lid);
  double e2 = 0.0, n2 = 0.0, ie2 = 0.0, in2 = 0.0;
  int count = 0, interiorCount = 0;
  for (const auto& r : refs) {
    fvm::Vec3 v = fvm::Vec3::Zero();
    if (r.axis == "y_center") {
      v = fvm::interpolateStructuredCellVector3D(mesh, u, {0.5, r.coord, 0.5}, &bc);
    } else if (r.axis == "x_center") {
      v = fvm::interpolateStructuredCellVector3D(mesh, u, {r.coord, 0.5, 0.5}, &bc);
    } else {
      continue;
    }
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
          std::sqrt(ie2 / std::max(interiorCount, 1))};
}

int main() {
  auto refs = readCourantSweepReference();
  fvm::Cavity3DLid referenceLid{0, 0.0, 1, 1.0};

  struct Case {
    const char* label;
    int nx;
    int ny;
    int nz;
    int warmup1Steps;
    int warmup2Steps;
    int finalSteps;
    double maxCourant;
  };
  const std::vector<Case> cases = {
      {"unconstrained_16x16x10", 16, 16, 10, 20, 20, 30, std::numeric_limits<double>::infinity()},
      {"courant_1_16x16x10", 16, 16, 10, 20, 20, 30, 1.0},
      {"courant_1_24x24x16_representative", 24, 24, 16, 40, 40, 240, 1.0}};

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/cavity3d_courant_reference_sweep.csv");
  csv << "label,nx,ny,nz,total_steps,nominal_final_dt,max_courant_target,max_courant,"
         "relative_all_l2,relative_interior_l2,rms_all,rms_interior,max_div\n";

  double representativeMax = 0.0;
  double representativeRms = 0.0;
  for (const auto& c : cases) {
    fvm::Cavity3DCase caseMeshCfg;
    caseMeshCfg.nx = c.nx;
    caseMeshCfg.ny = c.ny;
    caseMeshCfg.nz = c.nz;
    caseMeshCfg.cosineMesh = true;
    auto caseMesh = fvm::makeCavityMesh3D(caseMeshCfg);
    std::vector<fvm::Cavity3DStage> stages = {{1000, c.warmup1Steps, 0.02, c.maxCourant},
                                             {1000, c.warmup2Steps, 0.01, c.maxCourant},
                                             {1000, c.finalSteps, 0.08, c.maxCourant}};
    auto sol = fvm::solveCavityProjection3DContinuation(8, stages, 1.0, true, -1.0, 2,
                                                        referenceLid, c.nx, c.ny, c.nz, true);
    auto err = referenceError(caseMesh, sol.u, refs, referenceLid);
    csv << c.label << "," << caseMesh.nx << "," << caseMesh.ny << "," << caseMesh.nz
        << "," << sol.steps << ",0.08," << c.maxCourant << "," << sol.maxCourant
        << "," << err.relativeAll << "," << err.relativeInterior
        << "," << err.rmsAll << "," << err.rmsInterior << "," << sol.maxDiv << "\n";
    check(std::isfinite(err.rmsAll), "3D cavity Courant reference sweep RMS finite");
    check(sol.maxDiv <= 1e-10, "3D cavity Courant reference sweep continuity at tolerance");
    if (std::string(c.label) == "courant_1_24x24x16_representative") {
      representativeMax = sol.maxCourant;
      representativeRms = err.rmsAll;
    }
  }
  check(representativeMax <= 1.0, "3D cavity Courant reference sweep representative respects Co<=1");
  check(representativeRms <= 0.02, "3D cavity Courant reference sweep representative reaches 2% RMS");
  std::cout << "cavity3d_courant_sweep_representative_max_courant=" << representativeMax
            << " cavity3d_courant_sweep_representative_rms=" << representativeRms << "\n";
}
