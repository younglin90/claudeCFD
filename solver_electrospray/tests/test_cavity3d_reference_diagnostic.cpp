#include "TestUtil.hpp"
#include "fvm/IncompressibleSolver3D.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>

struct RefPoint {
  std::string axis;
  double coord = 0.0;
  std::string component;
  double value = 0.0;
};

struct MappingCandidate {
  std::string refAxis;
  char lineAxis = 'x';
  int component = 0;
  bool flipCoord = false;
  double allL2 = 0.0;
  double interiorL2 = 0.0;
};

static std::vector<RefPoint> readReference() {
  std::string path = std::string(FVM_SOURCE_DIR) +
                     "/reference/3d_cavity/albensoeder_kuhlmann_fig20_digitized.csv";
  std::ifstream in(path);
  check(in.good(), "3D cavity digitized reference CSV is readable");
  std::vector<RefPoint> refs;
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

int main() {
  fvm::Cavity3DCase cfg;
  cfg.Re = 1000;
  cfg.lid = {0, 0.0, 1, 1.0};
  cfg.nx = 12;
  cfg.ny = 12;
  cfg.nz = 8;
  cfg.cosineMesh = true;
  auto mesh = fvm::makeCavityMesh3D(cfg);
  auto sol = fvm::solveCavityProjection3D(cfg);
  fvm::VelocityBC3D bc = fvm::makeCavityVelocityBC3D(cfg.lid);
  auto refs = readReference();
  check(refs.size() >= 20, "3D cavity digitized reference has enough samples");

  double e2 = 0.0;
  double n2 = 0.0;
  double interiorE2 = 0.0;
  double interiorN2 = 0.0;
  int count = 0;
  int interiorCount = 0;
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/cavity3d_reference_l2.csv");
  csv << "source,nx,ny,nz,cosine_mesh,Re,steps,dt,lid_normal_axis,lid_side,lid_velocity_component,"
         "axis,coord,component,computed,reference,error\n";
  for (const auto& r : refs) {
    fvm::Vec3 v = fvm::Vec3::Zero();
    if (r.axis == "y_center") {
      v = fvm::interpolateStructuredCellVector3D(mesh, sol.u, {0.5, r.coord, 0.5}, &bc);
    } else if (r.axis == "x_center") {
      v = fvm::interpolateStructuredCellVector3D(mesh, sol.u, {r.coord, 0.5, 0.5}, &bc);
    } else {
      continue;
    }
    double computed = r.component == "ux" ? v.x() : v.y();
    double err = computed - r.value;
    e2 += fvm::sqr(err);
    n2 += fvm::sqr(r.value);
    ++count;
    if (r.coord > 0.05 && r.coord < 0.95) {
      interiorE2 += fvm::sqr(err);
      interiorN2 += fvm::sqr(r.value);
      ++interiorCount;
    }
    csv << "TNO_ECN_E_11_042_Fig20_Albensoeder_Kuhlmann,"
        << mesh.nx << "," << mesh.ny << "," << mesh.nz << "," << (cfg.cosineMesh ? 1 : 0) << ","
        << cfg.Re << "," << cfg.steps << "," << cfg.dt << ","
        << cfg.lid.normalAxis << "," << cfg.lid.side << "," << cfg.lid.velocityComponent << ","
        << r.axis << "," << r.coord << "," << r.component << ","
        << computed << "," << r.value << "," << err << "\n";
  }
  double relL2 = std::sqrt(e2 / std::max(n2, 1e-30));
  double interiorRelL2 = std::sqrt(interiorE2 / std::max(interiorN2, 1e-30));
  double rmsAll = std::sqrt(e2 / std::max(count, 1));
  double rmsInterior = std::sqrt(interiorE2 / std::max(interiorCount, 1));
  csv << "summary_all,,,,,,,,,,,,,,," << relL2 << ",relative_l2\n";
  csv << "summary_interior,,,,,,,,,,,,,,," << interiorRelL2 << ",relative_l2\n";
  csv << "summary_all,,,,,,,,,,,,,,," << rmsAll << ",rms\n";
  csv << "summary_interior,,,,,,,,,,,,,,," << rmsInterior << ",rms\n";

  std::vector<MappingCandidate> candidates;
  std::array<std::string, 2> refAxes = {"x_center", "y_center"};
  std::array<char, 3> lineAxes = {'x', 'y', 'z'};
  for (const std::string& refAxis : refAxes) {
    for (char lineAxis : lineAxes) {
      for (int component = 0; component < 3; ++component) {
        for (bool flip : {false, true}) {
          double ce2 = 0.0, cn2 = 0.0, cie2 = 0.0, cin2 = 0.0;
          for (const auto& r : refs) {
            if (r.axis != refAxis) continue;
            double q = flip ? 1.0 - r.coord : r.coord;
            fvm::Vec3 x{0.5, 0.5, 0.5};
            if (lineAxis == 'x') x.x() = q;
            if (lineAxis == 'y') x.y() = q;
            if (lineAxis == 'z') x.z() = q;
            fvm::Vec3 v = fvm::interpolateStructuredCellVector3D(mesh, sol.u, x, &bc);
            double computed = v[component];
            double err = computed - r.value;
            ce2 += fvm::sqr(err);
            cn2 += fvm::sqr(r.value);
            if (r.coord > 0.05 && r.coord < 0.95) {
              cie2 += fvm::sqr(err);
              cin2 += fvm::sqr(r.value);
            }
          }
          candidates.push_back({refAxis, lineAxis, component, flip,
                                std::sqrt(ce2 / std::max(cn2, 1e-30)),
                                std::sqrt(cie2 / std::max(cin2, 1e-30))});
        }
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.interiorL2 < b.interiorL2;
  });
  std::ofstream sweep("benchmark_logs/cavity3d_reference_mapping_sweep.csv");
  sweep << "rank,ref_axis,line_axis,component,flip_coord,all_l2,interior_l2\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    sweep << i + 1 << "," << candidates[i].refAxis << "," << candidates[i].lineAxis
          << "," << candidates[i].component << "," << (candidates[i].flipCoord ? 1 : 0)
          << "," << candidates[i].allL2 << "," << candidates[i].interiorL2 << "\n";
  }

  check(std::isfinite(relL2), "3D cavity reference diagnostic L2 finite");
  check(std::isfinite(interiorRelL2), "3D cavity reference diagnostic interior L2 finite");
  check(std::isfinite(rmsAll), "3D cavity reference diagnostic RMS finite");
  check(std::isfinite(rmsInterior), "3D cavity reference diagnostic interior RMS finite");
  check(!candidates.empty() && std::isfinite(candidates.front().interiorL2),
        "3D cavity reference mapping sweep finite");
  std::cout << "cavity3d_reference_digitized_l2=" << relL2
            << " cavity3d_reference_interior_l2=" << interiorRelL2
            << " cavity3d_reference_rms=" << rmsAll
            << " cavity3d_reference_interior_rms=" << rmsInterior
            << " cavity3d_reference_max_courant=" << sol.maxCourant
            << " cavity3d_reference_best_mapping_axis=" << candidates.front().refAxis
            << " line=" << candidates.front().lineAxis
            << " component=" << candidates.front().component
            << " flip=" << (candidates.front().flipCoord ? 1 : 0)
            << " best_interior_l2=" << candidates.front().interiorL2
            << " gate_status=partial target_l2=0.02\n";
}
