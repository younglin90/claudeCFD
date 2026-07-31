#include "fvm/Mesh3D.hpp"
#include "fvm/MeshQuality3D.hpp"
#include "fvm/OpenFoamPolyMeshReader3D.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct MeshValidateOptions {
  std::string mode = "builtin_hex";
  std::filesystem::path openFoamPolyMeshDir;
  std::filesystem::path output;
  int nx = 8;
  int ny = 16;
  int nz = 8;
  double lx = 1.0;
  double ly = 4.0;
  double lz = 1.0;
  double skew = 0.0;
};

std::string jsonEscape(const std::string& s) {
  std::ostringstream os;
  for (char ch : s) {
    switch (ch) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default: os << ch; break;
    }
  }
  return os.str();
}

int parseIntArg(const std::string& s, const std::string& name) {
  try {
    size_t consumed = 0;
    const int value = std::stoi(s, &consumed);
    if (consumed != s.size()) throw std::invalid_argument("trailing text");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(name + " requires an integer");
  }
}

double parseDoubleArg(const std::string& s, const std::string& name) {
  try {
    size_t consumed = 0;
    const double value = std::stod(s, &consumed);
    if (consumed != s.size()) throw std::invalid_argument("trailing text");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(name + " requires a scalar");
  }
}

MeshValidateOptions parseOptions(int argc, char** argv) {
  MeshValidateOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(name + " requires a value");
      return argv[++i];
    };
    if (arg == "--builtin-hex") {
      opt.mode = "builtin_hex";
    } else if (arg == "--openfoam-polyMesh") {
      opt.mode = "openfoam_polyMesh";
      opt.openFoamPolyMeshDir = need(arg);
    } else if (arg == "--nx") {
      opt.nx = parseIntArg(need(arg), arg);
    } else if (arg == "--ny") {
      opt.ny = parseIntArg(need(arg), arg);
    } else if (arg == "--nz") {
      opt.nz = parseIntArg(need(arg), arg);
    } else if (arg == "--lx") {
      opt.lx = parseDoubleArg(need(arg), arg);
    } else if (arg == "--ly") {
      opt.ly = parseDoubleArg(need(arg), arg);
    } else if (arg == "--lz") {
      opt.lz = parseDoubleArg(need(arg), arg);
    } else if (arg == "--skew") {
      opt.skew = parseDoubleArg(need(arg), arg);
    } else if (arg == "--output") {
      opt.output = need(arg);
    } else if (arg == "--help") {
      std::cout
          << "usage: electrospray_mesh_validate [--builtin-hex --nx N --ny N --nz N]\n"
          << "       electrospray_mesh_validate --openfoam-polyMesh constant/polyMesh\n"
          << "       [--lx L --ly L --lz L --skew S --output mesh_summary.csv]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opt.mode == "openfoam_polyMesh" && opt.openFoamPolyMeshDir.empty()) {
    throw std::runtime_error("--openfoam-polyMesh requires a directory");
  }
  if (opt.nx <= 0 || opt.ny <= 0 || opt.nz <= 0) {
    throw std::runtime_error("builtin mesh dimensions must be positive");
  }
  return opt;
}

int boundaryFaceCount(const fvm::Mesh3D& mesh) {
  int count = 0;
  for (const auto& f : mesh.faces) {
    if (!f.internal()) ++count;
  }
  return count;
}

std::string meshJson(const fvm::Mesh3D& mesh, const fvm::MeshQualityReport3D& quality,
                     const MeshValidateOptions& opt,
                     const fvm::OpenFoamPolyMeshReadReport3D& foamReport, bool ok) {
  std::ostringstream os;
  os << std::setprecision(12);
  os << "{\n";
  os << "  \"status\": \"" << (ok ? "pass" : "fail") << "\",\n";
  os << "  \"mesh_mode\": \"" << jsonEscape(opt.mode) << "\",\n";
  os << "  \"cells\": " << quality.cells << ",\n";
  os << "  \"faces\": " << quality.faces << ",\n";
  os << "  \"points\": " << mesh.points.size() << ",\n";
  os << "  \"internal_faces\": " << quality.internalFaces << ",\n";
  os << "  \"boundary_faces\": " << boundaryFaceCount(mesh) << ",\n";
  os << "  \"patch_count\": " << mesh.patches.size() << ",\n";
  os << "  \"finite\": " << (quality.finite ? "true" : "false") << ",\n";
  os << "  \"non_positive_volume_count\": " << quality.nonPositiveVolumeCount << ",\n";
  os << "  \"zero_area_face_count\": " << quality.zeroAreaFaceCount << ",\n";
  os << "  \"min_volume\": " << quality.minVolume << ",\n";
  os << "  \"max_volume\": " << quality.maxVolume << ",\n";
  os << "  \"min_face_area\": " << quality.minFaceArea << ",\n";
  os << "  \"max_face_area\": " << quality.maxFaceArea << ",\n";
  os << "  \"max_non_orthogonality_deg\": " << quality.maxNonOrthogonalityDeg << ",\n";
  os << "  \"mean_non_orthogonality_deg\": " << quality.meanNonOrthogonalityDeg << ",\n";
  os << "  \"max_skewness\": " << quality.maxSkewness << ",\n";
  os << "  \"max_aspect_ratio\": " << quality.maxAspectRatio << ",\n";
  os << "  \"openfoam_neighbour_faces\": " << foamReport.neighbours << ",\n";
  os << "  \"patches\": [\n";
  for (size_t p = 0; p < mesh.patches.size(); ++p) {
    os << "    {\"name\": \"" << jsonEscape(mesh.patches[p].name)
       << "\", \"faces\": " << mesh.patches[p].faces.size() << "}";
    if (p + 1 != mesh.patches.size()) os << ",";
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";
  return os.str();
}

void writeCsv(const std::filesystem::path& path, const fvm::Mesh3D& mesh,
              const fvm::MeshQualityReport3D& q, const MeshValidateOptions& opt, bool ok) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  std::ofstream csv(path);
  if (!csv) throw std::runtime_error("failed to write mesh validation CSV: " + path.string());
  csv << std::setprecision(12);
  csv << "status,mesh_mode,points,cells,faces,internal_faces,boundary_faces,patches,"
         "finite,non_positive_volume_count,zero_area_face_count,min_volume,max_volume,"
         "min_face_area,max_face_area,max_non_orthogonality_deg,mean_non_orthogonality_deg,"
         "max_skewness,max_aspect_ratio\n";
  csv << (ok ? "pass" : "fail") << "," << opt.mode << "," << mesh.points.size() << ","
      << q.cells << "," << q.faces << "," << q.internalFaces << "," << boundaryFaceCount(mesh)
      << "," << mesh.patches.size() << "," << (q.finite ? 1 : 0) << ","
      << q.nonPositiveVolumeCount << "," << q.zeroAreaFaceCount << "," << q.minVolume << ","
      << q.maxVolume << "," << q.minFaceArea << "," << q.maxFaceArea << ","
      << q.maxNonOrthogonalityDeg << "," << q.meanNonOrthogonalityDeg << ","
      << q.maxSkewness << "," << q.maxAspectRatio << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const MeshValidateOptions opt = parseOptions(argc, argv);
    fvm::OpenFoamPolyMeshReadReport3D foamReport;
    fvm::Mesh3D mesh;
    if (opt.mode == "openfoam_polyMesh") {
      mesh = fvm::readOpenFoamPolyMesh3D(opt.openFoamPolyMeshDir, &foamReport);
    } else {
      mesh = fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, opt.lx, opt.ly, opt.lz, opt.skew);
    }

    const auto quality = fvm::meshQualityReport3D(mesh);
    const bool ok = quality.finite && quality.nonPositiveVolumeCount == 0 &&
                    quality.zeroAreaFaceCount == 0 && quality.cells > 0 && quality.faces > 0;
    if (!opt.output.empty()) writeCsv(opt.output, mesh, quality, opt, ok);
    std::cout << meshJson(mesh, quality, opt, foamReport, ok);
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "{\n  \"status\": \"error\",\n  \"error\": \"" << jsonEscape(e.what()) << "\"\n}\n";
    return 2;
  }
}
