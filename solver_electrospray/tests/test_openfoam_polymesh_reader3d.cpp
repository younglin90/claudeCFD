#include "TestUtil.hpp"
#include "fvm/MeshQuality3D.hpp"
#include "fvm/OpenFoamPolyMeshReader3D.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

void writeFile(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to write fixture: " + path.string());
  out << text;
}

}  // namespace

int main() {
  const std::filesystem::path dir = "benchmark_logs/openfoam_polyMesh_fixture/constant/polyMesh";
  writeFile(dir / "points",
            "FoamFile { version 2.0; format ascii; class vectorField; object points; }\n"
            "12\n"
            "(\n"
            "(0 0 0)\n(1 0 0)\n(1 1 0)\n(0 1 0)\n"
            "(0 0 1)\n(1 0 1)\n(1 1 1)\n(0 1 1)\n"
            "(2 0 0)\n(2 1 0)\n(2 0 1)\n(2 1 1)\n"
            ")\n");
  writeFile(dir / "faces",
            "FoamFile { version 2.0; format ascii; class faceList; object faces; }\n"
            "11\n"
            "(\n"
            "4(1 5 6 2)\n"
            "4(0 3 7 4)\n"
            "4(8 10 11 9)\n"
            "4(0 4 5 1)\n"
            "4(1 5 10 8)\n"
            "4(3 2 6 7)\n"
            "4(2 9 11 6)\n"
            "4(0 1 2 3)\n"
            "4(1 8 9 2)\n"
            "4(4 7 6 5)\n"
            "4(5 6 11 10)\n"
            ")\n");
  writeFile(dir / "owner",
            "FoamFile { version 2.0; format ascii; class labelList; object owner; }\n"
            "11\n"
            "(\n0\n0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n)\n");
  writeFile(dir / "neighbour",
            "FoamFile { version 2.0; format ascii; class labelList; object neighbour; }\n"
            "1\n"
            "(\n1\n)\n");
  writeFile(dir / "boundary",
            "FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }\n"
            "6\n"
            "(\n"
            "xmin\n{\n type wall;\n nFaces 1;\n startFace 1;\n}\n"
            "xmax\n{\n type patch;\n nFaces 1;\n startFace 2;\n}\n"
            "ymin\n{\n type wall;\n nFaces 2;\n startFace 3;\n}\n"
            "ymax\n{\n type wall;\n nFaces 2;\n startFace 5;\n}\n"
            "zmin\n{\n type symmetryPlane;\n nFaces 2;\n startFace 7;\n}\n"
            "zmax\n{\n type symmetryPlane;\n nFaces 2;\n startFace 9;\n}\n"
            ")\n");

  fvm::OpenFoamPolyMeshReadReport3D report;
  const fvm::Mesh3D mesh = fvm::readOpenFoamPolyMesh3D(dir, &report);
  const auto quality = fvm::meshQualityReport3D(mesh);

  check(report.points == 12, "OpenFOAM reader point count");
  check(report.faces == 11, "OpenFOAM reader face count");
  check(report.neighbours == 1, "OpenFOAM reader neighbour count");
  check(report.cells == 2, "OpenFOAM reader cell count");
  check(report.patches == 6, "OpenFOAM reader patch count");
  check(report.boundaryFaces == 10, "OpenFOAM reader boundary face count");
  check(mesh.cells.size() == 2, "OpenFOAM mesh has two cells");
  check(mesh.faces.size() == 11, "OpenFOAM mesh has eleven faces");
  check(mesh.faces[0].internal(), "OpenFOAM first face is internal");
  check(mesh.patches[0].name == "xmin", "OpenFOAM patch name xmin preserved");
  check(mesh.patches[1].name == "xmax", "OpenFOAM patch name xmax preserved");
  check(mesh.patches[2].faces.size() == 2, "OpenFOAM patch range face count preserved");
  check(mesh.faces[1].patch == 0, "OpenFOAM face patch index preserved");
  check(mesh.faces[2].patch == 1, "OpenFOAM second boundary patch index preserved");
  check(quality.finite, "OpenFOAM mesh quality finite");
  check(quality.nonPositiveVolumeCount == 0, "OpenFOAM mesh positive volumes");
  check(quality.zeroAreaFaceCount == 0, "OpenFOAM mesh positive face areas");
  check(quality.internalFaces == 1, "OpenFOAM mesh internal face count");
  check(quality.maxNonOrthogonalityDeg < 1e-9, "OpenFOAM orthogonal fixture non-orthogonal angle");

  std::ofstream csv("benchmark_logs/openfoam_polymesh_reader3d.csv");
  csv << "case,points,cells,faces,patches,boundary_faces,max_non_orthogonality_deg,"
         "max_aspect_ratio\n";
  csv << "two_hex_openfoam," << report.points << "," << report.cells << "," << report.faces
      << "," << report.patches << "," << report.boundaryFaces << ","
      << quality.maxNonOrthogonalityDeg << "," << quality.maxAspectRatio << "\n";

  std::cout << "openfoam_polyMesh_points=" << report.points
            << " cells=" << report.cells
            << " faces=" << report.faces
            << " patches=" << report.patches
            << " max_non_ortho=" << quality.maxNonOrthogonalityDeg << "\n";
}
