#include "TestUtil.hpp"
#include "fvm/OpenFoamFieldReader3D.hpp"

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

std::string scalarField(const std::string& name, const std::string& internal,
                        const std::string& xminValue, const std::string& xmaxValue) {
  return "FoamFile { version 2.0; format ascii; class volScalarField; object " + name + "; }\n"
         "dimensions [0 0 0 0 0 0 0];\n"
         "internalField " + internal + ";\n"
         "boundaryField\n"
         "{\n"
         "xmin { type fixedValue; value uniform " + xminValue + "; }\n"
         "xmax { type fixedValue; value uniform " + xmaxValue + "; }\n"
         "ymin { type zeroGradient; }\n"
         "ymax { type zeroGradient; }\n"
         "zmin { type symmetryPlane; }\n"
         "zmax { type symmetryPlane; }\n"
         "}\n";
}

}  // namespace

int main() {
  const std::filesystem::path caseDir = "benchmark_logs/openfoam_case_fixture";
  const std::filesystem::path meshDir = caseDir / "constant" / "polyMesh";
  writeFile(meshDir / "points",
            "FoamFile { version 2.0; format ascii; class vectorField; object points; }\n"
            "12\n"
            "(\n"
            "(0 0 0)\n(1 0 0)\n(1 1 0)\n(0 1 0)\n"
            "(0 0 1)\n(1 0 1)\n(1 1 1)\n(0 1 1)\n"
            "(2 0 0)\n(2 1 0)\n(2 0 1)\n(2 1 1)\n"
            ")\n");
  writeFile(meshDir / "faces",
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
  writeFile(meshDir / "owner",
            "FoamFile { version 2.0; format ascii; class labelList; object owner; }\n"
            "11\n"
            "(\n0\n0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n)\n");
  writeFile(meshDir / "neighbour",
            "FoamFile { version 2.0; format ascii; class labelList; object neighbour; }\n"
            "1\n"
            "(\n1\n)\n");
  writeFile(meshDir / "boundary",
            "FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }\n"
            "6\n"
            "(\n"
            "xmin { type patch; nFaces 1; startFace 1; }\n"
            "xmax { type patch; nFaces 1; startFace 2; }\n"
            "ymin { type wall; nFaces 2; startFace 3; }\n"
            "ymax { type wall; nFaces 2; startFace 5; }\n"
            "zmin { type symmetryPlane; nFaces 2; startFace 7; }\n"
            "zmax { type symmetryPlane; nFaces 2; startFace 9; }\n"
            ")\n");

  writeFile(caseDir / "0" / "U",
            "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
            "dimensions [0 1 -1 0 0 0 0];\n"
            "internalField uniform (0 0 0);\n"
            "boundaryField\n"
            "{\n"
            "xmin { type fixedValue; value uniform (0 1 0); }\n"
            "xmax { type zeroGradient; }\n"
            "ymin { type noSlip; }\n"
            "ymax { type noSlip; }\n"
            "zmin { type symmetryPlane; }\n"
            "zmax { type symmetryPlane; }\n"
            "}\n");
  writeFile(caseDir / "0" / "p", scalarField("p", "uniform 0", "0", "0"));
  writeFile(caseDir / "0" / "alpha", scalarField("alpha", "uniform 0", "1", "0"));
  writeFile(caseDir / "0" / "phi", scalarField("phi", "uniform 0", "2180", "0"));
  writeFile(caseDir / "0" / "rhoE",
            scalarField("rhoE", "nonuniform List<scalar> 2 ( 0.1 -0.2 )", "0", "0"));

  fvm::OpenFoamPolyMeshReadReport3D meshReport;
  const fvm::Mesh3D mesh = fvm::readOpenFoamPolyMesh3D(meshDir, &meshReport);
  fvm::OpenFoamCaseValidationReport3D fieldReport;
  const auto fields = fvm::readOpenFoamCaseFields3D(caseDir, mesh, &fieldReport);

  check(fields.hasU && fields.hasP && fields.hasAlpha && fields.hasPhi && fields.hasRhoE,
        "OpenFOAM field reader loads all expected fields");
  check(fields.U.internal.size() == mesh.cells.size(), "OpenFOAM U cell count");
  check(fields.p.internal.size() == mesh.cells.size(), "OpenFOAM p cell count");
  check(fields.rhoE.internalForm == "nonuniform", "OpenFOAM nonuniform scalar form");
  check(std::abs(fields.rhoE.internal[0] - 0.1) < 1e-14, "OpenFOAM nonuniform scalar value 0");
  check(std::abs(fields.rhoE.internal[1] + 0.2) < 1e-14, "OpenFOAM nonuniform scalar value 1");
  check(fields.U.boundary.at("xmin").type == "fixedValue", "OpenFOAM U xmin fixedValue");
  check(std::abs(fields.U.boundary.at("xmin").uniformValue.y() - 1.0) < 1e-14,
        "OpenFOAM U xmin vector value");
  check(fields.U.boundary.at("ymin").type == "noSlip", "OpenFOAM U wall noSlip");
  check(fields.alpha.boundary.at("xmin").uniformValue == 1.0, "OpenFOAM alpha inlet value");
  check(fields.phi.boundary.at("xmin").uniformValue == 2180.0, "OpenFOAM phi electrode value");
  check(fieldReport.fieldsRead == 5, "OpenFOAM case reads five fields");
  check(fieldReport.missingPatchBoundaryEntries == 0, "OpenFOAM field BC covers all patches");
  check(fieldReport.unknownPatchBoundaryEntries == 0, "OpenFOAM field BC has no unknown patches");

  std::ofstream csv("benchmark_logs/openfoam_field_reader3d.csv");
  csv << "case,fields_read,boundary_entries,missing_patch_bc,unknown_patch_bc,"
         "rhoE0,rhoE1,U_xmin_y,phi_xmin\n";
  csv << "two_hex_case," << fieldReport.fieldsRead << "," << fieldReport.boundaryEntries
      << "," << fieldReport.missingPatchBoundaryEntries << ","
      << fieldReport.unknownPatchBoundaryEntries << "," << fields.rhoE.internal[0]
      << "," << fields.rhoE.internal[1] << ","
      << fields.U.boundary.at("xmin").uniformValue.y() << ","
      << fields.phi.boundary.at("xmin").uniformValue << "\n";

  std::cout << "openfoam_fields_read=" << fieldReport.fieldsRead
            << " boundary_entries=" << fieldReport.boundaryEntries
            << " rhoE0=" << fields.rhoE.internal[0]
            << " U_xmin_y=" << fields.U.boundary.at("xmin").uniformValue.y() << "\n";
}
