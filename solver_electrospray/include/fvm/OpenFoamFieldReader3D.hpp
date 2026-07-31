#pragma once

#include "fvm/OpenFoamPolyMeshReader3D.hpp"

#include <filesystem>
#include <map>
#include <set>
#include <sstream>

namespace fvm {

struct OpenFoamScalarPatchField3D {
  std::string patch;
  std::string type = "zeroGradient";
  bool hasUniformValue = false;
  double uniformValue = 0.0;
};

struct OpenFoamVectorPatchField3D {
  std::string patch;
  std::string type = "zeroGradient";
  bool hasUniformValue = false;
  Vec3 uniformValue = Vec3::Zero();
};

struct OpenFoamScalarField3D {
  std::string name;
  ScalarField internal;
  std::string internalForm = "uniform";
  double uniformInternalValue = 0.0;
  std::map<std::string, OpenFoamScalarPatchField3D> boundary;
};

struct OpenFoamVectorField3D {
  std::string name;
  VectorField3 internal;
  std::string internalForm = "uniform";
  Vec3 uniformInternalValue = Vec3::Zero();
  std::map<std::string, OpenFoamVectorPatchField3D> boundary;
};

struct OpenFoamCaseFields3D {
  bool hasU = false;
  bool hasP = false;
  bool hasAlpha = false;
  bool hasPhi = false;
  bool hasRhoE = false;
  OpenFoamVectorField3D U;
  OpenFoamScalarField3D p;
  OpenFoamScalarField3D alpha;
  OpenFoamScalarField3D phi;
  OpenFoamScalarField3D rhoE;
};

struct OpenFoamCaseValidationReport3D {
  int fieldsRead = 0;
  int scalarFieldsRead = 0;
  int vectorFieldsRead = 0;
  int boundaryEntries = 0;
  int missingPatchBoundaryEntries = 0;
  int unknownPatchBoundaryEntries = 0;
  std::vector<std::string> missingFiles;
  std::vector<std::string> missingPatchEntries;
  std::vector<std::string> unknownPatchEntries;
};

namespace detail {

inline bool tokenIsScalar3D(const std::string& token) {
  try {
    size_t consumed = 0;
    (void)std::stod(token, &consumed);
    return consumed == token.size();
  } catch (const std::exception&) {
    return false;
  }
}

inline size_t findToken3D(const std::vector<std::string>& tokens, const std::string& key) {
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == key) return i;
  }
  throw std::runtime_error("missing OpenFOAM field token: " + key);
}

inline Vec3 readFoamVectorValue3D(const std::vector<std::string>& tokens, size_t& i,
                                  const std::string& context) {
  expectToken3D(tokens, i, "(", context);
  const double x = readDouble3D(tokens, i, context);
  const double y = readDouble3D(tokens, i, context);
  const double z = readDouble3D(tokens, i, context);
  expectToken3D(tokens, i, ")", context);
  return Vec3{x, y, z};
}

inline void skipOptionalSemicolon3D(const std::vector<std::string>& tokens, size_t& i) {
  if (i < tokens.size() && tokens[i] == ";") ++i;
}

inline ScalarField readScalarInternalField3D(const std::vector<std::string>& tokens, size_t& i,
                                             int nCells, const std::string& context,
                                             std::string& form, double& uniformValue) {
  if (i >= tokens.size()) throw std::runtime_error("missing scalar internalField value in " + context);
  const std::string mode = tokens[i++];
  if (mode == "uniform") {
    uniformValue = readDouble3D(tokens, i, context);
    skipOptionalSemicolon3D(tokens, i);
    form = "uniform";
    return ScalarField(static_cast<size_t>(nCells), uniformValue);
  }
  if (mode != "nonuniform") {
    throw std::runtime_error("unsupported scalar internalField mode '" + mode + "' in " + context);
  }
  while (i < tokens.size() && !tokenIsScalar3D(tokens[i])) ++i;
  const int n = readInt3D(tokens, i, context);
  if (n != nCells) throw std::runtime_error("nonuniform scalar field cell count mismatch in " + context);
  expectToken3D(tokens, i, "(", context);
  ScalarField values(static_cast<size_t>(n), 0.0);
  for (int c = 0; c < n; ++c) values[static_cast<size_t>(c)] = readDouble3D(tokens, i, context);
  expectToken3D(tokens, i, ")", context);
  skipOptionalSemicolon3D(tokens, i);
  form = "nonuniform";
  return values;
}

inline VectorField3 readVectorInternalField3D(const std::vector<std::string>& tokens, size_t& i,
                                              int nCells, const std::string& context,
                                              std::string& form, Vec3& uniformValue) {
  if (i >= tokens.size()) throw std::runtime_error("missing vector internalField value in " + context);
  const std::string mode = tokens[i++];
  if (mode == "uniform") {
    uniformValue = readFoamVectorValue3D(tokens, i, context);
    skipOptionalSemicolon3D(tokens, i);
    form = "uniform";
    return VectorField3(static_cast<size_t>(nCells), uniformValue);
  }
  if (mode != "nonuniform") {
    throw std::runtime_error("unsupported vector internalField mode '" + mode + "' in " + context);
  }
  while (i < tokens.size() && !tokenIsScalar3D(tokens[i])) ++i;
  const int n = readInt3D(tokens, i, context);
  if (n != nCells) throw std::runtime_error("nonuniform vector field cell count mismatch in " + context);
  expectToken3D(tokens, i, "(", context);
  VectorField3 values(static_cast<size_t>(n), Vec3::Zero());
  for (int c = 0; c < n; ++c) values[static_cast<size_t>(c)] = readFoamVectorValue3D(tokens, i, context);
  expectToken3D(tokens, i, ")", context);
  skipOptionalSemicolon3D(tokens, i);
  form = "nonuniform";
  return values;
}

inline std::map<std::string, OpenFoamScalarPatchField3D> readScalarBoundaryField3D(
    const std::vector<std::string>& tokens, const std::string& context) {
  std::map<std::string, OpenFoamScalarPatchField3D> out;
  size_t i = findToken3D(tokens, "boundaryField") + 1;
  expectToken3D(tokens, i, "{", context);
  while (i < tokens.size() && tokens[i] != "}") {
    OpenFoamScalarPatchField3D bc;
    bc.patch = tokens[i++];
    expectToken3D(tokens, i, "{", context);
    while (i < tokens.size() && tokens[i] != "}") {
      const std::string key = tokens[i++];
      if (key == "type") {
        if (i >= tokens.size()) throw std::runtime_error("missing boundary type in " + context);
        bc.type = tokens[i++];
        skipOptionalSemicolon3D(tokens, i);
      } else if (key == "value") {
        if (i < tokens.size() && tokens[i] == "uniform") ++i;
        bc.uniformValue = readDouble3D(tokens, i, context);
        bc.hasUniformValue = true;
        skipOptionalSemicolon3D(tokens, i);
      } else {
        while (i < tokens.size() && tokens[i] != ";" && tokens[i] != "}") ++i;
        skipOptionalSemicolon3D(tokens, i);
      }
    }
    expectToken3D(tokens, i, "}", context);
    out[bc.patch] = bc;
  }
  expectToken3D(tokens, i, "}", context);
  return out;
}

inline std::map<std::string, OpenFoamVectorPatchField3D> readVectorBoundaryField3D(
    const std::vector<std::string>& tokens, const std::string& context) {
  std::map<std::string, OpenFoamVectorPatchField3D> out;
  size_t i = findToken3D(tokens, "boundaryField") + 1;
  expectToken3D(tokens, i, "{", context);
  while (i < tokens.size() && tokens[i] != "}") {
    OpenFoamVectorPatchField3D bc;
    bc.patch = tokens[i++];
    expectToken3D(tokens, i, "{", context);
    while (i < tokens.size() && tokens[i] != "}") {
      const std::string key = tokens[i++];
      if (key == "type") {
        if (i >= tokens.size()) throw std::runtime_error("missing boundary type in " + context);
        bc.type = tokens[i++];
        skipOptionalSemicolon3D(tokens, i);
      } else if (key == "value") {
        if (i < tokens.size() && tokens[i] == "uniform") ++i;
        bc.uniformValue = readFoamVectorValue3D(tokens, i, context);
        bc.hasUniformValue = true;
        skipOptionalSemicolon3D(tokens, i);
      } else {
        while (i < tokens.size() && tokens[i] != ";" && tokens[i] != "}") ++i;
        skipOptionalSemicolon3D(tokens, i);
      }
    }
    expectToken3D(tokens, i, "}", context);
    out[bc.patch] = bc;
  }
  expectToken3D(tokens, i, "}", context);
  return out;
}

inline std::set<std::string> meshPatchNameSet3D(const Mesh3D& mesh) {
  std::set<std::string> names;
  for (const Patch3D& p : mesh.patches) names.insert(p.name);
  return names;
}

template <class BoundaryMap>
inline void accumulatePatchValidation3D(const Mesh3D& mesh, const std::string& fieldName,
                                        const BoundaryMap& boundary,
                                        OpenFoamCaseValidationReport3D& report) {
  const std::set<std::string> patchNames = meshPatchNameSet3D(mesh);
  report.boundaryEntries += static_cast<int>(boundary.size());
  for (const Patch3D& p : mesh.patches) {
    if (boundary.find(p.name) == boundary.end()) {
      ++report.missingPatchBoundaryEntries;
      report.missingPatchEntries.push_back(fieldName + ":" + p.name);
    }
  }
  for (const auto& kv : boundary) {
    if (patchNames.find(kv.first) == patchNames.end()) {
      ++report.unknownPatchBoundaryEntries;
      report.unknownPatchEntries.push_back(fieldName + ":" + kv.first);
    }
  }
}

}  // namespace detail

inline OpenFoamScalarField3D readOpenFoamScalarField3D(const std::filesystem::path& path,
                                                       const Mesh3D& mesh,
                                                       const std::string& name) {
  const std::string context = path.string();
  const auto tokens = detail::foamTokens3D(detail::readTextFile3D(path));
  OpenFoamScalarField3D field;
  field.name = name;
  size_t internal = detail::findToken3D(tokens, "internalField") + 1;
  field.internal = detail::readScalarInternalField3D(
      tokens, internal, static_cast<int>(mesh.cells.size()), context,
      field.internalForm, field.uniformInternalValue);
  field.boundary = detail::readScalarBoundaryField3D(tokens, context);
  return field;
}

inline OpenFoamVectorField3D readOpenFoamVectorField3D(const std::filesystem::path& path,
                                                       const Mesh3D& mesh,
                                                       const std::string& name) {
  const std::string context = path.string();
  const auto tokens = detail::foamTokens3D(detail::readTextFile3D(path));
  OpenFoamVectorField3D field;
  field.name = name;
  size_t internal = detail::findToken3D(tokens, "internalField") + 1;
  field.internal = detail::readVectorInternalField3D(
      tokens, internal, static_cast<int>(mesh.cells.size()), context,
      field.internalForm, field.uniformInternalValue);
  field.boundary = detail::readVectorBoundaryField3D(tokens, context);
  return field;
}

inline OpenFoamCaseFields3D readOpenFoamCaseFields3D(
    const std::filesystem::path& caseDir, const Mesh3D& mesh,
    OpenFoamCaseValidationReport3D* report = nullptr) {
  OpenFoamCaseFields3D fields;
  OpenFoamCaseValidationReport3D localReport;
  const std::filesystem::path zero = caseDir / "0";

  auto readScalarIfExists = [&](const std::string& file, OpenFoamScalarField3D& target,
                                bool& flag) {
    const auto path = zero / file;
    if (!std::filesystem::exists(path)) {
      localReport.missingFiles.push_back(("0/" + file));
      return;
    }
    target = readOpenFoamScalarField3D(path, mesh, file);
    flag = true;
    ++localReport.fieldsRead;
    ++localReport.scalarFieldsRead;
    detail::accumulatePatchValidation3D(mesh, file, target.boundary, localReport);
  };

  auto readVectorIfExists = [&](const std::string& file, OpenFoamVectorField3D& target,
                                bool& flag) {
    const auto path = zero / file;
    if (!std::filesystem::exists(path)) {
      localReport.missingFiles.push_back(("0/" + file));
      return;
    }
    target = readOpenFoamVectorField3D(path, mesh, file);
    flag = true;
    ++localReport.fieldsRead;
    ++localReport.vectorFieldsRead;
    detail::accumulatePatchValidation3D(mesh, file, target.boundary, localReport);
  };

  readVectorIfExists("U", fields.U, fields.hasU);
  readScalarIfExists("p", fields.p, fields.hasP);
  readScalarIfExists("alpha", fields.alpha, fields.hasAlpha);
  readScalarIfExists("phi", fields.phi, fields.hasPhi);
  readScalarIfExists("rhoE", fields.rhoE, fields.hasRhoE);

  if (report) *report = localReport;
  return fields;
}

inline std::string openFoamCaseFieldsSummaryJson3D(
    const OpenFoamCaseValidationReport3D& report) {
  std::ostringstream os;
  os << "{";
  os << "\"fields_read\":" << report.fieldsRead << ",";
  os << "\"scalar_fields_read\":" << report.scalarFieldsRead << ",";
  os << "\"vector_fields_read\":" << report.vectorFieldsRead << ",";
  os << "\"boundary_entries\":" << report.boundaryEntries << ",";
  os << "\"missing_patch_boundary_entries\":" << report.missingPatchBoundaryEntries << ",";
  os << "\"unknown_patch_boundary_entries\":" << report.unknownPatchBoundaryEntries;
  os << "}";
  return os.str();
}

}  // namespace fvm
