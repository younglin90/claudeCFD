#pragma once

#include "fvm/Mesh3D.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fvm {

struct OpenFoamPatchRange3D {
  std::string name;
  int startFace = 0;
  int nFaces = 0;
};

struct OpenFoamPolyMeshReadReport3D {
  int points = 0;
  int faces = 0;
  int owners = 0;
  int neighbours = 0;
  int cells = 0;
  int patches = 0;
  int boundaryFaces = 0;
};

namespace detail {

inline std::string readTextFile3D(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to read OpenFOAM file: " + path.string());
  return std::string(std::istreambuf_iterator<char>(in), {});
}

inline std::string stripFoamComments3D(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size();) {
    if (i + 1 < text.size() && text[i] == '/' && text[i + 1] == '/') {
      i += 2;
      while (i < text.size() && text[i] != '\n') ++i;
      continue;
    }
    if (i + 1 < text.size() && text[i] == '/' && text[i + 1] == '*') {
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/')) ++i;
      i = std::min(i + 2, text.size());
      continue;
    }
    out.push_back(text[i++]);
  }
  return out;
}

inline std::vector<std::string> foamTokens3D(const std::string& text) {
  const std::string clean = stripFoamComments3D(text);
  std::vector<std::string> tokens;
  for (size_t i = 0; i < clean.size();) {
    const unsigned char ch = static_cast<unsigned char>(clean[i]);
    if (std::isspace(ch)) {
      ++i;
      continue;
    }
    if (clean[i] == '(' || clean[i] == ')' || clean[i] == '{' || clean[i] == '}' ||
        clean[i] == ';') {
      tokens.emplace_back(1, clean[i++]);
      continue;
    }
    if (std::isdigit(ch) || clean[i] == '-' || clean[i] == '+' || clean[i] == '.') {
      const size_t start = i;
      while (i < clean.size() && !std::isspace(static_cast<unsigned char>(clean[i])) &&
             clean[i] != '(' && clean[i] != ')' && clean[i] != '{' && clean[i] != '}' &&
             clean[i] != ';') {
        ++i;
      }
      tokens.push_back(clean.substr(start, i - start));
      continue;
    }
    const size_t start = i;
    while (i < clean.size() && !std::isspace(static_cast<unsigned char>(clean[i])) &&
           clean[i] != '(' && clean[i] != ')' && clean[i] != '{' && clean[i] != '}' &&
           clean[i] != ';') {
      ++i;
    }
    tokens.push_back(clean.substr(start, i - start));
  }
  return tokens;
}

inline bool parseIntToken3D(const std::string& token, int& value) {
  if (token.empty()) return false;
  size_t i = 0;
  if (token[i] == '+' || token[i] == '-') ++i;
  if (i == token.size()) return false;
  for (; i < token.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(token[i]))) return false;
  }
  value = std::stoi(token);
  return true;
}

inline bool parseUnsignedToken3D(const std::string& token, int& value) {
  if (token.empty()) return false;
  for (char ch : token) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
  }
  value = std::stoi(token);
  return true;
}

inline int readInt3D(const std::vector<std::string>& tokens, size_t& i,
                    const std::string& context) {
  if (i >= tokens.size()) throw std::runtime_error("unexpected EOF reading int in " + context);
  int value = 0;
  if (!parseIntToken3D(tokens[i], value)) {
    throw std::runtime_error("expected integer in " + context + ", got '" + tokens[i] + "'");
  }
  ++i;
  return value;
}

inline double readDouble3D(const std::vector<std::string>& tokens, size_t& i,
                           const std::string& context) {
  if (i >= tokens.size()) throw std::runtime_error("unexpected EOF reading scalar in " + context);
  try {
    size_t consumed = 0;
    const double value = std::stod(tokens[i], &consumed);
    if (consumed != tokens[i].size()) throw std::invalid_argument("trailing text");
    ++i;
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("expected scalar in " + context + ", got '" + tokens[i] + "'");
  }
}

inline void expectToken3D(const std::vector<std::string>& tokens, size_t& i,
                          const std::string& expected, const std::string& context) {
  if (i >= tokens.size() || tokens[i] != expected) {
    throw std::runtime_error("expected '" + expected + "' in " + context);
  }
  ++i;
}

inline size_t findFoamListStart3D(const std::vector<std::string>& tokens,
                                  const std::string& context, int& count) {
  for (size_t i = 0; i + 1 < tokens.size(); ++i) {
    if (parseUnsignedToken3D(tokens[i], count) && tokens[i + 1] == "(") {
      return i + 2;
    }
  }
  throw std::runtime_error("failed to find OpenFOAM list in " + context);
}

inline std::vector<Vec3> parseFoamPoints3D(const std::filesystem::path& path) {
  const auto tokens = foamTokens3D(readTextFile3D(path));
  int n = 0;
  size_t i = findFoamListStart3D(tokens, path.string(), n);
  std::vector<Vec3> points;
  points.reserve(static_cast<size_t>(n));
  for (int p = 0; p < n; ++p) {
    expectToken3D(tokens, i, "(", path.string());
    const double x = readDouble3D(tokens, i, path.string());
    const double y = readDouble3D(tokens, i, path.string());
    const double z = readDouble3D(tokens, i, path.string());
    expectToken3D(tokens, i, ")", path.string());
    points.push_back(Vec3{x, y, z});
  }
  return points;
}

inline std::vector<std::vector<int>> parseFoamFaces3D(const std::filesystem::path& path) {
  const auto tokens = foamTokens3D(readTextFile3D(path));
  int n = 0;
  size_t i = findFoamListStart3D(tokens, path.string(), n);
  std::vector<std::vector<int>> faces;
  faces.reserve(static_cast<size_t>(n));
  for (int f = 0; f < n; ++f) {
    int faceSize = -1;
    if (i < tokens.size() && parseUnsignedToken3D(tokens[i], faceSize)) {
      ++i;
      expectToken3D(tokens, i, "(", path.string());
      std::vector<int> ids;
      ids.reserve(static_cast<size_t>(faceSize));
      for (int k = 0; k < faceSize; ++k) ids.push_back(readInt3D(tokens, i, path.string()));
      expectToken3D(tokens, i, ")", path.string());
      faces.push_back(std::move(ids));
    } else {
      expectToken3D(tokens, i, "(", path.string());
      std::vector<int> ids;
      while (i < tokens.size() && tokens[i] != ")") ids.push_back(readInt3D(tokens, i, path.string()));
      expectToken3D(tokens, i, ")", path.string());
      faces.push_back(std::move(ids));
    }
  }
  return faces;
}

inline std::vector<int> parseFoamIntList3D(const std::filesystem::path& path) {
  const auto tokens = foamTokens3D(readTextFile3D(path));
  int n = 0;
  size_t i = findFoamListStart3D(tokens, path.string(), n);
  std::vector<int> values;
  values.reserve(static_cast<size_t>(n));
  for (int k = 0; k < n; ++k) values.push_back(readInt3D(tokens, i, path.string()));
  return values;
}

inline std::vector<OpenFoamPatchRange3D> parseFoamBoundary3D(
    const std::filesystem::path& path) {
  const auto tokens = foamTokens3D(readTextFile3D(path));
  int n = 0;
  size_t i = findFoamListStart3D(tokens, path.string(), n);
  std::vector<OpenFoamPatchRange3D> patches;
  patches.reserve(static_cast<size_t>(n));
  for (int p = 0; p < n; ++p) {
    if (i >= tokens.size() || tokens[i] == ")") break;
    OpenFoamPatchRange3D patch;
    patch.name = tokens[i++];
    expectToken3D(tokens, i, "{", path.string());
    int depth = 1;
    while (i < tokens.size() && depth > 0) {
      const std::string key = tokens[i++];
      if (key == "{") {
        ++depth;
      } else if (key == "}") {
        --depth;
      } else if (depth == 1 && key == "nFaces") {
        patch.nFaces = readInt3D(tokens, i, path.string());
      } else if (depth == 1 && key == "startFace") {
        patch.startFace = readInt3D(tokens, i, path.string());
      }
    }
    if (patch.nFaces < 0 || patch.startFace < 0) {
      throw std::runtime_error("invalid boundary patch range in " + path.string());
    }
    patches.push_back(patch);
  }
  return patches;
}

inline void applyOpenFoamPatches3D(Mesh3D& mesh,
                                   const std::vector<OpenFoamPatchRange3D>& ranges) {
  mesh.patches.clear();
  for (Face3D& f : mesh.faces) f.patch = -1;
  mesh.patches.reserve(ranges.size());
  for (size_t p = 0; p < ranges.size(); ++p) {
    Patch3D patch;
    patch.name = ranges[p].name;
    for (int k = 0; k < ranges[p].nFaces; ++k) {
      const int face = ranges[p].startFace + k;
      if (face < 0 || face >= static_cast<int>(mesh.faces.size())) {
        throw std::runtime_error("OpenFOAM boundary patch face out of range");
      }
      if (mesh.faces[face].internal()) {
        throw std::runtime_error("OpenFOAM boundary patch references internal face");
      }
      patch.faces.push_back(face);
      mesh.faces[face].patch = static_cast<int>(p);
    }
    mesh.patches.push_back(std::move(patch));
  }
}

}  // namespace detail

inline Mesh3D readOpenFoamPolyMesh3D(const std::filesystem::path& polyMeshDir,
                                     OpenFoamPolyMeshReadReport3D* report = nullptr) {
  const auto points = detail::parseFoamPoints3D(polyMeshDir / "points");
  const auto facePoints = detail::parseFoamFaces3D(polyMeshDir / "faces");
  const auto owners = detail::parseFoamIntList3D(polyMeshDir / "owner");
  const auto neighbours = detail::parseFoamIntList3D(polyMeshDir / "neighbour");
  const auto patchRanges = detail::parseFoamBoundary3D(polyMeshDir / "boundary");

  if (owners.size() != facePoints.size()) {
    throw std::runtime_error("OpenFOAM owner list size does not match face count");
  }
  if (neighbours.size() > owners.size()) {
    throw std::runtime_error("OpenFOAM neighbour list is larger than face count");
  }

  int nCells = 0;
  for (int owner : owners) nCells = std::max(nCells, owner + 1);
  for (int nei : neighbours) nCells = std::max(nCells, nei + 1);
  if (nCells <= 0) throw std::runtime_error("OpenFOAM polyMesh contains no cells");

  Mesh3D mesh;
  mesh.points = points;
  mesh.faces.resize(facePoints.size());
  mesh.cells.resize(static_cast<size_t>(nCells));
  mesh.patches = {{"xmin", {}}, {"xmax", {}}, {"ymin", {}},
                  {"ymax", {}}, {"zmin", {}}, {"zmax", {}}};

  std::vector<std::unordered_set<int>> cellPointSets(static_cast<size_t>(nCells));
  for (size_t f = 0; f < facePoints.size(); ++f) {
    const int owner = owners[f];
    if (owner < 0 || owner >= nCells) throw std::runtime_error("OpenFOAM owner out of range");
    Face3D face;
    face.points = facePoints[f];
    face.owner = owner;
    face.neighbour = f < neighbours.size() ? neighbours[f] : -1;
    if (face.neighbour >= nCells) throw std::runtime_error("OpenFOAM neighbour out of range");
    for (int pt : face.points) {
      if (pt < 0 || pt >= static_cast<int>(mesh.points.size())) {
        throw std::runtime_error("OpenFOAM face point out of range");
      }
      cellPointSets[static_cast<size_t>(owner)].insert(pt);
      if (face.neighbour >= 0) cellPointSets[static_cast<size_t>(face.neighbour)].insert(pt);
    }
    mesh.faces[f] = std::move(face);
    mesh.cells[static_cast<size_t>(owner)].faces.push_back(static_cast<int>(f));
    if (mesh.faces[f].neighbour >= 0) {
      mesh.cells[static_cast<size_t>(mesh.faces[f].neighbour)].faces.push_back(static_cast<int>(f));
    }
  }

  for (int c = 0; c < nCells; ++c) {
    mesh.cells[static_cast<size_t>(c)].points.assign(cellPointSets[static_cast<size_t>(c)].begin(),
                                                     cellPointSets[static_cast<size_t>(c)].end());
  }

  mesh.computeGeometry();
  detail::applyOpenFoamPatches3D(mesh, patchRanges);

  if (report) {
    report->points = static_cast<int>(mesh.points.size());
    report->faces = static_cast<int>(mesh.faces.size());
    report->owners = static_cast<int>(owners.size());
    report->neighbours = static_cast<int>(neighbours.size());
    report->cells = static_cast<int>(mesh.cells.size());
    report->patches = static_cast<int>(mesh.patches.size());
    int boundaryFaces = 0;
    for (const auto& p : mesh.patches) boundaryFaces += static_cast<int>(p.faces.size());
    report->boundaryFaces = boundaryFaces;
  }
  return mesh;
}

}  // namespace fvm
