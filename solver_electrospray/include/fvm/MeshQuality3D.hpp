#pragma once

#include "fvm/Mesh3D.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace fvm {

struct MeshQualityReport3D {
  int cells = 0;
  int faces = 0;
  int internalFaces = 0;
  int nonPositiveVolumeCount = 0;
  int zeroAreaFaceCount = 0;
  bool finite = true;
  double minVolume = std::numeric_limits<double>::infinity();
  double maxVolume = 0.0;
  double minFaceArea = std::numeric_limits<double>::infinity();
  double maxFaceArea = 0.0;
  double maxNonOrthogonalityDeg = 0.0;
  double meanNonOrthogonalityDeg = 0.0;
  double maxSkewness = 0.0;
  double maxAspectRatio = 0.0;
};

inline double clampUnit(double x) {
  return std::max(-1.0, std::min(1.0, x));
}

inline double cellAspectRatio3D(const Mesh3D& mesh, const Cell3D& cell) {
  double minLen = std::numeric_limits<double>::infinity();
  double maxLen = 0.0;
  for (size_t i = 0; i < cell.points.size(); ++i) {
    for (size_t j = i + 1; j < cell.points.size(); ++j) {
      const double d = (mesh.points[cell.points[i]] - mesh.points[cell.points[j]]).norm();
      if (d > 1e-30) {
        minLen = std::min(minLen, d);
        maxLen = std::max(maxLen, d);
      }
    }
  }
  if (!std::isfinite(minLen) || minLen <= 0.0) return std::numeric_limits<double>::infinity();
  return maxLen / minLen;
}

inline MeshQualityReport3D meshQualityReport3D(const Mesh3D& mesh) {
  MeshQualityReport3D r;
  r.cells = static_cast<int>(mesh.cells.size());
  r.faces = static_cast<int>(mesh.faces.size());

  for (const Cell3D& c : mesh.cells) {
    r.finite = r.finite && std::isfinite(c.V) &&
               std::isfinite(c.centroid.x()) && std::isfinite(c.centroid.y()) && std::isfinite(c.centroid.z());
    if (!(c.V > 0.0)) ++r.nonPositiveVolumeCount;
    r.minVolume = std::min(r.minVolume, c.V);
    r.maxVolume = std::max(r.maxVolume, c.V);
    r.maxAspectRatio = std::max(r.maxAspectRatio, cellAspectRatio3D(mesh, c));
  }

  double nonOrthoSum = 0.0;
  for (const Face3D& f : mesh.faces) {
    const double area = f.Sf.norm();
    r.finite = r.finite && std::isfinite(area) &&
               std::isfinite(f.centroid.x()) && std::isfinite(f.centroid.y()) && std::isfinite(f.centroid.z());
    if (!(area > 0.0)) ++r.zeroAreaFaceCount;
    r.minFaceArea = std::min(r.minFaceArea, area);
    r.maxFaceArea = std::max(r.maxFaceArea, area);

    if (f.internal()) {
      ++r.internalFaces;
      const double dNorm = f.d.norm();
      if (area > 1e-30 && dNorm > 1e-30) {
        const double cosTheta = std::abs(f.Sf.dot(f.d)) / (area * dNorm);
        const double angle = std::acos(clampUnit(cosTheta)) * 180.0 / M_PI;
        r.maxNonOrthogonalityDeg = std::max(r.maxNonOrthogonalityDeg, angle);
        nonOrthoSum += angle;

        const Vec3 co = mesh.cells[f.owner].centroid;
        const double t = (f.centroid - co).dot(f.d) / std::max(f.d.squaredNorm(), 1e-30);
        const Vec3 projected = co + t * f.d;
        r.maxSkewness = std::max(r.maxSkewness, (f.centroid - projected).norm() / dNorm);
      }
    }
  }

  if (r.internalFaces > 0) {
    r.meanNonOrthogonalityDeg = nonOrthoSum / static_cast<double>(r.internalFaces);
  }
  if (!std::isfinite(r.minVolume)) r.minVolume = 0.0;
  if (!std::isfinite(r.minFaceArea)) r.minFaceArea = 0.0;
  r.finite = r.finite && std::isfinite(r.maxAspectRatio) && std::isfinite(r.maxSkewness) &&
             std::isfinite(r.maxNonOrthogonalityDeg);
  return r;
}

inline std::string meshQualityMarkdown3D(const MeshQualityReport3D& r) {
  std::ostringstream os;
  os << std::setprecision(12);
  os << "- cells: " << r.cells << "\n";
  os << "- faces: " << r.faces << "\n";
  os << "- internal_faces: " << r.internalFaces << "\n";
  os << "- finite: " << (r.finite ? "true" : "false") << "\n";
  os << "- non_positive_volume_count: " << r.nonPositiveVolumeCount << "\n";
  os << "- zero_area_face_count: " << r.zeroAreaFaceCount << "\n";
  os << "- min_volume: " << r.minVolume << "\n";
  os << "- max_volume: " << r.maxVolume << "\n";
  os << "- min_face_area: " << r.minFaceArea << "\n";
  os << "- max_face_area: " << r.maxFaceArea << "\n";
  os << "- max_non_orthogonality_deg: " << r.maxNonOrthogonalityDeg << "\n";
  os << "- mean_non_orthogonality_deg: " << r.meanNonOrthogonalityDeg << "\n";
  os << "- max_skewness: " << r.maxSkewness << "\n";
  os << "- max_aspect_ratio: " << r.maxAspectRatio << "\n";
  return os.str();
}

}  // namespace fvm
