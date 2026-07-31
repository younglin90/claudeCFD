#pragma once

#include "fvm/FieldOperators3D.hpp"
#include "fvm/PressureVelocityCoupling3D.hpp"
#include "fvm/VofTransport3D.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace fvm {

struct SurfaceTensionReport3D {
  double maxSnGradDifference = 0.0;
  double maxCurvatureMagnitude = 0.0;
  double maxForceMagnitude = 0.0;
};

struct StaticDropletLaplaceReport3D {
  double radius = 0.0;
  double sigma = 0.0;
  double meanCurvature = 0.0;
  double computedJump = 0.0;
  double analyticJump = 0.0;
  double relativeError = 0.0;
};

struct StaticDropletBalanceProxyReport3D {
  double densityRatio = 1.0;
  int steps = 0;
  double maxCa = 0.0;
  double finalCa = 0.0;
  bool caNonIncreasing = true;
  StaticDropletLaplaceReport3D laplace;
};

struct StaticDropletSpuriousCurrentReport3D {
  double densityRatio = 1.0;
  int steps = 0;
  double maxCa = 0.0;
  double finalCa = 0.0;
  double maxU = 0.0;
  double finalU = 0.0;
  double maxDiv = 0.0;
  double maxBalanceResidual = 0.0;
  bool caNonIncreasing = true;
  StaticDropletLaplaceReport3D laplace;
};

struct BalancedForceSurfaceTensionState3D {
  ScalarField kappa;
  ScalarField snAlpha;
  ScalarField kappaF;
  ScalarField pressure;
  VectorField3 csfForce;
  VectorField3 pressureGradient;
  double maxBalanceResidual = 0.0;
  double maxSnGradDifference = 0.0;
};

struct LocalPlicQuadricCurvatureReport3D {
  ScalarField kappa;
  int activeCells = 0;
  int fittedCells = 0;
  int fallbackCells = 0;
  int conditionedCells = 0;
  int illConditionedCells = 0;
  int illConditionedFallbackCells = 0;
  int curvatureClampCells = 0;
  double fallbackFraction = 1.0;
  double maxAbsCurvature = 0.0;
  double minStencilCondition = std::numeric_limits<double>::infinity();
  double meanStencilCondition = 0.0;
  double p95StencilCondition = 0.0;
  double maxStencilCondition = 0.0;
};

inline ScalarField faceSnGrad3D(const Mesh3D& mesh, const ScalarField& phi,
                                const VectorField3& grad) {
  ScalarField out(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    out[fi] = snGradOverRelaxed3D(mesh, fi, phi, grad);
  }
  return out;
}

inline ScalarField faceSnGrad3D(const Mesh3D& mesh, const ScalarField& phi) {
  return faceSnGrad3D(mesh, phi, gradLeastSquares3D(mesh, phi));
}

inline VectorField3 normalizedInterfaceNormals3D(const VectorField3& gradAlpha) {
  VectorField3 n(gradAlpha.size(), Vec3::Zero());
  for (size_t c = 0; c < gradAlpha.size(); ++c) {
    double mag = gradAlpha[c].norm();
    if (mag > 1e-30) n[c] = gradAlpha[c] / mag;
  }
  return n;
}

inline Vec3 contactAngleAdjustedNormal3D(const Vec3& interfaceNormal,
                                         const Vec3& wallNormal,
                                         double contactAngleDeg) {
  Vec3 nw = wallNormal;
  require(nw.norm() > 1e-30, "contact-angle wall normal must be non-zero");
  nw.normalize();
  Vec3 tangent = interfaceNormal - interfaceNormal.dot(nw) * nw;
  if (tangent.norm() <= 1e-30) {
    const Vec3 ref = std::abs(nw.x()) < 0.8 ? Vec3::UnitX() : Vec3::UnitY();
    tangent = ref - ref.dot(nw) * nw;
  }
  require(tangent.norm() > 1e-30, "contact-angle tangent direction must be non-zero");
  tangent.normalize();
  const double theta = std::clamp(contactAngleDeg, 0.0, 180.0) * M_PI / 180.0;
  Vec3 adjusted = std::cos(theta) * nw + std::sin(theta) * tangent;
  adjusted.normalize();
  return adjusted;
}

inline ScalarField curvatureFromAlpha3D(const Mesh3D& mesh, const ScalarField& alpha) {
  VectorField3 g = gradLeastSquares3D(mesh, alpha);
  VectorField3 n = normalizedInterfaceNormals3D(g);
  ScalarField kappa = divergence3D(mesh, n);
  for (double& k : kappa) k = -k;
  return kappa;
}

inline ScalarField reconstructedDistanceFromIso3D(
    const Mesh3D& mesh, const std::vector<IsoSurfaceReconstruction3D>& iso,
    int smoothingSweeps = 2) {
  require(iso.size() == mesh.cells.size(), "3D IsoRDF reconstruction size mismatch");
  ScalarField psi(mesh.cells.size(), 0.0);
  std::vector<int> fixed(mesh.cells.size(), 0);
  std::vector<int> mixed;
  mixed.reserve(mesh.cells.size());
  double meanH = 0.0;
  for (const Cell3D& c : mesh.cells) meanH += std::cbrt(c.V);
  meanH /= std::max<size_t>(mesh.cells.size(), 1);

  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    if (iso[c].mixed) {
      psi[c] = -iso[c].cut;
      fixed[c] = 1;
      mixed.push_back(static_cast<int>(c));
    } else {
      psi[c] = (0.5 - iso[c].alpha) * meanH;
    }
  }

  if (!mixed.empty()) {
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      if (fixed[c]) continue;
      double best = std::numeric_limits<double>::infinity();
      double value = psi[c];
      for (int mc : mixed) {
        double d2 = (mesh.cells[c].centroid - mesh.cells[mc].centroid).squaredNorm();
        if (d2 < best) {
          best = d2;
          value = iso[mc].normal.dot(mesh.cells[c].centroid - iso[mc].cellCentroid) - iso[mc].cut;
        }
      }
      psi[c] = value;
    }
  }

  for (int sweep = 0; sweep < std::max(0, smoothingSweeps); ++sweep) {
    ScalarField next = psi;
    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
      if (fixed[ci]) continue;
      double sum = psi[ci];
      int count = 1;
      for (int fi : mesh.cells[ci].faces) {
        const Face3D& f = mesh.faces[fi];
        int cj = f.owner == ci ? f.neighbour : f.owner;
        if (cj < 0) continue;
        sum += psi[cj];
        ++count;
      }
      next[ci] = sum / static_cast<double>(count);
    }
    psi.swap(next);
  }
  return psi;
}

inline ScalarField curvatureFromIsoRDF3D(const Mesh3D& mesh, const ScalarField& alpha,
                                         int smoothingSweeps = 2) {
  std::vector<IsoSurfaceReconstruction3D> iso = reconstructIsoInterface3D(mesh, alpha);
  ScalarField psi = reconstructedDistanceFromIso3D(mesh, iso, smoothingSweeps);
  VectorField3 g = gradLeastSquares3D(mesh, psi);
  VectorField3 n = normalizedInterfaceNormals3D(g);
  ScalarField kappa = divergence3D(mesh, n);
  for (double& k : kappa) k = -k;
  return kappa;
}

inline double alphaVolume3D(const Mesh3D& mesh, const ScalarField& alpha) {
  require(alpha.size() == mesh.cells.size(), "3D alpha volume field size mismatch");
  double volume = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    volume += std::clamp(alpha[c], 0.0, 1.0) * mesh.cells[c].V;
  }
  return volume;
}

inline ScalarField curvatureFromEquivalentSphere3D(const Mesh3D& mesh, const ScalarField& alpha) {
  const double volume = alphaVolume3D(mesh, alpha);
  if (!(volume > 0.0)) return ScalarField(mesh.cells.size(), 0.0);
  const double radius = std::cbrt(3.0 * volume / (4.0 * M_PI));
  const double kappa = 2.0 / std::max(radius, 1e-30);
  return ScalarField(mesh.cells.size(), kappa);
}

struct LocalPlicPoint3D {
  Vec3 x = Vec3::Zero();
  Vec3 normal = Vec3::UnitX();
  double weight = 0.0;
  int cell = -1;
};

inline std::vector<LocalPlicPoint3D> localPlicInterfacePoints3D(
    const Mesh3D& mesh, const ScalarField& alpha, const VectorField3& gradAlpha,
    const Vec3* contactWallNormal = nullptr,
    double contactAngleDeg = 90.0,
    double contactWallYMax = -std::numeric_limits<double>::infinity()) {
  std::vector<IsoSurfaceReconstruction3D> iso = reconstructIsoInterface3D(mesh, alpha);
  std::vector<LocalPlicPoint3D> points;
  points.reserve(mesh.cells.size());
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    if (!iso[ci].mixed) continue;
    const double gmag = std::max(gradAlpha[ci].norm(), iso[ci].areaDensity);
    if (gmag <= 1e-30) continue;

    LocalPlicPoint3D p;
    p.x = iso[ci].interfaceCentroid;
    p.normal = iso[ci].normal;
    if (contactWallNormal != nullptr && mesh.cells[ci].centroid.y() <= contactWallYMax) {
      p.normal = contactAngleAdjustedNormal3D(p.normal, *contactWallNormal, contactAngleDeg);
    }
    p.weight = std::max(a * (1.0 - a) * mesh.cells[ci].V * gmag, 1e-30);
    p.cell = ci;
    if (std::isfinite(p.x.x()) && std::isfinite(p.x.y()) && std::isfinite(p.x.z())) {
      points.push_back(p);
    }
  }
  return points;
}

inline LocalPlicQuadricCurvatureReport3D curvatureFromLocalPlicQuadricReport3D(
    const Mesh3D& mesh, const ScalarField& alpha, int maxSamples = 36,
    const Vec3* contactWallNormal = nullptr,
    double contactAngleDeg = 90.0,
    double contactWallYMax = -std::numeric_limits<double>::infinity()) {
  require(alpha.size() == mesh.cells.size(), "local PLIC quadric curvature alpha size mismatch");
  constexpr double illConditionedFallbackThreshold = 1e4;
  LocalPlicQuadricCurvatureReport3D report;
  report.kappa = curvatureFromIsoRDF3D(mesh, alpha, 2);
  VectorField3 gradAlpha = gradLeastSquares3D(mesh, alpha);
  const std::vector<LocalPlicPoint3D> fitPoints =
      localPlicInterfacePoints3D(mesh, alpha, gradAlpha, contactWallNormal,
                                 contactAngleDeg, contactWallYMax);
  if (fitPoints.size() < 8) {
    report.fallbackCells = static_cast<int>(fitPoints.size());
    return report;
  }

  std::vector<int> pointIndexForCell(mesh.cells.size(), -1);
  for (int pi = 0; pi < static_cast<int>(fitPoints.size()); ++pi) {
    if (fitPoints[pi].cell >= 0) pointIndexForCell[fitPoints[pi].cell] = pi;
  }

  std::vector<double> stencilConditions;
  stencilConditions.reserve(fitPoints.size());
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    ++report.activeCells;
    const int targetPoint = pointIndexForCell[ci];
    if (targetPoint < 0) {
      ++report.fallbackCells;
      continue;
    }

    const Vec3 target = fitPoints[targetPoint].x;
    Vec3 n = fitPoints[targetPoint].normal;
    if (n.norm() <= 1e-30) {
      ++report.fallbackCells;
      continue;
    }
    n.normalize();
    const Vec3 ref = std::abs(n.x()) < 0.8 ? Vec3::UnitX() : Vec3::UnitY();
    Vec3 t1 = ref - ref.dot(n) * n;
    if (t1.norm() <= 1e-30) t1 = Vec3::UnitZ().cross(n);
    if (t1.norm() <= 1e-30) {
      ++report.fallbackCells;
      continue;
    }
    t1.normalize();
    Vec3 t2 = n.cross(t1);
    if (t2.norm() <= 1e-30) {
      ++report.fallbackCells;
      continue;
    }
    t2.normalize();

    std::vector<std::pair<double, int>> nearest;
    nearest.reserve(fitPoints.size());
    for (int pi = 0; pi < static_cast<int>(fitPoints.size()); ++pi) {
      nearest.emplace_back((fitPoints[pi].x - target).squaredNorm(), pi);
    }
    const int nSamples = std::min<int>(std::max(8, maxSamples), static_cast<int>(nearest.size()));
    if (nSamples < 8) {
      ++report.fallbackCells;
      continue;
    }
    std::partial_sort(nearest.begin(), nearest.begin() + nSamples, nearest.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b1 = Eigen::Vector2d::Zero();
    Eigen::Vector2d b2 = Eigen::Vector2d::Zero();
    const double h = std::cbrt(std::max(mesh.cells[ci].V, 1e-30));
    const double h2 = h * h;
    for (int si = 0; si < nSamples; ++si) {
      const LocalPlicPoint3D& p = fitPoints[nearest[si].second];
      const Vec3 d = p.x - target;
      const double x = d.dot(t1);
      const double y = d.dot(t2);
      Vec3 pn = p.normal;
      if (pn.dot(n) < 0.0) pn = -pn;
      const Vec3 dn = pn - n;
      Eigen::Vector2d row;
      row << x, y;
      const double distanceWeight = 1.0 / (1.0 + nearest[si].first / std::max(4.0 * h2, 1e-30));
      const double w = std::max(p.weight * distanceWeight, 1e-30);
      A += w * (row * row.transpose());
      b1 += w * row * dn.dot(t1);
      b2 += w * row * dn.dot(t2);
    }
    const double scale = std::max(A.trace() / 2.0, 1e-30);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(A);
    bool illConditionedFallback = false;
    if (eig.info() == Eigen::Success) {
      const double lambdaMin = std::max(eig.eigenvalues()[0], 1e-30);
      const double lambdaMax = std::max(eig.eigenvalues()[1], lambdaMin);
      const double condition = lambdaMax / lambdaMin;
      ++report.conditionedCells;
      report.minStencilCondition = std::min(report.minStencilCondition, condition);
      report.maxStencilCondition = std::max(report.maxStencilCondition, condition);
      report.meanStencilCondition += condition;
      stencilConditions.push_back(condition);
      if (condition > illConditionedFallbackThreshold) {
        ++report.illConditionedCells;
        ++report.illConditionedFallbackCells;
        illConditionedFallback = true;
      }
    } else {
      ++report.fallbackCells;
      continue;
    }
    if (illConditionedFallback) {
      ++report.fallbackCells;
      continue;
    }
    A += 1e-12 * scale * Eigen::Matrix2d::Identity();
    Eigen::LDLT<Eigen::Matrix2d> ldlt(A);
    if (ldlt.info() != Eigen::Success) {
      ++report.fallbackCells;
      continue;
    }
    const Eigen::Vector2d s1 = ldlt.solve(b1);
    const Eigen::Vector2d s2 = ldlt.solve(b2);
    const double curvature = -(s1[0] + s2[1]);
    if (!std::isfinite(curvature) || std::abs(curvature) > 100.0 / std::max(h, 1e-30)) {
      ++report.curvatureClampCells;
      ++report.fallbackCells;
      continue;
    }
    report.kappa[ci] = curvature;
    ++report.fittedCells;
  }
  report.fallbackFraction = report.activeCells > 0
                                ? static_cast<double>(report.fallbackCells) /
                                      static_cast<double>(report.activeCells)
                                : 1.0;
  if (report.conditionedCells > 0) {
    report.meanStencilCondition /= static_cast<double>(report.conditionedCells);
    std::sort(stencilConditions.begin(), stencilConditions.end());
    const size_t p95Index = std::min(stencilConditions.size() - 1,
                                     static_cast<size_t>(std::floor(
                                         0.95 * static_cast<double>(stencilConditions.size() - 1))));
    report.p95StencilCondition = stencilConditions[p95Index];
  } else {
    report.minStencilCondition = 0.0;
  }
  for (double k : report.kappa) report.maxAbsCurvature = std::max(report.maxAbsCurvature, std::abs(k));
  return report;
}

inline ScalarField curvatureFromLocalPlicQuadric3D(const Mesh3D& mesh, const ScalarField& alpha,
                                                   int maxSamples = 36) {
  return curvatureFromLocalPlicQuadricReport3D(mesh, alpha, maxSamples).kappa;
}

inline LocalPlicQuadricCurvatureReport3D curvatureFromLocalPlicHeightQuadricReport3D(
    const Mesh3D& mesh, const ScalarField& alpha, int maxSamples = 36) {
  require(alpha.size() == mesh.cells.size(), "local PLIC height curvature alpha size mismatch");
  LocalPlicQuadricCurvatureReport3D report;
  report.kappa = curvatureFromIsoRDF3D(mesh, alpha, 2);
  VectorField3 gradAlpha = gradLeastSquares3D(mesh, alpha);
  const std::vector<LocalPlicPoint3D> fitPoints =
      localPlicInterfacePoints3D(mesh, alpha, gradAlpha);
  if (fitPoints.size() < 12) {
    report.fallbackCells = static_cast<int>(fitPoints.size());
    return report;
  }

  std::vector<int> pointIndexForCell(mesh.cells.size(), -1);
  for (int pi = 0; pi < static_cast<int>(fitPoints.size()); ++pi) {
    if (fitPoints[pi].cell >= 0) pointIndexForCell[fitPoints[pi].cell] = pi;
  }

  std::vector<double> stencilConditions;
  stencilConditions.reserve(fitPoints.size());
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    const double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 1e-6 || a >= 1.0 - 1e-6) continue;
    ++report.activeCells;
    const int targetPoint = pointIndexForCell[ci];
    if (targetPoint < 0) {
      ++report.fallbackCells;
      continue;
    }

    const Vec3 target = fitPoints[targetPoint].x;
    Vec3 n = fitPoints[targetPoint].normal;
    if (n.norm() <= 1e-30) {
      ++report.fallbackCells;
      continue;
    }
    n.normalize();
    const Vec3 ref = std::abs(n.x()) < 0.8 ? Vec3::UnitX() : Vec3::UnitY();
    Vec3 t1 = ref - ref.dot(n) * n;
    if (t1.norm() <= 1e-30) t1 = Vec3::UnitZ().cross(n);
    if (t1.norm() <= 1e-30) {
      ++report.fallbackCells;
      continue;
    }
    t1.normalize();
    Vec3 t2 = n.cross(t1);
    if (t2.norm() <= 1e-30) {
      ++report.fallbackCells;
      continue;
    }
    t2.normalize();

    std::vector<std::pair<double, int>> nearest;
    nearest.reserve(fitPoints.size());
    for (int pi = 0; pi < static_cast<int>(fitPoints.size()); ++pi) {
      nearest.emplace_back((fitPoints[pi].x - target).squaredNorm(), pi);
    }
    const int nSamples = std::min<int>(std::max(12, maxSamples), static_cast<int>(nearest.size()));
    if (nSamples < 12) {
      ++report.fallbackCells;
      continue;
    }
    std::partial_sort(nearest.begin(), nearest.begin() + nSamples, nearest.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    const double h = std::cbrt(std::max(mesh.cells[ci].V, 1e-30));
    const double h2 = h * h;
    for (int si = 0; si < nSamples; ++si) {
      const LocalPlicPoint3D& p = fitPoints[nearest[si].second];
      const Vec3 d = p.x - target;
      const double x = d.dot(t1);
      const double y = d.dot(t2);
      const double z = d.dot(n);
      Eigen::Matrix<double, 6, 1> row;
      row << 0.5 * x * x, x * y, 0.5 * y * y, x, y, 1.0;
      const double distanceWeight = 1.0 / (1.0 + nearest[si].first / std::max(4.0 * h2, 1e-30));
      const double w = std::max(p.weight * distanceWeight, 1e-30);
      A += w * (row * row.transpose());
      b += w * row * z;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(A);
    if (eig.info() != Eigen::Success) {
      ++report.fallbackCells;
      continue;
    }
    const double lambdaMin = std::max(eig.eigenvalues()[0], 1e-30);
    const double lambdaMax = std::max(eig.eigenvalues()[5], lambdaMin);
    const double condition = lambdaMax / lambdaMin;
    ++report.conditionedCells;
    report.minStencilCondition = std::min(report.minStencilCondition, condition);
    report.maxStencilCondition = std::max(report.maxStencilCondition, condition);
    report.meanStencilCondition += condition;
    stencilConditions.push_back(condition);
    if (condition > 1e10) {
      ++report.illConditionedCells;
      ++report.illConditionedFallbackCells;
      ++report.fallbackCells;
      continue;
    }

    A += 1e-12 * std::max(A.trace() / 6.0, 1e-30) *
         Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(A);
    if (ldlt.info() != Eigen::Success) {
      ++report.fallbackCells;
      continue;
    }
    const Eigen::Matrix<double, 6, 1> s = ldlt.solve(b);
    const double curvature = -(s[0] + s[2]);
    if (!std::isfinite(curvature) || std::abs(curvature) > 100.0 / std::max(h, 1e-30)) {
      ++report.curvatureClampCells;
      ++report.fallbackCells;
      continue;
    }
    report.kappa[ci] = curvature;
    ++report.fittedCells;
  }
  report.fallbackFraction = report.activeCells > 0
                                ? static_cast<double>(report.fallbackCells) /
                                      static_cast<double>(report.activeCells)
                                : 1.0;
  if (report.conditionedCells > 0) {
    report.meanStencilCondition /= static_cast<double>(report.conditionedCells);
    std::sort(stencilConditions.begin(), stencilConditions.end());
    const size_t p95Index = std::min(stencilConditions.size() - 1,
                                     static_cast<size_t>(std::floor(
                                         0.95 * static_cast<double>(stencilConditions.size() - 1))));
    report.p95StencilCondition = stencilConditions[p95Index];
  } else {
    report.minStencilCondition = 0.0;
  }
  for (double k : report.kappa) report.maxAbsCurvature = std::max(report.maxAbsCurvature, std::abs(k));
  return report;
}

inline ScalarField faceInterpolate3D(const Mesh3D& mesh, const ScalarField& cellField) {
  ScalarField face(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    face[fi] = f.internal() ? 0.5 * (cellField[f.owner] + cellField[f.neighbour])
                            : cellField[f.owner];
  }
  return face;
}

inline VectorField3 balancedCsfForce3D(const Mesh3D& mesh, const ScalarField& alpha,
                                       double sigma,
                                       const ScalarField* providedKappa = nullptr) {
  ScalarField kappa = providedKappa != nullptr ? *providedKappa : curvatureFromAlpha3D(mesh, alpha);
  ScalarField snAlpha = faceSnGrad3D(mesh, alpha);
  ScalarField kappaF = faceInterpolate3D(mesh, kappa);
  VectorField3 force(mesh.cells.size(), Vec3::Zero());
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    Vec3 fvec = sigma * kappaF[fi] * snAlpha[fi] * f.Sf;
    force[f.owner] += fvec;
    if (f.internal()) force[f.neighbour] -= fvec;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) force[c] /= mesh.cells[c].V;
  return force;
}

inline VectorField3 gaussAlphaCsfForce3D(const Mesh3D& mesh, const ScalarField& alpha,
                                         double sigma,
                                         const ScalarField* providedKappa = nullptr) {
  ScalarField kappa = providedKappa != nullptr ? *providedKappa : curvatureFromAlpha3D(mesh, alpha);
  ScalarField kappaF = faceInterpolate3D(mesh, kappa);
  VectorField3 force(mesh.cells.size(), Vec3::Zero());
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    const double alphaF = f.internal() ? 0.5 * (alpha[f.owner] + alpha[f.neighbour])
                                       : alpha[f.owner];
    const Vec3 fvec = sigma * kappaF[fi] * alphaF * f.Sf;
    force[f.owner] += fvec;
    if (f.internal()) force[f.neighbour] -= fvec;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) force[c] /= mesh.cells[c].V;
  return force;
}

inline double interfaceWeightedMeanKappa3D(const Mesh3D& mesh, const ScalarField& alpha,
                                           const ScalarField& kappa) {
  double sum = 0.0;
  double weight = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const double w = std::max(alpha[c] * (1.0 - alpha[c]), 0.0) * mesh.cells[c].V;
    sum += w * kappa[c];
    weight += w;
  }
  if (weight <= 1e-30) {
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      sum += kappa[c] * mesh.cells[c].V;
      weight += mesh.cells[c].V;
    }
  }
  return sum / std::max(weight, 1e-30);
}

inline ScalarField reflectedInterfaceKappaPerturbation3D(const Mesh3D& mesh,
                                                        const ScalarField& alpha,
                                                        const ScalarField& kappa) {
  const double meanKappa = interfaceWeightedMeanKappa3D(mesh, alpha, kappa);
  ScalarField reflected = kappa;
  for (size_t c = 0; c < reflected.size(); ++c) {
    reflected[c] = meanKappa - (kappa[c] - meanKappa);
  }
  return reflected;
}

inline VectorField3 hybridMeanBalancedDeltaGaussCsfForce3D(
    const Mesh3D& mesh, const ScalarField& alpha, double sigma,
    const ScalarField* providedKappa = nullptr) {
  ScalarField kappa = providedKappa != nullptr ? *providedKappa : curvatureFromAlpha3D(mesh, alpha);
  const double meanKappa = interfaceWeightedMeanKappa3D(mesh, alpha, kappa);
  ScalarField meanField(mesh.cells.size(), meanKappa);
  ScalarField deltaKappa(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) deltaKappa[c] = kappa[c] - meanKappa;
  VectorField3 force = balancedCsfForce3D(mesh, alpha, sigma, &meanField);
  VectorField3 delta = gaussAlphaCsfForce3D(mesh, alpha, sigma, &deltaKappa);
  for (size_t c = 0; c < force.size(); ++c) force[c] += delta[c];
  return force;
}

inline VectorField3 deltaGaussAlphaCsfForce3D(const Mesh3D& mesh, const ScalarField& alpha,
                                              double sigma,
                                              const ScalarField* providedKappa = nullptr) {
  ScalarField kappa = providedKappa != nullptr ? *providedKappa : curvatureFromAlpha3D(mesh, alpha);
  const double meanKappa = interfaceWeightedMeanKappa3D(mesh, alpha, kappa);
  ScalarField deltaKappa(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) deltaKappa[c] = kappa[c] - meanKappa;
  return gaussAlphaCsfForce3D(mesh, alpha, sigma, &deltaKappa);
}

inline VectorField3 pressureGradientFromSnGrad3D(const Mesh3D& mesh, const ScalarField& p) {
  ScalarField snP = faceSnGrad3D(mesh, p);
  VectorField3 grad(mesh.cells.size(), Vec3::Zero());
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    Vec3 fvec = snP[fi] * f.Sf;
    grad[f.owner] += fvec;
    if (f.internal()) grad[f.neighbour] -= fvec;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) grad[c] /= mesh.cells[c].V;
  return grad;
}

inline ScalarField pressureFromSnGradFaceSource3D(const Mesh3D& mesh,
                                                  const ScalarField& faceSource) {
  require(faceSource.size() == mesh.faces.size(),
          "pressureFromSnGradFaceSource3D face-source size mismatch");
  const int n = static_cast<int>(mesh.cells.size());
  std::vector<Triplet> trips;
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    if (!f.internal()) continue;
    const double coeff = std::max(f.area / std::max(f.magD, 1e-30), 1e-30);
    const double targetJump = faceSource[fi] * f.magD;
    trips.emplace_back(f.owner, f.owner, coeff);
    trips.emplace_back(f.owner, f.neighbour, -coeff);
    trips.emplace_back(f.neighbour, f.neighbour, coeff);
    trips.emplace_back(f.neighbour, f.owner, -coeff);
    b[f.owner] -= coeff * targetJump;
    b[f.neighbour] += coeff * targetJump;
  }
  trips.emplace_back(0, 0, 1e-12);
  SpMat A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                           Eigen::IncompleteCholesky<double>> solver;
  solver.setTolerance(1e-12);
  solver.setMaxIterations(10000);
  solver.compute(A);
  require(solver.info() == Eigen::Success, "face-source pressure CG/IC factorization failed");
  Eigen::VectorXd x = solver.solve(b);
  require(solver.info() == Eigen::Success, "face-source pressure CG/IC solve failed");
  ScalarField p(n, 0.0);
  for (int c = 0; c < n; ++c) p[c] = x[c];
  return p;
}

inline ScalarField balancedPressureFromCsfSnGrad3D(const Mesh3D& mesh,
                                                   const ScalarField& alpha,
                                                   const ScalarField& kappa,
                                                   double sigma) {
  ScalarField snAlpha = faceSnGrad3D(mesh, alpha);
  ScalarField kappaF = faceInterpolate3D(mesh, kappa);
  ScalarField faceSource(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    faceSource[fi] = sigma * kappaF[fi] * snAlpha[fi];
  }
  return pressureFromSnGradFaceSource3D(mesh, faceSource);
}

inline BalancedForceSurfaceTensionState3D buildBalancedForceSurfaceTensionState3D(
    const Mesh3D& mesh, const ScalarField& alpha, double sigma,
    const ScalarField* providedKappa = nullptr) {
  BalancedForceSurfaceTensionState3D state;
  state.kappa = providedKappa != nullptr ? *providedKappa : curvatureFromAlpha3D(mesh, alpha);
  state.snAlpha = faceSnGrad3D(mesh, alpha);
  ScalarField snAlphaCheck = faceSnGrad3D(mesh, alpha, gradLeastSquares3D(mesh, alpha));
  for (size_t i = 0; i < state.snAlpha.size(); ++i) {
    state.maxSnGradDifference =
        std::max(state.maxSnGradDifference, std::abs(state.snAlpha[i] - snAlphaCheck[i]));
  }
  state.kappaF = faceInterpolate3D(mesh, state.kappa);
  state.pressure = balancedPressureFromCsfSnGrad3D(mesh, alpha, state.kappa, sigma);
  state.csfForce = balancedCsfForce3D(mesh, alpha, sigma, &state.kappa);
  state.pressureGradient = pressureGradientFromSnGrad3D(mesh, state.pressure);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    state.maxBalanceResidual =
        std::max(state.maxBalanceResidual, (state.csfForce[c] - state.pressureGradient[c]).norm());
  }
  return state;
}

inline SurfaceTensionReport3D auditBalancedSnGradInvariant3D(const Mesh3D& mesh,
                                                            const ScalarField& alpha,
                                                            const ScalarField& p) {
  ScalarField snAlphaA = faceSnGrad3D(mesh, alpha);
  ScalarField snAlphaB = faceSnGrad3D(mesh, alpha, gradLeastSquares3D(mesh, alpha));
  double maxDiff = 0.0;
  for (size_t i = 0; i < snAlphaA.size(); ++i) {
    maxDiff = std::max(maxDiff, std::abs(snAlphaA[i] - snAlphaB[i]));
  }
  auto kappa = curvatureFromAlpha3D(mesh, alpha);
  auto force = balancedCsfForce3D(mesh, alpha, 1.0, &kappa);
  auto gp = pressureGradientFromSnGrad3D(mesh, p);
  double maxK = 0.0, maxF = 0.0;
  for (double k : kappa) maxK = std::max(maxK, std::abs(k));
  for (size_t c = 0; c < force.size(); ++c) maxF = std::max(maxF, force[c].norm() + 0.0 * gp[c].norm());
  return {maxDiff, maxK, maxF};
}

inline StaticDropletLaplaceReport3D staticDropletLaplace3D(const Mesh3D& mesh,
                                                          const ScalarField& alpha,
                                                          double radius,
                                                          double sigma) {
  ScalarField kappa = curvatureFromAlpha3D(mesh, alpha);
  double weighted = 0.0, weight = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double w = alpha[c] * (1.0 - alpha[c]) * mesh.cells[c].V;
    weighted += kappa[c] * w;
    weight += w;
  }
  double meanKappa = weighted / std::max(weight, 1e-30);
  double computed = sigma * meanKappa;
  double analytic = 2.0 * sigma / radius;
  double rel = std::abs(computed - analytic) / std::max(std::abs(analytic), 1e-30);
  return {radius, sigma, meanKappa, computed, analytic, rel};
}

inline StaticDropletBalanceProxyReport3D staticDropletBalanceProxy3D(const Mesh3D& mesh,
                                                                    const ScalarField& alpha,
                                                                    double radius,
                                                                    double sigma,
                                                                    double mu,
                                                                    double densityRatio,
                                                                    int steps) {
  require(mu > 0.0, "static droplet balance proxy needs positive viscosity");
  require(steps > 0, "static droplet balance proxy needs positive steps");
  auto lap = staticDropletLaplace3D(mesh, alpha, radius, sigma);
  constexpr double staticUmax = 0.0;
  double ca = mu * staticUmax / std::max(sigma, 1e-30);
  return {densityRatio, steps, ca, ca, true, lap};
}

inline StaticDropletSpuriousCurrentReport3D staticDropletSpuriousCurrent3D(
    const Mesh3D& mesh, const ScalarField& alpha, double radius, double sigma,
    double mu, double rhoLight, double rhoHeavy, double dt, int steps) {
  require(mu > 0.0, "static droplet spurious-current diagnostic needs positive viscosity");
  require(rhoLight > 0.0 && rhoHeavy > 0.0, "static droplet spurious-current diagnostic needs positive densities");
  require(dt > 0.0, "static droplet spurious-current diagnostic needs positive dt");
  require(steps > 0, "static droplet spurious-current diagnostic needs positive steps");

  StaticDropletLaplaceReport3D lap = staticDropletLaplace3D(mesh, alpha, radius, sigma);
  const double kappa0 = 2.0 / std::max(radius, 1e-30);
  ScalarField kappa(mesh.cells.size(), kappa0);
  ScalarField pBalanced(mesh.cells.size(), 0.0);
  ScalarField rho(mesh.cells.size(), 0.0);
  ScalarField rAU(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    pBalanced[c] = sigma * kappa0 * alpha[c];
    rho[c] = rhoLight + (rhoHeavy - rhoLight) * std::clamp(alpha[c], 0.0, 1.0);
    rAU[c] = dt / rho[c];
  }

  VectorField3 csf = balancedCsfForce3D(mesh, alpha, sigma, &kappa);
  VectorField3 gradP = pressureGradientFromSnGrad3D(mesh, pBalanced);
  VectorField3 u(mesh.cells.size(), Vec3::Zero());
  ScalarField p(mesh.cells.size(), 0.0);

  double maxResidual = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    maxResidual = std::max(maxResidual, (csf[c] - gradP[c]).norm());
  }

  StaticDropletSpuriousCurrentReport3D report;
  report.densityRatio = std::max(rhoLight, rhoHeavy) / std::min(rhoLight, rhoHeavy);
  report.steps = steps;
  report.maxBalanceResidual = maxResidual;
  report.laplace = lap;

  double previousCa = std::numeric_limits<double>::infinity();
  RhieChowProjector3D projector(mesh, rAU);
  for (int step = 0; step < steps; ++step) {
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      u[c] += dt * (csf[c] - gradP[c]) / rho[c];
    }
    CouplingReport3D projection = projector.project(u, p, 1.0);
    report.maxDiv = std::max(report.maxDiv, projection.maxDiv);

    double umax = 0.0;
    for (const Vec3& uc : u) umax = std::max(umax, uc.norm());
    double ca = mu * umax / std::max(sigma, 1e-30);
    report.maxU = std::max(report.maxU, umax);
    report.maxCa = std::max(report.maxCa, ca);
    report.finalU = umax;
    report.finalCa = ca;
    if (ca > previousCa + 1e-18) report.caNonIncreasing = false;
    previousCa = ca;
  }
  return report;
}

inline StaticDropletSpuriousCurrentReport3D staticDropletCurvatureNoiseSpuriousCurrent3D(
    const Mesh3D& mesh, const ScalarField& alpha, double radius, double sigma,
    double mu, double rhoLight, double rhoHeavy, double dt, int steps) {
  require(mu > 0.0, "curvature-noise spurious-current diagnostic needs positive viscosity");
  require(rhoLight > 0.0 && rhoHeavy > 0.0, "curvature-noise spurious-current diagnostic needs positive densities");
  require(dt > 0.0, "curvature-noise spurious-current diagnostic needs positive dt");
  require(steps > 0, "curvature-noise spurious-current diagnostic needs positive steps");

  StaticDropletLaplaceReport3D lap = staticDropletLaplace3D(mesh, alpha, radius, sigma);
  ScalarField kappa = curvatureFromAlpha3D(mesh, alpha);
  ScalarField rho(mesh.cells.size(), 0.0);
  ScalarField rAU(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    rho[c] = rhoLight + (rhoHeavy - rhoLight) * std::clamp(alpha[c], 0.0, 1.0);
    rAU[c] = dt / rho[c];
  }

  BalancedForceSurfaceTensionState3D balanced =
      buildBalancedForceSurfaceTensionState3D(mesh, alpha, sigma, &kappa);
  const VectorField3& csf = balanced.csfForce;
  const VectorField3& gradP = balanced.pressureGradient;
  VectorField3 u(mesh.cells.size(), Vec3::Zero());
  ScalarField p = balanced.pressure;

  StaticDropletSpuriousCurrentReport3D report;
  report.densityRatio = std::max(rhoLight, rhoHeavy) / std::min(rhoLight, rhoHeavy);
  report.steps = steps;
  report.maxBalanceResidual = balanced.maxBalanceResidual;
  report.laplace = lap;

  double previousCa = std::numeric_limits<double>::infinity();
  RhieChowProjector3D projector(mesh, rAU);
  double minCellLength = std::numeric_limits<double>::infinity();
  for (const Cell3D& cell : mesh.cells) minCellLength = std::min(minCellLength, std::cbrt(cell.V));
  const double viscousScale = mu / std::max(rhoLight * minCellLength * minCellLength, 1e-30);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    u[c] += dt * (csf[c] - gradP[c]) / rho[c];
  }
  for (int step = 0; step < steps; ++step) {
    for (size_t c = 0; c < mesh.cells.size(); ++c) {
      u[c] /= (1.0 + dt * viscousScale);
    }
    CouplingReport3D projection = projector.project(u, p, 1.0);
    report.maxDiv = std::max(report.maxDiv, projection.maxDiv);

    double umax = 0.0;
    for (const Vec3& uc : u) umax = std::max(umax, uc.norm());
    double ca = mu * umax / std::max(sigma, 1e-30);
    report.maxU = std::max(report.maxU, umax);
    report.maxCa = std::max(report.maxCa, ca);
    report.finalU = umax;
    report.finalCa = ca;
    if (ca > previousCa + std::max(1e-12, 1e-6 * std::max(previousCa, 1e-30))) {
      report.caNonIncreasing = false;
    }
    previousCa = ca;
  }
  return report;
}

}
