#pragma once

#include "fvm/FieldOperators3D.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace fvm {

enum class VofAdvectionScheme3D {
  AlgebraicTVD,
  IsoAdvector
};

struct VofTransportOptions3D {
  VofAdvectionScheme3D scheme = VofAdvectionScheme3D::AlgebraicTVD;
  double tvdBlend = 1.0;
  double compression = 0.0;
  int correctionSweeps = 3;
  double postSharpening = 0.0;
  int postSharpeningSweeps = 0;
  const ScalarField* boundaryAlpha = nullptr;
};

struct VofTransportReport3D {
  double initialMass = 0.0;
  double finalMass = 0.0;
  double relativeMassDrift = 0.0;
  double minAlpha = 0.0;
  double maxAlpha = 0.0;
};

inline double vofMass3D(const Mesh3D& mesh, const ScalarField& alpha) {
  require(alpha.size() == mesh.cells.size(), "3D VoF mass field size mismatch");
  double m = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) m += alpha[c] * mesh.cells[c].V;
  return m;
}

inline std::pair<double, double> vofBounds3D(const ScalarField& alpha) {
  require(!alpha.empty(), "3D VoF bounds need non-empty field");
  auto [lo, hi] = std::minmax_element(alpha.begin(), alpha.end());
  return {*lo, *hi};
}

inline double tetraVolumeExact3D(const Vec3& a, const Vec3& b, const Vec3& c,
                                 const Vec3& d) {
  return std::abs((b - a).dot((c - a).cross(d - a))) / 6.0;
}

inline Vec3 edgePlaneIntersection3D(const Vec3& wet, double sw,
                                    const Vec3& dry, double sd) {
  const double t = sw / std::max(sw - sd, 1e-30);
  return wet + std::clamp(t, 0.0, 1.0) * (dry - wet);
}

inline double clippedTetraWetVolume3D(const std::array<Vec3, 4>& p,
                                      const std::array<double, 4>& signedDistance) {
  std::vector<int> wet;
  std::vector<int> dry;
  wet.reserve(4);
  dry.reserve(4);
  for (int i = 0; i < 4; ++i) {
    if (signedDistance[i] >= 0.0) wet.push_back(i);
    else dry.push_back(i);
  }
  const double full = tetraVolumeExact3D(p[0], p[1], p[2], p[3]);
  if (wet.empty()) return 0.0;
  if (wet.size() == 4) return full;

  if (wet.size() == 1) {
    const int w = wet.front();
    const Vec3 i0 = edgePlaneIntersection3D(p[w], signedDistance[w], p[dry[0]], signedDistance[dry[0]]);
    const Vec3 i1 = edgePlaneIntersection3D(p[w], signedDistance[w], p[dry[1]], signedDistance[dry[1]]);
    const Vec3 i2 = edgePlaneIntersection3D(p[w], signedDistance[w], p[dry[2]], signedDistance[dry[2]]);
    return tetraVolumeExact3D(p[w], i0, i1, i2);
  }

  if (wet.size() == 3) {
    const int d = dry.front();
    const Vec3 i0 = edgePlaneIntersection3D(p[wet[0]], signedDistance[wet[0]], p[d], signedDistance[d]);
    const Vec3 i1 = edgePlaneIntersection3D(p[wet[1]], signedDistance[wet[1]], p[d], signedDistance[d]);
    const Vec3 i2 = edgePlaneIntersection3D(p[wet[2]], signedDistance[wet[2]], p[d], signedDistance[d]);
    return std::clamp(full - tetraVolumeExact3D(p[d], i0, i1, i2), 0.0, full);
  }

  const int w0 = wet[0];
  const int w1 = wet[1];
  const int d0 = dry[0];
  const int d1 = dry[1];
  const Vec3 i00 = edgePlaneIntersection3D(p[w0], signedDistance[w0], p[d0], signedDistance[d0]);
  const Vec3 i01 = edgePlaneIntersection3D(p[w0], signedDistance[w0], p[d1], signedDistance[d1]);
  const Vec3 i10 = edgePlaneIntersection3D(p[w1], signedDistance[w1], p[d0], signedDistance[d0]);
  const Vec3 i11 = edgePlaneIntersection3D(p[w1], signedDistance[w1], p[d1], signedDistance[d1]);
  double v = 0.0;
  v += tetraVolumeExact3D(p[w0], p[w1], i10, i11);
  v += tetraVolumeExact3D(p[w0], i10, i00, i11);
  v += tetraVolumeExact3D(p[w0], i00, i01, i11);
  return std::clamp(v, 0.0, full);
}

inline bool cellSupportsExactPlicPlaneCut3D(const Cell3D& c) {
  return c.points.size() == 4 || c.points.size() == 8;
}

inline std::vector<std::array<int, 4>> cellTetraDecomposition3D(const Cell3D& c) {
  if (c.points.size() == 4) {
    return {{{c.points[0], c.points[1], c.points[2], c.points[3]}}};
  }
  if (c.points.size() == 8) {
    return {{{c.points[0], c.points[1], c.points[2], c.points[6]}},
            {{c.points[0], c.points[2], c.points[3], c.points[6]}},
            {{c.points[0], c.points[3], c.points[7], c.points[6]}},
            {{c.points[0], c.points[7], c.points[4], c.points[6]}},
            {{c.points[0], c.points[4], c.points[5], c.points[6]}},
            {{c.points[0], c.points[5], c.points[1], c.points[6]}}};
  }
  require(false, "exact 3D PLIC plane cut currently supports tetra and hex cells");
  return {};
}

inline double exactPlicPlaneCutVolumeFraction3D(const Mesh3D& mesh, int celli,
                                               const Vec3& normalIn, double cut) {
  require(celli >= 0 && celli < static_cast<int>(mesh.cells.size()),
          "exact 3D PLIC plane cut cell index out of range");
  Vec3 normal = normalIn;
  const double magN = normal.norm();
  require(magN > 1e-30, "exact 3D PLIC plane cut needs non-zero normal");
  normal /= magN;

  const Cell3D& cell = mesh.cells[celli];
  double wetVolume = 0.0;
  double totalVolume = 0.0;
  for (const auto& tet : cellTetraDecomposition3D(cell)) {
    std::array<Vec3, 4> p = {mesh.points[tet[0]], mesh.points[tet[1]],
                             mesh.points[tet[2]], mesh.points[tet[3]]};
    std::array<double, 4> s = {
        normal.dot(p[0] - cell.centroid) - cut,
        normal.dot(p[1] - cell.centroid) - cut,
        normal.dot(p[2] - cell.centroid) - cut,
        normal.dot(p[3] - cell.centroid) - cut};
    totalVolume += tetraVolumeExact3D(p[0], p[1], p[2], p[3]);
    wetVolume += clippedTetraWetVolume3D(p, s);
  }
  require(totalVolume > 0.0, "exact 3D PLIC plane cut got non-positive decomposed volume");
  return std::clamp(wetVolume / totalVolume, 0.0, 1.0);
}

inline double exactPlicCutForAlpha3D(const Mesh3D& mesh, int celli,
                                     const Vec3& normalIn, double alpha) {
  require(celli >= 0 && celli < static_cast<int>(mesh.cells.size()),
          "exact 3D PLIC cut inversion cell index out of range");
  Vec3 normal = normalIn;
  const double magN = normal.norm();
  require(magN > 1e-30, "exact 3D PLIC cut inversion needs non-zero normal");
  normal /= magN;
  const Cell3D& cell = mesh.cells[celli];
  require(cellSupportsExactPlicPlaneCut3D(cell),
          "exact 3D PLIC cut inversion currently supports tetra and hex cells");

  double minS = std::numeric_limits<double>::infinity();
  double maxS = -std::numeric_limits<double>::infinity();
  for (int pi : cell.points) {
    const double s = normal.dot(mesh.points[pi] - cell.centroid);
    minS = std::min(minS, s);
    maxS = std::max(maxS, s);
  }
  if (alpha <= 1e-12) return maxS + 1e-12 * std::max(maxS - minS, 1.0);
  if (alpha >= 1.0 - 1e-12) return minS - 1e-12 * std::max(maxS - minS, 1.0);

  double lo = minS;
  double hi = maxS;
  for (int iter = 0; iter < 48; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double frac = exactPlicPlaneCutVolumeFraction3D(mesh, celli, normal, mid);
    if (frac > alpha) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

inline void enforceBoundedMass3D(const Mesh3D& mesh, ScalarField& alpha, double targetMass,
                                 int sweeps = 3) {
  for (double& a : alpha) a = std::clamp(a, 0.0, 1.0);
  for (int sweep = 0; sweep < std::max(1, sweeps); ++sweep) {
    double mass = vofMass3D(mesh, alpha);
    double deficit = targetMass - mass;
    if (std::abs(deficit) <= 1e-14 * std::max(std::abs(targetMass), 1.0)) break;
    double capacity = 0.0;
    if (deficit > 0.0) {
      for (size_t c = 0; c < mesh.cells.size(); ++c) capacity += (1.0 - alpha[c]) * mesh.cells[c].V;
      if (capacity <= 1e-30) break;
      double frac = std::min(1.0, deficit / capacity);
      for (size_t c = 0; c < mesh.cells.size(); ++c) alpha[c] += frac * (1.0 - alpha[c]);
    } else {
      for (size_t c = 0; c < mesh.cells.size(); ++c) capacity += alpha[c] * mesh.cells[c].V;
      if (capacity <= 1e-30) break;
      double frac = std::min(1.0, -deficit / capacity);
      for (double& a : alpha) a -= frac * a;
    }
    for (double& a : alpha) a = std::clamp(a, 0.0, 1.0);
  }
}

inline void sharpenInterfaceConservative3D(const Mesh3D& mesh, ScalarField& alpha,
                                           double targetMass, double strength,
                                           int sweeps, int correctionSweeps = 3) {
  const double s = std::clamp(strength, 0.0, 1.0);
  if (s <= 0.0 || sweeps <= 0) return;
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    for (double& a : alpha) {
      const double ac = std::clamp(a, 0.0, 1.0);
      const double delta = s * (2.0 * ac - 1.0) * ac * (1.0 - ac);
      a = std::clamp(ac + delta, 0.0, 1.0);
    }
    enforceBoundedMass3D(mesh, alpha, targetMass, correctionSweeps);
  }
}

inline ScalarField vofCompressionFlux3D(const Mesh3D& mesh, const ScalarField& alpha,
                                        double compression) {
  ScalarField flux(mesh.faces.size(), 0.0);
  if (compression <= 0.0) return flux;
  VectorField3 grad = gradLeastSquares3D(mesh, alpha);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    if (!f.internal()) continue;
    Vec3 gf = 0.5 * (grad[f.owner] + grad[f.neighbour]);
    double magG = gf.norm();
    if (magG <= 1e-30) continue;
    double af = std::clamp(0.5 * (alpha[f.owner] + alpha[f.neighbour]), 0.0, 1.0);
    flux[fi] = compression * af * (1.0 - af) * gf.normalized().dot(f.Sf);
  }
  return flux;
}

struct IsoSurfaceReconstruction3D {
  Vec3 normal = Vec3::UnitX();
  double cut = 0.0;
  double alpha = 0.0;
  Vec3 cellCentroid = Vec3::Zero();
  Vec3 interfaceCentroid = Vec3::Zero();
  double areaDensity = 0.0;
  bool mixed = false;
};

inline std::vector<Vec3> cellReconstructionSamples3D(const Mesh3D& mesh, int celli) {
  std::vector<Vec3> samples;
  const Cell3D& c = mesh.cells[celli];
  samples.reserve(1 + c.points.size() + c.faces.size());
  samples.push_back(c.centroid);
  for (int pi : c.points) samples.push_back(mesh.points[pi]);
  for (int fi : c.faces) samples.push_back(mesh.faces[fi].centroid);
  return samples;
}

inline IsoSurfaceReconstruction3D reconstructIsoSurface3D(const Mesh3D& mesh,
                                                          const ScalarField& alpha,
                                                          int celli,
                                                          const VectorField3& gradAlpha) {
  IsoSurfaceReconstruction3D iso;
  iso.cellCentroid = mesh.cells[celli].centroid;
  iso.interfaceCentroid = iso.cellCentroid;
  iso.alpha = std::clamp(alpha[celli], 0.0, 1.0);
  if (iso.alpha <= 1e-12 || iso.alpha >= 1.0 - 1e-12) return iso;
  iso.mixed = true;
  iso.normal = gradAlpha[celli];
  double magN = iso.normal.norm();
  if (magN <= 1e-30) {
    iso.normal = Vec3::UnitX();
  } else {
    iso.normal /= magN;
  }

  if (cellSupportsExactPlicPlaneCut3D(mesh.cells[celli])) {
    iso.cut = exactPlicCutForAlpha3D(mesh, celli, iso.normal, iso.alpha);
  } else {
    std::vector<Vec3> samples = cellReconstructionSamples3D(mesh, celli);
    std::vector<double> s;
    s.reserve(samples.size());
    for (const Vec3& x : samples) s.push_back(iso.normal.dot(x - mesh.cells[celli].centroid));
    std::sort(s.begin(), s.end());
    const double below = std::clamp(1.0 - iso.alpha, 0.0, 1.0);
    const double pos = below * static_cast<double>(s.size() - 1);
    const int lo = static_cast<int>(std::floor(pos));
    const int hi = std::min<int>(lo + 1, static_cast<int>(s.size()) - 1);
    const double frac = pos - static_cast<double>(lo);
    iso.cut = (1.0 - frac) * s[lo] + frac * s[hi];
  }
  iso.interfaceCentroid = mesh.cells[celli].centroid + iso.cut * iso.normal;
  iso.areaDensity = gradAlpha[celli].norm();
  return iso;
}

inline std::vector<IsoSurfaceReconstruction3D> reconstructIsoInterface3D(
    const Mesh3D& mesh, const ScalarField& alpha) {
  require(alpha.size() == mesh.cells.size(), "3D iso interface alpha size mismatch");
  VectorField3 gradAlpha = gradLeastSquares3D(mesh, alpha);
  // plicRDF-style normal refinement: neighbour-average the interface normal direction so
  // the reconstruction plane uses a smoother, curvature-consistent normal (approximating
  // the reconstructed-distance-function normal) rather than the raw alpha-gradient, while
  // preserving the |grad(alpha)| interface area density per cell.
  VectorField3 normalSource = gradAlpha;
  for (int sweep = 0; sweep < 2; ++sweep) {
    VectorField3 next = normalSource;
    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
      const double a = std::clamp(alpha[ci], 0.0, 1.0);
      if (a <= 1e-12 || a >= 1.0 - 1e-12) continue;
      if (normalSource[ci].norm() <= 1e-30) continue;
      Vec3 acc = normalSource[ci].normalized();
      int count = 1;
      for (int fi : mesh.cells[ci].faces) {
        const Face3D& f = mesh.faces[fi];
        int cj = f.owner == ci ? f.neighbour : f.owner;
        if (cj < 0) continue;
        const double m = normalSource[cj].norm();
        if (m <= 1e-30) continue;
        acc += normalSource[cj] / m;
        ++count;
      }
      if (count > 0 && acc.norm() > 1e-30) {
        next[ci] = acc.normalized() * gradAlpha[ci].norm();
      }
    }
    normalSource.swap(next);
  }
  std::vector<IsoSurfaceReconstruction3D> iso(mesh.cells.size());
  // Per-cell disjoint write: iteration c writes only iso[c] via reconstructIsoSurface3D,
  // which allocates only local scratch (samples/sort or the PLIC bisection) and reads
  // mesh/alpha/normalSource read-only. No cross-iteration accumulation. Bit-exact at any
  // thread count. Hot: reconstructIsoSurface3D runs a 48-iteration PLIC bisection per
  // interface cell, and this is invoked twice per solver step via isoAdvectorFaceFlux3D.
  FVM_PARALLEL_FOR
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) {
    iso[c] = reconstructIsoSurface3D(mesh, alpha, c, normalSource);
  }
  return iso;
}

inline double isoSurfaceWetValue3D(const Mesh3D& mesh, int celli,
                                   const IsoSurfaceReconstruction3D& iso,
                                   const Vec3& x) {
  if (iso.alpha <= 1e-12) return 0.0;
  if (iso.alpha >= 1.0 - 1e-12) return 1.0;
  double h = std::cbrt(mesh.cells[celli].V);
  double signedDistance = iso.normal.dot(x - mesh.cells[celli].centroid) - iso.cut;
  double width = std::max(0.08 * h, 1e-30);
  return std::clamp(0.5 + 0.5 * signedDistance / width, 0.0, 1.0);
}

inline std::vector<Vec3> faceAdvectionSamples3D(const Mesh3D& mesh, int facei,
                                                double faceFlux, double dt) {
  const Face3D& f = mesh.faces[facei];
  std::vector<Vec3> samples;
  samples.reserve(1 + 2 * f.points.size());
  Vec3 n = f.Sf / std::max(f.area, 1e-30);
  double un = faceFlux / std::max(f.area, 1e-30);
  Vec3 shift = -0.5 * un * dt * n;
  samples.push_back(f.centroid + shift);
  for (size_t i = 0; i < f.points.size(); ++i) {
    const Vec3& a = mesh.points[f.points[i]];
    const Vec3& b = mesh.points[f.points[(i + 1) % f.points.size()]];
    samples.push_back(a + shift);
    samples.push_back(0.5 * (a + b) + shift);
  }
  return samples;
}

inline double clippedSweptTetWetVolume3D(const std::array<Vec3, 4>& p,
                                         const IsoSurfaceReconstruction3D& iso) {
  std::array<double, 4> s = {
      iso.normal.dot(p[0] - iso.cellCentroid) - iso.cut,
      iso.normal.dot(p[1] - iso.cellCentroid) - iso.cut,
      iso.normal.dot(p[2] - iso.cellCentroid) - iso.cut,
      iso.normal.dot(p[3] - iso.cellCentroid) - iso.cut};
  return clippedTetraWetVolume3D(p, s);
}

inline double exactSweptFacePlicWetFraction3D(const Mesh3D& mesh, int facei, int upwind,
                                             double faceFlux, double dt,
                                             const IsoSurfaceReconstruction3D& iso) {
  if (iso.alpha <= 1e-12) return 0.0;
  if (iso.alpha >= 1.0 - 1e-12) return 1.0;
  const Face3D& f = mesh.faces[facei];
  if (f.area <= 1e-30 || f.points.size() < 3) {
    return isoSurfaceWetValue3D(mesh, upwind, iso, f.centroid);
  }

  const double distance = std::abs(faceFlux) * dt / std::max(f.area, 1e-30);
  if (distance <= 1e-30) return isoSurfaceWetValue3D(mesh, upwind, iso, f.centroid);
  const Vec3 faceNormal = f.Sf / std::max(f.area, 1e-30);
  const Vec3 backShift = -std::copysign(distance, faceFlux) * faceNormal;
  double wetVolume = 0.0;
  double totalVolume = 0.0;

  auto accumulateTet = [&](const std::array<Vec3, 4>& tet) {
    const double volume = tetraVolumeExact3D(tet[0], tet[1], tet[2], tet[3]);
    totalVolume += volume;
    wetVolume += clippedSweptTetWetVolume3D(tet, iso);
  };

  const Vec3 p0 = mesh.points[f.points[0]];
  for (size_t i = 1; i + 1 < f.points.size(); ++i) {
    const Vec3 p1 = mesh.points[f.points[i]];
    const Vec3 p2 = mesh.points[f.points[i + 1]];
    const Vec3 q0 = p0 + backShift;
    const Vec3 q1 = p1 + backShift;
    const Vec3 q2 = p2 + backShift;
    accumulateTet({p0, p1, p2, q2});
    accumulateTet({p0, p1, q2, q1});
    accumulateTet({p0, q1, q2, q0});
  }

  if (totalVolume <= 1e-30) return isoSurfaceWetValue3D(mesh, upwind, iso, f.centroid);
  return std::clamp(wetVolume / totalVolume, 0.0, 1.0);
}

inline ScalarField isoAdvectorFaceFlux3D(const Mesh3D& mesh, const ScalarField& alpha,
                                         const ScalarField& faceFlux, double dt,
                                         const ScalarField* boundaryAlpha = nullptr) {
  require(alpha.size() == mesh.cells.size(), "3D isoAdvector alpha size mismatch");
  require(faceFlux.size() == mesh.faces.size(), "3D isoAdvector face flux size mismatch");
  if (boundaryAlpha) {
    require(boundaryAlpha->size() == mesh.faces.size(),
            "3D isoAdvector boundary alpha size mismatch");
  }
  require(dt > 0.0, "3D isoAdvector needs positive dt");
  std::vector<IsoSurfaceReconstruction3D> iso = reconstructIsoInterface3D(mesh, alpha);

  ScalarField flux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    const double mdot = faceFlux[fi];
    if (std::abs(mdot) <= 1e-30) {
      flux[fi] = 0.0;
      continue;
    }
    int upwind = f.owner;
    double wet = 0.0;
    if (f.internal()) {
      if (mdot < 0.0) upwind = f.neighbour;
      wet = exactSweptFacePlicWetFraction3D(mesh, fi, upwind, mdot, dt, iso[upwind]);
    } else if (boundaryAlpha && mdot < 0.0) {
      wet = std::clamp((*boundaryAlpha)[fi], 0.0, 1.0);
    } else {
      wet = exactSweptFacePlicWetFraction3D(mesh, fi, upwind, mdot, dt, iso[upwind]);
    }
    flux[fi] = mdot * std::clamp(wet, 0.0, 1.0);
  }
  return flux;
}

inline VofTransportReport3D advectVof3D(const Mesh3D& mesh, ScalarField& alpha,
                                        const ScalarField& faceFlux, double dt,
                                        const VofTransportOptions3D& opt = {}) {
  require(alpha.size() == mesh.cells.size(), "3D VoF alpha size mismatch");
  require(faceFlux.size() == mesh.faces.size(), "3D VoF face flux size mismatch");
  double initialMass = vofMass3D(mesh, alpha);

  ScalarField flux =
      opt.scheme == VofAdvectionScheme3D::IsoAdvector
          ? isoAdvectorFaceFlux3D(mesh, alpha, faceFlux, dt, opt.boundaryAlpha)
          : convectionFaceFluxUpwindTVD3D(mesh, alpha, faceFlux, opt.tvdBlend);
  ScalarField cflux = vofCompressionFlux3D(mesh, alpha, opt.compression);
  for (size_t fi = 0; fi < flux.size(); ++fi) flux[fi] += cflux[fi];
  double targetMass = initialMass;
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    if (!mesh.faces[fi].internal()) targetMass -= dt * flux[fi];
  }
  ScalarField div = explicitDivFaceFlux3D(mesh, flux);
  for (size_t c = 0; c < alpha.size(); ++c) alpha[c] -= dt * div[c];

  sharpenInterfaceConservative3D(mesh, alpha, targetMass, opt.postSharpening,
                                 opt.postSharpeningSweeps, opt.correctionSweeps);
  enforceBoundedMass3D(mesh, alpha, targetMass, opt.correctionSweeps);
  auto [amin, amax] = vofBounds3D(alpha);
  double finalMass = vofMass3D(mesh, alpha);
  double drift = std::abs(finalMass - targetMass) / std::max(std::abs(targetMass), 1e-30);
  return {initialMass, finalMass, drift, amin, amax};
}

inline ScalarField divergenceFreeBoxFlux3D(const Mesh3D& mesh, double scale = 1.0) {
  ScalarField faceFlux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    if (!f.internal()) continue;
    const Vec3& x = f.centroid;
    Vec3 u{std::pow(std::sin(M_PI * x.x()), 2.0) * std::sin(2.0 * M_PI * x.y()) * std::sin(2.0 * M_PI * x.z()),
           -std::sin(2.0 * M_PI * x.x()) * std::pow(std::sin(M_PI * x.y()), 2.0) * std::sin(2.0 * M_PI * x.z()),
           0.0};
    faceFlux[fi] = scale * u.dot(f.Sf);
  }
  return faceFlux;
}

}
