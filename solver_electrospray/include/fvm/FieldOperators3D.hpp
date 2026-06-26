#pragma once

#include "fvm/Mesh3D.hpp"
#include "fvm/Parallel.hpp"

namespace fvm {

inline VectorField3 gradLeastSquares3D(const Mesh3D& mesh, const ScalarField& phi) {
  VectorField3 g(mesh.cells.size(), Vec3::Zero());
  FVM_PARALLEL_FOR
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Vec3 b = Vec3::Zero();
    for (int fi : mesh.cells[ci].faces) {
      const Face3D& f = mesh.faces[fi];
      int cj = f.owner == ci ? f.neighbour : f.owner;
      if (cj < 0) continue;
      Vec3 r = mesh.cells[cj].centroid - mesh.cells[ci].centroid;
      double w = 1.0 / std::max(r.squaredNorm(), 1e-30);
      A += w * (r * r.transpose());
      b += w * r * (phi[cj] - phi[ci]);
    }
    g[ci] = A.ldlt().solve(b);
  }
  return g;
}

inline VectorField3 gradGreenGauss3D(const Mesh3D& mesh, const ScalarField& phi) {
  VectorField3 g(mesh.cells.size(), Vec3::Zero());
  for (const Face3D& f : mesh.faces) {
    double pf = f.internal() ? 0.5 * (phi[f.owner] + phi[f.neighbour]) : phi[f.owner];
    g[f.owner] += pf * f.Sf;
    if (f.internal()) g[f.neighbour] -= pf * f.Sf;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) g[c] /= mesh.cells[c].V;
  return g;
}

inline ScalarField divergence3D(const Mesh3D& mesh, const VectorField3& u) {
  ScalarField div(mesh.cells.size(), 0.0);
  for (const Face3D& f : mesh.faces) {
    Vec3 uf = f.internal() ? 0.5 * (u[f.owner] + u[f.neighbour]) : u[f.owner];
    double flux = uf.dot(f.Sf);
    div[f.owner] += flux;
    if (f.internal()) div[f.neighbour] -= flux;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) div[c] /= mesh.cells[c].V;
  return div;
}

inline double snGradOverRelaxed3D(const Mesh3D& mesh, int fi, const ScalarField& phi,
                                  const VectorField3& grad) {
  const Face3D& f = mesh.faces[fi];
  double jump = f.internal() ? (phi[f.neighbour] - phi[f.owner]) : 0.0;
  Vec3 gf = f.internal() ? 0.5 * (grad[f.owner] + grad[f.neighbour]) : grad[f.owner];
  return jump * f.Delta.norm() / std::max(f.d.norm() * f.area, 1e-30)
       + gf.dot(f.k) / std::max(f.area, 1e-30);
}

inline ScalarField laplacianExplicit3D(const Mesh3D& mesh, const ScalarField& phi) {
  VectorField3 g = gradLeastSquares3D(mesh, phi);
  ScalarField lap(mesh.cells.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    double flux = f.area * snGradOverRelaxed3D(mesh, fi, phi, g);
    lap[f.owner] += flux;
    if (f.internal()) lap[f.neighbour] -= flux;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) lap[c] /= mesh.cells[c].V;
  return lap;
}

inline ScalarField explicitDivFaceFlux3D(const Mesh3D& mesh, const ScalarField& faceFlux) {
  ScalarField div(mesh.cells.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    div[f.owner] += faceFlux[fi];
    if (f.internal()) div[f.neighbour] -= faceFlux[fi];
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) div[c] /= mesh.cells[c].V;
  return div;
}

inline ScalarField convectionFaceFluxUpwindTVD3D(const Mesh3D& mesh, const ScalarField& phi,
                                                 const ScalarField& faceFlux,
                                                 double deferredBlend = 1.0) {
  require(faceFlux.size() == mesh.faces.size(), "3D TVD convection face flux size mismatch");
  ScalarField flux(mesh.faces.size(), 0.0);
  VectorField3 grad = gradLeastSquares3D(mesh, phi);
  const double blend = std::clamp(deferredBlend, 0.0, 1.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    const double mdot = faceFlux[fi];
    int up = f.owner;
    int dn = f.neighbour;
    if (f.internal() && mdot < 0.0) {
      up = f.neighbour;
      dn = f.owner;
    }
    double facePhi = phi[up];
    if (f.internal()) {
      Vec3 rUpFace = f.centroid - mesh.cells[up].centroid;
      Vec3 rUpDn = mesh.cells[dn].centroid - mesh.cells[up].centroid;
      double denom = phi[dn] - phi[up];
      double upstreamExtrap = grad[up].dot(rUpDn);
      double r = upstreamExtrap / (denom + (std::abs(denom) < 1e-30 ? 1e-30 : 0.0));
      double limited = phi[up] + 0.5 * vanLeer(r) * (phi[dn] - phi[up]);
      double linear = phi[up] + grad[up].dot(rUpFace);
      double correction = limited - phi[up];
      if (!std::isfinite(correction)) correction = linear - phi[up];
      facePhi = phi[up] + blend * correction;
      double lo = std::min(phi[up], phi[dn]);
      double hi = std::max(phi[up], phi[dn]);
      facePhi = std::clamp(facePhi, lo, hi);
    }
    flux[fi] = mdot * facePhi;
  }
  return flux;
}

inline ScalarField divConvectionUpwindTVD3D(const Mesh3D& mesh, const ScalarField& phi,
                                            const ScalarField& faceFlux,
                                            double deferredBlend = 1.0) {
  return explicitDivFaceFlux3D(mesh, convectionFaceFluxUpwindTVD3D(mesh, phi, faceFlux, deferredBlend));
}

}
