#pragma once

#include "fvm/FvMatrix.hpp"
#include "fvm/FieldOperators3D.hpp"

namespace fvm {

inline ScalarField rhieChowFlux3D(const Mesh3D& mesh, const VectorField3& HbyA,
                                  const ScalarField& p, const ScalarField& rAU) {
  VectorField3 gp = gradLeastSquares3D(mesh, p);
  ScalarField flux(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    Vec3 hf = HbyA[f.owner];
    double rf = rAU[f.owner];
    double dp = 0.0;
    Vec3 gpf = gp[f.owner];
    if (f.internal()) {
      hf = 0.5 * (HbyA[f.owner] + HbyA[f.neighbour]);
      rf = 0.5 * (rAU[f.owner] + rAU[f.neighbour]);
      dp = p[f.neighbour] - p[f.owner];
      gpf = 0.5 * (gp[f.owner] + gp[f.neighbour]);
    }
    const double normalGrad = dp / std::max(f.d.norm(), 1e-30);
    const Vec3 n = f.Sf / std::max(f.area, 1e-30);
    double correction = rf * (normalGrad - gpf.dot(n)) * f.area;
    flux[fi] = hf.dot(f.Sf) - correction;
  }
  return flux;
}

inline double pressureCheckerboardMetric3D(const Mesh3D& mesh, const ScalarField& p) {
  if (mesh.nx <= 0 || mesh.ny <= 0 || mesh.nz <= 0) return 0.0;
  double alternating = 0.0;
  double energy = 0.0;
  for (int k = 0; k < mesh.nz; ++k) {
    for (int j = 0; j < mesh.ny; ++j) {
      for (int i = 0; i < mesh.nx; ++i) {
        int c = k * mesh.nx * mesh.ny + j * mesh.nx + i;
        double s = ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
        alternating += s * p[c];
        energy += p[c] * p[c];
      }
    }
  }
  return std::abs(alternating) / std::sqrt(std::max(energy * p.size(), 1e-30));
}

}
