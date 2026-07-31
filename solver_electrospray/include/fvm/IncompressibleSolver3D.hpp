#pragma once

#include "fvm/PressureVelocityCoupling3D.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <string>

namespace fvm {

struct VelocityBC3D {
  std::function<Vec3(const Vec3&)> value;
};

struct Cavity3DLid {
  int normalAxis = 1;
  double side = 1.0;
  int velocityComponent = 0;
  double velocity = 1.0;
};

struct IncompressibleResult3D {
  VectorField3 u;
  ScalarField p;
  double maxDiv = 0.0;
  int steps = 0;
  double maxCourant = 0.0;
};

struct Cavity3DCase {
  int n = 8;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int Re = 100;
  int steps = 250;
  double dt = 0.01;
  double pressureVelocityRelaxation = 1.0;
  bool includeConvection = true;
  double convectionDeferredBlend = 1.0;
  double pressurePredictorSign = -1.0;
  int pressureCorrectors = 1;
  Cavity3DLid lid;
  bool cosineMesh = false;
  double momentumIlutDropTol = 1e-3;
  int momentumIlutFillFactor = 10;
  double maxCourant = std::numeric_limits<double>::infinity();
};

struct Cavity3DStage {
  int Re = 100;
  int steps = 100;
  double dt = 0.01;
  double maxCourant = std::numeric_limits<double>::infinity();
};

struct Cavity3DProfilePoint {
  std::string axis;
  double coord = 0.0;
  std::string component;
  double value = 0.0;
};

inline Vec3 cavityVelocityBC3D(const Vec3& x) {
  Cavity3DLid lid;
  if (std::abs(x[lid.normalAxis] - lid.side) < 1e-10) {
    Vec3 v = Vec3::Zero();
    v[lid.velocityComponent] = lid.velocity;
    return v;
  }
  return {0.0, 0.0, 0.0};
}

inline VelocityBC3D makeCavityVelocityBC3D(const Cavity3DLid& lid) {
  return VelocityBC3D{[lid](const Vec3& x) -> Vec3 {
    if (std::abs(x[lid.normalAxis] - lid.side) < 1e-10) {
      Vec3 v = Vec3::Zero();
      v[lid.velocityComponent] = lid.velocity;
      return v;
    }
    return Vec3::Zero();
  }};
}

inline Mesh3D makeCavityMesh3D(const Cavity3DCase& cfg) {
  int nx = cfg.nx > 0 ? cfg.nx : cfg.n;
  int ny = cfg.ny > 0 ? cfg.ny : cfg.n;
  int nz = cfg.nz > 0 ? cfg.nz : cfg.n;
  return cfg.cosineMesh ? Mesh3D::cosineHexGrid(nx, ny, nz)
                        : Mesh3D::hexGrid(nx, ny, nz, 1.0, 1.0, 1.0, 0.0);
}

inline double maxCourantNumber3D(const Mesh3D& mesh, const VectorField3& u,
                                 const VelocityBC3D& bc, double dt) {
  require(u.size() == mesh.cells.size(), "3D Courant field size mismatch");
  ScalarField outflow(mesh.cells.size(), 0.0);
  for (const Face3D& f : mesh.faces) {
    Vec3 uf = f.internal() ? 0.5 * (u[f.owner] + u[f.neighbour]) : bc.value(f.centroid);
    double phi = uf.dot(f.Sf);
    if (f.internal()) {
      if (phi >= 0.0) outflow[f.owner] += phi;
      else outflow[f.neighbour] -= phi;
    } else if (phi > 0.0) {
      outflow[f.owner] += phi;
    }
  }
  double maxCo = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    maxCo = std::max(maxCo, dt * outflow[c] / std::max(mesh.cells[c].V, 1e-30));
  }
  return maxCo;
}

inline double maxCourantNumberFromFaceFlux3D(const Mesh3D& mesh, const ScalarField& faceFlux,
                                             double dt) {
  require(faceFlux.size() == mesh.faces.size(), "3D face-flux Courant size mismatch");
  ScalarField outflow(mesh.cells.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    double phi = faceFlux[fi];
    if (f.internal()) {
      if (phi >= 0.0) outflow[f.owner] += phi;
      else outflow[f.neighbour] -= phi;
    } else if (phi > 0.0) {
      outflow[f.owner] += phi;
    }
  }
  double maxCo = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    maxCo = std::max(maxCo, dt * outflow[c] / std::max(mesh.cells[c].V, 1e-30));
  }
  return maxCo;
}

inline ScalarField diffusionRAU3D(const Mesh3D& mesh, double nu, double dt) {
  ScalarField diag(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) diag[c] = mesh.cells[c].V / dt;
  for (const Face3D& f : mesh.faces) {
    double coeff = nu * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
    diag[f.owner] += coeff;
    if (f.internal()) diag[f.neighbour] += coeff;
  }
  ScalarField rAU(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) rAU[c] = mesh.cells[c].V / std::max(diag[c], 1e-30);
  return rAU;
}

inline int courantSubstepCount3D(const Mesh3D& mesh, const VectorField3& u,
                                 const VelocityBC3D& bc, double dt,
                                 double maxCourant) {
  if (!std::isfinite(maxCourant) || maxCourant <= 0.0) return 1;
  double co = maxCourantNumber3D(mesh, u, bc, dt);
  if (!std::isfinite(co)) return 1;
  constexpr double growthSafety = 0.8;
  return std::max(1, static_cast<int>(std::ceil(co / (growthSafety * maxCourant))));
}

inline int courantSubstepCountFromFaceFlux3D(const Mesh3D& mesh, const ScalarField& faceFlux,
                                             double dt, double maxCourant) {
  if (!std::isfinite(maxCourant) || maxCourant <= 0.0) return 1;
  double co = maxCourantNumberFromFaceFlux3D(mesh, faceFlux, dt);
  if (!std::isfinite(co)) return 1;
  constexpr double growthSafety = 0.8;
  return std::max(1, static_cast<int>(std::ceil(co / (growthSafety * maxCourant))));
}

inline ScalarField convectiveDivergenceComponent3D(const Mesh3D& mesh, const VectorField3& u,
                                                   const VelocityBC3D& bc, int component) {
  ScalarField div(mesh.cells.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    Vec3 uf = f.internal() ? 0.5 * (u[f.owner] + u[f.neighbour]) : bc.value(f.centroid);
    double phi = uf.dot(f.Sf);
    double q = 0.0;
    if (f.internal()) {
      q = phi >= 0.0 ? u[f.owner][component] : u[f.neighbour][component];
    } else {
      q = phi >= 0.0 ? u[f.owner][component] : bc.value(f.centroid)[component];
    }
    double faceTransport = phi * q;
    div[f.owner] += faceTransport;
    if (f.internal()) div[f.neighbour] -= faceTransport;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) div[c] /= mesh.cells[c].V;
  return div;
}

inline ScalarField convectiveDivergenceComponent3D(const Mesh3D& mesh, const VectorField3& u,
                                                   const VelocityBC3D& bc,
                                                   const ScalarField& faceFlux,
                                                   int component,
                                                   double deferredBlend = 1.0) {
  require(faceFlux.size() == mesh.faces.size(), "3D projected convection flux size mismatch");
  ScalarField phi(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < u.size(); ++c) phi[c] = u[c][component];
  ScalarField flux = convectionFaceFluxUpwindTVD3D(mesh, phi, faceFlux, deferredBlend);
  ScalarField div = explicitDivFaceFlux3D(mesh, flux);
  // Boundary-normal flux is zero for the closed cavity projector. Keep a conservative
  // fallback for future open boundaries that pass nonzero boundary face fluxes.
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    if (f.internal() || std::abs(faceFlux[fi]) < 1e-30) continue;
    double phi = faceFlux[fi];
    double q = phi >= 0.0 ? u[f.owner][component] : bc.value(f.centroid)[component];
    div[f.owner] += (phi * q - flux[fi]) / mesh.cells[f.owner].V;
  }
  return div;
}

struct CachedImplicitDiffusionSolver3D {
  const Mesh3D& mesh;
  double nu = 0.0;
  double dt = 0.0;
  SpMat A;
  SpMat scaledA;
  ScalarField rowScale;
  Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>> solver;
  std::vector<std::pair<int, double>> boundaryCoeff;

  CachedImplicitDiffusionSolver3D(const Mesh3D& mesh_, double nu_, double dt_,
                                  double ilutDropTol = 1e-3,
                                  int ilutFillFactor = 10)
      : mesh(mesh_), nu(nu_), dt(dt_) {
    const int n = static_cast<int>(mesh.cells.size());
    std::vector<Triplet> trips;
    for (int c = 0; c < n; ++c) trips.emplace_back(c, c, mesh.cells[c].V / dt);
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      const Face3D& f = mesh.faces[fi];
      double coeff = nu * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
      if (f.internal()) {
        trips.emplace_back(f.owner, f.owner, coeff);
        trips.emplace_back(f.owner, f.neighbour, -coeff);
        trips.emplace_back(f.neighbour, f.neighbour, coeff);
        trips.emplace_back(f.neighbour, f.owner, -coeff);
      } else {
        trips.emplace_back(f.owner, f.owner, coeff);
        boundaryCoeff.emplace_back(fi, coeff);
      }
    }
    A.resize(n, n);
    A.setFromTriplets(trips.begin(), trips.end());
    rowScale.assign(n, 1.0);
    std::vector<Triplet> scaledTrips;
    scaledTrips.reserve(static_cast<size_t>(A.nonZeros()));
    for (int c = 0; c < n; ++c) rowScale[c] = 1.0 / std::max(std::abs(A.coeff(c, c)), 1e-30);
    for (int k = 0; k < A.outerSize(); ++k) {
      for (SpMat::InnerIterator it(A, k); it; ++it) {
        scaledTrips.emplace_back(it.row(), it.col(), rowScale[it.row()] * it.value());
      }
    }
    scaledA.resize(n, n);
    scaledA.setFromTriplets(scaledTrips.begin(), scaledTrips.end());
    solver.preconditioner().setDroptol(ilutDropTol);
    solver.preconditioner().setFillfactor(ilutFillFactor);
    solver.setTolerance(1e-9);
    solver.setMaxIterations(10000);
    solver.compute(scaledA);
    require(solver.info() == Eigen::Success, "3D BiCGSTAB/ILUT factorization failed");
  }

  Eigen::VectorXd solveScalar(const ScalarField& old, const ScalarField& bcValue) {
    const int n = static_cast<int>(mesh.cells.size());
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    for (int c = 0; c < n; ++c) b[c] = mesh.cells[c].V * old[c] / dt;
    for (const auto& [fi, coeff] : boundaryCoeff) {
      b[mesh.faces[fi].owner] += coeff * bcValue[fi];
    }
    if (b.norm() == 0.0) return Eigen::VectorXd::Zero(n);
    Eigen::VectorXd scaledB = b;
    for (int c = 0; c < n; ++c) scaledB[c] *= rowScale[c];
    Eigen::VectorXd x = solver.solve(scaledB);
    double rel = (A * x - b).norm() / std::max(b.norm(), 1e-30);
    require(solver.info() == Eigen::Success || rel < 1e-8,
            "3D BiCGSTAB/ILUT solve failed rel=" + std::to_string(rel));
    return x;
  }

  VectorField3 solveVector(const VectorField3& old, const VelocityBC3D& bc) {
    ScalarField ux(old.size()), uy(old.size()), uz(old.size());
    for (size_t c = 0; c < old.size(); ++c) {
      ux[c] = old[c].x();
      uy[c] = old[c].y();
      uz[c] = old[c].z();
    }
    ScalarField bcx(mesh.faces.size(), 0.0), bcy(mesh.faces.size(), 0.0), bcz(mesh.faces.size(), 0.0);
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      Vec3 v = bc.value(mesh.faces[fi].centroid);
      bcx[fi] = v.x();
      bcy[fi] = v.y();
      bcz[fi] = v.z();
    }
    Eigen::VectorXd sx = solveScalar(ux, bcx);
    Eigen::VectorXd sy = solveScalar(uy, bcy);
    Eigen::VectorXd sz = solveScalar(uz, bcz);
    VectorField3 out(old.size(), Vec3::Zero());
    for (int c = 0; c < static_cast<int>(old.size()); ++c) out[c] = {sx[c], sy[c], sz[c]};
    return out;
  }
};

inline void advanceCavityProjectionSubstep3D(const Mesh3D& mesh,
                                             VectorField3& u,
                                             ScalarField& p,
                                             ScalarField& projectedFaceFlux,
                                             double& maxDiv,
                                             double nu,
                                             double dt,
                                             const VelocityBC3D& bc,
                                             double pressureVelocityRelaxation,
                                             bool includeConvection,
                                             double convectionDeferredBlend,
                                             double pressurePredictorSign,
                                             int pressureCorrectors,
                                             double momentumIlutDropTol,
                                             int momentumIlutFillFactor) {
  ScalarField rAU = diffusionRAU3D(mesh, nu, dt);
  CachedImplicitDiffusionSolver3D momentum(mesh, nu, dt,
                                           momentumIlutDropTol,
                                           momentumIlutFillFactor);
  RhieChowProjector3D projector(mesh, rAU);
  VectorField3 rhs = u;
  if (pressurePredictorSign != 0.0) {
    VectorField3 gp = gradLeastSquares3D(mesh, p);
    for (size_t c = 0; c < rhs.size(); ++c) rhs[c] += pressurePredictorSign * dt * gp[c];
  }
  if (includeConvection) {
    ScalarField cx = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 0,
                                                     convectionDeferredBlend);
    ScalarField cy = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 1,
                                                     convectionDeferredBlend);
    ScalarField cz = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 2,
                                                     convectionDeferredBlend);
    for (size_t c = 0; c < rhs.size(); ++c) {
      rhs[c].x() -= dt * cx[c];
      rhs[c].y() -= dt * cy[c];
      rhs[c].z() -= dt * cz[c];
    }
  }
  u = momentum.solveVector(rhs, bc);
  for (int corr = 0; corr < std::max(1, pressureCorrectors); ++corr) {
    CouplingReport3D cr = projector.project(u, p, pressureVelocityRelaxation);
    maxDiv = cr.maxDiv;
    projectedFaceFlux = cr.faceFlux;
  }
}

inline double advanceCavityProjectionCourantLimited3D(const Mesh3D& mesh,
                                                      VectorField3& u,
                                                      ScalarField& p,
                                                      ScalarField& projectedFaceFlux,
                                                      double& maxDiv,
                                                      double nu,
                                                      double dt,
                                                      const VelocityBC3D& bc,
                                                      double pressureVelocityRelaxation,
                                                      bool includeConvection,
                                                      double convectionDeferredBlend,
                                                      double pressurePredictorSign,
                                                      int pressureCorrectors,
                                                      double momentumIlutDropTol,
                                                      int momentumIlutFillFactor,
                                                      double maxCourant) {
  if (!std::isfinite(maxCourant) || maxCourant <= 0.0) {
    advanceCavityProjectionSubstep3D(mesh, u, p, projectedFaceFlux, maxDiv, nu, dt, bc,
                                     pressureVelocityRelaxation, includeConvection,
                                     convectionDeferredBlend, pressurePredictorSign,
                                     pressureCorrectors, momentumIlutDropTol,
                                     momentumIlutFillFactor);
    return maxCourantNumberFromFaceFlux3D(mesh, projectedFaceFlux, dt);
  }

  double remaining = dt;
  double maxObserved = 0.0;
  while (remaining > std::max(1e-14, 1e-10 * dt)) {
    int planned = courantSubstepCountFromFaceFlux3D(mesh, projectedFaceFlux, remaining, maxCourant);
    double subDt = remaining / static_cast<double>(std::max(planned, 1));
    bool accepted = false;
    for (int attempt = 0; attempt < 8; ++attempt) {
      VectorField3 oldU = u;
      ScalarField oldP = p;
      ScalarField oldFlux = projectedFaceFlux;
      double oldMaxDiv = maxDiv;
      advanceCavityProjectionSubstep3D(mesh, u, p, projectedFaceFlux, maxDiv, nu, subDt, bc,
                                       pressureVelocityRelaxation, includeConvection,
                                       convectionDeferredBlend, pressurePredictorSign,
                                       pressureCorrectors, momentumIlutDropTol,
                                       momentumIlutFillFactor);
      double co = maxCourantNumberFromFaceFlux3D(mesh, projectedFaceFlux, subDt);
      if (co <= maxCourant * (1.0 + 1e-10) || subDt <= std::max(1e-14, 1e-8 * dt)) {
        maxObserved = std::max(maxObserved, co);
        remaining -= subDt;
        accepted = true;
        break;
      }
      u = std::move(oldU);
      p = std::move(oldP);
      projectedFaceFlux = std::move(oldFlux);
      maxDiv = oldMaxDiv;
      subDt *= 0.5;
    }
    require(accepted, "3D Courant-limited cavity substep retry failed");
  }
  return maxObserved;
}

inline IncompressibleResult3D solveCavityProjection3D(int n, int Re, int steps, double dt,
                                                      double pressureVelocityRelaxation = 1.0,
                                                      bool includeConvection = false,
                                                      double pressurePredictorSign = 0.0,
                                                      int pressureCorrectors = 1,
                                                      Cavity3DLid lid = {}) {
  Cavity3DCase cfg;
  cfg.n = n;
  cfg.Re = Re;
  cfg.steps = steps;
  cfg.dt = dt;
  cfg.pressureVelocityRelaxation = pressureVelocityRelaxation;
  cfg.includeConvection = includeConvection;
  cfg.pressurePredictorSign = pressurePredictorSign;
  cfg.pressureCorrectors = pressureCorrectors;
  cfg.lid = lid;
  Mesh3D mesh = makeCavityMesh3D(cfg);
  const double nu = 1.0 / static_cast<double>(Re);
  VectorField3 u(mesh.cells.size(), Vec3::Zero());
  ScalarField p(mesh.cells.size(), 0.0);
  VelocityBC3D bc = makeCavityVelocityBC3D(lid);
  ScalarField rAU = diffusionRAU3D(mesh, nu, dt);
  CachedImplicitDiffusionSolver3D momentum(mesh, nu, dt,
                                           cfg.momentumIlutDropTol,
                                           cfg.momentumIlutFillFactor);
  RhieChowProjector3D projector(mesh, rAU);
  ScalarField projectedFaceFlux(mesh.faces.size(), 0.0);
  double maxDiv = 0.0;
  double maxCourant = 0.0;
  for (int step = 0; step < steps; ++step) {
    if (std::isfinite(cfg.maxCourant)) {
      maxCourant = std::max(maxCourant,
                            advanceCavityProjectionCourantLimited3D(
                                mesh, u, p, projectedFaceFlux, maxDiv, nu, dt, bc,
                                pressureVelocityRelaxation, includeConvection,
                                cfg.convectionDeferredBlend, pressurePredictorSign,
                                pressureCorrectors, cfg.momentumIlutDropTol,
                                cfg.momentumIlutFillFactor, cfg.maxCourant));
      continue;
    }
    VectorField3 rhs = u;
    if (pressurePredictorSign != 0.0) {
      VectorField3 gp = gradLeastSquares3D(mesh, p);
      for (size_t c = 0; c < rhs.size(); ++c) rhs[c] += pressurePredictorSign * dt * gp[c];
    }
    if (includeConvection) {
      ScalarField cx = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 0,
                                                       cfg.convectionDeferredBlend);
      ScalarField cy = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 1,
                                                       cfg.convectionDeferredBlend);
      ScalarField cz = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 2,
                                                       cfg.convectionDeferredBlend);
      for (size_t c = 0; c < rhs.size(); ++c) {
        rhs[c].x() -= dt * cx[c];
        rhs[c].y() -= dt * cy[c];
        rhs[c].z() -= dt * cz[c];
      }
    }
    u = momentum.solveVector(rhs, bc);
    for (int corr = 0; corr < std::max(1, pressureCorrectors); ++corr) {
      CouplingReport3D cr = projector.project(u, p, pressureVelocityRelaxation);
      maxDiv = cr.maxDiv;
      projectedFaceFlux = cr.faceFlux;
    }
    maxCourant = std::max(maxCourant, maxCourantNumberFromFaceFlux3D(mesh, projectedFaceFlux, dt));
  }
  return {u, p, maxDiv, steps, maxCourant};
}

inline IncompressibleResult3D solveCavityProjection3D(const Cavity3DCase& cfg) {
  Mesh3D mesh = makeCavityMesh3D(cfg);
  const double nu = 1.0 / static_cast<double>(cfg.Re);
  VectorField3 u(mesh.cells.size(), Vec3::Zero());
  ScalarField p(mesh.cells.size(), 0.0);
  VelocityBC3D bc = makeCavityVelocityBC3D(cfg.lid);
  ScalarField rAU = diffusionRAU3D(mesh, nu, cfg.dt);
  CachedImplicitDiffusionSolver3D momentum(mesh, nu, cfg.dt,
                                           cfg.momentumIlutDropTol,
                                           cfg.momentumIlutFillFactor);
  RhieChowProjector3D projector(mesh, rAU);
  ScalarField projectedFaceFlux(mesh.faces.size(), 0.0);
  double maxDiv = 0.0;
  double maxCourant = 0.0;
  for (int step = 0; step < cfg.steps; ++step) {
    if (std::isfinite(cfg.maxCourant)) {
      maxCourant = std::max(maxCourant,
                            advanceCavityProjectionCourantLimited3D(
                                mesh, u, p, projectedFaceFlux, maxDiv, nu, cfg.dt, bc,
                                cfg.pressureVelocityRelaxation, cfg.includeConvection,
                                cfg.convectionDeferredBlend, cfg.pressurePredictorSign,
                                cfg.pressureCorrectors, cfg.momentumIlutDropTol,
                                cfg.momentumIlutFillFactor, cfg.maxCourant));
      continue;
    }
    VectorField3 rhs = u;
    if (cfg.pressurePredictorSign != 0.0) {
      VectorField3 gp = gradLeastSquares3D(mesh, p);
      for (size_t c = 0; c < rhs.size(); ++c) rhs[c] += cfg.pressurePredictorSign * cfg.dt * gp[c];
    }
    if (cfg.includeConvection) {
      ScalarField cx = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 0,
                                                       cfg.convectionDeferredBlend);
      ScalarField cy = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 1,
                                                       cfg.convectionDeferredBlend);
      ScalarField cz = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 2,
                                                       cfg.convectionDeferredBlend);
      for (size_t c = 0; c < rhs.size(); ++c) {
        rhs[c].x() -= cfg.dt * cx[c];
        rhs[c].y() -= cfg.dt * cy[c];
        rhs[c].z() -= cfg.dt * cz[c];
      }
    }
    u = momentum.solveVector(rhs, bc);
    for (int corr = 0; corr < std::max(1, cfg.pressureCorrectors); ++corr) {
      CouplingReport3D cr = projector.project(u, p, cfg.pressureVelocityRelaxation);
      maxDiv = cr.maxDiv;
      projectedFaceFlux = cr.faceFlux;
    }
    maxCourant = std::max(maxCourant, maxCourantNumberFromFaceFlux3D(mesh, projectedFaceFlux, cfg.dt));
  }
  return {u, p, maxDiv, cfg.steps, maxCourant};
}

inline IncompressibleResult3D solveCavityProjection3DContinuation(
    int n, const std::vector<Cavity3DStage>& stages,
    double pressureVelocityRelaxation = 1.0,
    bool includeConvection = true,
    double pressurePredictorSign = -1.0,
    int pressureCorrectors = 1,
    Cavity3DLid lid = {},
    int nxIn = 0,
    int nyIn = 0,
    int nzIn = 0,
    bool cosineMesh = false,
    double convectionDeferredBlend = 1.0,
    double momentumIlutDropTol = 1e-3,
    int momentumIlutFillFactor = 10) {
  Cavity3DCase meshCfg;
  meshCfg.n = n;
  meshCfg.nx = nxIn;
  meshCfg.ny = nyIn;
  meshCfg.nz = nzIn;
  meshCfg.cosineMesh = cosineMesh;
  Mesh3D mesh = makeCavityMesh3D(meshCfg);
  VectorField3 u(mesh.cells.size(), Vec3::Zero());
  ScalarField p(mesh.cells.size(), 0.0);
  VelocityBC3D bc = makeCavityVelocityBC3D(lid);
  ScalarField projectedFaceFlux(mesh.faces.size(), 0.0);
  double maxDiv = 0.0;
  double maxCourant = 0.0;
  int totalSteps = 0;
  for (const Cavity3DStage& stage : stages) {
    const double nu = 1.0 / static_cast<double>(stage.Re);
    ScalarField rAU = diffusionRAU3D(mesh, nu, stage.dt);
    CachedImplicitDiffusionSolver3D momentum(mesh, nu, stage.dt,
                                             momentumIlutDropTol,
                                             momentumIlutFillFactor);
    RhieChowProjector3D projector(mesh, rAU);
    for (int step = 0; step < stage.steps; ++step) {
      if (std::isfinite(stage.maxCourant)) {
        maxCourant = std::max(maxCourant,
                              advanceCavityProjectionCourantLimited3D(
                                  mesh, u, p, projectedFaceFlux, maxDiv, nu, stage.dt, bc,
                                  pressureVelocityRelaxation, includeConvection,
                                  convectionDeferredBlend, pressurePredictorSign,
                                  pressureCorrectors, momentumIlutDropTol,
                                  momentumIlutFillFactor, stage.maxCourant));
        ++totalSteps;
        continue;
      }
      VectorField3 rhs = u;
      if (pressurePredictorSign != 0.0) {
        VectorField3 gp = gradLeastSquares3D(mesh, p);
        for (size_t c = 0; c < rhs.size(); ++c) rhs[c] += pressurePredictorSign * stage.dt * gp[c];
      }
      if (includeConvection) {
        ScalarField cx = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 0,
                                                         convectionDeferredBlend);
        ScalarField cy = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 1,
                                                         convectionDeferredBlend);
        ScalarField cz = convectiveDivergenceComponent3D(mesh, u, bc, projectedFaceFlux, 2,
                                                         convectionDeferredBlend);
        for (size_t c = 0; c < rhs.size(); ++c) {
          rhs[c].x() -= stage.dt * cx[c];
          rhs[c].y() -= stage.dt * cy[c];
          rhs[c].z() -= stage.dt * cz[c];
        }
      }
      u = momentum.solveVector(rhs, bc);
      for (int corr = 0; corr < std::max(1, pressureCorrectors); ++corr) {
        CouplingReport3D cr = projector.project(u, p, pressureVelocityRelaxation);
        maxDiv = cr.maxDiv;
        projectedFaceFlux = cr.faceFlux;
      }
      maxCourant = std::max(maxCourant, maxCourantNumberFromFaceFlux3D(mesh, projectedFaceFlux, stage.dt));
      ++totalSteps;
    }
  }
  return {u, p, maxDiv, totalSteps, maxCourant};
}

inline Vec3 interpolateStructuredCellVector3D(const Mesh3D& mesh, const VectorField3& field,
                                              const Vec3& x, const VelocityBC3D* bc = nullptr) {
  require(mesh.nx > 0 && mesh.ny > 0 && mesh.nz > 0, "3D interpolation requires structured mesh dimensions");
  require(field.size() == mesh.cells.size(), "3D interpolation field size mismatch");
  if (bc != nullptr &&
      (std::abs(x.x()) < 1e-12 || std::abs(x.x() - 1.0) < 1e-12 ||
       std::abs(x.y()) < 1e-12 || std::abs(x.y() - 1.0) < 1e-12 ||
       std::abs(x.z()) < 1e-12 || std::abs(x.z() - 1.0) < 1e-12)) {
    return bc->value(x);
  }

  auto cellId = [&mesh](int i, int j, int k) { return k * mesh.nx * mesh.ny + j * mesh.nx + i; };
  std::vector<double> xs(mesh.nx), ys(mesh.ny), zs(mesh.nz);
  for (int i = 0; i < mesh.nx; ++i) xs[i] = mesh.cells[cellId(i, 0, 0)].centroid.x();
  for (int j = 0; j < mesh.ny; ++j) ys[j] = mesh.cells[cellId(0, j, 0)].centroid.y();
  for (int k = 0; k < mesh.nz; ++k) zs[k] = mesh.cells[cellId(0, 0, k)].centroid.z();

  auto bracket = [](const std::vector<double>& a, double v) {
    if (a.size() == 1) return std::array<double, 3>{0.0, 0.0, 0.0};
    if (v <= a.front()) return std::array<double, 3>{0.0, 1.0, 0.0};
    if (v >= a.back()) {
      double last = static_cast<double>(a.size() - 1);
      return std::array<double, 3>{last - 1.0, last, 1.0};
    }
    auto hiIt = std::upper_bound(a.begin(), a.end(), v);
    int hi = static_cast<int>(hiIt - a.begin());
    int lo = hi - 1;
    double t = (v - a[lo]) / std::max(a[hi] - a[lo], 1e-30);
    return std::array<double, 3>{static_cast<double>(lo), static_cast<double>(hi), t};
  };

  auto bx = bracket(xs, x.x());
  auto by = bracket(ys, x.y());
  auto bz = bracket(zs, x.z());
  int i0 = static_cast<int>(bx[0]), i1 = static_cast<int>(bx[1]);
  int j0 = static_cast<int>(by[0]), j1 = static_cast<int>(by[1]);
  int k0 = static_cast<int>(bz[0]), k1 = static_cast<int>(bz[1]);
  double tx = bx[2], ty = by[2], tz = bz[2];

  Vec3 out = Vec3::Zero();
  for (int kk = 0; kk <= 1; ++kk) {
    double wz = kk ? tz : 1.0 - tz;
    int k = kk ? k1 : k0;
    for (int jj = 0; jj <= 1; ++jj) {
      double wy = jj ? ty : 1.0 - ty;
      int j = jj ? j1 : j0;
      for (int ii = 0; ii <= 1; ++ii) {
        double wx = ii ? tx : 1.0 - tx;
        int i = ii ? i1 : i0;
        out += (wx * wy * wz) * field[cellId(i, j, k)];
      }
    }
  }
  return out;
}

inline std::vector<Cavity3DProfilePoint> sampleCavityCenterlineProfiles3D(
    const Mesh3D& mesh, const VectorField3& u, const VelocityBC3D& bc, int intervals = 8) {
  std::vector<Cavity3DProfilePoint> samples;
  for (int s = 0; s <= intervals; ++s) {
    double q = static_cast<double>(s) / static_cast<double>(std::max(intervals, 1));
    Vec3 uyLine = interpolateStructuredCellVector3D(mesh, u, {0.5, q, 0.5}, &bc);
    Vec3 vxLine = interpolateStructuredCellVector3D(mesh, u, {q, 0.5, 0.5}, &bc);
    Vec3 wzLine = interpolateStructuredCellVector3D(mesh, u, {0.5, 0.5, q}, &bc);
    samples.push_back({"y_center", q, "ux", uyLine.x()});
    samples.push_back({"x_center", q, "uy", vxLine.y()});
    samples.push_back({"z_center", q, "uz", wzLine.z()});
  }
  return samples;
}

}
