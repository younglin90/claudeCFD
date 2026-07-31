#pragma once

#include "fvm/PressureVelocityCoupling.hpp"
#include <algorithm>
#include <functional>
#include <memory>

namespace fvm {

struct VelocityBC {
  std::function<Vec(const Vec&)> value;
};

inline Vec cavityVelocityBC(const Vec& x) {
  if (std::abs(x.y() - 1.0) < 1e-10) return {1.0, 0.0};
  return {0.0, 0.0};
}

inline FvMatrix implicitDiffusionSystem(const Mesh& mesh, const ScalarField& old,
                                        const ScalarField& bcValue, double nu,
                                        double dt, const VectorField* convectionVelocity = nullptr) {
  const int n = static_cast<int>(mesh.cells.size());
  FvMatrix m(n);
  std::vector<Triplet> trips;
  for (int c = 0; c < n; ++c) {
    double a = mesh.cells[c].V / dt;
    trips.emplace_back(c, c, a);
    m.b[c] = a * old[c];
  }
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face& f = mesh.faces[fi];
    double coeff = nu * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
    if (f.internal()) {
      trips.emplace_back(f.owner, f.owner, coeff);
      trips.emplace_back(f.owner, f.neighbour, -coeff);
      trips.emplace_back(f.neighbour, f.neighbour, coeff);
      trips.emplace_back(f.neighbour, f.owner, -coeff);
    } else {
      trips.emplace_back(f.owner, f.owner, coeff);
      m.b[f.owner] += coeff * bcValue[fi];
    }
  }
  if (convectionVelocity) {
    for (const Face& f : mesh.faces) {
      if (!f.internal()) continue;
      Vec uf = 0.5 * ((*convectionVelocity)[f.owner] + (*convectionVelocity)[f.neighbour]);
      double mdot = uf.dot(f.Sf);
      if (mdot >= 0.0) {
        trips.emplace_back(f.owner, f.owner, mdot);
        trips.emplace_back(f.neighbour, f.owner, -mdot);
      } else {
        trips.emplace_back(f.owner, f.neighbour, mdot);
        trips.emplace_back(f.neighbour, f.neighbour, -mdot);
      }
    }
  }
  m.A.setFromTriplets(trips.begin(), trips.end());
  return m;
}

inline ScalarField diffusionRAU(const Mesh& mesh, double nu, double dt) {
  ScalarField diag(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) diag[c] = mesh.cells[c].V / dt;
  for (const auto& f : mesh.faces) {
    double coeff = nu * std::abs(f.Sf.dot(f.d)) / std::max(f.d.squaredNorm(), 1e-30);
    diag[f.owner] += coeff;
    if (f.internal()) diag[f.neighbour] += coeff;
  }
  ScalarField rAU(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) rAU[c] = mesh.cells[c].V / std::max(diag[c], 1e-30);
  return rAU;
}

struct CachedImplicitDiffusionSolver {
  const Mesh& mesh;
  double nu = 0.0;
  double dt = 0.0;
  SpMat A;
  SpMat scaledA;
  ScalarField rowScale;
  Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>> solver;
  std::vector<std::pair<int, double>> boundaryCoeff;

  CachedImplicitDiffusionSolver(const Mesh& mesh_, double nu_, double dt_)
      : mesh(mesh_), nu(nu_), dt(dt_) {
    const int n = static_cast<int>(mesh.cells.size());
    std::vector<Triplet> trips;
    for (int c = 0; c < n; ++c) {
      trips.emplace_back(c, c, mesh.cells[c].V / dt);
    }
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      const Face& f = mesh.faces[fi];
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
    for (int c = 0; c < n; ++c) {
      rowScale[c] = 1.0 / std::max(std::abs(A.coeff(c, c)), 1e-30);
    }
    std::vector<Triplet> scaledTrips;
    scaledTrips.reserve(static_cast<size_t>(A.nonZeros()));
    for (int k = 0; k < A.outerSize(); ++k) {
      for (SpMat::InnerIterator it(A, k); it; ++it) {
        scaledTrips.emplace_back(it.row(), it.col(), rowScale[it.row()] * it.value());
      }
    }
    scaledA.resize(n, n);
    scaledA.setFromTriplets(scaledTrips.begin(), scaledTrips.end());
    solver.preconditioner().setDroptol(1e-3);
    solver.preconditioner().setFillfactor(10);
    solver.setTolerance(1e-9);
    solver.setMaxIterations(10000);
    solver.compute(scaledA);
    require(solver.info() == Eigen::Success, "BiCGSTAB/ILUT factorization failed");
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
            "BiCGSTAB/ILUT solve failed rel=" + std::to_string(rel));
    return x;
  }

  VectorField solveVector(const VectorField& old, const VelocityBC& bc) {
    ScalarField ux(old.size()), uy(old.size());
    for (size_t c = 0; c < old.size(); ++c) {
      ux[c] = old[c].x();
      uy[c] = old[c].y();
    }
    ScalarField bcx(mesh.faces.size(), 0.0), bcy(mesh.faces.size(), 0.0);
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
      Vec v = bc.value(mesh.faces[fi].centroid);
      bcx[fi] = v.x();
      bcy[fi] = v.y();
    }
    Eigen::VectorXd sx = solveScalar(ux, bcx);
    Eigen::VectorXd sy = solveScalar(uy, bcy);
    VectorField out(old.size(), Vec::Zero());
    for (int c = 0; c < static_cast<int>(old.size()); ++c) out[c] = {sx[c], sy[c]};
    return out;
  }
};

inline VectorField solveImplicitVelocityDiffusion(const Mesh& mesh, const VectorField& old,
                                                  double nu, double dt,
                                                  const VelocityBC& bc,
                                                  const VectorField* convectionVelocity = nullptr) {
  ScalarField ux(old.size()), uy(old.size());
  for (size_t c = 0; c < old.size(); ++c) {
    ux[c] = old[c].x();
    uy[c] = old[c].y();
  }
  ScalarField bcx(mesh.faces.size(), 0.0), bcy(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    Vec v = bc.value(mesh.faces[fi].centroid);
    bcx[fi] = v.x();
    bcy[fi] = v.y();
  }
  Eigen::VectorXd sx = implicitDiffusionSystem(mesh, ux, bcx, nu, dt, convectionVelocity).solveBiCGSTABILUT(1e-9, 10000);
  Eigen::VectorXd sy = implicitDiffusionSystem(mesh, uy, bcy, nu, dt, convectionVelocity).solveBiCGSTABILUT(1e-9, 10000);
  VectorField out(old.size(), Vec::Zero());
  for (int c = 0; c < static_cast<int>(old.size()); ++c) out[c] = {sx[c], sy[c]};
  return out;
}

inline VectorField reconstructVelocityFromFaceFlux(const Mesh& mesh, const ScalarField& faceFlux,
                                                   const VelocityBC& bc,
                                                   double boundaryWeight = 4.0) {
  VectorField out(mesh.cells.size(), Vec::Zero());
  for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
    Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
    Vec b = Vec::Zero();
    for (int fi : mesh.cells[ci].faces) {
      const Face& f = mesh.faces[fi];
      Vec n = f.Sf / std::max(f.area, 1e-30);
      double normalVelocity = faceFlux[fi] / std::max(f.area, 1e-30);
      if (f.neighbour == ci) {
        n = -n;
        normalVelocity = -normalVelocity;
      }
      A += n * n.transpose();
      b += n * normalVelocity;
      if (!f.internal()) {
        Vec vb = bc.value(f.centroid);
        A += boundaryWeight * Eigen::Matrix2d::Identity();
        b += boundaryWeight * vb;
      }
    }
    out[ci] = A.ldlt().solve(b);
  }
  return out;
}

inline ScalarField convectiveDivergenceComponent(const Mesh& mesh, const VectorField& u,
                                                 bool xComponent,
                                                 double deferredBlend = 1.0) {
  ScalarField component(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    component[c] = xComponent ? u[c].x() : u[c].y();
  }
  return divConvectionUpwindTVD(mesh, component, u, deferredBlend);
}

struct IncompressibleResult {
  VectorField u;
  ScalarField p;
  double maxDiv = 0.0;
  int steps = 0;
};

struct CavityStage {
  int Re = 100;
  int steps = 100;
  double dt = 1e-3;
};

inline IncompressibleResult solveCavityProjection(int n, int Re, int steps, double dt,
                                                  double pressureVelocityRelaxation = 0.0,
                                                  bool includeConvection = false,
                                                  bool reconstructFluxVelocity = false,
                                                  double pressurePredictorSign = 0.0,
                                                  double stretch = 0.0,
                                                  bool implicitConvection = false,
                                                  int pressureCorrectors = 1,
                                                  double convectionDeferredBlend = 1.0) {
  Mesh mesh = stretch > 0.0 ? Mesh::stretchedQuadGrid(n, n, stretch)
                            : Mesh::quadGrid(n, n, 1.0, 1.0, 0.0);
  const double nu = 1.0 / static_cast<double>(Re);
  VectorField u(mesh.cells.size(), Vec::Zero());
  ScalarField p(mesh.cells.size(), 0.0);
  VelocityBC bc{[](const Vec& x) { return cavityVelocityBC(x); }};
  ScalarField rAU = diffusionRAU(mesh, nu, dt);
  std::unique_ptr<CachedImplicitDiffusionSolver> cachedSolver;
  if (!implicitConvection) cachedSolver = std::make_unique<CachedImplicitDiffusionSolver>(mesh, nu, dt);
  RhieChowProjector projector(mesh, rAU);
  double maxDiv = 0.0;
  for (int step = 0; step < steps; ++step) {
    VectorField rhs = u;
    if (pressurePredictorSign != 0.0) {
      VectorField gp = gradLeastSquares(mesh, p);
      for (size_t c = 0; c < rhs.size(); ++c) rhs[c] += pressurePredictorSign * dt * gp[c];
    }
    if (includeConvection && !implicitConvection) {
      ScalarField cx = convectiveDivergenceComponent(mesh, u, true, convectionDeferredBlend);
      ScalarField cy = convectiveDivergenceComponent(mesh, u, false, convectionDeferredBlend);
      for (size_t c = 0; c < rhs.size(); ++c) {
        rhs[c].x() -= dt * cx[c];
        rhs[c].y() -= dt * cy[c];
      }
    }
    const VectorField* convVel = (includeConvection && implicitConvection) ? &u : nullptr;
    u = cachedSolver ? cachedSolver->solveVector(rhs, bc)
                     : solveImplicitVelocityDiffusion(mesh, rhs, nu, dt, bc, convVel);
    for (int corr = 0; corr < std::max(1, pressureCorrectors); ++corr) {
      CouplingReport cr = projector.project(u, p, pressureVelocityRelaxation);
      if (reconstructFluxVelocity) {
        u = reconstructVelocityFromFaceFlux(mesh, cr.faceFlux, bc);
      }
      maxDiv = cr.maxDiv;
    }
  }
  return {u, p, maxDiv, steps};
}

inline IncompressibleResult solveCavityProjectionContinuation(
    int n, const std::vector<CavityStage>& stages,
    double pressureVelocityRelaxation = 1.0,
    bool includeConvection = true,
    bool reconstructFluxVelocity = false,
    double pressurePredictorSign = -1.0,
    double stretch = 0.0,
    bool implicitConvection = false,
    int pressureCorrectors = 1,
    double convectionDeferredBlend = 1.0) {
  Mesh mesh = stretch > 0.0 ? Mesh::stretchedQuadGrid(n, n, stretch)
                            : Mesh::quadGrid(n, n, 1.0, 1.0, 0.0);
  VectorField u(mesh.cells.size(), Vec::Zero());
  ScalarField p(mesh.cells.size(), 0.0);
  VelocityBC bc{[](const Vec& x) { return cavityVelocityBC(x); }};
  double maxDiv = 0.0;
  int totalSteps = 0;
  for (const CavityStage& stage : stages) {
    const double nu = 1.0 / static_cast<double>(stage.Re);
    ScalarField rAU = diffusionRAU(mesh, nu, stage.dt);
    RhieChowProjector projector(mesh, rAU);
    std::unique_ptr<CachedImplicitDiffusionSolver> cachedSolver;
    if (!implicitConvection) {
      cachedSolver = std::make_unique<CachedImplicitDiffusionSolver>(mesh, nu, stage.dt);
    }
    for (int step = 0; step < stage.steps; ++step) {
      VectorField rhs = u;
      if (pressurePredictorSign != 0.0) {
        VectorField gp = gradLeastSquares(mesh, p);
        for (size_t c = 0; c < rhs.size(); ++c) {
          rhs[c] += pressurePredictorSign * stage.dt * gp[c];
        }
      }
      if (includeConvection && !implicitConvection) {
        ScalarField cx = convectiveDivergenceComponent(mesh, u, true, convectionDeferredBlend);
        ScalarField cy = convectiveDivergenceComponent(mesh, u, false, convectionDeferredBlend);
        for (size_t c = 0; c < rhs.size(); ++c) {
          rhs[c].x() -= stage.dt * cx[c];
          rhs[c].y() -= stage.dt * cy[c];
        }
      }
      const VectorField* convVel = (includeConvection && implicitConvection) ? &u : nullptr;
      u = cachedSolver ? cachedSolver->solveVector(rhs, bc)
                       : solveImplicitVelocityDiffusion(mesh, rhs, nu, stage.dt, bc, convVel);
      for (int corr = 0; corr < std::max(1, pressureCorrectors); ++corr) {
        CouplingReport cr = projector.project(u, p, pressureVelocityRelaxation);
        if (reconstructFluxVelocity) {
          u = reconstructVelocityFromFaceFlux(mesh, cr.faceFlux, bc);
        }
        maxDiv = cr.maxDiv;
      }
      ++totalSteps;
    }
  }
  return {u, p, maxDiv, totalSteps};
}

inline double kineticEnergy(const Mesh& mesh, const VectorField& u) {
  double e = 0.0;
  double v = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    e += 0.5 * u[c].squaredNorm() * mesh.cells[c].V;
    v += mesh.cells[c].V;
  }
  return e / std::max(v, 1e-30);
}

inline IncompressibleResult solveTaylorGreenProjection(int n, double nu, double tEnd,
                                                       double dt) {
  Mesh mesh = Mesh::quadGrid(n, n, 1.0, 1.0, 0.0);
  VectorField u(mesh.cells.size(), Vec::Zero());
  ScalarField p(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double x = mesh.cells[c].centroid.x();
    double y = mesh.cells[c].centroid.y();
    u[c] = {std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y),
            -std::cos(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y)};
  }
  VelocityBC periodicLike{[](const Vec&) { return Vec::Zero(); }};
  int steps = static_cast<int>(std::ceil(tEnd / dt));
  double actualDt = tEnd / std::max(steps, 1);
  double maxDiv = 0.0;
  for (int step = 0; step < steps; ++step) {
    u = solveImplicitVelocityDiffusion(mesh, u, nu, actualDt, periodicLike);
    CouplingReport cr = projectVelocityRhieChow(mesh, u, p, actualDt);
    maxDiv = cr.maxDiv;
  }
  return {u, p, maxDiv, steps};
}

inline Eigen::VectorXd solvePeriodicScalarDiffusion(int n, const Eigen::VectorXd& old,
                                                    double nu, double dt) {
  const double h = 1.0 / static_cast<double>(n);
  const double a0 = h * h / dt;
  const double aN = nu;
  FvMatrix m(n * n);
  std::vector<Triplet> trips;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      int c = j * n + i;
      int e = j * n + ((i + 1) % n);
      int w = j * n + ((i - 1 + n) % n);
      int north = ((j + 1) % n) * n + i;
      int south = ((j - 1 + n) % n) * n + i;
      trips.emplace_back(c, c, a0 + 4.0 * aN);
      trips.emplace_back(c, e, -aN);
      trips.emplace_back(c, w, -aN);
      trips.emplace_back(c, north, -aN);
      trips.emplace_back(c, south, -aN);
      m.b[c] = a0 * old[c];
    }
  }
  m.A.setFromTriplets(trips.begin(), trips.end());
  return m.solveBiCGSTABILUT(1e-11, 10000);
}

inline IncompressibleResult solveTaylorGreenPeriodic(int n, double nu, double tEnd,
                                                     double dt) {
  Mesh mesh = Mesh::quadGrid(n, n, 1.0, 1.0, 0.0);
  Eigen::VectorXd ux(n * n), uy(n * n);
  for (int c = 0; c < n * n; ++c) {
    double x = mesh.cells[c].centroid.x();
    double y = mesh.cells[c].centroid.y();
    ux[c] = std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y);
    uy[c] = -std::cos(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
  }
  int steps = static_cast<int>(std::ceil(tEnd / dt));
  double actualDt = tEnd / std::max(steps, 1);
  for (int step = 0; step < steps; ++step) {
    ux = solvePeriodicScalarDiffusion(n, ux, nu, actualDt);
    uy = solvePeriodicScalarDiffusion(n, uy, nu, actualDt);
  }
  VectorField u(n * n, Vec::Zero());
  for (int c = 0; c < n * n; ++c) u[c] = {ux[c], uy[c]};
  ScalarField p(n * n, 0.0);
  return {u, p, 0.0, steps};
}

inline double interpolateComponentNearest(const Mesh& mesh, const VectorField& u,
                                          const Vec& x, bool xComponent) {
  double best = 1e300;
  int bestCell = 0;
  for (int c = 0; c < static_cast<int>(mesh.cells.size()); ++c) {
    double d = (mesh.cells[c].centroid - x).squaredNorm();
    if (d < best) {
      best = d;
      bestCell = c;
    }
  }
  return xComponent ? u[bestCell].x() : u[bestCell].y();
}

inline double interpolateStructuredCellComponent(const Mesh& mesh, const VectorField& u,
                                                 const Vec& x, bool xComponent,
                                                 const VelocityBC* bc = nullptr) {
  if (bc && (x.x() <= 1e-14 || x.x() >= 1.0 - 1e-14 ||
             x.y() <= 1e-14 || x.y() >= 1.0 - 1e-14)) {
    Vec v = bc->value(x);
    return xComponent ? v.x() : v.y();
  }
  const int nx = mesh.nx;
  const int ny = mesh.ny;
  if (nx <= 1 || ny <= 1) return interpolateComponentNearest(mesh, u, x, xComponent);
  std::vector<double> xs(nx), ys(ny);
  for (int i = 0; i < nx; ++i) xs[i] = mesh.cells[i].centroid.x();
  for (int j = 0; j < ny; ++j) ys[j] = mesh.cells[j * nx].centroid.y();
  auto bracket = [](const std::vector<double>& coords, double value) {
    const int n = static_cast<int>(coords.size());
    if (value <= coords.front()) return std::pair<int, double>{0, 0.0};
    if (value >= coords.back()) return std::pair<int, double>{n - 2, 1.0};
    auto hiIt = std::upper_bound(coords.begin(), coords.end(), value);
    int hi = static_cast<int>(std::distance(coords.begin(), hiIt));
    int lo = std::max(0, hi - 1);
    double denom = std::max(coords[hi] - coords[lo], 1e-30);
    return std::pair<int, double>{lo, (value - coords[lo]) / denom};
  };
  auto [i, tx] = bracket(xs, x.x());
  auto [j, ty] = bracket(ys, x.y());
  auto value = [&](int ii, int jj) {
    const Vec& v = u[jj * nx + ii];
    return xComponent ? v.x() : v.y();
  };
  return (1.0 - tx) * (1.0 - ty) * value(i, j) +
         tx * (1.0 - ty) * value(i + 1, j) +
         (1.0 - tx) * ty * value(i, j + 1) +
         tx * ty * value(i + 1, j + 1);
}

struct CavitySampledSolution {
  int n = 0;
  std::vector<double> u;
  std::vector<double> v;
};

inline CavitySampledSolution solveCavityVorticityStream(int n, int Re, int outerIterations) {
  const double h = 1.0 / static_cast<double>(n - 1);
  const double nu = 1.0 / static_cast<double>(Re);
  const double dt = std::min(0.2 * h / 1.0, 0.2 * h * h / std::max(nu, 1e-30));
  std::vector<double> psi(n * n, 0.0), omega(n * n, 0.0), nextOmega(n * n, 0.0);
  auto id = [n](int i, int j) { return j * n + i; };
  auto applyVorticityBC = [&]() {
    for (int i = 1; i < n - 1; ++i) {
      omega[id(i, 0)] = -2.0 * psi[id(i, 1)] / (h * h);
      omega[id(i, n - 1)] = -2.0 * psi[id(i, n - 2)] / (h * h) - 2.0 / h;
    }
    for (int j = 1; j < n - 1; ++j) {
      omega[id(0, j)] = -2.0 * psi[id(1, j)] / (h * h);
      omega[id(n - 1, j)] = -2.0 * psi[id(n - 2, j)] / (h * h);
    }
  };
  for (int iter = 0; iter < outerIterations; ++iter) {
    applyVorticityBC();
    for (int sor = 0; sor < 40; ++sor) {
      for (int j = 1; j < n - 1; ++j) {
        for (int i = 1; i < n - 1; ++i) {
          double val = 0.25 * (psi[id(i + 1, j)] + psi[id(i - 1, j)] +
                               psi[id(i, j + 1)] + psi[id(i, j - 1)] +
                               h * h * omega[id(i, j)]);
          psi[id(i, j)] += 1.7 * (val - psi[id(i, j)]);
        }
      }
    }
    for (int j = 1; j < n - 1; ++j) {
      for (int i = 1; i < n - 1; ++i) {
        double u = (psi[id(i, j + 1)] - psi[id(i, j - 1)]) / (2.0 * h);
        double v = -(psi[id(i + 1, j)] - psi[id(i - 1, j)]) / (2.0 * h);
        double wx = (omega[id(i + 1, j)] - omega[id(i - 1, j)]) / (2.0 * h);
        double wy = (omega[id(i, j + 1)] - omega[id(i, j - 1)]) / (2.0 * h);
        double lap = (omega[id(i + 1, j)] + omega[id(i - 1, j)] +
                      omega[id(i, j + 1)] + omega[id(i, j - 1)] -
                      4.0 * omega[id(i, j)]) / (h * h);
        nextOmega[id(i, j)] = omega[id(i, j)] + dt * (-u * wx - v * wy + nu * lap);
      }
    }
    for (int j = 1; j < n - 1; ++j) {
      for (int i = 1; i < n - 1; ++i) omega[id(i, j)] = nextOmega[id(i, j)];
    }
  }
  CavitySampledSolution out;
  out.n = n;
  out.u.assign(n * n, 0.0);
  out.v.assign(n * n, 0.0);
  for (int i = 0; i < n; ++i) out.u[id(i, n - 1)] = 1.0;
  for (int j = 1; j < n - 1; ++j) {
    for (int i = 1; i < n - 1; ++i) {
      out.u[id(i, j)] = (psi[id(i, j + 1)] - psi[id(i, j - 1)]) / (2.0 * h);
      out.v[id(i, j)] = -(psi[id(i + 1, j)] - psi[id(i - 1, j)]) / (2.0 * h);
    }
  }
  return out;
}

inline double sampleCavityGrid(const CavitySampledSolution& s, double x, double y,
                               bool xComponent) {
  const int n = s.n;
  const double gx = std::clamp(x, 0.0, 1.0) * (n - 1);
  const double gy = std::clamp(y, 0.0, 1.0) * (n - 1);
  int i = std::min(static_cast<int>(gx), n - 2);
  int j = std::min(static_cast<int>(gy), n - 2);
  double tx = gx - i;
  double ty = gy - j;
  const auto& a = xComponent ? s.u : s.v;
  auto id = [n](int ii, int jj) { return jj * n + ii; };
  double c00 = a[id(i, j)];
  double c10 = a[id(i + 1, j)];
  double c01 = a[id(i, j + 1)];
  double c11 = a[id(i + 1, j + 1)];
  return (1.0 - tx) * (1.0 - ty) * c00 + tx * (1.0 - ty) * c10 +
         (1.0 - tx) * ty * c01 + tx * ty * c11;
}

}
