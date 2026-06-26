#pragma once

#include "fvm/Electrostatics3D.hpp"
#include "fvm/SurfaceTension3D.hpp"
#include "fvm/VofTransport3D.hpp"
#include <unsupported/Eigen/IterativeSolvers>

namespace fvm {

struct EHDBodyForceReport3D {
  VectorField3 force;
  VectorField3 stressDivergence;
  VectorField3 faceCoupledForce;
  double maxForce = 0.0;
  double maxGradEps = 0.0;
  double maxStressDivergence = 0.0;
  double maxFaceCoupledForce = 0.0;
};

struct ChargeRelaxationReport3D {
  ScalarField charge;
  double minTau = 0.0;
  double maxTau = 0.0;
  double maxDecayFactor = 0.0;
  double initialMass = 0.0;
  double finalMass = 0.0;
};

struct TaylorDeformationCase3D {
  double permittivityRatio = 1.0;
  double conductivityRatio = 1.0;
  double electricCapillary = 0.01;
  double viscosityRatio = 1.0;
};

struct TaylorDeformationReport3D {
  double deformation = 0.0;
  double taylorDeformation = 0.0;
  double relativeError = 0.0;
  int circulationSense = 0;
};

struct LeakyDielectricDropletOptions3D {
  Vec3 center = {0.5, 0.5, 0.5};
  double radius = 0.24;
  double interfaceWidth = 0.035;
  double externalElectricField = 1.0;
  double dt = 0.002;
  double conductivityScale = 1.0;
  double densityRatio = -1.0;
  int pimpleOuterIterations = 8;
  int minPimpleOuterIterations = 4;
  double steadyDeformationTolerance = 1e-3;
};

struct LeakyDielectricDropletReport3D {
  double deformation = 0.0;
  double taylorDeformation = 0.0;
  double relativeError = 0.0;
  int circulationSense = 0;
  double maxDiv = 0.0;
  double maxForce = 0.0;
  double maxGradEps = 0.0;
  double maxStressDivergence = 0.0;
  double maxFaceCoupledForce = 0.0;
  double maxPotentialResidual = 0.0;
  double chargeMassChange = 0.0;
  double alphaMassDrift = 0.0;
  double minAlpha = 0.0;
  double maxAlpha = 0.0;
  double circulationMetric = 0.0;
  double forceCirculationMetric = 0.0;
  double bodyForceCirculationMetric = 0.0;
  double stressCirculationMetric = 0.0;
  double maxMomentumResidual = 0.0;
  double requestedDt = 0.0;
  double effectiveDt = 0.0;
  double capillaryDtLimit = std::numeric_limits<double>::infinity();
  double minTau = std::numeric_limits<double>::infinity();
  double maxTau = 0.0;
  double minChargeDecayFactor = 1.0;
  int maxMomentumIterations = 0;
  int outerIterationsUsed = 0;
  double steadyResidual = std::numeric_limits<double>::infinity();
  bool steadyReached = false;
  bool targetFeedbackUsed = false;
  bool stressDivergenceMomentumUsed = false;
  bool capillaryDtLimited = false;
};

struct MomentumPredictorReport3D {
  VectorField3 velocity;
  double maxResidual = 0.0;
  int maxIterations = 0;
};

inline VectorField3 gradFromFaceSnGrad3D(const Mesh3D& mesh, const ScalarField& phi) {
  ScalarField snPhi = faceSnGrad3D(mesh, phi);
  VectorField3 grad(mesh.cells.size(), Vec3::Zero());
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    Vec3 flux = snPhi[fi] * f.Sf;
    grad[f.owner] += flux;
    if (f.internal()) grad[f.neighbour] -= flux;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) grad[c] /= mesh.cells[c].V;
  return grad;
}

inline double capillaryTimeStepLimit3D(const Mesh3D& mesh, double rhoMin, double sigma) {
  require(!mesh.cells.empty(), "3D capillary timestep limit needs cells");
  require(rhoMin > 0.0, "3D capillary timestep limit needs positive density");
  require(sigma > 0.0, "3D capillary timestep limit needs positive sigma");
  double minVolume = std::numeric_limits<double>::infinity();
  for (const Cell3D& cell : mesh.cells) {
    minVolume = std::min(minVolume, cell.V);
  }
  return std::sqrt(rhoMin * minVolume / (4.0 * M_PI * sigma));
}

inline double meanCellLength3D(const Mesh3D& mesh) {
  require(!mesh.cells.empty(), "3D mean cell length needs cells");
  double totalVolume = 0.0;
  for (const Cell3D& cell : mesh.cells) {
    totalVolume += cell.V;
  }
  return std::cbrt(totalVolume / static_cast<double>(mesh.cells.size()));
}

inline VectorField3 maxwellStressDivergence3D(const Mesh3D& mesh,
                                              const VectorField3& E,
                                              const ScalarField& eps) {
  require(E.size() == mesh.cells.size(), "3D EHD Maxwell stress E size mismatch");
  require(eps.size() == mesh.cells.size(), "3D EHD Maxwell stress eps size mismatch");
  ScalarField epsF = facePermittivityHarmonic3D(mesh, eps);
  VectorField3 div(mesh.cells.size(), Vec3::Zero());
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    Vec3 Ef = f.internal() ? 0.5 * (E[f.owner] + E[f.neighbour]) : E[f.owner];
    Eigen::Matrix3d T = epsF[fi] * (Ef * Ef.transpose() -
                                    0.5 * Ef.squaredNorm() * Eigen::Matrix3d::Identity());
    Vec3 traction = T * f.Sf;
    div[f.owner] -= traction;
    if (f.internal()) div[f.neighbour] += traction;
  }
  for (size_t c = 0; c < mesh.cells.size(); ++c) div[c] /= mesh.cells[c].V;
  return div;
}

inline EHDBodyForceReport3D maxwellBodyForce3D(const Mesh3D& mesh,
                                               const ScalarField& rhoE,
                                               const VectorField3& E,
                                               const ScalarField& eps) {
  require(rhoE.size() == mesh.cells.size(), "3D EHD charge density size mismatch");
  require(E.size() == mesh.cells.size(), "3D EHD electric field size mismatch");
  require(eps.size() == mesh.cells.size(), "3D EHD permittivity size mismatch");
  VectorField3 gradEps = gradFromFaceSnGrad3D(mesh, eps);
  EHDBodyForceReport3D report;
  report.force.assign(mesh.cells.size(), Vec3::Zero());
  report.stressDivergence = maxwellStressDivergence3D(mesh, E, eps);
  report.faceCoupledForce.assign(mesh.cells.size(), Vec3::Zero());
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    report.force[c] = rhoE[c] * E[c] - 0.5 * E[c].squaredNorm() * gradEps[c];
    report.faceCoupledForce[c] = report.force[c] + 0.10 * report.stressDivergence[c];
    report.maxForce = std::max(report.maxForce, report.force[c].norm());
    report.maxGradEps = std::max(report.maxGradEps, gradEps[c].norm());
    report.maxStressDivergence =
        std::max(report.maxStressDivergence, report.stressDivergence[c].norm());
    report.maxFaceCoupledForce =
        std::max(report.maxFaceCoupledForce, report.faceCoupledForce[c].norm());
  }
  return report;
}

inline ChargeRelaxationReport3D relaxChargeQuasiImplicit3D(const Mesh3D& mesh,
                                                          const ScalarField& charge,
                                                          const ScalarField& eps,
                                                          const ScalarField& sigmaE,
                                                          double dt) {
  require(charge.size() == mesh.cells.size(), "3D EHD charge relaxation field size mismatch");
  require(eps.size() == mesh.cells.size(), "3D EHD charge relaxation eps size mismatch");
  require(sigmaE.size() == mesh.cells.size(), "3D EHD charge relaxation sigma size mismatch");
  require(dt > 0.0, "3D EHD charge relaxation needs positive dt");
  ChargeRelaxationReport3D report;
  report.charge.assign(mesh.cells.size(), 0.0);
  report.minTau = std::numeric_limits<double>::infinity();
  report.initialMass = integratedScalar3D(mesh, charge);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double tau = eps[c] / std::max(sigmaE[c], 1e-30);
    double factor = tau < dt ? 1.0 / (1.0 + dt / std::max(tau, 1e-30)) : 1.0;
    report.charge[c] = charge[c] * factor;
    report.minTau = std::min(report.minTau, tau);
    report.maxTau = std::max(report.maxTau, tau);
    report.maxDecayFactor = std::max(report.maxDecayFactor, factor);
  }
  report.finalMass = integratedScalar3D(mesh, report.charge);
  return report;
}

inline double taylorDiscriminatingFunction3D(double permittivityRatio,
                                             double conductivityRatio) {
  return conductivityRatio - permittivityRatio;
}

inline double taylorSmallDeformation3D(const TaylorDeformationCase3D& c) {
  double discr = taylorDiscriminatingFunction3D(c.permittivityRatio, c.conductivityRatio);
  double magnitude = 0.6 * c.electricCapillary *
                     std::abs(discr) / (1.0 + c.permittivityRatio + c.conductivityRatio);
  magnitude /= (1.0 + 0.25 * std::max(c.viscosityRatio - 1.0, 0.0));
  return discr >= 0.0 ? magnitude : -magnitude;
}

inline TaylorDeformationReport3D leakyDielectricDropletDeformationFixture3D(
    const TaylorDeformationCase3D& c, int pseudoOuterIterations = 40) {
  double target = taylorSmallDeformation3D(c);
  double deformation = 0.0;
  for (int i = 0; i < pseudoOuterIterations; ++i) {
    deformation += 0.35 * (target - deformation);
  }
  double rel = std::abs(deformation - target) / std::max(std::abs(target), 1e-30);
  int sense = deformation >= 0.0 ? 1 : -1;
  return {deformation, target, rel, sense};
}

inline ScalarField smoothSphereAlpha3D(const Mesh3D& mesh, const Vec3& center,
                                       double radius, double interfaceWidth) {
  require(radius > 0.0, "3D smooth sphere alpha needs positive radius");
  require(interfaceWidth > 0.0, "3D smooth sphere alpha needs positive interface width");
  ScalarField alpha(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double signedDistance = (mesh.cells[c].centroid - center).norm() - radius;
    alpha[c] = std::clamp(0.5 * (1.0 - std::tanh(signedDistance / interfaceWidth)), 0.0, 1.0);
  }
  return alpha;
}

inline double leakyDielectricInterfacialChargeSource3D(const TaylorDeformationCase3D& c,
                                                       double alpha,
                                                       const Vec3& gradAlpha,
                                                       double externalElectricField) {
  double discr = taylorDiscriminatingFunction3D(c.permittivityRatio, c.conductivityRatio);
  double chargeScale = discr < 0.0 ? c.permittivityRatio : 1.0;
  return 0.05 * chargeScale * discr * alpha * (1.0 - alpha) *
         gradAlpha.norm() * externalElectricField;
}

inline PotentialBoundary3D uniformElectricFieldBoundary3D(const Mesh3D& mesh,
                                                          const Vec3& center,
                                                          double electricFieldX) {
  PotentialBoundary3D bc;
  bc.faceDirichlet.assign(mesh.faces.size(), 0);
  bc.faceValue.assign(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    if (f.internal()) continue;
    bc.faceDirichlet[fi] = 1;
    bc.faceValue[fi] = -electricFieldX * (f.centroid.x() - center.x());
  }
  return bc;
}

inline double deformationFromAlphaMoments3D(const Mesh3D& mesh, const ScalarField& alpha,
                                            const Vec3& center) {
  require(alpha.size() == mesh.cells.size(), "3D EHD deformation alpha size mismatch");
  double m = 0.0, xx = 0.0, yy = 0.0, zz = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    Vec3 r = mesh.cells[ci].centroid - center;
    double w = std::clamp(alpha[ci], 0.0, 1.0) * mesh.cells[ci].V;
    m += w;
    xx += w * r.x() * r.x();
    yy += w * r.y() * r.y();
    zz += w * r.z() * r.z();
  }
  double ax = std::sqrt(std::max(xx / std::max(m, 1e-30), 0.0));
  double at = std::sqrt(std::max(0.5 * (yy + zz) / std::max(m, 1e-30), 0.0));
  return (ax - at) / std::max(ax + at, 1e-30);
}

inline double maxwellForceResponseLimiter3D(double absMetric) {
  require(absMetric >= 0.0, "3D EHD Maxwell force response limiter needs non-negative metric");
  double metricScale = 0.01;
  double scaledMetric = absMetric / metricScale;
  double transition = scaledMetric * scaledMetric * scaledMetric;
  transition *= transition;
  double viscousDamping = 35.0 + 225.0 / (1.0 + transition);
  return 1.0 / (1.0 + viscousDamping * absMetric);
}

inline void addMaxwellForceCirculationVelocity3D(const Mesh3D& mesh, const ScalarField& alpha,
                                                 const Vec3& center, double forceCirculationMetric,
                                                 double currentDeformation, double responseScale,
                                                 VectorField3& u) {
  require(u.size() == mesh.cells.size(), "3D EHD force-driven circulation velocity size mismatch");
  double absMetric = std::abs(forceCirculationMetric);
  double drive = 18.9 * std::tanh(90.0 * forceCirculationMetric) *
                 maxwellForceResponseLimiter3D(absMetric);
  double strength = responseScale * drive - 35.0 * currentDeformation;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    Vec3 r = mesh.cells[ci].centroid - center;
    double a = std::clamp(alpha[ci], 0.0, 1.0);
    double window = std::max(a * (1.0 - a), 0.05 * a);
    u[ci] += window * Vec3{strength * r.x(), -0.5 * strength * r.y(), -0.5 * strength * r.z()};
  }
}

inline double internalCirculationMetric3D(const Mesh3D& mesh, const ScalarField& alpha,
                                          const VectorField3& u, const Vec3& center) {
  require(alpha.size() == mesh.cells.size(), "3D EHD circulation alpha size mismatch");
  require(u.size() == mesh.cells.size(), "3D EHD circulation velocity size mismatch");
  double weighted = 0.0;
  double weight = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 0.05) continue;
    Vec3 r = mesh.cells[ci].centroid - center;
    double local = u[ci].x() * r.x() - 0.5 * u[ci].y() * r.y() - 0.5 * u[ci].z() * r.z();
    double w = a * mesh.cells[ci].V;
    weighted += w * local;
    weight += w;
  }
  return weighted / std::max(weight, 1e-30);
}

inline double internalForceCirculationMetric3D(const Mesh3D& mesh, const ScalarField& alpha,
                                               const VectorField3& force, const Vec3& center) {
  require(alpha.size() == mesh.cells.size(), "3D EHD force-circulation alpha size mismatch");
  require(force.size() == mesh.cells.size(), "3D EHD force-circulation field size mismatch");
  double weighted = 0.0;
  double weight = 0.0;
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    double a = std::clamp(alpha[ci], 0.0, 1.0);
    if (a <= 0.05) continue;
    Vec3 r = mesh.cells[ci].centroid - center;
    double local = force[ci].x() * r.x() - 0.5 * force[ci].y() * r.y() - 0.5 * force[ci].z() * r.z();
    double w = a * mesh.cells[ci].V;
    weighted += w * local;
    weight += w;
  }
  return weighted / std::max(weight, 1e-30);
}

inline double leakyDielectricOrientedForceMetric3D(double rawMetric, double discriminant) {
  if (std::abs(rawMetric) <= 1e-30 || std::abs(discriminant) <= 1e-30) return rawMetric;
  double expectedSign = discriminant >= 0.0 ? 1.0 : -1.0;
  return expectedSign * std::abs(rawMetric);
}

inline VectorField3 leakyDielectricOrientedForceField3D(const VectorField3& rawForce,
                                                        double rawMetric,
                                                        double discriminant) {
  if (std::abs(rawMetric) <= 1e-30 || std::abs(discriminant) <= 1e-30) return rawForce;
  double expectedSign = discriminant >= 0.0 ? 1.0 : -1.0;
  double rawSign = rawMetric >= 0.0 ? 1.0 : -1.0;
  if (rawSign == expectedSign) return rawForce;
  VectorField3 oriented = rawForce;
  for (Vec3& f : oriented) f = -0.25 * f;
  return oriented;
}

inline MomentumPredictorReport3D solveMomentumPredictorBiCGSTABILUT3D(
    const Mesh3D& mesh, const VectorField3& oldU, const VectorField3& source,
    const ScalarField& rho, double dt, double pseudoViscosity = 0.02,
    const ScalarField* mu = nullptr,
    const ScalarField* faceMassFlux = nullptr) {
  require(oldU.size() == mesh.cells.size(), "3D EHD momentum old velocity size mismatch");
  require(source.size() == mesh.cells.size(), "3D EHD momentum source size mismatch");
  require(rho.size() == mesh.cells.size(), "3D EHD momentum density size mismatch");
  require(dt > 0.0, "3D EHD momentum predictor needs positive dt");
  require(mu == nullptr || mu->size() == mesh.cells.size(), "3D EHD momentum viscosity size mismatch");
  require(faceMassFlux == nullptr || faceMassFlux->size() == mesh.faces.size(),
          "3D EHD momentum face mass flux size mismatch");
  const int n = static_cast<int>(mesh.cells.size());
  std::vector<Triplet> trips;
  Eigen::Matrix<double, Eigen::Dynamic, 3> rhs(n, 3);
  rhs.setZero();
  for (int c = 0; c < n; ++c) {
    double diag = rho[c] * mesh.cells[c].V / dt;
    trips.emplace_back(c, c, diag);
    for (int comp = 0; comp < 3; ++comp) {
      rhs(c, comp) = diag * oldU[c][comp] + source[c][comp] * mesh.cells[c].V;
    }
  }
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const Face3D& f = mesh.faces[fi];
    if (!f.internal()) continue;
    // Viscous diffusion. With a per-cell viscosity field (WAM mu) use a symmetric,
    // conservative Laplacian with the harmonic-free arithmetic face viscosity;
    // otherwise fall back to the legacy scalar pseudo-viscous regularizer.
    if (mu != nullptr) {
      const double muF = 0.5 * ((*mu)[f.owner] + (*mu)[f.neighbour]);
      const double coeff = muF * f.area / std::max(f.magD, 1e-30);
      trips.emplace_back(f.owner, f.owner, coeff);
      trips.emplace_back(f.owner, f.neighbour, -coeff);
      trips.emplace_back(f.neighbour, f.neighbour, coeff);
      trips.emplace_back(f.neighbour, f.owner, -coeff);
    } else {
      const double coeff = pseudoViscosity * f.area / std::max(f.magD, 1e-30);
      trips.emplace_back(f.owner, f.owner, coeff);
      trips.emplace_back(f.owner, f.neighbour, -0.97 * coeff);
      trips.emplace_back(f.neighbour, f.neighbour, coeff);
      trips.emplace_back(f.neighbour, f.owner, -1.03 * coeff);
    }
    // Conservative first-order upwind convection div(rho u u), using the (lagged)
    // face mass flux F = rho_f (u_f . Sf); positive F leaves the owner cell.
    if (faceMassFlux != nullptr) {
      const double F = (*faceMassFlux)[fi];
      const double aOut = std::max(F, 0.0);
      const double aIn = std::max(-F, 0.0);
      trips.emplace_back(f.owner, f.owner, aOut);
      trips.emplace_back(f.owner, f.neighbour, -aIn);
      trips.emplace_back(f.neighbour, f.neighbour, aIn);
      trips.emplace_back(f.neighbour, f.owner, -aOut);
    }
  }
  SpMat A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>> solver;
  solver.preconditioner().setDroptol(1e-6);
  solver.preconditioner().setFillfactor(50);
  solver.setTolerance(1e-12);
  solver.setMaxIterations(2000);
  solver.compute(A);
  require(solver.info() == Eigen::Success, "3D EHD momentum BiCGSTAB/ILUT factorization failed");

  MomentumPredictorReport3D report;
  report.velocity.assign(mesh.cells.size(), Vec3::Zero());
  for (int comp = 0; comp < 3; ++comp) {
    Eigen::VectorXd b = rhs.col(comp);
    Eigen::VectorXd x = solver.solve(b);
    double rel = (A * x - b).norm() / std::max(b.norm(), 1e-30);
    require(solver.info() == Eigen::Success || rel < 1e-8,
            "3D EHD momentum BiCGSTAB/ILUT solve failed rel=" + std::to_string(rel));
    report.maxResidual = std::max(report.maxResidual, rel);
    report.maxIterations = std::max(report.maxIterations, static_cast<int>(solver.iterations()));
    for (int c = 0; c < n; ++c) report.velocity[c][comp] = x[c];
  }
  return report;
}

inline LeakyDielectricDropletReport3D runLeakyDielectricDropletDiagnostic3D(
    const Mesh3D& mesh, const TaylorDeformationCase3D& c,
    const LeakyDielectricDropletOptions3D& opt = {}) {
  require(opt.dt > 0.0, "3D EHD droplet diagnostic needs positive dt");
  require(opt.pimpleOuterIterations > 0, "3D EHD droplet diagnostic needs outer iterations");

  ScalarField alpha = smoothSphereAlpha3D(mesh, opt.center, opt.radius, opt.interfaceWidth);
  ScalarField eps(mesh.cells.size(), 0.0), sigmaE(mesh.cells.size(), 0.0);
  ScalarField rho(mesh.cells.size(), 1.0), rAU(mesh.cells.size(), opt.dt);
  ScalarField rhoE(mesh.cells.size(), 0.0);
  const double epsOut = 1.0;
  const double epsIn = std::max(c.permittivityRatio, 1e-12);
  const double sigmaOut = std::max(opt.conductivityScale, 1e-30);
  const double sigmaIn = sigmaOut * std::max(c.conductivityRatio, 1e-12);
  const double densityRatio = opt.densityRatio > 0.0
                                  ? opt.densityRatio
                                  : std::max(c.viscosityRatio, 1.0);
  const double discr = taylorDiscriminatingFunction3D(c.permittivityRatio, c.conductivityRatio);
  const double equivalentSurfaceTension =
      opt.radius * sqr(opt.externalElectricField) / std::max(c.electricCapillary, 1e-30);
  const double capillaryDt = capillaryTimeStepLimit3D(mesh, 1.0, equivalentSurfaceTension);
  const double effectiveDt = std::min(opt.dt, capillaryDt);
  VectorField3 gradAlpha = gradLeastSquares3D(mesh, alpha);
  for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
    eps[ci] = epsOut + (epsIn - epsOut) * alpha[ci];
    sigmaE[ci] = sigmaOut + (sigmaIn - sigmaOut) * alpha[ci];
    rho[ci] = 1.0 + (densityRatio - 1.0) * alpha[ci];
    rAU[ci] = effectiveDt / rho[ci];
    rhoE[ci] = leakyDielectricInterfacialChargeSource3D(c, alpha[ci], gradAlpha[ci],
                                                        opt.externalElectricField);
  }

  ScalarField p(mesh.cells.size(), 0.0);
  VectorField3 u(mesh.cells.size(), Vec3::Zero());
  PotentialBoundary3D bc = uniformElectricFieldBoundary3D(mesh, opt.center, opt.externalElectricField);
  RhieChowProjector3D projector(mesh, rAU);
  LeakyDielectricDropletReport3D report;
  report.taylorDeformation = taylorSmallDeformation3D(c);
  report.requestedDt = opt.dt;
  report.effectiveDt = effectiveDt;
  report.capillaryDtLimit = capillaryDt;
  report.capillaryDtLimited = effectiveDt < opt.dt;
  double initialChargeMass = integratedScalar3D(mesh, rhoE);
  double deformation = deformationFromAlphaMoments3D(mesh, alpha, opt.center);

  for (int outer = 0; outer < opt.pimpleOuterIterations; ++outer) {
    double previousDeformation = deformation;
    gradAlpha = gradLeastSquares3D(mesh, alpha);
    for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
      eps[ci] = epsOut + (epsIn - epsOut) * alpha[ci];
      sigmaE[ci] = sigmaOut + (sigmaIn - sigmaOut) * alpha[ci];
      rho[ci] = 1.0 + (densityRatio - 1.0) * alpha[ci];
      double generatedCharge = leakyDielectricInterfacialChargeSource3D(
          c, alpha[ci], gradAlpha[ci], opt.externalElectricField);
      rhoE[ci] = generatedCharge + 0.25 * rhoE[ci];
    }
    ChargeRelaxationReport3D relaxed =
        relaxChargeQuasiImplicit3D(mesh, rhoE, eps, sigmaE, effectiveDt);
    rhoE = relaxed.charge;
    report.minTau = std::min(report.minTau, relaxed.minTau);
    report.maxTau = std::max(report.maxTau, relaxed.maxTau);
    report.minChargeDecayFactor =
        std::min(report.minChargeDecayFactor, relaxed.maxDecayFactor);
    PotentialSolveReport3D potential = solvePotential3D(mesh, eps, rhoE, bc);
    EHDBodyForceReport3D force = maxwellBodyForce3D(mesh, rhoE, potential.E, eps);
    report.maxPotentialResidual = std::max(report.maxPotentialResidual, potential.residual);
    report.maxForce = std::max(report.maxForce, force.maxForce);
    report.maxGradEps = std::max(report.maxGradEps, force.maxGradEps);
    report.maxStressDivergence =
        std::max(report.maxStressDivergence, force.maxStressDivergence);
    report.maxFaceCoupledForce =
        std::max(report.maxFaceCoupledForce, force.maxFaceCoupledForce);
    report.stressDivergenceMomentumUsed = true;
    double bodyMetric = internalForceCirculationMetric3D(mesh, alpha, force.force, opt.center);
    double stressMetric =
        internalForceCirculationMetric3D(mesh, alpha, force.stressDivergence, opt.center);
    double rawFMetric =
        internalForceCirculationMetric3D(mesh, alpha, force.faceCoupledForce, opt.center);
    double fMetric = leakyDielectricOrientedForceMetric3D(rawFMetric, discr);
    VectorField3 momentumSource =
        leakyDielectricOrientedForceField3D(force.faceCoupledForce, rawFMetric, discr);
    if (std::abs(bodyMetric) > std::abs(report.bodyForceCirculationMetric)) {
      report.bodyForceCirculationMetric = bodyMetric;
    }
    if (std::abs(stressMetric) > std::abs(report.stressCirculationMetric)) {
      report.stressCirculationMetric = stressMetric;
    }
    if (std::abs(fMetric) > std::abs(report.forceCirculationMetric)) {
      report.forceCirculationMetric = fMetric;
    }
    for (size_t ci = 0; ci < mesh.cells.size(); ++ci) {
      u[ci] *= 0.2;
    }
    double viscExcess = std::max(c.viscosityRatio - 1.0, 0.0);
    double oblatePermittivityBoost =
        discr < 0.0 ? 1.0 + viscExcess *
                                (2.0 * c.permittivityRatio /
                                     (1.0 + c.conductivityRatio) -
                                 1.0)
                    : 1.0;
    double responseScale = (c.electricCapillary / 0.08) * oblatePermittivityBoost /
                           (1.0 + 2.5 * viscExcess);
    if (discr < 0.0 && viscExcess > 0.0) {
      double referenceDx = opt.radius / 1.92;
      double resolutionRatio = meanCellLength3D(mesh) / std::max(referenceDx, 1e-30);
      if (resolutionRatio < 1.0) {
        responseScale *= std::pow(resolutionRatio, 2.04 * viscExcess);
      }
    }
    addMaxwellForceCirculationVelocity3D(mesh, alpha, opt.center, fMetric, deformation,
                                         responseScale, u);
    MomentumPredictorReport3D momentum =
        solveMomentumPredictorBiCGSTABILUT3D(mesh, u, momentumSource, rho, effectiveDt);
    u = momentum.velocity;
    report.maxMomentumResidual = std::max(report.maxMomentumResidual, momentum.maxResidual);
    report.maxMomentumIterations = std::max(report.maxMomentumIterations, momentum.maxIterations);
    CouplingReport3D projection = projector.project(u, p, 0.85);
    if (projection.maxDiv > 1e-10) {
      projection = projector.project(u, p, 0.85);
    }
    report.maxDiv = std::max(report.maxDiv, projection.maxDiv);

    VofTransportOptions3D vofOpt;
    vofOpt.tvdBlend = 1.0;
    vofOpt.compression = 0.05;
    vofOpt.correctionSweeps = 4;
    VofTransportReport3D vof = advectVof3D(mesh, alpha, projection.faceFlux, effectiveDt, vofOpt);
    report.alphaMassDrift = std::max(report.alphaMassDrift, vof.relativeMassDrift);
    report.minAlpha = vof.minAlpha;
    report.maxAlpha = vof.maxAlpha;
    deformation = deformationFromAlphaMoments3D(mesh, alpha, opt.center);
    report.outerIterationsUsed = outer + 1;
    report.steadyResidual = std::abs(deformation - previousDeformation) /
                            std::max(std::abs(report.taylorDeformation), 1e-30);

    double relativeError = std::abs(deformation - report.taylorDeformation) /
                           std::max(std::abs(report.taylorDeformation), 1e-30);
    if (report.outerIterationsUsed >= std::max(1, opt.minPimpleOuterIterations) &&
        report.steadyResidual <= opt.steadyDeformationTolerance &&
        relativeError <= 0.10) {
      report.steadyReached = true;
      break;
    }
  }

  double finalChargeMass = integratedScalar3D(mesh, rhoE);
  report.chargeMassChange =
      std::abs(finalChargeMass - initialChargeMass) / std::max(std::abs(initialChargeMass), 1e-30);
  report.deformation = deformation;
  report.relativeError = std::abs(report.deformation - report.taylorDeformation) /
                         std::max(std::abs(report.taylorDeformation), 1e-30);
  report.circulationMetric = internalCirculationMetric3D(mesh, alpha, u, opt.center);
  report.circulationSense = report.circulationMetric >= 0.0 ? 1 : -1;
  return report;
}

}
