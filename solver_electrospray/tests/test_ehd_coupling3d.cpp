#include "TestUtil.hpp"
#include "fvm/EHDCoupling3D.hpp"
#include <filesystem>
#include <fstream>

static void runMaxwellForce(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(8, 8, 8, 1.0, 1.0, 1.0, 0.1);
  fvm::ScalarField eps(mesh.cells.size(), 0.0), rhoE(mesh.cells.size(), 0.0);
  fvm::VectorField3 E(mesh.cells.size(), fvm::Vec3::Zero());
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    double r = (x - fvm::Vec3{0.5, 0.5, 0.5}).norm();
    eps[c] = r < 0.28 ? 4.0 : 1.0;
    rhoE[c] = 0.01 * std::sin(M_PI * x.x()) * std::sin(M_PI * x.y());
    E[c] = {1.0, 0.15 * std::cos(M_PI * x.y()), -0.1 * std::sin(M_PI * x.z())};
  }
  auto report = fvm::maxwellBodyForce3D(mesh, rhoE, E, eps);
  csv << "maxwell_force," << mesh.cells.size() << "," << report.maxForce
      << "," << report.maxGradEps
      << ",nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,"
      << report.maxStressDivergence << ",nan,nan,nan,nan,nan,nan,nan\n";
  check(report.maxForce > 0.0 && std::isfinite(report.maxForce),
        "3D EHD Maxwell body force finite and active");
  check(report.maxGradEps > 0.0 && std::isfinite(report.maxGradEps),
        "3D EHD grad(eps) face-snGrad path active");
  check(report.maxStressDivergence > 0.0 && std::isfinite(report.maxStressDivergence),
        "3D EHD face Maxwell-stress divergence active");
}

static void runChargeRelaxation(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(6, 6, 6);
  fvm::ScalarField q(mesh.cells.size(), 0.0), eps(mesh.cells.size(), 0.0), sigma(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    q[c] = 0.1 + 0.2 * std::sin(M_PI * x.x()) * std::sin(M_PI * x.y()) * std::sin(M_PI * x.z());
    eps[c] = x.x() < 0.5 ? 2.0 : 1.0;
    sigma[c] = x.x() < 0.5 ? 100.0 : 0.2;
  }
  auto report = fvm::relaxChargeQuasiImplicit3D(mesh, q, eps, sigma, 0.1);
  double decay = std::abs(report.finalMass) / std::max(std::abs(report.initialMass), 1e-30);
  csv << "charge_relaxation," << mesh.cells.size() << ",nan,nan,"
      << report.minTau << "," << report.maxTau << "," << decay
      << ",nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan\n";
  check(report.minTau < 0.1, "3D EHD charge relaxation has tau_e < dt region");
  check(decay < 1.0, "3D EHD quasi-implicit charge relaxation damps charge");
}

static void runForceResponseLimiter() {
  double small = fvm::maxwellForceResponseLimiter3D(0.003);
  double moderate = fvm::maxwellForceResponseLimiter3D(0.03);
  double large = fvm::maxwellForceResponseLimiter3D(0.15);
  check(small > moderate && moderate > large,
        "3D EHD Maxwell force response limiter damps larger force metrics");
  check(std::isfinite(small) && std::isfinite(moderate) && std::isfinite(large),
        "3D EHD Maxwell force response limiter finite");
  check(fvm::maxwellForceResponseLimiter3D(0.0) == 1.0,
        "3D EHD Maxwell force response limiter leaves zero metric unchanged");
}

static void runCoupledStiffCharge(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(6, 6, 6, 1.0, 1.0, 1.0, 0.05);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 4;
  opt.minPimpleOuterIterations = 4;
  opt.steadyDeformationTolerance = 1.0;
  opt.dt = 0.01;
  opt.conductivityScale = 2000.0;
  fvm::TaylorDeformationCase3D stiff{1.2, 3.0, 0.02, 1.0};
  auto report = fvm::runLeakyDielectricDropletDiagnostic3D(mesh, stiff, opt);
  csv << "coupled_stiff_charge," << mesh.cells.size() << ","
      << report.maxForce << "," << report.maxGradEps << ","
      << report.minTau << "," << report.maxTau << ","
      << report.minChargeDecayFactor
      << ",nan,nan,nan,nan,nan," << report.maxDiv << ","
      << report.maxPotentialResidual << "," << report.chargeMassChange << ","
      << opt.pimpleOuterIterations << "," << report.alphaMassDrift << ","
      << report.minAlpha << "," << report.maxAlpha << ","
      << report.circulationMetric << "," << report.outerIterationsUsed << ","
      << report.steadyResidual << "," << (report.steadyReached ? 1 : 0) << ","
      << report.forceCirculationMetric << "," << report.maxMomentumResidual << ","
      << report.maxMomentumIterations << "," << (report.targetFeedbackUsed ? 1 : 0) << ","
      << report.maxStressDivergence << "," << (report.stressDivergenceMomentumUsed ? 1 : 0)
      << ",nan,nan,nan,"
      << report.requestedDt << "," << report.effectiveDt << "," << report.capillaryDtLimit
      << "," << (report.capillaryDtLimited ? 1 : 0) << "\n";
  check(report.minTau < opt.dt, "3D EHD coupled droplet has tau_e < dt region");
  check(report.minChargeDecayFactor < 1.0,
        "3D EHD coupled droplet applies quasi-implicit charge relaxation before potential solve");
  check(report.maxPotentialResidual <= 1e-8,
        "3D EHD coupled stiff-charge potential solve remains converged");
  check(report.maxForce > 0.0 && report.maxGradEps > 0.0,
        "3D EHD coupled stiff-charge Maxwell force remains active");
  check(report.stressDivergenceMomentumUsed,
        "3D EHD coupled stiff-charge momentum uses face Maxwell-stress divergence");
  check(report.effectiveDt <= report.capillaryDtLimit * (1.0 + 1e-12),
        "3D EHD coupled stiff-charge respects capillary timestep limit");
  check(report.capillaryDtLimited,
        "3D EHD coupled stiff-charge applies capillary timestep limiter when requested dt is too large");
}

static void writeAndCheckTaylorCase(std::ofstream& csv, const std::string& name,
                                    const fvm::Mesh3D& mesh,
                                    const fvm::TaylorDeformationCase3D& c,
                                    const fvm::LeakyDielectricDropletOptions3D& opt,
                                    bool& haveProlate, bool& haveOblate) {
  auto report = fvm::runLeakyDielectricDropletDiagnostic3D(mesh, c, opt);
  int expectedSense = fvm::taylorSmallDeformation3D(c) >= 0.0 ? 1 : -1;
  haveProlate = haveProlate || expectedSense > 0;
  haveOblate = haveOblate || expectedSense < 0;
  csv << name << "," << mesh.cells.size() << ","
      << report.maxForce << "," << report.maxGradEps << ","
      << report.minTau << "," << report.maxTau << ","
      << report.minChargeDecayFactor << ","
      << report.deformation << "," << report.taylorDeformation << ","
      << report.relativeError << "," << report.circulationSense << ","
      << expectedSense << "," << report.maxDiv << ","
      << report.maxPotentialResidual << "," << report.chargeMassChange << ","
      << opt.pimpleOuterIterations << "," << report.alphaMassDrift << ","
      << report.minAlpha << "," << report.maxAlpha << ","
      << report.circulationMetric << "," << report.outerIterationsUsed << ","
      << report.steadyResidual << "," << (report.steadyReached ? 1 : 0) << ","
      << report.forceCirculationMetric << "," << report.maxMomentumResidual << ","
      << report.maxMomentumIterations << "," << (report.targetFeedbackUsed ? 1 : 0) << ","
      << report.maxStressDivergence << "," << (report.stressDivergenceMomentumUsed ? 1 : 0)
      << "," << c.permittivityRatio << "," << c.conductivityRatio << ","
      << c.viscosityRatio << ","
      << report.requestedDt << "," << report.effectiveDt << "," << report.capillaryDtLimit
      << "," << (report.capillaryDtLimited ? 1 : 0)
      << "," << report.bodyForceCirculationMetric
      << "," << report.stressCirculationMetric
      << "," << (opt.densityRatio > 0.0 ? opt.densityRatio : c.viscosityRatio) << "\n";
  csv.flush();
  check(report.relativeError <= 0.10, "3D EHD Taylor deformation within 10%");
  check(report.circulationSense == expectedSense, "3D EHD internal velocity circulation sense correct");
  check(std::abs(report.circulationMetric) > 1e-12, "3D EHD internal circulation metric active");
  check(report.maxDiv <= 1e-10, "3D EHD droplet diagnostic Rhie-Chow projection continuity");
  check(report.maxPotentialResidual <= 1e-8, "3D EHD droplet diagnostic potential residual");
  check(report.maxForce > 0.0 && report.maxGradEps > 0.0, "3D EHD droplet diagnostic force active");
  check(report.maxMomentumResidual <= 1e-8, "3D EHD momentum BiCGSTAB/ILUT residual");
  check(report.maxMomentumIterations > 0, "3D EHD momentum BiCGSTAB/ILUT iterations recorded");
  check(!report.targetFeedbackUsed, "3D EHD droplet diagnostic avoids Taylor-target feedback forcing");
  check(report.stressDivergenceMomentumUsed,
        "3D EHD droplet momentum uses face Maxwell-stress divergence");
  check(std::abs(report.forceCirculationMetric) > 1e-12,
        "3D EHD Maxwell force circulation metric active");
  check((report.forceCirculationMetric >= 0.0 ? 1 : -1) == expectedSense,
        "3D EHD Maxwell force circulation sense correct");
  check(report.alphaMassDrift <= 1e-3, "3D EHD droplet diagnostic VoF mass conservation");
  check(report.minAlpha >= -1e-14 && report.maxAlpha <= 1.0 + 1e-14,
        "3D EHD droplet diagnostic VoF boundedness");
  check(report.steadyReached, "3D EHD droplet diagnostic steady deformation reached");
  check(report.steadyResidual <= opt.steadyDeformationTolerance,
        "3D EHD droplet diagnostic steady deformation residual");
  check(report.effectiveDt <= report.capillaryDtLimit * (1.0 + 1e-12),
        "3D EHD droplet diagnostic respects capillary timestep limit");
  check(report.effectiveDt <= report.requestedDt,
        "3D EHD droplet diagnostic never exceeds requested timestep");
}

static void runTaylorCases(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(8, 8, 8, 1.0, 1.0, 1.0, 0.08);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 120;
  opt.minPimpleOuterIterations = 20;
  opt.steadyDeformationTolerance = 1e-2;
  opt.dt = 0.002;
  opt.externalElectricField = 1.0;
  std::vector<fvm::TaylorDeformationCase3D> cases = {
      {1.2, 3.0, 0.08, 1.0},
      {4.0, 1.0, 0.08, 1.0},
      {1.5, 4.0, 0.06, 2.0},
      {4.0, 1.0, 0.08, 2.0},
  };
  bool haveProlate = false, haveOblate = false;
  for (size_t i = 0; i < cases.size(); ++i) {
    writeAndCheckTaylorCase(csv, "droplet_case_" + std::to_string(i), mesh, cases[i], opt,
                            haveProlate, haveOblate);
  }
  check(haveProlate && haveOblate, "3D EHD Taylor fixture spans prolate and oblate regimes");
}

static void runResolvedTaylorCases(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(10, 10, 10, 1.0, 1.0, 1.0, 0.06);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 140;
  opt.minPimpleOuterIterations = 24;
  opt.steadyDeformationTolerance = 1.2e-2;
  opt.dt = 0.0015;
  opt.externalElectricField = 1.0;
  std::vector<fvm::TaylorDeformationCase3D> cases = {
      {1.2, 3.0, 0.08, 1.0},
      {4.0, 1.0, 0.08, 1.0},
      {1.5, 4.0, 0.06, 2.0},
      {4.0, 1.0, 0.08, 2.0},
  };
  bool haveProlate = false, haveOblate = false;
  for (size_t i = 0; i < cases.size(); ++i) {
    writeAndCheckTaylorCase(csv, "resolved_droplet_case_" + std::to_string(i),
                            mesh, cases[i], opt, haveProlate, haveOblate);
  }
  check(haveProlate && haveOblate, "3D EHD resolved Taylor fixture spans prolate and oblate regimes");
}

static void runIrregularTaylorCases(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(8, 8, 8, 1.0, 1.0, 1.0, 0.10);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 150;
  opt.minPimpleOuterIterations = 24;
  opt.steadyDeformationTolerance = 1.25e-2;
  opt.dt = 0.0015;
  opt.externalElectricField = 1.0;
  std::vector<fvm::TaylorDeformationCase3D> cases = {
      {1.2, 3.0, 0.08, 1.0},
      {4.0, 1.0, 0.08, 1.0},
  };
  bool haveProlate = false, haveOblate = false;
  for (size_t i = 0; i < cases.size(); ++i) {
    writeAndCheckTaylorCase(csv, "irregular_droplet_case_" + std::to_string(i),
                            mesh, cases[i], opt, haveProlate, haveOblate);
  }
  check(haveProlate && haveOblate, "3D EHD irregular Taylor fixture spans prolate and oblate regimes");
}

static void runTetrahedralTaylorCases(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::tetraGrid(5, 5, 5, 1.0, 1.0, 1.0, 0.08);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 220;
  opt.minPimpleOuterIterations = 32;
  opt.steadyDeformationTolerance = 1.25e-2;
  opt.dt = 0.0012;
  opt.externalElectricField = 1.0;
  std::vector<fvm::TaylorDeformationCase3D> cases = {
      {1.2, 3.0, 0.08, 1.0},
      {4.0, 1.0, 0.08, 1.0},
  };
  bool haveProlate = false, haveOblate = false;
  for (size_t i = 0; i < cases.size(); ++i) {
    writeAndCheckTaylorCase(csv, "tetra_droplet_case_" + std::to_string(i),
                            mesh, cases[i], opt, haveProlate, haveOblate);
  }
  check(haveProlate && haveOblate,
        "3D EHD tetrahedral Taylor fixture spans prolate and oblate regimes");
}

static fvm::Mesh3D irregularConnectedTetraMesh() {
  auto mesh = fvm::Mesh3D::tetraGrid(5, 5, 5, 1.0, 1.0, 1.0, 0.02);
  for (auto& p : mesh.points) {
    bool boundary = p.x() < 1e-12 || p.x() > 1.0 - 1e-12 ||
                    p.y() < 1e-12 || p.y() > 1.0 - 1e-12 ||
                    p.z() < 1e-12 || p.z() > 1.0 - 1e-12;
    if (boundary) continue;
    double sx = std::sin(17.0 * p.x() + 3.0 * p.y() + 5.0 * p.z());
    double sy = std::sin(7.0 * p.x() + 19.0 * p.y() + 2.0 * p.z());
    double sz = std::sin(11.0 * p.x() + 13.0 * p.y() + 23.0 * p.z());
    p.x() += 0.006 * sx;
    p.y() += 0.005 * sy;
    p.z() += 0.004 * sz;
  }
  mesh.nx = 0;
  mesh.ny = 0;
  mesh.nz = 0;
  mesh.computeGeometry();
  return mesh;
}

static void runIrregularConnectedTetrahedralTaylorCases(std::ofstream& csv) {
  auto mesh = irregularConnectedTetraMesh();
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 240;
  opt.minPimpleOuterIterations = 36;
  opt.steadyDeformationTolerance = 1.25e-2;
  opt.dt = 0.0012;
  opt.externalElectricField = 1.0;
  std::vector<fvm::TaylorDeformationCase3D> cases = {
      {1.2, 3.0, 0.08, 1.0},
      {4.0, 1.0, 0.08, 1.0},
  };
  bool haveProlate = false, haveOblate = false;
  for (size_t i = 0; i < cases.size(); ++i) {
    writeAndCheckTaylorCase(csv, "irregular_tetra_droplet_case_" + std::to_string(i),
                            mesh, cases[i], opt, haveProlate, haveOblate);
  }
  check(haveProlate && haveOblate,
        "3D EHD irregular connected tetrahedral fixture spans prolate and oblate regimes");
}

static fvm::Mesh3D connectedPrismMesh(int n, double jitter) {
  const int np = n + 1;
  auto id = [np](int i, int j, int k) { return k * np * np + j * np + i; };
  std::vector<fvm::Vec3> points;
  points.reserve(np * np * np);
  for (int k = 0; k < np; ++k) {
    for (int j = 0; j < np; ++j) {
      for (int i = 0; i < np; ++i) {
        fvm::Vec3 p{static_cast<double>(i) / n,
                    static_cast<double>(j) / n,
                    static_cast<double>(k) / n};
        bool boundary = i == 0 || i == n || j == 0 || j == n || k == 0 || k == n;
        if (!boundary) {
          p.x() += jitter * std::sin(13.0 * p.x() + 5.0 * p.y() + 2.0 * p.z());
          p.y() += jitter * std::sin(3.0 * p.x() + 17.0 * p.y() + 7.0 * p.z());
          p.z() += jitter * std::sin(11.0 * p.x() + 19.0 * p.y() + 23.0 * p.z());
        }
        points.push_back(p);
      }
    }
  }

  std::vector<std::vector<std::vector<int>>> cellFaces;
  cellFaces.reserve(2 * n * n * n);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        int p000 = id(i, j, k);
        int p100 = id(i + 1, j, k);
        int p110 = id(i + 1, j + 1, k);
        int p010 = id(i, j + 1, k);
        int p001 = id(i, j, k + 1);
        int p101 = id(i + 1, j, k + 1);
        int p111 = id(i + 1, j + 1, k + 1);
        int p011 = id(i, j + 1, k + 1);
        cellFaces.push_back({
            {p000, p100, p110},
            {p001, p111, p101},
            {p000, p001, p101, p100},
            {p100, p101, p111, p110},
            {p110, p111, p001, p000},
        });
        cellFaces.push_back({
            {p000, p110, p010},
            {p001, p011, p111},
            {p000, p001, p111, p110},
            {p110, p111, p011, p010},
            {p010, p011, p001, p000},
        });
      }
    }
  }
  return fvm::Mesh3D::fromCellFaces(points, cellFaces);
}

static void addTetCell(std::vector<std::vector<std::vector<int>>>& cellFaces,
                       int a, int b, int c, int d) {
  cellFaces.push_back({
      {a, c, b},
      {a, b, d},
      {b, c, d},
      {c, a, d},
  });
}

static fvm::Mesh3D connectedMixedPolyMesh(int n, double jitter) {
  const int np = n + 1;
  auto id = [np](int i, int j, int k) { return k * np * np + j * np + i; };
  std::vector<fvm::Vec3> points;
  points.reserve(np * np * np);
  for (int k = 0; k < np; ++k) {
    for (int j = 0; j < np; ++j) {
      for (int i = 0; i < np; ++i) {
        fvm::Vec3 p{static_cast<double>(i) / n,
                    static_cast<double>(j) / n,
                    static_cast<double>(k) / n};
        bool boundary = i == 0 || i == n || j == 0 || j == n || k == 0 || k == n;
        if (!boundary) {
          p.x() += jitter * std::sin(29.0 * p.x() + 7.0 * p.y() + 3.0 * p.z());
          p.y() += jitter * std::sin(5.0 * p.x() + 23.0 * p.y() + 11.0 * p.z());
          p.z() += jitter * std::sin(17.0 * p.x() + 13.0 * p.y() + 31.0 * p.z());
        }
        points.push_back(p);
      }
    }
  }

  std::vector<std::vector<std::vector<int>>> cellFaces;
  cellFaces.reserve(4 * n * n * n);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        int p000 = id(i, j, k);
        int p100 = id(i + 1, j, k);
        int p110 = id(i + 1, j + 1, k);
        int p010 = id(i, j + 1, k);
        int p001 = id(i, j, k + 1);
        int p101 = id(i + 1, j, k + 1);
        int p111 = id(i + 1, j + 1, k + 1);
        int p011 = id(i, j + 1, k + 1);

        bool tetraCell = (i + j + k) % 2 == 0;
        if (tetraCell) {
          addTetCell(cellFaces, p000, p100, p110, p111);
          addTetCell(cellFaces, p000, p110, p010, p111);
          addTetCell(cellFaces, p000, p010, p011, p111);
          addTetCell(cellFaces, p000, p011, p001, p111);
          addTetCell(cellFaces, p000, p001, p101, p111);
          addTetCell(cellFaces, p000, p101, p100, p111);
        } else {
          cellFaces.push_back({
              {p000, p100, p110},
              {p001, p111, p101},
              {p000, p001, p101},
              {p000, p101, p100},
              {p100, p101, p111},
              {p100, p111, p110},
              {p110, p111, p001},
              {p110, p001, p000},
          });
          cellFaces.push_back({
              {p000, p110, p010},
              {p001, p011, p111},
              {p000, p001, p111},
              {p000, p111, p110},
              {p110, p111, p011},
              {p110, p011, p010},
              {p010, p011, p001},
              {p010, p001, p000},
          });
        }
      }
    }
  }
  return fvm::Mesh3D::fromCellFaces(points, cellFaces);
}

static void runPrismTaylorCases(std::ofstream& csv) {
  auto mesh = connectedPrismMesh(6, 0.002);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 280;
  opt.minPimpleOuterIterations = 44;
  opt.steadyDeformationTolerance = 1.25e-2;
  opt.dt = 0.0012;
  opt.externalElectricField = 0.80;
  bool haveProlate = false, haveOblate = false;
  writeAndCheckTaylorCase(csv, "prism_droplet_case_0", mesh, {1.2, 3.0, 0.08, 1.0},
                          opt, haveProlate, haveOblate);
  opt.externalElectricField = 0.20;
  opt.pimpleOuterIterations = 320;
  opt.minPimpleOuterIterations = 48;
  writeAndCheckTaylorCase(csv, "prism_droplet_case_1", mesh, {4.0, 1.0, 0.28, 1.0},
                          opt, haveProlate, haveOblate);
  check(haveProlate && haveOblate,
        "3D EHD connected prism polyhedral fixture spans prolate and oblate regimes");
}

static void runMixedPolyTaylorCases(std::ofstream& csv) {
  auto mesh = connectedMixedPolyMesh(5, 0.004);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 160;
  opt.minPimpleOuterIterations = 20;
  opt.steadyDeformationTolerance = 1.25e-2;
  opt.dt = 0.0012;
  opt.externalElectricField = 0.10;
  bool haveProlate = false, haveOblate = false;
  writeAndCheckTaylorCase(csv, "mixed_poly_droplet_case_0", mesh, {1.2, 3.0, 0.08, 1.0},
                          opt, haveProlate, haveOblate);
  opt.externalElectricField = 1.00;
  opt.pimpleOuterIterations = 160;
  opt.minPimpleOuterIterations = 20;
  writeAndCheckTaylorCase(csv, "mixed_poly_droplet_case_1", mesh, {4.0, 1.0, 0.04, 1.0},
                          opt, haveProlate, haveOblate);
  check(haveProlate && haveOblate,
        "3D EHD mixed tetra/prism polyhedral fixture spans prolate and oblate regimes");
}

static void runDensityStressTaylorCases(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(8, 8, 8, 1.0, 1.0, 1.0, 0.10);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 260;
  opt.minPimpleOuterIterations = 24;
  opt.steadyDeformationTolerance = 1.25e-2;
  opt.dt = 0.0012;
  opt.densityRatio = 1000.0;
  opt.externalElectricField = 0.80;
  bool haveProlate = false, haveOblate = false;
  writeAndCheckTaylorCase(csv, "density_stress_droplet_case_0", mesh,
                          {1.2, 3.0, 0.08, 1.0}, opt, haveProlate, haveOblate);
  opt.externalElectricField = 0.20;
  writeAndCheckTaylorCase(csv, "density_stress_droplet_case_1", mesh,
                          {4.0, 1.0, 0.08, 1.0}, opt, haveProlate, haveOblate);
  check(haveProlate && haveOblate,
        "3D EHD explicit density-ratio stress fixture spans prolate and oblate regimes");
}

static void runMixedDensityStressTaylorCases(std::ofstream& csv) {
  auto mesh = connectedMixedPolyMesh(5, 0.004);
  fvm::LeakyDielectricDropletOptions3D opt;
  opt.pimpleOuterIterations = 320;
  opt.minPimpleOuterIterations = 24;
  opt.steadyDeformationTolerance = 1.25e-2;
  opt.dt = 0.0012;
  opt.densityRatio = 1000.0;
  opt.externalElectricField = 0.40;
  bool haveProlate = false, haveOblate = false;
  writeAndCheckTaylorCase(csv, "mixed_density_stress_droplet_case_0", mesh,
                          {1.2, 3.0, 0.08, 1.0}, opt, haveProlate, haveOblate);
  opt.externalElectricField = 0.10;
  writeAndCheckTaylorCase(csv, "mixed_density_stress_droplet_case_1", mesh,
                          {4.0, 1.0, 0.04, 1.0}, opt, haveProlate, haveOblate);
  opt.externalElectricField = 0.20;
  writeAndCheckTaylorCase(csv, "mixed_density_stress_droplet_case_2", mesh,
                          {1.5, 4.0, 0.08, 1.0}, opt, haveProlate, haveOblate);
  opt.externalElectricField = 0.05;
  writeAndCheckTaylorCase(csv, "mixed_density_stress_droplet_case_3", mesh,
                          {5.0, 1.0, 0.16, 1.0}, opt, haveProlate, haveOblate);
  check(haveProlate && haveOblate,
        "3D EHD mixed-cell density-ratio stress fixture spans prolate and oblate regimes");
}

int main() {
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/ehd_coupling3d.csv");
  csv << "case,cells,max_force,max_grad_eps,min_tau,max_tau,charge_decay,"
         "deformation_D,deformation_D_T,deformation_relative_error,"
         "circulation_sense,expected_circulation_sense,max_div,"
         "potential_residual,charge_mass_change,pimple_outer_iterations,"
         "alpha_mass_drift,min_alpha,max_alpha,circulation_metric,"
         "outer_iterations_used,steady_residual,steady_reached,force_circulation_metric,"
         "momentum_residual,momentum_iterations,target_feedback_used,"
         "max_stress_divergence,stress_momentum_used,"
         "permittivity_ratio,conductivity_ratio,viscosity_ratio,"
         "requested_dt,effective_dt,capillary_dt_limit,capillary_dt_limited,"
         "body_force_circulation_metric,stress_circulation_metric,density_ratio\n";
  runMaxwellForce(csv);
  runChargeRelaxation(csv);
  runForceResponseLimiter();
  runCoupledStiffCharge(csv);
  runTaylorCases(csv);
  runResolvedTaylorCases(csv);
  runIrregularTaylorCases(csv);
  runTetrahedralTaylorCases(csv);
  runIrregularConnectedTetrahedralTaylorCases(csv);
  runPrismTaylorCases(csv);
  runMixedPolyTaylorCases(csv);
  runDensityStressTaylorCases(csv);
  runMixedDensityStressTaylorCases(csv);
  std::cout << "ehd_coupling3d=terminal_ehd_coupling_passed"
            << " maxwell_force=active"
            << " charge_relaxation=quasi_implicit"
            << " taylor_cases=4\n";
}
