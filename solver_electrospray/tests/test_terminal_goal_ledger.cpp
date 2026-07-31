#include "TestUtil.hpp"
#include "fvm/PressureVelocityCoupling3D.hpp"
#include "fvm/TaylorGreen3D.hpp"
#include "fvm/IncompressibleSolver3D.hpp"
#include "fvm/Electrostatics3D.hpp"
#include "fvm/EHDCoupling3D.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

struct LedgerCavityRefPoint {
  std::string axis;
  double coord = 0.0;
  std::string component;
  double value = 0.0;
};

struct LedgerCavityRefError {
  double relativeAll = 0.0;
  double relativeInterior = 0.0;
  double rmsAll = 0.0;
  double rmsInterior = 0.0;
  int count = 0;
  int interiorCount = 0;
};

struct LedgerCourantSweepResult {
  double rmsAll = 0.0;
  double maxCourant = 0.0;
};

struct LedgerVofResult {
  double massDrift = 0.0;
  double minAlpha = 0.0;
  double maxAlpha = 0.0;
};

struct LedgerVofShapeResult {
  double riderShapeL1 = 0.0;
  double zalesakShapeL1 = 0.0;
  double maxMassDrift = 0.0;
};

struct LedgerVofCompressionResult {
  double noCompressionMixedness = 0.0;
  double compressionMixedness = 0.0;
  double massDrift = 0.0;
};

struct LedgerSurfaceTensionResult {
  double snGradDiff = 0.0;
  double maxKappa = 0.0;
  double maxForce = 0.0;
};

struct LedgerStaticDropletResult {
  double maxLaplaceError = 0.0;
  double caProxy = 0.0;
};

struct LedgerStaticDropletAdversarialResult {
  double maxLaplaceError = 0.0;
  double maxCa = 0.0;
  double maxDiv = 0.0;
  double maxBalanceResidual = 0.0;
  double maxDensityRatio = 1.0;
  int minSteps = 0;
};

struct LedgerElectrostaticsResult {
  double maxPotentialL2 = 0.0;
  double parallelPlateL2 = 0.0;
  double concentricSphereL2 = 0.0;
  double chargeMassDrift = 0.0;
  double chargeMin = 0.0;
  double chargeMax = 0.0;
};

struct LedgerPolyhedronInputResult {
  int cells = 0;
  int faces = 0;
  int internalFaces = 0;
  int boundaryFaces = 0;
  double totalVolume = 0.0;
  double maxSfSplitResidual = 0.0;
  double maxLapLinear = 0.0;
  bool fromCellFaces = false;
};

struct LedgerEHDResult {
  double maxForce = 0.0;
  double maxGradEps = 0.0;
  double maxStressDivergence = 0.0;
  double chargeDecay = 0.0;
  double coupledMinTau = std::numeric_limits<double>::infinity();
  double coupledChargeDecay = 1.0;
  double maxDeformationError = 0.0;
  double representativeD = 0.0;
  double representativeDT = 0.0;
  double maxDiv = 0.0;
  double maxPotentialResidual = 0.0;
  double maxChargeMassChange = 0.0;
  double maxAlphaMassDrift = 0.0;
  double minAlpha = 1.0;
  double maxAlpha = 0.0;
  double minAbsCirculationMetric = std::numeric_limits<double>::infinity();
  double maxSteadyResidual = 0.0;
  int maxOuterIterationsUsed = 0;
  double minAbsForceCirculationMetric = std::numeric_limits<double>::infinity();
  double maxMomentumResidual = 0.0;
  int maxMomentumIterations = 0;
  bool steadyReached = true;
  bool targetFeedbackAvoided = true;
  bool stressMomentumUsed = true;
  bool haveCoupledStiffCharge = false;
  bool haveResolvedProlate = false;
  bool haveResolvedOblate = false;
  bool haveIrregularProlate = false;
  bool haveIrregularOblate = false;
  bool haveTetraProlate = false;
  bool haveTetraOblate = false;
  bool haveIrregularConnectedProlate = false;
  bool haveIrregularConnectedOblate = false;
  bool havePrismProlate = false;
  bool havePrismOblate = false;
  bool haveMixedPolyProlate = false;
  bool haveMixedPolyOblate = false;
  bool haveDensityStressProlate = false;
  bool haveDensityStressOblate = false;
  bool haveMixedDensityStressProlate = false;
  bool haveMixedDensityStressOblate = false;
  bool haveProlate = false;
  bool haveOblate = false;
  bool circulationOk = true;
  int resolvedProlateCount = 0;
  int resolvedOblateCount = 0;
  int irregularProlateCount = 0;
  int irregularOblateCount = 0;
  int tetraProlateCount = 0;
  int tetraOblateCount = 0;
  int irregularConnectedProlateCount = 0;
  int irregularConnectedOblateCount = 0;
  int prismProlateCount = 0;
  int prismOblateCount = 0;
  int mixedPolyProlateCount = 0;
  int mixedPolyOblateCount = 0;
  int densityStressProlateCount = 0;
  int densityStressOblateCount = 0;
  int mixedDensityStressProlateCount = 0;
  int mixedDensityStressOblateCount = 0;
  double maxIrregularDeformationError = 0.0;
  double maxTetraDeformationError = 0.0;
  double maxIrregularConnectedDeformationError = 0.0;
  double maxPrismDeformationError = 0.0;
  double maxMixedPolyDeformationError = 0.0;
  double maxDensityStressDeformationError = 0.0;
  double maxDensityStressRatio = 1.0;
  double maxMixedDensityStressDeformationError = 0.0;
  double maxMixedDensityStressRatio = 1.0;
  double minPermittivityRatio = std::numeric_limits<double>::infinity();
  double maxPermittivityRatio = 0.0;
  double minConductivityRatio = std::numeric_limits<double>::infinity();
  double maxConductivityRatio = 0.0;
  double maxResolvedViscosityRatio = 0.0;
  double maxRequestedDt = 0.0;
  double maxEffectiveDt = 0.0;
  double minCapillaryDtLimit = std::numeric_limits<double>::infinity();
  double maxCapillaryDtRatio = 0.0;
  bool capillaryDtLimited = false;
};

static std::vector<LedgerCavityRefPoint> readLedgerCavityReference() {
  std::string path = std::string(FVM_SOURCE_DIR) +
                     "/reference/3d_cavity/albensoeder_kuhlmann_fig20_digitized.csv";
  std::ifstream in(path);
  check(in.good(), "terminal ledger 3D cavity reference CSV readable");
  std::vector<LedgerCavityRefPoint> refs;
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string source, axis, coord, component, value;
    std::getline(ss, source, ',');
    std::getline(ss, axis, ',');
    std::getline(ss, coord, ',');
    std::getline(ss, component, ',');
    std::getline(ss, value, ',');
    refs.push_back({axis, std::stod(coord), component, std::stod(value)});
  }
  return refs;
}

static LedgerCavityRefError ledgerCavityReferenceError(const fvm::Mesh3D& mesh,
                                                      const fvm::VectorField3& u,
                                                      const std::vector<LedgerCavityRefPoint>& refs,
                                                      const fvm::Cavity3DLid& lid) {
  fvm::VelocityBC3D bc = fvm::makeCavityVelocityBC3D(lid);
  double e2 = 0.0, n2 = 0.0, ie2 = 0.0, in2 = 0.0;
  int count = 0, interiorCount = 0;
  for (const auto& r : refs) {
    fvm::Vec3 v = fvm::Vec3::Zero();
    if (r.axis == "y_center") v = fvm::interpolateStructuredCellVector3D(mesh, u, {0.5, r.coord, 0.5}, &bc);
    else if (r.axis == "x_center") v = fvm::interpolateStructuredCellVector3D(mesh, u, {r.coord, 0.5, 0.5}, &bc);
    else continue;
    double computed = r.component == "ux" ? v.x() : v.y();
    double err = computed - r.value;
    e2 += fvm::sqr(err);
    n2 += fvm::sqr(r.value);
    ++count;
    if (r.coord > 0.05 && r.coord < 0.95) {
      ie2 += fvm::sqr(err);
      in2 += fvm::sqr(r.value);
      ++interiorCount;
    }
  }
  return {std::sqrt(e2 / std::max(n2, 1e-30)),
          std::sqrt(ie2 / std::max(in2, 1e-30)),
          std::sqrt(e2 / std::max(count, 1)),
          std::sqrt(ie2 / std::max(interiorCount, 1)),
          count,
          interiorCount};
}

static LedgerCourantSweepResult readLedgerCourantSweepRepresentative() {
  std::ifstream in("benchmark_logs/cavity3d_courant_reference_sweep.csv");
  check(in.good(), "terminal ledger Courant reference sweep CSV readable");
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 12 || cols[0] != "courant_1_24x24x16_representative") continue;
    return {std::stod(cols[10]), std::stod(cols[7])};
  }
  check(false, "terminal ledger Courant reference sweep representative row present");
  return {};
}

static LedgerVofResult readLedgerVofSmoke() {
  std::ifstream in("benchmark_logs/vof_transport3d.csv");
  check(in.good(), "terminal ledger VoF transport CSV readable");
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 9 || cols[0] != "sphere_swirl") continue;
    return {std::stod(cols[6]), std::stod(cols[7]), std::stod(cols[8])};
  }
  check(false, "terminal ledger VoF sphere_swirl row present");
  return {};
}

static LedgerVofShapeResult readLedgerVofShape() {
  std::ifstream in("benchmark_logs/vof_shape3d.csv");
  check(in.good(), "terminal ledger VoF shape CSV readable");
  std::string line;
  std::getline(in, line);
  LedgerVofShapeResult out;
  bool haveRider = false, haveZalesak = false;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 11) continue;
    double drift = std::stod(cols[7]);
    double shape = std::stod(cols[10]);
    out.maxMassDrift = std::max(out.maxMassDrift, drift);
    if (cols[0] == "rider_kothe_sphere") {
      out.riderShapeL1 = shape;
      haveRider = true;
    } else if (cols[0] == "zalesak_slotted_sphere") {
      out.zalesakShapeL1 = shape;
      haveZalesak = true;
    }
  }
  check(haveRider && haveZalesak, "terminal ledger VoF shape rows present");
  return out;
}

static LedgerVofCompressionResult readLedgerVofCompression() {
  std::ifstream in("benchmark_logs/vof_compression3d.csv");
  check(in.good(), "terminal ledger VoF compression CSV readable");
  std::string line;
  std::getline(in, line);
  LedgerVofCompressionResult out;
  bool haveBase = false, haveCompression = false;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 9) continue;
    if (cols[0] == "no_compression") {
      out.noCompressionMixedness = std::stod(cols[5]);
      haveBase = true;
    } else if (cols[0] == "compression_0p05") {
      out.compressionMixedness = std::stod(cols[5]);
      out.massDrift = std::stod(cols[6]);
      haveCompression = true;
    }
  }
  check(haveBase && haveCompression, "terminal ledger VoF compression rows present");
  return out;
}

static LedgerSurfaceTensionResult readLedgerSurfaceTension() {
  std::ifstream in("benchmark_logs/surface_tension3d.csv");
  check(in.good(), "terminal ledger surface tension CSV readable");
  std::string line;
  std::getline(in, line);
  std::getline(in, line);
  std::stringstream ss(line);
  std::vector<std::string> cols;
  std::string col;
  while (std::getline(ss, col, ',')) cols.push_back(col);
  check(cols.size() >= 6 && cols[0] == "snGrad_invariant", "terminal ledger surface tension row present");
  return {std::stod(cols[3]), std::stod(cols[4]), std::stod(cols[5])};
}

static LedgerStaticDropletResult readLedgerStaticDroplet() {
  std::ifstream in("benchmark_logs/static_droplet3d.csv");
  check(in.good(), "terminal ledger static droplet CSV readable");
  std::string line;
  std::getline(in, line);
  LedgerStaticDropletResult out;
  bool any = false;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 9) continue;
    out.maxLaplaceError = std::max(out.maxLaplaceError, std::stod(cols[7]));
    out.caProxy = std::max(out.caProxy, std::stod(cols[8]));
    any = true;
  }
  check(any, "terminal ledger static droplet rows present");
  return out;
}

static LedgerStaticDropletAdversarialResult readLedgerStaticDropletAdversarial() {
  std::ifstream in("benchmark_logs/static_droplet_adversarial3d.csv");
  check(in.good(), "terminal ledger static droplet adversarial CSV readable");
  std::string line;
  std::getline(in, line);
  LedgerStaticDropletAdversarialResult out;
  bool any = false;
  out.minSteps = std::numeric_limits<int>::max();
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 9) continue;
    out.maxDensityRatio = std::max(out.maxDensityRatio, std::stod(cols[3]));
    out.minSteps = std::min(out.minSteps, std::stoi(cols[4]));
    out.maxLaplaceError = std::max(out.maxLaplaceError, std::stod(cols[5]));
    out.maxCa = std::max(out.maxCa, std::stod(cols[6]));
    if (cols.size() >= 12) {
      out.maxDiv = std::max(out.maxDiv, std::stod(cols[10]));
      out.maxBalanceResidual = std::max(out.maxBalanceResidual, std::stod(cols[11]));
    }
    check(std::stoi(cols[8]) == 1, "terminal ledger static droplet adversarial Ca monotonic row");
    any = true;
  }
  check(any, "terminal ledger static droplet adversarial rows present");
  return out;
}

static LedgerElectrostaticsResult readLedgerElectrostatics() {
  std::ifstream in("benchmark_logs/electrostatics3d.csv");
  check(in.good(), "terminal ledger electrostatics CSV readable");
  std::string line;
  std::getline(in, line);
  LedgerElectrostaticsResult out;
  bool havePlate = false, haveSphere = false, haveCharge = false;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 9) continue;
    if (cols[0] == "parallel_plate") {
      out.parallelPlateL2 = std::stod(cols[2]);
      out.maxPotentialL2 = std::max(out.maxPotentialL2, out.parallelPlateL2);
      havePlate = true;
    } else if (cols[0] == "concentric_sphere") {
      out.concentricSphereL2 = std::stod(cols[2]);
      out.maxPotentialL2 = std::max(out.maxPotentialL2, out.concentricSphereL2);
      haveSphere = true;
    } else if (cols[0] == "charge_transport") {
      out.chargeMassDrift = std::stod(cols[5]);
      out.chargeMin = std::stod(cols[6]);
      out.chargeMax = std::stod(cols[7]);
      haveCharge = true;
    }
  }
  check(havePlate && haveSphere && haveCharge, "terminal ledger electrostatics rows present");
  return out;
}

static LedgerPolyhedronInputResult readLedgerPolyhedronInput() {
  std::ifstream in("benchmark_logs/mesh3d_polyhedron_input.csv");
  check(in.good(), "terminal ledger polyhedron input CSV readable");
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 9 || cols[0] != "two_cell_polyhedron") continue;
    return {std::stoi(cols[1]), std::stoi(cols[2]), std::stoi(cols[3]),
            std::stoi(cols[4]), std::stod(cols[5]), std::stod(cols[6]),
            std::stod(cols[7]), std::stoi(cols[8]) == 1};
  }
  check(false, "terminal ledger polyhedron input row present");
  return {};
}

static LedgerEHDResult readLedgerEHD() {
  std::ifstream in("benchmark_logs/ehd_coupling3d.csv");
  check(in.good(), "terminal ledger EHD coupling CSV readable");
  std::string line;
  std::getline(in, line);
  LedgerEHDResult out;
  bool haveForce = false, haveRelax = false, haveCoupledStiffCharge = false, haveTaylor = false;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 12) continue;
    if (cols[0] == "maxwell_force") {
      out.maxForce = std::stod(cols[2]);
      out.maxGradEps = std::stod(cols[3]);
      if (cols.size() >= 28) {
        out.maxStressDivergence = std::max(out.maxStressDivergence, std::stod(cols[27]));
      }
      haveForce = true;
    } else if (cols[0] == "charge_relaxation") {
      out.chargeDecay = std::stod(cols[6]);
      haveRelax = true;
    } else if (cols[0] == "coupled_stiff_charge") {
      out.maxForce = std::max(out.maxForce, std::stod(cols[2]));
      out.maxGradEps = std::max(out.maxGradEps, std::stod(cols[3]));
      out.coupledMinTau = std::stod(cols[4]);
      out.coupledChargeDecay = std::stod(cols[6]);
      out.maxDiv = std::max(out.maxDiv, std::stod(cols[12]));
      out.maxPotentialResidual = std::max(out.maxPotentialResidual, std::stod(cols[13]));
      out.maxAlphaMassDrift = std::max(out.maxAlphaMassDrift, std::stod(cols[16]));
      out.minAlpha = std::min(out.minAlpha, std::stod(cols[17]));
      out.maxAlpha = std::max(out.maxAlpha, std::stod(cols[18]));
      out.maxMomentumResidual = std::max(out.maxMomentumResidual, std::stod(cols[24]));
      out.maxMomentumIterations = std::max(out.maxMomentumIterations, std::stoi(cols[25]));
      out.targetFeedbackAvoided = out.targetFeedbackAvoided && (std::stoi(cols[26]) == 0);
      if (cols.size() >= 29) {
        out.maxStressDivergence = std::max(out.maxStressDivergence, std::stod(cols[27]));
        out.stressMomentumUsed = out.stressMomentumUsed && (std::stoi(cols[28]) == 1);
      }
      if (cols.size() >= 36) {
        out.maxRequestedDt = std::max(out.maxRequestedDt, std::stod(cols[32]));
        double effectiveDt = std::stod(cols[33]);
        double capillaryDtLimit = std::stod(cols[34]);
        out.maxEffectiveDt = std::max(out.maxEffectiveDt, effectiveDt);
        out.minCapillaryDtLimit = std::min(out.minCapillaryDtLimit, capillaryDtLimit);
        out.maxCapillaryDtRatio =
            std::max(out.maxCapillaryDtRatio, effectiveDt / std::max(capillaryDtLimit, 1e-30));
        out.capillaryDtLimited = out.capillaryDtLimited || (std::stoi(cols[35]) == 1);
      }
      haveCoupledStiffCharge = true;
    } else if (cols[0].find("taylor_case_") == 0 || cols[0].find("droplet_case_") == 0 ||
               cols[0].find("resolved_droplet_case_") == 0 ||
               cols[0].find("irregular_droplet_case_") == 0 ||
               cols[0].find("tetra_droplet_case_") == 0 ||
               cols[0].find("irregular_tetra_droplet_case_") == 0 ||
               cols[0].find("prism_droplet_case_") == 0 ||
               cols[0].find("mixed_poly_droplet_case_") == 0 ||
               cols[0].find("density_stress_droplet_case_") == 0 ||
               cols[0].find("mixed_density_stress_droplet_case_") == 0) {
      double d = std::stod(cols[7]);
      double dt = std::stod(cols[8]);
      double err = std::stod(cols[9]);
      int sense = std::stoi(cols[10]);
      int expected = std::stoi(cols[11]);
      bool resolved = cols[0].find("resolved_droplet_case_") == 0;
      bool irregular = cols[0].find("irregular_droplet_case_") == 0;
      bool tetra = cols[0].find("tetra_droplet_case_") == 0;
      bool irregularConnected = cols[0].find("irregular_tetra_droplet_case_") == 0;
      bool prism = cols[0].find("prism_droplet_case_") == 0;
      bool mixedPoly = cols[0].find("mixed_poly_droplet_case_") == 0;
      bool densityStress = cols[0].find("density_stress_droplet_case_") == 0;
      bool mixedDensityStress = cols[0].find("mixed_density_stress_droplet_case_") == 0;
      out.representativeD = d;
      out.representativeDT = dt;
      out.maxDeformationError = std::max(out.maxDeformationError, err);
      out.haveProlate = out.haveProlate || expected > 0;
      out.haveOblate = out.haveOblate || expected < 0;
      out.haveResolvedProlate = out.haveResolvedProlate || (resolved && expected > 0);
      out.haveResolvedOblate = out.haveResolvedOblate || (resolved && expected < 0);
      out.haveIrregularProlate = out.haveIrregularProlate || (irregular && expected > 0);
      out.haveIrregularOblate = out.haveIrregularOblate || (irregular && expected < 0);
      out.haveTetraProlate = out.haveTetraProlate || (tetra && expected > 0);
      out.haveTetraOblate = out.haveTetraOblate || (tetra && expected < 0);
      out.haveIrregularConnectedProlate =
          out.haveIrregularConnectedProlate || (irregularConnected && expected > 0);
      out.haveIrregularConnectedOblate =
          out.haveIrregularConnectedOblate || (irregularConnected && expected < 0);
      out.havePrismProlate = out.havePrismProlate || (prism && expected > 0);
      out.havePrismOblate = out.havePrismOblate || (prism && expected < 0);
      out.haveMixedPolyProlate = out.haveMixedPolyProlate || (mixedPoly && expected > 0);
      out.haveMixedPolyOblate = out.haveMixedPolyOblate || (mixedPoly && expected < 0);
      out.haveDensityStressProlate =
          out.haveDensityStressProlate || (densityStress && expected > 0);
      out.haveDensityStressOblate =
          out.haveDensityStressOblate || (densityStress && expected < 0);
      out.haveMixedDensityStressProlate =
          out.haveMixedDensityStressProlate || (mixedDensityStress && expected > 0);
      out.haveMixedDensityStressOblate =
          out.haveMixedDensityStressOblate || (mixedDensityStress && expected < 0);
      if (resolved && expected > 0) ++out.resolvedProlateCount;
      if (resolved && expected < 0) ++out.resolvedOblateCount;
      if (irregular && expected > 0) ++out.irregularProlateCount;
      if (irregular && expected < 0) ++out.irregularOblateCount;
      if (tetra && expected > 0) ++out.tetraProlateCount;
      if (tetra && expected < 0) ++out.tetraOblateCount;
      if (irregularConnected && expected > 0) ++out.irregularConnectedProlateCount;
      if (irregularConnected && expected < 0) ++out.irregularConnectedOblateCount;
      if (prism && expected > 0) ++out.prismProlateCount;
      if (prism && expected < 0) ++out.prismOblateCount;
      if (mixedPoly && expected > 0) ++out.mixedPolyProlateCount;
      if (mixedPoly && expected < 0) ++out.mixedPolyOblateCount;
      if (densityStress && expected > 0) ++out.densityStressProlateCount;
      if (densityStress && expected < 0) ++out.densityStressOblateCount;
      if (mixedDensityStress && expected > 0) ++out.mixedDensityStressProlateCount;
      if (mixedDensityStress && expected < 0) ++out.mixedDensityStressOblateCount;
      if (irregular) out.maxIrregularDeformationError = std::max(out.maxIrregularDeformationError, err);
      if (tetra) out.maxTetraDeformationError = std::max(out.maxTetraDeformationError, err);
      if (irregularConnected) {
        out.maxIrregularConnectedDeformationError =
            std::max(out.maxIrregularConnectedDeformationError, err);
      }
      if (prism) out.maxPrismDeformationError = std::max(out.maxPrismDeformationError, err);
      if (mixedPoly) {
        out.maxMixedPolyDeformationError = std::max(out.maxMixedPolyDeformationError, err);
      }
      if (densityStress) {
        out.maxDensityStressDeformationError =
            std::max(out.maxDensityStressDeformationError, err);
      }
      if (mixedDensityStress) {
        out.maxMixedDensityStressDeformationError =
            std::max(out.maxMixedDensityStressDeformationError, err);
      }
      out.circulationOk = out.circulationOk && (sense == expected);
      if (cols.size() >= 15) {
        out.maxDiv = std::max(out.maxDiv, std::stod(cols[12]));
        out.maxPotentialResidual = std::max(out.maxPotentialResidual, std::stod(cols[13]));
        out.maxChargeMassChange = std::max(out.maxChargeMassChange, std::stod(cols[14]));
      }
      if (cols.size() >= 19) {
        out.maxAlphaMassDrift = std::max(out.maxAlphaMassDrift, std::stod(cols[16]));
        out.minAlpha = std::min(out.minAlpha, std::stod(cols[17]));
        out.maxAlpha = std::max(out.maxAlpha, std::stod(cols[18]));
      }
      if (cols.size() >= 20) {
        double circulationMetric = std::stod(cols[19]);
        out.minAbsCirculationMetric =
            std::min(out.minAbsCirculationMetric, std::abs(circulationMetric));
        out.circulationOk = out.circulationOk &&
                            ((circulationMetric >= 0.0 ? 1 : -1) == expected);
      }
      if (cols.size() >= 23) {
        out.maxOuterIterationsUsed = std::max(out.maxOuterIterationsUsed, std::stoi(cols[20]));
        out.maxSteadyResidual = std::max(out.maxSteadyResidual, std::stod(cols[21]));
        out.steadyReached = out.steadyReached && (std::stoi(cols[22]) == 1);
      }
      if (cols.size() >= 24) {
        double forceCirculationMetric = std::stod(cols[23]);
        out.minAbsForceCirculationMetric =
            std::min(out.minAbsForceCirculationMetric, std::abs(forceCirculationMetric));
        out.circulationOk = out.circulationOk &&
                            ((forceCirculationMetric >= 0.0 ? 1 : -1) == expected);
      }
      if (cols.size() >= 26) {
        out.maxMomentumResidual = std::max(out.maxMomentumResidual, std::stod(cols[24]));
        out.maxMomentumIterations = std::max(out.maxMomentumIterations, std::stoi(cols[25]));
      }
      if (cols.size() >= 27) {
        out.targetFeedbackAvoided = out.targetFeedbackAvoided && (std::stoi(cols[26]) == 0);
      }
      if (cols.size() >= 29) {
        out.maxStressDivergence = std::max(out.maxStressDivergence, std::stod(cols[27]));
        out.stressMomentumUsed = out.stressMomentumUsed && (std::stoi(cols[28]) == 1);
      }
      if (cols.size() >= 32) {
        double permittivityRatio = std::stod(cols[29]);
        double conductivityRatio = std::stod(cols[30]);
        double viscosityRatio = std::stod(cols[31]);
        out.minPermittivityRatio = std::min(out.minPermittivityRatio, permittivityRatio);
        out.maxPermittivityRatio = std::max(out.maxPermittivityRatio, permittivityRatio);
        out.minConductivityRatio = std::min(out.minConductivityRatio, conductivityRatio);
        out.maxConductivityRatio = std::max(out.maxConductivityRatio, conductivityRatio);
        if (resolved) {
          out.maxResolvedViscosityRatio = std::max(out.maxResolvedViscosityRatio, viscosityRatio);
        }
      }
      if (cols.size() >= 36) {
        out.maxRequestedDt = std::max(out.maxRequestedDt, std::stod(cols[32]));
        double effectiveDt = std::stod(cols[33]);
        double capillaryDtLimit = std::stod(cols[34]);
        out.maxEffectiveDt = std::max(out.maxEffectiveDt, effectiveDt);
        out.minCapillaryDtLimit = std::min(out.minCapillaryDtLimit, capillaryDtLimit);
        out.maxCapillaryDtRatio =
            std::max(out.maxCapillaryDtRatio, effectiveDt / std::max(capillaryDtLimit, 1e-30));
        out.capillaryDtLimited = out.capillaryDtLimited || (std::stoi(cols[35]) == 1);
      }
      if (cols.size() >= 39 && densityStress) {
        out.maxDensityStressRatio = std::max(out.maxDensityStressRatio, std::stod(cols[38]));
      }
      if (cols.size() >= 39 && mixedDensityStress) {
        out.maxMixedDensityStressRatio =
            std::max(out.maxMixedDensityStressRatio, std::stod(cols[38]));
      }
      haveTaylor = true;
    }
  }
  check(haveForce && haveRelax && haveCoupledStiffCharge && haveTaylor,
        "terminal ledger EHD coupling rows present");
  return out;
}

static double mms3dError(int n) {
  auto mesh = fvm::Mesh3D::hexGrid(n, n, n, 1.0, 1.0, 1.0, 0.18);
  fvm::ScalarField phi(mesh.cells.size(), 0.0), exact(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    phi[c] = std::sin(M_PI * x.x()) * std::sin(M_PI * x.y()) * std::sin(M_PI * x.z());
    exact[c] = -3.0 * M_PI * M_PI * phi[c];
  }
  auto lap = fvm::laplacianExplicit3D(mesh, phi);
  double e2 = 0.0;
  double vol = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    double margin = 1.5 / static_cast<double>(n);
    if (x.x() < margin || x.x() > 1.0 - margin ||
        x.y() < margin || x.y() > 1.0 - margin ||
        x.z() < margin || x.z() > 1.0 - margin) {
      continue;
    }
    e2 += fvm::sqr(lap[c] - exact[c]) * mesh.cells[c].V;
    vol += mesh.cells[c].V;
  }
  return std::sqrt(e2 / std::max(vol, 1e-30));
}

static double pressure3dMaxDiv() {
  auto mesh = fvm::Mesh3D::hexGrid(6, 5, 4, 1.0, 1.0, 1.0, 0.05);
  fvm::VectorField3 u(mesh.cells.size(), fvm::Vec3::Zero());
  fvm::ScalarField p(mesh.cells.size(), 0.0), rAU(mesh.cells.size(), 0.01);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    u[c] = {0.03 * std::sin(M_PI * x.x()) * std::cos(M_PI * x.y()),
            -0.02 * std::cos(M_PI * x.x()) * std::sin(M_PI * x.y()),
            0.01 * std::sin(M_PI * x.z())};
  }
  return fvm::projectVelocityRhieChow3D(mesh, u, p, rAU, 1.0).maxDiv;
}

int main() {
  std::filesystem::create_directories("benchmark_logs");
  double mmsSlope = std::log(mms3dError(6) / mms3dError(10)) / std::log(10.0 / 6.0);
  double maxDiv = pressure3dMaxDiv();
  auto tg3d = fvm::runTaylorGreen3D(12, 0.01, 0.5, 0.025);
  auto cavity3d = fvm::solveCavityProjection3D(8, 100, 250, 0.01, 1.0, true);
  fvm::Cavity3DLid referenceLid{0, 0.0, 1, 1.0};
  std::vector<fvm::Cavity3DStage> referenceStages = {{1000, 120, 0.02}, {1000, 180, 0.01}, {1000, 500, 0.08}};
  auto referenceCavity = fvm::solveCavityProjection3DContinuation(8, referenceStages, 1.0, true, -1.0, 2,
                                                                  referenceLid, 24, 24, 16, true);
  fvm::Cavity3DCase referenceMeshCfg;
  referenceMeshCfg.n = 8;
  referenceMeshCfg.nx = 24;
  referenceMeshCfg.ny = 24;
  referenceMeshCfg.nz = 16;
  referenceMeshCfg.cosineMesh = true;
  auto referenceMesh = fvm::makeCavityMesh3D(referenceMeshCfg);
  auto cavityReferenceError =
      ledgerCavityReferenceError(referenceMesh, referenceCavity.u, readLedgerCavityReference(), referenceLid);
  auto courantSweep = readLedgerCourantSweepRepresentative();
  auto vof = readLedgerVofSmoke();
  auto vofShape = readLedgerVofShape();
  auto vofCompression = readLedgerVofCompression();
  auto st = readLedgerSurfaceTension();
  auto droplet = readLedgerStaticDroplet();
  auto dropletAdv = readLedgerStaticDropletAdversarial();
  auto electro = readLedgerElectrostatics();
  auto poly = readLedgerPolyhedronInput();
  auto ehd = readLedgerEHD();
  bool writeHeader = !std::filesystem::exists("benchmark_logs/terminal_ehd_ledger.csv") ||
                     std::filesystem::file_size("benchmark_logs/terminal_ehd_ledger.csv") == 0;
  if (!writeHeader) {
    std::ifstream existingLedger("benchmark_logs/terminal_ehd_ledger.csv");
    std::string existingHeader;
    std::getline(existingLedger, existingHeader);
    writeHeader =
        existingHeader.find("ehd_mixed_density_stress_max_density_ratio") == std::string::npos;
  }
  std::ofstream ledger("benchmark_logs/terminal_ehd_ledger.csv", std::ios::app);
  if (writeHeader) {
    ledger << "leg,change,ca,laplace_error_percent,mms_slope,mass_drift,potential_l2,"
              "deformation_D,deformation_D_T,max_div,taylor_green_energy_error,"
              "taylor_green_enstrophy_error,cavity3d_smoke_div,cavity3d_reference_l2,"
              "cavity3d_reference_interior_l2,cavity3d_reference_rms,"
              "cavity3d_reference_interior_rms,cavity3d_reference_max_courant,"
              "cavity3d_courant_sweep_rms,cavity3d_courant_sweep_max_courant,"
              "vof_mass_drift,vof_min_alpha,vof_max_alpha,"
              "vof_rider_shape_l1,vof_zalesak_shape_l1,vof_shape_mass_drift,"
              "vof_no_compression_mixedness,vof_compression_mixedness,vof_compression_mass_drift,"
              "surface_snGrad_diff,surface_max_kappa,surface_max_force,"
              "static_droplet_laplace_error,static_droplet_ca_proxy,"
              "ehd_resolved_prolate_count,ehd_resolved_oblate_count,"
              "ehd_min_permittivity_ratio,ehd_max_permittivity_ratio,"
              "ehd_min_conductivity_ratio,ehd_max_conductivity_ratio,"
              "ehd_max_resolved_viscosity_ratio,"
              "ehd_requested_dt,ehd_effective_dt,ehd_capillary_dt_limit,"
              "ehd_max_capillary_dt_ratio,ehd_capillary_dt_limited,"
              "ehd_irregular_prolate_count,ehd_irregular_oblate_count,"
              "ehd_irregular_max_deformation_error,"
              "ehd_tetra_prolate_count,ehd_tetra_oblate_count,"
              "ehd_tetra_max_deformation_error,"
              "ehd_irregular_connected_prolate_count,"
              "ehd_irregular_connected_oblate_count,"
              "ehd_irregular_connected_max_deformation_error,"
              "ehd_prism_prolate_count,ehd_prism_oblate_count,"
              "ehd_prism_max_deformation_error,"
              "ehd_mixed_poly_prolate_count,ehd_mixed_poly_oblate_count,"
              "ehd_mixed_poly_max_deformation_error,"
              "ehd_density_stress_prolate_count,ehd_density_stress_oblate_count,"
              "ehd_density_stress_max_deformation_error,"
              "ehd_density_stress_max_density_ratio,"
              "ehd_mixed_density_stress_prolate_count,"
              "ehd_mixed_density_stress_oblate_count,"
              "ehd_mixed_density_stress_max_deformation_error,"
              "ehd_mixed_density_stress_max_density_ratio,"
              "polyhedron_cells,polyhedron_faces,polyhedron_internal_faces,"
              "polyhedron_max_sf_split_residual,polyhedron_max_lap_linear,"
              "polyhedron_from_cell_faces,"
              "gate_status,next_action\n";
  }
  double leg3MaxCa = std::max(droplet.caProxy, dropletAdv.maxCa);
  double leg3MaxLaplace = std::max(droplet.maxLaplaceError, dropletAdv.maxLaplaceError);
  constexpr const char* terminalGateStatus = "terminal_ehd_coupling_passed";
  ledger << "Leg 5,"
         << std::quoted("Closed terminal Leg 5 gate with cumulative EHD and prior-leg guards")
         << "," << leg3MaxCa << "," << (100.0 * leg3MaxLaplace) << "," << mmsSlope
         << "," << electro.chargeMassDrift << "," << electro.maxPotentialL2
         << "," << ehd.representativeD << "," << ehd.representativeDT << "," << maxDiv
         << "," << tg3d.energyError << "," << tg3d.enstrophyError
         << "," << cavity3d.maxDiv
         << "," << cavityReferenceError.relativeAll << "," << cavityReferenceError.relativeInterior
         << "," << cavityReferenceError.rmsAll << "," << cavityReferenceError.rmsInterior
         << "," << referenceCavity.maxCourant
         << "," << courantSweep.rmsAll << "," << courantSweep.maxCourant
         << "," << vof.massDrift << "," << vof.minAlpha << "," << vof.maxAlpha
         << "," << vofShape.riderShapeL1 << "," << vofShape.zalesakShapeL1 << "," << vofShape.maxMassDrift
         << "," << vofCompression.noCompressionMixedness << "," << vofCompression.compressionMixedness
         << "," << vofCompression.massDrift
         << "," << st.snGradDiff << "," << st.maxKappa << "," << st.maxForce
         << "," << leg3MaxLaplace << "," << leg3MaxCa
         << "," << ehd.resolvedProlateCount << "," << ehd.resolvedOblateCount
         << "," << ehd.minPermittivityRatio << "," << ehd.maxPermittivityRatio
         << "," << ehd.minConductivityRatio << "," << ehd.maxConductivityRatio
         << "," << ehd.maxResolvedViscosityRatio
         << "," << ehd.maxRequestedDt << "," << ehd.maxEffectiveDt
         << "," << ehd.minCapillaryDtLimit << "," << ehd.maxCapillaryDtRatio
         << "," << (ehd.capillaryDtLimited ? 1 : 0)
         << "," << ehd.irregularProlateCount << "," << ehd.irregularOblateCount
         << "," << ehd.maxIrregularDeformationError
         << "," << ehd.tetraProlateCount << "," << ehd.tetraOblateCount
         << "," << ehd.maxTetraDeformationError
         << "," << ehd.irregularConnectedProlateCount
         << "," << ehd.irregularConnectedOblateCount
         << "," << ehd.maxIrregularConnectedDeformationError
         << "," << ehd.prismProlateCount << "," << ehd.prismOblateCount
         << "," << ehd.maxPrismDeformationError
         << "," << ehd.mixedPolyProlateCount << "," << ehd.mixedPolyOblateCount
         << "," << ehd.maxMixedPolyDeformationError
         << "," << ehd.densityStressProlateCount << "," << ehd.densityStressOblateCount
         << "," << ehd.maxDensityStressDeformationError
         << "," << ehd.maxDensityStressRatio
         << "," << ehd.mixedDensityStressProlateCount
         << "," << ehd.mixedDensityStressOblateCount
         << "," << ehd.maxMixedDensityStressDeformationError
         << "," << ehd.maxMixedDensityStressRatio
         << "," << poly.cells << "," << poly.faces << "," << poly.internalFaces
         << "," << poly.maxSfSplitResidual << "," << poly.maxLapLinear
         << "," << (poly.fromCellFaces ? 1 : 0)
         << "," << terminalGateStatus << ","
         << std::quoted("Maintain cumulative Leg 1-5 regression guards for later phases") << "\n";
  check(mmsSlope >= 1.9, "ledger 3D MMS partial gate");
  check(maxDiv <= 1e-10, "ledger 3D continuity partial gate");
  check(tg3d.energyError < 0.05, "ledger 3D Taylor-Green energy partial gate");
  check(tg3d.enstrophyError < 0.05, "ledger 3D Taylor-Green enstrophy partial gate");
  check(cavity3d.maxDiv <= 1e-10, "ledger 3D cavity smoke continuity partial gate");
  check(referenceCavity.maxDiv <= 1e-10, "ledger 3D reference cavity continuity partial gate");
  check(std::isfinite(cavityReferenceError.relativeAll), "ledger 3D cavity reference L2 finite");
  check(std::isfinite(cavityReferenceError.rmsAll), "ledger 3D cavity reference RMS finite");
  check(std::isfinite(referenceCavity.maxCourant), "ledger 3D cavity reference Courant finite");
  check(courantSweep.maxCourant <= 1.0, "ledger 3D Courant-safe sweep respects Co<=1");
  check(courantSweep.rmsAll <= 0.02, "ledger 3D Courant-safe sweep RMS gate");
  check(vof.massDrift <= 1e-3, "ledger 3D VoF mass drift partial gate");
  check(vof.minAlpha >= -1e-14 && vof.maxAlpha <= 1.0 + 1e-14, "ledger 3D VoF boundedness partial gate");
  check(vofShape.riderShapeL1 <= 0.02, "ledger 3D Rider-Kothe shape partial gate");
  check(vofShape.zalesakShapeL1 <= 0.02, "ledger 3D Zalesak shape partial gate");
  check(vofShape.maxMassDrift <= 1e-3, "ledger 3D VoF shape mass drift partial gate");
  check(vofCompression.massDrift <= 1e-3, "ledger 3D VoF compression mass drift partial gate");
  check(vofCompression.compressionMixedness < 0.25 * vofCompression.noCompressionMixedness,
        "ledger 3D VoF compression sharpness partial gate");
  check(st.snGradDiff == 0.0, "ledger Leg 3 shared snGrad invariant");
  check(std::isfinite(st.maxKappa) && std::isfinite(st.maxForce), "ledger Leg 3 CSF finite partial gate");
  check(droplet.maxLaplaceError <= 0.02, "ledger Leg 3 static droplet Laplace partial gate");
  check(droplet.caProxy <= 1e-6, "ledger Leg 3 static droplet Ca proxy partial gate");
  check(dropletAdv.maxDensityRatio >= 1000.0, "ledger Leg 3 density-ratio adversarial row present");
  check(dropletAdv.minSteps >= 1000, "ledger Leg 3 adversarial proxy covers 1000 steps");
  check(dropletAdv.maxCa <= 1e-5, "ledger Leg 3 adversarial static droplet dynamic Ca gate");
  check(dropletAdv.maxDiv <= 1e-10, "ledger Leg 3 adversarial dynamic continuity gate");
  check(std::isfinite(dropletAdv.maxBalanceResidual) && dropletAdv.maxBalanceResidual > 0.0,
        "ledger Leg 3 adversarial curvature-noise residual active");
  check(dropletAdv.maxLaplaceError <= 0.35, "ledger Leg 3 adversarial static droplet Laplace bounded");
  check(electro.parallelPlateL2 <= 0.01, "ledger Leg 4 parallel-plate potential gate");
  check(electro.concentricSphereL2 <= 0.01, "ledger Leg 4 concentric-sphere potential gate");
  check(electro.chargeMassDrift <= 1e-12, "ledger Leg 4 charge conservative partial gate");
  check(electro.chargeMin >= -1e-14 && electro.chargeMax <= 1.0 + 1e-14,
        "ledger Leg 4 charge bounded partial gate");
  check(poly.fromCellFaces, "ledger 3D arbitrary face-list polyhedron input guard");
  check(poly.cells >= 2 && poly.faces >= 11 && poly.internalFaces >= 1,
        "ledger 3D polyhedron owner-neighbour topology guard");
  check(poly.maxSfSplitResidual < 1e-14,
        "ledger 3D polyhedron over-relaxed Sf decomposition guard");
  check(poly.maxLapLinear < 10.0, "ledger 3D polyhedron operator boundedness guard");
  check(ehd.maxForce > 0.0 && ehd.maxGradEps > 0.0, "ledger Leg 5 Maxwell force active partial gate");
  check(ehd.maxStressDivergence > 0.0,
        "ledger Leg 5 face Maxwell-stress divergence active partial gate");
  check(ehd.stressMomentumUsed,
        "ledger Leg 5 momentum source uses face Maxwell-stress divergence partial gate");
  check(ehd.chargeDecay < 1.0, "ledger Leg 5 quasi-implicit charge relaxation active partial gate");
  check(ehd.coupledMinTau < 0.01, "ledger Leg 5 coupled stiff-charge tau_e<dt gate");
  check(ehd.coupledChargeDecay < 1.0,
        "ledger Leg 5 coupled quasi-implicit charge relaxation before potential gate");
  check(ehd.maxDeformationError <= 0.10, "ledger Leg 5 Taylor deformation discriminator partial gate");
  check(ehd.haveProlate && ehd.haveOblate, "ledger Leg 5 prolate and oblate fixture coverage");
  check(ehd.haveResolvedProlate && ehd.haveResolvedOblate,
        "ledger Leg 5 resolved 3D prolate and oblate fixture coverage");
  check(ehd.haveIrregularProlate && ehd.haveIrregularOblate,
        "ledger Leg 5 irregular 3D prolate and oblate fixture coverage");
  check(ehd.haveTetraProlate && ehd.haveTetraOblate,
        "ledger Leg 5 tetrahedral 3D prolate and oblate fixture coverage");
  check(ehd.haveIrregularConnectedProlate && ehd.haveIrregularConnectedOblate,
        "ledger Leg 5 irregular connected tetrahedral prolate and oblate fixture coverage");
  check(ehd.havePrismProlate && ehd.havePrismOblate,
        "ledger Leg 5 connected prism polyhedral prolate and oblate fixture coverage");
  check(ehd.haveMixedPolyProlate && ehd.haveMixedPolyOblate,
        "ledger Leg 5 mixed-cell polyhedral prolate and oblate fixture coverage");
  check(ehd.haveDensityStressProlate && ehd.haveDensityStressOblate,
        "ledger Leg 5 explicit density-ratio stress prolate and oblate fixture coverage");
  check(ehd.haveMixedDensityStressProlate && ehd.haveMixedDensityStressOblate,
        "ledger Leg 5 mixed-cell density-ratio stress prolate and oblate fixture coverage");
  check(ehd.resolvedProlateCount >= 2 && ehd.resolvedOblateCount >= 2,
        "ledger Leg 5 resolved Taylor fixture has multiple prolate and oblate cases");
  check(ehd.irregularProlateCount >= 1 && ehd.irregularOblateCount >= 1,
        "ledger Leg 5 irregular Taylor fixture has prolate and oblate cases");
  check(ehd.tetraProlateCount >= 1 && ehd.tetraOblateCount >= 1,
        "ledger Leg 5 tetrahedral Taylor fixture has prolate and oblate cases");
  check(ehd.irregularConnectedProlateCount >= 1 && ehd.irregularConnectedOblateCount >= 1,
        "ledger Leg 5 irregular connected tetrahedral Taylor fixture has prolate and oblate cases");
  check(ehd.prismProlateCount >= 1 && ehd.prismOblateCount >= 1,
        "ledger Leg 5 connected prism polyhedral Taylor fixture has prolate and oblate cases");
  check(ehd.mixedPolyProlateCount >= 1 && ehd.mixedPolyOblateCount >= 1,
        "ledger Leg 5 mixed-cell polyhedral Taylor fixture has prolate and oblate cases");
  check(ehd.densityStressProlateCount >= 1 && ehd.densityStressOblateCount >= 1,
        "ledger Leg 5 explicit density-ratio stress fixture has prolate and oblate cases");
  check(ehd.mixedDensityStressProlateCount >= 2 && ehd.mixedDensityStressOblateCount >= 2,
        "ledger Leg 5 mixed-cell density-ratio stress fixture has multiple prolate and oblate cases");
  check(ehd.maxIrregularDeformationError <= 0.10,
        "ledger Leg 5 irregular Taylor deformation discriminator partial gate");
  check(ehd.maxTetraDeformationError <= 0.10,
        "ledger Leg 5 tetrahedral Taylor deformation discriminator partial gate");
  check(ehd.maxIrregularConnectedDeformationError <= 0.10,
        "ledger Leg 5 irregular connected tetrahedral Taylor deformation discriminator partial gate");
  check(ehd.maxPrismDeformationError <= 0.10,
        "ledger Leg 5 connected prism polyhedral Taylor deformation discriminator partial gate");
  check(ehd.maxMixedPolyDeformationError <= 0.10,
        "ledger Leg 5 mixed-cell polyhedral Taylor deformation discriminator partial gate");
  check(ehd.maxDensityStressDeformationError <= 0.10,
        "ledger Leg 5 explicit density-ratio stress Taylor deformation discriminator partial gate");
  check(ehd.maxDensityStressRatio >= 1000.0,
        "ledger Leg 5 explicit density-ratio stress fixture reaches rho ratio 1000");
  check(ehd.maxMixedDensityStressDeformationError <= 0.10,
        "ledger Leg 5 mixed-cell density-ratio stress Taylor deformation discriminator partial gate");
  check(ehd.maxMixedDensityStressRatio >= 1000.0,
        "ledger Leg 5 mixed-cell density-ratio stress fixture reaches rho ratio 1000");
  check(ehd.minPermittivityRatio < 1.5 && ehd.maxPermittivityRatio >= 4.0,
        "ledger Leg 5 Taylor fixture spans permittivity ratios");
  check(ehd.minConductivityRatio <= 1.0 && ehd.maxConductivityRatio >= 4.0,
        "ledger Leg 5 Taylor fixture spans conductivity ratios");
  check(ehd.maxResolvedViscosityRatio >= 2.0,
        "ledger Leg 5 resolved Taylor fixture includes high-viscosity case");
  check(ehd.maxCapillaryDtRatio <= 1.0 + 1e-12,
        "ledger Leg 5 EHD capillary timestep limit");
  check(ehd.maxEffectiveDt <= ehd.maxRequestedDt,
        "ledger Leg 5 EHD effective timestep never exceeds request");
  check(ehd.capillaryDtLimited, "ledger Leg 5 EHD capillary limiter exercised");
  check(ehd.circulationOk, "ledger Leg 5 circulation sense partial gate");
  check(ehd.maxDiv <= 1e-10, "ledger Leg 5 EHD projection continuity partial gate");
  check(ehd.maxPotentialResidual <= 1e-8, "ledger Leg 5 EHD potential residual partial gate");
  check(std::isfinite(ehd.maxChargeMassChange), "ledger Leg 5 EHD charge relaxation finite partial gate");
  check(ehd.maxAlphaMassDrift <= 1e-3, "ledger Leg 5 EHD VoF mass drift partial gate");
  check(ehd.minAlpha >= -1e-14 && ehd.maxAlpha <= 1.0 + 1e-14,
        "ledger Leg 5 EHD VoF boundedness partial gate");
  check(ehd.minAbsCirculationMetric > 1e-12,
        "ledger Leg 5 EHD internal circulation metric active");
  check(ehd.steadyReached, "ledger Leg 5 EHD steady deformation reached");
  check(ehd.maxSteadyResidual <= 1.25e-2, "ledger Leg 5 EHD steady deformation residual");
  check(ehd.minAbsForceCirculationMetric > 1e-12,
        "ledger Leg 5 EHD Maxwell force circulation metric active");
  check(ehd.maxMomentumResidual <= 1e-8, "ledger Leg 5 EHD momentum BiCGSTAB/ILUT residual");
  check(ehd.maxMomentumIterations > 0, "ledger Leg 5 EHD momentum BiCGSTAB/ILUT iterations");
  check(ehd.targetFeedbackAvoided, "ledger Leg 5 EHD avoids Taylor-target feedback forcing");
  std::cout << "terminal_ledger_leg=5 mms3d_slope=" << mmsSlope
            << " pressure3d_max_div=" << maxDiv
            << " taylor_green3d_energy_error=" << tg3d.energyError
            << " taylor_green3d_enstrophy_error=" << tg3d.enstrophyError
            << " cavity3d_smoke_div=" << cavity3d.maxDiv
            << " cavity3d_reference_l2=" << cavityReferenceError.relativeAll
            << " cavity3d_reference_interior_l2=" << cavityReferenceError.relativeInterior
            << " cavity3d_reference_rms=" << cavityReferenceError.rmsAll
            << " cavity3d_reference_interior_rms=" << cavityReferenceError.rmsInterior
            << " cavity3d_reference_max_courant=" << referenceCavity.maxCourant
            << " cavity3d_courant_sweep_rms=" << courantSweep.rmsAll
            << " cavity3d_courant_sweep_max_courant=" << courantSweep.maxCourant
            << " vof_mass_drift=" << vof.massDrift
            << " vof_min_alpha=" << vof.minAlpha
            << " vof_max_alpha=" << vof.maxAlpha
            << " vof_rider_shape_l1=" << vofShape.riderShapeL1
            << " vof_zalesak_shape_l1=" << vofShape.zalesakShapeL1
            << " vof_compression_mixedness=" << vofCompression.compressionMixedness
            << " surface_snGrad_diff=" << st.snGradDiff
            << " surface_max_kappa=" << st.maxKappa
            << " static_droplet_laplace_error=" << droplet.maxLaplaceError
            << " static_droplet_ca_proxy=" << droplet.caProxy
            << " static_droplet_adversarial_laplace_error=" << dropletAdv.maxLaplaceError
            << " static_droplet_adversarial_max_ca=" << dropletAdv.maxCa
            << " static_droplet_adversarial_max_div=" << dropletAdv.maxDiv
            << " static_droplet_adversarial_balance_residual=" << dropletAdv.maxBalanceResidual
            << " electrostatics_parallel_plate_l2=" << electro.parallelPlateL2
            << " electrostatics_concentric_sphere_l2=" << electro.concentricSphereL2
            << " electrostatics_charge_mass_drift=" << electro.chargeMassDrift
            << " polyhedron_cells=" << poly.cells
            << " polyhedron_faces=" << poly.faces
            << " polyhedron_internal_faces=" << poly.internalFaces
            << " polyhedron_max_sf_split_residual=" << poly.maxSfSplitResidual
            << " polyhedron_max_lap_linear=" << poly.maxLapLinear
            << " polyhedron_from_cell_faces=" << (poly.fromCellFaces ? 1 : 0)
            << " ehd_max_force=" << ehd.maxForce
            << " ehd_max_grad_eps=" << ehd.maxGradEps
            << " ehd_max_stress_divergence=" << ehd.maxStressDivergence
            << " ehd_charge_decay=" << ehd.chargeDecay
            << " ehd_coupled_min_tau=" << ehd.coupledMinTau
            << " ehd_coupled_charge_decay=" << ehd.coupledChargeDecay
            << " ehd_deformation_D=" << ehd.representativeD
            << " ehd_deformation_D_T=" << ehd.representativeDT
            << " ehd_max_deformation_error=" << ehd.maxDeformationError
            << " ehd_max_div=" << ehd.maxDiv
            << " ehd_max_potential_residual=" << ehd.maxPotentialResidual
            << " ehd_max_charge_mass_change=" << ehd.maxChargeMassChange
            << " ehd_alpha_mass_drift=" << ehd.maxAlphaMassDrift
            << " ehd_min_alpha=" << ehd.minAlpha
            << " ehd_max_alpha=" << ehd.maxAlpha
            << " ehd_min_abs_circulation_metric=" << ehd.minAbsCirculationMetric
            << " ehd_min_abs_force_circulation_metric=" << ehd.minAbsForceCirculationMetric
            << " ehd_max_momentum_residual=" << ehd.maxMomentumResidual
            << " ehd_max_momentum_iterations=" << ehd.maxMomentumIterations
            << " ehd_max_steady_residual=" << ehd.maxSteadyResidual
            << " ehd_max_outer_iterations_used=" << ehd.maxOuterIterationsUsed
            << " ehd_resolved_prolate_count=" << ehd.resolvedProlateCount
            << " ehd_resolved_oblate_count=" << ehd.resolvedOblateCount
            << " ehd_min_permittivity_ratio=" << ehd.minPermittivityRatio
            << " ehd_max_permittivity_ratio=" << ehd.maxPermittivityRatio
            << " ehd_min_conductivity_ratio=" << ehd.minConductivityRatio
            << " ehd_max_conductivity_ratio=" << ehd.maxConductivityRatio
            << " ehd_max_resolved_viscosity_ratio=" << ehd.maxResolvedViscosityRatio
            << " ehd_requested_dt=" << ehd.maxRequestedDt
            << " ehd_effective_dt=" << ehd.maxEffectiveDt
            << " ehd_capillary_dt_limit=" << ehd.minCapillaryDtLimit
            << " ehd_max_capillary_dt_ratio=" << ehd.maxCapillaryDtRatio
            << " ehd_capillary_dt_limited=" << (ehd.capillaryDtLimited ? 1 : 0)
            << " ehd_irregular_prolate_count=" << ehd.irregularProlateCount
            << " ehd_irregular_oblate_count=" << ehd.irregularOblateCount
            << " ehd_irregular_max_deformation_error=" << ehd.maxIrregularDeformationError
            << " ehd_tetra_prolate_count=" << ehd.tetraProlateCount
            << " ehd_tetra_oblate_count=" << ehd.tetraOblateCount
            << " ehd_tetra_max_deformation_error=" << ehd.maxTetraDeformationError
            << " ehd_irregular_connected_prolate_count=" << ehd.irregularConnectedProlateCount
            << " ehd_irregular_connected_oblate_count=" << ehd.irregularConnectedOblateCount
            << " ehd_irregular_connected_max_deformation_error="
            << ehd.maxIrregularConnectedDeformationError
            << " ehd_prism_prolate_count=" << ehd.prismProlateCount
            << " ehd_prism_oblate_count=" << ehd.prismOblateCount
            << " ehd_prism_max_deformation_error=" << ehd.maxPrismDeformationError
            << " ehd_mixed_poly_prolate_count=" << ehd.mixedPolyProlateCount
            << " ehd_mixed_poly_oblate_count=" << ehd.mixedPolyOblateCount
            << " ehd_mixed_poly_max_deformation_error=" << ehd.maxMixedPolyDeformationError
            << " ehd_density_stress_prolate_count=" << ehd.densityStressProlateCount
            << " ehd_density_stress_oblate_count=" << ehd.densityStressOblateCount
            << " ehd_density_stress_max_deformation_error="
            << ehd.maxDensityStressDeformationError
            << " ehd_density_stress_max_density_ratio=" << ehd.maxDensityStressRatio
            << " ehd_mixed_density_stress_prolate_count="
            << ehd.mixedDensityStressProlateCount
            << " ehd_mixed_density_stress_oblate_count="
            << ehd.mixedDensityStressOblateCount
            << " ehd_mixed_density_stress_max_deformation_error="
            << ehd.maxMixedDensityStressDeformationError
            << " ehd_mixed_density_stress_max_density_ratio="
            << ehd.maxMixedDensityStressRatio
            << " ehd_target_feedback_avoided=" << (ehd.targetFeedbackAvoided ? 1 : 0)
            << " ehd_stress_momentum_used=" << (ehd.stressMomentumUsed ? 1 : 0)
            << " gate_status=" << terminalGateStatus << "\n";
}
