#pragma once
// Named-patch boundary-condition framework for the resolved-nozzle Candido setup (P7, task #15).
//
// The built-in structured-box path classifies boundaries geometrically. A resolved-nozzle external
// mesh instead carries named patches (liquid_inlet, nozzle_wall, collector, outlet). This header
// defines the role/BC types and a per-face role map so the solver can honor the paper's per-patch
// boundary conditions (Candido & Pascoa 2023, Sec II.D) instead of box geometry.
//
// Design contract: when a CandidoBoundaryConditions is absent or inactive, the solver MUST take the
// existing geometric path byte-for-byte (regression-safe). The per-face role array is captured before
// Mesh3D::computeGeometry() re-tags faces to the box convention; face indices are stable across
// computeGeometry(), so faceRole[fi] stays valid.

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

#include "../fvm/Mesh3D.hpp"

namespace electrospray {

enum class BcRole { Inlet, NozzleWall, Collector, Outlet, Symmetry, Unknown };

// Scalar/vector field condition kind on a patch. ZeroGradient (homogeneous Neumann) is the default
// for every field unless a role overrides it.
enum class BcKind { ZeroGradient, Dirichlet };

// Per-patch boundary condition. Fields cover all five P7 increments so the struct is stable as they
// land: potential (phi), velocity, alpha (VOF), pressure, charge (rho_e).
struct PatchBc {
  BcRole role = BcRole::Unknown;
  std::string name;

  // Electric potential phi.
  BcKind potentialKind = BcKind::ZeroGradient;
  double potentialValue = 0.0;       // used when potentialKind==Dirichlet && !potentialIsRunVoltage
  bool potentialIsRunVoltage = false;  // Dirichlet to the run's dimensionless electrode voltage U

  // Velocity u.
  BcKind velocityKind = BcKind::ZeroGradient;
  fvm::Vec3 velocityValue = fvm::Vec3::Zero();  // Dirichlet wall velocity (e.g. moving collector)
  bool velocityNoSlip = false;        // u = 0 Dirichlet (nozzle wall)
  bool velocityInletProfile = false;  // fully-developed parabolic inlet (mean-forced)

  // Volume fraction alpha.
  BcKind alphaKind = BcKind::ZeroGradient;
  double alphaValue = 0.0;            // Dirichlet alpha (inlet = 1)
  bool alphaContactAngle = false;     // collector-wall contact-angle curvature treatment

  // Pressure p.
  BcKind pressureKind = BcKind::ZeroGradient;
  bool pressureTotalZero = false;     // total-pressure outlet p0 = p + 0.5*rho|u|^2 = 0 (gauge)

  // Electric charge density rho_e.
  BcKind chargeKind = BcKind::ZeroGradient;
  double chargeValue = 0.0;           // Dirichlet charge (neutral inflow = 0)
};

// Full BC set for a run: one PatchBc per named patch + a per-face index into `patches` (-1 = none).
struct CandidoBoundaryConditions {
  std::vector<PatchBc> patches;
  std::vector<int> faceRole;  // size == mesh.faces.size(); value is index into patches, or -1
  bool active = false;

  int patchIndexOfFace(int fi) const {
    if (!active || fi < 0 || fi >= static_cast<int>(faceRole.size())) return -1;
    const int pi = faceRole[fi];
    if (pi < 0 || pi >= static_cast<int>(patches.size())) return -1;
    return pi;
  }
  BcRole roleOfFace(int fi) const {
    const int pi = patchIndexOfFace(fi);
    return pi < 0 ? BcRole::Unknown : patches[pi].role;
  }
  const PatchBc* bcOfFace(int fi) const {
    const int pi = patchIndexOfFace(fi);
    return pi < 0 ? nullptr : &patches[pi];
  }
};

// Map an OpenFOAM patch name to a role by case-insensitive substring. Order matters: more specific
// names are checked first. Unknown names map to Outlet-like behavior only if explicitly chosen by the
// caller; here Unknown is returned so the caller can decide (default: treat as Outlet/zeroGradient).
inline BcRole bcRoleFromPatchName(const std::string& rawName) {
  std::string n = rawName;
  std::transform(n.begin(), n.end(), n.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  auto has = [&](const char* sub) { return n.find(sub) != std::string::npos; };
  if (has("inlet") || has("feed")) return BcRole::Inlet;
  if (has("nozzle") || has("electrode") || has("capillary") || has("needle")) return BcRole::NozzleWall;
  if (has("collector") || has("ground") || has("plate") || has("target")) return BcRole::Collector;
  if (has("outlet") || has("atm") || has("far") || has("open") || has("ambient")) return BcRole::Outlet;
  if (has("symm") || has("wedge") || has("axis")) return BcRole::Symmetry;
  if (has("wall")) return BcRole::NozzleWall;  // generic wall -> no-slip electrode wall
  return BcRole::Unknown;
}

// Paper-default per-patch BC (Candido & Pascoa 2023, Sec II.D). Values that depend on the run
// (electrode voltage, inlet velocity magnitude, moving-wall speed) are left to be filled by the
// caller; this sets the kinds/flags per role.
inline PatchBc paperDefaultPatchBc(BcRole role) {
  PatchBc b;
  b.role = role;
  switch (role) {
    case BcRole::Inlet:
      b.potentialKind = BcKind::Dirichlet; b.potentialIsRunVoltage = true;  // phi = U
      b.velocityInletProfile = true;                                        // parabolic u_in
      b.alphaKind = BcKind::Dirichlet; b.alphaValue = 1.0;                  // liquid
      b.chargeKind = BcKind::Dirichlet; b.chargeValue = 0.0;               // neutral inflow
      break;
    case BcRole::NozzleWall:
      b.potentialKind = BcKind::Dirichlet; b.potentialIsRunVoltage = true;  // electrode phi = U
      b.velocityNoSlip = true;                                              // u = 0
      // pressure/alpha/charge: zeroGradient (default)
      break;
    case BcRole::Collector:
      b.potentialKind = BcKind::Dirichlet; b.potentialValue = 0.0;          // ground
      b.velocityKind = BcKind::Dirichlet;                                   // moving wall (value set by caller)
      b.alphaContactAngle = true;                                           // contact angle 51deg
      break;
    case BcRole::Outlet:
      b.pressureKind = BcKind::Dirichlet; b.pressureTotalZero = true;       // total p0 = 0
      b.chargeKind = BcKind::Dirichlet; b.chargeValue = 0.0;               // neutral inflow / zeroGrad out
      // velocity/alpha: zeroGradient (mixed in/out)
      break;
    case BcRole::Symmetry:
    case BcRole::Unknown:
    default:
      break;  // all zeroGradient
  }
  return b;
}

}  // namespace electrospray
