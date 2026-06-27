// P7 (task #15): unit test for the named-patch boundary-condition framework + the first consumer
// (potential Dirichlet). Verifies (a) the inactive/null path is byte-identical to the geometric box
// path (regression-safe), (b) a named-patch BC whose roles coincide with the geometry reproduces the
// geometric potential field, and (c) flipping a patch role/value changes the field (the tag is
// actually consumed).

#include "TestUtil.hpp"
#include "electrospray/CandidoTaylorConeJet3D.hpp"

#include <algorithm>
#include <cmath>

int main() {
  using namespace electrospray;
  CandidoTaylorConeJetSetup setup;
  CandidoConeJetSmokeOptions3D opt;
  const double ly = setup.collectorDistance / setup.outerDiameter;
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  fvm::Mesh3D mesh = fvm::Mesh3D::hexGrid(opt.nx, opt.ny, opt.nz, lx, ly, lz, opt.skew);
  const double voltage = 7.5;  // arbitrary dimensionless test electrode voltage

  // (1) Geometric reference (bc == nullptr).
  const fvm::PotentialBoundary3D geom =
      candidoPotentialBoundary3D(mesh, setup, opt, voltage, nullptr);

  // (2) Build a named-patch BC whose face roles coincide with the geometric classification:
  //     nozzle electrode (y<=yTol & r<=0.5) -> NozzleWall (phi=U); collector (y>=ly-yTol) -> phi=0.
  CandidoBoundaryConditions bc;
  bc.faceRole.assign(mesh.faces.size(), -1);
  PatchBc nozzle = paperDefaultPatchBc(BcRole::NozzleWall);
  nozzle.name = "nozzle_wall";
  PatchBc collector = paperDefaultPatchBc(BcRole::Collector);
  collector.name = "collector";
  bc.patches = {nozzle, collector};  // index 0 = nozzle electrode, 1 = collector ground
  const double yTol = ly / std::max(opt.ny, 1) * 0.55;
  const fvm::Vec3 axis{0.5 * lx, 0.0, 0.5 * lz};
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const fvm::Face3D& f = mesh.faces[fi];
    if (f.internal()) continue;
    const double r = std::hypot(f.centroid.x() - axis.x(), f.centroid.z() - axis.z());
    if (f.centroid.y() <= yTol && r <= 0.5) {
      bc.faceRole[fi] = 0;  // nozzle electrode
    } else if (f.centroid.y() >= ly - yTol) {
      bc.faceRole[fi] = 1;  // collector ground
    }
  }
  bc.active = true;
  const fvm::PotentialBoundary3D named =
      candidoPotentialBoundary3D(mesh, setup, opt, voltage, &bc);

  int dirichletCount = 0, mismatch = 0;
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    if (named.faceDirichlet[fi]) ++dirichletCount;
    if (named.faceDirichlet[fi] != geom.faceDirichlet[fi]) ++mismatch;
    if (named.faceDirichlet[fi] &&
        std::abs(named.faceValue[fi] - geom.faceValue[fi]) > 1e-12) ++mismatch;
  }
  check(dirichletCount > 0, "named-patch potential BC sets Dirichlet faces");
  check(mismatch == 0, "named-patch potential BC matches geometric where roles coincide");

  // (3) Flip the collector to the electrode voltage U -> the field must change (tag is consumed).
  CandidoBoundaryConditions bc2 = bc;
  bc2.patches[1].potentialIsRunVoltage = true;  // collector -> U instead of 0
  const fvm::PotentialBoundary3D flipped =
      candidoPotentialBoundary3D(mesh, setup, opt, voltage, &bc2);
  int changed = 0;
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    if (flipped.faceDirichlet[fi] &&
        std::abs(flipped.faceValue[fi] - geom.faceValue[fi]) > 1e-9) ++changed;
  }
  check(changed > 0, "flipping the collector potential to U changes the BC (tag consumed)");

  // (4) An inactive BC must be byte-identical to the geometric path.
  CandidoBoundaryConditions inactive;
  inactive.active = false;
  const fvm::PotentialBoundary3D viaInactive =
      candidoPotentialBoundary3D(mesh, setup, opt, voltage, &inactive);
  int inactiveMismatch = 0;
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    if (viaInactive.faceDirichlet[fi] != geom.faceDirichlet[fi]) ++inactiveMismatch;
    if (std::abs(viaInactive.faceValue[fi] - geom.faceValue[fi]) > 1e-15) ++inactiveMismatch;
  }
  check(inactiveMismatch == 0, "inactive named-patch BC is byte-identical to the geometric path");

  // (5) Role mapping from OpenFOAM patch names.
  check(bcRoleFromPatchName("liquid_inlet") == BcRole::Inlet, "name->role: liquid_inlet");
  check(bcRoleFromPatchName("nozzle_wall") == BcRole::NozzleWall, "name->role: nozzle_wall");
  check(bcRoleFromPatchName("collector") == BcRole::Collector, "name->role: collector");
  check(bcRoleFromPatchName("outlet") == BcRole::Outlet, "name->role: outlet");
  check(bcRoleFromPatchName("ATMOSPHERE") == BcRole::Outlet, "name->role: case-insensitive");

  std::cout << "test_candido_named_patch_boundary3d=pass dirichlet_faces=" << dirichletCount << "\n";
  return 0;
}
