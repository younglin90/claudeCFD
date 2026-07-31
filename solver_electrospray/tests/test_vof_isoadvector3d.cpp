#include "TestUtil.hpp"
#include "fvm/VofTransport3D.hpp"
#include <filesystem>
#include <fstream>

static fvm::ScalarField tiltedInterfaceAlpha(const fvm::Mesh3D& mesh) {
  fvm::ScalarField alpha(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    double phi = x.x() + 0.35 * x.y() - 0.55;
    alpha[c] = std::clamp(0.5 - phi / 0.18, 0.0, 1.0);
  }
  return alpha;
}

int main() {
  auto mesh = fvm::Mesh3D::hexGrid(14, 12, 10, 1.0, 1.0, 1.0, 0.06);
  auto alpha = tiltedInterfaceAlpha(mesh);
  auto faceFlux = fvm::divergenceFreeBoxFlux3D(mesh, 0.05);

  fvm::ScalarField isoFlux = fvm::isoAdvectorFaceFlux3D(mesh, alpha, faceFlux, 0.01);
  fvm::ScalarField algFlux = fvm::convectionFaceFluxUpwindTVD3D(mesh, alpha, faceFlux, 1.0);
  double maxDiff = 0.0;
  double maxIso = 0.0;
  for (size_t fi = 0; fi < isoFlux.size(); ++fi) {
    maxDiff = std::max(maxDiff, std::abs(isoFlux[fi] - algFlux[fi]));
    maxIso = std::max(maxIso, std::abs(isoFlux[fi]));
    check(std::isfinite(isoFlux[fi]), "3D isoAdvector face flux finite");
  }

  double initialMass = fvm::vofMass3D(mesh, alpha);
  fvm::VofTransportOptions3D opt;
  opt.scheme = fvm::VofAdvectionScheme3D::IsoAdvector;
  opt.compression = 0.0;
  opt.correctionSweeps = 5;
  fvm::VofTransportReport3D report;
  for (int step = 0; step < 40; ++step) {
    report = fvm::advectVof3D(mesh, alpha, faceFlux, 0.01, opt);
  }
  double finalMass = fvm::vofMass3D(mesh, alpha);
  auto [amin, amax] = fvm::vofBounds3D(alpha);
  double drift = std::abs(finalMass - initialMass) / std::max(std::abs(initialMass), 1e-30);

  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/vof_isoadvector3d.csv");
  csv << "case,cells,faces,steps,dt,max_iso_flux,max_iso_minus_algebraic,"
         "initial_mass,final_mass,relative_mass_drift,min_alpha,max_alpha\n";
  csv << "tilted_interface," << mesh.cells.size() << "," << mesh.faces.size()
      << ",40,0.01," << maxIso << "," << maxDiff << "," << initialMass << ","
      << finalMass << "," << drift << "," << amin << "," << amax << "\n";

  check(maxIso > 0.0, "3D isoAdvector face flux active");
  check(maxDiff > 1e-8, "3D isoAdvector geometric flux differs from algebraic TVD flux");
  check(drift <= 1e-3, "3D isoAdvector mass drift bounded");
  check(report.relativeMassDrift <= 1e-3, "3D isoAdvector per-step mass drift bounded");
  check(amin >= -1e-14 && amax <= 1.0 + 1e-14, "3D isoAdvector alpha bounded");

  auto oneCell = fvm::Mesh3D::hexGrid(1, 1, 1);
  fvm::ScalarField empty(oneCell.cells.size(), 0.0);
  fvm::ScalarField boundaryFlux(oneCell.faces.size(), 0.0);
  fvm::ScalarField boundaryAlpha(oneCell.faces.size(), 0.0);
  int inflowFace = -1;
  for (int fi = 0; fi < static_cast<int>(oneCell.faces.size()); ++fi) {
    if (!oneCell.faces[fi].internal() && oneCell.faces[fi].patch == 2) {
      inflowFace = fi;
      break;
    }
  }
  check(inflowFace >= 0, "3D isoAdvector boundary-alpha guard has an inflow face");
  boundaryFlux[inflowFace] = -0.05;
  boundaryAlpha[inflowFace] = 1.0;
  const fvm::ScalarField noBoundaryAlphaFlux =
      fvm::isoAdvectorFaceFlux3D(oneCell, empty, boundaryFlux, 0.01);
  const fvm::ScalarField prescribedBoundaryAlphaFlux =
      fvm::isoAdvectorFaceFlux3D(oneCell, empty, boundaryFlux, 0.01, &boundaryAlpha);
  check(std::abs(noBoundaryAlphaFlux[inflowFace]) <= 1e-14,
        "3D isoAdvector empty owner does not inject liquid without boundary alpha");
  check(std::abs(prescribedBoundaryAlphaFlux[inflowFace] - boundaryFlux[inflowFace]) <=
            1e-14,
        "3D isoAdvector prescribed boundary alpha injects liquid on inflow");

  std::cout << "vof_isoadvector3d_max_flux=" << maxIso
            << " vof_isoadvector3d_max_diff_vs_algebraic=" << maxDiff
            << " vof_isoadvector3d_mass_drift=" << drift
            << " vof_isoadvector3d_min_alpha=" << amin
            << " vof_isoadvector3d_max_alpha=" << amax
            << " boundary_alpha_flux=" << prescribedBoundaryAlphaFlux[inflowFace] << "\n";
}
