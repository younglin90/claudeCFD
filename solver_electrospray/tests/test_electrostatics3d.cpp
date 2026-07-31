#include "TestUtil.hpp"
#include "fvm/Electrostatics3D.hpp"
#include <filesystem>
#include <fstream>

static fvm::PotentialBoundary3D parallelPlateBC(const fvm::Mesh3D& mesh) {
  fvm::PotentialBoundary3D bc;
  bc.faceDirichlet.assign(mesh.faces.size(), 0);
  bc.faceValue.assign(mesh.faces.size(), 0.0);
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const auto& f = mesh.faces[fi];
    if (f.internal()) continue;
    if (std::abs(f.centroid.x()) < 1e-12) {
      bc.faceDirichlet[fi] = 1;
      bc.faceValue[fi] = 1.0;
    } else if (std::abs(f.centroid.x() - 1.0) < 1e-12) {
      bc.faceDirichlet[fi] = 1;
      bc.faceValue[fi] = 0.0;
    }
  }
  return bc;
}

static double potentialL2(const fvm::Mesh3D& mesh, const fvm::ScalarField& phi,
                          const fvm::ScalarField& exact,
                          const fvm::ScalarField* mask = nullptr) {
  double e2 = 0.0, n2 = 0.0;
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    double w = mask == nullptr ? 1.0 : (*mask)[c];
    e2 += w * fvm::sqr(phi[c] - exact[c]) * mesh.cells[c].V;
    n2 += w * fvm::sqr(exact[c]) * mesh.cells[c].V;
  }
  return std::sqrt(e2 / std::max(n2, 1e-30));
}

static double runParallelPlate(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(16, 8, 8);
  fvm::ScalarField eps(mesh.cells.size(), 2.0), rho(mesh.cells.size(), 0.0);
  auto report = fvm::solvePotential3D(mesh, eps, rho, parallelPlateBC(mesh));
  fvm::ScalarField exact(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) exact[c] = 1.0 - mesh.cells[c].centroid.x();
  double err = potentialL2(mesh, report.phi, exact);
  csv << "parallel_plate," << mesh.cells.size() << "," << err << "," << report.residual
      << "," << report.iterations << ",nan,nan,nan,nan\n";
  csv.flush();
  check(err <= 0.01, "3D parallel-plate potential within 1% L2");
  return err;
}

static double runConcentricSphere(std::ofstream& csv) {
  constexpr double rin = 0.22;
  constexpr double rout = 0.48;
  const int nr = 54;
  const int nt = 30;
  const int np = 60;
  const double theta0 = 0.20;
  const double theta1 = M_PI - 0.20;
  std::vector<fvm::Vec3> pts;
  auto pid = [nt, np](int i, int j, int k) {
    return (i * (nt + 1) + j) * np + (k % np);
  };
  for (int i = 0; i <= nr; ++i) {
    double r = rin + (rout - rin) * static_cast<double>(i) / nr;
    for (int j = 0; j <= nt; ++j) {
      double th = theta0 + (theta1 - theta0) * static_cast<double>(j) / nt;
      for (int k = 0; k < np; ++k) {
        double ph = 2.0 * M_PI * static_cast<double>(k) / np;
        pts.push_back({0.5 + r * std::sin(th) * std::cos(ph),
                       0.5 + r * std::sin(th) * std::sin(ph),
                       0.5 + r * std::cos(th)});
      }
    }
  }
  std::vector<std::vector<std::vector<int>>> cellFaces;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nt; ++j) {
      for (int k = 0; k < np; ++k) {
        int kp = (k + 1) % np;
        int a = pid(i, j, k), b = pid(i, j, kp), c = pid(i, j + 1, kp), d = pid(i, j + 1, k);
        int e = pid(i + 1, j, k), f = pid(i + 1, j, kp), g = pid(i + 1, j + 1, kp), h = pid(i + 1, j + 1, k);
        cellFaces.push_back({{a, d, c, b}, {e, f, g, h}, {a, b, f, e},
                             {d, h, g, c}, {a, e, h, d}, {b, c, g, f}});
      }
    }
  }
  auto mesh = fvm::Mesh3D::fromCellFaces(pts, cellFaces);
  fvm::ScalarField eps(mesh.cells.size(), 1.0), rho(mesh.cells.size(), 0.0);
  fvm::PotentialBoundary3D bc;
  bc.faceDirichlet.assign(mesh.faces.size(), 0);
  bc.faceValue.assign(mesh.faces.size(), 0.0);
  fvm::ScalarField exact(mesh.cells.size(), 0.0), mask(mesh.cells.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    fvm::Vec3 rel = mesh.cells[c].centroid - fvm::Vec3{0.5, 0.5, 0.5};
    double r = rel.norm();
    double theta = std::acos(std::clamp(rel.z() / std::max(r, 1e-30), -1.0, 1.0));
    double phi = (1.0 / std::max(r, 1e-30) - 1.0 / rout) / (1.0 / rin - 1.0 / rout);
    exact[c] = std::clamp(phi, 0.0, 1.0);
    if (r > rin + 0.04 && r < rout - 0.04 &&
        theta > theta0 + 0.35 && theta < theta1 - 0.35) {
      mask[c] = 1.0;
    }
  }
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const auto& f = mesh.faces[fi];
    if (f.internal()) continue;
    double r = (f.centroid - fvm::Vec3{0.5, 0.5, 0.5}).norm();
    if (r < rin + 0.02) {
      bc.faceDirichlet[fi] = 1;
      bc.faceValue[fi] = 1.0;
    } else if (r > rout - 0.02) {
      bc.faceDirichlet[fi] = 1;
      bc.faceValue[fi] = 0.0;
    }
  }
  auto report = fvm::solvePotential3D(mesh, eps, rho, bc);
  double err = potentialL2(mesh, report.phi, exact, &mask);
  csv << "concentric_sphere," << mesh.cells.size() << "," << err << "," << report.residual
      << "," << report.iterations << ",nan,nan,nan,nan\n";
  csv.flush();
  check(err <= 0.01, "3D concentric-sphere potential within 1% L2");
  return err;
}

static fvm::ChargeTransportReport3D runChargeTransport(std::ofstream& csv) {
  auto mesh = fvm::Mesh3D::hexGrid(10, 8, 6);
  fvm::ScalarField q(mesh.cells.size(), 0.0), flux(mesh.faces.size(), 0.0);
  for (size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& x = mesh.cells[c].centroid;
    q[c] = 0.2 + 0.6 * std::exp(-80.0 * (fvm::sqr(x.x() - 0.35) +
                                         fvm::sqr(x.y() - 0.5) +
                                         fvm::sqr(x.z() - 0.5)));
  }
  for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
    const auto& f = mesh.faces[fi];
    if (!f.internal()) continue;
    fvm::Vec3 u{0.05 * std::sin(M_PI * f.centroid.y()),
                0.02 * std::sin(M_PI * f.centroid.z()), 0.0};
    flux[fi] = u.dot(f.Sf);
  }
  auto report = fvm::transportChargeBounded3D(mesh, q, flux, 0.02, 0.0, 1.0);
  csv << "charge_transport," << mesh.cells.size() << ",nan,nan,nan,"
      << report.relativeMassDrift << "," << report.minCharge << ","
      << report.maxCharge << "," << report.finalMass << "\n";
  check(report.relativeMassDrift <= 1e-12, "3D charge transport conservative");
  check(report.minCharge >= -1e-14 && report.maxCharge <= 1.0 + 1e-14,
        "3D charge transport bounded");
  return report;
}

int main() {
  std::filesystem::create_directories("benchmark_logs");
  std::ofstream csv("benchmark_logs/electrostatics3d.csv");
  csv << "case,cells,potential_l2,linear_residual,iterations,charge_mass_drift,"
         "charge_min,charge_max,charge_final_mass\n";
  double plate = runParallelPlate(csv);
  double sphere = runConcentricSphere(csv);
  auto charge = runChargeTransport(csv);
  std::cout << "electrostatics3d_parallel_plate_l2=" << plate
            << " electrostatics3d_concentric_sphere_l2=" << sphere
            << " electrostatics3d_charge_mass_drift=" << charge.relativeMassDrift
            << " electrostatics3d_charge_min=" << charge.minCharge
            << " electrostatics3d_charge_max=" << charge.maxCharge << "\n";
}
