#include "electrospray/CandidoTaylorConeJet3D.hpp"
#include "fvm/Mesh3D.hpp"
#include "fvm/MeshQuality3D.hpp"
#include "fvm/OpenFoamFieldReader3D.hpp"
#include "fvm/OpenFoamPolyMeshReader3D.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path sourceRoot() {
#ifdef FVM_SOURCE_DIR
  return std::filesystem::path(FVM_SOURCE_DIR);
#else
  return std::filesystem::current_path();
#endif
}

struct RunnerOptions {
  std::filesystem::path casePath;
  std::filesystem::path caseDir;
  std::filesystem::path outputDir;
};

std::string readText(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to read case file: " + path.string());
  return std::string(std::istreambuf_iterator<char>(in), {});
}

std::string jsonEscape(const std::string& s) {
  std::ostringstream os;
  for (char ch : s) {
    switch (ch) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default: os << ch; break;
    }
  }
  return os.str();
}

std::optional<std::string> jsonString(const std::string& text, const std::string& key) {
  const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch m;
  if (std::regex_search(text, m, re)) return m[1].str();
  return std::nullopt;
}

std::optional<double> jsonNumber(const std::string& text, const std::string& key) {
  const std::regex re("\"" + key +
                      "\"\\s*:\\s*(-?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
  std::smatch m;
  if (!std::regex_search(text, m, re)) return std::nullopt;
  return std::stod(m[1].str());
}

std::optional<bool> jsonBool(const std::string& text, const std::string& key) {
  const std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch m;
  if (!std::regex_search(text, m, re)) return std::nullopt;
  return m[1].str() == "true";
}

int jsonIntOr(const std::string& text, const std::string& key, int fallback) {
  const auto v = jsonNumber(text, key);
  return v ? static_cast<int>(*v) : fallback;
}

double jsonDoubleOr(const std::string& text, const std::string& key, double fallback) {
  const auto v = jsonNumber(text, key);
  return v ? *v : fallback;
}

bool jsonBoolOr(const std::string& text, const std::string& key, bool fallback) {
  const auto v = jsonBool(text, key);
  return v ? *v : fallback;
}

void printDefaultsJson(std::ostream& os) {
  // Single source of truth for case defaults: emit the solver struct defaults so the
  // GUI can mirror them (eliminates GUI/C++ default drift). Keys match the json keys
  // parsed by smokeOptionsFromJson and the setup parsing in main().
  const electrospray::CandidoConeJetSmokeOptions3D o;
  const electrospray::CandidoTaylorConeJetSetup s;
  os << std::setprecision(12) << std::boolalpha;
  os << "{\n";
  os << "  \"nx\": " << o.nx << ",\n";
  os << "  \"ny\": " << o.ny << ",\n";
  os << "  \"nz\": " << o.nz << ",\n";
  os << "  \"steps\": " << o.steps << ",\n";
  os << "  \"skew\": " << o.skew << ",\n";
  os << "  \"cfl\": " << o.cfl << ",\n";
  os << "  \"radial_window_outer_diameters\": " << o.radialWindowOuterDiameters << ",\n";
  os << "  \"pseudo_viscosity\": " << o.pseudoViscosity << ",\n";
  os << "  \"vof_compression\": " << o.vofCompression << ",\n";
  os << "  \"vof_post_sharpening\": " << o.vofPostSharpening << ",\n";
  os << "  \"vof_post_sharpening_sweeps\": " << o.vofPostSharpeningSweeps << ",\n";
  os << "  \"use_vof_inlet_boundary_alpha\": " << o.useVofInletBoundaryAlpha << ",\n";
  os << "  \"alpha_interface_width_outer_diameters\": " << o.alphaInterfaceWidthOuterDiameters << ",\n";
  os << "  \"normalized_liquid_conductivity\": " << o.normalizedLiquidConductivity << ",\n";
  os << "  \"normalized_gas_conductivity\": " << o.normalizedGasConductivity << ",\n";
  os << "  \"use_dimensional_electrical_scaling\": " << o.useDimensionalElectricalScaling << ",\n";
  os << "  \"charge_limit_base\": " << o.chargeLimitBase << ",\n";
  os << "  \"charge_subcycles\": " << o.chargeSubcycles << ",\n";
  os << "  \"conservative_charge_bounding\": " << o.conservativeChargeBounding << ",\n";
  os << "  \"quasi_implicit_charge_relaxation\": " << o.quasiImplicitChargeRelaxation << ",\n";
  os << "  \"quasi_implicit_bulk_conduction\": " << o.quasiImplicitBulkConduction << ",\n";
  os << "  \"use_rayleigh_charge_limit\": " << o.useRayleighChargeLimit << ",\n";
  os << "  \"use_interface_localized_charge_redistribution\": "
     << o.useInterfaceLocalizedChargeRedistribution << ",\n";
  os << "  \"interface_charge_redistribution_liquid_floor\": "
     << o.interfaceChargeRedistributionLiquidFloor << ",\n";
  os << "  \"use_interfacial_ohmic_charge_source\": " << o.useInterfacialOhmicChargeSource << ",\n";
  os << "  \"interfacial_ohmic_charge_source_scale\": " << o.interfacialOhmicChargeSourceScale << ",\n";
  os << "  \"use_conductivity_potential_charge_closure\": "
     << o.useConductivityPotentialChargeClosure << ",\n";
  os << "  \"suppress_nozzle_conductive_charge_flux\": " << o.suppressNozzleConductiveChargeFlux << ",\n";
  os << "  \"collector_only_conductive_charge_flux\": " << o.collectorOnlyConductiveChargeFlux << ",\n";
  os << "  \"apply_conductive_boundary_filters_in_implicit_ohmic\": "
     << o.applyConductiveBoundaryFiltersInImplicitOhmic << ",\n";
  os << "  \"use_poisson_face_conductive_current\": " << o.usePoissonFaceConductiveCurrent << ",\n";
  os << "  \"implicit_ohmic_charge_projection\": " << o.implicitOhmicChargeProjection << ",\n";
  os << "  \"refresh_potential_after_charge_advance\": " << o.refreshPotentialAfterChargeAdvance << ",\n";
  os << "  \"use_electric_relaxation_timestep_limit\": " << o.useElectricRelaxationTimeStepLimit << ",\n";
  os << "  \"electric_relaxation_timestep_safety\": " << o.electricRelaxationTimeStepSafety << ",\n";
  os << "  \"use_poisson_face_maxwell_force\": " << o.usePoissonFaceMaxwellForce << ",\n";
  os << "  \"use_poisson_hybrid_maxwell_force\": " << o.usePoissonHybridMaxwellForce << ",\n";
  os << "  \"use_poisson_bounded_vector_maxwell_force\": " << o.usePoissonBoundedVectorMaxwellForce << ",\n";
  os << "  \"use_tomar_conducting_surface_force\": " << o.useTomarConductingSurfaceForce << ",\n";
  os << "  \"use_open_atmospheric_boundary_flux\": " << o.useOpenAtmosphericBoundaryFlux << ",\n";
  os << "  \"use_boundary_charge_advection\": " << o.useBoundaryChargeAdvection << ",\n";
  os << "  \"use_fully_developed_inlet_velocity_boundary\": "
     << o.useFullyDevelopedInletVelocityBoundary << ",\n";
  os << "  \"use_moving_collector_wall\": " << o.useMovingCollectorWall << ",\n";
  os << "  \"use_preconditioned_paper_current_jet\": " << o.usePreconditionedPaperCurrentJet << ",\n";
  os << "  \"preconditioned_jet_tip_y_over_inner_diameter\": "
     << o.preconditionedJetTipYOverInnerDiameter << ",\n";
  os << "  \"preconditioned_jet_radius_inner_diameters\": "
     << o.preconditionedJetRadiusInnerDiameters << ",\n";
  os << "  \"preconditioned_jet_interface_width_inner_diameters\": "
     << o.preconditionedJetInterfaceWidthInnerDiameters << ",\n";
  os << "  \"preconditioned_jet_velocity_scale\": " << o.preconditionedJetVelocityScale << ",\n";
  os << "  \"use_contact_angle_curvature\": " << o.useContactAngleCurvature << ",\n";
  os << "  \"contact_angle_curvature_wall_band_cells\": " << o.contactAngleCurvatureWallBandCells << ",\n";
  os << "  \"electric_drive_reference_scale\": " << o.electricDriveReferenceScale << ",\n";
  os << "  \"electric_drive_ca_exponent\": " << o.electricDriveCaExponent << ",\n";
  os << "  \"poisson_tangential_limit_factor\": " << o.poissonTangentialLimitFactor << ",\n";
  os << "  \"poisson_tangential_limit_floor_fraction\": " << o.poissonTangentialLimitFloorFraction << ",\n";
  os << "  \"surface_tension_drive_scale\": " << o.surfaceTensionDriveScale << ",\n";
  os << "  \"use_electric_force_timestep_limit\": " << o.useElectricForceTimeStepLimit << ",\n";
  os << "  \"electric_force_timestep_safety\": " << o.electricForceTimeStepSafety << ",\n";
  os << "  \"inner_diameter\": " << s.innerDiameter << ",\n";
  os << "  \"outer_diameter\": " << s.outerDiameter << ",\n";
  os << "  \"nozzle_length\": " << s.nozzleLength << ",\n";
  os << "  \"collector_distance\": " << s.collectorDistance << ",\n";
  os << "  \"collector_diameter\": " << s.collectorDiameter << ",\n";
  os << "  \"collector_speed\": " << s.collectorSpeed << ",\n";
  os << "  \"contact_angle_deg\": " << s.contactAngleDeg << ",\n";
  os << "  \"validation_voltage\": " << s.validationVoltage << ",\n";
  os << "  \"validation_flow_rate\": " << s.validationFlowRate << ",\n";
  os << "  \"liquid_density\": " << s.liquidDensity << ",\n";
  os << "  \"gas_density\": " << s.gasDensity << ",\n";
  os << "  \"liquid_viscosity\": " << s.liquidViscosity << ",\n";
  os << "  \"gas_viscosity\": " << s.gasViscosity << ",\n";
  os << "  \"surface_tension\": " << s.surfaceTension << ",\n";
  os << "  \"liquid_relative_permittivity\": " << s.liquidRelativePermittivity << ",\n";
  os << "  \"gas_relative_permittivity\": " << s.gasRelativePermittivity << ",\n";
  os << "  \"liquid_conductivity\": " << s.liquidConductivity << ",\n";
  os << "  \"gas_conductivity\": " << s.gasConductivity << "\n";
  os << "}\n";
  os << std::noboolalpha;
}

RunnerOptions parseArgs(int argc, char** argv) {
  RunnerOptions opt;
  opt.outputDir = sourceRoot() / "runs" / "gui_case";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(name + " requires a value");
      return argv[++i];
    };
    if (arg == "--case") {
      opt.casePath = need(arg);
    } else if (arg == "--case-dir") {
      opt.caseDir = need(arg);
    } else if (arg == "--output-dir") {
      opt.outputDir = need(arg);
    } else if (arg == "--print-defaults") {
      printDefaultsJson(std::cout);
      std::exit(0);
    } else if (arg == "--help") {
      std::cout << "usage: electrospray_case_runner --case case.json [--output-dir runs/name]\n"
                << "       electrospray_case_runner --case-dir openfoam_case [--output-dir runs/name]\n"
                << "       electrospray_case_runner --print-defaults\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opt.casePath.empty() && opt.caseDir.empty()) {
    throw std::runtime_error("--case or --case-dir is required");
  }
  return opt;
}

fvm::Mesh3D buildCaseMesh(const std::string& text, const std::string& meshMode,
                          fvm::OpenFoamPolyMeshReadReport3D* foamReport) {
  if (meshMode == "openfoam_polyMesh") {
    const auto path = jsonString(text, "openfoam_polyMesh");
    if (!path || path->empty()) throw std::runtime_error("openfoam_polyMesh path is required");
    return fvm::readOpenFoamPolyMesh3D(*path, foamReport);
  }
  const int nx = jsonIntOr(text, "nx", 8);
  const int ny = jsonIntOr(text, "ny", 16);
  const int nz = jsonIntOr(text, "nz", 8);
  const double lx = jsonDoubleOr(text, "lx", 1.0);
  const double ly = jsonDoubleOr(text, "ly", 4.0);
  const double lz = jsonDoubleOr(text, "lz", 1.0);
  const double skew = jsonDoubleOr(text, "skew", 0.0);
  return fvm::Mesh3D::hexGrid(nx, ny, nz, lx, ly, lz, skew);
}

void writeMeshSummaryJson(const std::filesystem::path& out, const fvm::Mesh3D& mesh,
                          const fvm::MeshQualityReport3D& q,
                          const std::string& meshMode,
                          const fvm::OpenFoamPolyMeshReadReport3D& foamReport) {
  std::ofstream os(out);
  if (!os) throw std::runtime_error("failed to write summary JSON: " + out.string());
  os << std::setprecision(12);
  os << "{\n";
  os << "  \"status\": \"pass\",\n";
  os << "  \"run_mode\": \"validate_mesh\",\n";
  os << "  \"mesh_mode\": \"" << jsonEscape(meshMode) << "\",\n";
  os << "  \"points\": " << mesh.points.size() << ",\n";
  os << "  \"cells\": " << q.cells << ",\n";
  os << "  \"faces\": " << q.faces << ",\n";
  os << "  \"internal_faces\": " << q.internalFaces << ",\n";
  os << "  \"patch_count\": " << mesh.patches.size() << ",\n";
  os << "  \"openfoam_neighbour_faces\": " << foamReport.neighbours << ",\n";
  os << "  \"max_non_orthogonality_deg\": " << q.maxNonOrthogonalityDeg << ",\n";
  os << "  \"max_aspect_ratio\": " << q.maxAspectRatio << ",\n";
  os << "  \"patches\": [\n";
  for (size_t p = 0; p < mesh.patches.size(); ++p) {
    os << "    {\"name\": \"" << jsonEscape(mesh.patches[p].name)
       << "\", \"faces\": " << mesh.patches[p].faces.size() << "}";
    if (p + 1 != mesh.patches.size()) os << ",";
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";
}

void writeStringArrayJson(std::ostream& os, const std::vector<std::string>& items,
                          int indent) {
  os << "[";
  if (!items.empty()) os << "\n";
  for (size_t i = 0; i < items.size(); ++i) {
    os << std::string(static_cast<size_t>(indent), ' ')
       << "\"" << jsonEscape(items[i]) << "\"";
    if (i + 1 != items.size()) os << ",";
    os << "\n";
  }
  if (!items.empty()) os << std::string(static_cast<size_t>(indent - 2), ' ');
  os << "]";
}

void writeScalarFieldJson(std::ostream& os, const fvm::OpenFoamScalarField3D& field,
                          bool present, int indent) {
  const std::string pad(static_cast<size_t>(indent), ' ');
  os << "{\n";
  os << pad << "  \"present\": " << (present ? "true" : "false") << ",\n";
  if (present) {
    os << pad << "  \"internal_form\": \"" << jsonEscape(field.internalForm) << "\",\n";
    os << pad << "  \"uniform_internal_value\": " << field.uniformInternalValue << ",\n";
    os << pad << "  \"boundary\": [\n";
    size_t n = 0;
    for (const auto& kv : field.boundary) {
      const auto& bc = kv.second;
      os << pad << "    {\"patch\": \"" << jsonEscape(bc.patch)
         << "\", \"type\": \"" << jsonEscape(bc.type)
         << "\", \"has_value\": " << (bc.hasUniformValue ? "true" : "false")
         << ", \"value\": " << bc.uniformValue << "}";
      if (++n != field.boundary.size()) os << ",";
      os << "\n";
    }
    os << pad << "  ]\n";
  } else {
    os << pad << "  \"boundary\": []\n";
  }
  os << pad << "}";
}

void writeVectorFieldJson(std::ostream& os, const fvm::OpenFoamVectorField3D& field,
                          bool present, int indent) {
  const std::string pad(static_cast<size_t>(indent), ' ');
  os << "{\n";
  os << pad << "  \"present\": " << (present ? "true" : "false") << ",\n";
  if (present) {
    os << pad << "  \"internal_form\": \"" << jsonEscape(field.internalForm) << "\",\n";
    os << pad << "  \"uniform_internal_value\": ["
       << field.uniformInternalValue.x() << ", "
       << field.uniformInternalValue.y() << ", "
       << field.uniformInternalValue.z() << "],\n";
    os << pad << "  \"boundary\": [\n";
    size_t n = 0;
    for (const auto& kv : field.boundary) {
      const auto& bc = kv.second;
      os << pad << "    {\"patch\": \"" << jsonEscape(bc.patch)
         << "\", \"type\": \"" << jsonEscape(bc.type)
         << "\", \"has_value\": " << (bc.hasUniformValue ? "true" : "false")
         << ", \"value\": [" << bc.uniformValue.x() << ", "
         << bc.uniformValue.y() << ", " << bc.uniformValue.z() << "]}";
      if (++n != field.boundary.size()) os << ",";
      os << "\n";
    }
    os << pad << "  ]\n";
  } else {
    os << pad << "  \"boundary\": []\n";
  }
  os << pad << "}";
}

void writeOpenFoamCaseSummaryJson(const std::filesystem::path& out,
                                  const fvm::Mesh3D& mesh,
                                  const fvm::MeshQualityReport3D& q,
                                  const fvm::OpenFoamPolyMeshReadReport3D& meshReport,
                                  const fvm::OpenFoamCaseFields3D& fields,
                                  const fvm::OpenFoamCaseValidationReport3D& fieldReport) {
  std::ofstream os(out);
  if (!os) throw std::runtime_error("failed to write OpenFOAM case summary: " + out.string());
  os << std::setprecision(12);
  const bool ok = q.finite && q.nonPositiveVolumeCount == 0 && q.zeroAreaFaceCount == 0 &&
                  fieldReport.fieldsRead >= 5 &&
                  fieldReport.missingPatchBoundaryEntries == 0 &&
                  fieldReport.unknownPatchBoundaryEntries == 0;
  os << "{\n";
  os << "  \"status\": \"" << (ok ? "pass" : "fail") << "\",\n";
  os << "  \"run_mode\": \"openfoam_case_validate\",\n";
  os << "  \"mesh_mode\": \"openfoam_polyMesh\",\n";
  os << "  \"points\": " << mesh.points.size() << ",\n";
  os << "  \"cells\": " << q.cells << ",\n";
  os << "  \"faces\": " << q.faces << ",\n";
  os << "  \"patch_count\": " << mesh.patches.size() << ",\n";
  os << "  \"openfoam_neighbour_faces\": " << meshReport.neighbours << ",\n";
  os << "  \"max_non_orthogonality_deg\": " << q.maxNonOrthogonalityDeg << ",\n";
  os << "  \"max_aspect_ratio\": " << q.maxAspectRatio << ",\n";
  os << "  \"fields_read\": " << fieldReport.fieldsRead << ",\n";
  os << "  \"missing_patch_boundary_entries\": "
     << fieldReport.missingPatchBoundaryEntries << ",\n";
  os << "  \"unknown_patch_boundary_entries\": "
     << fieldReport.unknownPatchBoundaryEntries << ",\n";
  os << "  \"patches\": [\n";
  for (size_t p = 0; p < mesh.patches.size(); ++p) {
    os << "    {\"name\": \"" << jsonEscape(mesh.patches[p].name)
       << "\", \"faces\": " << mesh.patches[p].faces.size() << "}";
    if (p + 1 != mesh.patches.size()) os << ",";
    os << "\n";
  }
  os << "  ],\n";
  os << "  \"missing_files\": ";
  writeStringArrayJson(os, fieldReport.missingFiles, 4);
  os << ",\n  \"missing_patch_entries\": ";
  writeStringArrayJson(os, fieldReport.missingPatchEntries, 4);
  os << ",\n  \"unknown_patch_entries\": ";
  writeStringArrayJson(os, fieldReport.unknownPatchEntries, 4);
  os << ",\n  \"fields\": {\n";
  os << "    \"U\": ";
  writeVectorFieldJson(os, fields.U, fields.hasU, 4);
  os << ",\n    \"p\": ";
  writeScalarFieldJson(os, fields.p, fields.hasP, 4);
  os << ",\n    \"alpha\": ";
  writeScalarFieldJson(os, fields.alpha, fields.hasAlpha, 4);
  os << ",\n    \"phi\": ";
  writeScalarFieldJson(os, fields.phi, fields.hasPhi, 4);
  os << ",\n    \"rhoE\": ";
  writeScalarFieldJson(os, fields.rhoE, fields.hasRhoE, 4);
  os << "\n  }\n";
  os << "}\n";
}

void writeOpenFoamBoundaryCsv(const std::filesystem::path& path,
                              const fvm::OpenFoamCaseFields3D& fields) {
  std::ofstream csv(path);
  if (!csv) throw std::runtime_error("failed to write boundary CSV: " + path.string());
  csv << "field,patch,type,has_value,value\n";
  if (fields.hasU) {
    for (const auto& kv : fields.U.boundary) {
      const auto& bc = kv.second;
      csv << "U," << bc.patch << "," << bc.type << "," << (bc.hasUniformValue ? 1 : 0)
          << ",(" << bc.uniformValue.x() << " " << bc.uniformValue.y() << " "
          << bc.uniformValue.z() << ")\n";
    }
  }
  auto writeScalar = [&](const std::string& name, const fvm::OpenFoamScalarField3D& field,
                         bool present) {
    if (!present) return;
    for (const auto& kv : field.boundary) {
      const auto& bc = kv.second;
      csv << name << "," << bc.patch << "," << bc.type << ","
          << (bc.hasUniformValue ? 1 : 0) << "," << bc.uniformValue << "\n";
    }
  };
  writeScalar("p", fields.p, fields.hasP);
  writeScalar("alpha", fields.alpha, fields.hasAlpha);
  writeScalar("phi", fields.phi, fields.hasPhi);
  writeScalar("rhoE", fields.rhoE, fields.hasRhoE);
}

electrospray::CandidoConeJetSmokeOptions3D smokeOptionsFromJson(const std::string& text) {
  electrospray::CandidoConeJetSmokeOptions3D opt;
  opt.nx = jsonIntOr(text, "nx", opt.nx);
  opt.ny = jsonIntOr(text, "ny", opt.ny);
  opt.nz = jsonIntOr(text, "nz", opt.nz);
  opt.steps = jsonIntOr(text, "steps", opt.steps);
  opt.skew = jsonDoubleOr(text, "skew", opt.skew);
  opt.cfl = jsonDoubleOr(text, "cfl", opt.cfl);
  opt.radialWindowOuterDiameters =
      jsonDoubleOr(text, "radial_window_outer_diameters", opt.radialWindowOuterDiameters);
  opt.pseudoViscosity = jsonDoubleOr(text, "pseudo_viscosity", opt.pseudoViscosity);
  opt.vofCompression = jsonDoubleOr(text, "vof_compression", opt.vofCompression);
  opt.vofPostSharpening = jsonDoubleOr(text, "vof_post_sharpening", opt.vofPostSharpening);
  opt.vofPostSharpeningSweeps =
      jsonIntOr(text, "vof_post_sharpening_sweeps", opt.vofPostSharpeningSweeps);
  opt.useVofInletBoundaryAlpha =
      jsonBoolOr(text, "use_vof_inlet_boundary_alpha", opt.useVofInletBoundaryAlpha);
  opt.alphaInterfaceWidthOuterDiameters =
      jsonDoubleOr(text, "alpha_interface_width_outer_diameters",
                   opt.alphaInterfaceWidthOuterDiameters);
  opt.normalizedLiquidConductivity =
      jsonDoubleOr(text, "normalized_liquid_conductivity", opt.normalizedLiquidConductivity);
  opt.normalizedGasConductivity =
      jsonDoubleOr(text, "normalized_gas_conductivity", opt.normalizedGasConductivity);
  opt.useDimensionalElectricalScaling =
      jsonBoolOr(text, "use_dimensional_electrical_scaling", opt.useDimensionalElectricalScaling);
  opt.chargeLimitBase = jsonDoubleOr(text, "charge_limit_base", opt.chargeLimitBase);
  opt.chargeSubcycles = jsonIntOr(text, "charge_subcycles", opt.chargeSubcycles);
  opt.conservativeChargeBounding =
      jsonBoolOr(text, "conservative_charge_bounding", opt.conservativeChargeBounding);
  opt.quasiImplicitChargeRelaxation =
      jsonBoolOr(text, "quasi_implicit_charge_relaxation", opt.quasiImplicitChargeRelaxation);
  opt.quasiImplicitBulkConduction =
      jsonBoolOr(text, "quasi_implicit_bulk_conduction", opt.quasiImplicitBulkConduction);
  opt.useRayleighChargeLimit =
      jsonBoolOr(text, "use_rayleigh_charge_limit", opt.useRayleighChargeLimit);
  opt.useInterfaceLocalizedChargeRedistribution =
      jsonBoolOr(text, "use_interface_localized_charge_redistribution",
                 opt.useInterfaceLocalizedChargeRedistribution);
  opt.interfaceChargeRedistributionLiquidFloor =
      jsonDoubleOr(text, "interface_charge_redistribution_liquid_floor",
                   opt.interfaceChargeRedistributionLiquidFloor);
  opt.useInterfacialOhmicChargeSource =
      jsonBoolOr(text, "use_interfacial_ohmic_charge_source",
                 opt.useInterfacialOhmicChargeSource);
  opt.interfacialOhmicChargeSourceScale =
      jsonDoubleOr(text, "interfacial_ohmic_charge_source_scale",
                   opt.interfacialOhmicChargeSourceScale);
  opt.useConductivityPotentialChargeClosure =
      jsonBoolOr(text, "use_conductivity_potential_charge_closure",
                 opt.useConductivityPotentialChargeClosure);
  opt.suppressNozzleConductiveChargeFlux =
      jsonBoolOr(text, "suppress_nozzle_conductive_charge_flux",
                 opt.suppressNozzleConductiveChargeFlux);
  opt.collectorOnlyConductiveChargeFlux =
      jsonBoolOr(text, "collector_only_conductive_charge_flux",
                 opt.collectorOnlyConductiveChargeFlux);
  opt.applyConductiveBoundaryFiltersInImplicitOhmic =
      jsonBoolOr(text, "apply_conductive_boundary_filters_in_implicit_ohmic",
                 opt.applyConductiveBoundaryFiltersInImplicitOhmic);
  opt.usePoissonFaceConductiveCurrent =
      jsonBoolOr(text, "use_poisson_face_conductive_current",
                 opt.usePoissonFaceConductiveCurrent);
  opt.implicitOhmicChargeProjection =
      jsonBoolOr(text, "implicit_ohmic_charge_projection", opt.implicitOhmicChargeProjection);
  opt.refreshPotentialAfterChargeAdvance =
      jsonBoolOr(text, "refresh_potential_after_charge_advance",
                 opt.refreshPotentialAfterChargeAdvance);
  opt.useElectricRelaxationTimeStepLimit =
      jsonBoolOr(text, "use_electric_relaxation_timestep_limit", opt.useElectricRelaxationTimeStepLimit);
  opt.electricRelaxationTimeStepSafety =
      jsonDoubleOr(text, "electric_relaxation_timestep_safety",
                   opt.electricRelaxationTimeStepSafety);
  opt.usePoissonFaceMaxwellForce =
      jsonBoolOr(text, "use_poisson_face_maxwell_force", opt.usePoissonFaceMaxwellForce);
  opt.usePoissonHybridMaxwellForce =
      jsonBoolOr(text, "use_poisson_hybrid_maxwell_force", opt.usePoissonHybridMaxwellForce);
  opt.usePoissonBoundedVectorMaxwellForce =
      jsonBoolOr(text, "use_poisson_bounded_vector_maxwell_force",
                 opt.usePoissonBoundedVectorMaxwellForce);
  opt.useTomarConductingSurfaceForce =
      jsonBoolOr(text, "use_tomar_conducting_surface_force", opt.useTomarConductingSurfaceForce);
  opt.useOpenAtmosphericBoundaryFlux =
      jsonBoolOr(text, "use_open_atmospheric_boundary_flux", opt.useOpenAtmosphericBoundaryFlux);
  opt.useBoundaryChargeAdvection =
      jsonBoolOr(text, "use_boundary_charge_advection", opt.useBoundaryChargeAdvection);
  opt.useFullyDevelopedInletVelocityBoundary =
      jsonBoolOr(text, "use_fully_developed_inlet_velocity_boundary",
                 opt.useFullyDevelopedInletVelocityBoundary);
  opt.useMovingCollectorWall =
      jsonBoolOr(text, "use_moving_collector_wall", opt.useMovingCollectorWall);
  opt.usePreconditionedPaperCurrentJet =
      jsonBoolOr(text, "use_preconditioned_paper_current_jet",
                 opt.usePreconditionedPaperCurrentJet);
  opt.preconditionedJetTipYOverInnerDiameter =
      jsonDoubleOr(text, "preconditioned_jet_tip_y_over_inner_diameter",
                   opt.preconditionedJetTipYOverInnerDiameter);
  opt.preconditionedJetRadiusInnerDiameters =
      jsonDoubleOr(text, "preconditioned_jet_radius_inner_diameters",
                   opt.preconditionedJetRadiusInnerDiameters);
  opt.preconditionedJetInterfaceWidthInnerDiameters =
      jsonDoubleOr(text, "preconditioned_jet_interface_width_inner_diameters",
                   opt.preconditionedJetInterfaceWidthInnerDiameters);
  opt.preconditionedJetVelocityScale =
      jsonDoubleOr(text, "preconditioned_jet_velocity_scale",
                   opt.preconditionedJetVelocityScale);
  opt.useContactAngleCurvature =
      jsonBoolOr(text, "use_contact_angle_curvature", opt.useContactAngleCurvature);
  opt.contactAngleCurvatureWallBandCells =
      jsonDoubleOr(text, "contact_angle_curvature_wall_band_cells",
                   opt.contactAngleCurvatureWallBandCells);
  opt.electricDriveReferenceScale =
      jsonDoubleOr(text, "electric_drive_reference_scale", opt.electricDriveReferenceScale);
  opt.electricDriveCaExponent =
      jsonDoubleOr(text, "electric_drive_ca_exponent", opt.electricDriveCaExponent);
  opt.poissonTangentialLimitFactor =
      jsonDoubleOr(text, "poisson_tangential_limit_factor", opt.poissonTangentialLimitFactor);
  opt.poissonTangentialLimitFloorFraction =
      jsonDoubleOr(text, "poisson_tangential_limit_floor_fraction",
                   opt.poissonTangentialLimitFloorFraction);
  opt.surfaceTensionDriveScale =
      jsonDoubleOr(text, "surface_tension_drive_scale", opt.surfaceTensionDriveScale);
  opt.useElectricForceTimeStepLimit =
      jsonBoolOr(text, "use_electric_force_timestep_limit", opt.useElectricForceTimeStepLimit);
  opt.electricForceTimeStepSafety =
      jsonDoubleOr(text, "electric_force_timestep_safety", opt.electricForceTimeStepSafety);
  return opt;
}

struct ExternalMeshNormalization {
  fvm::Mesh3D mesh;
  bool inletFromPatch = false;
  double inletPlaneY = 0.0;
};

ExternalMeshNormalization normalizeExternalCandidoMesh(
    const fvm::Mesh3D& raw,
    const electrospray::CandidoTaylorConeJetSetup& setup,
    const electrospray::CandidoConeJetSmokeOptions3D& opt) {
  ExternalMeshNormalization out;
  out.mesh = raw;
  fvm::Mesh3D& mesh = out.mesh;
  const double d0 = std::max(setup.outerDiameter, 1e-300);
  const double lx = opt.radialWindowOuterDiameters;
  const double lz = opt.radialWindowOuterDiameters;
  if (mesh.points.empty()) return out;

  // Jet axis = centre of the x-z bounding box; map it to (0.5 lx, *, 0.5 lz) so the
  // solver's geometric classifiers (which assume an axis at 0.5 lx) line up
  // regardless of the external mesh's radial extent.
  double xMin = mesh.points.front().x(), xMax = xMin;
  double zMin = mesh.points.front().z(), zMax = zMin;
  double yMin = mesh.points.front().y();
  for (const auto& p : mesh.points) {
    xMin = std::min(xMin, p.x());
    xMax = std::max(xMax, p.x());
    zMin = std::min(zMin, p.z());
    zMax = std::max(zMax, p.z());
    yMin = std::min(yMin, p.y());
  }
  const double xc = 0.5 * (xMin + xMax);
  const double zc = 0.5 * (zMin + zMax);

  // Inlet plane = mean y of the named "liquid_inlet" patch (robust when the bore
  // inlet is not at the minimum y); fall back to the minimum y for box/cut-cylinder
  // meshes whose inlet sits at y_min.
  double inletY = yMin;
  double sum = 0.0;
  int cnt = 0;
  for (const auto& patch : raw.patches) {
    if (patch.name == "liquid_inlet") {
      for (int fi : patch.faces) {
        sum += raw.faces[static_cast<size_t>(fi)].centroid.y();
        ++cnt;
      }
    }
  }
  if (cnt > 0) {
    inletY = sum / cnt;
    out.inletFromPatch = true;
  }
  out.inletPlaneY = inletY;

  // Scale by 1/outerDiameter, shift the inlet plane to y = 0, centre the axis.
  for (auto& p : mesh.points) {
    p.x() = (p.x() - xc) / d0 + 0.5 * lx;
    p.y() = (p.y() - inletY) / d0;
    p.z() = (p.z() - zc) / d0 + 0.5 * lz;
  }
  // Reset to the six box patches so computeGeometry() re-tags boundary faces with the
  // structured-box convention the solver's geometric BC classifiers expect.
  mesh.patches = {{"xmin", {}}, {"xmax", {}}, {"ymin", {}},
                  {"ymax", {}}, {"zmin", {}}, {"zmax", {}}};
  mesh.computeGeometry();
  return out;
}

void writeSmokeHistoryCsv(const std::filesystem::path& path,
                          const electrospray::CandidoConeJetSmokeReport3D& report) {
  std::ofstream csv(path);
  if (!csv) throw std::runtime_error("failed to write history CSV: " + path.string());
  csv << std::setprecision(12);
  csv << "step,time,mass,min_alpha,max_alpha,tip_y,centroid_y,radial_asymmetry,"
         "max_div,potential_residual,electric_force,csf_force,curvature,"
         "conductive_current,convective_current,total_current,max_velocity,wave_y_over_di,"
         "wave_asymmetry,morphology_volume_di3\n";
  for (const auto& h : report.history) {
    csv << h.step << "," << h.time << "," << h.mass << "," << h.minAlpha << ","
        << h.maxAlpha << "," << h.tipY << "," << h.centroidY << ","
        << h.radialAsymmetry << "," << h.maxDiv << "," << h.potentialResidual << ","
        << h.electricForce << "," << h.csfForce << "," << h.curvature << ","
        << h.conductiveCurrent << "," << h.convectiveCurrent << "," << h.totalCurrent << ","
        << h.maxVelocity << "," << h.waveYOverDi << "," << h.waveAsymmetry << ","
        << h.morphologyVolumeDi3 << "\n";
  }
}

void writeSmokeSummaryJson(const std::filesystem::path& path,
                           const electrospray::CandidoConeJetSmokeReport3D& r,
                           const std::string& caseName,
                           const std::string& meshMode,
                           const electrospray::CandidoConeJetSmokeOptions3D& opt) {
  std::ofstream os(path);
  if (!os) throw std::runtime_error("failed to write summary JSON: " + path.string());
  os << std::setprecision(12);
  os << "{\n";
  os << "  \"status\": \"pass\",\n";
  os << "  \"run_mode\": \"candido_smoke\",\n";
  os << "  \"case_name\": \"" << jsonEscape(caseName) << "\",\n";
  os << "  \"mesh_mode\": \"" << jsonEscape(meshMode) << "\",\n";
  os << "  \"options\": {"
     << "\"nx\": " << opt.nx << ", \"ny\": " << opt.ny << ", \"nz\": " << opt.nz
     << ", \"cfl\": " << opt.cfl
     << ", \"radial_window_outer_diameters\": " << opt.radialWindowOuterDiameters
     << ", \"vof_compression\": " << opt.vofCompression
     << ", \"use_poisson_face_maxwell_force\": "
     << (opt.usePoissonFaceMaxwellForce ? "true" : "false")
     << ", \"use_open_atmospheric_boundary_flux\": "
     << (opt.useOpenAtmosphericBoundaryFlux ? "true" : "false")
     << ", \"use_electric_relaxation_timestep_limit\": "
     << (opt.useElectricRelaxationTimeStepLimit ? "true" : "false")
     << "},\n";
  os << "  \"target_ca_e\": " << r.targetCaE << ",\n";
  os << "  \"computed_ca_e\": " << r.computedCaE << ",\n";
  os << "  \"voltage\": " << r.voltage << ",\n";
  os << "  \"cells\": " << r.cells << ",\n";
  os << "  \"faces\": " << r.faces << ",\n";
  os << "  \"steps\": " << r.steps << ",\n";
  os << "  \"dt\": " << r.dt << ",\n";
  os << "  \"initial_mass\": " << r.initialMass << ",\n";
  os << "  \"final_mass\": " << r.finalMass << ",\n";
  os << "  \"alpha_mass_drift\": " << r.alphaMassDrift << ",\n";
  os << "  \"mass_budget_residual\": " << r.massBudgetResidual << ",\n";
  os << "  \"relative_mass_budget_residual\": " << r.relativeMassBudgetResidual << ",\n";
  os << "  \"min_alpha\": " << r.minAlpha << ",\n";
  os << "  \"max_alpha\": " << r.maxAlpha << ",\n";
  os << "  \"initial_tip_y\": " << r.initialTipY << ",\n";
  os << "  \"final_tip_y\": " << r.finalTipY << ",\n";
  os << "  \"tip_displacement\": " << r.tipDisplacement << ",\n";
  os << "  \"max_div\": " << r.maxDiv << ",\n";
  os << "  \"max_potential_residual\": " << r.maxPotentialResidual << ",\n";
  os << "  \"max_electric_force\": " << r.maxElectricForce << ",\n";
  os << "  \"max_csf_force\": " << r.maxCsfForce << ",\n";
  os << "  \"max_curvature\": " << r.maxCurvature << ",\n";
  os << "  \"curvature_fallback_fraction\": " << r.curvatureFallbackFraction << ",\n";
  os << "  \"max_charge\": " << r.maxCharge << ",\n";
  os << "  \"min_charge\": " << r.minCharge << ",\n";
  os << "  \"final_integrated_charge\": " << r.finalIntegratedCharge << ",\n";
  os << "  \"relative_charge_budget_residual\": " << r.relativeChargeBudgetResidual << ",\n";
  os << "  \"final_radial_asymmetry\": " << r.finalRadialAsymmetry << ",\n";
  os << "  \"final_midplane_jet_radius\": " << r.finalMidplaneJetRadius << ",\n";
  os << "  \"max_velocity\": " << r.maxVelocity << ",\n";
  // P5: paper observables surfaced as first-class outputs - the electric Courant
  // number dt/tau_e (paper limits this to <=0.1), the connected alpha=0.5 silhouette
  // volume V = sum(pi x^2) (paper morphology metric), and the jet current i_e.
  os << "  \"electric_courant\": " << r.dtOverElectricRelaxationLimit << ",\n";
  os << "  \"min_electric_force_cfl_raw\": " << r.minElectricForceCflRaw << ",\n";
  os << "  \"min_adaptive_dt\": " << r.minAdaptiveDt << ",\n";
  if (!r.history.empty()) {
    const auto& last = r.history.back();
    os << "  \"morphology_alpha05_silhouette_di3\": " << last.rayAlpha05SilhouetteVolumeDi3 << ",\n";
    os << "  \"morphology_connected_di3\": " << last.connectedMorphologyVolumeDi3 << ",\n";
    os << "  \"alpha05_convective_current\": " << last.alpha05ConvectiveCurrent << ",\n";
    os << "  \"jet_total_current\": " << last.totalCurrent << "\n";
  } else {
    os << "  \"morphology_alpha05_silhouette_di3\": 0,\n";
    os << "  \"morphology_connected_di3\": 0,\n";
    os << "  \"alpha05_convective_current\": 0,\n";
    os << "  \"jet_total_current\": 0\n";
  }
  os << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const RunnerOptions args = parseArgs(argc, argv);
    if (!args.caseDir.empty()) {
      std::filesystem::create_directories(args.outputDir);
      fvm::OpenFoamPolyMeshReadReport3D meshReport;
      const fvm::Mesh3D mesh =
          fvm::readOpenFoamPolyMesh3D(args.caseDir / "constant" / "polyMesh", &meshReport);
      const auto quality = fvm::meshQualityReport3D(mesh);
      fvm::OpenFoamCaseValidationReport3D fieldReport;
      const auto fields = fvm::readOpenFoamCaseFields3D(args.caseDir, mesh, &fieldReport);
      writeOpenFoamCaseSummaryJson(args.outputDir / "summary.json", mesh, quality,
                                   meshReport, fields, fieldReport);
      writeOpenFoamBoundaryCsv(args.outputDir / "openfoam_boundary_fields.csv", fields);
      const bool ok = quality.finite && quality.nonPositiveVolumeCount == 0 &&
                      quality.zeroAreaFaceCount == 0 && fieldReport.fieldsRead >= 5 &&
                      fieldReport.missingPatchBoundaryEntries == 0 &&
                      fieldReport.unknownPatchBoundaryEntries == 0;
      std::cout << "case_runner_status=" << (ok ? "pass" : "fail")
                << " run_mode=openfoam_case_validate output_dir=" << args.outputDir
                << " cells=" << quality.cells
                << " fields_read=" << fieldReport.fieldsRead
                << " missing_patch_bc=" << fieldReport.missingPatchBoundaryEntries
                << " unknown_patch_bc=" << fieldReport.unknownPatchBoundaryEntries << "\n";
      return ok ? 0 : 1;
    }

    const std::string text = readText(args.casePath);
    const std::string runMode = jsonString(text, "run_mode").value_or("candido_smoke");
    const std::string meshMode = jsonString(text, "mesh_mode").value_or("builtin_hex");
    const std::string caseName = jsonString(text, "case_name").value_or("gui_case");

    std::filesystem::create_directories(args.outputDir);
    if (runMode == "validate_mesh") {
      fvm::OpenFoamPolyMeshReadReport3D foamReport;
      fvm::Mesh3D mesh = buildCaseMesh(text, meshMode, &foamReport);
      const auto q = fvm::meshQualityReport3D(mesh);
      if (!q.finite || q.nonPositiveVolumeCount != 0 || q.zeroAreaFaceCount != 0) {
        throw std::runtime_error("mesh quality validation failed");
      }
      writeMeshSummaryJson(args.outputDir / "summary.json", mesh, q, meshMode, foamReport);
      std::cout << "case_runner_status=pass run_mode=validate_mesh output_dir="
                << args.outputDir << "\n";
      return 0;
    }

    auto smokeOpt = smokeOptionsFromJson(text);
    electrospray::CandidoTaylorConeJetSetup setup;
    setup.innerDiameter = jsonDoubleOr(text, "inner_diameter", setup.innerDiameter);
    setup.outerDiameter = jsonDoubleOr(text, "outer_diameter", setup.outerDiameter);
    setup.nozzleLength = jsonDoubleOr(text, "nozzle_length", setup.nozzleLength);
    setup.collectorDistance = jsonDoubleOr(text, "collector_distance", setup.collectorDistance);
    setup.collectorDiameter = jsonDoubleOr(text, "collector_diameter", setup.collectorDiameter);
    setup.validationVoltage = jsonDoubleOr(text, "validation_voltage", setup.validationVoltage);
    setup.validationFlowRate = jsonDoubleOr(text, "validation_flow_rate", setup.validationFlowRate);
    setup.collectorSpeed = jsonDoubleOr(text, "collector_speed", setup.collectorSpeed);
    setup.contactAngleDeg = jsonDoubleOr(text, "contact_angle_deg", setup.contactAngleDeg);
    setup.liquidDensity = jsonDoubleOr(text, "liquid_density", setup.liquidDensity);
    setup.gasDensity = jsonDoubleOr(text, "gas_density", setup.gasDensity);
    setup.liquidViscosity = jsonDoubleOr(text, "liquid_viscosity", setup.liquidViscosity);
    setup.gasViscosity = jsonDoubleOr(text, "gas_viscosity", setup.gasViscosity);
    setup.surfaceTension = jsonDoubleOr(text, "surface_tension", setup.surfaceTension);
    setup.liquidRelativePermittivity =
        jsonDoubleOr(text, "liquid_relative_permittivity", setup.liquidRelativePermittivity);
    setup.gasRelativePermittivity =
        jsonDoubleOr(text, "gas_relative_permittivity", setup.gasRelativePermittivity);
    setup.liquidConductivity = jsonDoubleOr(text, "liquid_conductivity", setup.liquidConductivity);
    setup.gasConductivity = jsonDoubleOr(text, "gas_conductivity", setup.gasConductivity);

    const double targetCaE = jsonDoubleOr(text, "target_ca_e", 0.35);

    // External-mesh path (spike): run the solver on a user-supplied OpenFOAM
    // polyMesh instead of the built-in structured hex box. The mesh is normalized
    // into the solver's nondimensional box convention before injection.
    fvm::Mesh3D externalMesh;
    const fvm::Mesh3D* externalMeshPtr = nullptr;
    int externalInletFaces = 0;
    bool inletFromPatch = false;
    if (meshMode == "openfoam_polyMesh") {
      const auto polyPath = jsonString(text, "openfoam_polyMesh");
      if (!polyPath || polyPath->empty()) {
        throw std::runtime_error(
            "openfoam_polyMesh path is required for candido_smoke on an external mesh");
      }
      fvm::OpenFoamPolyMeshReadReport3D meshReport;
      const fvm::Mesh3D raw = fvm::readOpenFoamPolyMesh3D(*polyPath, &meshReport);
      const auto norm = normalizeExternalCandidoMesh(raw, setup, smokeOpt);
      externalMesh = norm.mesh;
      inletFromPatch = norm.inletFromPatch;
      for (const auto& f : externalMesh.faces) {
        if (electrospray::candidoIsInletBoundaryFace3D(externalMesh, f, setup, smokeOpt)) {
          ++externalInletFaces;
        }
      }
      externalMeshPtr = &externalMesh;
    } else if (meshMode != "builtin_hex") {
      throw std::runtime_error("unknown mesh_mode for candido_smoke: " + meshMode);
    }

    const auto report =
        electrospray::runCandidoConeJetSmoke3D(targetCaE, setup, smokeOpt, externalMeshPtr);
    writeSmokeSummaryJson(args.outputDir / "summary.json", report, caseName, meshMode, smokeOpt);
    writeSmokeHistoryCsv(args.outputDir / "history.csv", report);
    std::cout << "case_runner_status=pass run_mode=candido_smoke output_dir=" << args.outputDir
              << " mesh_mode=" << meshMode
              << " external_inlet_faces=" << externalInletFaces
              << " inlet_from_patch=" << (inletFromPatch ? 1 : 0)
              << " cells=" << report.cells << " steps=" << report.steps
              << " mass_drift=" << report.alphaMassDrift
              << " max_div=" << report.maxDiv << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "case_runner_error=" << e.what() << "\n";
    return 2;
  }
}
