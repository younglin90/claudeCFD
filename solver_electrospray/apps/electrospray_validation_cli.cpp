#include "electrospray/Validation.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage() {
  std::cerr << "usage: electrospray_validation_cli [--format FORMAT] [--output path]"
               " [--check-*] [--write-artifacts]\n"
               "formats: json, markdown, reduced-step-json, plume-json, microthruster-json,"
               " application-json, artifact-status-json, submission-claim-audit-json\n";
}

std::filesystem::path sourceRoot() {
#ifdef FVM_SOURCE_DIR
  return std::filesystem::path(FVM_SOURCE_DIR);
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path artifactDir() {
  return sourceRoot() / "docs" / "electrospray";
}

std::string readTextFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open artifact: " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool nonEmptyFile(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec) && std::filesystem::file_size(path, ec) > 0;
}

bool pngFile(const std::filesystem::path& path) {
  if (!nonEmptyFile(path)) return false;
  std::ifstream in(path, std::ios::binary);
  const unsigned char expected[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  unsigned char actual[8] = {};
  in.read(reinterpret_cast<char*>(actual), 8);
  return in.gcount() == 8 && std::equal(std::begin(expected), std::end(expected), std::begin(actual));
}

const std::map<std::string, std::filesystem::path>& artifactPaths() {
  static const std::map<std::string, std::filesystem::path> paths = {
      {"validation_report", artifactDir() / "validation_report.md"},
      {"validation_summary", artifactDir() / "validation_summary.json"},
      {"reduced_step", artifactDir() / "reduced_phase_pair_step_report.json"},
      {"plume", artifactDir() / "plume_impingement_report.json"},
      {"microthruster", artifactDir() / "microthruster_operating_point_report.json"},
      {"application", artifactDir() / "application_report.json"},
      {"huh_wirz_metadata", artifactDir() / "huh_wirz_conejet_benchmark_metadata.json"},
      {"das_saintillan_metadata", artifactDir() / "das_saintillan_droplet_benchmark_metadata.json"},
      {"external_benchmark_readiness", artifactDir() / "external_benchmark_readiness_report.json"},
      {"submission_claim_audit", artifactDir() / "submission_claim_audit.json"},
      {"submission_readiness_matrix", artifactDir() / "submission_readiness_matrix.md"},
      {"cone_jet_error_budget_table", artifactDir() / "cone_jet_error_budget_table.md"},
      {"external_benchmark_numeric_comparison_table",
       artifactDir() / "external_benchmark_numeric_comparison_table.md"},
      {"full_cfd_huh_wirz_nonbreakup_comparison_table",
       artifactDir() / "full_cfd_huh_wirz_nonbreakup_comparison_table.md"},
      {"full_cfd_huh_wirz_subgrid_breakup_comparison_table",
       artifactDir() / "full_cfd_huh_wirz_subgrid_breakup_comparison_table.md"},
      {"taylor_cone_voltage_ramp_balance_table",
       artifactDir() / "taylor_cone_voltage_ramp_balance_table.md"},
      {"coupled_droplet_grid_refinement_table",
       artifactDir() / "coupled_droplet_grid_refinement_table.md"},
      {"dielectric_maxwell_droplet_history_table",
       artifactDir() / "dielectric_maxwell_droplet_history_table.md"},
      {"huh_wirz_same_path_grid_refinement_table",
       artifactDir() / "huh_wirz_same_path_grid_refinement_table.md"},
      {"full_cfd_readiness_report", artifactDir() / "full_cfd_readiness_report.json"},
      {"full_cfd_readiness_gates", artifactDir() / "full_cfd_readiness_gates.md"},
      {"field_contour_manifest", artifactDir() / "field_contour_manifest.md"},
      {"figure_manifest", artifactDir() / "figure_manifest.md"},
      {"figure_cone_jet_error_budget", artifactDir() / "figures" / "cone_jet_error_budget.png"},
      {"figure_coupled_droplet_grid_refinement",
       artifactDir() / "figures" / "coupled_droplet_grid_refinement.png"},
      {"figure_external_benchmark_numeric_comparison",
       artifactDir() / "figures" / "external_benchmark_numeric_comparison.png"},
      {"figure_taylor_cone_voltage_ramp",
       artifactDir() / "figures" / "taylor_cone_voltage_ramp.png"},
  };
  return paths;
}

bool artifactOk(const std::string& name, const std::filesystem::path& path) {
  if (name.rfind("figure_", 0) == 0 && path.extension() == ".png") return pngFile(path);
  return nonEmptyFile(path);
}

bool artifactsPresent(const std::vector<std::string>& names) {
  const auto& paths = artifactPaths();
  for (const std::string& name : names) {
    auto it = paths.find(name);
    if (it == paths.end() || !artifactOk(name, it->second)) return false;
  }
  return true;
}

std::string artifactStatusJson() {
  std::ostringstream os;
  os << "{";
  bool first = true;
  for (const auto& [name, path] : artifactPaths()) {
    if (!first) os << ",";
    first = false;
    os << "\"" << electrospray::jsonEscape(name) << "\":"
       << (artifactOk(name, path) ? "true" : "false");
  }
  os << "}\n";
  return os.str();
}

std::filesystem::path artifactForFormat(const std::string& format) {
  if (format == "reduced-step-json") return artifactPaths().at("reduced_step");
  if (format == "plume-json") return artifactPaths().at("plume");
  if (format == "microthruster-json") return artifactPaths().at("microthruster");
  if (format == "application-json") return artifactPaths().at("application");
  if (format == "submission-claim-audit-json") return artifactPaths().at("submission_claim_audit");
  throw std::runtime_error("no file-backed artifact for format: " + format);
}

int checkArtifacts(const std::vector<std::string>& names,
                   const std::string& okMessage,
                   const std::string& staleMessage) {
  if (artifactsPresent(names)) {
    std::cout << okMessage << "\n";
    return 0;
  }
  std::cout << staleMessage << "\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string format = "json";
  std::filesystem::path output;
  bool hasOutput = false;
  bool writeArtifacts = false;
  bool checkArtifactsFlag = false;
  bool checkReducedStepArtifact = false;
  bool checkPlumeArtifact = false;
  bool checkMicrothrusterArtifact = false;
  bool checkApplicationArtifact = false;
  bool checkSubmissionClaimAuditArtifact = false;
  bool checkAllArtifacts = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--format") {
      if (i + 1 >= argc) {
        usage();
        return 2;
      }
      format = argv[++i];
    } else if (arg == "--output") {
      if (i + 1 >= argc) {
        usage();
        return 2;
      }
      output = argv[++i];
      hasOutput = true;
    } else if (arg == "--write-artifacts") {
      writeArtifacts = true;
    } else if (arg == "--check-artifacts") {
      checkArtifactsFlag = true;
    } else if (arg == "--check-reduced-step-artifact") {
      checkReducedStepArtifact = true;
    } else if (arg == "--check-plume-artifact") {
      checkPlumeArtifact = true;
    } else if (arg == "--check-microthruster-artifact") {
      checkMicrothrusterArtifact = true;
    } else if (arg == "--check-application-artifact") {
      checkApplicationArtifact = true;
    } else if (arg == "--check-submission-claim-audit-artifact") {
      checkSubmissionClaimAuditArtifact = true;
    } else if (arg == "--check-all-artifacts") {
      checkAllArtifacts = true;
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage();
      return 2;
    }
  }

  const std::vector<std::string> supportedFormats = {
      "json",
      "markdown",
      "reduced-step-json",
      "plume-json",
      "microthruster-json",
      "application-json",
      "artifact-status-json",
      "submission-claim-audit-json",
  };
  if (std::find(supportedFormats.begin(), supportedFormats.end(), format) == supportedFormats.end()) {
    std::cerr << "unsupported format: " << format << "\n";
    return 2;
  }

  try {
    if (checkAllArtifacts) {
      std::vector<std::string> names;
      for (const auto& [name, _] : artifactPaths()) names.push_back(name);
      return checkArtifacts(names, "all validation artifacts current", "validation artifacts stale");
    }
    if (checkArtifactsFlag) {
      return checkArtifacts({"validation_report", "validation_summary"},
                            "validation artifacts current",
                            "validation artifacts stale");
    }
    if (checkReducedStepArtifact) {
      return checkArtifacts({"reduced_step"}, "reduced step artifact current",
                            "reduced step artifact stale");
    }
    if (checkPlumeArtifact) {
      return checkArtifacts({"plume"}, "plume artifact current", "plume artifact stale");
    }
    if (checkMicrothrusterArtifact) {
      return checkArtifacts({"microthruster"}, "microthruster artifact current",
                            "microthruster artifact stale");
    }
    if (checkApplicationArtifact) {
      return checkArtifacts({"application"}, "application artifact current",
                            "application artifact stale");
    }
    if (checkSubmissionClaimAuditArtifact) {
      return checkArtifacts({"submission_claim_audit"}, "submission claim audit artifact current",
                            "submission claim audit artifact stale");
    }

    std::vector<electrospray::ValidationResult> results =
        electrospray::runCoreValidationSuite();
    std::string text;
    if (format == "markdown") {
      text = electrospray::validationMarkdown(results);
    } else if (format == "json") {
      text = electrospray::validationSummaryJson(results) + "\n";
    } else if (format == "artifact-status-json") {
      text = artifactStatusJson();
    } else {
      text = readTextFile(artifactForFormat(format));
      if (text.empty() || text.back() != '\n') text.push_back('\n');
    }

    if (writeArtifacts) {
      std::filesystem::path root = sourceRoot();
      electrospray::writeTextFile(root / "docs" / "electrospray" / "validation_report_cpp.md",
                                  electrospray::validationMarkdown(results));
      electrospray::writeTextFile(root / "docs" / "electrospray" / "validation_summary_cpp.json",
                                  electrospray::validationSummaryJson(results) + "\n");
      std::cout << "C++ validation artifacts written\n";
      return 0;
    }

    if (hasOutput) {
      electrospray::writeTextFile(output, text);
    } else {
      std::cout << text;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "electrospray_validation_cli error: " << e.what() << "\n";
    return 1;
  }
}
