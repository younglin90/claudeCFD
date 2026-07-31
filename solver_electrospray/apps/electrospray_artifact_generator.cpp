#include "electrospray/Validation.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ArtifactEntry {
  std::filesystem::path relativePath;
  std::uintmax_t bytes = 0;
  std::uint64_t hash = 0;
  std::string kind;
  bool ok = false;
};

std::filesystem::path sourceRoot() {
#ifdef FVM_SOURCE_DIR
  return std::filesystem::path(FVM_SOURCE_DIR);
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path parseOutputDir(int argc, char** argv) {
  std::filesystem::path out = sourceRoot() / "build" / "generated_artifacts";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--output-dir") {
      if (i + 1 >= argc) throw std::runtime_error("--output-dir requires a path");
      out = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: electrospray_artifact_generator [--output-dir path]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return out;
}

std::vector<unsigned char> readBinary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to read artifact: " + path.string());
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(in), {});
}

std::uint64_t fnv1a64(const std::vector<unsigned char>& bytes) {
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char c : bytes) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ull;
  }
  return h;
}

std::string kindFor(const std::filesystem::path& p) {
  const std::string ext = p.extension().string();
  if (ext == ".json") return "json";
  if (ext == ".md") return "markdown";
  if (ext == ".csv") return "csv";
  if (ext == ".png") return "png";
  if (ext == ".txt" || ext == ".log") return "text";
  return "other";
}

bool validArtifact(const std::filesystem::path& p, const std::vector<unsigned char>& bytes) {
  if (bytes.empty()) return false;
  if (p.extension() == ".png") {
    const unsigned char sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    return bytes.size() >= 8 && std::equal(std::begin(sig), std::end(sig), bytes.begin());
  }
  return true;
}

bool trackedExtension(const std::filesystem::path& p) {
  const std::string ext = p.extension().string();
  return ext == ".json" || ext == ".md" || ext == ".csv" || ext == ".png" ||
         ext == ".txt" || ext == ".log";
}

std::vector<ArtifactEntry> collectArtifacts(const std::filesystem::path& root) {
  const std::filesystem::path src = sourceRoot();
  const std::vector<std::filesystem::path> dirs = {src / "docs" / "electrospray", src / "results"};
  std::vector<ArtifactEntry> entries;
  for (const auto& dir : dirs) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) continue;
    for (const auto& it : std::filesystem::recursive_directory_iterator(dir, ec)) {
      if (ec || !it.is_regular_file(ec) || !trackedExtension(it.path())) continue;
      std::vector<unsigned char> bytes = readBinary(it.path());
      ArtifactEntry e;
      e.relativePath = std::filesystem::relative(it.path(), root, ec);
      if (ec) e.relativePath = it.path().filename();
      e.bytes = bytes.size();
      e.hash = fnv1a64(bytes);
      e.kind = kindFor(it.path());
      e.ok = validArtifact(it.path(), bytes);
      entries.push_back(e);
    }
  }
  std::sort(entries.begin(), entries.end(), [](const ArtifactEntry& a, const ArtifactEntry& b) {
    return a.relativePath.generic_string() < b.relativePath.generic_string();
  });
  return entries;
}

void writeText(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to write: " + path.string());
  out << text;
}

std::string manifestCsv(const std::vector<ArtifactEntry>& entries) {
  std::ostringstream os;
  os << "path,bytes,fnv1a64,kind,status\n";
  os << std::hex << std::setfill('0');
  for (const ArtifactEntry& e : entries) {
    os << e.relativePath.generic_string() << "," << std::dec << e.bytes << ","
       << std::hex << std::setw(16) << e.hash << std::dec << "," << e.kind << ","
       << (e.ok ? "ok" : "invalid") << "\n";
  }
  return os.str();
}

std::string reportMarkdown(const std::vector<ArtifactEntry>& entries,
                           const std::vector<electrospray::ValidationResult>& results) {
  const int passed = static_cast<int>(std::count_if(results.begin(), results.end(),
                                                    [](const auto& r) { return r.passed; }));
  int invalid = 0;
  int png = 0;
  for (const ArtifactEntry& e : entries) {
    if (!e.ok) ++invalid;
    if (e.kind == "png") ++png;
  }
  std::ostringstream os;
  os << "# C++ Artifact Equivalence Report\n\n";
  os << "- artifact_count: " << entries.size() << "\n";
  os << "- png_count: " << png << "\n";
  os << "- invalid_count: " << invalid << "\n";
  os << "- validation_results: " << results.size() << "\n";
  os << "- validation_passed: " << passed << "\n";
  os << "- validation_summary_status: " << (passed == static_cast<int>(results.size()) ? "pass" : "fail") << "\n\n";
  os << "This report is generated by the C++ artifact generator and checks that tracked validation artifacts are present, non-empty, and hashable without Python runners.\n";
  return os.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::filesystem::path outDir = parseOutputDir(argc, argv);
    std::filesystem::create_directories(outDir);
    const auto results = electrospray::runCoreValidationSuite();
    writeText(outDir / "validation_summary.json", electrospray::validationSummaryJson(results) + "\n");
    writeText(outDir / "validation_report.md", electrospray::validationMarkdown(results));
    const auto artifacts = collectArtifacts(sourceRoot());
    writeText(outDir / "artifact_manifest.csv", manifestCsv(artifacts));
    writeText(outDir / "artifact_equivalence_report.md", reportMarkdown(artifacts, results));

    const int passed = static_cast<int>(std::count_if(results.begin(), results.end(),
                                                      [](const auto& r) { return r.passed; }));
    const int invalid = static_cast<int>(std::count_if(artifacts.begin(), artifacts.end(),
                                                       [](const auto& e) { return !e.ok; }));
    const int png = static_cast<int>(std::count_if(artifacts.begin(), artifacts.end(),
                                                   [](const auto& e) { return e.kind == "png"; }));
    std::cout << "artifact_generator_results=" << passed << "/" << results.size()
              << " artifacts=" << artifacts.size()
              << " png=" << png
              << " invalid=" << invalid
              << " output_dir=" << outDir << "\n";
    if (passed != static_cast<int>(results.size())) return 1;
    if (artifacts.size() < 20 || png < 1 || invalid != 0) return 1;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "artifact_generator_error=" << e.what() << "\n";
    return 2;
  }
}
