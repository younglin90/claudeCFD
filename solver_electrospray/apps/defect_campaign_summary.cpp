#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Row = std::map<std::string, std::string>;
using Table = std::map<std::string, Row>;

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream input(line);
  while (std::getline(input, field, ',')) fields.push_back(field);
  if (!line.empty() && line.back() == ',') fields.emplace_back();
  return fields;
}

Table readRows(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open " + path.string());
  std::string headerLine;
  if (!std::getline(in, headerLine)) throw std::runtime_error("empty csv " + path.string());
  std::vector<std::string> headers = splitCsvLine(headerLine);
  Table table;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> values = splitCsvLine(line);
    Row row;
    for (size_t i = 0; i < headers.size(); ++i) row[headers[i]] = i < values.size() ? values[i] : "";
    table[row.at("case_id")] = row;
  }
  return table;
}

double asDouble(const Row& row, const std::string& key) { return std::stod(row.at(key)); }

std::map<std::string, double> normalize(const std::map<std::string, double>& values) {
  double maximum = 0.0;
  for (const auto& [_, value] : values) maximum = std::max(maximum, value);
  std::map<std::string, double> out;
  for (const auto& [key, value] : values) out[key] = maximum <= 0.0 ? 0.0 : value / maximum;
  return out;
}

std::string formatFixed(double value, int precision) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

}  // namespace

int main() {
  try {
    const std::filesystem::path root = "results";
    const std::filesystem::path outputDir = root / "defect_campaign";
    Table screening = readRows(root / "defect_screening" / "defect_screening_summary.csv");
    Table geometry = readRows(root / "defect_electrostatic_geometry" / "electrostatic_geometry_summary.csv");
    Table timeHistory = readRows(root / "defect_ehd_time_history" / "time_history_summary.csv");

    std::vector<std::string> caseIds;
    for (const auto& [caseId, _] : screening) caseIds.push_back(caseId);

    std::map<std::string, double> proxyOffset, plume, field, highFieldOffset, finalOffset, growth;
    for (const std::string& caseId : caseIds) {
      proxyOffset[caseId] = asDouble(screening.at(caseId), "nonaxisymmetric_offset");
      plume[caseId] = asDouble(screening.at(caseId), "plume_divergence_proxy_deg");
      field[caseId] = asDouble(geometry.at(caseId), "field_enhancement");
      highFieldOffset[caseId] = asDouble(geometry.at(caseId), "high_field_offset_um");
      finalOffset[caseId] = asDouble(timeHistory.at(caseId), "final_offset");
      growth[caseId] = asDouble(timeHistory.at(caseId), "offset_growth");
    }
    auto normProxy = normalize(proxyOffset);
    auto normPlume = normalize(plume);
    auto normField = normalize(field);
    auto normHighFieldOffset = normalize(highFieldOffset);
    auto normFinalOffset = normalize(finalOffset);
    auto normGrowth = normalize(growth);

    struct CombinedRow {
      std::string caseId;
      double score = 0.0;
    };
    std::vector<CombinedRow> combinedRows;
    for (const std::string& caseId : caseIds) {
      const double score = 0.18 * normProxy[caseId] + 0.12 * normPlume[caseId] + 0.20 * normField[caseId] +
                           0.15 * normHighFieldOffset[caseId] + 0.25 * normFinalOffset[caseId] +
                           0.10 * normGrowth[caseId];
      combinedRows.push_back({caseId, score});
    }
    std::sort(combinedRows.begin(), combinedRows.end(), [](const CombinedRow& a, const CombinedRow& b) {
      return a.score > b.score;
    });

    std::filesystem::create_directories(outputDir);
    {
      std::ofstream csv(outputDir / "combined_defect_campaign.csv");
      csv << "case_id,label,difficulty,screening_cae,screening_offset,plume_divergence_proxy_deg,"
             "field_enhancement,high_field_offset_um,reduced_final_offset,reduced_offset_growth,"
             "reduced_max_pass_metric,reduced_class,campaign_score\n";
      for (const auto& item : combinedRows) {
        const std::string& caseId = item.caseId;
        const Row& s = screening.at(caseId);
        const Row& g = geometry.at(caseId);
        const Row& t = timeHistory.at(caseId);
        csv << caseId << ',' << s.at("label") << ',' << s.at("difficulty") << ',' << s.at("local_cae") << ','
            << s.at("nonaxisymmetric_offset") << ',' << s.at("plume_divergence_proxy_deg") << ','
            << g.at("field_enhancement") << ',' << g.at("high_field_offset_um") << ',' << t.at("final_offset")
            << ',' << t.at("offset_growth") << ',' << t.at("max_pass_metric") << ',' << t.at("risk_class") << ','
            << formatFixed(item.score, 6) << '\n';
      }
    }

    std::ofstream md(outputDir / "defect_campaign_report.md");
    md << "# Electrospray Tip-Defect Calculation Campaign\n\n"
       << "This report combines three calculation levels: dimensional defect screening, 3D electrostatic geometry, and "
          "stabilized reduced 3D EHD drift response.\n"
       << "The campaign score is only a prioritization metric for the next simulations, not a validated physical "
          "instability threshold.\n\n"
       << "## Priority Order\n\n"
       << "| Rank | Difficulty | Case | Score | Field enh. | High-field offset (um) | Final offset | Growth | Reduced "
          "class |\n"
       << "|---:|---:|---|---:|---:|---:|---:|---:|---|\n";
    int rank = 1;
    for (const auto& item : combinedRows) {
      const Row& s = screening.at(item.caseId);
      const Row& g = geometry.at(item.caseId);
      const Row& t = timeHistory.at(item.caseId);
      md << "| " << rank++ << " | " << s.at("difficulty") << " | " << s.at("label") << " | "
         << formatFixed(item.score, 3) << " | " << formatFixed(asDouble(g, "field_enhancement"), 2) << " | "
         << formatFixed(asDouble(g, "high_field_offset_um"), 2) << " | " << formatFixed(asDouble(t, "final_offset"), 4)
         << " | " << formatFixed(asDouble(t, "offset_growth"), 2) << " | " << t.at("risk_class") << " |\n";
    }
    md << "\n## Calculation Interpretation\n\n"
       << "- Start manuscript figures from the normal, 10 deg bent, 0.10Do bump, 0.10Do split, and severe oxidized cases.\n"
       << "- Bump defects mainly amplify local electric field; bending mainly shifts the high-field centroid; split defects "
          "dominate combined whipping risk.\n"
       << "- Oxidation reduces local electrostatic enhancement in the geometry solve, but rough asymmetric drift still "
          "increases reduced plume/offset risk.\n"
       << "- Projection-enabled long-time 3D whipping remains a solver-development target, because the coarse PIMPLE-style "
          "reduced run became non-finite before 40 steps.\n";

    std::cout << "Wrote combined campaign report to " << outputDir.string() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "defect_campaign_summary error: " << e.what() << "\n";
    return 1;
  }
}
