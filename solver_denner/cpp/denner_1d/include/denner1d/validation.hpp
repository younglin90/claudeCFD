#pragma once

#include "denner1d/types.hpp"

#include <string>
#include <vector>

namespace denner1d {

ErrorMetrics compare(const PrimitiveState& got, const PrimitiveState& ref);
std::string metrics_json(const std::string& case_id, const ErrorMetrics& m, int cells);
void write_comparison_png(const std::string& path,
                          const PrimitiveState& got,
                          const PrimitiveState& ref,
                          const std::string& title);
int validate_cases(const std::vector<std::string>& selected, const std::string& out_dir);

}  // namespace denner1d
