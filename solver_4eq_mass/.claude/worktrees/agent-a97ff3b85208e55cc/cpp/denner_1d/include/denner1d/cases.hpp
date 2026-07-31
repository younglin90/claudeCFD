#pragma once

#include "denner1d/types.hpp"

#include <vector>

namespace denner1d {

std::vector<CaseDefinition> all_cases();
CaseDefinition find_case(const std::string& id_or_prefix);
PrimitiveState initial_state(const CaseDefinition& c);
PrimitiveState reference_state(const CaseDefinition& c);

}  // namespace denner1d
