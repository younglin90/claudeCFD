#pragma once

#include "denner1d/types.hpp"

namespace denner1d {

PrimitiveState solve_case(const CaseDefinition& c);
void refresh_thermo(PrimitiveState& s, const Phase& a, const Phase& b);

}  // namespace denner1d
