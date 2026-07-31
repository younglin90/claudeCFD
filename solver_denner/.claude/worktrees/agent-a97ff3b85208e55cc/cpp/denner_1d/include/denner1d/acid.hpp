#pragma once

#include "denner1d/types.hpp"

// Faithful Denner ACID pressure-based solver (JCP 367, 2018).
// See docs/denner_acid_faithful.md for the method spec.
//   - 4-eq one-fluid (continuity + momentum + enthalpy-energy) + VOF colour function
//   - stiffened-gas EOS, collocated FV, implicit (Backward Euler)
//   - fully-coupled u-p block-tridiagonal solve (block-Thomas in 1D) + MWI advecting
//     velocity (Rhie-Chow), segregated enthalpy in the outer loop
//   - ACID interface treatment (added incrementally)
namespace denner1d {

// Stiffened-gas phase from Denner Table 1 spec: (gamma, Pi, rho0, a0) at (p0, T0).
// Returns a Phase with b=0 (pure stiffened gas), kv=cv derived from rho0/a0.
Phase denner_sg_phase(double gamma, double pinf, double rho0, double a0,
                      double p0 = 1.0e5, double T0 = 300.0);

PrimitiveState solve_case_acid(const CaseDefinition& c);

}  // namespace denner1d
