#pragma once

#include "denner1d/types.hpp"

namespace denner1d {

struct PhaseProps {
    double rho = 0.0;
    double c = 0.0;
    double h = 0.0;
    double e_vol = 0.0;
    double zeta = 0.0;
    double phi = 0.0;
    double dh_dp = 0.0;
    double cp = 0.0;
    double dEdp = 0.0;
    double dEdT = 0.0;
};

// Lean thermo for the h->T inner Newton (the hottest kernel: ~60 EOS evals per cell per
// compute_R). Only rho, h, and their T-partials phi=drho/dT|p, cp=dh/dT|p -- it OMITS the
// sound speed c (a sqrt + extra div) and the p-partials that the full phase_props always
// computes. Header-inline so it folds into the caller and the compiler optimizes it in place
// (phase_props lives in a separate TU, so a call there cannot be inlined/DCE'd without LTO).
// The rho/h/phi/cp formulas are byte-identical to phase_props -> T_from_hstat is unchanged.
struct PhaseThermo { double rho, h, phi, cp; };
inline PhaseThermo phase_thermo(double p, double T, const Phase& ph) {
    const double gm1 = ph.gamma - 1.0;
    const double A = ph.kv * T * gm1 + ph.b * (p + ph.pinf) + 1.0e-300;
    const double ppinf = p + ph.pinf;
    PhaseThermo o;
    o.rho = ppinf / A;
    o.h   = ph.gamma * ph.kv * T + ph.b * p + ph.eta;
    o.phi = -ppinf * ph.kv * gm1 / (A * A + 1.0e-300);
    o.cp  = ph.gamma * ph.kv;
    return o;
}

Phase air_phase();
Phase water_liquid_phase();
Phase water_vapor_phase();

PhaseProps phase_props(double p, double T, const Phase& phase);
double mixture_density(double p, double T, double alpha, const Phase& a, const Phase& b);
double mixture_sound_speed(double p, double T, double alpha, const Phase& a, const Phase& b);
double mixture_enthalpy(double p, double T, double alpha, const Phase& a, const Phase& b);
double mixture_internal_energy_density(double p, double T, double alpha, const Phase& a, const Phase& b);
bool recover_pressure_temperature_from_density_energy(double rho,
                                                      double internal_energy_density,
                                                      double alpha,
                                                      const Phase& a,
                                                      const Phase& b,
                                                      double& p,
                                                      double& T);

}  // namespace denner1d
