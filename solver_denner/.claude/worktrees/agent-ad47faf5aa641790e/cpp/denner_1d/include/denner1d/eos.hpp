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
