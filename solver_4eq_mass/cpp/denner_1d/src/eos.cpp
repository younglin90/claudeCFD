#include "denner1d/eos.hpp"

#include <algorithm>
#include <cmath>

namespace denner1d {
namespace {
constexpr double eps = 1.0e-300;
}

Phase air_phase() {
    // Denner Table-1 air: gamma=1.4, Pi=0, rho0=1.157, a0=347.8 at p0=1e5,T0=300
    // => R=p0/(rho0 T0)=288.10, cv=R/(gamma-1)=720.25 (a0=347.85).
    return Phase{1.4, 0.0, 0.0, 720.25, 0.0};
}

Phase water_liquid_phase() {
    return Phase{1.187, 7.028e8, 6.61e-4, 3610.0, -1.177788e6};
}

Phase water_vapor_phase() {
    return Phase{1.467, 0.0, 0.0, 955.0, 2.077616e6};
}

PhaseProps phase_props(double p, double T, const Phase& ph) {
    const double gm1 = ph.gamma - 1.0;
    const double A = ph.kv * T * gm1 + ph.b * (p + ph.pinf) + eps;
    const double ppinf = p + ph.pinf;
    PhaseProps out;
    out.rho = ppinf / A;
    out.h = ph.gamma * ph.kv * T + ph.b * p + ph.eta;
    const double one_minus = std::max(1.0 - ph.b * out.rho, 1.0e-12);
    out.c = std::sqrt(std::max(ph.gamma * ppinf / (out.rho * one_minus + eps), 0.0));
    const double inv_A2 = 1.0 / (A * A + eps);
    out.zeta = ph.kv * T * gm1 * inv_A2;
    out.phi = -ppinf * ph.kv * gm1 * inv_A2;
    out.dh_dp = ph.b;
    out.cp = ph.gamma * ph.kv;
    out.e_vol = out.rho * out.h - p;
    out.dEdp = out.rho * out.dh_dp + out.h * out.zeta - 1.0;
    out.dEdT = out.rho * out.cp + out.h * out.phi;
    return out;
}

double mixture_density(double p, double T, double alpha, const Phase& a, const Phase& b) {
    alpha = std::clamp(alpha, 0.0, 1.0);
    return alpha * phase_props(p, T, a).rho + (1.0 - alpha) * phase_props(p, T, b).rho;
}

double mixture_sound_speed(double p, double T, double alpha, const Phase& a, const Phase& b) {
    alpha = std::clamp(alpha, 0.0, 1.0);
    const auto pa = phase_props(p, T, a);
    const auto pb = phase_props(p, T, b);
    const double rho = alpha * pa.rho + (1.0 - alpha) * pb.rho;
    const double compress = alpha / (pa.rho * pa.c * pa.c + eps)
                          + (1.0 - alpha) / (pb.rho * pb.c * pb.c + eps);
    return std::sqrt(std::max(1.0 / (rho * compress + eps), 0.0));
}

double mixture_enthalpy(double p, double T, double alpha, const Phase& a, const Phase& b) {
    alpha = std::clamp(alpha, 0.0, 1.0);
    const auto pa = phase_props(p, T, a);
    const auto pb = phase_props(p, T, b);
    const double rho = alpha * pa.rho + (1.0 - alpha) * pb.rho + eps;
    return (alpha * pa.rho * pa.h + (1.0 - alpha) * pb.rho * pb.h) / rho;
}

double mixture_internal_energy_density(double p, double T, double alpha, const Phase& a, const Phase& b) {
    alpha = std::clamp(alpha, 0.0, 1.0);
    return alpha * phase_props(p, T, a).e_vol + (1.0 - alpha) * phase_props(p, T, b).e_vol;
}

bool recover_pressure_temperature_from_density_energy(double rho_target,
                                                      double e_target,
                                                      double alpha,
                                                      const Phase& a,
                                                      const Phase& b,
                                                      double& p,
                                                      double& T) {
    alpha = std::clamp(alpha, 0.0, 1.0);
    p = std::max(p, 1.0);
    T = std::max(T, 1.0e-6);
    bool converged = false;
    for (int it = 0; it < 20; ++it) {
        const auto pa = phase_props(p, T, a);
        const auto pb = phase_props(p, T, b);
        const double rho = alpha * pa.rho + (1.0 - alpha) * pb.rho;
        const double e = alpha * pa.e_vol + (1.0 - alpha) * pb.e_vol;
        const double f_rho = rho - rho_target;
        const double f_e = e - e_target;
        const double dr_dp = alpha * pa.zeta + (1.0 - alpha) * pb.zeta;
        const double dr_dT = alpha * pa.phi + (1.0 - alpha) * pb.phi;
        const double de_dp = alpha * pa.dEdp + (1.0 - alpha) * pb.dEdp;
        const double de_dT = alpha * pa.dEdT + (1.0 - alpha) * pb.dEdT;
        const double det = dr_dp * de_dT - dr_dT * de_dp;
        if (!std::isfinite(det) || std::abs(det) < eps) break;
        const double dp = (-f_rho * de_dT + dr_dT * f_e) / det;
        const double dT = (-dr_dp * f_e + f_rho * de_dp) / det;
        double damping = 1.0;
        bool accepted = false;
        for (int ls = 0; ls < 10; ++ls) {
            const double p_try = p + damping * dp;
            const double T_try = T + damping * dT;
            if (std::isfinite(p_try) && std::isfinite(T_try) && p_try > 0.0 && T_try > 0.0) {
                p = std::max(p_try, 1.0);
                T = std::max(T_try, 1.0e-6);
                accepted = true;
                break;
            }
            damping *= 0.5;
        }
        if (!accepted) break;
        const double p_tol = 1.0e-10 * std::max(std::abs(p), 1.0);
        const double T_tol = 1.0e-10 * std::max(std::abs(T), 1.0);
        if (std::abs(dp) <= p_tol && std::abs(dT) <= T_tol) {
            converged = true;
            break;
        }
    }
    p = std::max(p, 1.0);
    T = std::max(T, 1.0e-6);
    return converged && std::isfinite(p) && std::isfinite(T);
}

}  // namespace denner1d
