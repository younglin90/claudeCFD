// 1-D physical source terms. Port of source_terms.py; gravity path first.
#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "cfd/eos.hpp"
#include "cfd/primitive.hpp"
#include "cfd/five_eq/step.hpp"

namespace cfd::five_eq {

struct PhaseChangeConfig {
    bool enabled = false;
    bool isothermal = false;
    bool saturation_pressure_target = false;
    double tau = 1.0e-4;
    double T_sat = 373.15;
    std::optional<double> p_sat; // unset: Antoine relation; zero is a valid explicit target
    double latent_heat = 2.257e6;
};

struct HeatConductionConfig {
    bool enabled = false;
    bool primitive_temperature = false;
    double k_liquid = 0.6, k_vapor = 0.025;
    double T_left = std::numeric_limits<double>::quiet_NaN();
    double T_right = std::numeric_limits<double>::quiet_NaN();
};

inline double saturation_pressure_water(double T) {
    const double Tc = std::fmin(std::fmax(T - 273.15, 1.0), 100.0);
    return std::pow(10.0, 8.07131 - 1730.63 / (233.426 + Tc)) * 133.322368;
}

inline double lee_gamma(const PrimW& W, const EOS& eos1, const EOS& eos2,
                        const PhaseChangeConfig& cfg) {
    const double a = W.alpha1, rho1 = eos1.density(W.p, W.T1), rho2 = eos2.density(W.p, W.T2);
    const double psat = std::fmax(cfg.p_sat ? *cfg.p_sat : saturation_pressure_water(a * W.T1 + (1.0 - a) * W.T2), 1.0);
    const double tau = std::fmax(cfg.tau, PRIM_EPS);
    const double evap = (a * rho1 / tau) * std::fmax((psat - W.p) / psat, 0.0);
    const double cond = -((1.0 - a) * rho2 / tau) * std::fmax((W.p - psat) / psat, 0.0);
    if (std::fabs(W.p - psat) / psat >= 1.0e-10) return evap + cond;
    const double dT = (a * W.T1 + (1.0 - a) * W.T2 - cfg.T_sat) / std::fmax(cfg.T_sat, 1.0);
    return dT >= 0.0 ? (a * rho1 / tau) * dT : -((1.0 - a) * rho2 / tau) * (-dT);
}

inline StepResult apply_gravity_source(const StepResult& W, double dt, double gravity,
                                       const EOS& eos1, const EOS& eos2) {
    if (gravity == 0.0 || dt == 0.0) return W;
    StepResult out = W;
    const std::size_t n = W.alpha.size();
    for (std::size_t i = 0; i < n; ++i) {
        const PrimW wi{W.alpha[i], W.T1[i], W.T2[i], W.u[i], W.p[i]};
        ConsU U = prim_to_cons_W(wi, eos1, eos2);
        const double rho = std::fmax(U.m1 + U.m2, PRIM_EPS);
        const double u = U.mom / rho;
        const double du = gravity * dt;
        U.mom += rho * du;
        U.rhoE += rho * (u * du + 0.5 * du * du);
        const PrimW wo = cons_to_prim_W(U, eos1, eos2, 1.0e-9, 30, wi.T1, wi.T2);
        out.alpha[i] = wo.alpha1; out.T1[i] = wo.T1; out.T2[i] = wo.T2;
        out.u[i] = wo.u; out.p[i] = wo.p;
    }
    return out;
}

inline bool is_hydrostatic_equilibrium(const StepResult& W, double dx, double gravity,
                                       const EOS& eos1, const EOS& eos2) {
    if (gravity == 0.0 || W.alpha.empty()) return false;
    double umax = 0.0;
    for (double u : W.u) if (!std::isfinite(u)) return false; else umax = std::fmax(umax, std::fabs(u));
    if (umax > std::sqrt(std::numeric_limits<double>::epsilon()) * std::fmax(umax, 1.0)) return false;
    if (W.alpha.size() < 2) return true;
    double residual = 0.0;
    for (std::size_t i = 0; i + 1 < W.alpha.size(); ++i) {
        if (std::fabs(W.alpha[i + 1] - W.alpha[i]) >= 0.25) continue;
        const double rho_l = W.alpha[i] * eos1.density(W.p[i], W.T1[i]) +
                             (1.0 - W.alpha[i]) * eos2.density(W.p[i], W.T2[i]);
        const double rho_r = W.alpha[i + 1] * eos1.density(W.p[i + 1], W.T1[i + 1]) +
                             (1.0 - W.alpha[i + 1]) * eos2.density(W.p[i + 1], W.T2[i + 1]);
        const double target = 0.5 * (rho_l + rho_r) * gravity;
        residual = std::fmax(residual, std::fabs((W.p[i + 1] - W.p[i]) / dx - target) / std::fmax(std::fabs(target), 1.0));
    }
    return residual <= std::fmax(dx * dx, std::sqrt(std::numeric_limits<double>::epsilon()));
}

inline StepResult apply_phase_change_source(const StepResult& W, double dt,
                                            const EOS& eos1, const EOS& eos2,
                                            const PhaseChangeConfig& cfg) {
    if (!cfg.enabled || dt == 0.0) return W;
    StepResult out = W;
    for (std::size_t i = 0; i < W.alpha.size(); ++i) {
        const PrimW wi{W.alpha[i], W.T1[i], W.T2[i], W.u[i], W.p[i]};
        const double rho1 = eos1.density(wi.p, wi.T1), rho2 = eos2.density(wi.p, wi.T2);
        ConsU U = prim_to_cons_W(wi, eos1, eos2);
        double gamma = lee_gamma(wi, eos1, eos2, cfg);
        gamma = std::fmin(std::fmax(gamma, -U.m2 / std::fmax(std::fabs(dt), PRIM_EPS)),
                          U.m1 / std::fmax(std::fabs(dt), PRIM_EPS));
        const double q1 = U.m1 - dt * gamma, q2 = U.m2 + dt * gamma;
        const double rho = std::fmax(U.m1 + U.m2, PRIM_EPS), u = U.mom / rho;
        const double volume1 = q1 / std::fmax(rho1, PRIM_EPS), volume2 = q2 / std::fmax(rho2, PRIM_EPS);
        const double alpha = std::fmin(std::fmax(volume1 / std::fmax(volume1 + volume2, PRIM_EPS), 0.0), 1.0);
        if (cfg.isothermal) {
            out.alpha[i] = alpha;
            if (cfg.saturation_pressure_target)
                out.p[i] = cfg.p_sat ? *cfg.p_sat : saturation_pressure_water(alpha * wi.T1 + (1.0 - alpha) * wi.T2);
            continue;
        }
        const double internal = std::fmax(U.rhoE - 0.5 * rho * u * u - dt * gamma * cfg.latent_heat, PRIM_EPS);
        const double rho_new = std::fmax(q1 + q2, PRIM_EPS);
        U = ConsU{q1, q2, rho_new * u, internal + 0.5 * rho_new * u * u, alpha};
        const PrimW wo = cons_to_prim_W(U, eos1, eos2, 1.0e-9, 30, wi.T1, wi.T2);
        out.alpha[i] = wo.alpha1; out.T1[i] = wo.T1; out.T2[i] = wo.T2; out.u[i] = wo.u; out.p[i] = wo.p;
    }
    return out;
}

inline std::vector<double> heat_divergence(const StepResult& W, double dx,
                                           const HeatConductionConfig& cfg) {
    const int n = static_cast<int>(W.alpha.size());
    std::vector<double> T(n), k(n), flux(n + 1), div(n);
    for (int i = 0; i < n; ++i) { T[i] = W.alpha[i] * W.T1[i] + (1.0 - W.alpha[i]) * W.T2[i]; k[i] = W.alpha[i] * cfg.k_liquid + (1.0 - W.alpha[i]) * cfg.k_vapor; }
    const double left = std::isfinite(cfg.T_left) ? cfg.T_left : T.front();
    const double right = std::isfinite(cfg.T_right) ? cfg.T_right : T.back();
    flux[0] = -k[0] * (T[0] - left) / dx;
    for (int i = 1; i < n; ++i) flux[i] = -0.5 * (k[i - 1] + k[i]) * (T[i] - T[i - 1]) / dx;
    flux[n] = -k[n - 1] * (right - T[n - 1]) / dx;
    for (int i = 0; i < n; ++i) div[i] = -(flux[i + 1] - flux[i]) / dx;
    return div;
}

inline StepResult apply_heat_conduction_source(const StepResult& W, double dt, double dx,
                                               const EOS& eos1, const EOS& eos2,
                                               const HeatConductionConfig& cfg) {
    if (!cfg.enabled || dt == 0.0) return W;
    StepResult out = W; const auto div = heat_divergence(W, dx, cfg);
    for (std::size_t i = 0; i < W.alpha.size(); ++i) {
        const PrimW wi{W.alpha[i], W.T1[i], W.T2[i], W.u[i], W.p[i]};
        if (cfg.primitive_temperature) {
            const double rho1 = eos1.density(wi.p, wi.T1), rho2 = eos2.density(wi.p, wi.T2);
            const double rho_cv = wi.alpha1 * rho1 * eos1.kv + (1.0 - wi.alpha1) * rho2 * eos2.kv;
            const double dT = dt * div[i] / std::fmax(rho_cv, PRIM_EPS);
            out.T1[i] = std::fmax(wi.T1 + dT, 1.0); out.T2[i] = std::fmax(wi.T2 + dT, 1.0);
        } else {
            ConsU U = prim_to_cons_W(wi, eos1, eos2); U.rhoE += dt * div[i];
            const PrimW wo = cons_to_prim_W(U, eos1, eos2, 1.0e-9, 30, wi.T1, wi.T2);
            out.alpha[i] = wo.alpha1; out.T1[i] = wo.T1; out.T2[i] = wo.T2; out.u[i] = wo.u; out.p[i] = wo.p;
        }
    }
    return out;
}

// Exact source composition from source_terms.py::apply_source_terms.  The
// primitive-temperature heat policy deliberately takes precedence and leaves
// gravity out of that branch, matching the Python public solver contract.
inline StepResult apply_source_terms(const StepResult& W, double dt, double dx,
                                     const EOS& eos1, const EOS& eos2,
                                     double gravity, const PhaseChangeConfig& phase,
                                     const HeatConductionConfig& heat,
                                     double alpha_pure_tol = 0.0) {
    if (dt == 0.0) return W;
    if (gravity == 0.0 && !phase.enabled && !heat.enabled) return W;
    if (heat.enabled && heat.primitive_temperature) {
        StepResult out = apply_heat_conduction_source(W, dt, dx, eos1, eos2, heat);
        return (phase.enabled && phase.isothermal)
            ? apply_phase_change_source(out, dt, eos1, eos2, phase) : out;
    }
    if (phase.enabled && phase.isothermal && gravity == 0.0 && !heat.enabled)
        return apply_phase_change_source(W, dt, eos1, eos2, phase);

    StepResult out = W;
    const std::vector<double> heat_div = heat.enabled ? heat_divergence(W, dx, heat)
                                                       : std::vector<double>{};
    for (std::size_t i = 0; i < W.alpha.size(); ++i) {
        const PrimW wi{W.alpha[i], W.T1[i], W.T2[i], W.u[i], W.p[i]};
        ConsU U = prim_to_cons_W(wi, eos1, eos2);
        if (gravity != 0.0) {
            const double rho = std::fmax(U.m1 + U.m2, PRIM_EPS);
            const double u = U.mom / rho, du = gravity * dt;
            U.mom += rho * du;
            U.rhoE += rho * (u * du + 0.5 * du * du);
        }
        if (phase.enabled) {
            const double rho1 = eos1.density(wi.p, wi.T1), rho2 = eos2.density(wi.p, wi.T2);
            double gamma = lee_gamma(wi, eos1, eos2, phase);
            const double dt_abs = std::fmax(std::fabs(dt), PRIM_EPS);
            gamma = std::fmin(std::fmax(gamma, -U.m2 / dt_abs), U.m1 / dt_abs);
            const double q1 = U.m1 - dt * gamma, q2 = U.m2 + dt * gamma;
            const double rho = std::fmax(U.m1 + U.m2, PRIM_EPS), u = U.mom / rho;
            const double internal = std::fmax(U.rhoE - 0.5 * rho * u * u - dt * gamma * phase.latent_heat, PRIM_EPS);
            const double rho_new = std::fmax(q1 + q2, PRIM_EPS);
            const double v1 = q1 / std::fmax(rho1, PRIM_EPS), v2 = q2 / std::fmax(rho2, PRIM_EPS);
            const double alpha = std::fmin(std::fmax(v1 / std::fmax(v1 + v2, PRIM_EPS), 0.0), 1.0);
            U = ConsU{q1, q2, rho_new * u, internal + 0.5 * rho_new * u * u, alpha};
        }
        if (heat.enabled) U.rhoE += dt * heat_div[i];
        const PrimW wo = cons_to_prim_W(U, eos1, eos2, 1.0e-9, 30, wi.T1, wi.T2,
                                        alpha_pure_tol);
        out.alpha[i] = wo.alpha1; out.T1[i] = wo.T1; out.T2[i] = wo.T2;
        out.u[i] = wo.u; out.p[i] = wo.p;
    }
    return out;
}

} // namespace cfd::five_eq
