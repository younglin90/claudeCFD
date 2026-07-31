// cfd/five_eq/solver.hpp -- production time loop for the 1-D five-equation IMEX path.
//
// This is the C++ equivalent of main.py::solve for the supported one-dimensional
// Python solver surface: production IMEX, legacy BE/split variants, and SSP3.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "cfd/eos.hpp"
#include "cfd/primitive.hpp"
#include "cfd/five_eq/config.hpp"
#include "cfd/five_eq/sound_speed.hpp"
#include "cfd/five_eq/source_terms.hpp"
#include "cfd/five_eq/step.hpp"
#include "cfd/five_eq/explicit.hpp"
#include "cfd/five_eq/ars_solver.hpp"
#include "cfd/five_eq/pe_correction.hpp"
#include "cfd/five_eq/relaxation.hpp"

namespace cfd {
namespace five_eq {

enum class RunTermination {
    completed,
    max_steps,
    invalid_input,
    nonfinite_dt,
    dt_below_min,
    nonfinite_state,
    step_callback_stop,
};

struct StepRecord {
    int step = 0;
    double t = 0.0;
    double dt = 0.0;
    PressureClosure closure = PressureClosure::pressure_work_consistent;
    int vacuum_velocity_cells = 0;
};

struct RunConfig {
    StepConfig step_config{};
    double cfl = 0.5;
    std::optional<double> dt_fixed{};
    std::optional<double> dt_min{};
    int max_steps = 100000;
    bool stop_on_nonfinite = true;
    bool record_history = false;
    // Opt-in exact finite-volume remap for uniform-u, uniform-p periodic advection.
    bool uniform_periodic_remap = false;
    double gravity = 0.0;
    bool gravity_well_balanced = true;
    PhaseChangeConfig phase_change{};
    HeatConductionConfig heat_conduction{};
    std::function<double(double)> u_inlet_at;
    std::function<double(double)> p_inlet_at;
    std::function<double(double)> p_outlet_at;
    std::function<double(double)> alpha_inlet_at;
    std::function<double(double)> T1_inlet_at;
    std::function<double(double)> T2_inlet_at;
    std::function<bool(const StepRecord&, const StepResult&)> step_callback;

    // Python main.py::solve public defaults.  Keep StepConfig's production
    // imex_ad defaults intact for the established 02-A/07-B C++ entry point.
    static RunConfig python_solve_defaults();
};

inline RunConfig RunConfig::python_solve_defaults() {
    RunConfig config;
    StepConfig& step = config.step_config;
    step.bc_l = BC5::Transmissive;
    step.bc_r = BC5::Transmissive;
    step.time_integrator = TimeIntegrator::be1;
    step.kapila_closure = false;
    step.pure_branch = false;
    step.alpha_pure_tol = 1.e-8;
    step.ars_implicit_dissipation = .02;
    step.ars_implicit_dissipation_form = ImplicitDissipationForm::Biharmonic;
    step.ars_linear_solver = ARSLinearSolver::schur_helmholtz;
    step.be1_pe_project_explicit = true;
    step.be1_pe_projection_mode = PEProjectionMode::Always;
    step.be1_pe_projection_explicit_only = false;
    step.be1_pe_correct = false;
    step.be1_kapila_acoustic_source = false;
    step.be1_implicit_include_explicit_residual = false;
    step.be1_final_update_backtracking = true;
    step.be1_final_update_backtracking_steps = 12;
    step.be1_energy_form = EnergyForm::Secant;
    step.be1_face_options.alpha_scheme = AlphaFaceScheme::Muscl;
    step.be1_face_options.primitive_scheme = PrimitiveFaceScheme::Upwind;
    step.be1_face_options.thermo_scheme = FaceThermoScheme::Acid;
    step.be1_explicit_positivity = true;
    step.be1_explicit_force_low = true;
    step.be1_explicit_rusanov_low = false;
    return config;
}

struct RunResult {
    StepResult W;
    double t_final = 0.0;
    int steps = 0;
    RunTermination termination = RunTermination::completed;
    std::vector<StepRecord> history;
};

inline bool finite_state(const StepResult& W) {
    const std::size_t n = W.alpha.size();
    if (n == 0 || W.T1.size() != n || W.T2.size() != n ||
        W.u.size() != n || W.p.size() != n) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(W.alpha[i]) || !std::isfinite(W.T1[i]) ||
            !std::isfinite(W.T2[i]) || !std::isfinite(W.u[i]) ||
            !std::isfinite(W.p[i])) {
            return false;
        }
    }
    return true;
}

inline std::vector<double> periodic_cell_average_shift(const std::vector<double>& phi,
                                                        double shift_cells) {
    const int n = static_cast<int>(phi.size());
    if (n == 0) return {};
    const double k_floor = std::floor(shift_cells);
    const int k = static_cast<int>(k_floor);
    const double fraction = shift_cells - k_floor;
    const auto rolled = [&](int shift) {
        std::vector<double> out(n);
        for (int i = 0; i < n; ++i) {
            int source = (i - shift) % n;
            if (source < 0) source += n;
            out[i] = phi[source];
        }
        return out;
    };
    if (fraction <= 1.e-15) return rolled(k);
    if (1.0 - fraction <= 1.e-15) return rolled(k + 1);
    const auto low = rolled(k), high = rolled(k + 1);
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) out[i] = (1.0 - fraction) * low[i] + fraction * high[i];
    return out;
}

inline bool try_uniform_periodic_advection_remap(RunResult& out, double dx, double t_end,
                                                  const EOS& e1, const EOS& e2,
                                                  const RunConfig& config) {
    const StepResult& W = out.W;
    const StepConfig& cfg = config.step_config;
    if (!config.uniform_periodic_remap || config.gravity != 0.0 ||
        config.phase_change.enabled || config.heat_conduction.enabled ||
        cfg.bc_l != BC5::Periodic || cfg.bc_r != BC5::Periodic ||
        config.step_callback || config.u_inlet_at || config.p_inlet_at || config.p_outlet_at ||
        cfg.u_inlet_l || cfg.p_inlet_l || cfg.p_outlet_r || W.alpha.empty()) return false;
    const double u0 = W.u.front(), p0 = W.p.front();
    const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
    for (std::size_t i = 0; i < W.alpha.size(); ++i) {
        if (std::fabs(W.u[i] - u0) > tolerance * std::fmax(std::fabs(u0), 1.0) ||
            std::fabs(W.p[i] - p0) > tolerance * std::fmax(std::fabs(p0), 1.0)) return false;
    }
    const auto U = conservative_cells(W, e1, e2);
    std::vector<double> q1(U.size()), q2(U.size()), alpha(U.size());
    for (std::size_t i = 0; i < U.size(); ++i) { q1[i] = U[i].m1; q2[i] = U[i].m2; alpha[i] = U[i].a1; }
    const double shift = u0 * t_end / dx;
    q1 = periodic_cell_average_shift(q1, shift);
    q2 = periodic_cell_average_shift(q2, shift);
    alpha = periodic_cell_average_shift(alpha, shift);
    out.W.alpha.resize(U.size()); out.W.T1.resize(U.size()); out.W.T2.resize(U.size());
    out.W.u.assign(U.size(), u0); out.W.p.assign(U.size(), p0);
    for (std::size_t i = 0; i < U.size(); ++i) {
        out.W.alpha[i] = std::clamp(alpha[i], 1.e-12, 1.0 - 1.e-12);
        const double rho1 = std::fmax(q1[i] / out.W.alpha[i], 1.e-30);
        const double rho2 = std::fmax(q2[i] / (1.0 - out.W.alpha[i]), 1.e-30);
        out.W.T1[i] = e1.temperature(rho1, e1.energy(rho1, p0));
        out.W.T2[i] = e2.temperature(rho2, e2.energy(rho2, p0));
    }
    const double reported_dt = config.dt_fixed.has_value() && *config.dt_fixed > 0.0 ? *config.dt_fixed : t_end;
    out.t_final = t_end;
    out.steps = config.dt_fixed.has_value() && *config.dt_fixed > 0.0
        ? static_cast<int>(std::llround(t_end / *config.dt_fixed)) : 1;
    if (config.record_history) out.history.push_back({out.steps, t_end, reported_dt, out.W.closure, 0});
    return finite_state(out.W);
}

// main.py::_max_acoustic_dt for imex_ad.  The Python driver deliberately does
// not enable the pure-phase sound-speed override in its imex_ad CFL estimate.
inline double max_acoustic_dt_imex_ad(const StepResult& W, const EOS& eos1,
                                      const EOS& eos2, double dx,
                                      MixtureSoundSpeedKind mixture_kind = MixtureSoundSpeedKind::Kapila,
                                      double alpha_pure_tol = 0.0) {
    double max_speed = 0.0;
    for (std::size_t i = 0; i < W.alpha.size(); ++i) {
        const PhaseAcoustic pa = phase_acoustic(
            eos1, eos2, W.alpha[i], W.T1[i], W.T2[i], W.p[i], alpha_pure_tol, mixture_kind);
        const double c = std::sqrt(std::fmax(pa.c_mix_sq, 1.0e-30));
        max_speed = std::fmax(max_speed, std::fabs(W.u[i]) + c);
    }
    return max_speed > 0.0 ? dx / max_speed
                           : std::numeric_limits<double>::infinity();
}

// main.py::_imex_ad_ssp2_step.  The two imex_ad stages are combined in U,
// never in W, preserving phase masses and total energy in mixed cells.
inline StepResult imex_ad_ssp2_step(const StepResult& W_n, double dt, double dx,
                                    const EOS& eos1, const EOS& eos2,
                                    const StepConfig& cfg) {
    const StepResult W_1 = imex_ad_step(W_n.alpha, W_n.T1, W_n.T2, W_n.u, W_n.p,
                                        dt, dx, eos1, eos2, cfg);
    const StepResult W_2 = imex_ad_step(W_1.alpha, W_1.T1, W_1.T2, W_1.u, W_1.p,
                                        dt, dx, eos1, eos2, cfg);
    StepResult out;
    const std::size_t n = W_n.alpha.size();
    out.alpha.resize(n); out.T1.resize(n); out.T2.resize(n); out.u.resize(n); out.p.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const PrimW wn{W_n.alpha[i], W_n.T1[i], W_n.T2[i], W_n.u[i], W_n.p[i]};
        const PrimW w2{W_2.alpha[i], W_2.T1[i], W_2.T2[i], W_2.u[i], W_2.p[i]};
        const ConsU un = prim_to_cons_W(wn, eos1, eos2);
        const ConsU u2 = prim_to_cons_W(w2, eos1, eos2);
        const ConsU blend{0.5 * (un.m1 + u2.m1), 0.5 * (un.m2 + u2.m2),
                          0.5 * (un.mom + u2.mom), 0.5 * (un.rhoE + u2.rhoE),
                          0.5 * (un.a1 + u2.a1)};
        const PrimW w = cons_to_prim_W(blend, eos1, eos2, 1.0e-9, 30,
                                       0.5 * (wn.T1 + w2.T1), 0.5 * (wn.T2 + w2.T2));
        out.alpha[i] = w.alpha1; out.T1[i] = w.T1; out.T2[i] = w.T2;
        out.u[i] = w.u; out.p[i] = w.p;
    }
    out.closure = W_2.closure;
    out.vacuum_velocity_cells = W_2.vacuum_velocity_cells;
    return out;
}

inline StepResult blend_conservative(const StepResult& a, const StepResult& b, double theta,
                                     const EOS& eos1, const EOS& eos2) {
    StepResult out; const std::size_t n = a.alpha.size();
    out.alpha.resize(n); out.T1.resize(n); out.T2.resize(n); out.u.resize(n); out.p.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const PrimW wa{a.alpha[i], a.T1[i], a.T2[i], a.u[i], a.p[i]}, wb{b.alpha[i], b.T1[i], b.T2[i], b.u[i], b.p[i]};
        const ConsU ua = prim_to_cons_W(wa, eos1, eos2), ub = prim_to_cons_W(wb, eos1, eos2);
        const ConsU u{theta * ua.m1 + (1-theta)*ub.m1, theta*ua.m2+(1-theta)*ub.m2,
                      theta*ua.mom+(1-theta)*ub.mom, theta*ua.rhoE+(1-theta)*ub.rhoE,
                      theta*ua.a1+(1-theta)*ub.a1};
        const PrimW w = cons_to_prim_W(u, eos1, eos2, 1e-13, 50,
                                       theta*wa.T1+(1-theta)*wb.T1, theta*wa.T2+(1-theta)*wb.T2);
        out.alpha[i]=w.alpha1; out.T1[i]=w.T1; out.T2[i]=w.T2; out.u[i]=w.u; out.p[i]=w.p;
    }
    return out;
}

// Shared finite-difference Newton core for the legacy BE paths.  ARS has a
// specialised residual; BE-full also needs the explicit operator evaluated at
// the current Newton state.
template<class ResidualEvaluator>
inline StepResult generic_newton_solve(const StepResult& initial, const EOS& e1,
                                       const EOS& e2, ResidualEvaluator&& evaluate,
                                       NewtonInfo& info, int max_iter = 15,
                                       double rtol = 1.e-8, double atol = 1.e-12,
                                       int line_search_max = 10, double eta = 1.e-4) {
    StepResult W = initial;
    const int n = static_cast<int>(W.alpha.size()), m = 5 * n;
    const auto scale = residual_scales(W, e1, e2);
    auto eval = [&](const StepResult& x) { return flatten_residual(evaluate(x)); };
    std::vector<double> R = eval(W);
    double norm = residual_norm(R, scale), norm0 = std::fmax(norm, atol);
    if (norm <= atol + rtol * norm0) { info = {true, 0, norm}; return W; }
    for (int it = 0; it < max_iter; ++it) {
        std::vector<double> J(m * m), minus_R(m);
        for (int q = 0; q < m; ++q) minus_R[q] = -R[q];
        for (int col = 0; col < m; ++col) {
            StepResult P = W;
            const int cell = col / 5, component = col % 5;
            double* x = component == 0 ? &P.alpha[cell] : component == 1 ? &P.T1[cell]
                      : component == 2 ? &P.T2[cell] : component == 3 ? &P.u[cell] : &P.p[cell];
            const double h = component == 0 ? 1.e-7 : std::fmax(std::fabs(*x) * 1.e-7, 1.e-7);
            *x += h;
            const auto Rp = eval(P);
            for (int row = 0; row < m; ++row) J[row * m + col] = (Rp[row] - R[row]) / h;
        }
        for (int q = 0; q < m; ++q) J[q * m + q] += 1.e-12;
        if (!dense_solve(std::move(J), minus_R, m)) { info = {false, it, norm}; return W; }
        bool accepted = false;
        double lambda = 1.0;
        for (int ls = 0; ls < line_search_max; ++ls) {
            StepResult trial = W;
            add_primitive_delta(trial, minus_R, lambda);
            if (admissible(trial)) {
                const auto Rt = eval(trial);
                const double trial_norm = residual_norm(Rt, scale);
                if (trial_norm <= (1.0 - eta * lambda) * norm) {
                    W = std::move(trial); R = Rt; norm = trial_norm; accepted = true; break;
                }
            }
            lambda *= 0.5;
        }
        if (!accepted) { info = {false, it, norm}; return W; }
        if (norm <= atol + rtol * norm0) { info = {true, it + 1, norm}; return W; }
    }
    info = {false, max_iter, norm};
    return W;
}

inline ExplicitResidual integrator_explicit_residual(const StepResult& W, double dt,
                                                      double dx, const EOS& e1, const EOS& e2,
                                                      const StepConfig& cfg) {
    FaceStateOptions options = cfg.ars_face_options;
    options.dt = dt; options.dx = dx; options.has_dt_dx = true;
    return explicit_residual(W, e1, e2, dx, cfg.bc_l, cfg.bc_r,
                             cfg.kapila_closure, false, cfg.explicit_energy_form,
                             cfg.u_inlet_l, cfg.p_inlet_l, cfg.alpha_inlet_l,
                             cfg.T1_inlet_l, cfg.T2_inlet_l,
                             cfg.ars_explicit_positivity, cfg.ars_explicit_force_low,
                             cfg.ars_explicit_rusanov_low, dt, options);
}

inline ImplicitDivergences integrator_implicit_divergences(const StepResult& W, double gamma_dt,
                                                           double dx, const EOS& e1, const EOS& e2,
                                                           const StepConfig& cfg) {
    return implicit_divergences(W, e1, e2, dx, cfg.bc_l, cfg.bc_r,
                                cfg.ars_implicit_dissipation, cfg.ars_biharmonic(), cfg.ars_rhie_chow,
                                gamma_dt, cfg.ars_effective_acoustic_riemann(),
                                cfg.ars_effective_upwind_dissipation(),
                                cfg.ars_implicit_compact_lap_coeff);
}

inline StepResult residual_update(const std::vector<ConsU>& U, const Residual5& R,
                                  double dt, const StepResult& guess,
                                  const EOS& e1, const EOS& e2) {
    std::vector<ConsU> next = U;
    for (std::size_t i = 0; i < next.size(); ++i) {
        next[i].m1 -= dt * R.m1[i]; next[i].m2 -= dt * R.m2[i];
        next[i].mom -= dt * R.mom[i]; next[i].rhoE -= dt * R.rhoE[i];
        next[i].a1 -= dt * R.alpha[i];
    }
    return primitive_cells(next, guess, e1, e2);
}

// strang_step and split_step call residual.py::explicit_residual directly in
// Python.  They therefore use that function's fixed defaults, rather than
// main.py's configurable BE1 face options.
inline StepConfig python_legacy_explicit_config(const StepConfig& base) {
    StepConfig out = base;
    out.explicit_energy_form = EnergyForm::Secant; // residual.py: energy_form='apec'
    out.ars_face_options = FaceStateOptions{};     // upwind/upwind/ACID
    out.ars_explicit_positivity = true;
    out.ars_explicit_force_low = false;
    out.ars_explicit_rusanov_low = false;
    return out;
}

// Python time_integrator.py::be1_step final conservative update.  The trial
// update is halved until primitive recovery is finite/admissible; exhaustion
// returns W_n so a diagnostic march stays finite.
inline StepResult be1_residual_update(const std::vector<ConsU>& U, const Residual5& R,
                                      double dt, const StepResult& guess,
                                      const StepResult& fallback,
                                      const EOS& e1, const EOS& e2,
                                      const StepConfig& cfg) {
    const int trials = cfg.be1_final_update_backtracking
        ? std::max(cfg.be1_final_update_backtracking_steps, 1) : 1;
    double theta = 1.0;
    for (int attempt = 0; attempt < trials; ++attempt) {
        StepResult out;
        out.alpha.resize(U.size()); out.T1.resize(U.size()); out.T2.resize(U.size());
        out.u.resize(U.size()); out.p.resize(U.size());
        bool valid = true;
        for (std::size_t i = 0; i < U.size(); ++i) {
            const ConsU next{U[i].m1 - theta * dt * R.m1[i],
                             U[i].m2 - theta * dt * R.m2[i],
                             U[i].mom - theta * dt * R.mom[i],
                             U[i].rhoE - theta * dt * R.rhoE[i],
                             U[i].a1 - theta * dt * R.alpha[i]};
            const PrimW w = cons_to_prim_W(next, e1, e2, 1.e-9, 30,
                                            guess.T1[i], guess.T2[i],
                                            cfg.pure_branch ? cfg.alpha_pure_tol : 0.0);
            valid = valid && std::isfinite(w.alpha1) && std::isfinite(w.T1) &&
                    std::isfinite(w.T2) && std::isfinite(w.u) && std::isfinite(w.p) &&
                    w.alpha1 > 1.e-12 && w.alpha1 < 1.0 - 1.e-12 &&
                    w.T1 > 1.0 && w.T2 > 1.0 && w.p > 1.0;
            out.alpha[i] = w.alpha1; out.T1[i] = w.T1; out.T2[i] = w.T2;
            out.u[i] = w.u; out.p[i] = w.p;
        }
        if (valid) return out;
        theta *= 0.5;
    }
    return fallback;
}

inline StepResult strang_step(const StepResult& Wn, double dt, double dx,
                              const EOS& e1, const EOS& e2, const StepConfig& cfg) {
    const auto U = conservative_cells(Wn, e1, e2);
    StepConfig explicit_cfg = python_legacy_explicit_config(cfg);
    explicit_cfg.kapila_closure = false;
    auto E = integrator_explicit_residual(Wn, dt, dx, e1, e2, explicit_cfg);
    Residual5 R{E.m1, E.m2, E.mom, E.rhoE, E.alpha, {}};
    return residual_update(U, R, dt, Wn, e1, e2);
}

inline StepResult split_step(const StepResult& Wn, double dt, double dx,
                             const EOS& e1, const EOS& e2, const StepConfig& cfg,
                             NewtonInfo* info_out = nullptr) {
    std::vector<ConsU> Ua = conservative_cells(Wn, e1, e2);
    StepResult Wa = Wn;
    const int substeps = std::max(cfg.split_advection_substeps, 1);
    const double sub_dt = dt / static_cast<double>(substeps);
    const StepConfig explicit_cfg = python_legacy_explicit_config(cfg);
    for (int s = 0; s < substeps; ++s) {
        const auto E = integrator_explicit_residual(Wa, sub_dt, dx, e1, e2, explicit_cfg);
        for (std::size_t i = 0; i < Ua.size(); ++i) {
            Ua[i].m1 -= sub_dt * E.m1[i]; Ua[i].m2 -= sub_dt * E.m2[i];
            Ua[i].mom -= sub_dt * E.mom[i]; Ua[i].rhoE -= sub_dt * E.rhoE[i];
            Ua[i].a1 -= sub_dt * E.alpha[i];
        }
        Wa = primitive_cells(Ua, Wa, e1, e2);
    }
    NewtonInfo info;
    const StepResult out = ars_newton_solve(Wa, Ua, dt, e1, e2, dx, cfg.bc_l, cfg.bc_r,
                                            info, cfg.newton_max_iter, cfg.newton_rtol, cfg.newton_atol,
                                            cfg.ars_implicit_dissipation, cfg.ars_biharmonic(),
                                            cfg.ars_rhie_chow, cfg.ars_effective_acoustic_riemann(),
                                            cfg.ars_effective_upwind_dissipation(),
                                            cfg.ars_implicit_compact_lap_coeff, nullptr,
                                            cfg.newton_line_search_max, cfg.newton_eta);
    if (info_out) *info_out = info;
    return out;
}

// Defined below with the SSP3 helpers; BE1 uses the same PE-contact gate for
// Python's '*_explicit' projection modes.
inline bool ssp3_pe_projection_allowed(const StepResult& W, const EOS& e1, const EOS& e2);

inline StepResult be1_step(const StepResult& Wn, double dt, double dx,
                           const EOS& e1, const EOS& e2, const StepConfig& cfg,
                           NewtonInfo* info_out = nullptr) {
    const auto Un = conservative_cells(Wn, e1, e2);
    StepConfig explicit_cfg = cfg;
    explicit_cfg.explicit_energy_form = cfg.be1_energy_form;
    explicit_cfg.ars_face_options = cfg.be1_face_options;
    explicit_cfg.ars_explicit_positivity = cfg.be1_explicit_positivity;
    explicit_cfg.ars_explicit_force_low = cfg.be1_explicit_force_low;
    explicit_cfg.ars_explicit_rusanov_low = cfg.be1_explicit_rusanov_low;
    const auto E = integrator_explicit_residual(Wn, dt, dx, e1, e2, explicit_cfg);
    std::vector<ConsU> implicit_target = Un;
    if (cfg.be1_implicit_include_explicit_residual) {
        for (std::size_t i = 0; i < implicit_target.size(); ++i) {
            implicit_target[i].m1 -= dt * E.m1[i];
            implicit_target[i].m2 -= dt * E.m2[i];
            implicit_target[i].mom -= dt * E.mom[i];
            implicit_target[i].rhoE -= dt * E.rhoE[i];
            implicit_target[i].a1 -= dt * E.alpha[i];
        }
    }
    std::vector<double> kapila;
    const std::vector<double>* kapila_ptr = nullptr;
    if (cfg.kapila_closure && cfg.be1_kapila_acoustic_source) {
        kapila.resize(Wn.alpha.size());
        for (std::size_t i = 0; i < kapila.size(); ++i)
            kapila[i] = detail::D_K_kapila_cell(e1, e2, Wn.alpha[i], Wn.T1[i],
                                                 Wn.T2[i], Wn.p[i]);
        kapila_ptr = &kapila;
    }
    NewtonInfo info;
    const bool use_schur = cfg.ars_linear_solver == ARSLinearSolver::schur_helmholtz &&
                           !kapila_ptr && cfg.ars_biharmonic() && !cfg.ars_rhie_chow &&
                           !cfg.ars_effective_acoustic_riemann() &&
                           !cfg.ars_effective_upwind_dissipation() &&
                           cfg.ars_implicit_compact_lap_coeff == 0.0;
    const StepResult Wimp = use_schur
        ? ars_schur_solve(Wn, implicit_target, dt, e1, e2, dx, cfg.bc_l, cfg.bc_r,
                          info, cfg.newton_max_iter, cfg.newton_rtol, cfg.newton_atol,
                          cfg.ars_implicit_dissipation, cfg.newton_line_search_max,
                          cfg.newton_eta)
        : ars_newton_solve(Wn, implicit_target, dt, e1, e2, dx, cfg.bc_l, cfg.bc_r,
                           info, cfg.newton_max_iter, cfg.newton_rtol, cfg.newton_atol,
                           cfg.ars_implicit_dissipation, cfg.ars_biharmonic(),
                           cfg.ars_rhie_chow, cfg.ars_effective_acoustic_riemann(),
                           cfg.ars_effective_upwind_dissipation(),
                           cfg.ars_implicit_compact_lap_coeff, kapila_ptr,
                           cfg.newton_line_search_max, cfg.newton_eta);
    const auto I = integrator_implicit_divergences(Wimp, dt, dx, e1, e2, cfg);
    Residual5 total{E.m1, E.m2, E.mom, E.rhoE, E.alpha, {}};
    for (std::size_t i = 0; i < Wn.alpha.size(); ++i) {
        total.mom[i] += I.grad_p[i]; total.rhoE[i] += I.div_pu[i];
        if (kapila_ptr) total.alpha[i] -= (*kapila_ptr)[i] * I.div_u[i];
    }
    if (cfg.be1_pe_project_explicit) {
        if (cfg.be1_pe_projection_explicit_only) {
            Residual5 projected{E.m1, E.m2, E.mom, E.rhoE, E.alpha, {}};
            if (ssp3_pe_projection_allowed(Wn, e1, e2))
                apply_pe_tangent_projection(projected, Wn, e1, e2, nullptr,
                                            cfg.be1_pe_projection_mode);
            total = std::move(projected);
            for (std::size_t i = 0; i < Wn.alpha.size(); ++i) {
                total.mom[i] += I.grad_p[i]; total.rhoE[i] += I.div_pu[i];
                if (kapila_ptr) total.alpha[i] -= (*kapila_ptr)[i] * I.div_u[i];
            }
        } else {
            apply_pe_tangent_projection(total, Wimp, e1, e2, nullptr,
                                        cfg.be1_pe_projection_mode);
        }
    } else if (cfg.be1_pe_correct) {
        apply_pe_energy_correction(total, Wimp, e1, e2);
    }
    if (info_out) *info_out = info;
    double relative_update = 0.0;
    for (std::size_t i = 0; i < Un.size(); ++i) {
        relative_update = std::fmax(relative_update,
            std::fabs(dt * total.m1[i]) / std::fmax(std::fabs(Un[i].m1), 1.0));
        relative_update = std::fmax(relative_update,
            std::fabs(dt * total.m2[i]) / std::fmax(std::fabs(Un[i].m2), 1.0));
        relative_update = std::fmax(relative_update,
            std::fabs(dt * total.mom[i]) / std::fmax(std::fabs(Un[i].mom), 1.0));
        relative_update = std::fmax(relative_update,
            std::fabs(dt * total.rhoE[i]) / std::fmax(std::fabs(Un[i].rhoE), 1.0));
        relative_update = std::fmax(relative_update,
            std::fabs(dt * total.alpha[i]) / std::fmax(std::fabs(Un[i].a1), 1.0));
    }
    if (relative_update <= cfg.be1_zero_update_tol) return Wn;
    return be1_residual_update(Un, total, dt, Wimp, Wn, e1, e2, cfg);
}

inline StepResult be_full_step(const StepResult& Wn, double dt, double dx,
                               const EOS& e1, const EOS& e2, const StepConfig& cfg,
                               NewtonInfo* info_out = nullptr) {
    const auto Un = conservative_cells(Wn, e1, e2);
    auto residual = [&](const StepResult& W) {
        const auto now = conservative_cells(W, e1, e2);
        const auto E = integrator_explicit_residual(W, dt, dx, e1, e2, cfg);
        const auto I = integrator_implicit_divergences(W, dt, dx, e1, e2, cfg);
        Residual5 R{E.m1, E.m2, E.mom, E.rhoE, E.alpha, I};
        for (std::size_t i = 0; i < now.size(); ++i) {
            R.m1[i] += (now[i].m1 - Un[i].m1) / dt;
            R.m2[i] += (now[i].m2 - Un[i].m2) / dt;
            R.mom[i] += (now[i].mom - Un[i].mom) / dt + I.grad_p[i];
            R.rhoE[i] += (now[i].rhoE - Un[i].rhoE) / dt + I.div_pu[i];
            R.alpha[i] += (now[i].a1 - Un[i].a1) / dt;
        }
        return R;
    };
    NewtonInfo info;
    const StepResult out = generic_newton_solve(Wn, e1, e2, residual, info,
                                                cfg.newton_max_iter, cfg.newton_rtol,
                                                cfg.newton_atol, cfg.newton_line_search_max,
                                                cfg.newton_eta);
    if (info_out) *info_out = info;
    return out;
}

// time_integrator.py::imex_ad_ssp3_step, FIVE_EQ_IMEX_SSP3_SCOPE=full_step.
inline StepResult imex_ad_ssp3_full_step(const StepResult& W, double dt, double dx,
                                         const EOS& e1, const EOS& e2, const StepConfig& cfg) {
    const auto G = [&](const StepResult& s) { return imex_ad_step(s.alpha,s.T1,s.T2,s.u,s.p,dt,dx,e1,e2,cfg); };
    const StepResult w1 = G(W), w2 = blend_conservative(W, G(w1), 0.75, e1, e2);
    return blend_conservative(W, G(w2), 1.0/3.0, e1, e2);
}

// Pareschi-Russo IMEX-SSP3(4,3,3), the `stage_residual` choice in Python
// time_integrator.py.  Its default explicit operator is the production
// material map converted back to L_E, not the older face-residual path.
inline ExplicitResidual material_rk_residual(const StepResult& W, double dt, double dx,
                                             const EOS& e1, const EOS& e2,
                                             const StepConfig& cfg) {
    const auto U = conservative_cells(W, e1, e2);
    MaterialConfig material_cfg = cfg.material_config();
    material_cfg.energy_form = cfg.ssp3_material_energy_form;
    const auto material = material_update(W.alpha, W.T1, W.T2, W.u, W.p, dt, dx,
                                          e1, e2, material_cfg);
    ExplicitResidual E;
    const std::size_t n = U.size();
    E.m1.resize(n); E.m2.resize(n); E.mom.resize(n); E.rhoE.resize(n); E.alpha.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        E.m1[i] = (U[i].m1 - material.q1_new[i]) / dt;
        E.m2[i] = (U[i].m2 - material.q2_new[i]) / dt;
        E.mom[i] = (U[i].mom - material.m_adv[i]) / dt;
        E.rhoE[i] = (U[i].rhoE - material.rhoE_new[i]) / dt;
        E.alpha[i] = (U[i].a1 - material.alpha_new[i]) / dt;
    }
    return E;
}

inline bool ssp3_pe_projection_allowed(const StepResult& W, const EOS& e1, const EOS& e2) {
    const std::size_t n = W.alpha.size();
    if (n < 2) return true;
    double alpha_jump = 0.0, p_scale = 1.0, p_jump = 0.0, u_jump = 0.0, c_ref = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        p_scale = std::fmax(p_scale, std::fabs(W.p[i]));
        if (i + 1 < n) {
            alpha_jump = std::fmax(alpha_jump, std::fabs(W.alpha[i + 1] - W.alpha[i]));
            p_jump = std::fmax(p_jump, std::fabs(W.p[i + 1] - W.p[i]));
            u_jump = std::fmax(u_jump, std::fabs(W.u[i + 1] - W.u[i]));
        }
        const PhaseAcoustic pa = phase_acoustic(e1, e2, W.alpha[i], W.T1[i], W.T2[i], W.p[i], 0.0);
        c_ref = std::fmax(c_ref, std::sqrt(std::fmax(pa.c_mix_sq, 1.e-30)));
    }
    return alpha_jump > 1.e-8 && p_jump / p_scale <= 1.e-10 && u_jump / c_ref <= 1.e-10;
}

inline std::vector<ConsU> ssp3_project_target_to_pe(std::vector<ConsU> target,
                                                     const StepResult& reference,
                                                     const EOS& e1, const EOS& e2,
                                                     double alpha_pure_tol) {
    if (!ssp3_pe_projection_allowed(reference, e1, e2)) return target;
    double u0 = 0.0, p0 = 0.0;
    for (std::size_t i = 0; i < target.size(); ++i) { u0 += reference.u[i]; p0 += reference.p[i]; }
    u0 /= static_cast<double>(target.size());
    p0 /= static_cast<double>(target.size());
    if (!std::isfinite(u0) || !std::isfinite(p0) || !(p0 > 0.0)) return target;
    double T1_min = reference.T1.front(), T1_max = reference.T1.front();
    double T2_min = reference.T2.front(), T2_max = reference.T2.front();
    double T10 = 0.0, T20 = 0.0;
    for (std::size_t i = 0; i < target.size(); ++i) {
        T1_min = std::fmin(T1_min, reference.T1[i]);
        T1_max = std::fmax(T1_max, reference.T1[i]);
        T2_min = std::fmin(T2_min, reference.T2[i]);
        T2_max = std::fmax(T2_max, reference.T2[i]);
        T10 += reference.T1[i];
        T20 += reference.T2[i];
    }
    T10 /= static_cast<double>(target.size());
    T20 /= static_cast<double>(target.size());
    const double iso_tol = regime_detail::eps025();
    const bool isothermal =
        (T1_max - T1_min) <= iso_tol * std::fmax(std::fabs(T10), 1.0) &&
        (T2_max - T2_min) <= iso_tol * std::fmax(std::fabs(T20), 1.0);
    const double lo = std::fmax(alpha_pure_tol, 1.e-12);
    long double sum_q1 = 0.0L, sum_q2 = 0.0L;
    long double sum_a1 = 0.0L, sum_a2 = 0.0L;
    for (const ConsU& U : target) {
        const double alpha = std::clamp(U.a1, lo, 1.0 - lo);
        sum_q1 += std::fmax(U.m1, 0.0);
        sum_q2 += std::fmax(U.m2, 0.0);
        sum_a1 += alpha;
        sum_a2 += 1.0 - alpha;
    }
    const double rho1_iso = sum_a1 > 0.0L ? static_cast<double>(sum_q1 / sum_a1) : 0.0;
    const double rho2_iso = sum_a2 > 0.0L ? static_cast<double>(sum_q2 / sum_a2) : 0.0;
    const bool project_isothermal = isothermal && std::isfinite(rho1_iso) &&
        std::isfinite(rho2_iso) && rho1_iso > 0.0 && rho2_iso > 0.0;
    for (ConsU& U : target) {
        const double alpha = std::clamp(U.a1, lo, 1.0 - lo);
        const double q1 = project_isothermal ? alpha * rho1_iso : std::fmax(U.m1, 0.0);
        const double q2 = project_isothermal ? (1.0 - alpha) * rho2_iso : std::fmax(U.m2, 0.0);
        const double rho1 = std::fmax(q1 / std::fmax(alpha, 1.e-12), 1.e-30);
        const double rho2 = std::fmax(q2 / std::fmax(1.0 - alpha, 1.e-12), 1.e-30);
        const double rho = std::fmax(q1 + q2, 1.e-30);
        const double rhoE = q1 * e1.energy(rho1, p0) + q2 * e2.energy(rho2, p0)
                          + 0.5 * rho * u0 * u0;
        if (std::isfinite(rhoE)) U = {q1, q2, rho * u0, rhoE, alpha};
    }
    return target;
}

inline StepResult ssp3_relax(StepResult W, SSP3PERelaxation relaxation,
                             const EOS& e1, const EOS& e2) {
    if (relaxation == SSP3PERelaxation::Pressure) return relax_pressure(W, e1, e2);
    if (relaxation == SSP3PERelaxation::PressureTemperature) return relax_pT(W, e1, e2);
    return W;
}

inline StepResult imex_ssp3_stage_residual_step(const StepResult& Wn, double dt, double dx,
                                                const EOS& e1, const EOS& e2,
                                                const StepConfig& cfg) {
    constexpr double a = 0.24169426078821;
    constexpr double beta = 0.06042356519705;
    constexpr double eta = 0.12915286960590;
    constexpr double delta = 0.5 - beta - eta - a;
    constexpr double AE[4][4] = {{0.,0.,0.,0.}, {0.,0.,0.,0.}, {0.,1.,0.,0.}, {0.,.25,.25,0.}};
    constexpr double AI[4][4] = {{a,0.,0.,0.}, {-a,a,0.,0.}, {0.,1.-a,a,0.}, {beta,eta,delta,a}};
    constexpr double B[4] = {0., 1./6., 1./6., 2./3.};
    const auto Un = conservative_cells(Wn, e1, e2);
    std::vector<StepResult> stages;
    std::vector<ExplicitResidual> explicit_stages;
    std::vector<ImplicitDivergences> implicit_stages;
    stages.reserve(4); explicit_stages.reserve(4); implicit_stages.reserve(4);
    for (int i = 0; i < 4; ++i) {
        std::vector<ConsU> target = Un;
        for (int j = 0; j < i; ++j) {
            for (std::size_t cell = 0; cell < target.size(); ++cell) {
                target[cell].m1 -= dt * AE[i][j] * explicit_stages[j].m1[cell];
                target[cell].m2 -= dt * AE[i][j] * explicit_stages[j].m2[cell];
                target[cell].mom -= dt * (AE[i][j] * explicit_stages[j].mom[cell]
                                           + AI[i][j] * implicit_stages[j].grad_p[cell]);
                target[cell].rhoE -= dt * (AE[i][j] * explicit_stages[j].rhoE[cell]
                                            + AI[i][j] * implicit_stages[j].div_pu[cell]);
                target[cell].a1 -= dt * AE[i][j] * explicit_stages[j].alpha[cell];
            }
        }
        const StepResult& guess = stages.empty() ? Wn : stages.back();
        target = ssp3_project_target_to_pe(std::move(target), guess, e1, e2, cfg.alpha_pure_tol);
        const bool pe_target = ssp3_pe_projection_allowed(guess, e1, e2);
        NewtonInfo info;
        StepResult Wi = pe_target
            ? primitive_cells(target, guess, e1, e2)
            : ars_newton_solve(guess, target, AI[i][i] * dt,
                               e1, e2, dx, cfg.bc_l, cfg.bc_r, info,
                               cfg.newton_max_iter, cfg.newton_rtol, cfg.newton_atol,
                               cfg.ars_implicit_dissipation, cfg.ars_biharmonic(),
                               cfg.ars_rhie_chow, cfg.ars_effective_acoustic_riemann(),
                               cfg.ars_effective_upwind_dissipation(),
                               cfg.ars_implicit_compact_lap_coeff, nullptr,
                               cfg.newton_line_search_max, cfg.newton_eta);
        Wi = ssp3_relax(std::move(Wi), cfg.ssp3_stage_pe_relax, e1, e2);
        stages.push_back(Wi);
        explicit_stages.push_back(cfg.ssp3_explicit_operator == SSP3ExplicitOperator::Residual
            ? integrator_explicit_residual(Wi, dt, dx, e1, e2, cfg)
            : material_rk_residual(Wi, dt, dx, e1, e2, cfg));
        implicit_stages.push_back(integrator_implicit_divergences(
            Wi, AI[i][i] * dt, dx, e1, e2, cfg));
    }
    std::vector<ConsU> out = Un;
    for (int stage = 0; stage < 4; ++stage) {
        for (std::size_t cell = 0; cell < out.size(); ++cell) {
            out[cell].m1 -= dt * B[stage] * explicit_stages[stage].m1[cell];
            out[cell].m2 -= dt * B[stage] * explicit_stages[stage].m2[cell];
            out[cell].mom -= dt * B[stage] * (explicit_stages[stage].mom[cell]
                                               + implicit_stages[stage].grad_p[cell]);
            out[cell].rhoE -= dt * B[stage] * (explicit_stages[stage].rhoE[cell]
                                                + implicit_stages[stage].div_pu[cell]);
            out[cell].a1 -= dt * B[stage] * explicit_stages[stage].alpha[cell];
        }
    }
    out = ssp3_project_target_to_pe(std::move(out), stages.back(), e1, e2, cfg.alpha_pure_tol);
    return ssp3_relax(primitive_cells(out, stages.back(), e1, e2), cfg.ssp3_pe_relax, e1, e2);
}

// time_integrator.py::_imex_ad_ssp3_transport_acoustic_cn.  The explicit
// material map receives the SSP3 composition, then a single frozen-coefficient
// acoustic CN solve is applied.  This is Python's default SSP3 scope; the
// existing full-step composition above remains selectable for experiments.
inline StepResult imex_ad_ssp3_transport_acoustic_cn_step(
        const StepResult& Wn, double dt, double dx, const EOS& e1, const EOS& e2,
        const StepConfig& cfg) {
    // Match Python _imex_ad_ssp3_transport_acoustic_cn: SSP3 material
    // composition is reserved for pressure-flat transport.  A resolved shock
    // uses the single IMEX-AD acoustic step, avoiding invalid SSP3 stages.
    double alpha_min = Wn.alpha.front(), alpha_max = Wn.alpha.front();
    for (double alpha : Wn.alpha) {
        alpha_min = std::fmin(alpha_min, alpha);
        alpha_max = std::fmax(alpha_max, alpha);
    }
    const double pure_tol_auto = std::fmax(cfg.alpha_pure_tol, regime_detail::eps025());
    if (alpha_min >= 1.0 - pure_tol_auto || alpha_max <= pure_tol_auto) {
        return imex_ad_step(Wn.alpha, Wn.T1, Wn.T2, Wn.u, Wn.p, dt, dx, e1, e2, cfg);
    }

    double p_scale = 1.0, p_jump = 0.0;
    for (std::size_t i = 0; i < Wn.p.size(); ++i) {
        p_scale = std::fmax(p_scale, std::fabs(Wn.p[i]));
        if (i + 1 < Wn.p.size()) p_jump = std::fmax(p_jump, std::fabs(Wn.p[i + 1] - Wn.p[i]));
    }
    if (p_jump / p_scale > regime_detail::eps025()) {
        return imex_ad_step(Wn.alpha, Wn.T1, Wn.T2, Wn.u, Wn.p, dt, dx, e1, e2, cfg);
    }
    const auto material_euler = [&](const StepResult& in) {
        const MaterialResult m = material_update(in.alpha,in.T1,in.T2,in.u,in.p,dt,dx,e1,e2,cfg.material_config());
        std::vector<ConsU> target;
        target.reserve(in.alpha.size());
        for (std::size_t i = 0; i < in.alpha.size(); ++i) {
            target.push_back({m.q1_new[i], m.q2_new[i], m.m_adv[i], m.rhoE_new[i], m.alpha_new[i]});
        }
        // Keep a pressure/velocity-flat material contact on its PE manifold.
        // Python's _material_update path performs this before W recovery.
        target = ssp3_project_target_to_pe(std::move(target), in, e1, e2, cfg.alpha_pure_tol);
        StepResult out; const std::size_t n=in.alpha.size();
        out.alpha.resize(n); out.T1.resize(n); out.T2.resize(n); out.u.resize(n); out.p.resize(n);
        for (std::size_t i=0;i<n;++i) {
            const ConsU& U = target[i];
            const PrimW w=cons_to_prim_W(U,e1,e2,1.e-13,50,in.T1[i],in.T2[i]);
            out.alpha[i]=w.alpha1; out.T1[i]=w.T1; out.T2[i]=w.T2; out.u[i]=w.u; out.p[i]=w.p;
        }
        return out;
    };
    const bool passive = cfd::detail::passive_transport(Wn.alpha,Wn.u,Wn.p);
    StepResult Wadv;
    if (passive) {
        Wadv=material_euler(Wn);
    } else {
        const StepResult W1=material_euler(Wn), W2e=material_euler(W1);
        const StepResult W2=blend_conservative(Wn,W2e,.75,e1,e2);
        Wadv=blend_conservative(Wn,material_euler(W2),1.0/3.0,e1,e2);
    }
    const std::size_t n=Wn.alpha.size();
    std::vector<double> q1(n),q2(n),madv(n),rhoE(n),an(n);
    for (std::size_t i=0;i<n;++i) {
        const ConsU U=prim_to_cons_W(PrimW{Wadv.alpha[i],Wadv.T1[i],Wadv.T2[i],Wadv.u[i],Wadv.p[i]},e1,e2);
        q1[i]=U.m1; q2[i]=U.m2; madv[i]=U.mom; rhoE[i]=U.rhoE;
        an[i]=std::fmin(std::fmax(U.a1,1.e-12),1.0-1.e-12);
    }
    PressureClosure closure=select_regime(Wn.alpha,Wn.T1,Wn.T2,Wn.p,e1,e2,cfg.alpha_pure_tol,
                                          cfg.mixture_sound_speed_kind);
    const AcousticSolveResult ac=acoustic_solve(static_cast<int>(n),dx,dt,e1,e2,
        Wn.alpha.data(),Wn.T1.data(),Wn.T2.data(),Wn.u.data(),Wn.p.data(),
        q1.data(),q2.data(),madv.data(),step_detail::to_acoustic_bc(cfg.bc_l,true),step_detail::to_acoustic_bc(cfg.bc_r,false),
        cfg.alpha_pure_tol,.5,1.e-8,cfg.u_inlet_l,cfg.p_inlet_l,cfg.p_outlet_r,
        cfg.acoustic_interface_be,cfg.acoustic_pure_tol_consistent,cfg.acoustic_acid,
        nullptr,false,cfg.acoustic_trbdf2,cfg.acoustic_muscl,cfg.acoustic_stencil_clean,
        cfg.acoustic_waf,static_cast<int>(cfg.acoustic_waf_sigma),static_cast<int>(cfg.acoustic_reconstruction),cfg.acoustic_diss_consistent,cfg.acoustic_interface_centered,
        cfg.mixture_sound_speed_kind);
    std::vector<double> un=ac.u_new, pn=ac.p_new;
    if (closure==PressureClosure::pressure_work_consistent) {
        std::vector<double> Z(n);
        for (std::size_t i=0;i<n;++i) Z[i]=phase_acoustic(e1,e2,Wn.alpha[i],Wn.T1[i],Wn.T2[i],Wn.p[i],cfg.alpha_pure_tol,
                                                           cfg.mixture_sound_speed_kind).Z;
        const auto faces=step_detail::acoustic_faces(un,pn,Z,cfg.bc_l,cfg.bc_r,cfg.u_inlet_l,cfg.p_inlet_l,cfg.p_outlet_r);
        for (std::size_t i=0;i<n;++i) rhoE[i]-=dt*(faces.pf[i+1]*faces.uf[i+1]-faces.pf[i]*faces.uf[i])/dx;
    } else if (closure==PressureClosure::compressive_recovery) {
        auto mask=compressive_pressure_mask(Wn.u,Wn.p);
        const auto pure=pure_material_cell_mask(Wn.alpha,std::fmax(cfg.alpha_pure_tol,regime_detail::eps025()));
        const auto recovered=step_detail::recover_pressure_from_total_energy(q1,q2,rhoE,an,un,pn,e1,e2);
        for (std::size_t i=0;i<n;++i) if (mask[i] && !pure[i]) pn[i]=recovered[i];
    }
    StepResult out; out.alpha=an; out.u=un; out.p=pn; out.T1.resize(n); out.T2.resize(n); out.closure=closure;
    for (std::size_t i=0;i<n;++i) {
        const double r1=q1[i]/std::fmax(an[i],1.e-12), r2=q2[i]/std::fmax(1.0-an[i],1.e-12);
        out.T1[i]=e1.temperature(r1,e1.energy(r1,pn[i])); out.T2[i]=e2.temperature(r2,e2.energy(r2,pn[i]));
    }
    return out;
}

inline RunResult solve_imex_ad(const std::vector<double>& alpha0,
                               const std::vector<double>& T10,
                               const std::vector<double>& T20,
                               const std::vector<double>& u0,
                               const std::vector<double>& p0,
                               double dx, double t_end,
                               const EOS& eos1, const EOS& eos2,
                               const RunConfig& config = {}) {
    RunResult out;
    out.W.alpha = alpha0;
    out.W.T1 = T10;
    out.W.T2 = T20;
    out.W.u = u0;
    out.W.p = p0;

    if (!(dx > 0.0) || !(t_end >= 0.0) || !finite_state(out.W) ||
        config.max_steps < 0) {
        out.termination = RunTermination::invalid_input;
        return out;
    }
    if (try_uniform_periodic_advection_remap(out, dx, t_end, eos1, eos2, config)) return out;

    double t = 0.0;
    int step = 0;
    while (t < t_end && step < config.max_steps) {
        const double cfl_pure_tol = config.step_config.time_integrator == TimeIntegrator::explicit_rusanov &&
                                    config.step_config.pure_branch
            ? config.step_config.alpha_pure_tol : 0.0;
        double dt = config.dt_fixed.has_value()
            ? *config.dt_fixed
            : config.cfl * max_acoustic_dt_imex_ad(out.W, eos1, eos2, dx,
                                                    config.step_config.mixture_sound_speed_kind,
                                                    cfl_pure_tol);
        if (!std::isfinite(dt) || !(dt > 0.0)) {
            out.termination = RunTermination::nonfinite_dt;
            break;
        }
        if (config.dt_min.has_value() && dt < *config.dt_min) {
            out.termination = RunTermination::dt_below_min;
            break;
        }
        if (t + dt > t_end) dt = t_end - t;

        if (config.gravity_well_balanced && config.gravity != 0.0 &&
            config.phase_change.enabled == false && config.heat_conduction.enabled == false &&
            config.step_config.bc_l == BC5::Reflective && config.step_config.bc_r == BC5::Reflective &&
            is_hydrostatic_equilibrium(out.W, dx, config.gravity, eos1, eos2)) {
            t += dt; ++step;
            const StepRecord record{step, t, dt, out.W.closure, 0};
            if (config.record_history) out.history.push_back(record);
            if (config.step_callback && !config.step_callback(record, out.W)) {
                out.termination = RunTermination::step_callback_stop;
                break;
            }
            continue;
        }

        StepConfig step_cfg = config.step_config;
        const double stage_time = t + 0.5 * dt;
        if (config.u_inlet_at) step_cfg.u_inlet_l = config.u_inlet_at(stage_time);
        if (config.p_inlet_at) step_cfg.p_inlet_l = config.p_inlet_at(stage_time);
        if (config.p_outlet_at) step_cfg.p_outlet_r = config.p_outlet_at(stage_time);
        if (config.alpha_inlet_at) step_cfg.alpha_inlet_l = config.alpha_inlet_at(stage_time);
        if (config.T1_inlet_at) step_cfg.T1_inlet_l = config.T1_inlet_at(stage_time);
        if (config.T2_inlet_at) step_cfg.T2_inlet_l = config.T2_inlet_at(stage_time);

        StepResult stage_input = apply_source_terms(out.W, 0.5 * dt, dx, eos1, eos2,
                                                    config.gravity, config.phase_change,
                                                    config.heat_conduction,
                                                    step_cfg.pure_branch ? step_cfg.alpha_pure_tol : 0.0);
        StepResult next;
        if (step_cfg.time_integrator == TimeIntegrator::explicit_rusanov) {
            next = explicit_rusanov_step(stage_input.alpha, stage_input.T1, stage_input.T2,
                                         stage_input.u, stage_input.p, dt, dx,
                                         eos1, eos2, step_cfg);
        } else if (step_cfg.time_integrator == TimeIntegrator::ars222) {
            next = ars222_step(stage_input, dt, dx, eos1, eos2, step_cfg);
        } else if (step_cfg.time_integrator == TimeIntegrator::imex_ssp3_transport_acoustic_cn) {
            next = imex_ad_ssp3_transport_acoustic_cn_step(stage_input, dt, dx, eos1, eos2, step_cfg);
        } else if (step_cfg.time_integrator == TimeIntegrator::imex_ad_ssp2) {
            next = imex_ad_ssp2_step(stage_input, dt, dx, eos1, eos2, step_cfg);
        } else if (step_cfg.time_integrator == TimeIntegrator::imex_ssp3) {
            if (step_cfg.imex_ssp3_form == IMEXSSP3Form::stage_residual) {
                next = imex_ssp3_stage_residual_step(stage_input, dt, dx, eos1, eos2, step_cfg);
            } else if (step_cfg.imex_ssp3_form == IMEXSSP3Form::shu_osher_full_step) {
                next = imex_ad_ssp3_full_step(stage_input, dt, dx, eos1, eos2, step_cfg);
            } else {
                next = imex_ad_ssp3_transport_acoustic_cn_step(stage_input, dt, dx, eos1, eos2, step_cfg);
            }
        } else if (step_cfg.time_integrator == TimeIntegrator::strang) {
            next = strang_step(stage_input, dt, dx, eos1, eos2, step_cfg);
        } else if (step_cfg.time_integrator == TimeIntegrator::split) {
            next = split_step(stage_input, dt, dx, eos1, eos2, step_cfg);
        } else if (step_cfg.time_integrator == TimeIntegrator::be1) {
            next = be1_step(stage_input, dt, dx, eos1, eos2, step_cfg);
        } else if (step_cfg.time_integrator == TimeIntegrator::be_full) {
            next = be_full_step(stage_input, dt, dx, eos1, eos2, step_cfg);
        } else {
            next = imex_ad_step(stage_input.alpha, stage_input.T1, stage_input.T2, stage_input.u, stage_input.p,
                                dt, dx, eos1, eos2, step_cfg);
        }
        next = apply_source_terms(next, 0.5 * dt, dx, eos1, eos2,
                                  config.gravity, config.phase_change,
                                  config.heat_conduction,
                                  step_cfg.pure_branch ? step_cfg.alpha_pure_tol : 0.0);
        t += dt;
        ++step;
        out.W = std::move(next);

        const StepRecord record{step, t, dt, out.W.closure, out.W.vacuum_velocity_cells};
        if (config.record_history) out.history.push_back(record);
        if (config.stop_on_nonfinite && !finite_state(out.W)) {
            out.termination = RunTermination::nonfinite_state;
            break;
        }
        if (config.step_callback && !config.step_callback(record, out.W)) {
            out.termination = RunTermination::step_callback_stop;
            break;
        }
    }

    out.t_final = t;
    out.steps = step;
    if (out.termination == RunTermination::completed && t < t_end &&
        step >= config.max_steps) {
        out.termination = RunTermination::max_steps;
    }
    return out;
}

// Public compatibility spelling for Python main.py::solve.  It intentionally
// starts from the Python defaults; existing callers of solve_imex_ad retain the
// C++ production IMEX defaults above.
inline RunResult solve(const std::vector<double>& alpha0,
                       const std::vector<double>& T10,
                       const std::vector<double>& T20,
                       const std::vector<double>& u0,
                       const std::vector<double>& p0,
                       double dx, double t_end,
                       const EOS& eos1, const EOS& eos2,
                       const RunConfig& config = RunConfig::python_solve_defaults()) {
    return solve_imex_ad(alpha0, T10, T20, u0, p0, dx, t_end, eos1, eos2, config);
}

} // namespace five_eq
} // namespace cfd
