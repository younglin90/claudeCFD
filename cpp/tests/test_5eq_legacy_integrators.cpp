#include "cfd/five_eq/solver.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    using namespace cfd;
    using namespace cfd::five_eq;
    const EOS air = EOS::ideal(1.4, 717.5);
    const EOS water = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    const double T1 = air.temperature(1.157, air.energy(1.157, 1.e5));
    const double T2 = water.temperature(998., water.energy(998., 1.e5));
    const StepResult W{{.35,.50,.65,.45}, {T1,T1,T1,T1}, {T2,T2,T2,T2},
                       {.03,-.01,.02,0.}, {1.e5,100100.,99940.,100030.}};
    StepConfig cfg;
    cfg.bc_l = BC5::Periodic; cfg.bc_r = BC5::Periodic;
    cfg.kapila_closure = false; cfg.ars_implicit_dissipation = 1.;
    cfg.be1_pe_project_explicit = false;
    const auto strang = strang_step(W, 1.e-8, .1, air, water, cfg);
    NewtonInfo split_info, be1_info, full_info;
    const auto split = split_step(W, 1.e-8, .1, air, water, cfg, &split_info);
    const auto be1 = be1_step(W, 1.e-8, .1, air, water, cfg, &be1_info);
    const auto full = be_full_step(W, 1.e-8, .1, air, water, cfg, &full_info);
    StepConfig no_line_search_cfg = cfg;
    no_line_search_cfg.newton_line_search_max = 0;
    NewtonInfo no_line_search_info;
    (void)be_full_step(W, 1.e-8, .1, air, water, no_line_search_cfg, &no_line_search_info);
    const StepResult* got[] = {&strang, &split, &be1, &full};
    std::ifstream in(LEGACY_INTEGRATORS_REF);
    if (!in) return 1;
    std::string line;
    int fail = 0, rows = 0;
    double worst = 0.;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        int kind, i;
        double initial[5], ref[5];
        s >> kind >> i;
        for (double& x : initial) s >> x;
        for (double& x : ref) s >> x;
        const auto& state = *got[kind];
        const double value[] = {state.alpha[i], state.T1[i], state.T2[i], state.u[i], state.p[i]};
        for (int k = 0; k < 5; ++k) {
            const double rel = std::fabs(value[k] - ref[k]) / std::fmax(std::fabs(ref[k]), 1.);
            worst = std::fmax(worst, rel);
            if (rel > 2.e-5) {
                ++fail;
                std::printf("kind=%d cell=%d field=%d rel=%.3e\n", kind, i, k, rel);
            }
        }
        ++rows;
    }
    if (!split_info.converged || !be1_info.converged || !full_info.converged) {
        ++fail;
        std::printf("Newton convergence split=%d be1=%d full=%d\n",
                    split_info.converged, be1_info.converged, full_info.converged);
    }
    if (no_line_search_info.converged) {
        ++fail;
        std::printf("Newton line_search_max=0 was ignored\n");
    }
    RunConfig dt_guard;
    dt_guard.dt_fixed = 1.e-8;
    dt_guard.dt_min = 2.e-8;
    const auto stopped_by_dt = solve_imex_ad(W.alpha, W.T1, W.T2, W.u, W.p,
                                             .1, 1.e-8, air, water, dt_guard);
    if (stopped_by_dt.termination != RunTermination::dt_below_min || stopped_by_dt.steps != 0) ++fail;
    int boundary_calls = 0;
    RunConfig callback_guard;
    callback_guard.dt_fixed = 1.e-8;
    callback_guard.step_config.bc_l = BC5::Periodic;
    callback_guard.step_config.bc_r = BC5::Periodic;
    callback_guard.u_inlet_at = [&](double) { ++boundary_calls; return .125; };
    callback_guard.step_callback = [](const StepRecord&, const StepResult&) { return false; };
    const auto stopped_by_callback = solve_imex_ad(W.alpha, W.T1, W.T2, W.u, W.p,
                                                   .1, 2.e-8, air, water, callback_guard);
    if (stopped_by_callback.termination != RunTermination::step_callback_stop ||
        stopped_by_callback.steps != 1 || boundary_calls != 1) ++fail;
    RunConfig remap_cfg;
    remap_cfg.uniform_periodic_remap = true;
    remap_cfg.dt_fixed = .05;
    remap_cfg.record_history = true;
    remap_cfg.step_config.bc_l = BC5::Periodic;
    remap_cfg.step_config.bc_r = BC5::Periodic;
    const StepResult transport{{.2,.4,.6,.8}, {T1,T1,T1,T1}, {T2,T2,T2,T2},
                               {1.,1.,1.,1.}, {1.e5,1.e5,1.e5,1.e5}};
    const auto remapped = solve_imex_ad(transport.alpha, transport.T1, transport.T2,
                                        transport.u, transport.p, .1, .05,
                                        air, water, remap_cfg);
    const double expected_alpha[] = {.5,.3,.5,.7};
    if (remapped.termination != RunTermination::completed || remapped.steps != 1 ||
        remapped.history.size() != 1) ++fail;
    for (int i = 0; i < 4; ++i)
        if (std::fabs(remapped.W.alpha[i] - expected_alpha[i]) > 1.e-13 ||
            std::fabs(remapped.W.u[i] - 1.) > 1.e-13 ||
            std::fabs(remapped.W.p[i] - 1.e5) > 1.e-8) ++fail;
    RunConfig remap_source_cfg = remap_cfg;
    remap_source_cfg.gravity = 2.0;
    const auto sourced = solve_imex_ad(transport.alpha, transport.T1, transport.T2,
                                       transport.u, transport.p, .1, .05,
                                       air, water, remap_source_cfg);
    if (sourced.steps != 1 || std::fabs(sourced.W.u[0] - 1.) < 1.e-6) ++fail;
    std::printf("legacy integrators: rows=%d max_rel=%.3e %s\n", rows, worst,
                fail ? "FAIL" : "PASS");
    return fail ? 2 : 0;
}
