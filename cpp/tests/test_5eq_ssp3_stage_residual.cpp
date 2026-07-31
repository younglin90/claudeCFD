#include "cfd/five_eq/solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

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
    const auto got = imex_ssp3_stage_residual_step(W, 1.e-8, .1, air, water, cfg);
    RunConfig run_cfg;
    run_cfg.step_config = cfg;
    run_cfg.step_config.time_integrator = TimeIntegrator::imex_ssp3;
    run_cfg.step_config.imex_ssp3_form = IMEXSSP3Form::stage_residual;
    run_cfg.dt_fixed = 1.e-8;
    const auto dispatched = solve_imex_ad(W.alpha, W.T1, W.T2, W.u, W.p,
                                          .1, 1.e-8, air, water, run_cfg);
    for (std::size_t i = 0; i < W.alpha.size(); ++i) {
        const double delta = std::max({std::fabs(dispatched.W.alpha[i] - got.alpha[i]),
                                       std::fabs(dispatched.W.T1[i] - got.T1[i]),
                                       std::fabs(dispatched.W.T2[i] - got.T2[i]),
                                       std::fabs(dispatched.W.u[i] - got.u[i]),
                                       std::fabs(dispatched.W.p[i] - got.p[i])});
        if (delta > 1.e-12) return 3;
    }
    std::ifstream in(SSP3_STAGE_REF);
    if (!in) return 1;
    std::string line;
    int fail = 0, rows = 0;
    double worst = 0.;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        int i;
        double initial[5], ref[5];
        s >> i;
        for (double& x : initial) s >> x;
        for (double& x : ref) s >> x;
        const double value[] = {got.alpha[i], got.T1[i], got.T2[i], got.u[i], got.p[i]};
        for (int k = 0; k < 5; ++k) {
            const double rel = std::fabs(value[k] - ref[k]) / std::fmax(std::fabs(ref[k]), 1.);
            worst = std::fmax(worst, rel);
            if (rel > 3.e-5) {
                std::printf("  cell=%d field=%d got=%.17g ref=%.17g rel=%.3e\n",
                            i, k, value[k], ref[k], rel);
                ++fail;
            }
        }
        ++rows;
    }
    std::printf("ssp3 stage-residual: rows=%d max_rel=%.3e %s\n", rows, worst,
                fail ? "FAIL" : "PASS");
    return fail ? 2 : 0;
}
