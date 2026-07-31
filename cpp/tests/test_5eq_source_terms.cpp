#include "cfd/five_eq/source_terms.hpp"

#include <cmath>
#include <cstdio>

int main() {
    const cfd::EOS air = cfd::EOS::ideal(1.4, 717.5);
    const cfd::EOS water = cfd::EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    cfd::five_eq::StepResult W{{0.7}, {300.0}, {300.0}, {2.0}, {1.0e5}};
    const auto out = cfd::five_eq::apply_gravity_source(W, 0.25, -9.81, air, water);
    const double expected_u = 2.0 - 9.81 * 0.25;
    if (std::fabs(out.u[0] - expected_u) > 1.0e-10 ||
        std::fabs(out.alpha[0] - W.alpha[0]) > 1.0e-12 ||
        std::fabs(out.p[0] - W.p[0]) / W.p[0] > 1.0e-9) {
        std::fprintf(stderr, "gravity source mismatch u=%.17g p=%.17g\n", out.u[0], out.p[0]);
        return 1;
    }
    cfd::five_eq::PhaseChangeConfig pc;
    pc.enabled = true; pc.isothermal = true; pc.p_sat = 1.0e5; pc.tau = 1.0e-3;
    cfd::five_eq::StepResult vap{{0.8}, {380.0}, {380.0}, {0.0}, {9.0e4}};
    const auto changed = cfd::five_eq::apply_phase_change_source(vap, 1.0e-5, air, water, pc);
    if (!(changed.alpha[0] < vap.alpha[0]) || std::fabs(changed.T1[0] - vap.T1[0]) > 1.0e-12) {
        std::fprintf(stderr, "phase change direction mismatch alpha=%.17g\n", changed.alpha[0]);
        return 1;
    }
    cfd::five_eq::HeatConductionConfig heat;
    heat.enabled = true; heat.primitive_temperature = true; heat.k_liquid = heat.k_vapor = 1.0;
    cfd::five_eq::StepResult thermal{{0.5, 0.5, 0.5}, {300.0, 400.0, 500.0},
                                      {300.0, 400.0, 500.0}, {0.0, 0.0, 0.0}, {1e5, 1e5, 1e5}};
    const auto heated = cfd::five_eq::apply_heat_conduction_source(thermal, 0.1, 1.0, air, water, heat);
    if (!(heated.T1[1] > 399.0 && heated.T1[1] < 401.0)) {
        std::fprintf(stderr, "heat source mismatch T=%.17g\n", heated.T1[1]);
        return 1;
    }
    constexpr double dx = 0.01, g = -9.81;
    cfd::five_eq::StepResult hydro;
    for (int i = 0; i < 4; ++i) {
        const double x = (i + 0.5) * dx;
        hydro.alpha.push_back(1.0 - 1.0e-8); hydro.T1.push_back(300.0); hydro.T2.push_back(300.0);
        hydro.u.push_back(0.0); hydro.p.push_back(1.0e5 * std::exp(g * x / ((1.4 - 1.0) * 717.5 * 300.0)));
    }
    if (!cfd::five_eq::is_hydrostatic_equilibrium(hydro, dx, g, air, water)) {
        std::fprintf(stderr, "hydrostatic sensor mismatch\n"); return 1;
    }
    std::printf("gravity source passed\n");
    return 0;
}
