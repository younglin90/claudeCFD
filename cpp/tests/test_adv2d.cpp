// test_adv2d.cpp — 2D scalar advection (constant velocity) on the criss-cross
// mesh, C++ solver vs Python. first-order + upwind + SSP-RK3, dt_fixed.
#include "cfd/solver2d.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
using namespace cfd;
#ifndef ADV2D_REF
#  define ADV2D_REF "adv2d_ref.txt"
#endif

int main() {
    Mesh m = criss_cross_box(4, 1.0);
    Advection2D eq;
    eq.velocity = [](double, double, double& ax, double& ay) { ax = 1.0; ay = 0.5; };

    const int N = m.n_cells();
    std::vector<double> U0(N);
    for (int i = 0; i < N; ++i) {
        double cx = m.cell_centers[i*2+0], cy = m.cell_centers[i*2+1];
        U0[i] = std::cos(2*M_PI*cx) * std::cos(2*M_PI*cy);
    }
    Solve2DResult r = solve_adv2d(m, eq, U0, 5e-3, /*SSP-RK3*/2, 0.4, /*dt_fixed*/1e-3);

    std::ifstream fin(ADV2D_REF);
    if (!fin) { std::printf("test_adv2d: cannot open %s\n", ADV2D_REF); return 2; }
    int Nref, nsteps; fin >> Nref >> nsteps;
    int fail = 0; double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref; fin >> ref;
        double d = std::fabs(ref) > 1e-12 ? std::fabs(ref) : 1.0;
        double rel = std::fabs(r.U[i] - ref) / d;
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-12) { if (fail < 6)
            std::printf("  [FAIL] cell %d got=%.17g ref=%.17g rel=%.3e\n", i, r.U[i], ref, rel);
            ++fail; }
    }
    if (r.n_steps != nsteps) { std::printf("  [FAIL] n_steps %d vs %d\n", r.n_steps, nsteps); ++fail; }
    std::printf("test_adv2d: n_steps=%d max_rel=%.3e\n", r.n_steps, max_rel);
    if (fail == 0) { std::printf("test_adv2d: ALL PASS (64 cells match Python)\n"); return 0; }
    std::printf("test_adv2d: %d FAILURES\n", fail); return 1;
}
