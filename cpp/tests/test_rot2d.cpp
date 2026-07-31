// test_rot2d.cpp — integrated 2D high-order solver: rotation advection on the
// criss-cross mesh with T-MLP-u-L (BJ-vertex) reconstruction + upwind + SSP-RK3,
// vs Python (reconstruction='mlp_u1'). Validates recon+flux+integrator together.
#include "cfd/solver2d.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
using namespace cfd;
#ifndef ROT_REF
#  define ROT_REF "rot_mlpu1_ref.txt"
#endif

int main() {
    Mesh m = criss_cross_box(8, 1.0);
    ReconCtx ctx = build_recon_ctx(m);
    Advection2D eq;
    eq.velocity = [](double x, double y, double& ax, double& ay) { ax = 0.5 - y; ay = x - 0.5; };

    const int N = m.n_cells();
    std::vector<double> U0(N);
    for (int i = 0; i < N; ++i) {
        double cx = m.cell_centers[i*2+0], cy = m.cell_centers[i*2+1];
        U0[i] = std::cos(2*M_PI*cx) * std::cos(2*M_PI*cy);
    }
    Solve2DResult r = solve_adv2d(m, eq, U0, 1e-2, /*SSP-RK3*/2, 0.4, /*dt_fixed*/2e-3,
                                  1000000, RECON_BJ_VERTEX, &ctx);

    std::ifstream fin(ROT_REF);
    if (!fin) { std::printf("test_rot2d: cannot open %s\n", ROT_REF); return 2; }
    int Nref, ns; fin >> Nref >> ns;
    int fail = 0; double mr = 0;
    for (int i = 0; i < N; ++i) {
        double ref; fin >> ref;
        double d = std::fabs(ref) > 1e-12 ? std::fabs(ref) : 1.0;
        double rel = std::fabs(r.U[i]-ref)/d; mr = std::max(mr, rel);
        if (rel > 1e-12) { if (fail < 6)
            std::printf("  [FAIL] cell %d got=%.17g ref=%.17g rel=%.3e\n", i, r.U[i], ref, rel);
            ++fail; }
    }
    if (r.n_steps != ns) { std::printf("  [FAIL] n_steps %d vs %d\n", r.n_steps, ns); ++fail; }
    std::printf("test_rot2d: n_steps=%d max_rel=%.3e\n", r.n_steps, mr);
    if (fail == 0) { std::printf("test_rot2d: ALL PASS (256 cells, high-order solver matches Python)\n"); return 0; }
    std::printf("test_rot2d: %d FAILURES\n", fail); return 1;
}
