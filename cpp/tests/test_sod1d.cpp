// test_sod1d.cpp — 1D Euler Sod shock tube, C++ solver vs Python reference.
// Two reconstructions (first-order, minmod-MUSCL), same config (LLF,
// forward-Euler, dt_fixed=5e-4, t=0.2, transmissive). Bit-comparable to
// solver/solve_T-MLP-u.
#include "cfd/solver1d.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>

using namespace cfd;
#ifndef SOD_REF
#  define SOD_REF "sod1d_ref.txt"
#endif
#ifndef SOD_REF_MINMOD
#  define SOD_REF_MINMOD "sod1d_ref_minmod.txt"
#endif

static int compare_case(const char* tag, int recon, const char* ref_path) {
    const int N = 200;
    Mesh m = build_structured_1d(N, 1.0, false);
    Euler1D eq{1.4};
    std::vector<double> U0(3 * N);
    for (int i = 0; i < N; ++i) {
        double x = m.cell_centers[i];
        double W[3] = {x < 0.5 ? 1.0 : 0.125, 0.0, x < 0.5 ? 1.0 : 0.1}, U[3];
        eq.prim_to_cons(W, U);
        U0[0 * N + i] = U[0]; U0[1 * N + i] = U[1]; U0[2 * N + i] = U[2];
    }
    Solve1DResult r = solve_euler1d(m, eq, U0, 0.2, /*integrator FE*/0,
                                    0.4, /*dt_fixed*/5e-4, 200000, recon);

    std::ifstream fin(ref_path);
    if (!fin) { std::printf("  [%s] cannot open ref %s\n", tag, ref_path); return 1; }
    int Nref, nsteps_ref; fin >> Nref >> nsteps_ref;
    int fail = 0; double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref[3]; fin >> ref[0] >> ref[1] >> ref[2];
        double Uc[3] = {r.U[0 * N + i], r.U[1 * N + i], r.U[2 * N + i]}, Wc[3];
        eq.cons_to_prim(Uc, Wc);
        for (int v = 0; v < 3; ++v) {
            double d = std::fabs(ref[v]) > 1e-12 ? std::fabs(ref[v]) : 1.0;
            double rel = std::fabs(Wc[v] - ref[v]) / d;
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-12) { if (fail < 4)
                std::printf("  [%s FAIL] cell %d var %d got=%.17g ref=%.17g rel=%.3e\n",
                            tag, i, v, Wc[v], ref[v], rel); ++fail; }
        }
    }
    if (r.n_steps != nsteps_ref) { std::printf("  [%s FAIL] n_steps %d vs %d\n",
                                               tag, r.n_steps, nsteps_ref); ++fail; }
    std::printf("  [%s] n_steps=%d max_rel=%.3e %s\n", tag, r.n_steps, max_rel,
                fail == 0 ? "PASS" : "FAIL");
    return fail;
}

int main() {
    int fail = 0;
    fail += compare_case("first_order", 0, SOD_REF);
    fail += compare_case("minmod", 1, SOD_REF_MINMOD);
    if (fail == 0) { std::printf("test_sod1d: ALL PASS (both reconstructions match Python)\n"); return 0; }
    std::printf("test_sod1d: %d FAILURES\n", fail);
    return 1;
}
