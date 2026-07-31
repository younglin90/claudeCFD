// test_euler2d.cpp — 2D Euler (Gaussian pressure pulse) on a triangulated box,
// C++ vs Python. LLF + SSP-RK3, first-order and BJ-vertex (mlp_u1) recon.
#include "cfd/solver_euler2d.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
using namespace cfd;
#ifndef EUL2D_FO_REF
#  define EUL2D_FO_REF "eul2d_fo_ref.txt"
#endif
#ifndef EUL2D_MLPU1_REF
#  define EUL2D_MLPU1_REF "eul2d_mlpu1_ref.txt"
#endif
#ifndef EUL2D_HLLC_REF
#  define EUL2D_HLLC_REF "eul2d_hllc_ref.txt"
#endif
#ifndef EUL2D_REFL_REF
#  define EUL2D_REFL_REF "eul2d_refl_ref.txt"
#endif

static int run_case(const char* tag, const Mesh& m, const Euler2D& eq,
                    const std::vector<double>& U0, int recon, const ReconCtx* ctx,
                    const char* ref, int flux = FLUX_LLF,
                    const std::vector<BC2D>* bcs = nullptr, double t_end = 5e-3) {
    Solve2DResult r = solve_euler2d(m, eq, U0, t_end, 2, 0.4, 1e-3, 1000000, recon, ctx, flux, bcs);
    std::ifstream fin(ref);
    if (!fin) { std::printf("  [%s] cannot open %s\n", tag, ref); return 1; }
    int N, ns; fin >> N >> ns;
    int fail = 0; double mr = 0;
    for (int i = 0; i < N; ++i) {
        double ref4[4]; fin >> ref4[0] >> ref4[1] >> ref4[2] >> ref4[3];
        for (int v = 0; v < 4; ++v) {
            double got = r.U[(size_t)v*N+i];
            double aerr = std::fabs(got - ref4[v]);
            double rel = aerr / (std::fabs(ref4[v]) > 1e-300 ? std::fabs(ref4[v]) : 1.0);
            mr = std::max(mr, rel);
            // combined: machine-precision abs OR tight rel (near-zero momenta have
            // huge rel but ~1e-17 abs from FP summation-order differences).
            if (aerr > 1e-11 && rel > 1e-11) { if (fail < 6)
                std::printf("  [%s FAIL] cell %d var %d got=%.17g ref=%.17g abs=%.3e rel=%.3e\n",
                            tag, i, v, got, ref4[v], aerr, rel); ++fail; }
        }
    }
    if (r.n_steps != ns) { std::printf("  [%s FAIL] n_steps %d vs %d\n", tag, r.n_steps, ns); ++fail; }
    std::printf("  [%s] n_steps=%d max_rel=%.3e %s\n", tag, r.n_steps, mr, fail?"FAIL":"PASS");
    return fail;
}

int main() {
    Mesh m = triangulate_box(8, 8, 1.0, 1.0);
    ReconCtx ctx = build_recon_ctx(m);
    Euler2D eq{1.4};
    const int N = m.n_cells();
    std::vector<double> U0(4 * N);
    for (int i = 0; i < N; ++i) {
        double cx = m.cell_centers[i*2+0], cy = m.cell_centers[i*2+1];
        double r2 = (cx-0.5)*(cx-0.5) + (cy-0.5)*(cy-0.5);
        double W[4] = {1.0, 0.0, 0.0, 1.0 + 0.2*std::exp(-50.0*r2)}, U[4];
        eq.prim_to_cons(W, U);
        for (int v = 0; v < 4; ++v) U0[(size_t)v*N+i] = U[v];
    }
    int fail = 0;
    fail += run_case("first_order", m, eq, U0, RECON_FIRST, nullptr, EUL2D_FO_REF);
    fail += run_case("mlp_u1", m, eq, U0, RECON_BJ_VERTEX, &ctx, EUL2D_MLPU1_REF);
    fail += run_case("hllc", m, eq, U0, RECON_BJ_VERTEX, &ctx, EUL2D_HLLC_REF, FLUX_HLLC);

    // reflective-wall case (stronger pulse, t=1e-2) — patch tags 1..4 all slip walls.
    std::vector<BC2D> bcs(5);
    for (int k = 1; k <= 4; ++k) bcs[k].kind = 1;
    std::vector<double> U0r(4 * N);
    for (int i = 0; i < N; ++i) {
        double cx = m.cell_centers[i*2+0], cy = m.cell_centers[i*2+1];
        double r2 = (cx-0.5)*(cx-0.5) + (cy-0.5)*(cy-0.5);
        double W[4] = {1.0, 0.0, 0.0, 1.0 + 0.5*std::exp(-80.0*r2)}, U[4];
        eq.prim_to_cons(W, U);
        for (int v = 0; v < 4; ++v) U0r[(size_t)v*N+i] = U[v];
    }
    fail += run_case("reflective", m, eq, U0r, RECON_FIRST, nullptr, EUL2D_REFL_REF,
                     FLUX_LLF, &bcs, 1e-2);
    if (fail == 0) { std::printf("test_euler2d: ALL PASS (2D Euler matches Python)\n"); return 0; }
    std::printf("test_euler2d: %d FAILURES\n", fail); return 1;
}
