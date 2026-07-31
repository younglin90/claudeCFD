// bench_euler2d.cpp — wall-time benchmark of the 2D Euler solver (mlp_u1 recon,
// LLF, SSP-RK3) for Python-vs-C++ speed comparison. Args: [Nx] [steps].
#include "cfd/solver_euler2d.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
using namespace cfd;

int main(int argc, char** argv) {
    int Nx = argc > 1 ? std::atoi(argv[1]) : 48;
    int steps = argc > 2 ? std::atoi(argv[2]) : 50;
    Mesh m = triangulate_box(Nx, Nx, 1.0, 1.0);
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
    double dt = 2e-4, t_end = dt * steps;
    auto t0 = std::chrono::high_resolution_clock::now();
    Solve2DResult r = solve_euler2d(m, eq, U0, t_end, 2, 0.4, dt, 1000000,
                                    RECON_BJ_VERTEX, &ctx);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::printf("cpp bench: cells=%d steps=%d wall=%.4f s  (%.3e s/step, rhoE[mid]=%.10g)\n",
                N, r.n_steps, sec, sec / r.n_steps, r.U[(size_t)3*N + N/2]);
    return 0;
}
