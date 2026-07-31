// leveque_compare.cpp — LeVeque-style solid-body rotation on the criss-cross
// mesh. After one full revolution the exact solution equals the initial field,
// so global_E1 = mean|u_final - u0| is a clean accuracy gate. Compares the
// baseline mlp_u1 (BJ-vertex, idw_p=0) against T-MLP-u-L (idw_p=2). Args: [N] [revs].
#include "cfd/solver2d.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace cfd;

static double run(const Mesh& m, const ReconCtx& ctx, const std::vector<double>& U0,
                  double t_end, int recon, double& umin, double& umax) {
    Advection2D eq;
    eq.velocity = [](double x, double y, double& ax, double& ay) {
        ax = 2.0 * M_PI * (0.5 - y); ay = 2.0 * M_PI * (x - 0.5);
    };
    Solve2DResult r = solve_adv2d(m, eq, U0, t_end, /*SSP-RK3*/2, /*cfl*/0.3,
                                  /*dt_fixed*/-1.0, 10000000, recon, &ctx);
    const int N = m.n_cells();
    double e1 = 0.0; umin = 1e300; umax = -1e300;
    for (int i = 0; i < N; ++i) {
        e1 += std::fabs(r.U[i] - U0[i]) * m.cell_volumes[i];
        umin = std::min(umin, r.U[i]); umax = std::max(umax, r.U[i]);
    }
    std::printf("  (n_steps=%d)\n", r.n_steps);
    return e1; // domain area = 1 so this is the area-weighted L1 error
}

int main(int argc, char** argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : 64;
    double revs = argc > 2 ? std::atof(argv[2]) : 1.0;
    Mesh m = criss_cross_box(N, 1.0);
    const int NC = m.n_cells();

    // smooth cosine-bell hump centred at (0.75, 0.5), radius 0.15.
    std::vector<double> U0(NC);
    for (int i = 0; i < NC; ++i) {
        double cx = m.cell_centers[i*2+0], cy = m.cell_centers[i*2+1];
        double r = std::sqrt((cx-0.75)*(cx-0.75) + (cy-0.5)*(cy-0.5)) / 0.15;
        U0[i] = r < 1.0 ? 0.25 * (1.0 + std::cos(M_PI * r)) : 0.0;
    }

    ReconCtx ctx_bj  = build_recon_ctx(m, 0.0);  // mlp_u1
    ReconCtx ctx_idw = build_recon_ctx(m, 2.0);  // T-MLP-u-L (IDW)

    double umn, umx;
    std::printf("LeVeque rotation: N=%d cells=%d revs=%.2f\n", N, NC, revs);
    std::printf("mlp_u1 (BJ-vertex):\n");
    double e_bj  = run(m, ctx_bj,  U0, revs, RECON_BJ_VERTEX, umn, umx);
    std::printf("    global_E1=%.6e  u in [%.4f, %.4f]\n", e_bj, umn, umx);
    std::printf("T-MLP-u-L (IDW p=2):\n");
    double e_idw = run(m, ctx_idw, U0, revs, RECON_BJ_VERTEX, umn, umx);
    std::printf("    global_E1=%.6e  u in [%.4f, %.4f]\n", e_idw, umn, umx);
    std::printf("ratio E1(T-MLP-u-L)/E1(mlp_u1) = %.4f  (%s)\n",
                e_idw / e_bj, e_idw < e_bj ? "T-MLP-u-L better" : "mlp_u1 better");
    return 0;
}
