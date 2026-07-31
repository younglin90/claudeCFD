// leveque_bench.cpp — canonical LeVeque-Zalesak solid-body rotation test
// (slotted cylinder + cone + cosine hump, rigid rotation about (.5,.5), period 1).
// After one revolution exact = initial, so per-body L1 + shape diagnostics gauge
// accuracy. Compares baseline mlp_u1 (idw_p=0) vs T-MLP-u-L (idw_p=2). Args:[N].
#include "cfd/solver2d.hpp"
#include "cfd/io_vtk.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

using namespace cfd;

static double phi0(double x, double y) {
    const double r0 = 0.15;
    double r1 = std::sqrt((x-0.5)*(x-0.5) + (y-0.75)*(y-0.75)) / r0;
    bool in_slot = (std::fabs(x-0.5) < 0.025) && (y < 0.85);
    double slot = (r1 <= 1.0 && !in_slot) ? 1.0 : 0.0;
    double r2 = std::sqrt((x-0.5)*(x-0.5) + (y-0.25)*(y-0.25)) / r0;
    double cone = (r2 <= 1.0) ? 1.0 - r2 : 0.0;
    double r3 = std::sqrt((x-0.25)*(x-0.25) + (y-0.5)*(y-0.5)) / r0;
    double hump = (r3 <= 1.0) ? 0.25*(1.0 + std::cos(M_PI*r3)) : 0.0;
    return slot + cone + hump;
}

struct Metrics { double L1_total, L1_cone, L1_hump, L1_slot, cone_peak, hump_peak, slot_max, gmin, gmax; };

static Metrics run(const Mesh& m, const ReconCtx& ctx, const std::vector<double>& U0,
                   const char* dumpfile = nullptr, int recon = RECON_BJ_VERTEX,
                   const ReconCtxO2* ctx_o2 = nullptr, const char* label = nullptr) {
    Advection2D eq;
    eq.velocity = [](double x, double y, double& ax, double& ay) {
        ax = -2.0*M_PI*(y-0.5); ay = 2.0*M_PI*(x-0.5);
    };
    auto t0_ = std::chrono::steady_clock::now();
    Solve2DResult r = solve_adv2d(m, eq, U0, 1.0, 2, 0.3, -1.0, 10000000, recon, &ctx, ctx_o2);
    double wall_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    std::printf("[WALL] %s wall=%.3fs steps=%d\n", label ? label : "run", wall_, r.n_steps);
    std::fflush(stdout);
    const int N = m.n_cells();
    if (dumpfile) { FILE* fh=std::fopen(dumpfile,"w");
        for(int i=0;i<N;++i) std::fprintf(fh,"%.6g %.6g %.6g\n",
            m.cell_centers[i*2],m.cell_centers[i*2+1],r.U[i]); std::fclose(fh);
        cfd::write_vtk_unstructured_2d(std::string(dumpfile)+".vtk", m, {{"g", r.U.data()}}); }
    Metrics M{0,0,0,0,-1e9,-1e9,0,1e9,-1e9};
    double r0 = 0.15;
    for (int i = 0; i < N; ++i) {
        double cx = m.cell_centers[i*2+0], cy = m.cell_centers[i*2+1];
        double V = m.cell_volumes[i], u = r.U[i], e = std::fabs(u - U0[i]) * V;
        M.L1_total += e;
        double r2 = std::sqrt((cx-0.5)*(cx-0.5)+(cy-0.25)*(cy-0.25))/r0;
        double r3 = std::sqrt((cx-0.25)*(cx-0.25)+(cy-0.5)*(cy-0.5))/r0;
        double r1 = std::sqrt((cx-0.5)*(cx-0.5)+(cy-0.75)*(cy-0.75))/r0;
        if (r2 <= 1.0) { M.L1_cone += e; M.cone_peak = std::max(M.cone_peak, u); }
        if (r3 <= 1.0) { M.L1_hump += e; M.hump_peak = std::max(M.hump_peak, u); }
        if (r1 <= 1.0) M.L1_slot += e;
        // slot-fill: max u inside the (returned) slot notch (should stay ~0)
        if (std::fabs(cx-0.5) < 0.025 && cy < 0.85 && r1 <= 1.0) M.slot_max = std::max(M.slot_max, u);
        M.gmin = std::min(M.gmin, u); M.gmax = std::max(M.gmax, u);
    }
    std::printf("    (n_steps=%d)\n", r.n_steps);
    return M;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : 100;
    Mesh m = criss_cross_box(N, 1.0);
    const int NC = m.n_cells();
    std::vector<double> U0(NC);
    for (int i = 0; i < NC; ++i) U0[i] = phi0(m.cell_centers[i*2+0], m.cell_centers[i*2+1]);
    ReconCtx bj  = build_recon_ctx(m, 0.0);
    ReconCtx idw = build_recon_ctx(m, 2.0);
    std::printf("LeVeque-Zalesak rotation: N=%d cells=%d, 1 revolution\n", N, NC);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    const bool mlponly = std::getenv("LEV_MLPONLY") != nullptr;  // skip the slow T-MLP-u-L + BVD lines
    auto t_all_ = std::chrono::steady_clock::now();
    std::printf("  mlp_u1 (BJ-vertex):\n");   Metrics a = run(m, bj,  U0, "lev_mlpu1.txt", RECON_BJ_VERTEX, nullptr, "mlp_u1");
    auto pr=[](const char* t, const Metrics& M){ std::printf(
        "  %-10s L1=%.4e cone=%.4e hump=%.4e slot=%.4e | cone_pk=%.3f slot_fill=%.3f range[%.3f,%.3f]\n",
        t, M.L1_total,M.L1_cone,M.L1_hump,M.L1_slot,M.cone_peak,M.slot_max,M.gmin,M.gmax); };
    if (mlponly) { pr("mlp_u1",a); return 0; }
    std::printf("  T-MLP-u-L (IDW p=2):\n");   Metrics b = run(m, idw, U0, "lev_tmlpul.txt", RECON_BJ_VERTEX, nullptr, "T-MLP-u-L");
    if (std::getenv("LEV_NOBVD")) { pr("mlp_u1",a); pr("T-MLP-u-L",b);
        std::printf("  T-MLP-u-L vs mlp_u1: L1 ratio %.4f, cone_pk %.3f vs %.3f, slot_fill %.3f vs %.3f\n",
                    b.L1_total/a.L1_total, b.cone_peak,a.cone_peak, b.slot_max,a.slot_max); return 0; }
    std::printf("  BVD (smooth BJ / sharp O2):\n");
    Metrics d = run(m, bj, U0, "lev_bvd.txt", RECON_BVD, &c2, "BVD");
    pr("mlp_u1",a); pr("T-MLP-u-L",b); pr("BVD",d);
    std::printf("  BVD vs mlp_u1: L1 ratio %.4f, cone_pk %.3f vs %.3f, slot_fill %.3f vs %.3f\n",
                d.L1_total/a.L1_total, d.cone_peak, a.cone_peak, d.slot_max, a.slot_max);
    std::printf("[WALL] TOTAL wall=%.3fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all_).count());
    return 0;
}
