// double_mach_bench.cpp — Double-Mach reflection (Woodward-Colella). Mach-10
// shock at 60deg hitting a reflecting wall. Domain [0,4]x[0,1], t_end=0.2.
// post-shock (rho8,u7.1447,v-4.125,p116.5), pre (1.4,0,0,1). Time-dependent top.
// HLLC + SSP-RK3 + T-MLP-u-L face LMP bound for robustness. Args:[Nx][Ny][t_end].
#include "cfd/solver_euler2d.hpp"
#include "cfd/diagnostics.hpp"
#include "cfd/io_vtk.hpp"
#include "cfd/mesh.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

using namespace cfd;
static const double SQ3 = std::sqrt(3.0), X0 = 1.0/6.0;
static const double POST[4] = {8.0, 7.1447096, -4.125, 116.5};
static const double PRE[4]  = {1.4, 0.0, 0.0, 1.0};

struct Diag { double rho_min, rho_max, p_min; int n_steps; double t;
              double enstrophy, omega_peak; int vortex_count;
              int qn, qs, qm, ql; double qint; };

// Vorticity omega = dv/dx - du/dy via the LSQ vertex-stencil gradient (reuse
// ctx). In the slip-line ROI, count coherent vortices = local |omega| maxima
// above a threshold (KH rollups), accumulate enstrophy and peak. This is the
// Double-Mach gate signal: sharper/less-diffusive recon => more rollups.
static void vorticity_metrics(const Mesh& m, const Euler2D& eq, const ReconCtx& c,
                              const std::vector<double>& U, Diag& d) {
    const int N = m.n_cells();
    std::vector<double> u(N), v(N), om(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double Uc[4] = {U[0*N+i],U[1*N+i],U[2*N+i],U[3*N+i]}, W[4];
        eq.cons_to_prim(Uc, W); u[i] = W[1]; v[i] = W[2];
    }
    // LSQ gradients of u and v from the vertex stencil.
    for (int ci = 0; ci < N; ++ci) {
        double ru0=0,ru1=0,rv0=0,rv1=0;
        for (int k = 0; k < c.max_nb; ++k) {
            int nb = c.nb[(size_t)ci*c.max_nb+k]; if (nb < 0) continue;
            double wk = c.w[(size_t)ci*c.max_nb+k];
            double dx = c.d[((size_t)ci*c.max_nb+k)*2+0], dy = c.d[((size_t)ci*c.max_nb+k)*2+1];
            ru0 += wk*dx*(u[nb]-u[ci]); ru1 += wk*dy*(u[nb]-u[ci]);
            rv0 += wk*dx*(v[nb]-v[ci]); rv1 += wk*dy*(v[nb]-v[ci]);
        }
        double dudy = c.ATA_inv[ci*4+2]*ru0 + c.ATA_inv[ci*4+3]*ru1;
        double dvdx = c.ATA_inv[ci*4+0]*rv0 + c.ATA_inv[ci*4+1]*rv1;
        om[ci] = dvdx - dudy;
    }
    // ROI = slip-line rollup region behind the triple point.
    auto inroi = [&](int i){ double x=m.cell_centers[i*2], y=m.cell_centers[i*2+1];
                             return x>=2.0 && x<=2.95 && y>=0.0 && y<=0.5; };
    double ens=0, pk=0; for (int i=0;i<N;++i) if (inroi(i)) {
        ens += om[i]*om[i]*m.cell_volumes[i]; pk = std::max(pk, std::fabs(om[i])); }
    // coherent vortex count: local |omega| maxima above 0.25*peak within ROI.
    int cnt = 0; double thr = 0.25*pk;
    for (int ci = 0; ci < N; ++ci) {
        if (!inroi(ci) || std::fabs(om[ci]) < thr) continue;
        bool ismax = true;
        for (int k = 0; k < c.max_nb; ++k) { int nb=c.nb[(size_t)ci*c.max_nb+k];
            if (nb>=0 && std::fabs(om[nb]) > std::fabs(om[ci])) { ismax=false; break; } }
        if (ismax) ++cnt;
    }
    d.enstrophy = ens; d.omega_peak = pk; d.vortex_count = cnt;
}

static Diag run(const Mesh& m, const Euler2D& eq, const std::vector<double>& U0,
                double t_end, int recon, const ReconCtx* ctx, const ReconCtx* metric_ctx,
                const ReconCtxO2* ctx_o2 = nullptr, const char* dumpfile = nullptr,
                const char* label = nullptr) {
    std::vector<BC2D> bcs(6);
    bcs[1].kind = 2; for (int v=0;v<4;++v) bcs[1].state[v] = POST[v];  // left inflow
    bcs[2].kind = 0;                                                    // right outflow
    bcs[3].kind = 2; for (int v=0;v<4;++v) bcs[3].state[v] = POST[v];  // bottom x<1/6
    bcs[4].kind = 1;                                                    // bottom wall (reflective)
    bcs[5].kind = 3;                                                    // top time-dependent
    bcs[5].func = [](double x, double, double t, double* W) {
        double xs = X0 + (1.0 + 20.0 * t) / SQ3;
        const double* s = (x < xs) ? POST : PRE;
        for (int v = 0; v < 4; ++v) W[v] = s[v];
    };
    static const int dmflux = []{ const char* f=std::getenv("DM_FLUX"); if(!f)return (int)FLUX_HLLC; std::string s=f; if(s=="hll")return (int)FLUX_HLL; if(s=="rhllc")return (int)FLUX_RHLLC; if(s=="llf")return (int)FLUX_LLF; if(s=="roe")return (int)FLUX_ROE_EF; if(s=="rroe")return (int)FLUX_RROE; return (int)FLUX_HLLC; }();
    static const double dmcfl = []{ const char* e=std::getenv("DM_CFL"); return e?std::atof(e):0.3; }();  // DM_CFL override (default 0.3; Cheng-like ~0.6 = fewer steps, less accumulated diffusion)
    auto t0_ = std::chrono::steady_clock::now();
    Solve2DResult r = solve_euler2d(m, eq, U0, t_end, 2, dmcfl, -1.0, 100000000,
                                    recon, ctx, dmflux, &bcs, nullptr, ctx_o2);
    double wall_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    std::printf("[WALL] %s wall=%.3fs steps=%d\n", label ? label : "run", wall_, r.n_steps);
    std::fflush(stdout);
    const int N = m.n_cells();
    Diag d{1e9, -1e9, 1e9, r.n_steps, r.t};
    for (int i = 0; i < N; ++i) {
        double U[4] = {r.U[0*N+i],r.U[1*N+i],r.U[2*N+i],r.U[3*N+i]}, W[4];
        eq.cons_to_prim(U, W);
        d.rho_min = std::min(d.rho_min, W[0]); d.rho_max = std::max(d.rho_max, W[0]);
        d.p_min = std::min(d.p_min, W[3]);
    }
    vorticity_metrics(m, eq, *metric_ctx, r.U, d);
    QDiag q = q_criterion_roi(m, eq, *metric_ctx, r.U, 2.0, 2.95, 0.0, 0.5);
    d.qn=q.n_vortices; d.qs=q.n_small; d.qm=q.n_mid; d.ql=q.n_large; d.qint=q.q_integral;
    const char* dump = dumpfile ? dumpfile : std::getenv("DM_DUMP");
    if (dump) { FILE* fh = std::fopen(dump, "w");
        for (int i = 0; i < N; ++i) { double U[4]={r.U[0*N+i],r.U[1*N+i],r.U[2*N+i],r.U[3*N+i]},W[4];
            eq.cons_to_prim(U,W); std::fprintf(fh,"%.6g %.6g %.6g\n",
            m.cell_centers[i*2],m.cell_centers[i*2+1],W[0]); } std::fclose(fh);
        cfd::write_vtk2d_euler(std::string(dump)+".vtk", m, eq, r.U); }
    return d;
}

int main(int argc, char** argv) {
    // S1 default flux (2026-07-07, user): HLLC + blend=0 (pure HLLC, contact-preserving) + PVRS
    // (Toro pressure-based wave speeds). setenv-if-unset (3rd arg 0) so an explicit env overrides.
    // DM_FLUX already defaults to HLLC below; these set the HLLC sub-options.
    setenv("HLLC_HLLBLEND", "0", 0);
    setenv("HLLC_PVRS", "1", 0);
    // MUSCL member limiter = MLP-u2 (Venkatakrishnan, eps->0) per Cheng 2021 / dln_paper.
    setenv("MLP_U2", "0.001", 0);
    int Nx = argc > 1 ? std::atoi(argv[1]) : 240;
    int Ny = argc > 2 ? std::atoi(argv[2]) : 60;
    double t_end = argc > 3 ? std::atof(argv[3]) : 0.2;
    double idw_p = argc > 4 ? std::atof(argv[4]) : 2.0;
    const char* ghf = std::getenv("GRADED_HF");
    // DM_MESH2D: load an external unstructured .mesh2d (Nn Ne, node xy, tri indices) e.g. a
    // gmsh Delaunay near-equilateral mesh of [0,4]x[0,1] — removes the right-isoceles single-
    // diagonal anisotropy that biases/damps the oblique slip-line KH rollup. Boundary tags by
    // face centroid (1 left,2 right,3 bottom x<1/6 post-shock,4 bottom wall,5 top) = DM scheme.
    const char* mfile = std::getenv("DM_MESH2D");
    Mesh m;
    if (mfile) {
        std::ifstream in(mfile);
        int Nn = 0, Ne = 0; in >> Nn >> Ne;
        std::vector<double> nds(2 * Nn);
        for (int i = 0; i < Nn; ++i) in >> nds[2*i] >> nds[2*i+1];
        std::vector<std::vector<int>> els(Ne, std::vector<int>(3));
        for (int i = 0; i < Ne; ++i) in >> els[i][0] >> els[i][1] >> els[i][2];
        auto classify = [](double cx, double cy, double, double) -> int {
            const double xs = 1.0/6.0;
            if (cx <= 1e-6)        return 1;                 // left inflow
            if (cx >= 4.0 - 1e-6)  return 2;                 // right outflow
            if (cy <= 1e-6)        return (cx < xs) ? 3 : 4; // bottom: post-shock / reflective wall
            return 5;                                        // top time-dependent
        };
        m = build_unstructured_2d(nds, els, classify, {"left","right","bottom_post","bottom_wall","top"});
        m.kind = "double_mach";
        std::printf("DM_MESH2D=%s cells=%d nodes=%d\n", mfile, m.n_cells(), Nn);
    } else {
        m = ghf ? double_mach_mesh_graded(std::atof(ghf),
                       std::getenv("GRADED_HC") ? std::atof(std::getenv("GRADED_HC")) : 0.02)
                 : double_mach_mesh(Nx, Ny, 4.0, 1.0);
    }
    // DM_BJ_IDW: inverse-distance weighting power for the BVD/MUSCL P1 LSQ gradient ctx.
    // DEFAULT 2 = 1/d^2 -> COMMON with the P2 (THINC/QQ hessian) LSQ which is hardcoded 1/d^2
    // (reconstruct2d_o2.hpp:63). Unifies P1 gradient and P2 hessian weighting (user directive).
    static const double bj_idw = []{ const char* e=std::getenv("DM_BJ_IDW"); return e?std::atof(e):2.0; }();
    ReconCtx bj = build_recon_ctx(m, bj_idw), idw = build_recon_ctx(m, idw_p);
    std::printf("(idw_p=%.1f bj_idw=%.1f)\n", idw_p, bj_idw);
    Euler2D eq{1.4};
    const int N = m.n_cells();
    std::vector<double> U0(4 * N);
    for (int i = 0; i < N; ++i) {
        double cx = m.cell_centers[i*2+0], cy = m.cell_centers[i*2+1];
        const double* s = (cx < X0 + cy / SQ3) ? POST : PRE;
        double U[4]; eq.prim_to_cons(s, U);
        for (int v = 0; v < 4; ++v) U0[(size_t)v*N+i] = U[v];
    }
    std::printf("Double-Mach reflection: cells=%d grid=%dx%d t_end=%.3f\n", N, Nx, Ny, t_end);
    // DM_BVD_ONLY=1 -> convergence mode: run only mlp_u1 + BVD (skip idw T-MLP-u-L
    // and order-2) to cut compute. Default runs all 4.
    const bool bvd_only = std::getenv("DM_BVD_ONLY") != nullptr;
    const bool dm_single = std::getenv("DM_SINGLE") != nullptr;   // BVD only (fast sweep)
    const bool dm_mlponly = std::getenv("DM_MLPONLY") != nullptr; // mlp_u1 line only (fast dump)
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    // pure mlp_u1 = BJ-vertex multidim limiter (the baseline for comparison figures).
    Diag a{};
    auto t_all_ = std::chrono::steady_clock::now();
    if (!dm_single) a = run(m, eq, U0, t_end, RECON_BJ_VERTEX, &bj,  &bj, nullptr, "cmp_mlpu1.txt", "mlp_u1");
    if (dm_mlponly) { std::printf("  mlp_line: posOK=%d rho[%.3f,%.3f] p_min=%.4f ens=%.1f Qvort=%d Qint=%.2f\n",
        (a.rho_min>0&&a.p_min>0), a.rho_min, a.rho_max, a.p_min, a.enstrophy, a.qn, a.qint); return 0; }
    Diag b{}, o2{};
    if (!bvd_only && !dm_single) {
        b = run(m, eq, U0, t_end, RECON_TMLPU_L, &idw, &bj, nullptr, "cmp_tmlpul.txt", "T-MLP-u-L");
        // order-2 (quadratic MLP) — genuine high-order; smoke-test stability + vortices
        o2 = run(m, eq, U0, t_end, RECON_O2, &bj, &bj, &c2, "cmp_o2.txt", "ORDER-2");
        std::printf("  ORDER-2      : posOK=%d rho[%.3f,%.3f] p_min=%.4f | Q-crit vort=%d (S%d/M%d/L%d) Qint=%.2f\n",
                    (o2.rho_min>0&&o2.p_min>0), o2.rho_min, o2.rho_max, o2.p_min, o2.qn,o2.qs,o2.qm,o2.ql,o2.qint);
    }
    Diag bv = run(m, eq, U0, t_end, RECON_BVD, &bj, &bj, &c2, "cmp_bvd.txt", "BVD");
    std::printf("  BVD          : posOK=%d rho[%.3f,%.3f] p_min=%.4f | Q-crit vort=%d (S%d/M%d/L%d) Qint=%.2f\n",
                (bv.rho_min>0&&bv.p_min>0), bv.rho_min, bv.rho_max, bv.p_min, bv.qn,bv.qs,bv.qm,bv.ql,bv.qint);
    std::printf("  mlp_u1_tmlpu : %s | rawW: vort=%d ens=%.1f | Q-crit: vort=%d (S%d/M%d/L%d) Qint=%.2f\n",
                (a.rho_min>0&&a.p_min>0)?"posOK":"VIOL", a.vortex_count, a.enstrophy, a.qn,a.qs,a.qm,a.ql,a.qint);
    std::printf("  T-MLP-u-L    : %s | rawW: vort=%d ens=%.1f | Q-crit: vort=%d (S%d/M%d/L%d) Qint=%.2f\n",
                (b.rho_min>0&&b.p_min>0)?"posOK":"VIOL", b.vortex_count, b.enstrophy, b.qn,b.qs,b.qm,b.ql,b.qint);
    std::printf("  DM gate BVD vs mlp_u1: vortices %d vs %d (%s); size-bins S/M/L %d/%d/%d vs %d/%d/%d; Qint ratio %.3f\n",
                bv.qn, a.qn, bv.qn>a.qn?"BVD MORE":(bv.qn==a.qn?"tie":"fewer"),
                bv.qs,bv.qm,bv.ql, a.qs,a.qm,a.ql, bv.qint/std::max(a.qint,1e-30));
    if (!bvd_only)
        std::printf("  DM gate (Q-criterion, robust): vortices %d vs %d (%s); size-bins S/M/L %d/%d/%d vs %d/%d/%d; Qint ratio %.3f\n",
                b.qn, a.qn, b.qn>a.qn?"T-MLP-u-L MORE":(b.qn==a.qn?"tie":"fewer"),
                b.qs,b.qm,b.ql, a.qs,a.qm,a.ql, b.qint/std::max(a.qint,1e-30));
    std::printf("[WALL] TOTAL wall=%.3fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all_).count());
    return 0;
}
