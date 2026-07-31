// mach3_bench.cpp — Mach-3 forward-facing-step flow (Emery/Woodward-Colella).
// Domain [0,3]x[0,1] minus step [0.6,3]x[0,0.2]; uniform Mach-3 inflow
// (rho=1.4, u=3, v=0, p=1); inflow=dirichlet, outflow=transmissive, walls=slip.
// HLLC flux + SSP-RK3. Compares mlp_u1 vs T-MLP-u-L (shock sharpness, positivity).
// Args: [Nx] [Ny] [t_end].
#include "cfd/solver_euler2d.hpp"
#include "cfd/diagnostics.hpp"
#include "cfd/io_vtk.hpp"
#include "cfd/mesh.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>

using namespace cfd;

struct Diag { double rho_min, rho_max, p_min, grad_max; int n_steps;
              int vortex_count; double enstrophy; int v_small, v_mid, v_large;
              int v_genuine, v_stable, q_active_cells, q_shock_cells, q_contact_rejected;
              double lambda_ci_peak; };

static Diag run(const Mesh& m, const Euler2D& eq, const std::vector<double>& U0,
                double t_end, int recon, const ReconCtx* ctx, const std::vector<BC2D>& bcs,
                const ReconCtx* metric_ctx, const ReconCtx* shock_ctx = nullptr,
                const char* dumpfile = nullptr, const ReconCtxO2* ctx_o2 = nullptr,
                const char* label = nullptr) {
    static const char* mf = std::getenv("M3_FLUX");
    static const int fk = (mf && mf[0]=='l') ? FLUX_LLF
                        : (mf && mf[0]=='r' && mf[1]=='o') ? FLUX_RROE          // roe = rotated-Roe hybrid
                        : (mf && mf[0]=='r') ? FLUX_RHLLC
                        : (mf && mf[0]=='h' && mf[1]=='l' && mf[2]=='l' && mf[3]=='\0') ? FLUX_HLL  // hll = Cheng2021 (plain HLL)
                        : FLUX_HLLC;                                            // llf/roe/rhllc/hll/hllc
    static const int integ = std::getenv("M3_INT") ? std::atoi(std::getenv("M3_INT")) : 2;   // 1=SSP-RK2 (Python), 2=RK3
    static const double cfl = std::getenv("M3_CFL") ? std::atof(std::getenv("M3_CFL")) : 0.4; // Python=0.35
    auto t0_ = std::chrono::steady_clock::now();
    Solve2DResult r = solve_euler2d(m, eq, U0, t_end, integ, cfl, -1.0, 10000000,
                                    recon, ctx, fk, &bcs, shock_ctx, ctx_o2);
    double wall_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    std::printf("[WALL] %s wall=%.3fs steps=%d\n", label ? label : "run", wall_, r.n_steps);
    std::fflush(stdout);
    const int N = m.n_cells();
    Diag d{1e9, -1e9, 1e9, 0.0, r.n_steps};
    std::vector<double> rho(N);
    for (int i = 0; i < N; ++i) {
        double U[4] = {r.U[0*N+i], r.U[1*N+i], r.U[2*N+i], r.U[3*N+i]}, W[4];
        eq.cons_to_prim(U, W);
        rho[i] = W[0];
        d.rho_min = std::min(d.rho_min, W[0]); d.rho_max = std::max(d.rho_max, W[0]);
        d.p_min = std::min(d.p_min, W[3]);
    }
    for (int f = 0; f < m.n_faces(); ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        if (n < 0) continue;
        d.grad_max = std::max(d.grad_max, std::fabs(rho[o] - rho[n]));
    }
    const char* dump = dumpfile ? dumpfile : std::getenv("M3_DUMP");
    if (dump) { FILE* fh = std::fopen(dump, "w");
        for (int i = 0; i < N; ++i) std::fprintf(fh,"%.6g %.6g %.6g\n",
            m.cell_centers[i*2], m.cell_centers[i*2+1], rho[i]); std::fclose(fh);
        cfd::write_vtk2d_euler(std::string(dump)+".vtk", m, eq, r.U); }
    // calibrated slip-stream ROI (from mach3_ref.png): x[0.7,2.5] y[0.6,0.95]
    QDiag q = q_criterion_roi(m, eq, *metric_ctx, r.U, 0.7, 2.5, 0.6, 0.95);
    d.vortex_count = q.n_vortices; d.enstrophy = q.q_integral;
    d.v_small = q.n_small; d.v_mid = q.n_mid; d.v_large = q.n_large;
    d.v_genuine = q.n_genuine; d.v_stable = q.n_stable_genuine;
    d.q_active_cells = q.n_active_cells; d.q_shock_cells = q.n_shock_cells;
    d.q_contact_rejected = q.n_contact_rejected;
    d.lambda_ci_peak = q.lambda_ci_peak;
    return d;
}

int main(int argc, char** argv) {
    int Nx = argc > 1 ? std::atoi(argv[1]) : 120;
    int Ny = argc > 2 ? std::atoi(argv[2]) : 40;
    double t_end = argc > 3 ? std::atof(argv[3]) : 0.5;
    double idw_p = argc > 4 ? std::atof(argv[4]) : 2.0;
    const char* ghf = std::getenv("GRADED_HF");
    const char* mesh_env = std::getenv("M3_MESH");
    const bool uniform_mesh = mesh_env && std::string(mesh_env) == "uniform";
    // M3_MESH2D: load an external unstructured .mesh2d (Nn Ne, node xy, tri idx) e.g. a gmsh
    // Delaunay uniform mesh of the L-shaped step domain. Boundary tags by face centroid:
    // 1 left inflow (x=0), 2 right outflow (x=3), 3 everything else (top/bottom/step walls).
    const char* m2d = std::getenv("M3_MESH2D");
    Mesh m;
    if (m2d) {
        std::ifstream in(m2d);
        int Nn = 0, Ne = 0; in >> Nn >> Ne;
        std::vector<double> nds(2 * Nn);
        for (int i = 0; i < Nn; ++i) in >> nds[2*i] >> nds[2*i+1];
        std::vector<std::vector<int>> els(Ne, std::vector<int>(3));
        for (int i = 0; i < Ne; ++i) in >> els[i][0] >> els[i][1] >> els[i][2];
        auto classify = [](double cx, double, double, double) -> int {
            if (cx <= 1e-6)       return 1;   // left inflow
            if (cx >= 3.0 - 1e-6) return 2;   // right outflow
            return 3;                          // top/bottom/step walls
        };
        m = build_unstructured_2d(nds, els, classify, {"inflow","outflow","wall"});
        m.kind = "forward_step";
        std::printf("M3_MESH2D=%s cells=%d nodes=%d\n", m2d, m.n_cells(), Nn);
    } else {
        m = ghf ? forward_step_mesh_graded(std::atof(ghf),
                       std::getenv("GRADED_HC") ? std::atof(std::getenv("GRADED_HC")) : 0.02)
                 : (uniform_mesh ? forward_step_mesh(Nx, Ny)
                                 : forward_step_mesh_roi_graded(Nx, Ny));
    }
    ReconCtx bj = build_recon_ctx(m, 0.0), idw = build_recon_ctx(m, idw_p);
    Euler2D eq{1.4};
    const int N = m.n_cells();
    double Win[4] = {1.4, 3.0, 0.0, 1.0}, Uin[4]; eq.prim_to_cons(Win, Uin);
    std::vector<double> U0(4 * N);
    for (int i = 0; i < N; ++i) for (int v = 0; v < 4; ++v) U0[(size_t)v*N+i] = Uin[v];
    std::vector<BC2D> bcs(4);
    bcs[1].kind = 2; for (int v=0;v<4;++v) bcs[1].state[v] = Win[v]; // inflow dirichlet
    bcs[2].kind = 0;                                                  // outflow transmissive
    bcs[3].kind = 1;                                                  // walls reflective

    std::printf("Mach-3 forward step: cells=%d  grid=%dx%d  mesh=%s  t_end=%.2f\n",
                N, Nx, Ny, m.kind.c_str(), t_end);
    const bool bvd_only = std::getenv("M3_BVD_ONLY") != nullptr;  // skip T-MLP-u-L
    const bool mlponly = std::getenv("M3_MLPONLY") != nullptr;    // mlp_u1 line only (fast dump)
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    auto t_all_ = std::chrono::steady_clock::now();
    if (std::getenv("M3_CONLY")) {  // T-MLP-u-C line ONLY. M3_O2 -> genuine P2 (RECON_O2); M3_DUMP override for parallel runs
        int rc = std::getenv("M3_O2") ? RECON_O2 : RECON_BVD;
        const char* dmp = std::getenv("M3_DUMP") ? std::getenv("M3_DUMP") : "m3_bvd.txt";
        Diag cv = run(m, eq, U0, t_end, rc, &bj, bcs, &bj, nullptr, dmp, &c2, "C_line");
        std::printf("  C_line: rho[%.4f,%.4f] p_min=%.4f max|drho|=%.4f rollups=%d (small=%d mid=%d large=%d genuine=%d stable=%d) ens=%.3f lci_peak=%.4f qcells=%d shockcells=%d contactrej=%d\n",
            cv.rho_min, cv.rho_max, cv.p_min, cv.grad_max, cv.vortex_count, cv.v_small, cv.v_mid, cv.v_large,
            cv.v_genuine, cv.v_stable, cv.enstrophy, cv.lambda_ci_peak, cv.q_active_cells, cv.q_shock_cells, cv.q_contact_rejected);
        return 0;
    }
    Diag a = run(m, eq, U0, t_end, RECON_BJ_VERTEX, &bj,  bcs, &bj, nullptr, "m3_mlpu1.txt", nullptr, "mlp_u1");
    if (mlponly) { std::printf("  mlp_line: rho[%.4f,%.4f] p_min=%.4f max|drho|=%.4f rollups=%d (small=%d mid=%d large=%d genuine=%d stable=%d) ens=%.3f lci_peak=%.4f qcells=%d shockcells=%d contactrej=%d\n",
        a.rho_min, a.rho_max, a.p_min, a.grad_max, a.vortex_count, a.v_small, a.v_mid, a.v_large,
        a.v_genuine, a.v_stable, a.enstrophy, a.lambda_ci_peak, a.q_active_cells, a.q_shock_cells, a.q_contact_rejected); return 0; }
    Diag b{};
    if (!bvd_only) b = run(m, eq, U0, t_end, RECON_BJ_VERTEX, &idw, bcs, &bj, nullptr, "m3_tmlpul.txt", nullptr, "T-MLP-u-L");
    Diag bv = run(m, eq, U0, t_end, RECON_BVD, &bj, bcs, &bj, nullptr, "m3_bvd.txt", &c2, "BVD");
    std::printf("  BVD       : steps=%d rho[%.4f,%.4f] p_min=%.4f max|drho|=%.4f | rollups=%d Qint=%.3f\n",
                bv.n_steps, bv.rho_min, bv.rho_max, bv.p_min, bv.grad_max, bv.vortex_count, bv.enstrophy);
    std::printf("  BVD vs mlp_u1: shock %.4f, rollups %d vs %d\n",
                bv.grad_max/a.grad_max, bv.vortex_count, a.vortex_count);
    std::printf("  mlp_u1    : steps=%d rho[%.4f,%.4f] p_min=%.4f max|drho|=%.4f | upper-rollups=%d enstrophy=%.3f\n",
                a.n_steps, a.rho_min, a.rho_max, a.p_min, a.grad_max, a.vortex_count, a.enstrophy);
    std::printf("  T-MLP-u-L : steps=%d rho[%.4f,%.4f] p_min=%.4f max|drho|=%.4f | upper-rollups=%d enstrophy=%.3f\n",
                b.n_steps, b.rho_min, b.rho_max, b.p_min, b.grad_max, b.vortex_count, b.enstrophy);
    std::printf("  Mach3 gate: shock %.4f (%s), rollups %d vs %d, enstrophy ratio %.4f; positivity %s\n",
                b.grad_max / a.grad_max, b.grad_max > a.grad_max ? "sharper" : "softer",
                b.vortex_count, a.vortex_count, b.enstrophy/std::max(a.enstrophy,1e-30),
                (a.rho_min > 0 && a.p_min > 0 && b.rho_min > 0 && b.p_min > 0) ? "OK both" : "VIOLATION");
    std::printf("[WALL] TOTAL wall=%.3fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all_).count());
    return 0;
}
