// apps/rmi3d_bench.cpp — 3D single-fluid Richtmyer-Meshkov instability (RMI).
//
// First 3D physics result for the Stage-1 Cartesian-hex Euler solver and the
// MUSCL+RHLLC baseline for a later deg3t-BVD comparison.
//
//   Single-fluid Euler, gamma=1.4 everywhere (Atwood via density only).
//   Domain [0,Lx]x[0,Ly]x[0,Lz], Lx=4, Ly=Lz=1.
//   A Mach Ms=1.5 shock (moving +x, post-shock state on the left) hits a
//   cos-cos perturbed light/heavy interface at x0=1. The shock crosses the
//   interface and the RMI spike/bubble structure grows.
//
//   x_min = dirichlet (post-shock light gas, PRIMITIVE W), x_max = transmissive,
//   y,z = periodic.
//
// Env knobs: RMI_NX/RMI_NY/RMI_NZ (grid), RMI_T (t_end). Defaults below were
// picked from a short calibration so the full run fits a ~14 min budget.
#include "cfd/mesh.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/solver_euler3d.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>

using namespace cfd;

static int env_int(const char* k, int dflt) {
    const char* v = std::getenv(k);
    return v ? std::atoi(v) : dflt;
}
static double env_dbl(const char* k, double dflt) {
    const char* v = std::getenv(k);
    return v ? std::atof(v) : dflt;
}

int main() {
    Euler3D eq; eq.gamma = 1.4;

    // ---- Geometry ----
    const double Lx = 4.0, Ly = 1.0, Lz = 1.0;
    const int Nx = env_int("RMI_NX", 160);
    const int Ny = env_int("RMI_NY", 48);
    const int Nz = env_int("RMI_NZ", 48);
    const double t_end = env_dbl("RMI_T", 2.5);
    // Diagnostic knobs (defaults reproduce the prompt's locked full-run config).
    const double cfl    = env_dbl("RMI_CFL", 0.4);
    // RMI_RECON: 0=first-order, 1=MUSCL(BJ vertex), 2=deg3t-BVD Euler (RECON3_BVD).
    const int recon_sel = env_int("RMI_RECON", 1);
    const int recon     = (recon_sel == 2) ? (int)RECON3_BVD
                        : (recon_sel == 1) ? (int)RECON3_BJ_VERTEX
                        : (int)RECON3_FIRST;
    const int flux      = env_int("RMI_FLUX",  (int)FLUX3_RHLLC);       // 0=llf,1=hllc,2=rhllc,4=hll

    // ---- Physics constants (exact, do not alter) ----
    const double xs = 0.5;     // initial shock location
    const double x0 = 1.0;     // mean interface location
    const double a0 = 0.05;    // perturbation amplitude
    // Post-shock light gas (Rankine-Hugoniot, Ms=1.5, gamma=1.4 into rho1=1,p1=1):
    const double rho_ps = 1.86207, u_ps = 0.82168, p_ps = 2.45833;
    // Pre-shock light gas:
    const double rho_pre = 1.0, p_pre = 1.0;
    // Heavy gas (Atwood A=(3-1)/(3+1)=0.5):
    const double rho_hvy = 3.0, p_hvy = 1.0;

    Mesh m = build_structured_3d(Nx, Ny, Nz, Lx, Ly, Lz,
                                 /*px*/false, /*py*/true, /*pz*/true);
    const int N = m.n_cells();

    // ---- Initial condition ----
    std::vector<double> U0((size_t)5 * N);
    for (int c = 0; c < N; ++c) {
        double x = m.cell_centers[(size_t)c * 3 + 0];
        double y = m.cell_centers[(size_t)c * 3 + 1];
        double z = m.cell_centers[(size_t)c * 3 + 2];
        double xi = x0 + a0 * std::cos(2.0*M_PI*y/Ly) * std::cos(2.0*M_PI*z/Lz);
        double W[5];
        if (x < xs) {                 // post-shock light gas
            W[0]=rho_ps; W[1]=u_ps; W[2]=0; W[3]=0; W[4]=p_ps;
        } else if (x < xi) {          // pre-shock light gas
            W[0]=rho_pre; W[1]=0; W[2]=0; W[3]=0; W[4]=p_pre;
        } else {                      // heavy gas
            W[0]=rho_hvy; W[1]=0; W[2]=0; W[3]=0; W[4]=p_hvy;
        }
        double Uc[5]; eq.prim_to_cons(W, Uc);
        for (int v = 0; v < 5; ++v) U0[(size_t)v * N + c] = Uc[v];
    }
    // Total mass Σ(rho*Vol) at t=0 (conservation reference; x-boundaries are open so
    // the only legitimate change is the net x in/out mass flux over the run).
    double mass0 = 0.0;
    for (int c = 0; c < N; ++c) mass0 += U0[(size_t)0 * N + c] * m.cell_volumes[c];

    ReconCtx3D ctx = build_recon_ctx_3d(m);
    // o2 P2 LSQ context for the deg3t-BVD Euler path (built once; geometry-only).
    ReconCtx3DO2 o2ctx;
    const bool use_bvd = (recon == (int)RECON3_BVD);
    if (use_bvd) o2ctx = build_recon_ctx_3d_o2(m);

    // ---- Boundary conditions ----
    // tags (px=false,py=true,pz=true): 1 = x_min, 2 = x_max.
    std::vector<BC3D> bcs(3);
    bcs[1].kind = 2;  // x_min dirichlet, PRIMITIVE W = post-shock light gas
    bcs[1].state[0]=rho_ps; bcs[1].state[1]=u_ps; bcs[1].state[2]=0;
    bcs[1].state[3]=0;      bcs[1].state[4]=p_ps;
    bcs[2].kind = 0;  // x_max transmissive

    std::printf("RMI3D start: grid=%dx%dx%d (N=%d) t_end=%.3f integrator=SSP-RK3 cfl=%.3f flux=%d recon=%d\n",
                Nx, Ny, Nz, N, t_end, cfl, flux, recon);
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();
    Solve3DResult R = solve_euler3d(m, eq, U0, t_end, /*integrator*/2,
                                    cfl, /*dt_fixed*/-1.0, /*max_steps*/100000000,
                                    recon, &ctx, flux, &bcs,
                                    use_bvd ? &o2ctx : nullptr);
    auto t1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();

    // ---- Diagnostics ----
    double rmin=1e300, rmax=-1e300, pmin=1e300, pmax=-1e300;
    bool finite = true;
    for (int c = 0; c < N; ++c) {
        double u[5] = {R.U[0*N+c],R.U[1*N+c],R.U[2*N+c],R.U[3*N+c],R.U[4*N+c]};
        double w[5]; eq.cons_to_prim(u, w);
        if (!std::isfinite(w[0]) || !std::isfinite(w[4])) finite = false;
        if (w[0] < rmin) rmin = w[0]; if (w[0] > rmax) rmax = w[0];
        if (w[4] < pmin) pmin = w[4]; if (w[4] > pmax) pmax = w[4];
    }
    bool positivity = (rmin > 0.0) && (pmin > 0.0) && finite;
    std::printf("RMI3D done: rho=[%.6f,%.6f] p=[%.6f,%.6f] positivity=%s finite=%d steps=%d t=%.6f wall=%.1fs\n",
                rmin, rmax, pmin, pmax, positivity ? "OK" : "FAIL", (int)finite,
                R.n_steps, R.t, wall);
    // Mass-conservation report. y,z periodic; x open (dirichlet in / transmissive out),
    // so the net inflow ~ rho_ps*u_ps*(Ly*Lz)*t_final is the EXPECTED Σrho increase.
    double mass1 = 0.0;
    for (int c = 0; c < N; ++c) mass1 += R.U[(size_t)0 * N + c] * m.cell_volumes[c];
    double inflow_est = rho_ps * u_ps * (Ly * Lz) * R.t;   // crude (interface not yet at outlet)
    std::printf("RMI3D mass: Sigma_rho0=%.8e Sigma_rho1=%.8e drift=%.3e expected_inflow=%.3e residual=%.3e\n",
                mass0, mass1, mass1 - mass0, inflow_est, (mass1 - mass0) - inflow_est);
    std::fflush(stdout);

    // ---- Slice dumps ----
    auto cidx = [Nx, Ny](int i, int j, int k) { return (k * Ny + j) * Nx + i; };

    // z = Lz/2 slice (k = Nz/2): "x y rho" over all (i,j).
    const char* zslice_path = use_bvd ? "/tmp/mbq/rmi3d_bvd.txt" : "/tmp/mbq/rmi3d_mus.txt";
    {
        int k = Nz / 2;
        FILE* fp = std::fopen(zslice_path, "w");
        if (fp) {
            for (int j = 0; j < Ny; ++j)
                for (int i = 0; i < Nx; ++i) {
                    int c = cidx(i, j, k);
                    double x = m.cell_centers[(size_t)c * 3 + 0];
                    double y = m.cell_centers[(size_t)c * 3 + 1];
                    double rho = R.U[(size_t)0 * N + c];
                    std::fprintf(fp, "%.6f %.6f %.6f\n", x, y, rho);
                }
            std::fclose(fp);
        }
    }
    // y = Ly/2 slice (j = Ny/2): "x z rho" over all (i,k).
    {
        int j = Ny / 2;
        FILE* fp = std::fopen("/tmp/mbq/rmi3d_mus_xz.txt", "w");
        if (fp) {
            for (int k = 0; k < Nz; ++k)
                for (int i = 0; i < Nx; ++i) {
                    int c = cidx(i, j, k);
                    double x = m.cell_centers[(size_t)c * 3 + 0];
                    double z = m.cell_centers[(size_t)c * 3 + 2];
                    double rho = R.U[(size_t)0 * N + c];
                    std::fprintf(fp, "%.6f %.6f %.6f\n", x, z, rho);
                }
            std::fclose(fp);
        }
    }
    std::printf("RMI3D slices written: %s (z-mid), /tmp/mbq/rmi3d_mus_xz.txt (y-mid)\n", zslice_path);
    std::printf("RMI3D_GRID %dx%dx%d\nRMI3D_TFINAL %.6f\n", Nx, Ny, Nz, R.t);
    return 0;
}
