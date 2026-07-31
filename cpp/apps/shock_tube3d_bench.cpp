// apps/shock_tube3d_bench.cpp — 3D compressible Navier-Stokes viscous shock tube
// (Daru-Tenaud, JCP 2009 / Comput. Fluids 2001).
//
// Validates the NS extension (viscous flux + no-slip adiabatic wall + MLP-limited
// deg3t-BVD recon) in 3D. A Sod-like diaphragm at x=0.5 drives a shock + contact
// down a CLOSED tube; the contact slides along the bottom NO-SLIP wall and rolls up
// into the characteristic lambda-shock + primary vortex at the contact/wall junction.
//
//   Domain [0,1] x [0,0.5] x [0,Lz] (Lz thin, z PERIODIC — quasi-2D).
//   IC (gamma=1.4): x<0.5 -> rho=120, u=v=w=0, p=120/gamma ;
//                   x>=0.5-> rho=1.2,  u=v=w=0, p=1.2/gamma.  (a=1 both sides, Ms~2.37)
//   BC: y=0 no-slip adiabatic wall (kind 4); y=0.5 slip wall (kind 1, symmetry plane);
//       x=0 & x=1 no-slip adiabatic walls (kind 4, closed tube); z periodic.
//   mu from Re: mu = rho_R * a * L / Re = 1.2 * 1 * 1 / Re. k = mu*cp/Pr, Pr=0.72.
//
// Env: SHKTUBE_NX/NY/NZ (200/100/8), SHKTUBE_T (1.0), SHKTUBE_RE (200), SHKTUBE_MU
//      (overrides Re if set). Dumps a z-mid "x y rho" slice to /tmp/mbq/shktube3d.txt.
#include "cfd/mesh.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/viscous3d.hpp"
#include "cfd/solver_euler3d.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace cfd;

static int env_int(const char* k, int dflt) { const char* v = std::getenv(k); return v ? std::atoi(v) : dflt; }
static double env_dbl(const char* k, double dflt) { const char* v = std::getenv(k); return v ? std::atof(v) : dflt; }

int main() {
    Euler3D eq; eq.gamma = 1.4;

    // ---- Geometry ----
    const double Lx = 1.0, Ly = 0.5, Lz = env_dbl("SHKTUBE_LZ", 0.04);
    // Default grid 120x60x4: the strong Daru-Tenaud shock + fine dy forces a small
    // CFL-limited dt (~5e-4), so 200x100 needs ~2e4 steps (>12 min). 120x60x4 (~1540
    // steps, ~75 s @24 threads) resolves the lambda-shock + primary vortex within budget.
    const int Nx = env_int("SHKTUBE_NX", 120);
    const int Ny = env_int("SHKTUBE_NY", 60);
    const int Nz = env_int("SHKTUBE_NZ", 4);
    const double t_end = env_dbl("SHKTUBE_T", 1.0);
    const double cfl   = env_dbl("SHKTUBE_CFL", 0.2);   // strong-shock viscous: low CFL for positivity (0.2 verified stable to t=1)
    const double Re    = env_dbl("SHKTUBE_RE", 200.0);

    // ---- Viscosity from Re (mu = rho_R * a * L / Re); SHKTUBE_MU overrides. ----
    const double rho_R = 1.2, a_ref = 1.0, L_ref = 1.0;
    double mu = rho_R * a_ref * L_ref / Re;
    { const char* e = std::getenv("SHKTUBE_MU"); if (e) mu = std::atof(e); }
    ViscousParams vp; vp.mu = mu; vp.Pr = 0.72; vp.R = 1.0;

    // z PERIODIC; x,y are walls (non-periodic).
    Mesh m = build_structured_3d(Nx, Ny, Nz, Lx, Ly, Lz,
                                 /*px*/false, /*py*/false, /*pz*/true);
    const int N = m.n_cells();

    // ---- Initial condition (Daru-Tenaud diaphragm at x=0.5) ----
    const double rhoL = 120.0, pL = 120.0 / eq.gamma;
    const double rhoR = 1.2,   pR = 1.2 / eq.gamma;
    std::vector<double> U0((size_t)5 * N);
    for (int c = 0; c < N; ++c) {
        double x = m.cell_centers[(size_t)c * 3 + 0];
        double W[5];
        if (x < 0.5) { W[0]=rhoL; W[1]=0; W[2]=0; W[3]=0; W[4]=pL; }
        else         { W[0]=rhoR; W[1]=0; W[2]=0; W[3]=0; W[4]=pR; }
        double Uc[5]; eq.prim_to_cons(W, Uc);
        for (int v = 0; v < 5; ++v) U0[(size_t)v * N + c] = Uc[v];
    }
    double mass0 = 0.0;
    for (int c = 0; c < N; ++c) mass0 += U0[(size_t)0 * N + c] * m.cell_volumes[c];

    // ---- Contexts ----
    ReconCtx3D ctx = build_recon_ctx_3d(m);             // face-nbr LSQ (MUSCL + viscous grads)
    ReconCtx3DO2 o2ctx = build_recon_ctx_3d_o2(m);      // o2 P2 (deg3t-BVD)

    // ---- Boundary conditions ----
    // tags (px=false,py=false,pz=true): 1=x_min, 2=x_max, 3=y_min, 4=y_max.
    std::vector<BC3D> bcs(5);
    bcs[1].kind = 4;   // x_min : no-slip adiabatic wall (closed tube, left)
    bcs[2].kind = 4;   // x_max : no-slip adiabatic wall (closed tube, right)
    bcs[3].kind = 4;   // y_min : NO-SLIP adiabatic wall (the bottom wall — boundary layer)
    bcs[4].kind = 1;   // y_max : slip wall (symmetry plane at the tube mid-height)

    // SHKTUBE_RECON: 0=first-order, 1=MUSCL(BJ vertex), 4=deg3t-BVD+MLP (default).
    const int recon_sel = env_int("SHKTUBE_RECON", 4);
    const int recon = (recon_sel == 0) ? (int)RECON3_FIRST
                    : (recon_sel == 1) ? (int)RECON3_BJ_VERTEX
                    : (int)RECON3_BVD;
    const int flux  = (int)FLUX3_RHLLC;

    std::printf("ShockTube3D start: grid=%dx%dx%d (N=%d) Lz=%.3f t_end=%.3f cfl=%.3f\n",
                Nx, Ny, Nz, N, Lz, t_end, cfl);
    std::printf("ShockTube3D physics: Daru-Tenaud Ms~2.37, Re=%.1f mu=%.6f Pr=%.2f flux=RHLLC recon=deg3t-BVD+MLP integrator=SSP-RK3\n",
                Re, mu, vp.Pr);
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();
    Solve3DResult R = solve_euler3d(m, eq, U0, t_end, /*integrator*/2,
                                    cfl, /*dt_fixed*/-1.0, /*max_steps*/100000000,
                                    recon, &ctx, flux, &bcs, &o2ctx, &vp);
    auto t1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();

    // ---- Diagnostics ----
    double rmin=1e300, rmax=-1e300, pmin=1e300, pmax=-1e300, umax=0.0;
    bool finite = true;
    for (int c = 0; c < N; ++c) {
        double u[5] = {R.U[0*N+c],R.U[1*N+c],R.U[2*N+c],R.U[3*N+c],R.U[4*N+c]};
        double w[5]; eq.cons_to_prim(u, w);
        if (!std::isfinite(w[0]) || !std::isfinite(w[4])) finite = false;
        if (w[0] < rmin) rmin = w[0]; if (w[0] > rmax) rmax = w[0];
        if (w[4] < pmin) pmin = w[4]; if (w[4] > pmax) pmax = w[4];
        double sp = std::sqrt(w[1]*w[1]+w[2]*w[2]+w[3]*w[3]); if (sp > umax) umax = sp;
    }
    bool positivity = (rmin > 0.0) && (pmin > 0.0) && finite;
    std::printf("ShockTube3D done: rho=[%.6f,%.6f] p=[%.6f,%.6f] |u|max=%.4f positivity=%s finite=%d steps=%d t=%.6f wall=%.1fs\n",
                rmin, rmax, pmin, pmax, umax, positivity ? "OK" : "FAIL", (int)finite, R.n_steps, R.t, wall);
    double mass1 = 0.0;
    for (int c = 0; c < N; ++c) mass1 += R.U[(size_t)0 * N + c] * m.cell_volumes[c];
    std::printf("ShockTube3D mass: Sigma_rho0=%.8e Sigma_rho1=%.8e drift=%.3e (closed tube ⇒ should be ~0)\n",
                mass0, mass1, mass1 - mass0);
    std::fflush(stdout);

    // ---- z-mid slice dump: "x y rho" over all (i,j) at k=Nz/2. ----
    auto cidx = [Nx, Ny](int i, int j, int k) { return (k * Ny + j) * Nx + i; };
    {
        int k = Nz / 2;
        FILE* fp = std::fopen("/tmp/mbq/shktube3d.txt", "w");
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
            std::printf("ShockTube3D slice written: /tmp/mbq/shktube3d.txt (z-mid, %d points)\n", Nx*Ny);
        }
    }
    std::printf("SHKTUBE3D_GRID %dx%dx%d\nSHKTUBE3D_RE %.1f\nSHKTUBE3D_MU %.6f\nSHKTUBE3D_TFINAL %.6f\n",
                Nx, Ny, Nz, Re, mu, R.t);
    return 0;
}
