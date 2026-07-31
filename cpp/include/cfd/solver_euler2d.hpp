// cfd/solver_euler2d.hpp — 2D Euler finite-volume time march.
// Reconstruction on primitive variables (first-order or BJ-vertex MLP-u /
// T-MLP-u-L), LLF flux, SSP-RK3, transmissive BC. Port of the 2D Euler path of
// solver.py. State U is 4*N (var-major: rho, rho u, rho v, rho E).
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/euler2d.hpp"
#include "cfd/reconstruct2d.hpp"
#include "cfd/reconstruct2d_o2.hpp"
#include "cfd/solver2d.hpp"   // ReconMode

namespace cfd { constexpr int RECON_O2 = 3; }  // MLP-limited order-2
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cstdio>

namespace cfd {

enum FluxKind { FLUX_LLF = 0, FLUX_HLLC = 1, FLUX_RHLLC = 2, FLUX_RROE = 3, FLUX_HLL = 4, FLUX_SLAU2 = 5, FLUX_RSLAU2 = 6, FLUX_ROE_EF = 7, FLUX_HLLCM = 8, FLUX_RHLLCM = 9 };

// Boundary condition per patch tag. kind: 0 transmissive, 1 reflective (slip
// wall), 2 dirichlet (fixed W), 3 dirichlet_func (W = f(x,y,t)).
struct BC2D {
    int kind = 0;
    double state[4] = {0,0,0,0};
    std::function<void(double, double, double, double*)> func;
};

// Apply BC at a boundary face: given inner (reconstructed) wL, produce ghost wR.
inline void apply_bc(const BC2D& bc, const double wL[4], double nx, double ny,
                     double fx, double fy, double t, double wR[4]) {
    switch (bc.kind) {
    case 1: { // reflective: (u,v) -= 2 (u.n) n
        double un = wL[1]*nx + wL[2]*ny;
        wR[0] = wL[0]; wR[1] = wL[1] - 2*un*nx; wR[2] = wL[2] - 2*un*ny; wR[3] = wL[3];
        break; }
    case 2: for (int k=0;k<4;++k) wR[k] = bc.state[k]; break;
    case 3: bc.func(fx, fy, t, wR); break;
    default: for (int k=0;k<4;++k) wR[k] = wL[k]; break; // transmissive
    }
}

inline void euler2d_rhs(const Mesh& m, const Euler2D& eq,
                        const std::vector<double>& U, std::vector<double>& dUdt,
                        int recon, const ReconCtx* ctx,
                        std::vector<double>& Wc, std::vector<double>& WL, std::vector<double>& WR,
                        int flux = FLUX_LLF,
                        const std::vector<BC2D>* bcs = nullptr, double t = 0.0,
                        const ReconCtx* shock_ctx = nullptr,
                        const ReconCtxO2* ctx_o2 = nullptr) {
    const int N = m.n_cells(), Nf = m.n_faces();
    // cons -> prim
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double u[4] = {U[0*N+i], U[1*N+i], U[2*N+i], U[3*N+i]}, w[4];
        eq.cons_to_prim(u, w);
        Wc[0*N+i]=w[0]; Wc[1*N+i]=w[1]; Wc[2*N+i]=w[2]; Wc[3*N+i]=w[3];
    }
    const bool bvd = (recon == RECON_BVD) && ctx && ctx_o2;
    const bool o2 = (recon == RECON_O2) && ctx_o2;
    const bool ho = bvd || o2 || ((recon != RECON_FIRST) && ctx);
    // Ducros-type shear sensor s = w^2/(w^2+theta^2+eps) in [0,1]: ~1 on shear/
    // vortex (slip line, KH), ~0 on shocks (high dilatation). Used to relax the
    // sharp-candidate limiter on the shear layer only. Gated by SHEAR_RELAX env.
    static const char* SR = std::getenv("SHEAR_RELAX");
    static const double KRELAX = SR ? std::atof(SR) : 0.0;
    static const char* TM = std::getenv("TVB_M");
    static const double TVBM = TM ? std::atof(TM) : 0.0;
    static const char* VK = std::getenv("VENKAT_K");
    static const double VENKATK = VK ? std::atof(VK) : 0.0;
    static const bool MLP_HIER = std::getenv("MLP_HIER") != nullptr;
    std::vector<double> shear;
    if (bvd && KRELAX > 0.0 && ctx) {
        shear.assign(N, 0.0);
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double ru0=0,ru1=0,rv0=0,rv1=0, uc=Wc[1*N+ci], vc=Wc[2*N+ci];
            for (int k = 0; k < ctx->max_nb; ++k) {
                int nb = ctx->nb[(size_t)ci*ctx->max_nb+k]; if (nb < 0) continue;
                double wk = ctx->w[(size_t)ci*ctx->max_nb+k];
                double dx = ctx->d[((size_t)ci*ctx->max_nb+k)*2+0], dy = ctx->d[((size_t)ci*ctx->max_nb+k)*2+1];
                ru0 += wk*dx*(Wc[1*N+nb]-uc); ru1 += wk*dy*(Wc[1*N+nb]-uc);
                rv0 += wk*dx*(Wc[2*N+nb]-vc); rv1 += wk*dy*(Wc[2*N+nb]-vc);
            }
            double dudx = ctx->ATA_inv[ci*4+0]*ru0 + ctx->ATA_inv[ci*4+1]*ru1;
            double dudy = ctx->ATA_inv[ci*4+2]*ru0 + ctx->ATA_inv[ci*4+3]*ru1;
            double dvdx = ctx->ATA_inv[ci*4+0]*rv0 + ctx->ATA_inv[ci*4+1]*rv1;
            double dvdy = ctx->ATA_inv[ci*4+2]*rv0 + ctx->ATA_inv[ci*4+3]*rv1;
            double theta = dudx + dvdy, omega = dvdx - dudy;
            shear[ci] = (omega*omega) / (omega*omega + theta*theta + 1e-30);
        }
    }
    // TMLPU_GATED: genuine linear gated T-MLP-u (LSQ-residual AND gap-bimodality
    // gate -> cicsam compression at discontinuities / van_leer at smooth, uncapped
    // vertex-LMP). Per primitive variable. The LeVeque winner extended to Euler.
    static const bool TG = std::getenv("TMLPU_GATED") != nullptr;
    static const double TG_THR = []{ const char* e=std::getenv("TMLPU_THR"); return e?std::atof(e):0.2; }();
    static const double TG_CO  = []{ const char* e=std::getenv("TMLPU_CO"); return e?std::atof(e):0.38; }();
    static const double TG_CAP = []{ const char* e=std::getenv("TMLPU_CAP"); return e?std::atof(e):2.0; }();
    // Gate ONLY the RECON_BVD slot so the RECON_BJ_VERTEX (mlp_u1) baseline stays
    // pure in same-process comparisons (matches the scalar path in solver2d.hpp).
    if (TG && ctx && recon == RECON_BVD) {
        reconstruct_tmlpu_gated(m, *ctx, Wc, 4, WL, WR, TG_THR, TG_CO, TG_CAP);
    } else if (bvd) {
        reconstruct_bvd(m, *ctx, *ctx_o2, Wc, 4, WL, WR, /*face_bound*/true, /*sel_var density*/0,
                        shear.empty() ? nullptr : shear.data(), KRELAX, TVBM, VENKATK, MLP_HIER);
    } else if (o2) {
        static const bool O2_UNLIM = std::getenv("O2_UNLIMITED") != nullptr;  // pure P2 (3rd-order base test)
        WL.assign((size_t)4*Nf, 0.0); WR.assign((size_t)4*Nf, 0.0);
        std::vector<double> wl, wr;
        for (int v = 0; v < 4; ++v) {
            if (O2_UNLIM) reconstruct_o2_scalar(m, *ctx_o2, Wc, 4, v, wl, wr);
            else          reconstruct_o2_limited(m, *ctx_o2, Wc, 4, v, wl, wr,
                              shear.empty()?nullptr:shear.data(), KRELAX, TVBM, VENKATK, MLP_HIER); // genuine P2 + TVB/Venkat/hier
            for (int f = 0; f < Nf; ++f) { WL[(size_t)v*Nf+f]=wl[f]; WR[(size_t)v*Nf+f]=wr[f]; }
        }
    } else if (ho) reconstruct_bj_vertex(m, *ctx, Wc, 4, WL, WR, recon == RECON_TMLPU_L, shock_ctx);

    // ===== CPU-perf face/cell machinery (2026-07-09, result-preserving) =====
    // (a) Fall layout = FACE-INNERMOST [f*4+v]: one cache line per face for both the Pass-1 write and
    //     the Pass-2 gather (was [v*Nf+f] = 4 strided lines per face). Fall is local to this function
    //     -> pure addressing change, byte-exact.
    // (b) static persistent scratch (H2 pattern): no per-call malloc/zero/first-touch.
    // (c) CSR flatten of m.cell_faces + PRECOMPUTED gather sign (owner -1 / neighbour +1): removes the
    //     vector<vector> pointer chase and the face_owner[f]==i re-check from the Pass-2 hot loop.
    //     Face order per cell is preserved -> identical accumulation order -> byte-exact.
    // (d) Pass-1 loop fission internal/boundary: each face writes only its own Fall slot (no
    //     accumulation), so splitting the loop cannot change any result; removes the n<0 branch +
    //     BC machinery from the (dominant) internal-face loop.
    static const bool bc_cellghost = std::getenv("BC_CELLGHOST") != nullptr;
    static const bool bc_face_recon = std::getenv("BC_FACE_RECON") != nullptr; // OLD (pre-2026-07-03): reconstructed wL at boundary + BC ghost
    struct RhsMeshCache { const Mesh* mp=nullptr; int N=0, Nf=0;
        std::vector<int> off, fid, ifc, bfc; std::vector<double> sgn; };
    static RhsMeshCache MC;
    if (MC.mp != &m || MC.N != N || MC.Nf != Nf) {
        MC.mp=&m; MC.N=N; MC.Nf=Nf;
        MC.off.assign((size_t)N+1,0); MC.fid.clear(); MC.sgn.clear(); MC.ifc.clear(); MC.bfc.clear();
        for (int i = 0; i < N; ++i) {
            MC.off[i]=(int)MC.fid.size();
            for (int f : m.cell_faces[i]) { MC.fid.push_back(f); MC.sgn.push_back(m.face_owner[f]==i ? -1.0 : 1.0); }
        }
        MC.off[N]=(int)MC.fid.size();
        for (int f = 0; f < Nf; ++f) (m.face_neighbour[f] < 0 ? MC.bfc : MC.ifc).push_back(f);
    }
    static std::vector<double> Fall; Fall.assign((size_t)4*Nf, 0.0);   // [f*4+v] face-innermost
    double* __restrict FA = Fall.data();
    const double* __restrict WLp = WL.data(); const double* __restrict WRp = WR.data();
    const double* __restrict Wcp = Wc.data();
    // Pass 1a (parallel, race-free): INTERNAL faces — branch-free hot loop, local scalar caching.
    const int nIF = (int)MC.ifc.size(), nBF = (int)MC.bfc.size();
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < nIF; ++k) {
        const int f = MC.ifc[k];
        const int o = m.face_owner[f], n = m.face_neighbour[f];
        const double nx = m.face_normals[f*2+0], ny = m.face_normals[f*2+1];
        const double area = m.face_areas[f];
        double wL[4], wR[4];
        if (ho) for (int v = 0; v < 4; ++v) { wL[v] = WLp[(size_t)v*Nf+f]; wR[v] = WRp[(size_t)v*Nf+f]; }
        else    for (int v = 0; v < 4; ++v) { wL[v] = Wcp[(size_t)v*N+o];  wR[v] = Wcp[(size_t)v*N+n]; }
        double F[4];
        if (flux == FLUX_RROE)      rotated_roe_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_ROE_EF) roe_ef_face_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_RHLLCM) rotated_hllcm_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_HLLCM) hllcm_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_RHLLC) rotated_hllc_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_RSLAU2) rotated_slau2_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_SLAU2) slau2_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_HLLC) hllc_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_HLL)  hll_euler2d(eq, wL, wR, nx, ny, F);
        else                        llf_euler2d(eq, wL, wR, nx, ny, F);
        double* __restrict Ff = FA + (size_t)f*4;
        for (int v = 0; v < 4; ++v) Ff[v] = F[v]*area;
    }
    // Pass 1b (parallel, race-free): BOUNDARY faces — BC logic isolated here.
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < nBF; ++k) {
        const int f = MC.bfc[k];
        const int o = m.face_owner[f];
        const double nx = m.face_normals[f*2+0], ny = m.face_normals[f*2+1];
        const double area = m.face_areas[f];
        double wL[4], wR[4];
        if (ho) for (int v = 0; v < 4; ++v) wL[v] = WLp[(size_t)v*Nf+f];
        else    for (int v = 0; v < 4; ++v) wL[v] = Wcp[(size_t)v*N+o];
        if (bc_cellghost) {
            // boundary face state = owner cell-average (rho,u,v,p) on BOTH sides (no reconstruction
            // extrapolation): pure cell-value zero-gradient -> flux = F(cell), no overshoot.
            for (int v = 0; v < 4; ++v) { wL[v] = Wcp[(size_t)v*N+o]; wR[v] = wL[v]; }
        } else {
            // HARD rule (2026-07-03, extended): NO reconstruction at ANY boundary face.
            // Interior side = owner CELL AVERAGE; ghost = the BC-assigned values applied
            // to it (Neumann: same; reflective: mirrored cell average; Dirichlet: the
            // prescribed state exactly). Flux = Riemann(cell average, ghost).
            // BC_FACE_RECON: keep the RECONSTRUCTED wL (old 2nd-order boundary) for A/B isolation.
            if (!bc_face_recon) for (int v = 0; v < 4; ++v) wL[v] = Wcp[(size_t)v*N+o];
            int tag = m.face_bc_tag[f];
            const BC2D* bc = (bcs && tag > 0 && tag < (int)bcs->size()) ? &(*bcs)[tag] : nullptr;
            if (bc) apply_bc(*bc, wL, nx, ny, m.face_centers[f*2+0], m.face_centers[f*2+1], t, wR);
            else for (int v = 0; v < 4; ++v) wR[v] = wL[v];
        }
        double F[4];
        if (flux == FLUX_RROE)      rotated_roe_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_ROE_EF) roe_ef_face_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_RHLLCM) rotated_hllcm_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_HLLCM) hllcm_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_RHLLC) rotated_hllc_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_RSLAU2) rotated_slau2_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_SLAU2) slau2_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_HLLC) hllc_euler2d(eq, wL, wR, nx, ny, F);
        else if (flux == FLUX_HLL)  hll_euler2d(eq, wL, wR, nx, ny, F);
        else                        llf_euler2d(eq, wL, wR, nx, ny, F);
        double* __restrict Ff = FA + (size_t)f*4;
        for (int v = 0; v < 4; ++v) Ff[v] = F[v]*area;
    }
    // Pass 2 (parallel, race-free): each cell gathers its own faces via CSR + precomputed sign.
    // Same per-cell face order as m.cell_faces -> identical FP accumulation order (byte-exact).
    const int* __restrict off = MC.off.data(); const int* __restrict fid = MC.fid.data();
    const double* __restrict sgn = MC.sgn.data();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double acc[4] = {0,0,0,0};
        for (int k = off[i]; k < off[i+1]; ++k) {
            const double s = sgn[k];
            const double* __restrict Ff = FA + (size_t)fid[k]*4;
            for (int v = 0; v < 4; ++v) acc[v] += s * Ff[v];
        }
        const double inv = 1.0 / m.cell_volumes[i];
        for (int v = 0; v < 4; ++v) dUdt[(size_t)v*N+i] = acc[v] * inv;
    }
}

inline Solve2DResult solve_euler2d(const Mesh& m, const Euler2D& eq,
                                   const std::vector<double>& U0, double t_end,
                                   int integrator = 2, double cfl = 0.4,
                                   double dt_fixed = -1.0, int max_steps = 1000000,
                                   int recon = RECON_FIRST, const ReconCtx* ctx = nullptr,
                                   int flux = FLUX_LLF, const std::vector<BC2D>* bcs = nullptr,
                                   const ReconCtx* shock_ctx = nullptr,
                                   const ReconCtxO2* ctx_o2 = nullptr,
                                   double gravity_y = 0.0) {
    const int N = m.n_cells(); const int sz = 4 * N;
    std::vector<double> U = U0, L0(sz), U1(sz), L1(sz), U2(sz), L2(sz);
    std::vector<double> Wc(sz), WL, WR;
    double cur_t = 0.0;
    auto RHS = [&](const std::vector<double>& s, std::vector<double>& out) {
        euler2d_rhs(m, eq, s, out, recon, ctx, Wc, WL, WR, flux, bcs, cur_t, shock_ctx, ctx_o2);
        if (gravity_y != 0.0) {   // body force (0, g_y): y-momentum += ρ g_y, energy += ρ v g_y
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                out[(size_t)2*N+i] += s[(size_t)0*N+i] * gravity_y;   // ρ g_y
                out[(size_t)3*N+i] += s[(size_t)2*N+i] * gravity_y;   // (ρv) g_y
            }
        }
    };
    double h_min = 1e300;
    for (int i = 0; i < N; ++i) {
        double amax = 0.0; for (int f : m.cell_faces[i]) amax = std::max(amax, m.face_areas[f]);
        h_min = std::min(h_min, m.cell_volumes[i] / std::max(amax, 1e-30));
    }
    // dt-collapse divergence guard: abort if the CFL dt suddenly drops far below
    // the reference (first) dt, or the wave speed goes non-finite/huge. Threshold
    // overridable via DT_COLLAPSE_FACTOR (default 1e-3 = wmax grew 1000x).
    double dt_ref = -1.0;
    const char* dcf = std::getenv("DT_COLLAPSE_FACTOR");
    const double collapse = dcf ? std::atof(dcf) : 1e-3;
    double t = 0.0; int n = 0;
    if(const char*_e=std::getenv("CFD_MAXSTEP")){int _v=std::atoi(_e); if(_v>0&&_v<max_steps)max_steps=_v;}   // bench cap (per-cell timing)
    for (; n < max_steps && t < t_end; ++n) {
        double dt;
        if (dt_fixed > 0.0) dt = dt_fixed;
        else {
            double wmax = 0.0;
            #pragma omp parallel for reduction(max:wmax)
            for (int i = 0; i < N; ++i) {
                double u[4] = {U[0*N+i],U[1*N+i],U[2*N+i],U[3*N+i]};
                double w = eq.max_wave_speed(u, 1.0, 0.0);
                double w2 = eq.max_wave_speed(u, 0.0, 1.0);
                w = std::max(w, w2); if (w > wmax) wmax = w;
            }
            if (!std::isfinite(wmax) || wmax > 1e12) {
                std::fprintf(stderr, "DIVERGED: non-finite/huge wave speed (%.3e) at step %d t=%.6g\n",
                             wmax, n, t); break;
            }
            dt = cfl * h_min / std::max(wmax, 1e-30);
            if (dt_ref < 0.0) dt_ref = dt;
            else if (dt < collapse * dt_ref) {
                std::fprintf(stderr, "DIVERGED: dt collapsed (%.3e < %.1e*%.3e) at step %d t=%.6g\n",
                             dt, collapse, dt_ref, n, t); break;
            }
        }
        if (t + dt > t_end) dt = t_end - t;
        if (dt <= 0.0) break;
        cur_t = t;
        RHS(U, L0);
        if (integrator == 0) {
            #pragma omp parallel for
            for (int i = 0; i < sz; ++i) U[i] += dt*L0[i];
        } else if (integrator == 1) {
            #pragma omp parallel for
            for (int i = 0; i < sz; ++i) U1[i] = U[i] + dt*L0[i];
            RHS(U1, L1);
            #pragma omp parallel for
            for (int i = 0; i < sz; ++i) U[i] = 0.5*U[i] + 0.5*(U1[i] + dt*L1[i]);
        } else {
            #pragma omp parallel for
            for (int i = 0; i < sz; ++i) U1[i] = U[i] + dt*L0[i];
            RHS(U1, L1);
            #pragma omp parallel for
            for (int i = 0; i < sz; ++i) U2[i] = 0.75*U[i] + 0.25*(U1[i] + dt*L1[i]);
            RHS(U2, L2);
            #pragma omp parallel for
            for (int i = 0; i < sz; ++i) U[i] = (1.0/3.0)*U[i] + (2.0/3.0)*(U2[i] + dt*L2[i]);
        }
        t += dt;
    }
    return Solve2DResult{std::move(U), n, t};
}

} // namespace cfd
