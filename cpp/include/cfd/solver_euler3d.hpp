// cfd/solver_euler3d.hpp — 3D Euler finite-volume time march.
// Reconstruction on primitive variables (first-order or face-neighbour MUSCL),
// LLF/HLL/HLLC/rotated-HLLC flux, SSP-RK2/3, transmissive/reflective/dirichlet
// BC. 3D mirror of solver_euler2d.hpp. State U is 5*N (var-major: rho, rho u,
// rho v, rho w, rho E).
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/reconstruct3d_bvd_euler.hpp"
#include "cfd/reconstruct3d_bvd_euler_mlp.hpp"
#include "cfd/reconstruct3d_o2_unstr.hpp"
#include "cfd/reconstruct3d_unstr.hpp"
#include "cfd/reconstruct3d_tmlpu.hpp"
#include "cfd/viscous3d.hpp"
#include "cfd/io_vtk.hpp"   // BVD_CANDFLAG diagnostic buffer (mood_level_flag)
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cstdio>
#include <chrono>

namespace cfd {

// ── coarse phase profiler (env E3D_PROF): attribute RHS wall to prim/recon/flux/asm ──
// Phase boundaries in euler3d_rhs are SINGLE-THREADED (the omp-parallel is inside each
// phase), so plain accumulators are race-free. Printed by solve_euler3d after the run.
struct E3DProf { double prim=0, recon=0, ppfloor=0, flux=0, visc=0, asmb=0, mood=0; long ncall=0; };
inline E3DProf& e3d_prof(){ static E3DProf p; return p; }
inline bool e3d_prof_on(){ static const bool b = std::getenv("E3D_PROF")!=nullptr; return b; }
inline double e3d_ms(){ return std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Reconstruction modes (mirror solver2d.hpp ReconMode for the 3D path).
// RECON3_BVD = the Stage-3a EULER deg3t-BVD (o2-quad smooth candidate + 2 THINC,
// min-TBV, no clamping; needs a ReconCtx3DO2). 4 to leave room for future modes.
enum ReconMode3D { RECON3_FIRST = 0, RECON3_BJ_VERTEX = 1, RECON3_BVD = 4 };

enum FluxKind3D { FLUX3_LLF = 0, FLUX3_HLLC = 1, FLUX3_RHLLC = 2, FLUX3_RROE = 3, FLUX3_HLL = 4, FLUX3_SLAU2 = 5, FLUX3_RSLAU2 = 6 };

// Boundary condition per patch tag. kind: 0 transmissive, 1 reflective (slip
// wall), 2 dirichlet (fixed W), 3 dirichlet_func (W = f(x,y,z,t)).
struct BC3D {
    int kind = 0;
    double state[5] = {0,0,0,0,0};
    std::function<void(double, double, double, double, double*)> func;
};

// Apply BC at a boundary face: given inner (reconstructed) wL, produce ghost wR.
inline void apply_bc3d(const BC3D& bc, const double wL[5], double nx, double ny, double nz,
                       double fx, double fy, double fz, double t, double wR[5]) {
    switch (bc.kind) {
    case 1: { // reflective: (u,v,w) -= 2 (u.n) n
        double un = wL[1]*nx + wL[2]*ny + wL[3]*nz;
        wR[0] = wL[0];
        wR[1] = wL[1] - 2*un*nx; wR[2] = wL[2] - 2*un*ny; wR[3] = wL[3] - 2*un*nz;
        wR[4] = wL[4];
        break; }
    case 2: for (int k=0;k<5;++k) wR[k] = bc.state[k]; break;
    case 3: bc.func(fx, fy, fz, t, wR); break;
    case 4: { // no-slip adiabatic wall: velocity -> 0 AT the wall face (ghost = -inner),
              // adiabatic ∂p/∂n=∂T/∂n=0 (rho,p mirrored). (Contrast kind 1 slip wall
              // which only flips the normal velocity component.) The viscous face
              // gradient then sees ∂u/∂n ≠ 0 → the boundary layer.
        wR[0] = wL[0];
        wR[1] = -wL[1]; wR[2] = -wL[2]; wR[3] = -wL[3];
        wR[4] = wL[4];
        break; }
    default: for (int k=0;k<5;++k) wR[k] = wL[k]; break; // transmissive
    }
}

// Evaluate the (area-weighted) numerical flux for one face given its two PRIMITIVE
// face states wL/wR. Boundary faces (n<0) take the ghost from apply_bc3d. The flux
// MATH is untouched — this only dispatches the existing flux functions. Shared by
// the normal RHS Pass-1 and the MOOD first-order face recompute (the cascade).
inline void euler3d_face_flux(const Mesh& m, const Euler3D& eq, int f,
                              const double wL_in[5], const double wR_in[5],
                              int flux, const std::vector<BC3D>* bcs, double t,
                              double Fout[5], const double* Wcavg=nullptr, int Ncell=0) {
    int n = m.face_neighbour[f];
    double nx = m.face_normals[(size_t)f*3+0], ny = m.face_normals[(size_t)f*3+1], nz = m.face_normals[(size_t)f*3+2];
    double area = m.face_areas[f];
    double wL[5], wR[5];
    for (int v = 0; v < 5; ++v) { wL[v] = wL_in[v]; wR[v] = wR_in[v]; }
    if (n < 0) {
        // NO recon at ANY boundary face (2026-07-03): interior side = owner cell average,
        // ghost = BC-assigned values (Dirichlet exact, reflective mirrored cell average).
        // BC_FACE_RECON: keep the reconstructed wL_in at boundary (old 2nd-order boundary).
        static const bool bc_face_recon = std::getenv("BC_FACE_RECON") != nullptr;
        if (Wcavg && !bc_face_recon) { int o = m.face_owner[f];
            for (int v = 0; v < 5; ++v) wL[v] = Wcavg[(size_t)v*Ncell+o]; }
        int tag = m.face_bc_tag[f];
        const BC3D* bc = (bcs && tag > 0 && tag < (int)bcs->size()) ? &(*bcs)[tag] : nullptr;
        if (bc) apply_bc3d(*bc, wL, nx, ny, nz,
                       m.face_centers[(size_t)f*3+0], m.face_centers[(size_t)f*3+1], m.face_centers[(size_t)f*3+2], t, wR);
        else for (int v = 0; v < 5; ++v) wR[v] = wL[v];
    }
    double F[5];
    if (flux == FLUX3_RHLLC)     rotated_hllc_euler3d(eq, wL, wR, nx, ny, nz, F);
    else if (flux == FLUX3_RROE) rotated_roe_euler3d(eq, wL, wR, nx, ny, nz, F);
    else if (flux == FLUX3_SLAU2)  slau2_euler3d(eq, wL, wR, nx, ny, nz, F);
    else if (flux == FLUX3_RSLAU2) rotated_slau2_euler3d(eq, wL, wR, nx, ny, nz, F);
    else if (flux == FLUX3_HLLC) hllc_euler3d(eq, wL, wR, nx, ny, nz, F);
    else if (flux == FLUX3_HLL)  hll_euler3d(eq, wL, wR, nx, ny, nz, F);
    else                         llf_euler3d(eq, wL, wR, nx, ny, nz, F);
    for (int v = 0; v < 5; ++v) Fout[v] = F[v]*area;
}

// Physical Admissibility Detection for one cell's candidate conservative state.
// Admissible iff rho>1e-12 AND p>1e-12 AND all five components finite.
inline bool euler3d_pad_ok(const Euler3D& eq, const double Ucell[5]) {
    for (int v = 0; v < 5; ++v) if (!std::isfinite(Ucell[v])) return false;
    double w[5]; eq.cons_to_prim(Ucell, w);
    if (!std::isfinite(w[0]) || !std::isfinite(w[4])) return false;
    return (w[0] > 1e-12) && (w[4] > 1e-12);
}

inline void euler3d_rhs(const Mesh& m, const Euler3D& eq,
                        const std::vector<double>& U, std::vector<double>& dUdt,
                        int recon, const ReconCtx3D* ctx,
                        std::vector<double>& Wc, std::vector<double>& WL, std::vector<double>& WR,
                        int flux = FLUX3_LLF,
                        const std::vector<BC3D>* bcs = nullptr, double t = 0.0,
                        const ReconCtx3DO2* o2ctx = nullptr,
                        const ViscousParams* visc = nullptr) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const bool viscous = (visc != nullptr) && (visc->mu > 0.0) && ctx;
    const bool prof = e3d_prof_on(); double _t = prof ? e3d_ms() : 0.0;
    if (prof) e3d_prof().ncall++;
    // cons -> prim
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double u[5] = {U[0*N+i], U[1*N+i], U[2*N+i], U[3*N+i], U[4*N+i]}, w[5];
        eq.cons_to_prim(u, w);
        Wc[0*N+i]=w[0]; Wc[1*N+i]=w[1]; Wc[2*N+i]=w[2]; Wc[3*N+i]=w[3]; Wc[4*N+i]=w[4];
    }
    if (prof) { e3d_prof().prim += e3d_ms()-_t; _t = e3d_ms(); }
    // High-order reconstruction on PRIMITIVE variables. RECON3_BVD: when VISCOUS,
    // dispatch the MLP-LIMITED Euler deg3t-BVD (shock-stable, no MOOD); the inviscid
    // default keeps the Stage-3a unlimited BVD. Else MUSCL (BJ vertex).
    const bool bvd = (recon == RECON3_BVD) && o2ctx;
    const bool ho  = bvd || ((recon != RECON3_FIRST) && ctx);
    // DEFAULT = MLP-limited BVD (vertex-26 bounded smooth candidate, ψ∈[0,1]). The
    // legacy unlimited o2-quad candidate has NO slope limiter and over-sharpens strong
    // shocks into vacuum (Langseth diverges). Opt out to it via BVD_UNLIMITED (kept only
    // to reproduce that divergence); VISCOUS always uses MLP.
    static const bool bvd_unlim = (std::getenv("BVD_UNLIMITED") != nullptr);
    const bool unstr3d = (m.s3_nx == 0);   // build_unstructured_3d leaves s3_nx=0
    static const bool TMLPU_GATED_E1 = std::getenv("TMLPU_GATED") != nullptr;
    if (bvd && unstr3d && TMLPU_GATED_E1) reconstruct_tmlpu_gated_3d(m, Wc, 5, WL, WR);
    else if (bvd && unstr3d) reconstruct3d_bvd_gauss_unstr(m, *o2ctx, Wc, 5, WL, WR);
    else if (bvd) {
        if (bvd_unlim && !viscous) reconstruct3d_bvd_euler(m, *o2ctx, Wc, 5, WL, WR);
        else                       reconstruct3d_bvd_euler_mlp(m, *o2ctx, Wc, 5, WL, WR);
    }
    else if (ho)  reconstruct_bj_vertex_3d(m, *ctx, Wc, 5, WL, WR);
    if (prof) { e3d_prof().recon += e3d_ms()-_t; _t = e3d_ms(); }

    // PP-floor (positivity, a-priori): clamp face rho (var 0) & p (var 4) to a fraction
    // of the OWNING cell primitive so THINC over-sharpening cannot drive faces to vacuum.
    // The 3D BVD path otherwise lacks the 2D BVD_PPFLOOR; this is the 3D mirror. Env
    // BVD_PPFLOOR=<frac> (e.g. 0.2); unset/0 = off. cell-avg>0 => floored faces>0 =>
    // RHLLC/HLLC + CFL keep the cell-avg positive (Zhang-Shu). Fixes Langseth/octant/RMI
    // THINC->vacuum divergence without MOOD's recompute cost.
    static const double ppf3 = std::getenv("BVD_PPFLOOR") ? std::atof(std::getenv("BVD_PPFLOOR")) : 0.0;
    // gate on `ho` (not just `bvd`) so the mlp_u1 BASE (RECON3_BJ_VERTEX) can get the SAME
    // a-priori positivity floor as CZL for a FAIR 3D Euler comparison (BVD_PPFLOOR=0.1).
    // ho ⊇ bvd, so the BVD path is unchanged; unset BVD_PPFLOOR => ppf3=0 => no-op.
    if (ho && ppf3 > 0.0) {
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            int o = m.face_owner[f], nb = m.face_neighbour[f];
            double rL = ppf3*Wc[(size_t)0*N+o], pL = ppf3*Wc[(size_t)4*N+o];
            if (WL[(size_t)0*Nf+f] < rL) WL[(size_t)0*Nf+f] = rL;
            if (WL[(size_t)4*Nf+f] < pL) WL[(size_t)4*Nf+f] = pL;
            if (nb >= 0) {
                double rR = ppf3*Wc[(size_t)0*N+nb], pR = ppf3*Wc[(size_t)4*N+nb];
                if (WR[(size_t)0*Nf+f] < rR) WR[(size_t)0*Nf+f] = rR;
                if (WR[(size_t)4*Nf+f] < pR) WR[(size_t)4*Nf+f] = pR;
            }
        }
    }

    if (prof) { e3d_prof().ppfloor += e3d_ms()-_t; _t = e3d_ms(); }
    // Pass 1 (parallel, race-free): area-weighted CONVECTIVE flux per face.
    std::vector<double> Fall((size_t)5*Nf);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double nx = m.face_normals[(size_t)f*3+0], ny = m.face_normals[(size_t)f*3+1], nz = m.face_normals[(size_t)f*3+2];
        double area = m.face_areas[f];
        double wL[5], wR[5];
        for (int v = 0; v < 5; ++v) {
            if (ho) { wL[v] = WL[(size_t)v*Nf+f]; if (n>=0) wR[v] = WR[(size_t)v*Nf+f]; }
            else    { wL[v] = Wc[(size_t)v*N+o]; if (n>=0) wR[v] = Wc[(size_t)v*N+n]; }
        }
        if (n < 0) {
            // NO recon at ANY boundary face (2026-07-03): cell avg + BC ghost
            // BC_FACE_RECON: keep reconstructed wL at boundary (old 2nd-order boundary).
            static const bool bc_face_recon = std::getenv("BC_FACE_RECON") != nullptr;
            if (!bc_face_recon) for (int v = 0; v < 5; ++v) wL[v] = Wc[(size_t)v*N+o];
            int tag = m.face_bc_tag[f];
            const BC3D* bc = (bcs && tag > 0 && tag < (int)bcs->size()) ? &(*bcs)[tag] : nullptr;
            if (bc) apply_bc3d(*bc, wL, nx, ny, nz,
                           m.face_centers[(size_t)f*3+0], m.face_centers[(size_t)f*3+1], m.face_centers[(size_t)f*3+2], t, wR);
            else for (int v = 0; v < 5; ++v) wR[v] = wL[v];
        }
        double F[5];
        if (flux == FLUX3_RHLLC)     rotated_hllc_euler3d(eq, wL, wR, nx, ny, nz, F);
        else if (flux == FLUX3_RROE) rotated_roe_euler3d(eq, wL, wR, nx, ny, nz, F);
        else if (flux == FLUX3_SLAU2)  slau2_euler3d(eq, wL, wR, nx, ny, nz, F);
        else if (flux == FLUX3_RSLAU2) rotated_slau2_euler3d(eq, wL, wR, nx, ny, nz, F);
        else if (flux == FLUX3_HLLC) hllc_euler3d(eq, wL, wR, nx, ny, nz, F);
        else if (flux == FLUX3_HLL)  hll_euler3d(eq, wL, wR, nx, ny, nz, F);
        else                         llf_euler3d(eq, wL, wR, nx, ny, nz, F);
        for (int v = 0; v < 5; ++v) Fall[(size_t)v*Nf+f] = F[v]*area;
    }
    if (prof) { e3d_prof().flux += e3d_ms()-_t; _t = e3d_ms(); }
    // Pass 1b (viscous, NS): add −G_visc·area so Fall = (F_conv − G_visc)·area. Uses
    // FIRST-ORDER cell-average primitives (Wc) for the gradients (the diffusive flux
    // is a centred physics term, NOT the convective reconstruction). Gated: inviscid
    // path never enters here.
    if (viscous) {
        // boundary ghost closure bound to the solver BCs (wall / slip / etc).
        auto bc_ghost = [&](int f, const double wL[5], double wR[5]) {
            double nx = m.face_normals[(size_t)f*3+0], ny = m.face_normals[(size_t)f*3+1], nz = m.face_normals[(size_t)f*3+2];
            int tag = m.face_bc_tag[f];
            if (bcs && tag > 0 && tag < (int)bcs->size())
                apply_bc3d((*bcs)[tag], wL, nx, ny, nz,
                           m.face_centers[(size_t)f*3+0], m.face_centers[(size_t)f*3+1], m.face_centers[(size_t)f*3+2], t, wR);
            else for (int v = 0; v < 5; ++v) wR[v] = wL[v];
        };
        // DEFAULT (unstructured): skewness/non-orthogonality-corrected P2@face viscous gradient
        // (2nd-order on distorted meshes). Opt-out VISC_CENTROID for the legacy centroid scheme.
        static const bool visc_p2face = (std::getenv("VISC_CENTROID") == nullptr);
        if (unstr3d && o2ctx && visc_p2face) {
            std::vector<double> cu, cv, cw, cT;
            viscous3d_cell_coeffs_o2(m, *o2ctx, Wc, visc->R, cu, cv, cw, cT);
            viscous3d_add_face_flux_p2face(m, eq, *visc, Wc, cu, cv, cw, cT, bc_ghost, Fall);
        } else {
            std::vector<double> gu, gv, gw, gT;
            if (unstr3d && o2ctx) viscous3d_cell_gradients_o2(m, *o2ctx, Wc, visc->R, gu, gv, gw, gT);
            else                  viscous3d_cell_gradients(m, eq, *ctx, Wc, visc->R, gu, gv, gw, gT);
            viscous3d_add_face_flux(m, eq, *visc, Wc, gu, gv, gw, gT, bc_ghost, Fall);
        }
    }
    if (prof) { e3d_prof().visc += e3d_ms()-_t; _t = e3d_ms(); }
    // Pass 2 (parallel, race-free): each cell gathers its own faces (sign by owner/neighbour).
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double acc[5] = {0,0,0,0,0};
        for (int f : m.cell_faces[i]) {
            double s = (m.face_owner[f] == i) ? -1.0 : 1.0;  // owner -F, neighbour +F
            for (int v = 0; v < 5; ++v) acc[v] += s * Fall[(size_t)v*Nf+f];
        }
        double inv = 1.0 / m.cell_volumes[i];
        for (int v = 0; v < 5; ++v) dUdt[(size_t)v*N+i] = acc[v] * inv;
    }
    if (prof) e3d_prof().asmb += e3d_ms()-_t;
}

// ── a-posteriori MOOD positivity protection (Stage 3b) ─────────────────────────
//
// Wraps ONE forward-Euler sub-update U_new = U + dt*L(U) so U_new is admissible
// everywhere (rho>0, p>0, finite) or first-order-floored. The flux MATH is never
// changed; troubled FACES are re-evaluated with FIRST-ORDER reconstruction (the
// cell-average primitives Wc), which is positivity-preserving with RHLLC under CFL.
// A troubled face's first-order flux is used by BOTH adjacent cells → the flux
// stays SINGLE-VALUED → conservation is preserved exactly.
//
// Cascade (face-level):
//   0. high-order Fall (Pass-1) + per-face WL/WR retained.
//   1. candidate dUdt (Pass-2) and U_new = U + dt*dUdt.
//   2. PAD per cell → troubled set.
//   3. while troubled (≤ max_iter): troubled faces = faces with ≥1 troubled endpoint;
//      recompute each with first-order (Wc[owner],Wc[neighbour]); re-gather the cells
//      touching a troubled face; re-run PAD on just those cells.
//   4. a cell whose faces are already all first-order is left (first-order is the floor).
//
// Worklist-based: cascade iterations only scan the small set of cells touching a
// troubled face, never a full-N sweep (the initial PAD in step 2 is the only full-N pass).
struct MoodStats { long troubled_total = 0; int troubled_max = 0; long floored = 0; };

inline void euler3d_rhs_mood(const Mesh& m, const Euler3D& eq,
                             const std::vector<double>& U, std::vector<double>& dUdt,
                             double dt, int recon, const ReconCtx3D* ctx,
                             std::vector<double>& Wc, std::vector<double>& WL, std::vector<double>& WR,
                             int flux, const std::vector<BC3D>* bcs, double t,
                             const ReconCtx3DO2* o2ctx, MoodStats* stats = nullptr,
                             int max_iter = 3, const ViscousParams* visc = nullptr) {
    const int N = m.n_cells(), Nf = m.n_faces();
    // coarse phase profiler (env E3D_PROF), same accounting as euler3d_rhs. Phase
    // boundaries here are SINGLE-THREADED (every omp-parallel is inside a phase), so the
    // plain accumulators are race-free. Zero cost when E3D_PROF is unset.
    const bool prof = e3d_prof_on(); double _t = prof ? e3d_ms() : 0.0;
    if (prof) e3d_prof().ncall++;
    // cons -> prim (Wc are the first-order cell-average primitives used by the cascade).
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double u[5] = {U[0*N+i], U[1*N+i], U[2*N+i], U[3*N+i], U[4*N+i]}, w[5];
        eq.cons_to_prim(u, w);
        Wc[0*N+i]=w[0]; Wc[1*N+i]=w[1]; Wc[2*N+i]=w[2]; Wc[3*N+i]=w[3]; Wc[4*N+i]=w[4];
    }
    if (prof) { e3d_prof().prim += e3d_ms()-_t; _t = e3d_ms(); }
    // high-order reconstruction. DEFAULT = MLP-limited (vertex-26 bounded, ψ∈[0,1])
    // Euler BVD even inside the MOOD loop — the UNLIMITED o2-quad smooth candidate has
    // no slope limiter and over-sharpens strong-shock-into-low-density (Langseth) faster
    // than the a-posteriori cascade can floor it. The bounded candidate is the a-priori
    // fix; MOOD then only backstops the rare residual. BVD_UNLIMITED → legacy unlimited.
    static const bool mood_unlim = (std::getenv("BVD_UNLIMITED") != nullptr);
    const bool bvd = (recon == RECON3_BVD) && o2ctx;
    const bool ho  = bvd || ((recon != RECON3_FIRST) && ctx);
    const bool unstr3d = (m.s3_nx == 0);   // unstructured mesh
    static const bool TMLPU_GATED_E2 = std::getenv("TMLPU_GATED") != nullptr;
    if (bvd && unstr3d && TMLPU_GATED_E2) reconstruct_tmlpu_gated_3d(m, Wc, 5, WL, WR);
    else if (bvd && unstr3d) reconstruct3d_bvd_gauss_unstr(m, *o2ctx, Wc, 5, WL, WR);
    else if (bvd) {
        if (mood_unlim) reconstruct3d_bvd_euler(m, *o2ctx, Wc, 5, WL, WR);
        else            reconstruct3d_bvd_euler_mlp(m, *o2ctx, Wc, 5, WL, WR);
    }
    else if (ho)  reconstruct_bj_vertex_3d(m, *ctx, Wc, 5, WL, WR);

    // MUSCL candidate for the GRADED cascade (THINC→MUSCL→1st): a troubled THINC face
    // first drops to MUSCL (2nd-order, clean) and only to 1st-order if MUSCL is also bad
    // — far less diffusive than straight-to-1st (Diot 2012 degree-decrement cascade).
    // The graded MUSCL intermediate HURTS THINC/BVD over-sharpening (sub-cell binary needs
    // the aggressive 1st-order floor; the gentle MUSCL level fails to suppress the THINC
    // oscillation — measured: octant ens 67→143, Langseth blow-up). Diot's degree-decrement
    // is for polynomial Gibbs, not BVD. So DEFAULT = straight-to-1st (have_muscl=false);
    // opt-in MOOD_GRADED for the THINC→MUSCL→1st cascade (experimental).
    std::vector<double> WLmu, WRmu;
    const bool have_muscl = bvd && ctx && (std::getenv("MOOD_GRADED") != nullptr);
    if (have_muscl) reconstruct_bj_vertex_3d(m, *ctx, Wc, 5, WLmu, WRmu);
    if (prof) { e3d_prof().recon += e3d_ms()-_t; _t = e3d_ms(); }

    // Pass 1: high-order area-weighted flux per face (keep Fall).
    std::vector<double> Fall((size_t)5*Nf);
    // GRADED face level: 2=THINC(WL/WR), 1=MUSCL(WLmu/WRmu), 0=first-order(Wc). Init 2.
    std::vector<unsigned char> face_level((size_t)Nf, 2);
    // recompute one face's flux into Fall at its CURRENT level.
    auto recompute_face = [&](int f) {
        int o = m.face_owner[f], n = m.face_neighbour[f]; int lv = face_level[f];
        double wL[5], wR[5];
        for (int v = 0; v < 5; ++v) {
            if (lv >= 2)                   { wL[v]=WL[(size_t)v*Nf+f];   wR[v]=(n>=0)?WR[(size_t)v*Nf+f]:0.0; }
            else if (lv == 1 && have_muscl){ wL[v]=WLmu[(size_t)v*Nf+f]; wR[v]=(n>=0)?WRmu[(size_t)v*Nf+f]:0.0; }
            else                           { wL[v]=Wc[(size_t)v*N+o];    wR[v]=(n>=0)?Wc[(size_t)v*N+n]:0.0; }
        }
        // [FIX 2026-06-30] the FIRST-ORDER floor (lv==0) uses LLF (Rusanov, positivity-preserving
        // under CFL) regardless of the configured flux. MOOD's positivity guarantee assumes a PP
        // flux at first-order; non-PP fluxes (SLAU2) re-blow-up at first-order otherwise (rho 7.6e24).
        int fk = (lv == 0) ? (int)FLUX3_LLF : flux;
        double F[5]; euler3d_face_flux(m, eq, f, wL, wR, fk, bcs, t, F, Wc.data(), N);
        for (int v = 0; v < 5; ++v) Fall[(size_t)v*Nf+f] = F[v];
    };
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double wL[5], wR[5];
        for (int v = 0; v < 5; ++v) {
            if (ho) { wL[v] = WL[(size_t)v*Nf+f]; wR[v] = (n>=0) ? WR[(size_t)v*Nf+f] : 0.0; }
            else    { wL[v] = Wc[(size_t)v*N+o]; wR[v] = (n>=0) ? Wc[(size_t)v*N+n] : 0.0; }
        }
        double F[5];
        euler3d_face_flux(m, eq, f, wL, wR, flux, bcs, t, F, Wc.data(), N);
        for (int v = 0; v < 5; ++v) Fall[(size_t)v*Nf+f] = F[v];
    }
    if (prof) { e3d_prof().flux += e3d_ms()-_t; _t = e3d_ms(); }
    // Pass 2: candidate dUdt for every cell.
    auto gather_cell = [&](int i) {
        double acc[5] = {0,0,0,0,0};
        for (int f : m.cell_faces[i]) {
            double s = (m.face_owner[f] == i) ? -1.0 : 1.0;
            for (int v = 0; v < 5; ++v) acc[v] += s * Fall[(size_t)v*Nf+f];
        }
        double inv = 1.0 / m.cell_volumes[i];
        for (int v = 0; v < 5; ++v) dUdt[(size_t)v*N+i] = acc[v] * inv;
    };
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) gather_cell(i);
    if (prof) { e3d_prof().asmb += e3d_ms()-_t; _t = e3d_ms(); }

    // ── MOOD detection upgrade (relaxed-DMP + u2 curvature), density-driven, on the
    //    high-order BVD path. PAD alone is a LAGGING positivity detector (a positive
    //    overshoot grows until p<0); the relaxed-DMP is a LEADING overshoot detector
    //    (Jiang 2018 Eq.16/17) and the u2 curvature test (Diot 2012 Def.3.4) spares
    //    genuine smooth extrema (vortex peaks) so DMP adds no diffusion. Opt-out MOOD_NODMP.
    static const bool no_dmp = std::getenv("MOOD_NODMP") != nullptr;
    const bool use_dmp = bvd && o2ctx && !no_dmp;
    // u2 curvature-magnitude tolerance: canonical Diot 2012 (Comput.Fluids 64) Eq.10 uses a
    // MESH-DEPENDENT ε_i = (Δx_i)^{1/(2m)} = h^{1/6} (3D) so |X_min|/|X_max| ≥ 1−ε_i (more lenient
    // on coarse cells / under-resolved smooth extrema, → DMP as h→0). We previously hard-coded
    // ε=1/2 (sm≥0.5·am). DEFAULT now the canonical mesh-dependent form; opt-out MOOD_U2_FIXED.
    static const bool u2_meshtol = (std::getenv("MOOD_U2_FIXED") == nullptr);
    std::vector<double> rhomin, rhomax, deltad;
    std::vector<unsigned char> u2smooth;
    if (use_dmp) {
        rhomin.assign((size_t)N, 0.0); rhomax.assign((size_t)N, 0.0);
        deltad.assign((size_t)N, 0.0); u2smooth.assign((size_t)N, 0);
        // density P2 Hessian per cell (reuse the o2 LSQ): g_rho[ci*9+{3,4,5}]=hxx,hyy,hzz.
        std::vector<double> g_rho;
        reconstruct3d_o2_coeffs(m, *o2ctx, Wc, 5, 0, g_rho);
        const int mnb = o2ctx->max_nb;
        // DMP min/max stencil. CANONICAL MOOD (GP-MOOD Eq.22; Diot 2011/13; Dumbser 2014) uses the
        // COMPACT FACE-NEIGHBOUR set. We had the WIDE vertex node-ring (~70 cells on a tet): a gradual
        // SPREADING overshoot then hides inside its own inflated stencil max (rhomax rises with the blob)
        // -> the relative DMP never trips -> divergence. Compact = the overshoot must outrun its IMMEDIATE
        // face neighbours every step (a spreading front can't) -> caught -> demoted to 1st-order/LLF.
        // The u2 curvature (Hessian min/max) stays on the node-ring. Opt back to wide with MOOD_DMP_WIDE.
        static const bool dmp_wide = std::getenv("MOOD_DMP_WIDE") != nullptr;
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double rc = Wc[(size_t)0*N+i];                 // rho^n at cell i (sub-step start)
            double rmn = rc, rmx = rc;
            // interface NORMAL from the density gradient (P2 coeffs 0,1,2) for the rotation-invariant u2:
            // the old u2 tested only the AXIS-diagonal 2nd derivatives (hxx,hyy,hzz), blind to off-diagonal
            // curvature (hxy,hxz,hyz) -> on a TET whose interface normal is a cube-diagonal it mislabeled
            // the smeared interface "smooth" and un-limited it. The NORMAL curvature κ=n̂ᵀHn̂ (full Hessian,
            // incl off-diagonal) is what the interface actually sees -> canonical per-direction test on κ.
            double gx=g_rho[(size_t)i*9+0], gy=g_rho[(size_t)i*9+1], gz=g_rho[(size_t)i*9+2];
            double gn=std::sqrt(gx*gx+gy*gy+gz*gz); bool have_n=(gn>1e-30);
            double nx=have_n?gx/gn:0, ny=have_n?gy/gn:0, nz=have_n?gz/gn:0;
            auto kappa=[&](const double* h)->double{
                return nx*nx*h[3]+ny*ny*h[4]+nz*nz*h[5]+2*nx*ny*h[6]+2*nx*nz*h[7]+2*ny*nz*h[8]; };
            double ki=have_n?kappa(&g_rho[(size_t)i*9]):0.0, kmn=ki, kmx=ki;
            for (int k = 0; k < mnb; ++k) {
                int nb = o2ctx->nb[(size_t)i*mnb+k]; if (nb < 0) continue;
                if (dmp_wide) { double rn = Wc[(size_t)0*N+nb]; if (rn < rmn) rmn = rn; if (rn > rmx) rmx = rn; }
                if (have_n) { double kn=kappa(&g_rho[(size_t)nb*9]); if(kn<kmn)kmn=kn; if(kn>kmx)kmx=kn; }
            }
            if (!dmp_wide) {   // COMPACT: rho min/max over FACE-neighbours only (canonical DMP)
                for (int fc : m.cell_faces[i]) {
                    int nb = (m.face_owner[fc]==i) ? m.face_neighbour[fc] : m.face_owner[fc];
                    if (nb < 0) continue;
                    double rn = Wc[(size_t)0*N+nb]; if (rn < rmn) rmn = rn; if (rn > rmx) rmx = rn;
                }
            }
            rhomin[i]=rmn; rhomax[i]=rmx;
            double range = rmx - rmn;
            deltad[i] = std::max(1e-4*std::fabs(rc), 1e-3*range);   // δ: abs (dimensional) ∨ relative
            double du2 = std::cbrt(m.cell_volumes[i]);              // δ_u2 ~ local cell size h
            if (range < du2*du2*du2) { u2smooth[i] = 1; }           // plateau (near-uniform) → smooth
            else if (!have_n) { u2smooth[i] = 0; }                  // no normal → can't assess → strict
            else {
                bool smooth;
                if (kmx*kmn <= -du2) smooth=false;                 // curvature sign flip → Gibbs oscillation
                else { double am=std::max(std::fabs(kmx),std::fabs(kmn));
                       if (am < du2) smooth=true;                   // ~flat
                       else { double sm=std::min(std::fabs(kmx),std::fabs(kmn));
                              double tol=u2_meshtol?(1.0-std::pow(du2,1.0/6.0)):0.5;  // Diot2012 ε=h^{1/6}
                              smooth = sm >= tol*am; } }            // κ nearly uniform → smooth extremum
                u2smooth[i] = smooth ? 1 : 0;
            }
        }
    }
    // PRESSURE DMP (robustness, mirrors the density DMP). PAD only flags p<=0 (LAGGING);
    // the density DMP is BLIND to a spurious pressure collapse on a uniform-p field (e.g.
    // THINC over-sharpening on slip lines drives p far below the stencil min while density
    // stays in-bounds — the octant Case5 corruption). A relaxed DMP + u2 curvature on
    // pressure catches it while sparing genuine smooth vortex-core drops. Opt-out MOOD_NOPDMP.
    static const bool no_pdmp = std::getenv("MOOD_NOPDMP") != nullptr;
    const bool use_pdmp = use_dmp && !no_pdmp;
    std::vector<double> pmind, pmaxd, deltap; std::vector<unsigned char> p2smooth;
    if (use_pdmp) {
        pmind.assign((size_t)N,0.0); pmaxd.assign((size_t)N,0.0); deltap.assign((size_t)N,0.0); p2smooth.assign((size_t)N,0);
        std::vector<double> g_p;
        reconstruct3d_o2_coeffs(m, *o2ctx, Wc, 5, 4, g_p);     // pressure P2 (Wc var 4)
        const int mnb = o2ctx->max_nb;
        static const bool dmp_wide = std::getenv("MOOD_DMP_WIDE") != nullptr;   // compact face-neighbour DMP (canonical); see density DMP
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double pc = Wc[(size_t)4*N+i]; double pmn=pc, pmx=pc;
            double gx=g_p[(size_t)i*9+0],gy=g_p[(size_t)i*9+1],gz=g_p[(size_t)i*9+2];
            double gn=std::sqrt(gx*gx+gy*gy+gz*gz); bool have_n=(gn>1e-30);
            double nx=have_n?gx/gn:0, ny=have_n?gy/gn:0, nz=have_n?gz/gn:0;
            auto kappa=[&](const double* h)->double{ return nx*nx*h[3]+ny*ny*h[4]+nz*nz*h[5]+2*nx*ny*h[6]+2*nx*nz*h[7]+2*ny*nz*h[8]; };
            double ki=have_n?kappa(&g_p[(size_t)i*9]):0.0, kmn=ki, kmx=ki;
            for (int k=0;k<mnb;++k){ int nb=o2ctx->nb[(size_t)i*mnb+k]; if(nb<0)continue;
                if(dmp_wide){ double pn=Wc[(size_t)4*N+nb]; if(pn<pmn)pmn=pn; if(pn>pmx)pmx=pn; }
                if(have_n){ double kn=kappa(&g_p[(size_t)nb*9]); if(kn<kmn)kmn=kn; if(kn>kmx)kmx=kn; } }
            if(!dmp_wide){ for(int fc:m.cell_faces[i]){ int nb=(m.face_owner[fc]==i)?m.face_neighbour[fc]:m.face_owner[fc];
                if(nb<0)continue; double pn=Wc[(size_t)4*N+nb]; if(pn<pmn)pmn=pn; if(pn>pmx)pmx=pn; } }
            pmind[i]=pmn; pmaxd[i]=pmx; double range=pmx-pmn;
            deltap[i]=std::max(1e-4*std::fabs(pc), 1e-3*range);
            double du2=std::cbrt(m.cell_volumes[i]);
            if (range < du2*du2*du2) { p2smooth[i]=1; }
            else if (!have_n) { p2smooth[i]=0; }
            else { bool smooth;
                   if (kmx*kmn<=-du2) smooth=false;
                   else { double am=std::max(std::fabs(kmx),std::fabs(kmn));
                          if(am<du2) smooth=true;
                          else { double sm=std::min(std::fabs(kmx),std::fabs(kmn));
                                 double tol=u2_meshtol?(1.0-std::pow(du2,1.0/6.0)):0.5; smooth=sm>=tol*am; } }
                   p2smooth[i]=smooth?1:0; }
        }
    }
    // ABSOLUTE ceiling backstop (MOOD P1; GP-MOOD/Dumbser: PAD carries HARD physical bounds). Flags a
    // gross rho/p overshoot (> CEIL × global-max) regardless of the relative DMP. DEFAULT OFF (0): it is
    // a HEURISTIC (the CEIL factor is arbitrary) and merely MASKS symptoms — the real cases are fixed at
    // the ROOT instead (compact-DMP for spreading overshoot; DIRICHLET BC for the octant boundary
    // feedback blowup). Kept env-available (MOOD_ABS_CEIL=<factor>, e.g. 6) as a last-resort backstop only.
    static const double abs_ceil = []{ const char* e=std::getenv("MOOD_ABS_CEIL"); return (e&&e[0])?std::atof(e):0.0; }();
    double gmax_rho = 0.0, gmax_p = 0.0;
    if (abs_ceil > 0.0) {
        #pragma omp parallel for reduction(max:gmax_rho,gmax_p)
        for (int i = 0; i < N; ++i) {
            double r = U[(size_t)0*N+i]; if (r > gmax_rho) gmax_rho = r;
            double uc[5]; for(int v=0;v<5;++v) uc[v]=U[(size_t)v*N+i];
            double wc[5]; eq.cons_to_prim(uc, wc); if (wc[4] > gmax_p) gmax_p = wc[4];   // pressure at sub-step start
        }
    }
    // shared detector: troubled = !PAD OR (density-DMP violated AND not-smooth) OR
    // (pressure-DMP violated AND not-smooth). PAD = positivity floor.
    auto detect_bad = [&](int i) -> bool {
        double un[5];
        for (int v = 0; v < 5; ++v) un[v] = U[(size_t)v*N+i] + dt*dUdt[(size_t)v*N+i];
        if (!euler3d_pad_ok(eq, un)) return true;
        if (abs_ceil > 0.0 && un[0] > abs_ceil*gmax_rho) return true;   // gross density-overshoot backstop (absolute)
        double w[5]; eq.cons_to_prim(un, w);   // candidate primitives (rho,u,v,w,p), reused below
        if (abs_ceil > 0.0 && gmax_p > 0.0 && w[4] > abs_ceil*gmax_p) return true;   // gross PRESSURE-overshoot backstop (the octant boundary p→95 blowup)
        // WAVE-SPEED sanity (robustness): a candidate signal speed |u|+c spiking far beyond
        // the cell's time-n value = spurious THINC over-sharpening that COLLAPSES the CFL dt
        // (the Langseth failure: huge velocities, enstrophy blow-up, dt→0 — the rho/p DMP is
        // blind to it). factor 8 spares genuine shocks (~2-3x jump). Opt-out MOOD_NOSPEED.
        static const bool no_speed = std::getenv("MOOD_NOSPEED") != nullptr;
        if (!no_speed) {
            double g = eq.gamma;
            double sn = std::sqrt(Wc[(size_t)1*N+i]*Wc[(size_t)1*N+i]+Wc[(size_t)2*N+i]*Wc[(size_t)2*N+i]+Wc[(size_t)3*N+i]*Wc[(size_t)3*N+i])
                      + std::sqrt(g*Wc[(size_t)4*N+i]/Wc[(size_t)0*N+i]);
            double sc = std::sqrt(w[1]*w[1]+w[2]*w[2]+w[3]*w[3]) + std::sqrt(g*w[4]/w[0]);
            if (sc > 8.0*sn + 1e-12) return true;
        }
        // ABSOLUTE drain floor (drift-free): the relaxed DMP is RELATIVE to the time-n stencil
        // MIN, which DRIFTS DOWN as a region slowly drains → a gradual vacuum drain stays "in
        // bounds" every sub-step and is never flagged (the Langseth drift). Compare to the
        // stencil MAX (anchored to the high-side/ambient, does NOT drift down): flag if ρ or p
        // < FRAC·max. FRAC=0.05 spares the initial low feature (ρ=0.1 vs max~1 → 0.1>0.05).
        // Tunable MOOD_FLOOR_FRAC (0 = off).
        static const double floor_frac = []{ const char* e=std::getenv("MOOD_FLOOR_FRAC"); return (e&&e[0])?std::atof(e):0.05; }();
        if (floor_frac > 0.0) {
            if (use_dmp  && un[0] < floor_frac*rhomax[i]) return true;
            if (use_pdmp && w[4]  < floor_frac*pmaxd[i])  return true;
        }
        // ABSOLUTE buildup CEILING (drift-free, symmetric to the drain floor): the relative DMP +
        // u2 HIGH-side sparing miss a slow over-sharpening BUILDUP (octant 210k: rho drifts to 72,
        // enstrophy→6e4 — the stencil MAX rises with the buildup so it stays "in bounds"). Anchor to
        // the stencil MIN (does NOT rise during a localized buildup): flag rho/p > MIN/frac (=20×).
        // Skip near-vacuum stencils. Env MOOD_CEIL_FRAC. DEFAULT OFF (0.0): on the octant 210k slip-line
        // over-sharpening blowup the ceiling made it WORSE (ens 6e4→4e6, dt collapsed 240×) — flooring
        // the over-sharpened cells fed a sharpen→floor→dt-collapse cycle. The blowup is a RECON-level
        // (slip-line/shear) problem, not an a-posteriori-catchable one. Kept env-available for other cases.
        static const double ceil_frac = []{ const char* e=std::getenv("MOOD_CEIL_FRAC"); return (e&&e[0])?std::atof(e):0.0; }();
        if (ceil_frac > 0.0) {
            if (use_dmp  && rhomin[i] > 0.01*rhomax[i] && un[0] > rhomin[i]/ceil_frac) return true;
            if (use_pdmp && pmind[i]  > 0.01*pmaxd[i]  && w[4]  > pmind[i]/ceil_frac)  return true;
        }
        // ASYMMETRIC DMP (robustness fix): a low-side violation (UNDERSHOOT — density/
        // pressure draining toward vacuum) is positivity-critical and is ALWAYS flagged;
        // u2 must NOT spare it (the gradual-drain failure mode: u2 classifies a slow
        // vacuum drain as a "smooth extremum", lets it slip through, and the cell-average
        // drains over many accepted sub-steps until p<0 — the Langseth divergence). Only a
        // HIGH-side violation (overshoot — a peak that may be a genuine smooth extremum) is
        // eligible for the u2 accuracy-sparing. Opt-out (revert to symmetric) MOOD_SYMDMP.
        static const bool sym_dmp = std::getenv("MOOD_SYMDMP") != nullptr;
        // GROSS-overshoot cap (drift-free, MAX-anchored): the high-side DMP spares an overshoot when
        // u2smooth[i] (protect genuine smooth extrema). But at high resolution a sharp THINC(beta~1.6)
        // + RHLLC can OVER-SHARPEN a shock↔contact interaction into a spurious spike (Langseth
        // light-cylinder: rho→7.7 ≈ 4.5× stencil max) that u2 MIS-classifies as smooth → spared →
        // runaway → dt collapse. A genuine smooth extremum never exceeds the stencil MAX by a large
        // factor; so flag rho/p > CAP·(stencil max) UNCONDITIONALLY (ignore u2). CAP=3 spares real
        // peaks, kills the blowup. MAX-anchored (not MIN-anchored like MOOD_CEIL_FRAC) => no false
        // trips near low-density features. Env MOOD_OS_CAP (0 = off). Enables denser stable meshes.
        // DEFAULT OFF: a SPREADING overshoot raises the stencil MAX with it, so max-anchored 3× is
        // never crossed (Langseth tet N=33: cap=3 → byte-identical divergence). Recon-level, not
        // a-posteriori-catchable. Kept env-available for LOCALIZED-spike cases.
        static const double os_cap = []{ const char* e=std::getenv("MOOD_OS_CAP"); return (e&&e[0])?std::atof(e):0.0; }();
        if (use_dmp) {
            double rs = un[0];   // candidate density (U[0]=rho)
            if (rs < rhomin[i]-deltad[i] && (!sym_dmp || !u2smooth[i])) return true;  // undershoot → drain
            if (rs > rhomax[i]+deltad[i] && !u2smooth[i]) return true;                // overshoot → peak
            if (os_cap > 0.0 && rhomax[i] > 0.0 && rs > os_cap*rhomax[i]) return true;// GROSS overshoot → always flag
        }
        if (use_pdmp) {
            double ps = w[4];   // candidate pressure (reuse hoisted w)
            if (ps < pmind[i]-deltap[i] && (!sym_dmp || !p2smooth[i])) return true;   // pressure drain
            if (ps > pmaxd[i]+deltap[i] && !p2smooth[i]) return true;
            if (os_cap > 0.0 && pmaxd[i] > 0.0 && ps > os_cap*pmaxd[i]) return true;  // GROSS overshoot → always flag
        }
        return false;
    };

    // step 2: full-N PAD on the candidate U_new = U + dt*dUdt → initial troubled set.
    std::vector<unsigned char> troubled((size_t)N, 0);
    std::vector<int> work;   // current troubled-cell worklist
    {
        std::vector<int> local;
        #pragma omp parallel
        {
            std::vector<int> priv;
            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                if (detect_bad(i)) { troubled[i] = 1; priv.push_back(i); }
            }
            #pragma omp critical
            local.insert(local.end(), priv.begin(), priv.end());
        }
        work.swap(local);
    }
    const int troubled_initial = (int)work.size();   // initial troubled-cell count (for stats)

    // step 3: GRADED cascade (Diot 2012 degree-decrement). Each iteration: decrement the
    // level of every troubled cell's faces (THINC→MUSCL→1st), recompute those faces at the
    // new level, re-gather + re-detect the touched cells. A troubled THINC face drops to
    // the clean MUSCL candidate first; only to 1st-order if MUSCL is also troubled — this
    // preserves accuracy (MUSCL is 2nd-order) where straight-to-1st would over-diffuse.
    std::vector<int> touch_stamp((size_t)N, 0), face_stamp((size_t)Nf, 0);
    int epoch = 0;
    std::vector<int> changed, touched, next;
    static const int MITER=[]{const char*e=std::getenv("MOOD_ITER");return(e&&e[0])?std::atoi(e):-1;}();
    int mit=(MITER>0)?MITER:max_iter;   // env override (diagnose graded-cascade depth vs stability)
    for (int it = 0; it < mit && !work.empty(); ++it) {
        ++epoch;
        // (a) decrement the level of each troubled cell's faces (>0), once per face per iter.
        changed.clear();
        for (int ci : work) {
            for (int f : m.cell_faces[ci]) {
                if (face_level[f] > 0 && face_stamp[f] != epoch) {
                    face_stamp[f] = epoch;
                    face_level[f] = (face_level[f] == 2 && have_muscl) ? 1 : 0;   // 2→1(MUSCL)/→0 ; 1→0
                    changed.push_back(f);
                }
            }
        }
        if (changed.empty()) break;   // every troubled face is already first-order
        // (b) recompute the changed faces at their new (lower) level.
        #pragma omp parallel for
        for (size_t a = 0; a < changed.size(); ++a) recompute_face(changed[a]);
        // (c) re-gather the cells touching a changed face.
        touched.clear();
        for (int f : changed) {
            int o = m.face_owner[f], n = m.face_neighbour[f];
            if (o >= 0 && touch_stamp[o] != epoch) { touch_stamp[o] = epoch; touched.push_back(o); }
            if (n >= 0 && touch_stamp[n] != epoch) { touch_stamp[n] = epoch; touched.push_back(n); }
        }
        #pragma omp parallel for
        for (size_t a = 0; a < touched.size(); ++a) gather_cell(touched[a]);
        // (d) re-detect; a cell whose faces are ALL first-order is at the floor (PAD only —
        //     1st-order is monotone, so the DMP/u2 oscillation tests no longer apply).
        next.clear();
        for (int ci : touched) {
            bool all_lo = true;
            for (int f : m.cell_faces[ci]) if (face_level[f] > 0) { all_lo = false; break; }
            bool bad;
            if (all_lo) {
                double un[5];
                for (int v = 0; v < 5; ++v) un[v] = U[(size_t)v*N+ci] + dt*dUdt[(size_t)v*N+ci];
                bad = !euler3d_pad_ok(eq, un);
            } else bad = detect_bad(ci);
            if (!bad) { troubled[ci] = 0; continue; }
            troubled[ci] = 1;
            if (!all_lo) next.push_back(ci);   // still has high-order faces → keep cascading
        }
        work.swap(next);
    }
    // FAIL-SAFE FLOOR: if max_iter was exhausted with cells still troubled, FORCE all their
    // faces to first-order (the parachute MUST fully deploy). Without this, a graded cascade
    // (THINC→MUSCL→1st, 2 decrements) can run out of iterations on a propagating strong shock
    // and leave cells stranded at MUSCL — committing an unsafe flux → divergence (verified:
    // graded diverged at max_iter=3, stable at 12). This guarantees MOOD is unconditionally
    // fail-safe regardless of cascade depth.
    if (!work.empty()) {
        ++epoch; changed.clear();
        for (int ci : work) for (int f : m.cell_faces[ci])
            if (face_level[f] > 0 && face_stamp[f] != epoch) { face_stamp[f]=epoch; face_level[f]=0; changed.push_back(f); }
        #pragma omp parallel for
        for (size_t a=0;a<changed.size();++a) recompute_face(changed[a]);
        touched.clear();
        for (int f : changed) { int o=m.face_owner[f], n=m.face_neighbour[f];
            if (o>=0 && touch_stamp[o]!=epoch) { touch_stamp[o]=epoch; touched.push_back(o); }
            if (n>=0 && touch_stamp[n]!=epoch) { touch_stamp[n]=epoch; touched.push_back(n); } }
        #pragma omp parallel for
        for (size_t a=0;a<touched.size();++a) gather_cell(touched[a]);
    }

    // BVD_CANDFLAG: export a per-cell MOOD level for the paper diagnostic. 3D MOOD tracks a
    // per-FACE level (2=THINC/BVD,1=MUSCL,0=first-order); the cell level = MIN over its faces
    // (higher=less repaired), mapped to the same convention as 2D (2=BVD/P2 accepted,1=MUSCL/P1,
    // 0=P0/first-order). Overwrites each stage-call -> LAST call (final-time, last stage) wins.
    if (std::getenv("BVD_CANDFLAG")) {
        auto& ml = mood_level_flag(); ml.assign((size_t)N, 2);
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) { int lv = 2;
            for (int f : m.cell_faces[i]) { int fl = face_level[f]; if (fl < lv) lv = fl; }
            ml[i] = (signed char)lv; }
    }
    if (prof) { e3d_prof().mood += e3d_ms()-_t; _t = e3d_ms(); }

    // ── viscous (NS): the MOOD cascade above modifies only the CONVECTIVE reconstruction;
    //    add the centered diffusive flux (τ, q) to the post-cascade Fall, then a FINAL
    //    gather so dUdt = cascaded-convective + viscous. The viscous flux is a centred
    //    physics term (first-order cell-average gradients), not the convective recon, and
    //    is purely diffusive → adding it after the (convective) positivity cascade is safe.
    if (visc && visc->mu > 0.0 && ctx) {
        auto bc_ghost = [&](int f, const double wL[5], double wR[5]) {
            double nx = m.face_normals[(size_t)f*3+0], ny = m.face_normals[(size_t)f*3+1], nz = m.face_normals[(size_t)f*3+2];
            int tag = m.face_bc_tag[f];
            if (bcs && tag > 0 && tag < (int)bcs->size())
                apply_bc3d((*bcs)[tag], wL, nx, ny, nz,
                           m.face_centers[(size_t)f*3+0], m.face_centers[(size_t)f*3+1], m.face_centers[(size_t)f*3+2], t, wR);
            else for (int v = 0; v < 5; ++v) wR[v] = wL[v];
        };
        static const bool visc_p2face = (std::getenv("VISC_CENTROID") == nullptr);
        if (unstr3d && o2ctx && visc_p2face) {
            std::vector<double> cu, cv, cw, cT;
            viscous3d_cell_coeffs_o2(m, *o2ctx, Wc, visc->R, cu, cv, cw, cT);
            viscous3d_add_face_flux_p2face(m, eq, *visc, Wc, cu, cv, cw, cT, bc_ghost, Fall);
        } else {
            std::vector<double> gu, gv, gw, gT;
            if (unstr3d && o2ctx) viscous3d_cell_gradients_o2(m, *o2ctx, Wc, visc->R, gu, gv, gw, gT);
            else                  viscous3d_cell_gradients(m, eq, *ctx, Wc, visc->R, gu, gv, gw, gT);
            viscous3d_add_face_flux(m, eq, *visc, Wc, gu, gv, gw, gT, bc_ghost, Fall);
        }
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) gather_cell(i);
    }
    if (prof) { e3d_prof().visc += e3d_ms()-_t; }

    if (stats) {
        // troubled_total/max: cells flagged at least once (initial PAD set size).
        // floored: cells STILL inadmissible after the full cascade — i.e. even first-order
        // on all their faces did not restore rho>0,p>0 for this dt (should be 0; >0 = warning).
        stats->troubled_total += (long)troubled_initial;
        if (troubled_initial > stats->troubled_max) stats->troubled_max = troubled_initial;
        long still = 0; for (int i = 0; i < N; ++i) still += troubled[i];
        stats->floored += still;
    }
}

struct Solve3DResult { std::vector<double> U; int n_steps = 0; double t = 0.0; };

inline Solve3DResult solve_euler3d(const Mesh& m, const Euler3D& eq,
                                   const std::vector<double>& U0, double t_end,
                                   int integrator = 2, double cfl = 0.4,
                                   double dt_fixed = -1.0, int max_steps = 100000000,
                                   int recon = RECON3_FIRST, const ReconCtx3D* ctx = nullptr,
                                   int flux = FLUX3_LLF, const std::vector<BC3D>* bcs = nullptr,
                                   const ReconCtx3DO2* o2ctx = nullptr,
                                   const ViscousParams* visc = nullptr) {
    const int N = m.n_cells(); const int sz = 5 * N;
    std::vector<double> U = U0, L0(sz), U1(sz), L1(sz), U2(sz), L2(sz);
    std::vector<double> Wc(sz), WL, WR;
    // RECON3_BVD needs the o2 P2 LSQ context. If the caller did not pass one, build
    // it once here (geometry-only; reused across all steps/stages).
    ReconCtx3DO2 o2_local;
    if (recon == RECON3_BVD && !o2ctx) {
        o2_local = (m.s3_nx == 0) ? build_recon_ctx_3d_o2_unstr(m)   // unstructured node-ring stencil
                                  : build_recon_ctx_3d_o2(m);        // structured vertex-26
        o2ctx = &o2_local;
    }
    // RECON3_BVD viscous path needs a FACE-nbr ctx for the gradients; build if missing.
    ReconCtx3D ctx_local;
    const bool viscous = (visc != nullptr) && (visc->mu > 0.0);
    // MUSCL (BJ-vertex) needs the face-nbr LSQ ctx; viscous needs it for the gradients. Build if missing
    // (else recon silently falls back to FIRST-order because ho requires a non-null ctx).
    if ((viscous || recon == RECON3_BJ_VERTEX) && !ctx) { ctx_local = build_recon_ctx_3d(m); ctx = &ctx_local; }
    double cur_t = 0.0;
    auto RHS = [&](const std::vector<double>& s, std::vector<double>& out) {
        euler3d_rhs(m, eq, s, out, recon, ctx, Wc, WL, WR, flux, bcs, cur_t, o2ctx, visc);
    };
    // a-posteriori MOOD positivity protection. DEFAULT OFF: the deg3t-BVD path
    // (RECON3_BVD) runs WITHOUT MOOD unless MOOD_ON is explicitly set. The machinery
    // stays available behind the flag (opt-in) but is never invoked by default, so
    // every recon path is unchanged out of the box. MOOD_DIAG prints the per-step
    // troubled-cell count when MOOD is enabled.
    const bool mood_diag = std::getenv("MOOD_DIAG") != nullptr;
    const bool mood_off  = std::getenv("MOOD_OFF") != nullptr;
    // MOOD = a-posteriori positivity safeguard for the high-order deg3t-BVD recon
    // at strong shocks. DEFAULT-ON for RECON3_BVD (opt-out via MOOD_OFF); first-order
    // / MUSCL are positivity-stable and skip it. MOOD_ON forces it on any recon.
    const bool use_mood  = (std::getenv("MOOD_ON") != nullptr) || (recon == RECON3_BVD && !mood_off);
    MoodStats mood_step;
    // MOOD-protected forward-Euler RHS: returns a dUdt whose s + dt*dUdt is admissible.
    auto RHS_mood = [&](const std::vector<double>& s, std::vector<double>& out, double dtw) {
        if (use_mood)
            euler3d_rhs_mood(m, eq, s, out, dtw, recon, ctx, Wc, WL, WR, flux, bcs, cur_t, o2ctx, &mood_step, 3, visc);
        else
            euler3d_rhs(m, eq, s, out, recon, ctx, Wc, WL, WR, flux, bcs, cur_t, o2ctx, visc);
    };
    double h_min = 1e300;
    for (int i = 0; i < N; ++i) {
        double amax = 0.0; for (int f : m.cell_faces[i]) amax = std::max(amax, m.face_areas[f]);
        h_min = std::min(h_min, m.cell_volumes[i] / std::max(amax, 1e-30));
    }
    // Smallest cell linear size for the viscous (diffusive) dt limit (NS only).
    double hvisc_min = 1e300;
    if (viscous) { for (int i = 0; i < N; ++i) hvisc_min = std::min(hvisc_min, std::cbrt(m.cell_volumes[i])); }
    // dt-collapse divergence guard (DT_COLLAPSE_FACTOR env, default 1e-3).
    double dt_ref = -1.0;
    const char* dcf = std::getenv("DT_COLLAPSE_FACTOR");
    const double collapse = dcf ? std::atof(dcf) : 1e-3;
    double t = 0.0; int n = 0;
    if(const char*_e=std::getenv("CFD_MAXSTEP")){int _v=std::atoi(_e); if(_v>0&&_v<max_steps)max_steps=_v;}   // bench cap (per-cell timing)
    for (; n < max_steps && t < t_end; ++n) {
        double dt;
        if (dt_fixed > 0.0) dt = dt_fixed;
        else {
            double wmax = 0.0, rmin = 1e300;
            #pragma omp parallel for reduction(max:wmax) reduction(min:rmin)
            for (int i = 0; i < N; ++i) {
                double u[5] = {U[0*N+i],U[1*N+i],U[2*N+i],U[3*N+i],U[4*N+i]};
                double w = eq.max_wave_speed(u, 1.0, 0.0, 0.0);
                double w2 = eq.max_wave_speed(u, 0.0, 1.0, 0.0);
                double w3 = eq.max_wave_speed(u, 0.0, 0.0, 1.0);
                w = std::max(w, std::max(w2, w3)); if (w > wmax) wmax = w;
                double rho = u[0]; if (rho < rmin) rmin = rho;
            }
            if (!std::isfinite(wmax) || wmax > 1e12) {
                std::fprintf(stderr, "DIVERGED: non-finite/huge wave speed (%.3e) at step %d t=%.6g\n",
                             wmax, n, t); break;
            }
            dt = cfl * h_min / std::max(wmax, 1e-30);
            // viscous (diffusive) dt limit: dt <= cfl * rho * h² / (2 * mu * Cdiff),
            // Cdiff = max(4/3, gamma/Pr) bounds the largest diffusion coefficient
            // (momentum 4/3 nu vs thermal gamma/Pr nu). dt = min(convective, viscous).
            if (viscous) {
                double Cdiff = std::max(4.0/3.0, eq.gamma / visc->Pr);
                double dt_v = cfl * std::max(rmin, 1e-30) * hvisc_min * hvisc_min
                            / std::max(2.0 * visc->mu * Cdiff, 1e-30);
                if (dt_v < dt) dt = dt_v;
            }
            if (dt_ref < 0.0) dt_ref = dt;
            else if (dt < collapse * dt_ref) {
                std::fprintf(stderr, "DIVERGED: dt collapsed (%.3e < %.1e*%.3e) at step %d t=%.6g\n",
                             dt, collapse, dt_ref, n, t); break;
            }
        }
        if (t + dt > t_end) dt = t_end - t;
        if (dt <= 0.0) break;
        cur_t = t;
        // floored-cell dt-cut: if MOOD cannot restore admissibility even at first-order
        // (floored>0 ⇒ first-order positivity-CFL violated for those cells at this dt),
        // REJECT the step, restore U, halve dt, retry (≤ MOOD_DTCUT times, default 8 ⇒
        // dt/256). Keeps single-step positivity beyond what the global CFL alone gives.
        static const int mood_retry = []{ const char* e=std::getenv("MOOD_DTCUT"); return e?std::atoi(e):0; }();  // default OFF: DTCUT=8 caused crawl+blow-up (repeated dt-halve then commits bad step). floor->LLF (fix a) handles non-PP flux instead.
        const bool dtcut = use_mood && (mood_retry > 0);
        std::vector<double> U_save; if (dtcut) U_save = U;
        int retry = 0;
        while (true) {
            mood_step = MoodStats{};   // reset per-attempt MOOD counters
            // SSP-RK convex combination of forward-Euler sub-steps; each dUdt is MOOD-protected.
            RHS_mood(U, L0, dt);
            if (integrator == 0) {
                #pragma omp parallel for
                for (int i = 0; i < sz; ++i) U[i] += dt*L0[i];
            } else if (integrator == 1) {
                #pragma omp parallel for
                for (int i = 0; i < sz; ++i) U1[i] = U[i] + dt*L0[i];
                RHS_mood(U1, L1, dt);
                #pragma omp parallel for
                for (int i = 0; i < sz; ++i) U[i] = 0.5*U[i] + 0.5*(U1[i] + dt*L1[i]);
            } else {
                #pragma omp parallel for
                for (int i = 0; i < sz; ++i) U1[i] = U[i] + dt*L0[i];
                RHS_mood(U1, L1, dt);
                #pragma omp parallel for
                for (int i = 0; i < sz; ++i) U2[i] = 0.75*U[i] + 0.25*(U1[i] + dt*L1[i]);
                RHS_mood(U2, L2, dt);
                #pragma omp parallel for
                for (int i = 0; i < sz; ++i) U[i] = (1.0/3.0)*U[i] + (2.0/3.0)*(U2[i] + dt*L2[i]);
            }
            if (!dtcut || mood_step.floored == 0 || retry >= mood_retry) break;
            U = U_save; dt *= 0.5; ++retry;   // floored cells remain → reject & retry at dt/2
        }
        if (use_mood && mood_diag) {
            std::fprintf(stderr, "[MOOD] step %d t=%.6g dt=%.3e retry=%d troubled=%ld/%d floored=%ld\n",
                         n, t, dt, retry, mood_step.troubled_total, mood_step.troubled_max, mood_step.floored);
        }
        t += dt;
        // optional progress (env SOLVE3D_PROGRESS = step stride). Cheap; off by default.
        static const char* prg = std::getenv("SOLVE3D_PROGRESS");
        static const int prg_n = prg ? std::max(1, std::atoi(prg)) : 0;
        if (prg_n && (n % prg_n == 0)) {
            std::fprintf(stderr, "[step %d] t=%.6f dt=%.3e\n", n, t, dt); std::fflush(stderr);
        }
    }
    if (e3d_prof_on()) {
        const E3DProf& p = e3d_prof();
        double tot = p.prim + p.recon + p.ppfloor + p.flux + p.visc + p.asmb + p.mood;
        if (tot < 1e-9) tot = 1.0;
        std::printf("E3D_PROF: %ld RHS calls, summed %.0fms  ||  "
            "prim %.0fms(%.1f%%)  RECON %.0fms(%.1f%%)  ppfloor %.0fms(%.1f%%)  "
            "FLUX %.0fms(%.1f%%)  MOOD %.0fms(%.1f%%)  visc %.0fms(%.1f%%)  asm %.0fms(%.1f%%)\n",
            p.ncall, tot,
            p.prim,100*p.prim/tot, p.recon,100*p.recon/tot, p.ppfloor,100*p.ppfloor/tot,
            p.flux,100*p.flux/tot, p.mood,100*p.mood/tot, p.visc,100*p.visc/tot, p.asmb,100*p.asmb/tot);
        const U3Prof& r = u3_prof();
        double rtot = r.lsq + r.celld + r.facebvd; if (rtot < 1e-9) rtot = 1.0;
        long nit = u3_celld_iters().load(), nso = u3_celld_solves().load();
        // same one-line format as solver_advect3d.hpp (parser parity across 3D paths).
        std::printf("E3D_PROF (euler3d) RECON breakdown (summed %.0fms): o2-LSQ %.0fms(%.1f%%)  "
            "cell-D-Newton %.0fms(%.1f%%, avg %.2f Newton iters/solve over %ld solves)  "
            "face-quad+BVD %.0fms(%.1f%%)\n",
            rtot, r.lsq,100*r.lsq/rtot, r.celld,100*r.celld/rtot, nso?(double)nit/nso:0.0, nso, r.facebvd,100*r.facebvd/rtot);
    }
    return Solve3DResult{std::move(U), n, t};
}

} // namespace cfd
