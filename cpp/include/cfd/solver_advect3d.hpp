// cfd/solver_advect3d.hpp — conservative 3D scalar advection with a PRESCRIBED,
// possibly time-varying, divergence-free velocity field u(x,y,z,t).
//
//   dg/dt + div(u g) = 0,   F.n = (u.n) g,  upwind picks the upstream cell.
//
// The 3D analogue of solve_adv2d (solver2d.hpp) but for a TIME-DEPENDENT velocity
// (the Enright/LeVeque deformation field), so the velocity is evaluated at the
// correct SSP-RK stage time. Reconstruction is the Stage-1 MUSCL baseline
// (reconstruct_bj_vertex_3d, nvar=1); recon==0 falls back to first order. Since
// the velocity is divergence-free, Sum(g*V) is conserved to round-off.
//
// Header-only, new file. Does not modify euler3d / reconstruct3d / mesh.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/reconstruct3d_bvd.hpp"
#include "cfd/reconstruct3d_o2_unstr.hpp"
#include "cfd/reconstruct3d_unstr.hpp"
#include "cfd/reconstruct3d_tmlpu.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

namespace cfd {

// Reconstruction selector for the scalar advection path (mirrors RECON3_*).
constexpr int ADV3_FIRST = 0;       // first-order upwind (g_L=g[owner], g_R=g[nb])
constexpr int ADV3_BJ_VERTEX = 1;   // limited-linear MUSCL (reconstruct_bj_vertex_3d)
constexpr int ADV3_BVD = 4;         // deg3t-THINC-QQ + min-TBV BVD (reconstruct3d_bvd)

// Velocity callback: writes the 3-vector u(x,y,z,t) into uvw[3].
using Velocity3D = std::function<void(double, double, double, double, double*)>;

// One RHS evaluation:  dg/dt = -(1/V) Sum_faces sign * (u.n) g_upwind * area,
// with the velocity field sampled at face centres at stage time t.
//  - reconstruct (recon!=0 && ctx): g_L = owner-side face value, g_R = neighbour-side.
//  - first order: g_L = g[owner], g_R = g[neighbour].
//  - boundary faces (n<0): the deformation velocity vanishes on the cube wall by
//    construction so u.n ~ 0; we use g_R = g_L and flux = (u.n) g_L (~0).
inline void advect3d_rhs(const Mesh& m, const std::vector<double>& g, double t,
                         const Velocity3D& vel, std::vector<double>& dgdt,
                         int recon, const ReconCtx3D* ctx,
                         std::vector<double>& gL, std::vector<double>& gR,
                         const ReconCtx3DO2* o2ctx = nullptr,
                         double beta_l = 1.6, double beta_s = 0.8,
                         double mood_dt = -1.0) {   // >0 => a-posteriori scalar MOOD (compact DMP -> 1st-order)
    const int N = m.n_cells(), Nf = m.n_faces();
    const bool bvd = (recon == ADV3_BVD) && ctx && o2ctx;
    const bool ho = bvd || ((recon == ADV3_BJ_VERTEX) && ctx);
    const bool unstr = (m.s3_nx == 0);   // unstructured mesh -> GAUSS-THINC unstr recon
    // BVD_ABVD (adaptive TVD-BVD linear candidate): the Eq.26 upwind endpoint needs the
    // per-face advection speed a.n at the stage time (midpoint fallback is NOT TVD).
    static const bool ADV_ABVD = std::getenv("BVD_ABVD") != nullptr;
    std::vector<double> unf;
    if (bvd && unstr && ADV_ABVD) {
        unf.resize((size_t)Nf);
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            double uvw[3]; vel(m.face_centers[3*f], m.face_centers[3*f+1], m.face_centers[3*f+2], t, uvw);
            unf[f] = uvw[0]*m.face_normals[3*f] + uvw[1]*m.face_normals[3*f+1] + uvw[2]*m.face_normals[3*f+2];
        }
    }
    static const bool TMLPU_GATED_ADV = std::getenv("TMLPU_GATED") != nullptr;
    if (bvd && unstr && TMLPU_GATED_ADV) reconstruct_tmlpu_gated_3d(m, g, /*nvar*/1, gL, gR);
    else if (bvd && unstr) reconstruct3d_bvd_gauss_unstr(m, *o2ctx, g, /*nvar*/1, gL, gR, beta_l, beta_s,
                                                    unf.empty() ? nullptr : unf.data());
    else if (bvd) reconstruct3d_bvd(m, *o2ctx, *ctx, g, /*nvar*/1, gL, gR, beta_l, beta_s);
    else if (ho) reconstruct_bj_vertex_3d(m, *ctx, g, /*nvar*/1, gL, gR);

    // Pass 1 (race-free): per-face area-weighted upwind flux into Fall[f]. uA[f]=(u·n)·area (cached for
    // the MOOD first-order recompute so the same wind direction/magnitude is reused).
    std::vector<double> Fall((size_t)Nf), uA((size_t)Nf);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double nx = m.face_normals[(size_t)f * 3 + 0];
        double ny = m.face_normals[(size_t)f * 3 + 1];
        double nz = m.face_normals[(size_t)f * 3 + 2];
        double fx = m.face_centers[(size_t)f * 3 + 0];
        double fy = m.face_centers[(size_t)f * 3 + 1];
        double fz = m.face_centers[(size_t)f * 3 + 2];
        double uvw[3]; vel(fx, fy, fz, t, uvw);
        double un = uvw[0] * nx + uvw[1] * ny + uvw[2] * nz;
        uA[f] = un * m.face_areas[f];
        double uLval, uRval;
        if (ho) { uLval = gL[f]; uRval = (n >= 0) ? gR[f] : gL[f]; }
        else    { uLval = g[o];  uRval = (n >= 0) ? g[n]  : g[o]; }
        static const bool bc_face_recon = std::getenv("BC_FACE_RECON") != nullptr; // OLD: keep reconstructed boundary value
        if (n < 0) { if (!bc_face_recon) uLval = g[o]; uRval = uLval; }   // Neumann boundary: cell average, no recon (2026-07-03); BC_FACE_RECON keeps recon
        double gup = (un >= 0.0) ? uLval : uRval;   // upwind scalar
        Fall[f] = uA[f] * gup;
    }
    auto gather = [&](int i)->double{ double acc=0.0;
        for (int f : m.cell_faces[i]) acc += ((m.face_owner[f]==i)?-1.0:1.0)*Fall[f];
        return acc / m.cell_volumes[i]; };
    // Pass 2 (race-free): each cell gathers its faces (owner -F, neighbour +F).
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) dgdt[i] = gather(i);

    // SCALAR MOOD (a-posteriori): the sharp THINC interface can push the CONSERVATIVE update g+dt·L over
    // the local bound (deform sphere: g 1.0->1.6) — the reconstruction is bounded but the update is not,
    // and there is no positivity/limiter net. Detect cells whose candidate g_new violates the COMPACT
    // face-neighbour DMP (the canonical MOOD bound; strict = no smooth-extremum spare since the sphere is
    // a genuine discontinuity), demote ALL their faces to FIRST-ORDER upwind of cell averages, re-gather.
    if (mood_dt > 0.0 && ho) {
        std::vector<char> fo((size_t)Nf, 0);   // face -> use first-order
        std::vector<char> bad((size_t)N, 0);
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double gnew = g[i] + mood_dt * dgdt[i];
            double gmn = g[i], gmx = g[i];
            for (int fc : m.cell_faces[i]) { int nb=(m.face_owner[fc]==i)?m.face_neighbour[fc]:m.face_owner[fc];
                if (nb < 0) continue; double gn=g[nb]; if(gn<gmn)gmn=gn; if(gn>gmx)gmx=gn; }
            double d = 1e-9 + 1e-6*(gmx-gmn);
            if (gnew < gmn - d || gnew > gmx + d) bad[i] = 1;
        }
        bool any = false;
        for (int i = 0; i < N; ++i) if (bad[i]) { any = true; for (int fc : m.cell_faces[i]) fo[fc]=1; }
        if (any) {
            #pragma omp parallel for
            for (int f = 0; f < Nf; ++f) if (fo[f]) {   // first-order upwind of cell AVERAGES
                int o=m.face_owner[f], n=m.face_neighbour[f];
                double gup = (uA[f] >= 0.0) ? g[o] : ((n>=0)?g[n]:g[o]);
                Fall[f] = uA[f] * gup;
            }
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) dgdt[i] = gather(i);   // re-gather (only faces on bad cells changed)
        }
    }
}

// SSP-RK scalar advection march. integrator: 0 fwd-Euler, 1 SSP-RK2, 2 SSP-RK3.
// dt = cfl * h_min / |u|_max with |u|_max sampled at cell centres at the current
// step time, or dt_fixed if > 0. h_min = min(cell_volume / max incident face area).
// The velocity is evaluated at the proper stage time so the time-dependent
// deformation field is integrated 2nd/3rd-order accurately.
inline std::vector<double> solve_advect3d(
        const Mesh& m, const std::vector<double>& g0, double t_end,
        const Velocity3D& vel, double cfl = 0.5, double dt_fixed = -1.0,
        int integrator = 2, int recon = ADV3_BJ_VERTEX,
        const ReconCtx3D* ctx = nullptr, int* out_steps = nullptr,
        double* out_t = nullptr, const ReconCtx3DO2* o2ctx = nullptr,
        double beta_l = 1.6, double beta_s = 0.8, double t_start = 0.0) {
    const int N = m.n_cells();
    std::vector<double> g = g0, L0(N), g1(N), L1(N), g2(N), L2(N);
    std::vector<double> gL, gR;   // reused reconstruction buffers

    // The BVD path needs the o2 (P2-LSQ) context for the interface polynomial.
    // Build it once here if the caller did not supply one.
    ReconCtx3DO2 o2_local;
    const ReconCtx3DO2* o2 = o2ctx;
    if (recon == ADV3_BVD && o2 == nullptr) {
        o2_local = (m.s3_nx == 0) ? build_recon_ctx_3d_o2_unstr(m)   // unstructured node-ring
                                  : build_recon_ctx_3d_o2(m);
        o2 = &o2_local;
    }

    // characteristic length: min over cells of volume / max adjacent face area.
    double h_min = 1e300;
    for (int i = 0; i < N; ++i) {
        double amax = 0.0;
        for (int f : m.cell_faces[i]) amax = std::max(amax, m.face_areas[f]);
        h_min = std::min(h_min, m.cell_volumes[i] / std::max(amax, 1e-30));
    }

    auto umax_at = [&](double tt) {
        double um = 0.0;
        #pragma omp parallel for reduction(max:um)
        for (int i = 0; i < N; ++i) {
            double uvw[3];
            vel(m.cell_centers[(size_t)i * 3 + 0], m.cell_centers[(size_t)i * 3 + 1],
                m.cell_centers[(size_t)i * 3 + 2], tt, uvw);
            double s = std::sqrt(uvw[0]*uvw[0] + uvw[1]*uvw[1] + uvw[2]*uvw[2]);
            if (s > um) um = s;
        }
        return um;
    };

    // a-posteriori scalar MOOD on the BVD path (compact-DMP -> 1st-order upwind). DEFAULT ON; opt-out ADV_NOMOOD.
    const bool adv_mood = (recon == ADV3_BVD) && (std::getenv("ADV_NOMOOD") == nullptr);
    double t = t_start; int n = 0;   // t_start: resume the velocity clock for segmented (multi-dump) runs
    int max_steps = 100000000;
    if(const char*_e=std::getenv("CFD_MAXSTEP")){int _v=std::atoi(_e); if(_v>0&&_v<max_steps)max_steps=_v;}   // bench cap (per-cell timing)
    for (; n < max_steps && t < t_end; ++n) {
        double dt;
        if (dt_fixed > 0.0) dt = dt_fixed;
        else {
            double um = umax_at(t);
            // The deformation velocity vanishes identically at the turning point t = T/2
            // (the cos(pi t / T) modulation). A step that STARTS there gets um = 0, hence
            // an unbounded dt that the clip below turns into one giant under-resolved
            // jump. Sampling slightly ahead is not enough either: just past the turning
            // point the cosine is still tiny, so the step comes out hundreds of times too
            // large. Use instead the speed the field reaches over the whole period, which
            // gives the smallest CFL-admissible step -- conservative, and it costs one
            // step. The branch cannot fire anywhere else, where um is O(1), so runs that
            // do not resume exactly at the turning point are bit-identical to before.
            if (um <= 1e-12) {
                double uref = 0.0;
                for (int q = 0; q <= 20; ++q)
                    uref = std::max(uref, umax_at(t_start + (t_end - t_start) * q / 20.0));
                um = uref;
            }
            dt = cfl * h_min / std::max(um, 1e-30);
        }
        if (t + dt > t_end) dt = t_end - t;
        if (dt <= 0.0) break;

        // SSP-RK with velocity at the proper stage time (Shu-Osher node times:
        // RK2 stages at t, t+dt; RK3 stages at t, t+dt, t+dt/2).
        advect3d_rhs(m, g, t, vel, L0, recon, ctx, gL, gR, o2, beta_l, beta_s, adv_mood?dt:-1.0);
        if (integrator == 0) {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) g[i] += dt * L0[i];
        } else if (integrator == 1) {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) g1[i] = g[i] + dt * L0[i];
            advect3d_rhs(m, g1, t + dt, vel, L1, recon, ctx, gL, gR, o2, beta_l, beta_s, adv_mood?dt:-1.0);
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) g[i] = 0.5 * g[i] + 0.5 * (g1[i] + dt * L1[i]);
        } else {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) g1[i] = g[i] + dt * L0[i];
            advect3d_rhs(m, g1, t + dt, vel, L1, recon, ctx, gL, gR, o2, beta_l, beta_s, adv_mood?dt:-1.0);
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) g2[i] = 0.75 * g[i] + 0.25 * (g1[i] + dt * L1[i]);
            advect3d_rhs(m, g2, t + 0.5 * dt, vel, L2, recon, ctx, gL, gR, o2, beta_l, beta_s, adv_mood?dt:-1.0);
            #pragma omp parallel for
            for (int i = 0; i < N; ++i)
                g[i] = (1.0/3.0) * g[i] + (2.0/3.0) * (g2[i] + dt * L2[i]);
        }
        t += dt;
    }
    if (out_steps) *out_steps = n;
    if (out_t) *out_t = t;
    if (u3_prof_on()) {
        const U3Prof& r = u3_prof();
        double rtot = r.lsq + r.celld + r.facebvd; if (rtot < 1e-9) rtot = 1.0;
        long nit = u3_celld_iters().load(), nso = u3_celld_solves().load();
        std::printf("E3D_PROF (advect3d) RECON breakdown (summed %.0fms): o2-LSQ %.0fms(%.1f%%)  "
            "cell-D-Newton %.0fms(%.1f%%, avg %.2f Newton iters/solve over %ld solves)  "
            "face-quad+BVD %.0fms(%.1f%%)\n",
            rtot, r.lsq,100*r.lsq/rtot, r.celld,100*r.celld/rtot, nso?(double)nit/nso:0.0, nso,
            r.facebvd,100*r.facebvd/rtot);
    }
    return g;
}

} // namespace cfd
