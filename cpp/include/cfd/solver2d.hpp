// cfd/solver2d.hpp — 2D finite-volume time march for scalar advection.
// Port of the rhs / time loop in solver.py for the 2D first-order scalar path:
// midpoint face quadrature, upwind flux (velocity sampled at the face centre),
// transmissive BC, forward-Euler / SSP-RK2 / SSP-RK3 in time.
// (Euler2D + T-MLP-u-L reconstruction extend this next.)
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/advection.hpp"
#include "cfd/reconstruct2d.hpp"
#include "cfd/reconstruct_bvd.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace cfd {

struct Solve2DResult { std::vector<double> U; int n_steps = 0; double t = 0.0; };

// Reconstruction modes for the scalar 2D solver.
enum ReconMode { RECON_FIRST = 0, RECON_BJ_VERTEX = 1, RECON_TMLPU_L = 2, RECON_BVD = 4 };

// dUdt = -(1/V) sum_f F_f*area. Scalar advection, transmissive BC.
inline void adv2d_rhs(const Mesh& m, const Advection2D& eq,
                      const std::vector<double>& U, std::vector<double>& dUdt,
                      int recon = RECON_FIRST, const ReconCtx* ctx = nullptr,
                      std::vector<double>* WL = nullptr, std::vector<double>* WR = nullptr,
                      const ReconCtxO2* ctx_o2 = nullptr) {
    const int Nf = m.n_faces();
    const bool ho = (recon != RECON_FIRST) && ctx;
    // TMLPU_GATED: genuine linear T-MLP-u (LSQ-residual-gated cicsam/van_leer +
    // uncapped vertex-LMP). thr=smoothness_threshold, Co=Hyper-C Courant, cap=psi cap.
    static const bool TG = std::getenv("TMLPU_GATED") != nullptr;
    static const double TG_THR = []{ const char* e=std::getenv("TMLPU_THR"); return e?std::atof(e):0.20; }();
    static const double TG_CO  = []{ const char* e=std::getenv("TMLPU_CO"); return e?std::atof(e):0.38; }();
    static const double TG_CAP = []{ const char* e=std::getenv("TMLPU_CAP"); return e?std::atof(e):2.0; }();
    if (TG && ctx && recon == RECON_BVD)   // gated applies to the BVD scheme slot only
        reconstruct_tmlpu_gated(m, *ctx, U, 1, *WL, *WR, TG_THR, TG_CO, TG_CAP);
    else if (recon == RECON_BVD && ctx && ctx_o2) {
        // BVD_ABVD needs the per-face advection speed for the Eq.26 upwind endpoint
        // (midpoint fallback is NOT TVD -> diverges). Compute a.n once per call.
        static const bool ABVD = std::getenv("BVD_ABVD") != nullptr;
        std::vector<double> fa;
        if (ABVD) {
            fa.resize((size_t)Nf);
            #pragma omp parallel for
            for (int f = 0; f < Nf; ++f)
                fa[f] = eq.a_dot_n(m.face_centers[f*2+0], m.face_centers[f*2+1],
                                   m.face_normals[f*2+0], m.face_normals[f*2+1]);
        }
        reconstruct_bvd(m, *ctx, *ctx_o2, U, 1, *WL, *WR, false, 0,
                        nullptr, 0.0, 0.0, 0.0, false, ABVD ? fa.data() : nullptr);
    }
    else if (ho) reconstruct_bj_vertex(m, *ctx, U, 1, *WL, *WR, recon == RECON_TMLPU_L);
    const int N = m.n_cells();
    // Pass 1 (parallel): area-weighted flux per face. Pass 2 (parallel): each cell
    // gathers its own faces (race-free), matching the Euler 2D path.
    std::vector<double> Ff((size_t)Nf);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double nx = m.face_normals[f * 2 + 0], ny = m.face_normals[f * 2 + 1];
        double fx = m.face_centers[f * 2 + 0], fy = m.face_centers[f * 2 + 1];
        double uL, uR;
        if (n < 0)   { uL = U[o]; uR = uL; }             // Neumann boundary: cell average, no recon (2026-07-03)
        else if (ho) { uL = (*WL)[f]; uR = (*WR)[f]; }
        else         { uL = U[o]; uR = U[n]; }
        double adn = eq.a_dot_n(fx, fy, nx, ny);
        Ff[f] = upwind_advection(adn, uL, uR) * m.face_areas[f];
    }
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double acc = 0.0;
        for (int f : m.cell_faces[i]) acc += (m.face_owner[f] == i ? -1.0 : 1.0) * Ff[f];
        dUdt[i] = acc / m.cell_volumes[i];
    }
}

// integrator: 0=FE, 1=SSP-RK2, 2=SSP-RK3
inline Solve2DResult solve_adv2d(const Mesh& m, const Advection2D& eq,
                                 const std::vector<double>& U0, double t_end,
                                 int integrator = 2, double cfl = 0.4,
                                 double dt_fixed = -1.0, int max_steps = 1000000,
                                 int recon = RECON_FIRST, const ReconCtx* ctx = nullptr,
                                 const ReconCtxO2* ctx_o2 = nullptr) {
    const int N = m.n_cells(), Nf = m.n_faces();
    std::vector<double> U = U0, L0(N), U1(N), L1(N), U2(N), L2(N);
    std::vector<double> WL, WR;  // reused reconstruction buffers
    // Zalesak FCT (flux-corrected transport): low-order monotone upwind + BVD/
    // T-MLP-u high-order anti-diffusion limited cell-wise to the local maximum
    // principle -> sharp contacts WITHOUT the downwind ratcheting, fully LINEAR.
    static const bool FCT = std::getenv("FCT_ADV") != nullptr;
    double cur_dt = 0.0;
    auto fct_rate = [&](const std::vector<double>& s, std::vector<double>& out) {
        const bool ho = (recon != RECON_FIRST) && ctx;
        if (recon == RECON_BVD && ctx && ctx_o2) reconstruct_bvd(m, *ctx, *ctx_o2, s, 1, WL, WR, false, 0);
        else if (ho) reconstruct_bj_vertex(m, *ctx, s, 1, WL, WR, recon == RECON_TMLPU_L);
        std::vector<double> FL(Nf), A(Nf);   // low-order flux*area, anti-diffusive flux*area
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            int o=m.face_owner[f], n=m.face_neighbour[f];
            double adn=eq.a_dot_n(m.face_centers[f*2],m.face_centers[f*2+1],m.face_normals[f*2],m.face_normals[f*2+1]);
            double area=m.face_areas[f];
            double fl = upwind_advection(adn, s[o], (n>=0?s[n]:s[o]))*area;
            double uL = ho?WL[f]:s[o], uR = (n>=0)?(ho?WR[f]:s[n]):uL;
            double fh = upwind_advection(adn, uL, uR)*area;
            FL[f]=fl; A[f]=fh-fl;
        }
        // low-order update Ud, local bounds [mn,mx] from s.
        std::vector<double> Ud(N), mn(N), mx(N), Pp(N,0.0), Pm(N,0.0);
        #pragma omp parallel for
        for (int i=0;i<N;++i){ double acc=0, lo=s[i], hi=s[i];
            for(int f:m.cell_faces[i]){ double sg=(m.face_owner[f]==i)?-1.0:1.0; acc += sg*FL[f];
                int o=m.face_owner[f],nn=m.face_neighbour[f],nb=(o==i)?nn:o;
                if(nb>=0){ lo=std::min(lo,s[nb]); hi=std::max(hi,s[nb]); } }
            Ud[i]=s[i]+cur_dt*acc/m.cell_volumes[i]; mn[i]=lo; mx[i]=hi;
            // accumulate anti-diffusive contributions (sign into cell i)
            double pp=0,pm=0;
            for(int f:m.cell_faces[i]){ double sg=(m.face_owner[f]==i)?-1.0:1.0;
                double c=sg*A[f]/m.cell_volumes[i]; if(c>0)pp+=c; else pm-=c; }
            Pp[i]=pp; Pm[i]=pm; }
        std::vector<double> Rp(N), Rm(N);
        #pragma omp parallel for
        for (int i=0;i<N;++i){ double qp=(mx[i]-Ud[i])/std::max(cur_dt,1e-30), qm=(Ud[i]-mn[i])/std::max(cur_dt,1e-30);
            Rp[i]=Pp[i]>1e-30?std::min(1.0,qp/Pp[i]):0.0; Rm[i]=Pm[i]>1e-30?std::min(1.0,qm/Pm[i]):0.0; }
        // per-face limit coeff C_f, then gather limited rate.
        std::vector<double> C(Nf);
        #pragma omp parallel for
        for (int f=0;f<Nf;++f){ int o=m.face_owner[f],n=m.face_neighbour[f];
            if(n<0){ C[f]=0.0; continue; }
            // A[f] oriented owner->neighbour: A>0 raises owner-rate? sg(owner)=-1 -> lowers owner, raises neighbour.
            double cf = (A[f]<=0) ? std::min(Rm[o],Rp[n]) : std::min(Rp[o],Rm[n]);
            C[f]=cf; }
        #pragma omp parallel for
        for (int i=0;i<N;++i){ double acc=0;
            for(int f:m.cell_faces[i]){ double sg=(m.face_owner[f]==i)?-1.0:1.0; acc += sg*(FL[f]+C[f]*A[f]); }
            out[i]=acc/m.cell_volumes[i]; }
    };
    auto RHS = [&](const std::vector<double>& s, std::vector<double>& out) {
        if (FCT) fct_rate(s, out);
        else adv2d_rhs(m, eq, s, out, recon, ctx, &WL, &WR, ctx_o2);
    };
    // characteristic length per cell ~ volume / max adjacent face area; use min.
    double h_min = 1e300;
    for (int i = 0; i < N; ++i) {
        double amax = 0.0;
        for (int f : m.cell_faces[i]) amax = std::max(amax, m.face_areas[f]);
        double h = m.cell_volumes[i] / std::max(amax, 1e-30);
        h_min = std::min(h_min, h);
    }
    double t = 0.0; int n = 0;
    if(const char*_e=std::getenv("CFD_MAXSTEP")){int _v=std::atoi(_e); if(_v>0&&_v<max_steps)max_steps=_v;}   // bench cap (per-cell timing)
    for (; n < max_steps && t < t_end; ++n) {
        double dt;
        if (dt_fixed > 0.0) dt = dt_fixed;
        else {
            // max |a.n| over faces as wave-speed proxy
            double wmax = 0.0;
            for (int f = 0; f < m.n_faces(); ++f) {
                double adn = std::fabs(eq.a_dot_n(m.face_centers[f*2], m.face_centers[f*2+1],
                                                  m.face_normals[f*2], m.face_normals[f*2+1]));
                wmax = std::max(wmax, adn);
            }
            dt = cfl * h_min / std::max(wmax, 1e-30);
        }
        if (t + dt > t_end) dt = t_end - t;
        if (dt <= 0.0) break;
        cur_dt = dt;

        RHS(U, L0);
        if (integrator == 0) {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) U[i] += dt * L0[i];
        } else if (integrator == 1) {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) U1[i] = U[i] + dt * L0[i];
            RHS(U1, L1);
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) U[i] = 0.5 * U[i] + 0.5 * (U1[i] + dt * L1[i]);
        } else { // SSP-RK3 (Shu-Osher)
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) U1[i] = U[i] + dt * L0[i];
            RHS(U1, L1);
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) U2[i] = 0.75 * U[i] + 0.25 * (U1[i] + dt * L1[i]);
            RHS(U2, L2);
            #pragma omp parallel for
            for (int i = 0; i < N; ++i)
                U[i] = (1.0 / 3.0) * U[i] + (2.0 / 3.0) * (U2[i] + dt * L2[i]);
        }
        t += dt;
    }
    return Solve2DResult{std::move(U), n, t};
}

} // namespace cfd
