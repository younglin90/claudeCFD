// cfd/reconstruct3d_bvd_euler.hpp — EULER-specific 3D deg3t-THINC + min-TBV BVD.
//
// Stage 3a. Three face-value candidates per cell/variable, min-TBV BVD selected:
//   (0) o2 P2-quad : the UNLIMITED order-2 LSQ quadratic, evaluated for EVERY cell
//                    (the high-order smooth candidate — captures KH / shock vortices).
//   (1) THINC beta_l: deg3t sigmoid interface profile, sharper (beta=1.6).
//   (2) THINC beta_s: deg3t sigmoid interface profile, gentler (beta=0.8).
//
// Difference vs the ADVECTION BVD (reconstruct3d_bvd.hpp) — three deliberate changes:
//   * SMOOTH candidate is the UNLIMITED o2 P2-quadratic (NOT MUSCL): for Euler the
//     reconstruction must stay high-order in smooth regions; a MUSCL-linear base
//     diffuses vortices. The min-TBV BVD rejects the quadratic at discontinuities
//     (its TBV is large there) and the THINC profile is preferred instead.
//   * NO min/max clamping anywhere. The THINC face value is returned unclamped
//     (qf = qmin + 0.5*rng*(1+fa)); the o2-quad value is the raw polynomial eval.
//     Boundedness / positivity is a MOOD a-posteriori job (Stage 3b), NOT clamping
//     — clamping forces physical extrema down and over-diffuses vortices.
//   * Non-interface cells: the THINC candidate value EQUALS the o2-quad value, so the
//     BVD effectively chooses between {o2-quad, THINC} only where THINC fires
//     (cbar in (1e-6, 1-1e-6)); elsewhere o2-quad is selected.
//
// All cell-D / face-average THINC math is the FROZEN verified closed-form core
// (reconstruct3d_bvd_core.hpp); the interface polynomial P is the o2 P2 gradient +
// Hessian (reconstruct3d_o2.hpp). No Newton, no quadrature, QQ curvature kept.
//
// New file. reconstruct3d_bvd_core.hpp / reconstruct3d_o2.hpp / euler3d.hpp are
// FROZEN; this only consumes them. Reconstruction-only (the flux is never touched).
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/reconstruct3d_bvd_core.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace cfd {

// ── Per-(cell,var) THINC state (mirror of reconstruct3d_bvd.hpp:D3ThincCell).
//    hasint=0 → not an interface cell (THINC falls back to the o2-quad value).
struct D3ThincCellE {
    D3Poly P;            // unit-normal interface quadratic
    double D_l = 0.0;    // cell-D shift for beta_l
    double D_s = 0.0;    // cell-D shift for beta_s
    double kk_l = 0.0;   // beta_l / H
    double kk_s = 0.0;   // beta_s / H
    double qmin = 0.0;   // stencil min (physical) — MAPS the profile (not a clamp)
    double rng = 0.0;    // stencil range qmax-qmin
    int hasint = 0;      // 1 = interface cell
};

// Build the THINC per-cell state for one variable v over all cells. Identical
// construction to the advection BVD (interface-P from o2 grad+Hessian, deg3t cell-D
// per beta). Only interface cells (cbar in (1e-6,1-1e-6)) get hasint=1.
inline void de_build_thinc_cells(const Mesh& m, const ReconCtx3DO2& o2c,
                                 const std::vector<double>& W, int nvar, int v,
                                 double beta_l, double beta_s,
                                 std::vector<D3ThincCellE>& tc) {
    (void)nvar;
    const int N = m.n_cells();
    const double hx = m.s3_h[0], hy = m.s3_h[1], hz = m.s3_h[2];
    const double H = std::cbrt(hx * hy * hz);   // geometric-mean cell size (=h for a cube)
    tc.assign((size_t)N, D3ThincCellE{});

    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        double qbar = W[(size_t)v * N + ci];
        // stencil min/max over the vertex-26 neighbours (o2 stencil).
        double qmin = qbar, qmax = qbar;
        for (int k = 0; k < o2c.max_nb; ++k) {
            int nb = o2c.nb[(size_t)ci * o2c.max_nb + k];
            if (nb < 0) continue;
            double wn = W[(size_t)v * N + nb];
            if (wn < qmin) qmin = wn;
            if (wn > qmax) qmax = wn;
        }
        double rng = qmax - qmin;
        double cbar = (rng > 1e-14) ? (qbar - qmin) / rng : 0.5;
        // not an interface cell → leave hasint=0 (o2-quad fallback at faces).
        if (cbar <= 1e-6 || cbar >= 1.0 - 1e-6 || rng <= 1e-14) continue;

        // inline o2 P2 matvec (interface cells only): co[i] = Σ_k M·(W_nb − qbar).
        double gg[9];
        for (int i = 0; i < 9; ++i) {
            double co = 0.0;
            for (int k = 0; k < o2c.max_nb; ++k) {
                int nbk = o2c.nb[(size_t)ci * o2c.max_nb + k];
                if (nbk < 0) continue;
                co += o2c.M[((size_t)ci * 9 + i) * o2c.max_nb + k] * (W[(size_t)v * N + nbk] - qbar);
            }
            gg[i] = co;
        }
        double gx = gg[0], gy = gg[1], gz = gg[2];
        double hxx = gg[3], hyy = gg[4], hzz = gg[5];
        double hxy = gg[6], hxz = gg[7], hyz = gg[8];
        double gnorm = std::sqrt(gx * gx + gy * gy + gz * gz);
        if (gnorm < 1e-30) continue;   // no interface normal

        // unit normal n = grad/|grad|.
        double inv = 1.0 / gnorm;
        double nx = gx * inv, ny = gy * inv, nz = gz * inv;
        double Hxx = hxx, Hyy = hyy, Hzz = hzz, Hxy = hxy, Hxz = hxz, Hyz = hyz;
        // Hg = H * grad.
        double Hg_x = Hxx * gx + Hxy * gy + Hxz * gz;
        double Hg_y = Hxy * gx + Hyy * gy + Hyz * gz;
        double Hg_z = Hxz * gx + Hyz * gy + Hzz * gz;
        double inv3 = inv * inv * inv;   // 1/gnorm^3
        // normal-gradient n_ij = ∂n_i/∂x_j = H_ij/gnorm − g_i·Hg_j/gnorm^3.
        double n_xx = Hxx * inv - gx * Hg_x * inv3;
        double n_yy = Hyy * inv - gy * Hg_y * inv3;
        double n_zz = Hzz * inv - gz * Hg_z * inv3;
        double n_xy = Hxy * inv - gx * Hg_y * inv3;
        double n_yx = Hxy * inv - gy * Hg_x * inv3;   // H symmetric
        double n_xz = Hxz * inv - gx * Hg_z * inv3;
        double n_zx = Hxz * inv - gz * Hg_x * inv3;
        double n_yz = Hyz * inv - gy * Hg_z * inv3;
        double n_zy = Hyz * inv - gz * Hg_y * inv3;

        D3ThincCellE& t = tc[ci];
        t.P.c[0] = nx; t.P.c[1] = ny; t.P.c[2] = nz;
        t.P.c[3] = 0.5 * n_xx; t.P.c[4] = 0.5 * n_yy; t.P.c[5] = 0.5 * n_zz;
        t.P.c[6] = 0.5 * (n_xy + n_yx);
        t.P.c[7] = 0.5 * (n_xz + n_zx);
        t.P.c[8] = 0.5 * (n_yz + n_zy);

        double Q = 2.0 * cbar - 1.0;
        t.kk_l = beta_l / H;
        t.kk_s = beta_s / H;
        t.D_l = deg3t3d_cellD(t.P, t.kk_l, hx, hy, hz, Q);
        t.D_s = deg3t3d_cellD(t.P, t.kk_s, hx, hy, hz, Q);
        t.qmin = qmin;
        t.rng = rng;
        t.hasint = 1;
    }
}

// Evaluate one THINC candidate (D,kk) at a face for the cell owning this side.
// NO CLAMP (deliberate): qf = qmin + 0.5*rng*(1+fa), returned raw. The stencil
// qmin/rng only MAP the [-1,1] profile to physical units, they do not clamp.
// Non-interface cell → returns q_o2 (the o2-quad fallback value).
inline double de_thinc_face_val(const Mesh& m, const std::vector<D3ThincCellE>& tc,
                                int f, int cell, double D, double kk, double q_o2) {
    const D3ThincCellE& t = tc[cell];
    if (!t.hasint) return q_o2;   // non-interface → candidate equals the o2-quad value
    const double* fn = &m.face_normals[(size_t)f * 3];
    int axis = 0;
    double an = std::fabs(fn[0]), a1 = std::fabs(fn[1]), a2 = std::fabs(fn[2]);
    if (a1 > an) { an = a1; axis = 1; }
    if (a2 > an) { an = a2; axis = 2; }
    double off = m.face_centers[(size_t)f * 3 + axis] - m.cell_centers[(size_t)cell * 3 + axis];
    double hx = m.s3_h[0], hy = m.s3_h[1], hz = m.s3_h[2];
    double h0, h1;
    if (axis == 0)      { h0 = hy; h1 = hz; }
    else if (axis == 1) { h0 = hx; h1 = hz; }
    else                { h0 = hx; h1 = hy; }
    double fa = deg3t3d_face_avg(t.P, D, kk, axis, off, h0, h1);   // ∈[-1,1]
    return t.qmin + 0.5 * t.rng * (1.0 + fa);                      // UNCLAMPED
}

// Public entry: fill the final WL/WR (owner/neighbour face values) for nvar
// PRIMITIVE variables after the min-TBV BVD over {o2-quad, THINC_l, THINC_s}.
inline void reconstruct3d_bvd_euler(const Mesh& m, const ReconCtx3DO2& o2c,
                                    const std::vector<double>& W, int nvar,
                                    std::vector<double>& WL, std::vector<double>& WR,
                                    double beta_l = 1.6, double beta_s = 0.8) {
    const int N = m.n_cells(), Nf = m.n_faces();
    WL.assign((size_t)nvar * Nf, 0.0);
    WR.assign((size_t)nvar * Nf, 0.0);

    // ── a-priori strong-shock sensor (Colella-Woodward 1984 flattening idea).
    //    The smooth candidate here is the UNLIMITED o2-quad P2 and the THINC
    //    candidate over-sharpens; both drive strong-shock-into-low-density
    //    (Langseth) toward vacuum, faster than a-posteriori MOOD can floor the
    //    cell average. Where a cell sees a large pressure jump to a face-neighbour
    //    AND is compressing (genuine strong shock), force FIRST-ORDER (qbar) for
    //    that cell's faces — positivity-safe. Contacts/slip lines have ~0 pressure
    //    jump → keep THINC/o2-quad sharp (the vortex is preserved). Reconstruction-
    //    only (Riemann flux untouched). env BVD_SHOCK=0 off, BVD_SHOCK_P threshold
    //    (Colella-Woodward uses 1/3), BVD_SHOCK_NOCOMP=1 drops the compression gate.
    static const int    SHK  = []{ const char* e=std::getenv("BVD_SHOCK");        return (e&&e[0])?std::atoi(e):0;   }();  // opt-in (MLP is the real strong-shock fix)
    static const double SHKP = []{ const char* e=std::getenv("BVD_SHOCK_P");      return (e&&e[0])?std::atof(e):0.5; }();
    static const int    SHKC = []{ const char* e=std::getenv("BVD_SHOCK_NOCOMP"); return (e&&e[0])?std::atoi(e):0;   }();
    std::vector<unsigned char> shockcell;
    if (SHK && nvar >= 5) {
        const int ip = nvar - 1;   // pressure index (layout rho,u,v,w,p)
        shockcell.assign((size_t)N, 0);
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double pc = W[(size_t)ip * N + ci];
            double rp = 0.0, divv = 0.0;
            for (int f : m.cell_faces[ci]) {
                bool is_owner = (m.face_owner[f] == ci);
                int nb = is_owner ? m.face_neighbour[f] : m.face_owner[f];
                if (nb < 0) continue;
                double pn = W[(size_t)ip * N + nb];
                double mn = std::min(pc, pn);
                double jump = (mn > 0.0) ? std::fabs(pc - pn) / mn : 1e300;
                if (jump > rp) rp = jump;
                const double* fn = &m.face_normals[(size_t)f * 3];
                double sgn = is_owner ? 1.0 : -1.0;   // face normal points owner→neighbour
                for (int d = 0; d < 3; ++d)
                    divv += 0.5 * (W[(size_t)(1+d)*N + nb] + W[(size_t)(1+d)*N + ci]) * fn[d] * sgn;
            }
            shockcell[ci] = (rp > SHKP && (SHKC || divv < 0.0)) ? 1 : 0;
        }
    }

    // per-variable scratch (bounded memory): o2-quad face arrays + THINC face arrays.
    std::vector<double> WLq((size_t)Nf), WRq((size_t)Nf);   // o2-quad owner/neighbour
    std::vector<double> WLf_l((size_t)Nf), WRf_l((size_t)Nf);
    std::vector<double> WLf_s((size_t)Nf), WRf_s((size_t)Nf);
    std::vector<double> g;                                  // N*9 o2 coeffs (current var)
    std::vector<D3ThincCellE> tc;

    for (int v = 0; v < nvar; ++v) {
        // (0) UNLIMITED o2 P2-quad coeffs for ALL cells.
        reconstruct3d_o2_coeffs(m, o2c, W, nvar, v, g);

        // o2-quad face value (owner side WLq, neighbour side WRq). dof = face_centre
        // − cell_centre (per side). raw polynomial eval — no limiter, no clamp.
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            int o = m.face_owner[f], n = m.face_neighbour[f];
            double fcx = m.face_centers[(size_t)f * 3 + 0];
            double fcy = m.face_centers[(size_t)f * 3 + 1];
            double fcz = m.face_centers[(size_t)f * 3 + 2];
            // owner side
            {
                double wc = W[(size_t)v * N + o];
                double dx = fcx - m.cell_centers[(size_t)o * 3 + 0];
                double dy = fcy - m.cell_centers[(size_t)o * 3 + 1];
                double dz = fcz - m.cell_centers[(size_t)o * 3 + 2];
                const double* gg = &g[(size_t)o * 9];
                WLq[f] = wc + gg[0]*dx + gg[1]*dy + gg[2]*dz
                       + 0.5*gg[3]*dx*dx + 0.5*gg[4]*dy*dy + 0.5*gg[5]*dz*dz
                       + gg[6]*dx*dy + gg[7]*dx*dz + gg[8]*dy*dz;
            }
            // neighbour side (from the neighbour's own quadratic)
            if (n >= 0) {
                double wc = W[(size_t)v * N + n];
                double dx = fcx - m.cell_centers[(size_t)n * 3 + 0];
                double dy = fcy - m.cell_centers[(size_t)n * 3 + 1];
                double dz = fcz - m.cell_centers[(size_t)n * 3 + 2];
                const double* gg = &g[(size_t)n * 9];
                WRq[f] = wc + gg[0]*dx + gg[1]*dy + gg[2]*dz
                       + 0.5*gg[3]*dx*dx + 0.5*gg[4]*dy*dy + 0.5*gg[5]*dz*dz
                       + gg[6]*dx*dy + gg[7]*dx*dz + gg[8]*dy*dz;
            } else {
                WRq[f] = WLq[f];
            }
        }

        // (1,2) THINC candidate face arrays (unclamped; non-interface = o2-quad val).
        de_build_thinc_cells(m, o2c, W, nvar, v, beta_l, beta_s, tc);
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            int o = m.face_owner[f], n = m.face_neighbour[f];
            WLf_l[f] = de_thinc_face_val(m, tc, f, o, tc[o].D_l, tc[o].kk_l, WLq[f]);
            WLf_s[f] = de_thinc_face_val(m, tc, f, o, tc[o].D_s, tc[o].kk_s, WLq[f]);
            if (n >= 0) {
                WRf_l[f] = de_thinc_face_val(m, tc, f, n, tc[n].D_l, tc[n].kk_l, WRq[f]);
                WRf_s[f] = de_thinc_face_val(m, tc, f, n, tc[n].D_s, tc[n].kk_s, WRq[f]);
            } else {
                WRf_l[f] = WLf_l[f];
                WRf_s[f] = WLf_s[f];
            }
        }

        // default = o2-quad everywhere; interface cells may overwrite below.
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            WL[(size_t)v * Nf + f] = WLq[f];
            WR[(size_t)v * Nf + f] = WRq[f];
        }

        // min-TBV per cell over {o2-quad, THINC_l, THINC_s}. Only interface cells
        // have a real choice (elsewhere the THINC arrays == the o2-quad arrays, so
        // any pick writes the same value). TBV_i(c) = Σ_{f∈cell ci} |q_c(ci-side) − W_nb|.
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            bool shk = !shockcell.empty() && shockcell[ci];
            if (!tc[ci].hasint && !shk) continue;   // non-interface, non-shock → keep o2-quad (the default)
            double qbar_ci = W[(size_t)v * N + ci];
            if (shk) {
                // strong-shock cell: force FIRST-ORDER (qbar) on this cell's faces.
                for (int f : m.cell_faces[ci]) {
                    bool is_owner = (m.face_owner[f] == ci);
                    if (is_owner) WL[(size_t)v * Nf + f] = qbar_ci;
                    else          WR[(size_t)v * Nf + f] = qbar_ci;
                }
                continue;
            }
            (void)qbar_ci;
            double tbv_q = 0.0, tbv_l = 0.0, tbv_s = 0.0;
            for (int f : m.cell_faces[ci]) {
                bool is_owner = (m.face_owner[f] == ci);
                int nb = is_owner ? m.face_neighbour[f] : m.face_owner[f];
                double qnb = (nb >= 0) ? W[(size_t)v * N + nb] : W[(size_t)v * N + ci];
                double qq = is_owner ? WLq[f]   : WRq[f];
                double ql = is_owner ? WLf_l[f] : WRf_l[f];
                double qs = is_owner ? WLf_s[f] : WRf_s[f];
                tbv_q += std::fabs(qq - qnb);
                tbv_l += std::fabs(ql - qnb);
                tbv_s += std::fabs(qs - qnb);
            }
            int pick = 0; double best = tbv_q;
            if (tbv_l < best) { best = tbv_l; pick = 1; }
            if (tbv_s < best) { best = tbv_s; pick = 2; }

            for (int f : m.cell_faces[ci]) {
                bool is_owner = (m.face_owner[f] == ci);
                double val;
                if (pick == 0)      val = is_owner ? WLq[f]   : WRq[f];
                else if (pick == 1) val = is_owner ? WLf_l[f] : WRf_l[f];
                else                val = is_owner ? WLf_s[f] : WRf_s[f];
                if (is_owner) WL[(size_t)v * Nf + f] = val;
                else          WR[(size_t)v * Nf + f] = val;
            }
        }
    }
}

} // namespace cfd
