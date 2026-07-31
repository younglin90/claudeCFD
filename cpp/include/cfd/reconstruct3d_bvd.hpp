// cfd/reconstruct3d_bvd.hpp — 3D deg3t-THINC-QQ + min-TBV BVD reconstruction.
//
// Stage 2b final integration. Combines three face-value candidates per cell/var:
//   (0) MUSCL          : reconstruct_bj_vertex_3d (the diffusive but robust base).
//   (1) THINC-QQ beta_l: deg3t sigmoid interface profile, sharper (beta=1.6).
//   (2) THINC-QQ beta_s: deg3t sigmoid interface profile, gentler (beta=0.8).
// and picks, per cell and variable, the candidate with the smallest Total
// Boundary Variation (min-TBV BVD, Sun-Wang-Xiao). The THINC interface polynomial
// P is the unit-normal quadratic built from the o2 P2-LSQ gradient + Hessian; all
// cell-D / face-average math is the FROZEN, verified closed-form core
// (reconstruct3d_bvd_core.hpp) — no Newton, no quadrature, QQ curvature kept.
//
// Honors the project constraints: reconstruction-only (no artificial diffusion),
// exact-cell-D (the core solves the cell-average constraint via Cardano), QQ
// curvature (full quadratic P), closed-form (no Gauss, no per-cell Newton).
//
// New file. reconstruct3d_bvd_core.hpp & reconstruct3d_o2.hpp are FROZEN; this
// only consumes them. Builds the MUSCL candidate internally via reconstruct3d.hpp.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/reconstruct3d_bvd_core.hpp"
#include "cfd/reconstruct3d_thinc_tanh.hpp"   // tanh THINC/QQ baseline + thinc3d_* dispatch (env THINCQQ_SIG_TANH)
#include <vector>
#include <cmath>
#include <algorithm>

namespace cfd {

// ── Per-(cell,var) THINC state. hasint=0 → not an interface cell (THINC falls
//    back to MUSCL, i.e. its face candidates are never preferred / are skipped).
struct D3ThincCell {
    D3Poly P;            // unit-normal interface quadratic
    double D_l = 0.0;    // cell-D shift for beta_l
    double D_s = 0.0;    // cell-D shift for beta_s
    double kk_l = 0.0;   // beta_l / H
    double kk_s = 0.0;   // beta_s / H
    double qmin = 0.0;   // stencil min (physical)
    double rng = 0.0;    // stencil range qmax-qmin
    int hasint = 0;      // 1 = interface cell
};

// Build the THINC per-cell state for one variable v over all cells.
inline void d3_build_thinc_cells(const Mesh& m, const ReconCtx3DO2& o2c,
                                 const std::vector<double>& W, int nvar, int v,
                                 double beta_l, double beta_s,
                                 std::vector<D3ThincCell>& tc) {
    (void)nvar;
    const int N = m.n_cells();
    const double hx = m.s3_h[0], hy = m.s3_h[1], hz = m.s3_h[2];
    const double H = std::cbrt(hx * hy * hz);   // geometric-mean cell size (=h for a cube)
    tc.assign((size_t)N, D3ThincCell{});

    // o2 P2 coeffs are computed INLINE per interface cell only (the 26-stencil
    // matvec is the dominant cost; ~97% of cells are non-interface → skip them).
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
        // not an interface cell → leave hasint=0 (MUSCL fallback at faces).
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
        // Hessian (symmetric).
        double Hxx = hxx, Hyy = hyy, Hzz = hzz;
        double Hxy = hxy, Hxz = hxz, Hyz = hyz;
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
        double n_yx = Hxy * inv - gy * Hg_x * inv3;   // H is symmetric (Hyx=Hxy)
        double n_xz = Hxz * inv - gx * Hg_z * inv3;
        double n_zx = Hxz * inv - gz * Hg_x * inv3;
        double n_yz = Hyz * inv - gy * Hg_z * inv3;
        double n_zy = Hyz * inv - gz * Hg_y * inv3;

        D3ThincCell& t = tc[ci];
        t.P.c[0] = nx; t.P.c[1] = ny; t.P.c[2] = nz;
        t.P.c[3] = 0.5 * n_xx; t.P.c[4] = 0.5 * n_yy; t.P.c[5] = 0.5 * n_zz;
        t.P.c[6] = 0.5 * (n_xy + n_yx);
        t.P.c[7] = 0.5 * (n_xz + n_zx);
        t.P.c[8] = 0.5 * (n_yz + n_zy);

        double Q = 2.0 * cbar - 1.0;
        t.kk_l = beta_l / H;
        t.kk_s = beta_s / H;
        thinc3d_cellD_both(t.P, t.kk_l, t.kk_s, hx, hy, hz, Q, t.D_l, t.D_s);  // moments shared
        t.qmin = qmin;
        t.rng = rng;
        t.hasint = 1;
    }
}

// Evaluate one THINC candidate (beta given by D,kk) at a face for the cell `cell`
// that owns this side of the face. Returns the physical face value clamped to the
// cell's [qmin,qmax]. If the cell is not an interface cell, returns qbar (1st order).
inline double d3_thinc_face_val(const Mesh& m, const std::vector<D3ThincCell>& tc,
                                int f, int cell, double D, double kk, double qbar) {
    const D3ThincCell& t = tc[cell];
    if (!t.hasint) return qbar;
    const double* fn = &m.face_normals[(size_t)f * 3];
    // axis = argmax |face normal component| (structured hex → exactly one nonzero).
    int axis = 0;
    double an = std::fabs(fn[0]), a1 = std::fabs(fn[1]), a2 = std::fabs(fn[2]);
    if (a1 > an) { an = a1; axis = 1; }
    if (a2 > an) { an = a2; axis = 2; }
    // signed offset of the face centre from the cell centre along that axis.
    double off = m.face_centers[(size_t)f * 3 + axis] - m.cell_centers[(size_t)cell * 3 + axis];
    // tangential cell sizes per the core's axis rule.
    double hx = m.s3_h[0], hy = m.s3_h[1], hz = m.s3_h[2];
    double h0, h1;
    if (axis == 0)      { h0 = hy; h1 = hz; }
    else if (axis == 1) { h0 = hx; h1 = hz; }
    else                { h0 = hx; h1 = hy; }
    double fa = thinc3d_face_avg(t.P, D, kk, axis, off, h0, h1);   // ∈[-1,1] (deg3t closed-form or tanh quad baseline)
    double qf = t.qmin + 0.5 * t.rng * (1.0 + fa);
    double qlo = t.qmin, qhi = t.qmin + t.rng;
    return qf < qlo ? qlo : (qf > qhi ? qhi : qf);
}

// BETA-SHARED face eval: both betas at this (cell,face) sharing the face geometry + moments
// (computed ONCE). Writes the clamped owner/neighbour face values for beta_l and beta_s.
inline void d3_thinc_face_val_both(const Mesh& m, const std::vector<D3ThincCell>& tc,
                                   int f, int cell, double qbar, double& qf_l, double& qf_s) {
    const D3ThincCell& t = tc[cell];
    if (!t.hasint) { qf_l = qf_s = qbar; return; }
    const double* fn = &m.face_normals[(size_t)f * 3];
    int axis = 0; double an = std::fabs(fn[0]), a1 = std::fabs(fn[1]), a2 = std::fabs(fn[2]);
    if (a1 > an) { an = a1; axis = 1; }
    if (a2 > an) { an = a2; axis = 2; }
    double off = m.face_centers[(size_t)f * 3 + axis] - m.cell_centers[(size_t)cell * 3 + axis];
    double hx = m.s3_h[0], hy = m.s3_h[1], hz = m.s3_h[2]; double h0, h1;
    if (axis == 0)      { h0 = hy; h1 = hz; }
    else if (axis == 1) { h0 = hx; h1 = hz; }
    else                { h0 = hx; h1 = hy; }
    double fa_l, fa_s;
    thinc3d_face_avg_both(t.P, t.D_l, t.D_s, t.kk_l, t.kk_s, axis, off, h0, h1, fa_l, fa_s);
    double qlo = t.qmin, qhi = t.qmin + t.rng;
    double a = t.qmin + 0.5 * t.rng * (1.0 + fa_l); qf_l = a < qlo ? qlo : (a > qhi ? qhi : a);
    double b = t.qmin + 0.5 * t.rng * (1.0 + fa_s); qf_s = b < qlo ? qlo : (b > qhi ? qhi : b);
}

// Build the two THINC face-value arrays (owner side in WL*, neighbour side in WR*).
inline void d3_thinc_face_arrays(const Mesh& m, const std::vector<D3ThincCell>& tc,
                                 const std::vector<double>& W, int nvar, int v,
                                 std::vector<double>& WLf_l, std::vector<double>& WRf_l,
                                 std::vector<double>& WLf_s, std::vector<double>& WRf_s) {
    const int N = m.n_cells(), Nf = m.n_faces();
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double qo = W[(size_t)v * N + o];
        d3_thinc_face_val_both(m, tc, f, o, qo, WLf_l[(size_t)v*Nf+f], WLf_s[(size_t)v*Nf+f]);
        if (n >= 0) {
            double qn = W[(size_t)v * N + n];
            d3_thinc_face_val_both(m, tc, f, n, qn, WRf_l[(size_t)v*Nf+f], WRf_s[(size_t)v*Nf+f]);
        } else {
            WRf_l[(size_t)v * Nf + f] = WLf_l[(size_t)v * Nf + f];
            WRf_s[(size_t)v * Nf + f] = WLf_s[(size_t)v * Nf + f];
        }
    }
}

// Public entry: emit the final WL/WR after the min-TBV BVD selection over the
// three candidates {MUSCL, THINC_l, THINC_s}, per cell and variable.
inline void reconstruct3d_bvd(const Mesh& m, const ReconCtx3DO2& o2c,
                              const ReconCtx3D& bjc, const std::vector<double>& W,
                              int nvar, std::vector<double>& WL, std::vector<double>& WR,
                              double beta_l = 1.6, double beta_s = 0.8) {
    const int N = m.n_cells(), Nf = m.n_faces();
    // env override of the THINC sharpness betas (gentler beta => less over-sharpening,
    // better positivity on strong-shock / low-density cases). Mirrors the 2D BVD_BETA_L/S.
    static const double BL = []{ const char* e=std::getenv("BVD_BETA_L"); return (e&&e[0])?std::atof(e):-1.0; }();
    static const double BS = []{ const char* e=std::getenv("BVD_BETA_S"); return (e&&e[0])?std::atof(e):-1.0; }();
    if (BL > 0.0) beta_l = BL;
    if (BS > 0.0) beta_s = BS;

    // (0) MUSCL candidate (owner side WLm, neighbour side WRm) — all vars at once.
    std::vector<double> WLm, WRm;
    reconstruct_bj_vertex_3d(m, bjc, W, nvar, WLm, WRm);
    WL = WLm; WR = WRm;   // default everywhere = MUSCL; interface cells overwrite below

    // ── a-priori strong-shock sensor (Colella-Woodward 1984 flattening idea).
    //    The THINC face value is already clamped to [qmin,qmax]; divergence on
    //    strong-shock-into-low-density (Langseth) is NOT a face overshoot but the
    //    THINC sharpening draining the cell AVERAGE toward vacuum through the flux.
    //    A-posteriori MOOD cannot catch this (it develops gradually, faster than a
    //    cell-average floor). Fix: where a cell sees a large pressure jump to a
    //    face-neighbour AND is compressing (genuine strong shock), VETO THINC →
    //    robust MUSCL (positivity-safe; MUSCL completes Langseth). Contacts/slip
    //    lines have ~0 pressure jump → keep THINC sharp (the vortex is preserved).
    //    Reconstruction-only (the Riemann flux is untouched). env BVD_SHOCK=0 off,
    //    BVD_SHOCK_P = relative-pressure-jump threshold (Colella-Woodward uses 1/3).
    static const int    SHK  = []{ const char* e=std::getenv("BVD_SHOCK");   return (e&&e[0])?std::atoi(e):0;   }();  // opt-in
    static const double SHKP = []{ const char* e=std::getenv("BVD_SHOCK_P"); return (e&&e[0])?std::atof(e):0.5; }();
    static const int    SHKC = []{ const char* e=std::getenv("BVD_SHOCK_NOCOMP"); return (e&&e[0])?std::atoi(e):0; }();
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
                // net velocity convergence (compression) along outward face normals.
                const double* fn = &m.face_normals[(size_t)f * 3];
                double sgn = is_owner ? 1.0 : -1.0;   // face normal points owner→neighbour
                for (int d = 0; d < 3; ++d)
                    divv += 0.5 * (W[(size_t)(1+d)*N + nb] + W[(size_t)(1+d)*N + ci]) * fn[d] * sgn;
            }
            // strong compressive shock: big pressure jump AND net compression (divv<0).
            shockcell[ci] = (rp > SHKP && (SHKC || divv < 0.0)) ? 1 : 0;
        }
    }

    // THINC face arrays per variable (built inside the var loop to bound memory).
    std::vector<double> WLf_l((size_t)nvar * Nf), WRf_l((size_t)nvar * Nf);
    std::vector<double> WLf_s((size_t)nvar * Nf), WRf_s((size_t)nvar * Nf);
    std::vector<D3ThincCell> tc;

    for (int v = 0; v < nvar; ++v) {
        d3_build_thinc_cells(m, o2c, W, nvar, v, beta_l, beta_s, tc);
        d3_thinc_face_arrays(m, tc, W, nvar, v, WLf_l, WRf_l, WLf_s, WRf_s);

        // min-TBV per cell: for each candidate, the cell's total boundary
        // variation = Σ_{faces of ci} | qf_cand(this cell's side) − qbar(neighbour) |
        // (neighbour = the cell across the face, or ci itself on a boundary face).
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            if (!tc[ci].hasint) continue;   // non-interface cell → MUSCL (already the default)
            double qbar_ci = W[(size_t)v * N + ci];
            double tbv_m = 0.0, tbv_l = 0.0, tbv_s = 0.0;
            for (int f : m.cell_faces[ci]) {
                bool is_owner = (m.face_owner[f] == ci);
                int nb = is_owner ? m.face_neighbour[f] : m.face_owner[f];
                double qnb = (nb >= 0) ? W[(size_t)v * N + nb] : qbar_ci;
                // this cell's side of the face for each candidate.
                double qm  = is_owner ? WLm[(size_t)v * Nf + f]  : WRm[(size_t)v * Nf + f];
                double ql  = is_owner ? WLf_l[(size_t)v * Nf + f] : WRf_l[(size_t)v * Nf + f];
                double qs  = is_owner ? WLf_s[(size_t)v * Nf + f] : WRf_s[(size_t)v * Nf + f];
                tbv_m += std::fabs(qm - qnb);
                tbv_l += std::fabs(ql - qnb);
                tbv_s += std::fabs(qs - qnb);
            }
            // pick the smallest-TBV candidate. If this cell is not a THINC
            // interface cell, the THINC arrays equal the 1st-order value (qbar),
            // which keeps the contact sharp only where THINC fired; elsewhere the
            // MUSCL candidate is naturally selected (its TBV is the limited one).
            // strong-shock cells: veto THINC (force MUSCL) for positivity.
            bool allow_thinc = shockcell.empty() || !shockcell[ci];
            int pick = 0; double best = tbv_m;
            if (allow_thinc && tc[ci].hasint && tbv_l < best) { best = tbv_l; pick = 1; }
            if (allow_thinc && tc[ci].hasint && tbv_s < best) { best = tbv_s; pick = 2; }

            // write this cell's side of every incident face into the FINAL WL/WR.
            for (int f : m.cell_faces[ci]) {
                bool is_owner = (m.face_owner[f] == ci);
                double val;
                if (pick == 0) val = is_owner ? WLm[(size_t)v * Nf + f]  : WRm[(size_t)v * Nf + f];
                else if (pick == 1) val = is_owner ? WLf_l[(size_t)v * Nf + f] : WRf_l[(size_t)v * Nf + f];
                else val = is_owner ? WLf_s[(size_t)v * Nf + f] : WRf_s[(size_t)v * Nf + f];
                if (is_owner) WL[(size_t)v * Nf + f] = val;
                else          WR[(size_t)v * Nf + f] = val;
            }
        }
    }
}

} // namespace cfd
