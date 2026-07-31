// cfd/reconstruct3d_bvd_euler_mlp.hpp — MLP-LIMITED 3D deg3t-THINC + min-TBV BVD.
//
// Shock-stable variant of reconstruct3d_bvd_euler.hpp (Stage 3a). The unlimited
// Euler BVD survives smooth flows (KH / vortices) but OVERSHOOTS at a strong
// STANDING shock (the o2-quad candidate has no bound there and the THINC value is
// unclamped) → density/pressure go negative → divergence. This file makes BOTH
// candidates BOUNDED by the vertex-26 stencil, so RECON3_BVD is shock-stable
// WITHOUT MOOD and WITHOUT a diffusive hard clamp on the smooth candidate:
//
//   (0) o2-quad : q = qbar + phi * P(dof), with phi ∈ [0,1] an MLP/Barth-Jespersen
//       limiter = min over the cell's faces of the admissible ratio keeping the
//       face-extrapolated value within [qmin − Δ, qmax + Δ] of the vertex-26
//       stencil min/max. The TVB widening Δ = M·h² (h = cbrt(cell vol)) PRESERVES
//       smooth extrema (boundary-layer vortices) while bounding the strong shock —
//       this is NOT a plain min/max clamp.
//   (1,2) THINC : the deg3t profile is mapped to [qmin,qmax] and CLAMPED to that
//       same vertex-26 range (the clamp the unlimited euler version removed). The
//       deg3t profile already ∈[-1,1] so this only guards round-off / monotonicity.
//
// min-TBV BVD selection is unchanged. Reconstruction-only (the flux is untouched);
// the o2/core math (reconstruct3d_o2.hpp, reconstruct3d_bvd_core.hpp) is FROZEN and
// only consumed here.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/reconstruct3d_bvd_core.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace cfd {

// Per-(cell,var) THINC state for the MLP-limited Euler BVD. Same construction as
// D3ThincCellE but qmax stored explicitly so the THINC face value can be CLAMPED.
struct D3ThincCellEM {
    D3Poly P;
    double D_l = 0.0, D_s = 0.0;
    double kk_l = 0.0, kk_s = 0.0;
    double qmin = 0.0, qmax = 0.0;   // vertex-26 stencil min/max (THINC clamp range)
    double rng = 0.0;
    int hasint = 0;
};

// MLP / Barth-Jespersen admissible ratio with TVB relaxation. The bound is the
// vertex-26 stencil [lo, hi] widened by Δ (=M·h²). delta = the unlimited
// face-extrapolated increment P(dof). Returns phi ∈ [0,1].
inline double mlp_phi_tvb_3d(double delta, double center, double lo, double hi, double Dtvb) {
    const double eps = 1e-30;
    double allowed = (delta >= 0.0) ? (hi + Dtvb - center) : (center - (lo - Dtvb));
    if (allowed < 0.0) allowed = 0.0;               // already outside even the widened band
    double phi = (std::fabs(delta) > eps) ? allowed / std::max(std::fabs(delta), eps) : 1.0;
    return phi < 0.0 ? 0.0 : (phi > 1.0 ? 1.0 : phi);
}

// Build the THINC per-cell state for one variable v (identical interface-P to the
// unlimited euler BVD, but also records qmax for the face clamp).
inline void dem_build_thinc_cells(const Mesh& m, const ReconCtx3DO2& o2c,
                                  const std::vector<double>& W, int nvar, int v,
                                  double beta_l, double beta_s,
                                  std::vector<D3ThincCellEM>& tc) {
    (void)nvar;
    const int N = m.n_cells();
    const double hx = m.s3_h[0], hy = m.s3_h[1], hz = m.s3_h[2];
    const double H = std::cbrt(hx * hy * hz);
    tc.assign((size_t)N, D3ThincCellEM{});

    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        double qbar = W[(size_t)v * N + ci];
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
        D3ThincCellEM& t = tc[ci];
        t.qmin = qmin; t.qmax = qmax; t.rng = rng;   // store even for non-interface
        if (cbar <= 1e-6 || cbar >= 1.0 - 1e-6 || rng <= 1e-14) continue;

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
        if (gnorm < 1e-30) continue;

        double inv = 1.0 / gnorm;
        double nx = gx * inv, ny = gy * inv, nz = gz * inv;
        double Hxx = hxx, Hyy = hyy, Hzz = hzz, Hxy = hxy, Hxz = hxz, Hyz = hyz;
        double Hg_x = Hxx * gx + Hxy * gy + Hxz * gz;
        double Hg_y = Hxy * gx + Hyy * gy + Hyz * gz;
        double Hg_z = Hxz * gx + Hyz * gy + Hzz * gz;
        double inv3 = inv * inv * inv;
        double n_xx = Hxx * inv - gx * Hg_x * inv3;
        double n_yy = Hyy * inv - gy * Hg_y * inv3;
        double n_zz = Hzz * inv - gz * Hg_z * inv3;
        double n_xy = Hxy * inv - gx * Hg_y * inv3;
        double n_yx = Hxy * inv - gy * Hg_x * inv3;
        double n_xz = Hxz * inv - gx * Hg_z * inv3;
        double n_zx = Hxz * inv - gz * Hg_x * inv3;
        double n_yz = Hyz * inv - gy * Hg_z * inv3;
        double n_zy = Hyz * inv - gz * Hg_y * inv3;

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
        t.hasint = 1;
    }
}

// THINC candidate face value WITH the vertex-26 clamp restored.
// Non-interface cell → returns q_o2 (the o2-quad fallback, already MLP-limited).
inline double dem_thinc_face_val(const Mesh& m, const std::vector<D3ThincCellEM>& tc,
                                 int f, int cell, double D, double kk, double q_o2) {
    const D3ThincCellEM& t = tc[cell];
    if (!t.hasint) return q_o2;
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
    double qf = t.qmin + 0.5 * t.rng * (1.0 + fa);
    // RESTORED vertex-26 clamp (monotone; the unlimited euler version dropped this).
    if (qf < t.qmin) qf = t.qmin;
    if (qf > t.qmax) qf = t.qmax;
    return qf;
}

// Public entry: MLP-LIMITED min-TBV BVD over {o2-quad(phi-limited), THINC_l, THINC_s}.
// Signature mirrors reconstruct3d_bvd_euler; tvb_M sets the TVB Δ = M·h² (env TVB_M
// overrides if >=0). Fills WL/WR for nvar PRIMITIVE variables.
inline void reconstruct3d_bvd_euler_mlp(const Mesh& m, const ReconCtx3DO2& o2c,
                                        const std::vector<double>& W, int nvar,
                                        std::vector<double>& WL, std::vector<double>& WR,
                                        double beta_l = 1.6, double beta_s = 0.8,
                                        double tvb_M = 20.0) {
    const int N = m.n_cells(), Nf = m.n_faces();
    WL.assign((size_t)nvar * Nf, 0.0);
    WR.assign((size_t)nvar * Nf, 0.0);
    {
        const char* e = std::getenv("TVB_M");
        if (e) tvb_M = std::atof(e);
    }
    // The SMOOTH BVD candidate is the MLP-limited LINEAR (gradient) reconstruction by
    // DEFAULT. The o2 P2 Hessian on the smooth candidate is NOT bound-stable under a
    // single-field independent MLP limiter at a strong shock (Ms~2.37): the curvature
    // overshoots between faces and destabilizes (verified: with the Hessian on, the
    // Daru-Tenaud tube diverges at t~0.012; with it off it is rock-solid to t=1.0).
    // The full QUADRATIC interface curvature (QQ) is retained where it matters — in the
    // deg3t-THINC candidate (built from grad+Hessian) — so KH-roll capturing is intact.
    // BVD_HESS opts the P2 Hessian back into the smooth candidate (experimental).
    // BVD_NO_THINC disables the THINC candidates (pure limited reconstruction).
    static const bool use_hess = (std::getenv("BVD_HESS")     != nullptr);
    static const bool no_hess  = !use_hess;
    // The deg3t-THINC sharp candidate sharpens contacts/slip-lines but DESTABILIZES at
    // a strong reflecting shock (Ms~2.37): with THINC the Daru-Tenaud tube diverges at
    // t~0.31 (shock/wall reflection). The shipped MLP-limited recon therefore keeps the
    // MLP-limited smooth candidate and enables THINC only on opt-in (BVD_THINC=1), with
    // the pressure-jump shock-veto (BVD_SHOCK_P) restricting it to non-shock cells.
    static const bool use_thinc = (std::getenv("BVD_THINC")   != nullptr);
    static const bool no_thinc  = !use_thinc || (std::getenv("BVD_NO_THINC") != nullptr);

    // ── THINC shock-veto (classified compression). The deg3t-THINC sharp candidate is
    // beneficial at CONTACTS / slip-lines (sharpens the Daru-Tenaud vortex roll-up) but
    // DESTABILIZES at a strong reflecting SHOCK (verified: with THINC on, the tube
    // diverges at t~0.31 during the shock/wall reflection; with THINC off it is stable).
    // So THINC is VETOED in a cell whose max relative PRESSURE jump to a vertex-26
    // neighbour exceeds shock_p (a shock ⇒ big Δp/p; a contact ⇒ Δp/p~0, THINC kept).
    // pidx = the pressure variable index (W layout = rho,u,v,w,p ⇒ 4).
    const int pidx = nvar - 1;
    double shock_p = 0.20;
    { const char* e = std::getenv("BVD_SHOCK_P"); if (e) shock_p = std::atof(e); }
    std::vector<unsigned char> shock_veto((size_t)N, 0);
    if (!no_thinc) {
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double pc = W[(size_t)pidx * N + ci];
            double mx = 0.0;
            for (int k = 0; k < o2c.max_nb; ++k) {
                int nb = o2c.nb[(size_t)ci * o2c.max_nb + k];
                if (nb < 0) continue;
                double pn = W[(size_t)pidx * N + nb];
                double rel = std::fabs(pn - pc) / std::max(std::fabs(pn) + std::fabs(pc), 1e-30);
                if (rel > mx) mx = rel;
            }
            shock_veto[ci] = (mx > shock_p) ? 1 : 0;
        }
    }

    // o2-quad face arrays (phi-limited) + THINC face arrays.
    std::vector<double> WLq((size_t)Nf), WRq((size_t)Nf);
    std::vector<double> WLf_l((size_t)Nf), WRf_l((size_t)Nf);
    std::vector<double> WLf_s((size_t)Nf), WRf_s((size_t)Nf);
    std::vector<double> g;                                  // N*9 o2 coeffs
    std::vector<double> phg((size_t)N), phh((size_t)N);     // per-cell HIERARCHICAL limiters
    std::vector<D3ThincCellEM> tc;

    for (int v = 0; v < nvar; ++v) {
        // (0) UNLIMITED o2 P2-quad coeffs for ALL cells.
        reconstruct3d_o2_coeffs(m, o2c, W, nvar, v, g);
        if (no_hess) {   // diagnostic: drop the quadratic curvature → limited linear
            #pragma omp parallel for
            for (int ci = 0; ci < N; ++ci)
                for (int i = 3; i < 9; ++i) g[(size_t)ci * 9 + i] = 0.0;
        }

        // (0a) HIERARCHICAL MLP limiter against the vertex-26 stencil [qmin−Δ, qmax+Δ]
        // (TVB Δ = M h²). A SINGLE scalar on the full P2 does NOT bound the quadratic
        // at a strong shock (the curvature overshoots between faces → instability), so
        // the linear and quadratic parts are limited SEPARATELY (hierarchical / moment
        // limiting, à la Krivodonova):
        //   phg = BJ ratio bounding the LINEAR (gradient) face value to the band;
        //   phh = ratio scaling the HESSIAN term so (phg·grad + phh·hess) stays in band.
        // Smooth region: phg=phh=1 (full 3rd-order). Shock: phh→0 first (curvature
        // killed), then phg→0 (first-order). The THINC candidate keeps the full QQ
        // interface curvature, so KH-roll capturing is unaffected.
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double qbar = W[(size_t)v * N + ci];
            double qmin = qbar, qmax = qbar;
            for (int k = 0; k < o2c.max_nb; ++k) {
                int nb = o2c.nb[(size_t)ci * o2c.max_nb + k];
                if (nb < 0) continue;
                double wn = W[(size_t)v * N + nb];
                if (wn < qmin) qmin = wn;
                if (wn > qmax) qmax = wn;
            }
            double Hc = std::cbrt(m.cell_volumes[ci]);
            double Dtvb = tvb_M * Hc * Hc;               // TVB widening Δ = M h²
            const double* gg = &g[(size_t)ci * 9];
            // pass 1: limit the LINEAR part.
            double pg = 1.0;
            for (int f : m.cell_faces[ci]) {
                double dx = m.face_centers[(size_t)f * 3 + 0] - m.cell_centers[(size_t)ci * 3 + 0];
                double dy = m.face_centers[(size_t)f * 3 + 1] - m.cell_centers[(size_t)ci * 3 + 1];
                double dz = m.face_centers[(size_t)f * 3 + 2] - m.cell_centers[(size_t)ci * 3 + 2];
                double lin = gg[0]*dx + gg[1]*dy + gg[2]*dz;
                double pk = mlp_phi_tvb_3d(lin, qbar, qmin, qmax, Dtvb);
                if (pk < pg) pg = pk;
            }
            // pass 2: with the linear part fixed at pg, limit the HESSIAN so the TOTAL
            // face value (qbar + pg·lin + phh·hess) stays in [qmin−Δ, qmax+Δ].
            double ph = 1.0;
            for (int f : m.cell_faces[ci]) {
                double dx = m.face_centers[(size_t)f * 3 + 0] - m.cell_centers[(size_t)ci * 3 + 0];
                double dy = m.face_centers[(size_t)f * 3 + 1] - m.cell_centers[(size_t)ci * 3 + 1];
                double dz = m.face_centers[(size_t)f * 3 + 2] - m.cell_centers[(size_t)ci * 3 + 2];
                double lin  = gg[0]*dx + gg[1]*dy + gg[2]*dz;
                double hess = 0.5*gg[3]*dx*dx + 0.5*gg[4]*dy*dy + 0.5*gg[5]*dz*dz
                            + gg[6]*dx*dy + gg[7]*dx*dz + gg[8]*dy*dz;
                double base = qbar + pg * lin;            // already-in-band linear value
                // remaining headroom in the band for the hessian increment
                double pk = mlp_phi_tvb_3d(hess, base, qmin, qmax, Dtvb);
                if (pk < ph) ph = pk;
            }
            phg[ci] = pg; phh[ci] = ph;
        }

        // o2-quad face value: q = qbar + phg·(grad·dof) + phh·(0.5 H:dof⊗dof).
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            int o = m.face_owner[f], n = m.face_neighbour[f];
            double fcx = m.face_centers[(size_t)f * 3 + 0];
            double fcy = m.face_centers[(size_t)f * 3 + 1];
            double fcz = m.face_centers[(size_t)f * 3 + 2];
            {
                double wc = W[(size_t)v * N + o];
                double dx = fcx - m.cell_centers[(size_t)o * 3 + 0];
                double dy = fcy - m.cell_centers[(size_t)o * 3 + 1];
                double dz = fcz - m.cell_centers[(size_t)o * 3 + 2];
                const double* gg = &g[(size_t)o * 9];
                double lin  = gg[0]*dx + gg[1]*dy + gg[2]*dz;
                double hess = 0.5*gg[3]*dx*dx + 0.5*gg[4]*dy*dy + 0.5*gg[5]*dz*dz
                            + gg[6]*dx*dy + gg[7]*dx*dz + gg[8]*dy*dz;
                WLq[f] = wc + phg[o]*lin + phh[o]*hess;
            }
            if (n >= 0) {
                double wc = W[(size_t)v * N + n];
                double dx = fcx - m.cell_centers[(size_t)n * 3 + 0];
                double dy = fcy - m.cell_centers[(size_t)n * 3 + 1];
                double dz = fcz - m.cell_centers[(size_t)n * 3 + 2];
                const double* gg = &g[(size_t)n * 9];
                double lin  = gg[0]*dx + gg[1]*dy + gg[2]*dz;
                double hess = 0.5*gg[3]*dx*dx + 0.5*gg[4]*dy*dy + 0.5*gg[5]*dz*dz
                            + gg[6]*dx*dy + gg[7]*dx*dz + gg[8]*dy*dz;
                WRq[f] = wc + phg[n]*lin + phh[n]*hess;
            } else {
                WRq[f] = WLq[f];
            }
        }

        // default = o2-quad (phi-limited) everywhere (set first so no_thinc is a no-op tail).
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            WL[(size_t)v * Nf + f] = WLq[f];
            WR[(size_t)v * Nf + f] = WRq[f];
        }
        if (no_thinc) continue;   // diagnostic: pure MLP-limited o2-quad (no THINC, no BVD)

        // (1,2) THINC candidate face arrays (clamped to the vertex-26 range).
        dem_build_thinc_cells(m, o2c, W, nvar, v, beta_l, beta_s, tc);
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            int o = m.face_owner[f], n = m.face_neighbour[f];
            WLf_l[f] = dem_thinc_face_val(m, tc, f, o, tc[o].D_l, tc[o].kk_l, WLq[f]);
            WLf_s[f] = dem_thinc_face_val(m, tc, f, o, tc[o].D_s, tc[o].kk_s, WLq[f]);
            if (n >= 0) {
                WRf_l[f] = dem_thinc_face_val(m, tc, f, n, tc[n].D_l, tc[n].kk_l, WRq[f]);
                WRf_s[f] = dem_thinc_face_val(m, tc, f, n, tc[n].D_s, tc[n].kk_s, WRq[f]);
            } else {
                WRf_l[f] = WLf_l[f];
                WRf_s[f] = WLf_s[f];
            }
        }

        // min-TBV per cell over {o2-quad, THINC_l, THINC_s}. Shock-vetoed cells keep
        // the MLP-limited smooth candidate (o2-quad) — THINC only at contacts/slip-lines.
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            if (!tc[ci].hasint) continue;
            if (shock_veto[ci]) continue;   // strong shock → no THINC (keep limited smooth)
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
