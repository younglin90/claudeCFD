// cfd/five_eq/acoustic_solve.hpp — implicit acoustic (u,p) solve (M7).
//
// C++ port of solver/five_eq_IMEX/imex_ad.py::_solve_acoustic_ad (the n>=32
// production path with FIVE_EQ_IMEX_ACOUSTIC_RECON=weno5).  Validated
// bit-comparably (rel <= 1e-12) against tests/5eq_ref/acoustic_solve_ref.txt.
//
// The Python solve is ONE Newton step of a limiter-frozen linear acoustic
// operator anchored at W^n:  y = y0 - J^{-1} R(y0),  y0 = (u0, p0).  The
// operator's coefficients (Z = rho c, beta = rho c^2, rho) are frozen at the
// stage anchor.  Python builds J with torch autodiff under a *straight-through*
// rule: the WENO5 face VALUE is the full nonlinear Jiang-Shu reconstruction, but
// its Jacobian is frozen to the linear-OPTIMAL 5th-order stencil d=(1/10,6/10,
// 3/10) (imex_ad.py weno5_left_t: `lin + (weno - lin).detach()`).  Everything
// else in the residual is already linear in (u,p) with frozen coefficients OR a
// TVD limiter differentiated at the anchor.
//
// This port reproduces that with a fixed-width forward-mode dual number
// (`Dual`): one evaluation of the templated local stencil residual yields, in a
// single pass, BOTH the nonlinear residual value R(y0) (the `.v` field carries
// the nonlinear WENO5 value through) AND the straight-through analytic Jacobian
// row (the `.g` field carries grad of the linear-optimal stencil + the anchor
// tangent of the superbee limiter).  This is operation-for-operation identical
// to torch's vmap(jacrev(local_residual_torch)).
//
// Sparsity: per cell i the residual couples cells {i-3..i+3} in both the u and p
// blocks (i+-1 from the Z-Riemann core, i+-2/i+-3 from the WENO5 linear stencil),
// so each of the four n x n sub-blocks is 7-point (half-bandwidth 3); in the
// interleaved [u_i,p_i] ordering the 2n x 2n operator has half-bandwidth 7.  The
// solve below is a dense partial-pivot Gaussian elimination (self-contained, no
// LAPACK): bulletproof and exact to roundoff for the validation sizes, and it
// handles the periodic cyclic wrap entries transparently (they are just off-band
// entries of the dense matrix).  For the production speed goal (n up to ~800)
// swap in a banded LU (dgbtrf/dgbtrs-style, kl=ku=7 on the interleaved ordering,
// O(n*b^2)) for the non-periodic case and a bordered/Sherman-Morrison solve for
// the periodic wrap rows; the assembly below is unchanged.
//
// Config ported (production BASE_ENV): acoustic_recon=weno5, scheme=theta,
// theta_wave=0.5, wave_rel=1e-8, tvd=superbee, centered_interface=on, mixture=
// kapila, pure_branch on.  Default-off research flags (ACID, PURE_TOL_CONSISTENT,
// STENCIL_CLEAN, DISS_CONSISTENT, WAF, interface_be, trbdf2, muscl3/weno3/
// characteristic/bvd recon) are NOT taken on this path and are omitted.
#pragma once
#include "cfd/eos.hpp"
#include "cfd/five_eq/sound_speed.hpp"
#include "cfd/five_eq/weno.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace cfd {

enum class AcousticBC { reflective, transmissive, periodic, inlet, outlet };

namespace acoustic_detail {

constexpr double AC_EPS = 1e-30;

// ── fixed-width forward-mode dual number (value + 14 stencil partials) ────────
struct Dual {
    double v = 0.0;
    double g[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    Dual() = default;
    explicit Dual(double c) : v(c) {}
};
inline Dual operator+(const Dual& a, const Dual& b) {
    Dual r; r.v = a.v + b.v;
    for (int k = 0; k < 14; ++k) r.g[k] = a.g[k] + b.g[k];
    return r;
}
inline Dual operator-(const Dual& a, const Dual& b) {
    Dual r; r.v = a.v - b.v;
    for (int k = 0; k < 14; ++k) r.g[k] = a.g[k] - b.g[k];
    return r;
}
inline Dual operator-(const Dual& a) {
    Dual r; r.v = -a.v;
    for (int k = 0; k < 14; ++k) r.g[k] = -a.g[k];
    return r;
}
inline Dual operator*(const Dual& a, const Dual& b) {
    Dual r; r.v = a.v * b.v;
    for (int k = 0; k < 14; ++k) r.g[k] = a.g[k] * b.v + a.v * b.g[k];
    return r;
}
inline Dual operator*(const Dual& a, double c) {
    Dual r; r.v = a.v * c;
    for (int k = 0; k < 14; ++k) r.g[k] = a.g[k] * c;
    return r;
}
inline Dual operator*(double c, const Dual& a) { return a * c; }
inline Dual operator/(const Dual& a, double c) {
    Dual r; r.v = a.v / c;
    for (int k = 0; k < 14; ++k) r.g[k] = a.g[k] / c;
    return r;
}
// abs with torch semantics: d|x|/dx = sign(x), 0 at x==0.
inline Dual abs_d(const Dual& a) {
    double s = (a.v > 0.0) ? 1.0 : (a.v < 0.0 ? -1.0 : 0.0);
    Dual r; r.v = std::fabs(a.v);
    for (int k = 0; k < 14; ++k) r.g[k] = s * a.g[k];
    return r;
}
// minimum/maximum: gradient follows the selected branch (ties -> first arg).
inline Dual min_d(const Dual& a, const Dual& b) { return (a.v <= b.v) ? a : b; }
inline Dual max_d(const Dual& a, const Dual& b) { return (a.v >= b.v) ? a : b; }

// superbee limited slope (imex_ad.py limited_t, tvd_kind='superbee'):
//   same = (a*b)>0 ; val = sign(a)*max(min(2|a|,|b|), min(|a|,2|b|)); else 0.
inline Dual superbee_d(const Dual& a, const Dual& b) {
    bool same = (a.v * b.v) > 0.0;
    if (!same) return Dual(0.0);
    Dual aa = abs_d(a), bb = abs_d(b);
    Dual cand1 = min_d(aa * 2.0, bb);
    Dual cand2 = min_d(aa, bb * 2.0);
    Dual mx = max_d(cand1, cand2);
    double s = (a.v > 0.0) ? 1.0 : (a.v < 0.0 ? -1.0 : 0.0);
    return mx * s;
}

// WENO5-JS straight-through (imex_ad.py weno5_left_t): VALUE = nonlinear WENO5,
// GRADIENT = linear-optimal 5th-order stencil lin = 0.1*c0+0.6*c1+0.3*c2.
inline Dual weno5_st(const Dual& qmm, const Dual& qm, const Dual& q0,
                     const Dual& qp, const Dual& qpp) {
    Dual c0 = qmm * (1.0 / 3.0) - qm * (7.0 / 6.0) + q0 * (11.0 / 6.0);
    Dual c1 = qm * (-1.0 / 6.0) + q0 * (5.0 / 6.0) + qp * (1.0 / 3.0);
    Dual c2 = q0 * (1.0 / 3.0) + qp * (5.0 / 6.0) - qpp * (1.0 / 6.0);
    Dual lin = c0 * 0.1 + c1 * 0.6 + c2 * 0.3;
    double w = weno5_face_left(qmm.v, qm.v, q0.v, qp.v, qpp.v);
    Dual out; out.v = w;
    for (int k = 0; k < 14; ++k) out.g[k] = lin.g[k];  // = lin + (weno-lin).detach()
    return out;
}

// ── plain-double helpers for the old-time (explicit) face divergence ─────────
inline double superbee_pair(double a, double b) {
    bool same = (a * b) > 0.0;
    if (!same) return 0.0;
    double s = (a > 0.0) ? 1.0 : (a < 0.0 ? -1.0 : 0.0);
    double cand1 = std::fmin(2.0 * std::fabs(a), std::fabs(b));
    double cand2 = std::fmin(std::fabs(a), 2.0 * std::fabs(b));
    return s * std::fmax(cand1, cand2);
}

inline double eps025() { return std::pow(std::numeric_limits<double>::epsilon(), 0.25); }

// same resolved-pure material pair (_same_pure_pair_np).
inline bool same_pure(double a, double b, double alpha_pure_tol) {
    double pt = std::fmax(alpha_pure_tol, eps025());
    return ((a >= 1.0 - pt) && (b >= 1.0 - pt)) || ((a <= pt) && (b <= pt));
}

// ── dense partial-pivot linear solve (A x = b), A is (m x m) row-major ────────
inline void solve_dense_pp(std::vector<double>& A, std::vector<double>& b, int m) {
    for (int col = 0; col < m; ++col) {
        int piv = col;
        double best = std::fabs(A[col * m + col]);
        for (int r = col + 1; r < m; ++r) {
            double v = std::fabs(A[r * m + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) {
            for (int c = 0; c < m; ++c) std::swap(A[col * m + c], A[piv * m + c]);
            std::swap(b[col], b[piv]);
        }
        double diag = A[col * m + col];
        for (int r = col + 1; r < m; ++r) {
            double f = A[r * m + col] / diag;
            if (f == 0.0) continue;
            A[r * m + col] = 0.0;
            for (int c = col + 1; c < m; ++c) A[r * m + c] -= f * A[col * m + c];
            b[r] -= f * b[col];
        }
    }
    for (int r = m - 1; r >= 0; --r) {
        double s = b[r];
        for (int c = r + 1; c < m; ++c) s -= A[r * m + c] * b[c];  // b[c] = x_c
        b[r] = s / A[r * m + r];
    }
}

} // namespace acoustic_detail

// Result of the implicit acoustic solve.
struct AcousticSolveResult {
    std::vector<double> u_new;
    std::vector<double> p_new;
    // Diagnostics: the regularized 2n x 2n operator A (row-major) and RHS b=-R,
    // and the max-abs residual ||A*dy - b|| of the returned solution (proves the
    // assembly is a valid linear system solved to machine precision).
    std::vector<double> Amat;    // (2n)^2, regularized
    std::vector<double> Rhs;     // 2n, = -R
    double resid_self = 0.0;     // max_k |(A dy - b)_k| for the returned dy
};

// Implicit acoustic (u,p) solve.  Inputs are the stage-anchor primitive state
// W_n=(alpha,T1,T2,u0,p0) and the material-update conservative outputs
// (q1_new,q2_new,m_adv).  Production config: weno5 recon, theta scheme,
// theta_wave=0.5, tvd=superbee, kapila mixture, pure_branch (alpha_pure_tol).
inline AcousticSolveResult acoustic_solve(
        int n, double dx, double dt,
        const EOS& eos1, const EOS& eos2,
        const double* alpha, const double* T1, const double* T2,
        const double* u0, const double* p0,
        const double* q1_new, const double* q2_new, const double* m_adv,
        AcousticBC bc_l, AcousticBC bc_r, double alpha_pure_tol,
        double theta_wave = 0.5, double wave_rel = 1e-8) {
    using namespace acoustic_detail;
    const double EPS = AC_EPS;
    const bool periodic = (bc_l == AcousticBC::periodic && bc_r == AcousticBC::periodic);

    // ── frozen coefficients at W^n ───────────────────────────────────────────
    std::vector<double> Z(n), c_mix_sq(n), rho_star(n), beta(n);
    std::vector<char> u_mask(n);
    for (int i = 0; i < n; ++i) {
        PhaseAcoustic pa = phase_acoustic(eos1, eos2, alpha[i], T1[i], T2[i],
                                          p0[i], alpha_pure_tol);
        Z[i] = pa.Z;
        c_mix_sq[i] = pa.c_mix_sq;
        rho_star[i] = EOS::max2(q1_new[i] + q2_new[i], EPS);
        beta[i] = EOS::max2(pa.rho * pa.c_mix_sq, EPS);
        u_mask[i] = (u0[i] >= 0.0) ? 1 : 0;
    }

    // ── old-time faces (explicit divergence) via the weno5 muscl helper ──────
    // p_star,u_star first-order Z-Riemann on the ghost-extended state, then a
    // componentwise weno5/superbee high-order override on pure-bulk faces.
    const int nf = n + 1;
    std::vector<double> pf_old(nf), uf_old(nf);
    std::vector<char> high_face(nf, 0);
    {
        // ghost-extend (length n+2): reflective flips u (odd), transmissive/
        // outlet copies, periodic wraps.  Z copies the end cell (never flips).
        auto ext = [&](const double* a, bool odd) {
            std::vector<double> e(n + 2);
            for (int i = 0; i < n; ++i) e[i + 1] = a[i];
            if (periodic) { e[0] = a[n - 1]; e[n + 1] = a[0]; return e; }
            e[0] = (odd && bc_l == AcousticBC::reflective) ? -a[0] : a[0];
            e[n + 1] = (odd && bc_r == AcousticBC::reflective) ? -a[n - 1] : a[n - 1];
            return e;
        };
        std::vector<double> u_ext = ext(u0, true);
        std::vector<double> p_ext = ext(p0, false);
        std::vector<double> a_ext = ext(alpha, false);
        std::vector<double> Z_ext(n + 2);
        for (int i = 0; i < n; ++i) Z_ext[i + 1] = Z[i];
        if (periodic) { Z_ext[0] = Z[n - 1]; Z_ext[n + 1] = Z[0]; }
        else { Z_ext[0] = Z[0]; Z_ext[n + 1] = Z[n - 1]; }

        // first-order Z-Riemann on every face (_acoustic_faces_np).
        for (int f = 0; f < nf; ++f) {
            double ZL = Z_ext[f], ZR = Z_ext[f + 1];
            double den = EOS::max2(ZL + ZR, EPS);
            double pL = p_ext[f], pR = p_ext[f + 1];
            double uL = u_ext[f], uR = u_ext[f + 1];
            pf_old[f] = (ZR * pL + ZL * pR + ZL * ZR * (uL - uR)) / den;
            uf_old[f] = (pL - pR + ZL * uL + ZR * uR) / den;
        }
        if (bc_l == AcousticBC::reflective) { pf_old[0] = p0[0]; uf_old[0] = 0.0; }
        if (bc_r == AcousticBC::reflective) { pf_old[n] = p0[n - 1]; uf_old[n] = 0.0; }

        // pure-bulk high-order face mask (_pure_bulk_muscl_face_mask).
        double pt = std::fmax(alpha_pure_tol, eps025());
        if (alpha_pure_tol > 0.0) {
            for (int f = 0; f < nf; ++f) {
                double aL = a_ext[f], aR = a_ext[f + 1];
                bool pL = (aL >= 1.0 - pt) || (aL <= pt);
                bool pR = (aR >= 1.0 - pt) || (aR <= pt);
                high_face[f] = (pL && pR) ? 1 : 0;
            }
            high_face[0] = 0; high_face[n] = 0;
            if (nf > 3 && !periodic) { high_face[1] = 0; high_face[n - 1] = 0; }
        }
        // same-pure material face mask (_same_pure_material_face_mask).
        std::vector<char> same_face(nf, 0);
        if (alpha_pure_tol > 0.0) {
            for (int f = 0; f < nf; ++f) {
                double aL = a_ext[f], aR = a_ext[f + 1];
                bool p1 = (aL >= 1.0 - pt) && (aR >= 1.0 - pt);
                bool p2 = (aL <= pt) && (aR <= pt);
                same_face[f] = (p1 || p2) ? 1 : 0;
            }
            same_face[0] = 0; same_face[n] = 0;
            if (nf > 3 && !periodic) { same_face[1] = 0; same_face[n - 1] = 0; }
        }
        // weno5 face availability (face-based, ext window f-2..f+3).
        int L = n + 2;
        std::vector<char> weno5_face_ok(nf, 0);
        {
            int fmax = std::min(nf - 1, L - 4);
            for (int f = 2; f <= fmax; ++f) {
                bool clean = true;
                for (int d = -2; d <= 2; ++d)
                    clean = clean && same_pure(a_ext[f + d], a_ext[f + d + 1], alpha_pure_tol);
                weno5_face_ok[f] = clean ? 1 : 0;
            }
        }
        // high-order override loop (centered_interface on, tvd=superbee, weno5).
        for (int f = 0; f < nf; ++f) {
            if (!high_face[f]) continue;
            if (f <= 0 || f + 2 >= (int)p_ext.size()) continue;
            double pL, pR, uL, uR;
            if (same_face[f] && weno5_face_ok[f]) {
                pL = weno5_face_left(p_ext[f - 2], p_ext[f - 1], p_ext[f], p_ext[f + 1], p_ext[f + 2]);
                pR = weno5_face_left(p_ext[f + 3], p_ext[f + 2], p_ext[f + 1], p_ext[f], p_ext[f - 1]);
                uL = weno5_face_left(u_ext[f - 2], u_ext[f - 1], u_ext[f], u_ext[f + 1], u_ext[f + 2]);
                uR = weno5_face_left(u_ext[f + 3], u_ext[f + 2], u_ext[f + 1], u_ext[f], u_ext[f - 1]);
            } else {
                // same_face[f] || centered_interface(on): componentwise superbee.
                double spL = superbee_pair(p_ext[f] - p_ext[f - 1], p_ext[f + 1] - p_ext[f]);
                double spR = superbee_pair(p_ext[f + 1] - p_ext[f], p_ext[f + 2] - p_ext[f + 1]);
                double suL = superbee_pair(u_ext[f] - u_ext[f - 1], u_ext[f + 1] - u_ext[f]);
                double suR = superbee_pair(u_ext[f + 1] - u_ext[f], u_ext[f + 2] - u_ext[f + 1]);
                pL = p_ext[f] + 0.5 * spL; pR = p_ext[f + 1] - 0.5 * spR;
                uL = u_ext[f] + 0.5 * suL; uR = u_ext[f + 1] - 0.5 * suR;
            }
            double ZL = Z_ext[f], ZR = Z_ext[f + 1];
            double den = EOS::max2(ZL + ZR, EPS);
            pf_old[f] = (ZR * pL + ZL * pR + ZL * ZR * (uL - uR)) / den;
            uf_old[f] = (pL - pR + ZL * uL + ZR * uR) / den;
        }
    }
    std::vector<double> div_p_old(n), div_u_old(n);
    for (int i = 0; i < n; ++i) {
        div_p_old[i] = (pf_old[i + 1] - pf_old[i]) / dx;
        div_u_old[i] = (uf_old[i + 1] - uf_old[i]) / dx;
    }

    // ── per-cell theta (CN 0.5 on wave cells, BE 1.0 on flat cells) ──────────
    std::vector<double> theta(n);
    {
        auto pext = [&](int f) -> double {   // p_ext_old, extend odd=False
            if (f == 0) return periodic ? p0[n - 1] : p0[0];
            if (f == n + 1) return periodic ? p0[0] : p0[n - 1];
            return p0[f - 1];
        };
        std::vector<char> wave_face(nf);
        for (int f = 0; f < nf; ++f) {
            double a = pext(f), b = pext(f + 1);
            double rel = std::fabs(b - a) /
                         std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1.0);
            wave_face[f] = (rel > wave_rel) ? 1 : 0;
        }
        double tw = std::fmin(1.0, std::fmax(0.5, theta_wave));
        for (int i = 0; i < n; ++i)
            theta[i] = (wave_face[i] || wave_face[i + 1]) ? tw : 1.0;
    }

    // ── assemble R and J (2n x 2n), one dual pass per cell ───────────────────
    const int m = 2 * n;
    std::vector<double> A(m * m, 0.0), R(m, 0.0);
    for (int i = 0; i < n; ++i) {
        int il, ir, ill, irr, illl, irrr;
        bool lb = false, rb = false;
        if (periodic) {
            il = (i - 1 + n) % n; ir = (i + 1) % n;
            ill = (i - 2 + n) % n; irr = (i + 2) % n;
            illl = (i - 3 + n) % n; irrr = (i + 3) % n;
        } else {
            il = std::max(i - 1, 0); ir = std::min(i + 1, n - 1);
            ill = std::max(il - 1, 0); irr = std::min(ir + 1, n - 1);
            illl = std::max(ill - 1, 0); irrr = std::min(irr + 1, n - 1);
            lb = (i == 0); rb = (i == n - 1);
        }
        // weno5 per-cell availability (cell-based, clamped/periodic indices).
        bool avail_l, avail_r;
        if (periodic) { avail_l = true; avail_r = true; }
        else { avail_l = (i - 3 >= 0) && (i + 2 <= n - 1);
               avail_r = (i - 2 >= 0) && (i + 3 <= n - 1); }
        bool w5l = avail_l
            && same_pure(alpha[illl], alpha[ill], alpha_pure_tol)
            && same_pure(alpha[ill], alpha[il], alpha_pure_tol)
            && same_pure(alpha[il], alpha[i], alpha_pure_tol)
            && same_pure(alpha[i], alpha[ir], alpha_pure_tol)
            && same_pure(alpha[ir], alpha[irr], alpha_pure_tol);
        bool w5r = avail_r
            && same_pure(alpha[ill], alpha[il], alpha_pure_tol)
            && same_pure(alpha[il], alpha[i], alpha_pure_tol)
            && same_pure(alpha[i], alpha[ir], alpha_pure_tol)
            && same_pure(alpha[ir], alpha[irr], alpha_pure_tol)
            && same_pure(alpha[irr], alpha[irrr], alpha_pure_tol);

        // seed the 14 dual unknowns (order matches col_ids below).
        // z = [u_ll,u_l,u_c,u_r,u_rr, p_ll,p_l,p_c,p_r,p_rr, u_lll,u_rrr,p_lll,p_rrr]
        Dual u_ll, u_l, u_c, u_r, u_rr, p_ll, p_l, p_c, p_r, p_rr,
             u_lll, u_rrr, p_lll, p_rrr;
        u_ll.v = u0[ill];   u_ll.g[0] = 1.0;
        u_l.v  = u0[il];    u_l.g[1]  = 1.0;
        u_c.v  = u0[i];     u_c.g[2]  = 1.0;
        u_r.v  = u0[ir];    u_r.g[3]  = 1.0;
        u_rr.v = u0[irr];   u_rr.g[4] = 1.0;
        p_ll.v = p0[ill];   p_ll.g[5] = 1.0;
        p_l.v  = p0[il];    p_l.g[6]  = 1.0;
        p_c.v  = p0[i];     p_c.g[7]  = 1.0;
        p_r.v  = p0[ir];    p_r.g[8]  = 1.0;
        p_rr.v = p0[irr];   p_rr.g[9] = 1.0;
        u_lll.v = u0[illl]; u_lll.g[10] = 1.0;
        u_rrr.v = u0[irrr]; u_rrr.g[11] = 1.0;
        p_lll.v = p0[illl]; p_lll.g[12] = 1.0;
        p_rrr.v = p0[irrr]; p_rrr.g[13] = 1.0;

        double Z_l = Z[il], Z_c = Z[i], Z_r = Z[ir];
        double den_l = EOS::max2(Z_l + Z_c, EPS);
        double den_r = EOS::max2(Z_c + Z_r, EPS);
        char ho_l = high_face[i];      // left face of cell i is face i
        char ho_r = high_face[i + 1];

        // ── LEFT face (between il and i) ─────────────────────────────────────
        Dual p_fl_raw = (Z_c * p_l + Z_l * p_c + (Z_l * Z_c) * (u_l - u_c)) / den_l;
        Dual u_fl_raw = (p_l - p_c + Z_l * u_l + Z_c * u_c) / den_l;
        // centered_interface(on): componentwise superbee reconstructed states.
        Dual p_lh   = p_l + superbee_d(p_l - p_ll, p_c - p_l) * 0.5;
        Dual p_ch_l = p_c - superbee_d(p_c - p_l, p_r - p_c) * 0.5;
        Dual u_lh   = u_l + superbee_d(u_l - u_ll, u_c - u_l) * 0.5;
        Dual u_ch_l = u_c - superbee_d(u_c - u_l, u_r - u_c) * 0.5;
        if (w5l) {
            p_lh   = weno5_st(p_lll, p_ll, p_l, p_c, p_r);
            p_ch_l = weno5_st(p_rr, p_r, p_c, p_l, p_ll);
            u_lh   = weno5_st(u_lll, u_ll, u_l, u_c, u_r);
            u_ch_l = weno5_st(u_rr, u_r, u_c, u_l, u_ll);
        }
        Dual p_fl_ho = (Z_c * p_lh + Z_l * p_ch_l + (Z_l * Z_c) * (u_lh - u_ch_l)) / den_l;
        Dual u_fl_ho = (p_lh - p_ch_l + Z_l * u_lh + Z_c * u_ch_l) / den_l;
        Dual p_fl = ho_l ? p_fl_ho : p_fl_raw;
        Dual u_fl = ho_l ? u_fl_ho : u_fl_raw;

        // ── RIGHT face (between i and ir) ────────────────────────────────────
        Dual p_fr_raw = (Z_r * p_c + Z_c * p_r + (Z_c * Z_r) * (u_c - u_r)) / den_r;
        Dual u_fr_raw = (p_c - p_r + Z_c * u_c + Z_r * u_r) / den_r;
        Dual p_ch_r = p_c + superbee_d(p_c - p_l, p_r - p_c) * 0.5;
        Dual p_rh   = p_r - superbee_d(p_r - p_c, p_rr - p_r) * 0.5;
        Dual u_ch_r = u_c + superbee_d(u_c - u_l, u_r - u_c) * 0.5;
        Dual u_rh   = u_r - superbee_d(u_r - u_c, u_rr - u_r) * 0.5;
        if (w5r) {
            p_ch_r = weno5_st(p_ll, p_l, p_c, p_r, p_rr);
            p_rh   = weno5_st(p_rrr, p_rr, p_r, p_c, p_l);
            u_ch_r = weno5_st(u_ll, u_l, u_c, u_r, u_rr);
            u_rh   = weno5_st(u_rrr, u_rr, u_r, u_c, u_l);
        }
        Dual p_fr_ho = (Z_r * p_ch_r + Z_c * p_rh + (Z_c * Z_r) * (u_ch_r - u_rh)) / den_r;
        Dual u_fr_ho = (p_ch_r - p_rh + Z_c * u_ch_r + Z_r * u_rh) / den_r;
        Dual p_fr = ho_r ? p_fr_ho : p_fr_raw;
        Dual u_fr = ho_r ? u_fr_ho : u_fr_raw;

        // ── boundary face overrides (dirichlet inlet not on this ref path) ──
        Dual p_l_eff = p_l, p_r_eff = p_r;
        if (lb) {
            // p_fl_b = p_c (no left dirichlet); u_fl_b = 0 (reflective) else u_c.
            p_fl = p_c;
            u_fl = (bc_l == AcousticBC::reflective) ? Dual(0.0) : u_c;
            p_l_eff = p_c;
        }
        if (rb) {
            p_fr = p_c;   // no right pressure outlet on this path
            u_fr = (bc_r == AcousticBC::reflective) ? Dual(0.0) : u_c;
            p_r_eff = p_c;
        }

        Dual dp_back = (p_c - p_l_eff) / dx;
        Dual dp_forw = (p_r_eff - p_c) / dx;
        Dual dp_dx = u_mask[i] ? dp_back : dp_forw;

        double th = theta[i];
        Dual div_p = (p_fr - p_fl) * (th / dx) + Dual((1.0 - th) * div_p_old[i]);
        Dual div_u = (u_fr - u_fl) * (th / dx) + Dual((1.0 - th) * div_u_old[i]);
        Dual r_u = u_c * rho_star[i] - Dual(m_adv[i]) + div_p * dt;
        Dual r_p = p_c - Dual(p0[i]) + (dp_dx * u0[i] + div_u * beta[i]) * dt;

        // scatter into the global system.
        int cols[14] = { ill, il, i, ir, irr,
                         n + ill, n + il, n + i, n + ir, n + irr,
                         illl, irrr, n + illl, n + irrr };
        R[i]     = r_u.v;
        R[n + i] = r_p.v;
        for (int k = 0; k < 14; ++k) {
            A[i * m + cols[k]]       += r_u.g[k];
            A[(n + i) * m + cols[k]] += r_p.g[k];
        }
    }

    // ── diagonal Tikhonov floor + solve  (J + 1e-12*max(|diag|,1)*I) dy = -R ──
    double scale = 1.0;
    for (int k = 0; k < m; ++k) scale = std::fmax(scale, std::fabs(A[k * m + k]));
    double reg = 1e-12 * scale;
    for (int k = 0; k < m; ++k) A[k * m + k] += reg;
    std::vector<double> rhs(m);
    for (int k = 0; k < m; ++k) rhs[k] = -R[k];

    AcousticSolveResult out;
    out.Amat = A;                 // save regularized A before elimination
    out.Rhs = rhs;                // b = -R
    solve_dense_pp(A, rhs, m);    // rhs <- dy

    // residual of the returned solution: max_k |(A dy - b)_k| (long double accum).
    for (int r = 0; r < m; ++r) {
        long double s = 0.0L;
        for (int c = 0; c < m; ++c) s += (long double)out.Amat[r * m + c] * (long double)rhs[c];
        double res = std::fabs((double)(s - (long double)out.Rhs[r]));
        if (res > out.resid_self) out.resid_self = res;
    }

    out.u_new.resize(n);
    out.p_new.resize(n);
    for (int i = 0; i < n; ++i) {
        out.u_new[i] = u0[i] + rhs[i];
        out.p_new[i] = p0[i] + rhs[n + i];
    }
    return out;
}

} // namespace cfd
