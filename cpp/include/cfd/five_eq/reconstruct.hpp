// cfd/five_eq/reconstruct.hpp — tmlpu bounded primitive reconstruction (M5).
//
// C++ port of solver/five_eq_IMEX/reconstruction.py:70-232. The 1D bounded
// T-MLP-u face state is a TVD-limited increment additionally clipped by a local
// three-cell maximum-principle (LMP, Kim-Kim spirit) bound, with an optional
// MUSCL-Hancock time-centering factor (1 - C_u) on the increment. This header
// ports the per-face SCALAR kernels marked CFD_ROUTINE_SEQ; the array drivers
// reproduce reconstruct_upwind_faces / reconstruct_primitive_upwind_faces
// (upwind cell picked as L via u_face sign) and reconstruct_lr_faces exactly.
//
// PRODUCTION limiter = superbee (FIVE_EQ_IMEX_TMLPU_TVD=superbee, per
// docs/five_eq_IMEX_governing_equations_and_numerics.md + imex_ad.py:723,3033
// default 'superbee'). reconstruction.py's own primitive_tvd_kind() defaults to
// 'vanleer', so the two agree only when the env var is set; TvdLimiter::Superbee
// is the default here to match the production BASE_ENV. Every clamp/floor
// constant reproduces the Python guard EXACTLY (_EPS = 1e-30) so results are
// bit-comparable (rel <= 1e-12) vs tests/5eq_ref/reconstruct_ref.txt.
#pragma once
#include "cfd/eos.hpp"   // CFD_ROUTINE_SEQ, EOS::max2
#include <cmath>

namespace cfd {

// TVD limiter family (reconstruction.py::_tvd_limiter 70-89). Superbee is the
// production default; the others match the Python aliases in _TVD_PRIMITIVE_SCHEMES.
enum class TvdLimiter { Superbee, VanLeer, Minmod, MC, VanAlbada, Umist };

// psi(r) clamped to [0, 2]; returns 0 for r <= 0 or non-finite (reconstruction.py 70-89).
CFD_ROUTINE_SEQ
inline double tvd_limiter(double r, TvdLimiter kind) {
    if (!std::isfinite(r) || r <= 0.0) return 0.0;
    double psi;
    switch (kind) {
        case TvdLimiter::Minmod:
            psi = (1.0 < r) ? 1.0 : r;                         // min(1, r)
            break;
        case TvdLimiter::Superbee: {
            double a = (2.0 * r < 1.0) ? 2.0 * r : 1.0;        // min(2r, 1)
            double b = (r < 2.0) ? r : 2.0;                    // min(r, 2)
            psi = a > b ? a : b;                               // max(min(2r,1), min(r,2), 0)
            if (psi < 0.0) psi = 0.0;
            break;
        }
        case TvdLimiter::MC: {
            double m = 2.0 * r;                                // min(2r, 0.5(1+r), 2)
            double t = 0.5 * (1.0 + r);
            if (t < m) m = t;
            if (2.0 < m) m = 2.0;
            psi = m > 0.0 ? m : 0.0;                           // max(0, ...)
            break;
        }
        case TvdLimiter::VanAlbada:
            psi = (r * r + r) / (r * r + 1.0);                 // (r^2+r)/(r^2+1)
            break;
        case TvdLimiter::Umist: {
            double m = 2.0 * r;                                // min(2r, .25+.75r, .75+.25r, 2)
            double t1 = 0.25 + 0.75 * r;
            double t2 = 0.75 + 0.25 * r;
            if (t1 < m) m = t1;
            if (t2 < m) m = t2;
            if (2.0 < m) m = 2.0;
            psi = m > 0.0 ? m : 0.0;
            break;
        }
        case TvdLimiter::VanLeer:
        default:
            psi = 2.0 * r / (1.0 + r);                         // van Leer harmonic
            break;
    }
    if (!(psi > 0.0)) psi = 0.0;                               // max(0, min(2, psi)); NaN->0
    if (psi > 2.0) psi = 2.0;
    return psi;
}

// TVD-limited value from cell L to the L/R face (reconstruction.py::_limited_value
// 92-129). Stencil is (phi_LL, phi_L, phi_R) already ordered upwind->downwind.
// When has_courant, the increment is multiplied by (1 - clip(|courant|,0,1))
// (MUSCL-Hancock time centering). Then a local-max-principle bound clips psi so
// the result never leaves [min,max](phi_LL,phi_L,phi_R).
CFD_ROUTINE_SEQ
inline double limited_value(double phi_LL, double phi_L, double phi_R,
                            TvdLimiter kind, double courant, bool has_courant) {
    const double EPS = 1e-30;
    if (!(std::isfinite(phi_LL) && std::isfinite(phi_L) && std::isfinite(phi_R)))
        return phi_L;
    double num = phi_R - phi_L;
    double den = phi_L - phi_LL;
    if (std::fabs(num) <= 1.0e-300) return phi_L;
    if (std::fabs(den) <= 1.0e-300) return phi_L;
    double r = num / den;
    double psi = tvd_limiter(r, kind);
    if (psi <= 0.0) return phi_L;

    double delta = 0.5 * num;
    if (has_courant) {
        double c = std::fabs(courant);
        if (c > 1.0) c = 1.0;
        if (c < 0.0) c = 0.0;
        delta *= (1.0 - c);
    }
    double lo = phi_LL, hi = phi_LL;
    if (phi_L < lo) lo = phi_L; if (phi_L > hi) hi = phi_L;
    if (phi_R < lo) lo = phi_R; if (phi_R > hi) hi = phi_R;

    double psi_bound;
    if (delta > 0.0)       psi_bound = (hi - phi_L) / EOS::max2(delta, EPS);
    else if (delta < 0.0)  psi_bound = (lo - phi_L) / (delta < -EPS ? delta : -EPS);  // min(delta,-EPS)
    else                   psi_bound = 0.0;
    if (!std::isfinite(psi_bound)) psi_bound = 0.0;

    // psi = max(0, min(2, psi, psi_bound))
    double p = psi;
    if (2.0 < p) p = 2.0;
    if (psi_bound < p) p = psi_bound;
    if (p < 0.0) p = 0.0;

    double val = phi_L + p * delta;
    if (val < lo) val = lo;                                    // min(hi, max(lo, ...))
    if (val > hi) val = hi;
    return val;
}

// Face-state kernel used by the material fluxes: given a cell-centred field on
// the ghost-extended stencil and a face-velocity sign, reconstruct the face
// value from the UPWIND cell (reconstruct_upwind_faces semantics 160-197).
//   phi_ext : ghost-extended field, length n_ext.
//   f       : face index in [0, n_ext-2].
//   u_face  : face velocity (sign selects the upwind side / stencil).
//   courant : |u_face|*dt/dx (used only if has_courant).
// If the required 3-cell stencil runs off the array, returns the first-order
// upwind base value (matches the Python "continue" that keeps face[f]).
CFD_ROUTINE_SEQ
inline double reconstruct_upwind_face_value(const double* phi_ext, int n_ext,
                                            int f, double u_face,
                                            TvdLimiter kind,
                                            double courant, bool has_courant) {
    double base = (u_face >= 0.0) ? phi_ext[f] : phi_ext[f + 1];
    int i_ll, i_l, i_r;
    if (u_face >= 0.0) { i_ll = f - 1; i_l = f;     i_r = f + 1; }
    else               { i_ll = f + 2; i_l = f + 1; i_r = f;     }
    if (i_ll < 0 || i_ll >= n_ext || i_l < 0 || i_l >= n_ext ||
        i_r < 0 || i_r >= n_ext)
        return base;
    return limited_value(phi_ext[i_ll], phi_ext[i_l], phi_ext[i_r],
                         kind, courant, has_courant);
}

// Bounded third-order WENO value (reconstruction.py::_weno3_value).
CFD_ROUTINE_SEQ
inline double weno3_value(double phi_LL, double phi_L, double phi_R) {
    if (!(std::isfinite(phi_LL) && std::isfinite(phi_L) && std::isfinite(phi_R))) return phi_L;
    const double q0=1.5*phi_L-.5*phi_LL, q1=.5*(phi_L+phi_R);
    const double b0=(phi_L-phi_LL)*(phi_L-phi_LL), b1=(phi_R-phi_L)*(phi_R-phi_L);
    double scale=1.0; scale=std::fmax(scale,std::fabs(phi_LL)); scale=std::fmax(scale,std::fabs(phi_L)); scale=std::fmax(scale,std::fabs(phi_R));
    const double eps=1.e-12*scale*scale;
    const double a0=(1./3.)/((eps+b0)*(eps+b0)), a1=(2./3.)/((eps+b1)*(eps+b1));
    double val=(a0*q0+a1*q1)/EOS::max2(a0+a1,1.e-30);
    double lo=phi_LL, hi=phi_LL;
    if(phi_L<lo)lo=phi_L; if(phi_L>hi)hi=phi_L; if(phi_R<lo)lo=phi_R; if(phi_R>hi)hi=phi_R;
    if(val<lo)val=lo; if(val>hi)val=hi;
    return val;
}

// Array driver: reconstruct_upwind_faces (reconstruction.py 160-197) for a TVD /
// tmlpu scheme. out has length n_ext-1 (= n_face). dt/dx enable the MUSCL-Hancock
// courant factor (pass has_courant=false to disable, as for the 'u' variable when
// dt/dx are None — but production reconstruct_primitive_upwind_faces passes both).
// floor<0 means "no floor" (Python floor=None); otherwise face=max(face,floor).
inline void reconstruct_upwind_faces(const double* phi_ext, int n_ext,
                                     const double* u_face, TvdLimiter kind,
                                     double dt, double dx, bool has_courant,
                                     double floor, double* out) {
    const double EPS = 1e-30;
    int n_face = n_ext - 1;
    for (int f = 0; f < n_face; ++f) {
        double courant = 0.0;
        if (has_courant)
            courant = std::fabs(u_face[f]) * dt / EOS::max2(dx, EPS);
        double v = reconstruct_upwind_face_value(phi_ext, n_ext, f, u_face[f],
                                                 kind, courant, has_courant);
        if (floor >= 0.0 && v < floor) v = floor;
        out[f] = v;
    }
}

inline void reconstruct_weno3_upwind_faces(const double* phi_ext, int n_ext,
                                           const double* u_face, double floor, double* out) {
    const int n_face=n_ext-1;
    for(int f=0;f<n_face;++f) {
        double v=(u_face[f]>=0.)?phi_ext[f]:phi_ext[f+1];
        const int ill=u_face[f]>=0.?f-1:f+2, il=u_face[f]>=0.?f:f+1, ir=u_face[f]>=0.?f+1:f;
        if(ill>=0 && ill<n_ext && il>=0 && il<n_ext && ir>=0 && ir<n_ext)
            v=weno3_value(phi_ext[ill],phi_ext[il],phi_ext[ir]);
        if(floor>=0. && v<floor)v=floor;
        out[f]=v;
    }
}

// L/R face states for conservative solvers (reconstruct_lr_faces 200-232). No
// courant factor is applied (matches the Python: _limited_value called without
// courant). left[f] reconstructs from cell f to its right face; right[f] from
// cell f+1 to its left face (mirrored stencil). floor<0 disables the floor.
inline void reconstruct_lr_faces(const double* phi_ext, int n_ext,
                                 TvdLimiter kind, double floor,
                                 double* left, double* right) {
    int n_face = n_ext - 1;
    for (int f = 0; f < n_face; ++f) {
        left[f]  = phi_ext[f];
        right[f] = phi_ext[f + 1];
        if (f - 1 >= 0)
            left[f] = limited_value(phi_ext[f - 1], phi_ext[f], phi_ext[f + 1],
                                    kind, 0.0, false);
        if (f + 2 < n_ext)
            right[f] = limited_value(phi_ext[f + 2], phi_ext[f + 1], phi_ext[f],
                                     kind, 0.0, false);
        if (floor >= 0.0) {
            if (left[f]  < floor) left[f]  = floor;
            if (right[f] < floor) right[f] = floor;
        }
    }
}

} // namespace cfd
