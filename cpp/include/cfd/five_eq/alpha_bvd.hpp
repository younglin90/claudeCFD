// cfd/five_eq/alpha_bvd.hpp — adaptive_bvd volume-fraction face value (M4).
//
// C++ port of solver/five_eq_IMEX/explicit.py::_adaptive_bvd_alpha_face
// (410-456), _cicsam_alpha_face (63-114), _muscl_hancock_alpha_face (273-319)
// and _tvd_slope_1d (246-270). The face-value candidate is selected by the LOCAL
// alpha profile topology ONLY (never a case id): CICSAM compressive flux on a
// genuine sharp pure-material jump / narrow mixed layer, bounded MUSCL-Hancock
// TVD transport everywhere else.
//
// Header-only, POD, per-face scalar kernels marked CFD_ROUTINE_SEQ. The array
// driver does the whole-array topology reduction (a host-side branch decision)
// then the per-face loop. Every tolerance/clamp reproduces the Python EXACTLY so
// the result is bit-comparable (rel <= 1e-12) vs tests/5eq_ref/alpha_bvd_ref.txt.
#pragma once
#include "cfd/eos.hpp"   // CFD_ROUTINE_SEQ, EOS::max2
#include <algorithm>
#include <cmath>
#include <vector>

namespace cfd {

// Slope limiter family used by MUSCL-Hancock alpha transport (_tvd_slope_1d
// 246-270). Note this set (no vanalbada) differs from the primitive TvdLimiter.
enum class AlphaTvd { VanLeer, Minmod, Superbee, MC, Umist };

// Limiter slope in the upwind-oriented coordinate (_tvd_slope_1d 246-270).
// Returns 0 if d_up*d_down <= 0 or either is non-finite.
CFD_ROUTINE_SEQ
inline double tvd_slope_1d(double d_up, double d_down, AlphaTvd kind) {
    if (!(std::isfinite(d_up) && std::isfinite(d_down))) return 0.0;
    if (d_up * d_down <= 0.0) return 0.0;
    double sgn = (d_down > 0.0) ? 1.0 : (d_down < 0.0 ? -1.0 : 0.0);  // np.sign
    double au = std::fabs(d_up), ad = std::fabs(d_down);
    switch (kind) {
        case AlphaTvd::Minmod:
            return sgn * (au < ad ? au : ad);
        case AlphaTvd::Superbee: {
            double t1 = (2.0 * au < ad) ? 2.0 * au : ad;            // min(2|up|,|down|)
            double t2 = (au < 2.0 * ad) ? au : 2.0 * ad;            // min(|up|,2|down|)
            return sgn * (t1 > t2 ? t1 : t2);
        }
        case AlphaTvd::MC: {
            double m = 2.0 * au;                                    // min(2|up|,.5|up+down|,2|down|)
            double t = 0.5 * std::fabs(d_up + d_down);
            if (t < m) m = t;
            if (2.0 * ad < m) m = 2.0 * ad;
            return sgn * m;
        }
        case AlphaTvd::Umist: {
            double r = d_down / d_up;                               // psi(r)*d_up
            double m = 2.0 * r;
            double t1 = 0.25 + 0.75 * r;
            double t2 = 0.75 + 0.25 * r;
            if (t1 < m) m = t1;
            if (t2 < m) m = t2;
            if (2.0 < m) m = 2.0;
            if (m < 0.0) m = 0.0;                                   // max(0, min(...))
            return m * d_up;
        }
        case AlphaTvd::VanLeer:
        default:
            return 2.0 * d_up * d_down / (d_up + d_down);           // harmonic van Leer
    }
}

// CICSAM/CBC compressive face value for one interior face (_cicsam_alpha_face
// 76-95). base = the first-order upwind value at this face. far/up/down are the
// stencil cells (already picked by the caller per sign(u_face)). courant =
// clip(|u|dt/dx, 1e-12, 1). Returns base when the stencil is flat / non-monotone.
CFD_ROUTINE_SEQ
inline double cicsam_alpha_value(double base, double far, double up, double down,
                                 double courant) {
    double denom = down - far;
    if (std::fabs(denom) <= 1.0e-14) return base;
    double phi_c = (up - far) / denom;
    if (phi_c < 0.0 || phi_c > 1.0) return base;
    double phi_f = phi_c / courant;
    if (phi_f > 1.0) phi_f = 1.0;                                  // min(1, phi_c/Co)
    return far + phi_f * denom;
}

// MUSCL-Hancock TVD face value for one interior face (_muscl_hancock_alpha_face
// 286-299). far/up/down picked by the caller per sign(u_face). Clamped to the
// local [min,max](far,up,down).
CFD_ROUTINE_SEQ
inline double muscl_hancock_alpha_value(double far, double up, double down,
                                        double courant, AlphaTvd kind) {
    double slope = tvd_slope_1d(up - far, down - up, kind);
    double val = up + 0.5 * (1.0 - courant) * slope;
    double lo = far, hi = far;
    if (up < lo) lo = up; if (up > hi) hi = up;
    if (down < lo) lo = down; if (down > hi) hi = down;
    if (val < lo) val = lo;
    if (val > hi) val = hi;
    return val;
}

// np.isclose(a, b) with default rtol=1e-5, atol=1e-8: |a-b| <= atol + rtol*|b|.
CFD_ROUTINE_SEQ
inline bool np_isclose(double a, double b) {
    return std::fabs(a - b) <= (1.0e-8 + 1.0e-5 * std::fabs(b));
}

// clip helper matching np.clip(x, lo, hi).
CFD_ROUTINE_SEQ
inline double clip01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

// ── array drivers ──────────────────────────────────────────────────────────

// CICSAM alpha faces (_cicsam_alpha_face full array, 63-114). out length n_face.
inline void cicsam_alpha_face(const double* a_ext, int n_ext,
                              const double* u_face, int n_face,
                              double dt, double dx, double* out) {
    for (int f = 0; f < n_face; ++f)
        out[f] = (u_face[f] >= 0.0) ? a_ext[f] : a_ext[f + 1];     // upwind base
    for (int f = 1; f < n_face - 1; ++f) {
        double courant = std::fabs(u_face[f]) * dt / dx;
        if (courant < 1.0e-12) courant = 1.0e-12;                  // clip(., 1e-12, 1)
        if (courant > 1.0) courant = 1.0;
        double far, up, down;
        if (u_face[f] >= 0.0) { far = a_ext[f - 1]; up = a_ext[f];     down = a_ext[f + 1]; }
        else                  { far = a_ext[f + 2]; up = a_ext[f + 1]; down = a_ext[f];     }
        out[f] = cicsam_alpha_value(out[f], far, up, down, courant);
    }
    // Periodic-wrap duplicate-face handling (explicit.py 100-113).
    bool periodic = (n_ext >= 4) && np_isclose(a_ext[0], a_ext[n_ext - 2])
                                 && np_isclose(a_ext[n_ext - 1], a_ext[1]);
    if (periodic && n_face >= 2) {
        int f = n_face - 1;
        double courant = std::fabs(u_face[f]) * dt / dx;
        if (courant < 1.0e-12) courant = 1.0e-12;
        if (courant > 1.0) courant = 1.0;
        double far, up, down;
        if (u_face[f] >= 0.0) { far = a_ext[n_ext - 3]; up = a_ext[n_ext - 2]; down = a_ext[n_ext - 1]; }
        else                  { far = a_ext[2];         up = a_ext[1];         down = a_ext[0];         }
        double base = (u_face[f] >= 0.0) ? a_ext[f] : a_ext[f + 1];
        double val = cicsam_alpha_value(base, far, up, down, courant);
        out[0] = val;
        out[n_face - 1] = val;
    }
    for (int f = 0; f < n_face; ++f) out[f] = clip01(out[f]);
}

// MUSCL-Hancock alpha faces (_muscl_hancock_alpha_face full array, 273-319).
inline void muscl_hancock_alpha_face(const double* a_ext, int n_ext,
                                     const double* u_face, int n_face,
                                     double dt, double dx, AlphaTvd kind,
                                     double* out) {
    const double EPS = 1e-30;
    for (int f = 0; f < n_face; ++f)
        out[f] = (u_face[f] >= 0.0) ? a_ext[f] : a_ext[f + 1];
    for (int f = 1; f < n_face - 1; ++f) {
        double courant = std::fabs(u_face[f]) * dt / EOS::max2(dx, EPS);
        if (courant < 0.0) courant = 0.0;                          // clip(., 0, 1)
        if (courant > 1.0) courant = 1.0;
        double far, up, down;
        if (u_face[f] >= 0.0) { far = a_ext[f - 1]; up = a_ext[f];     down = a_ext[f + 1]; }
        else                  { far = a_ext[f + 2]; up = a_ext[f + 1]; down = a_ext[f];     }
        out[f] = muscl_hancock_alpha_value(far, up, down, courant, kind);
    }
    bool periodic = (n_ext >= 4) && np_isclose(a_ext[0], a_ext[n_ext - 2])
                                 && np_isclose(a_ext[n_ext - 1], a_ext[1]);
    if (periodic && n_face >= 2) {
        int f = n_face - 1;
        double courant = std::fabs(u_face[f]) * dt / EOS::max2(dx, EPS);
        if (courant < 0.0) courant = 0.0;
        if (courant > 1.0) courant = 1.0;
        double far, up, down;
        if (u_face[f] >= 0.0) { far = a_ext[n_ext - 3]; up = a_ext[n_ext - 2]; down = a_ext[n_ext - 1]; }
        else                  { far = a_ext[2];         up = a_ext[1];         down = a_ext[0];         }
        double val = muscl_hancock_alpha_value(far, up, down, courant, kind);
        out[0] = val;
        out[n_face - 1] = val;
    }
    for (int f = 0; f < n_face; ++f) out[f] = clip01(out[f]);
}

// adaptive_bvd volume-fraction face (_adaptive_bvd_alpha_face 410-456). Topology
// selector keyed on the local alpha profile only. alpha_pure_tol < 0 reproduces
// the Python default branch (uses 1e-12). out length n_face = n_ext - 1.
inline void adaptive_bvd_alpha_face(const double* a_ext, int n_ext,
                                    const double* u_face, int n_face,
                                    double dt, double dx, AlphaTvd kind,
                                    double alpha_pure_tol, double* out) {
    // pure_tol = max(eps^0.25, alpha_pure_tol_or_1e-12). np.finfo(float).eps^0.25
    // = 2^-13 = 0.0001220703125 EXACTLY (eps = 2^-52).
    const double EPS4 = 0.0001220703125;
    double apt = (alpha_pure_tol < 0.0) ? 1.0e-12 : alpha_pure_tol;
    double pure_tol = EPS4 > apt ? EPS4 : apt;
    double pure_band = pure_tol * (1.0 + 1.0e-9) + 1.0e-15;

    // interior = a_ext[1:-1] if n_ext > 2 else a_ext.
    int lo_idx = (n_ext > 2) ? 1 : 0;
    int hi_idx = (n_ext > 2) ? (n_ext - 1) : n_ext;   // exclusive
    double mn = a_ext[lo_idx], mx = a_ext[lo_idx];
    int low_count = 0, high_count = 0, mixed_count = 0;
    for (int i = lo_idx; i < hi_idx; ++i) {
        double a = a_ext[i];
        if (a < mn) mn = a;
        if (a > mx) mx = a;
        if (a <= pure_band) ++low_count;
        if (a >= 1.0 - pure_band) ++high_count;
        if (a > pure_band && a < 1.0 - pure_band) ++mixed_count;
    }
    bool has_low_pure = mn <= pure_band;
    bool has_high_pure = mx >= 1.0 - pure_band;

    bool has_sharp_pure_jump = false;
    for (int f = 0; f < n_face; ++f) {
        double al = a_ext[f], ar = a_ext[f + 1];
        if (((al <= pure_band) && (ar >= 1.0 - pure_band)) ||
            ((ar <= pure_band) && (al >= 1.0 - pure_band))) {
            has_sharp_pure_jump = true;
            break;
        }
    }
    int lh = low_count + high_count; if (lh < 1) lh = 1;
    bool has_narrow_mixed_layer = has_low_pure && has_high_pure && (mixed_count <= lh);

    if (has_low_pure && has_high_pure &&
        (has_sharp_pure_jump || has_narrow_mixed_layer)) {
        cicsam_alpha_face(a_ext, n_ext, u_face, n_face, dt, dx, out);
    } else {
        muscl_hancock_alpha_face(a_ext, n_ext, u_face, n_face, dt, dx, kind, out);
    }
}

inline void stacs_alpha_face(const double* a_ext, int n_ext, const double* u_face, int n_face, double* out) {
    for(int f=0;f<n_face;++f) out[f]=u_face[f]>=0.?a_ext[f]:a_ext[f+1];
    for(int f=1;f<n_face-1;++f) {
        const double far=u_face[f]>=0.?a_ext[f-1]:a_ext[f+2], up=u_face[f]>=0.?a_ext[f]:a_ext[f+1], down=u_face[f]>=0.?a_ext[f+1]:a_ext[f];
        const double den=down-far; if(std::fabs(den)<=1.e-14) continue; const double pc=(up-far)/den;
        double pf=pc;
        if(pc>=0. && pc<=1.) {
            if(pc<1./3.) pf=2.*pc;
            else if(pc<.5) pf=.5+.5*pc;
            else if(pc<2./3.) pf=1.5*pc;
            else pf=1.;
        }
        if(pf>=0. && pf<=1.) out[f]=far+pf*den;
    }
    for(int f=0;f<n_face;++f) out[f]=clip01(out[f]);
}

inline void mstacs_alpha_face(const double* a_ext, int n_ext, const double* u_face, int n_face,
                              double dt, double dx, double* out) {
    for(int f=0;f<n_face;++f) out[f]=u_face[f]>=0.?a_ext[f]:a_ext[f+1];
    const auto val=[&](int f,double far,double up,double down) {
        const double den=down-far; if(std::fabs(den)<=1.e-14) return up;
        const double pc=(up-far)/den; if(pc<0.||pc>1.) return up;
        double co=std::fabs(u_face[f])*dt/dx; if(co<1.e-12)co=1.e-12; if(co>1.)co=1.;
        const double pf=co<=.33?std::fmin(pc/std::fmax(co,1.e-12),1.):std::fmin(3.*pc,1.);
        return far+pf*den;
    };
    for(int f=1;f<n_face-1;++f) {
        const double far=u_face[f]>=0.?a_ext[f-1]:a_ext[f+2], up=u_face[f]>=0.?a_ext[f]:a_ext[f+1], down=u_face[f]>=0.?a_ext[f+1]:a_ext[f];
        const double den=down-far; if(std::fabs(den)>1.e-14) out[f]=val(f,far,up,down);
    }
    const bool periodic=n_ext>=4 && np_isclose(a_ext[0],a_ext[n_ext-2]) && np_isclose(a_ext[n_ext-1],a_ext[1]);
    if(periodic && n_face>=2) { const int f=n_face-1; const double far=u_face[f]>=0.?a_ext[n_ext-3]:a_ext[2], up=u_face[f]>=0.?a_ext[n_ext-2]:a_ext[1], down=u_face[f]>=0.?a_ext[n_ext-1]:a_ext[0];
        if(std::fabs(down-far)>1.e-14) out[0]=out[n_face-1]=val(f,far,up,down); }
    for(int f=0;f<n_face;++f) out[f]=clip01(out[f]);
}

inline void vanleer_alpha_face(const double* a_ext, int n_ext, const double* u_face, int n_face, double* out) {
    for(int f=0;f<n_face;++f) out[f]=u_face[f]>=0.?a_ext[f]:a_ext[f+1];
    for(int f=1;f<n_face-1;++f) {
        const double far=u_face[f]>=0.?a_ext[f-1]:a_ext[f+2], up=u_face[f]>=0.?a_ext[f]:a_ext[f+1], down=u_face[f]>=0.?a_ext[f+1]:a_ext[f];
        const double den=down-up; if(std::fabs(den)<=1.e-14) continue; const double ratio=(up-far)/den;
        if(!std::isfinite(ratio)||ratio<=0.) continue; double psi=2.*ratio/(1.+ratio); if(psi<0.)psi=0.;if(psi>2.)psi=2.;
        double v=up+.5*psi*den,lo=far,hi=far; if(up<lo)lo=up;if(up>hi)hi=up;if(down<lo)lo=down;if(down>hi)hi=down;
        out[f]=std::clamp(v,lo,hi);
    }
    for(int f=0;f<n_face;++f) out[f]=clip01(out[f]);
}

inline void thinc_alpha_face(const double* a_ext, int n_ext, const double* u_face, int n_face, double* out) {
    for(int f=0;f<n_face;++f) out[f]=u_face[f]>=0.?a_ext[f]:a_ext[f+1];
    constexpr double beta=1.6; const double tb=std::tanh(beta), cb=std::cosh(beta);
    const auto cell=[&](int i,bool right) { const double left=a_ext[i-1], centre=a_ext[i], r=a_ext[i+1], mn=std::fmin(left,r), range=std::fmax(left,r)-mn;
        if(range<=1.e-14) return centre; const double bar=(centre-mn)/range; if(bar<=1.e-12||bar>=1.-1.e-12)return centre;
        const double theta=r>=left?1.:-1., b=std::exp(theta*beta*(2.*bar-1.)), aa=(b/cb-1.)/tb;
        double val=right ? mn+.5*range*(1.+theta*(tb+aa)/std::fmax(1.+aa*tb,1.e-14)) : mn+.5*range*(1.+theta*aa);
        double lo=left,hi=left;if(centre<lo)lo=centre;if(centre>hi)hi=centre;if(r<lo)lo=r;if(r>hi)hi=r;return std::clamp(val,lo,hi); };
    for(int f=1;f<n_face-1;++f) out[f]=u_face[f]>=0.?cell(f,true):cell(f+1,false);
    for(int f=0;f<n_face;++f) out[f]=clip01(out[f]);
}

inline void thinc_bvd_alpha_face(const double* a_ext, int n_ext, const double* u_face, int n_face,
                                  double dt, double dx, AlphaTvd kind, double* out) {
    std::vector<double> smooth(n_face), sharp(n_face);
    muscl_hancock_alpha_face(a_ext,n_ext,u_face,n_face,dt,dx,kind,smooth.data());
    thinc_alpha_face(a_ext,n_ext,u_face,n_face,sharp.data());
    for(int f=0;f<n_face;++f) out[f]=smooth[f];
    for(int f=1;f<n_face-1;++f) { double co=std::fabs(u_face[f])*dt/EOS::max2(dx,1.e-30); if(co<0.)co=0.;if(co>1.)co=1.; if(co>=1.-1.e-12)continue;
        const double l=a_ext[f],r=a_ext[f+1],bs=std::fabs(smooth[f]-l)+std::fabs(r-smooth[f]),bt=std::fabs(sharp[f]-l)+std::fabs(r-sharp[f]); if(bt<bs)out[f]=sharp[f]; }
    const bool periodic=n_ext>=4 && np_isclose(a_ext[0],a_ext[n_ext-2]) && np_isclose(a_ext[n_ext-1],a_ext[1]); if(periodic&&n_face>=2)out[0]=out[n_face-1];
    for(int f=0;f<n_face;++f) out[f]=clip01(out[f]);
}

} // namespace cfd
