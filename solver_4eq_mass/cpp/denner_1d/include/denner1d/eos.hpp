#pragma once

#include <cmath>

#include "denner1d/types.hpp"

namespace denner1d {

struct PhaseProps {
    double rho = 0.0;
    double c = 0.0;
    double h = 0.0;
    double e_vol = 0.0;
    double zeta = 0.0;
    double phi = 0.0;
    double dh_dp = 0.0;
    double cp = 0.0;
    double dEdp = 0.0;
    double dEdT = 0.0;
};

// Lean thermo for the h->T inner Newton (the hottest kernel: ~60 EOS evals per cell per
// compute_R). Only rho, h, and their T-partials phi=drho/dT|p, cp=dh/dT|p -- it OMITS the
// sound speed c (a sqrt + extra div) and the p-partials that the full phase_props always
// computes. Header-inline so it folds into the caller and the compiler optimizes it in place
// (phase_props lives in a separate TU, so a call there cannot be inlined/DCE'd without LTO).
// The rho/h/phi/cp formulas are byte-identical to phase_props -> T_from_hstat is unchanged.
struct PhaseThermo { double rho, h, phi, cp; };
inline PhaseThermo phase_thermo(double p, double T, const Phase& ph) {
    const double gm1 = ph.gamma - 1.0;
    const double A = ph.kv * T * gm1 + ph.b * (p + ph.pinf) + 1.0e-300;
    const double ppinf = p + ph.pinf;
    PhaseThermo o;
    o.rho = ppinf / A;
    o.h   = ph.gamma * ph.kv * T + ph.b * p + ph.eta;
    o.phi = -ppinf * ph.kv * gm1 / (A * A + 1.0e-300);
    o.cp  = ph.gamma * ph.kv;
    return o;
}

// ---- volume fraction alpha  <->  mass fraction Y  (phase A) -------------------------------
// Denner mixture (both phases at the SAME p,T):
//     rho = alpha*rho_a + (1-alpha)*rho_b            (volume-fraction blend, Eq.37)
//     Y   = alpha*rho_a / rho                        (mass fraction of phase A)
// The inverse is EXPLICIT (no iteration): solving Y = alpha*rho_a/(alpha*rho_a+(1-alpha)*rho_b)
// for alpha gives
//     alpha*rho_a*(1-Y) = Y*(1-alpha)*rho_b  =>  alpha = Y*rho_b / (rho_a*(1-Y) + Y*rho_b),
// with rho_a = rho_a(p,T), rho_b = rho_b(p,T). Substituting back, the implied mixture density
// is the specific-volume (mass-fraction) blend
//     1/rho = Y/rho_a + (1-Y)/rho_b,   i.e.  rho = rho_a*rho_b/(rho_a*(1-Y)+Y*rho_b),
// identical to the volume-fraction blend evaluated at the corresponding alpha -- the two
// descriptions are the same mixture, only the transported variable differs.
// Floating point: alpha in {0,1} maps to Y in {0,1} EXACTLY and back (the off-phase term is a
// multiplication by 0.0), so a sharp interface with pure cells round-trips bit-for-bit.
inline double mass_fraction_from_alpha(double alpha, double rho_a, double rho_b) {
    const double num = alpha * rho_a;
    const double den = num + (1.0 - alpha) * rho_b;
    return den > 0.0 ? num / den : alpha;
}
inline double alpha_from_mass_fraction(double Y, double rho_a, double rho_b) {
    const double num = Y * rho_b;
    const double den = rho_a * (1.0 - Y) + num;
    return den > 0.0 ? num / den : Y;
}

// ---- closed-form NASG p-T-equilibrium inversion, given (v, e, Y) -- round 21 -----------------
// Given mixture specific volume v, specific internal energy e, and mass fraction Y of phase a,
// return the unique (p,T) the four-equation p-T-equilibrium closure implies -- i.e. the
// composition-frozen "UV flash" this solver needs once per step to remove the alpha-remap lag
// (docs/YADV_ROUND_21_PLAN.md sect.2). Per-phase NASG (eos.cpp): v_k = (g_k-1)*cv_k*T/(p+pinf_k)
// + b_k, h_k = g_k*cv_k*T + b_k*p + eta_k. Mixture at p-T equilibrium (mass fraction Y of a):
//     bbar = Y*b_a+(1-Y)*b_b        qbar = Y*eta_a+(1-Y)*eta_b
//     cpbar = Y*g_a*cv_a+(1-Y)*g_b*cv_b     cvbar = Y*cv_a+(1-Y)*cv_b   (cpbar-cvbar = Ka+Kb)
//     Ka = Y*(g_a-1)*cv_a           Kb = (1-Y)*(g_b-1)*cv_b
//     v(p,T) = bbar + T*S(p),  S(p) = Ka/(p+pinf_a) + Kb/(p+pinf_b)
//     h(p,T) = cpbar*T + bbar*p + qbar,   e = h - p*v
// Solving v(p,T)=v (i.e. T=(v-bbar)/S(p)) simultaneously with e(p,T)-p*v = e (i.e.
// T=(E0+p*W)/cpbar, W=v-bbar, E0=e-qbar) and clearing denominators gives a quadratic in p:
//     A2*p^2 + A1*p + A0 = 0
//     A2 = W*cvbar
//     A1 = cpbar*W*(pinf_a+pinf_b) - W*(Ka*pinf_b+Kb*pinf_a) - E0*(Ka+Kb)
//     A0 = cpbar*W*pinf_a*pinf_b   - E0*(Ka*pinf_b+Kb*pinf_a)
// Root selection: every phase pair this project uses has at most one phase with pinf!=0
// (verified against cases.cpp, docs/YADV_ROUND_21_PLAN.md sect.1), so pinf_a*pinf_b==0 =>
// A0<=0 always (E0>0 for admissible input) => either the two roots have opposite sign (take the
// positive one) or one root is exactly 0 (take the other, requiring it positive). This is NOT a
// general multi-stiffened-phase solver -- if a future case pairs two phases with pinf!=0 this
// root-selection argument needs revisiting (flagged, not built, since no such pair exists today).
// Prior art: Collis et al. 2025 sect.2.3 derives the same closed mixture pressure under the same
// at-most-one-stiffened-phase hypothesis (papers/library/md/newest5/2025_Collis_..._four_equation
// _thermodynamic_ENO.md); this derivation is independent, not transcribed from theirs (their
// equations are page images, not text). ok=false on ANY of: W<=0 (v below the covolume blend,
// unphysical), non-finite disc, no admissible positive root, p*<1, T* outside (1e-6,1e6) (the
// same ceiling T_from_hstat enforces -- this function can never itself trip the F2'' reason-5
// scan, acid.cpp:2328), or any non-finite output. Pure function: no state, no env reads.
struct MixPT { double p = 0.0; double T = 0.0; bool ok = false; };
inline MixPT pT_from_v_e_massfrac(double v, double e, double Y, const Phase& a, const Phase& b) {
    MixPT o;
    const double bbar = Y * a.b + (1.0 - Y) * b.b;
    const double qbar = Y * a.eta + (1.0 - Y) * b.eta;
    const double cpbar = Y * a.gamma * a.kv + (1.0 - Y) * b.gamma * b.kv;
    const double cvbar = Y * a.kv + (1.0 - Y) * b.kv;
    const double Ka = Y * (a.gamma - 1.0) * a.kv;
    const double Kb = (1.0 - Y) * (b.gamma - 1.0) * b.kv;
    const double W = v - bbar;
    const double E0 = e - qbar;
    if (!(W > 0.0)) return o;
    const double A2 = W * cvbar;
    const double A1 = cpbar * W * (a.pinf + b.pinf) - W * (Ka * b.pinf + Kb * a.pinf)
                     - E0 * (Ka + Kb);
    const double A0 = cpbar * W * a.pinf * b.pinf - E0 * (Ka * b.pinf + Kb * a.pinf);
    if (!(A2 > 0.0)) return o;
    const double disc = A1 * A1 - 4.0 * A2 * A0;
    if (!(disc >= 0.0) || !std::isfinite(disc)) return o;
    const double sq = std::sqrt(disc);
    const double qq = -0.5 * (A1 + std::copysign(sq, A1));
    double p_cand = -1.0;
    if (qq != 0.0) {
        const double r1 = qq / A2;
        const double r2 = A0 / qq;
        // A0<=0 in this project (verified, see comment above): at most one positive root exists
        // unless A0==0, in which case one root is exactly 0 -- pick the strictly positive one.
        if (r1 > 0.0 && std::isfinite(r1)) p_cand = r1;
        else if (r2 > 0.0 && std::isfinite(r2)) p_cand = r2;
    } else if (A0 == 0.0) {
        // qq==0 with A0==0: both roots are 0 -- no positive root, reject below.
    }
    if (!(p_cand >= 1.0) || !std::isfinite(p_cand)) return o;
    const double T_cand = (E0 + W * p_cand) / cpbar;
    if (!std::isfinite(T_cand) || !(T_cand > 1.0e-6) || !(T_cand < 1.0e6)) return o;
    o.p = p_cand;
    o.T = T_cand;
    o.ok = true;
    return o;
}

// ---- derivatives of alpha(Y, rho_a(p,T), rho_b(p,T)) at FIXED mass fraction Y ---------------
// From alpha = Y*rb / D with D = ra(1-Y) + Y*rb, the two identities alpha = Y*rb/D and
// 1-alpha = ra(1-Y)/D give
//     d(alpha)/d(ra) = -Y*rb*(1-Y)/D^2 = -alpha(1-alpha)/ra
//     d(alpha)/d(rb) = +Y*ra*(1-Y)/D^2 = +alpha(1-alpha)/rb
// Chaining through rho_k(p,T) with the EXISTING PhaseProps partials zeta = drho/dp|_T and
// phi = drho/dT|_p:
//     a_p = d(alpha)/dp|_{T,Y} = alpha(1-alpha) * ( zeta_b/rho_b - zeta_a/rho_a )
//     a_T = d(alpha)/dT|_{p,Y} = alpha(1-alpha) * ( phi_b /rho_b - phi_a /rho_a )
// zeta_k/rho_k is phase k's ISOTHERMAL compressibility (= 1/p for an ideal gas), -phi_k/rho_k
// its thermal expansivity. a_p < 0 for gas-in-liquid: compress -> gas volume fraction falls.
// The alpha(1-alpha) prefactor vanishes EXACTLY at both pure ends (a multiply by 0.0), so these
// are automatically consistent with the clamp(alpha,0,1) the residual applies -- no epsilon, no
// kink handling, no new constant.
// NOTE on a_T and the NASG covolume b: phi_k/rho_k = -kv_k(gamma_k-1)/A_k with
// A_k = kv_k(gamma_k-1)T + b_k(p+pinf_k) (eos.cpp), which is exactly -1/T for ANY phase with
// b_k = 0. So a_T is ALGEBRAICALLY zero whenever both phases have b == 0 (17 of this suite's 19
// cases -- only cases 14 and 15 use water_liquid_phase, b = 6.61e-4). It is NOT bitwise zero:
// phase_props evaluates phi/rho as (-ppinf*kv*gm1/A^2)/(ppinf/A), which does not round-trip
// ppinf and A exactly; the residual is <= ~2*eps*alpha(1-alpha)*|phi/rho| (measured worst
// 1.4e-17 over a wide (p,T,alpha,phase-pair) grid).
//
// Cross-check identity (asserted in denner1d_unit.cpp), exact for BOTH zeta and phi:
//     D_p + (rho_a - rho_b)*a_p  ==  rho * ( alpha*zeta_a/rho_a + (1-alpha)*zeta_b/rho_b )
//     where D_p = alpha*zeta_a + (1-alpha)*zeta_b is the FROZEN-alpha value acid.cpp uses today,
//     and the RHS is the (isothermal) Wood-type mixture compressibility.
struct AlphaDerivs { double a_p; double a_T; };
inline AlphaDerivs alpha_derivs_massfrac(double alpha,
                                         double zeta_a, double phi_a, double rho_a,
                                         double zeta_b, double phi_b, double rho_b) {
    const double w = alpha * (1.0 - alpha);
    AlphaDerivs o;
    o.a_p = w * (zeta_b / rho_b - zeta_a / rho_a);
    o.a_T = w * (phi_b  / rho_b - phi_a  / rho_a);
    return o;
}
inline double dalpha_dp_massfrac(double alpha, double zeta_a, double rho_a,
                                 double zeta_b, double rho_b) {
    return alpha * (1.0 - alpha) * (zeta_b / rho_b - zeta_a / rho_a);
}
inline double dalpha_dT_massfrac(double alpha, double phi_a, double rho_a,
                                 double phi_b, double rho_b) {
    return alpha * (1.0 - alpha) * (phi_b / rho_b - phi_a / rho_a);
}

Phase air_phase();
Phase water_liquid_phase();
Phase water_vapor_phase();

PhaseProps phase_props(double p, double T, const Phase& phase);
double mixture_density(double p, double T, double alpha, const Phase& a, const Phase& b);
double mixture_sound_speed(double p, double T, double alpha, const Phase& a, const Phase& b);
double mixture_enthalpy(double p, double T, double alpha, const Phase& a, const Phase& b);
double mixture_internal_energy_density(double p, double T, double alpha, const Phase& a, const Phase& b);
bool recover_pressure_temperature_from_density_energy(double rho,
                                                      double internal_energy_density,
                                                      double alpha,
                                                      const Phase& a,
                                                      const Phase& b,
                                                      double& p,
                                                      double& T);

}  // namespace denner1d
