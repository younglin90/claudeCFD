#include "denner1d/cases.hpp"
#include "denner1d/eos.hpp"
#include "denner1d/numerics.hpp"
#include "denner1d/solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "denner1d_unit FAIL: " << msg << "\n";
        ++failures;
    }
}
}  // namespace

int main() {
    // --- EOS sanity + conservative recovery round-trip -----------------------
    const auto air = denner1d::air_phase();
    const auto props = denner1d::phase_props(100000.0, 300.0, air);
    check(std::isfinite(props.rho) && props.rho > 0.0 && props.c > 0.0, "air phase_props");
    check(std::abs(denner1d::van_leer_phi(1.0) - 1.0) < 1.0e-12, "van_leer_phi(1)=1");

    const auto water = denner1d::water_liquid_phase();
    for (double alpha : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const double p0 = 8.0e6;
        const double T0 = 360.0;
        const double rho0 = denner1d::mixture_density(p0, T0, alpha, air, water);
        const double e0 = denner1d::mixture_internal_energy_density(p0, T0, alpha, air, water);
        double p = 1.1 * p0;
        double T = 0.9 * T0;
        const bool ok = denner1d::recover_pressure_temperature_from_density_energy(
            rho0, e0, alpha, air, water, p, T);
        check(ok, "recover converged");
        check(std::abs(p - p0) / p0 < 1.0e-9, "recover p");
        check(std::abs(T - T0) / T0 < 1.0e-9, "recover T");
    }

    // --- ACID_YADV: volume fraction <-> mass fraction round-trip -------------------
    // alpha -> Y -> alpha at a fixed (p,T), both phase orders. Two properties are asserted:
    //   (1) EXACT at the pure ends alpha in {0,1} (the off-phase term is a multiply by 0.0),
    //       which is what a sharp interface with pure cells needs;
    //   (2) in between, the error is machine eps times the CONDITION NUMBER of the map,
    //       kappa = max(rho_a/rho_b, rho_b/rho_a). This is not slop: Y = alpha*rho_a/rho
    //       compresses the whole alpha range into a sliver of the Y range when one phase is
    //       ~10^4x denser, so (1-Y) loses relative precision and the inverse magnifies it.
    //       Measured (scripts/yadv_cond.cpp): worst |d alpha| tracks eps*kappa to within 2x
    //       over p in [1e4,1e9], T in [250,1200], air/water/vapor pairs -- 1.7e-16 at
    //       kappa~1.5 up to 2.9e-12 at kappa~1.9e4 (water|air at p=1e4, T=1200).
    {
        double worst_ratio = 0.0;
        const denner1d::Phase pairs[][2] = {{air, water}, {water, air}};
        for (const auto& pr : pairs) {
            for (const double p0 : {1.0e4, 1.0e5, 8.0e6, 1.0e9}) {
                for (const double T0 : {250.0, 300.0, 360.0, 1200.0}) {
                    const double ra = denner1d::phase_props(p0, T0, pr[0]).rho;
                    const double rb = denner1d::phase_props(p0, T0, pr[1]).rho;
                    const double kappa = std::max(ra / rb, rb / ra);
                    // round 24: same function ACID_RECON_NULL uses (eos.hpp) -- both must use
                    // literally the same bound. The 2.220446049250313e-16 literal this replaced
                    // is asserted below instead of re-typed, so this test still pins the
                    // machine-epsilon value the bound is built from.
                    const double tol = denner1d::alpha_roundtrip_floor(ra, rb);
                    check(std::abs(tol - 8.0 * std::numeric_limits<double>::epsilon()
                                          * std::max(kappa, 1.0)) < 1e-30,
                          "alpha_roundtrip_floor matches its own formula");
                    check(std::numeric_limits<double>::epsilon() == 2.220446049250313e-16,
                          "machine epsilon is the expected IEEE-754 double value");
                    for (int k = 0; k <= 1000; ++k) {
                        const double al = static_cast<double>(k) / 1000.0;
                        const double Y = denner1d::mass_fraction_from_alpha(al, ra, rb);
                        const double al2 = denner1d::alpha_from_mass_fraction(Y, ra, rb);
                        const double err = std::abs(al2 - al);
                        worst_ratio = std::max(worst_ratio, err / tol);
                        if (k == 0) check(Y == 0.0 && al2 == 0.0, "Y round-trip exact at alpha=0");
                        if (k == 1000) check(Y == 1.0 && al2 == 1.0, "Y round-trip exact at alpha=1");
                        check(err <= tol, "alpha->Y->alpha round-trip within 8*eps*kappa");
                        // the mass-fraction (specific-volume) blend 1/rho = Y/ra + (1-Y)/rb must
                        // reproduce the volume-fraction blend rho = al*ra + (1-al)*rb exactly.
                        const double rho_v = al * ra + (1.0 - al) * rb;
                        const double rho_y = ra * rb / (ra * (1.0 - Y) + Y * rb);
                        check(std::abs(rho_y - rho_v) <= tol * rho_v, "Y/alpha mixture density agree");
                    }
                }
            }
        }
        if (worst_ratio > 1.0)
            std::cerr << "  round-trip worst err / (8 eps kappa) = " << worst_ratio << "\n";
    }

    // --- Phase 2 Stage 0: d(alpha)/dp, d(alpha)/dT at fixed mass fraction Y ----------------
    // Four properties, over the SAME (p,T,pair) grid the round-trip test above uses:
    //   (1) a_p, a_T reproduce a central FD of alpha_from_mass_fraction o phase_props;
    //   (2) the exact algebraic identity  D_p + (ra-rb)*a_p == rho*(al*za/ra + (1-al)*zb/rb),
    //       and its zeta->phi twin, to ~1e-12 relative;
    //   (3) for a b==0 / b==0 phase pair, a_T is zero to within the cancellation floor
    //       (~2*eps*al(1-al)*|phi/rho|). NOT bitwise zero: phase_props evaluates phi/rho as
    //       (-ppinf*kv*gm1/A^2)/(ppinf/A), which does not round-trip ppinf and A exactly.
    //       Measured worst over this grid: 1.39e-17 abs, 1.74*eps relative. See eos.hpp.
    //   (4) alpha in {0,1} => a_p == a_T == +0.0 EXACTLY (a multiply by 0.0), which is what
    //       makes these derivatives consistent with the residual's clamp(alpha,0,1).
    {
        const auto vapor = denner1d::water_vapor_phase();
        const double macheps = 2.220446049250313e-16;
        const denner1d::Phase pairs2[][2] = {{air, water}, {water, air},
                                             {air, vapor}, {vapor, air}};
        double worst_fd_p = 0.0, worst_fd_T = 0.0, worst_id = 0.0, worst_bzero = 0.0;
        for (const auto& pr : pairs2) {
            const bool both_b_zero = (pr[0].b == 0.0 && pr[1].b == 0.0);
            for (const double p0 : {1.0e4, 1.0e5, 8.0e6, 1.0e9}) {
                for (const double T0 : {250.0, 300.0, 360.0, 1200.0}) {
                    const auto pa = denner1d::phase_props(p0, T0, pr[0]);
                    const auto pb = denner1d::phase_props(p0, T0, pr[1]);
                    for (int k = 1; k <= 19; ++k) {
                        const double al = 0.05 * static_cast<double>(k);   // 0.05 .. 0.95
                        const double Y = denner1d::mass_fraction_from_alpha(al, pa.rho, pb.rho);
                        const auto d = denner1d::alpha_derivs_massfrac(
                            al, pa.zeta, pa.phi, pa.rho, pb.zeta, pb.phi, pb.rho);

                        // (1) central FD in p at fixed (Y, T)
                        const double hp = 1.0e-6 * p0;
                        const double ap_fd =
                            (denner1d::alpha_from_mass_fraction(
                                 Y, denner1d::phase_props(p0 + hp, T0, pr[0]).rho,
                                 denner1d::phase_props(p0 + hp, T0, pr[1]).rho)
                           - denner1d::alpha_from_mass_fraction(
                                 Y, denner1d::phase_props(p0 - hp, T0, pr[0]).rho,
                                 denner1d::phase_props(p0 - hp, T0, pr[1]).rho)) / (2.0 * hp);
                        // central FD in T at fixed (Y, p)
                        const double hT = 1.0e-6 * T0;
                        const double aT_fd =
                            (denner1d::alpha_from_mass_fraction(
                                 Y, denner1d::phase_props(p0, T0 + hT, pr[0]).rho,
                                 denner1d::phase_props(p0, T0 + hT, pr[1]).rho)
                           - denner1d::alpha_from_mass_fraction(
                                 Y, denner1d::phase_props(p0, T0 - hT, pr[0]).rho,
                                 denner1d::phase_props(p0, T0 - hT, pr[1]).rho)) / (2.0 * hT);
                        // Tolerance = max(1e-6 relative, an ABSOLUTE central-difference roundoff
                        // floor). BUG FOUND LIVE (round 5): the first version multiplied the
                        // floor by another 1e-6, which is wrong -- the floor already IS the
                        // achievable comparison precision (central-diff noise ~ eps/h), not a
                        // quantity to shrink further. This bit for the air|vapor pair, where
                        // a_p and a_T are ALGEBRAICALLY exactly zero (air and vapor share
                        // pinf=0, b=0, so zeta/rho = phi/rho are identical between the two
                        // phases -- see eos.hpp's a_T note) and the FD comparison is pure
                        // roundoff noise on both sides. Verified against a standalone probe:
                        // the analytic formula matches FD to full double precision wherever
                        // there is real signal (every air|water combo); only the near-zero
                        // air|vapor pair needs the floor, and needs it sized correctly.
                        const double floor_p = 16.0 * macheps / hp;
                        const double floor_T = 16.0 * macheps / hT;
                        const double tol_p = std::max(1.0e-6 * std::abs(d.a_p), floor_p);
                        const double tol_T = std::max(1.0e-6 * std::abs(d.a_T), floor_T);
                        worst_fd_p = std::max(worst_fd_p, std::abs(ap_fd - d.a_p) / tol_p);
                        worst_fd_T = std::max(worst_fd_T, std::abs(aT_fd - d.a_T) / tol_T);
                        check(std::abs(ap_fd - d.a_p) <= tol_p, "a_p vs central FD");
                        check(std::abs(aT_fd - d.a_T) <= tol_T, "a_T vs central FD");

                        // (2) the exact mixture-compressibility identity, p and T forms
                        const double rho = al * pa.rho + (1.0 - al) * pb.rho;
                        const double D_p = al * pa.zeta + (1.0 - al) * pb.zeta;
                        const double D_T = al * pa.phi  + (1.0 - al) * pb.phi;
                        const double lhs_p = D_p + (pa.rho - pb.rho) * d.a_p;
                        const double rhs_p = rho * (al * pa.zeta / pa.rho
                                                  + (1.0 - al) * pb.zeta / pb.rho);
                        const double lhs_T = D_T + (pa.rho - pb.rho) * d.a_T;
                        const double rhs_T = rho * (al * pa.phi / pa.rho
                                                  + (1.0 - al) * pb.phi / pb.rho);
                        worst_id = std::max(worst_id, std::abs(lhs_p - rhs_p) / std::abs(rhs_p));
                        worst_id = std::max(worst_id, std::abs(lhs_T - rhs_T) / std::abs(rhs_T));
                        check(std::abs(lhs_p - rhs_p) <= 1.0e-12 * std::abs(rhs_p),
                              "mixture-compressibility identity (zeta)");
                        check(std::abs(lhs_T - rhs_T) <= 1.0e-12 * std::abs(rhs_T),
                              "mixture-compressibility identity (phi)");

                        // (3) b==0 phase pair => a_T zero to the cancellation floor
                        if (both_b_zero) {
                            const double sc = al * (1.0 - al)
                                * std::max(std::abs(pa.phi / pa.rho), std::abs(pb.phi / pb.rho));
                            worst_bzero = std::max(worst_bzero, std::abs(d.a_T) / (macheps * sc));
                            check(std::abs(d.a_T) <= 8.0 * macheps * sc,
                                  "a_T == 0 (to cancellation floor) for b=0 phase pair");
                        }
                    }
                    // (4) endpoints are EXACT zeros (multiply by 0.0)
                    for (const double al : {0.0, 1.0}) {
                        const auto d0 = denner1d::alpha_derivs_massfrac(
                            al, pa.zeta, pa.phi, pa.rho, pb.zeta, pb.phi, pb.rho);
                        check(d0.a_p == 0.0 && d0.a_T == 0.0, "a_p==a_T==0 exactly at alpha in {0,1}");
                        check(denner1d::dalpha_dp_massfrac(al, pa.zeta, pa.rho, pb.zeta, pb.rho) == 0.0,
                              "dalpha_dp_massfrac exact 0 at endpoint");
                        check(denner1d::dalpha_dT_massfrac(al, pa.phi, pa.rho, pb.phi, pb.rho) == 0.0,
                              "dalpha_dT_massfrac exact 0 at endpoint");
                    }
                }
            }
        }
        std::cerr << "  Stage0 derivs: worst FD rel a_p=" << worst_fd_p
                  << " a_T=" << worst_fd_T << " ; worst identity rel=" << worst_id
                  << " ; worst |a_T|/(eps*scale) on b=0 pairs=" << worst_bzero << "\n";

        // --- Stage 0 deliverable 5: verify Phase-2 §1's numeric prediction at case15's state.
        // Printed, plus a LOOSE physical bound (the Wood-type mixture compressibility of a
        // bubbly liquid is orders of magnitude above the volume-blend value). Not a tuned
        // constant: the assertion is ">100x", the prediction is 521.56x.
        {
            const double p15 = 1.0e5;
            // case15 IC (cases.cpp): T = al*T(rho_air=1.3) + (1-al)*T(rho_water=1000)
            const double T15 = 348.2468430731;      // recompute if the IC ever changes
            const double al15 = 0.055;
            const auto pa = denner1d::phase_props(p15, T15, air);
            const auto pb = denner1d::phase_props(p15, T15, water);
            const auto d = denner1d::alpha_derivs_massfrac(
                al15, pa.zeta, pa.phi, pa.rho, pb.zeta, pb.phi, pb.rho);
            const double D_p = al15 * pa.zeta + (1.0 - al15) * pb.zeta;
            const double D_p_star = D_p + (pa.rho - pb.rho) * d.a_p;
            std::cerr << "  Stage0 case15 state: a_p=" << d.a_p << " a_T=" << d.a_T
                      << " D_p=" << D_p << " D_p*=" << D_p_star
                      << " ratio=" << D_p_star / D_p << " (Phase-2 sect.1 predicts ~500)\n";
            check(D_p_star / D_p > 100.0, "case15 continuity-diagonal defect exceeds 100x");
        }
    }

    // --- Phase 2 Stage 3a: the STARRED EOS-chain partials have exact closed forms ----------
    // With alpha = A(Y,p,T), N == D*(Y*h_a + (1-Y)*h_b) identically (N/D is exactly the
    // mass-fraction-weighted average, and Y := alpha*rho_a/rho by definition), so
    // hstat(Y,p,T) = Y*h_a(p,T) + (1-Y)*h_b(p,T) EXACTLY -- and NASG h_k = gamma_k*kv_k*T +
    // b_k*p + eta_k is LINEAR in T and p. Hence:
    //     hsT* = Y*cp_a + (1-Y)*cp_b     (strictly positive, in [min cp, max cp])
    //     hsp* = Y*b_a  + (1-Y)*b_b
    //     D_p* = rho^2*(Y*zeta_a/ra^2 + (1-Y)*zeta_b/rb^2)
    //     D_T* = rho^2*(Y*phi_a /ra^2 + (1-Y)*phi_b /rb^2)
    // hsp*/D_p* validate Stage 1 (round 6, already shipped, never checked this way before);
    // hsT*/D_T* validate Stage 3a. Grid includes a b!=0 pair (air|water) -- for b==0/eta==0
    // pairs (every other case) the identities hold trivially since N_T is an algebraic zero.
    {
        const auto vapor = denner1d::water_vapor_phase();
        const denner1d::Phase pairs3[][2] = {{air, water}, {water, air}, {air, vapor}};
        double worst = 0.0;
        for (const auto& pr : pairs3) {
            for (const double p0 : {1.0e4, 1.0e5, 8.0e6, 1.0e9}) {
                for (const double T0 : {6.942, 78.0, 300.0, 855.0, 3700.0}) {
                    const auto pa = denner1d::phase_props(p0, T0, pr[0]);
                    const auto pb = denner1d::phase_props(p0, T0, pr[1]);
                    for (int k = 1; k <= 19; ++k) {
                        const double al = 0.05 * static_cast<double>(k);
                        const double Y = denner1d::mass_fraction_from_alpha(al, pa.rho, pb.rho);
                        const auto d = denner1d::alpha_derivs_massfrac(
                            al, pa.zeta, pa.phi, pa.rho, pb.zeta, pb.phi, pb.rho);
                        const double D = al * pa.rho + (1.0 - al) * pb.rho;
                        const double D_p = al * pa.zeta + (1.0 - al) * pb.zeta;
                        const double D_T = al * pa.phi + (1.0 - al) * pb.phi;
                        const double N_p = al * (pa.zeta * pa.h + pa.rho * pa.dh_dp)
                                         + (1.0 - al) * (pb.zeta * pb.h + pb.rho * pb.dh_dp);
                        const double N_T = al * (pa.phi * pa.h + pa.rho * pa.cp)
                                         + (1.0 - al) * (pb.phi * pb.h + pb.rho * pb.cp);
                        const double D_ps = D_p + (pa.rho - pb.rho) * d.a_p;
                        const double D_Ts = D_T + (pa.rho - pb.rho) * d.a_T;
                        const double N_ps = N_p + (pa.rho * pa.h - pb.rho * pb.h) * d.a_p;
                        const double N_Ts = N_T + (pa.rho * pa.h - pb.rho * pb.h) * d.a_T;
                        const double hsT_star = (N_Ts * D - (al * pa.rho * pa.h + (1.0 - al) * pb.rho * pb.h) * D_Ts) / (D * D);
                        const double hsp_star = (N_ps * D - (al * pa.rho * pa.h + (1.0 - al) * pb.rho * pb.h) * D_ps) / (D * D);
                        const double hsT_closed = Y * pa.cp + (1.0 - Y) * pb.cp;
                        const double hsp_closed = Y * pr[0].b + (1.0 - Y) * pr[1].b;
                        const double D_p_closed = D * D * (Y * pa.zeta / (pa.rho * pa.rho)
                                                          + (1.0 - Y) * pb.zeta / (pb.rho * pb.rho));
                        const double D_T_closed = D * D * (Y * pa.phi / (pa.rho * pa.rho)
                                                          + (1.0 - Y) * pb.phi / (pb.rho * pb.rho));
                        // hsp_closed is EXACTLY 0 for a b=0/b=0 pair (air|vapor) at every Y --
                        // a legitimate closed-form zero, not underflow. A pure RELATIVE-error
                        // ratio blows up to nonsense there (found live, the same class of bug
                        // round 5's own unit test had). Use absolute-OR-relative combined:
                        // pass if the absolute difference is within an ABSOLUTE floor sized to
                        // this division's actual roundoff scale (D~O(1-1e4), N~O(1e5-1e10), so
                        // eps*N/D^2-family terms land well under 1e-9) OR within 1e-12 relative.
                        auto close = [](double got, double closed) {
                            return std::abs(got - closed) <= std::max(1.0e-9, 1.0e-12 * std::abs(closed));
                        };
                        auto abserr = [](double a_, double b_) { return std::abs(a_ - b_); };
                        worst = std::max(worst, abserr(hsT_star, hsT_closed));
                        worst = std::max(worst, abserr(hsp_star, hsp_closed));
                        worst = std::max(worst, abserr(D_ps, D_p_closed));
                        worst = std::max(worst, abserr(D_Ts, D_T_closed));
                        check(close(hsT_star, hsT_closed), "hsT* == Y*cp_a+(1-Y)*cp_b");
                        check(close(hsp_star, hsp_closed), "hsp* == Y*b_a+(1-Y)*b_b");
                        check(close(D_ps, D_p_closed), "D_p* closed form");
                        check(close(D_Ts, D_T_closed), "D_T* closed form");
                        check(hsT_star >= std::min(pa.cp, pb.cp) - 1.0e-9,
                              "hsT* bounded below by min(cp_a,cp_b)");
                        check(hsT_star <= std::max(pa.cp, pb.cp) + 1.0e-9,
                              "hsT* bounded above by max(cp_a,cp_b)");
                    }
                }
            }
        }
        std::cerr << "  Stage3a closed-form worst abs err = " << worst << "\n";
        // Pin the defect this stage addresses: the UNSTARRED hsT is negative at case14's
        // interface-region state (air|water, p=1e5, T=6.942K, alpha=0.5).
        {
            const auto pa = denner1d::phase_props(1.0e5, 6.942, air);
            const auto pb = denner1d::phase_props(1.0e5, 6.942, water);
            const double D = 0.5 * pa.rho + 0.5 * pb.rho;
            const double N = 0.5 * pa.rho * pa.h + 0.5 * pb.rho * pb.h;
            const double N_T = 0.5 * (pa.phi * pa.h + pa.rho * pa.cp)
                             + 0.5 * (pb.phi * pb.h + pb.rho * pb.cp);
            const double D_T = 0.5 * pa.phi + 0.5 * pb.phi;
            const double hsT_unstarred = (N_T * D - N * D_T) / (D * D);
            check(hsT_unstarred < 0.0, "unstarred hsT is negative at the case14 defect state");
        }
    }

    std::vector<double> a{1.0, 2.0, 3.0};
    auto g = denner1d::apply_ghost(a, "transmissive", "wall", 2, false);
    check(g.size() == 7, "ghost size");
    check(g[0] == 1.0 && g[1] == 1.0 && g[5] == 3.0 && g[6] == 2.0, "ghost values");

    // --- Round 21: pT_from_v_e_massfrac closed-form identity round-trip -------------------
    // Build (v,e,Y) exactly as eval_thermo/acid.cpp would from a KNOWN (p,T), then invert and
    // require the recovered (p,T) match. Also cross-checks against the existing INDEPENDENT
    // 2x2 Newton (recover_pressure_temperature_from_density_energy) at fixed alpha.
    {
        const auto vapor2 = denner1d::water_vapor_phase();
        const auto helium = denner1d::Phase{1.667, 0.0, 0.0, 3047.0, 0.0};
        const denner1d::Phase pairsPT[][2] = {
            {air, water}, {water, air}, {air, vapor2}, {vapor2, air}, {helium, air},
        };
        double worst_rel_p = 0.0, worst_rel_T = 0.0, worst_cross = 0.0;
        for (const auto& pr : pairsPT) {
            for (const double p0 : {1.0e4, 1.0e5, 1.0e7, 1.0e9, 1.5e10}) {
                for (const double T0 : {200.0, 300.0, 1.0e3, 1.0e4, 1.0e5}) {
                    for (const double Y : {0.0, 1.0e-4, 0.00116, 0.1, 0.5, 0.9, 1.0}) {
                        const auto pa = denner1d::phase_props(p0, T0, pr[0]);
                        const auto pb = denner1d::phase_props(p0, T0, pr[1]);
                        const double alpha = denner1d::alpha_from_mass_fraction(Y, pa.rho, pb.rho);
                        const double rho = alpha * pa.rho + (1.0 - alpha) * pb.rho;
                        if (!(rho > 0.0) || !std::isfinite(rho)) continue;
                        const double h_stat = (alpha * pa.rho * pa.h + (1.0 - alpha) * pb.rho * pb.h) / rho;
                        const double v = 1.0 / rho;
                        const double e = h_stat - p0 * v;
                        const auto r = denner1d::pT_from_v_e_massfrac(v, e, Y, pr[0], pr[1]);
                        check(r.ok, "pT_from_v_e_massfrac ok on a physically-built state");
                        if (!r.ok) continue;
                        const double rel_p = std::abs(r.p - p0) / p0;
                        const double rel_T = std::abs(r.T - T0) / T0;
                        worst_rel_p = std::max(worst_rel_p, rel_p);
                        worst_rel_T = std::max(worst_rel_T, rel_T);
                        check(rel_p < 1.0e-8, "pT_from_v_e_massfrac recovers p");
                        check(rel_T < 1.0e-8, "pT_from_v_e_massfrac recovers T");
                        // cross-check vs the existing independent frozen-alpha 2x2 Newton
                        double p2 = 1.1 * p0, T2 = 0.9 * T0;
                        const bool ok2 = denner1d::recover_pressure_temperature_from_density_energy(
                            rho, rho * e, alpha, pr[0], pr[1], p2, T2);
                        if (ok2) {
                            const double cross_p = std::abs(r.p - p2) / p0;
                            worst_cross = std::max(worst_cross, cross_p);
                            check(cross_p < 1.0e-6, "pT_from_v_e_massfrac vs independent Newton agree");
                        }
                    }
                }
            }
        }
        std::cerr << "  Round21 pT_from_v_e_massfrac: worst rel_p=" << worst_rel_p
                  << " worst rel_T=" << worst_rel_T << " worst vs-Newton=" << worst_cross << "\n";

        // rejection: an inadmissible input (v below the covolume blend) must return ok=false
        // and touch nothing -- no fallback, no clamp-and-continue.
        {
            const auto r = denner1d::pT_from_v_e_massfrac(-1.0, 1.0e5, 0.5, air, water);
            check(!r.ok, "pT_from_v_e_massfrac rejects v<=bbar");
        }
        // gas-gas degenerate pair (both pinf==0 => A0==0): must still pick the nonzero root.
        {
            const auto pa = denner1d::phase_props(1.0e5, 300.0, helium);
            const auto pb = denner1d::phase_props(1.0e5, 300.0, air);
            const double alpha = denner1d::alpha_from_mass_fraction(0.3, pa.rho, pb.rho);
            const double rho = alpha * pa.rho + (1.0 - alpha) * pb.rho;
            const double h_stat = (alpha * pa.rho * pa.h + (1.0 - alpha) * pb.rho * pb.h) / rho;
            const double v = 1.0 / rho;
            const double e = h_stat - 1.0e5 * v;
            const auto r = denner1d::pT_from_v_e_massfrac(v, e, 0.3, helium, air);
            check(r.ok, "pT_from_v_e_massfrac gas-gas degenerate ok");
            check(r.ok && std::abs(r.p - 1.0e5) / 1.0e5 < 1.0e-8, "pT_from_v_e_massfrac gas-gas p");
        }
    }

    // --- conservative operator invariant: static air-water interface stays static
    // (interface-equilibrium / Collis IEC property). This is the invariant the
    // double-flux recovery broke; the production scheme must hold it to roundoff.
    {
        const auto c01 = denner1d::find_case("01");
        const auto s = denner1d::solve_case(c01);
        double p0 = s.p.empty() ? 0.0 : s.p.front();
        double max_dp = 0.0, max_u = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < s.x.size(); ++i) {
            max_dp = std::max(max_dp, std::abs(s.p[i] - p0));
            max_u = std::max(max_u, std::abs(s.u[i]));
            finite = finite && std::isfinite(s.p[i]) && std::isfinite(s.u[i]);
        }
        check(finite, "case01 finite");
        check(max_dp < 1.0, "case01 interface-equilibrium pressure (|dp| < 1 Pa)");
        check(max_u < 1.0e-4, "case01 interface-equilibrium velocity (|u| < 1e-4)");
    }

    if (failures == 0) {
        std::cout << "denner1d_unit ok\n";
        return 0;
    }
    std::cerr << "denner1d_unit: " << failures << " failure(s)\n";
    return 1;
}
