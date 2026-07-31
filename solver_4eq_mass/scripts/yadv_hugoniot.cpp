// TASK B -- standalone, OUTSIDE the validation path. Computes, for cases 24 / 33 / 34, the
// post-shock state under TWO closures and prints them side by side:
//
//   (A) "alpha-held"  -- exactly what cpp/denner_1d/src/cases.cpp:compute_case24_shock builds:
//       Denner's mixture Rankine-Hugoniot Eqs.59-62 with an equivalent stiffened gas
//       (gamma_mix from Eq.57, Pihat from Eq.60), then  alpha_post := alpha_pre.
//       This is the reference the validation gates compare against.
//
//   (B) "Y-consistent" -- the mass fractions Y_k are held across the shock (no phase change)
//       and the mixture Rankine-Hugoniot jump conditions are imposed with the TRUE NASG
//       mixture EOS at a single (p,T); alpha_post is then a RESULT of the EOS, not an input.
//
// Neither cases.cpp nor validation.cpp is modified or linked-into; this file only uses the
// public EOS (phase_props) from libdenner1d.a and re-derives the two Hugoniots locally.
//
// build:
//   g++ -O2 -std=c++17 -Icpp/denner_1d/include scripts/yadv_hugoniot.cpp \
//       -o /tmp/yadv_hugoniot build-cpp/cpp/denner_1d/libdenner1d.a -fopenmp

#include "denner1d/eos.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using denner1d::Phase;
using denner1d::phase_props;

// ---- mixture at a single (p,T) --------------------------------------------------------------
static double rho_a_of(double p, double T, const Phase& a) { return phase_props(p, T, a).rho; }

// volume-fraction blend (Eq.37)
static double rho_mix_alpha(double p, double T, double al, const Phase& a, const Phase& b) {
    return al * phase_props(p, T, a).rho + (1.0 - al) * phase_props(p, T, b).rho;
}
// specific volume from the MASS fraction:  1/rho = Y/rho_a + (1-Y)/rho_b
static double v_mix_Y(double p, double T, double Y, const Phase& a, const Phase& b) {
    return Y / phase_props(p, T, a).rho + (1.0 - Y) / phase_props(p, T, b).rho;
}
// mixture specific enthalpy (mass-weighted -- identical to the alpha-weighted form of eos.cpp)
static double h_mix_Y(double p, double T, double Y, const Phase& a, const Phase& b) {
    return Y * phase_props(p, T, a).h + (1.0 - Y) * phase_props(p, T, b).h;
}

// T from (p, rho, alpha) -- byte-equivalent to cases.cpp:temperature_for_mixture_density_pressure
static double T_from_p_rho_alpha(double p, double rho, double al, const Phase& a, const Phase& b) {
    double lo = 1.0e-6, hi = 1.0;
    while (rho_mix_alpha(p, hi, al, a, b) > rho && hi < 1.0e9) hi *= 2.0;
    for (int k = 0; k < 200; ++k) {
        const double mid = 0.5 * (lo + hi);
        if (rho_mix_alpha(p, mid, al, a, b) > rho) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// T from (p, v, Y): specific volume increases with T at fixed p -> monotone bisection.
static double T_from_p_v_Y(double p, double v, double Y, const Phase& a, const Phase& b) {
    double lo = 1.0e-4, hi = 1.0;
    while (v_mix_Y(p, hi, Y, a, b) < v && hi < 1.0e10) hi *= 2.0;
    for (int k = 0; k < 200; ++k) {
        const double mid = 0.5 * (lo + hi);
        if (v_mix_Y(p, mid, Y, a, b) < v) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

struct State {
    double p = 0, u = 0, T = 0, rho = 0, alpha = 0, Y = 0;
};

int main() {
    const Phase air{1.4, 0.0, 0.0, 720.25, 0.0};           // == denner1d::air_phase()
    const Phase water{4.1, 4.4e8, 0.0, 474.2, 0.0};        // == cases.cpp denner_water
    const double Ms = 10.0;

    struct CaseSpec { const char* id; double alpha_air; };
    const CaseSpec specs[] = {{"24", 0.50}, {"33", 0.75}, {"34", 0.25}};

    for (const auto& cs : specs) {
        // ---------------- pre-shock state (identical for both closures) ----------------
        State pre;
        pre.p = 1.0e5;
        pre.u = 0.0;
        pre.alpha = cs.alpha_air;
        const double rho_air_ref = 1.1574, rho_water_ref = 998.0;
        double rho_seed = pre.alpha * rho_air_ref + (1.0 - pre.alpha) * rho_water_ref;
        pre.T = T_from_p_rho_alpha(pre.p, rho_seed, pre.alpha, air, water);
        const double ra_pre = rho_a_of(pre.p, pre.T, air);
        const double rb_pre = phase_props(pre.p, pre.T, water).rho;
        pre.rho = pre.alpha * ra_pre + (1.0 - pre.alpha) * rb_pre;
        pre.Y = pre.alpha * ra_pre / pre.rho;

        // ---------------- shock speed: Denner Eq.57-58 mixture sound speed ----------------
        const double cpa = phase_props(pre.p, pre.T, air).cp;
        const double cpb = phase_props(pre.p, pre.T, water).cp;
        const double inv_gm1 = pre.alpha / (air.gamma - 1.0) + (1.0 - pre.alpha) / (water.gamma - 1.0);
        const double gamma_mix = 1.0 + 1.0 / inv_gm1;
        const double cp_mix =
            (pre.alpha * ra_pre * cpa + (1.0 - pre.alpha) * rb_pre * cpb) / pre.rho;
        const double c_pre = std::sqrt((gamma_mix - 1.0) * cp_mix * pre.T);
        const double Vs = Ms * c_pre;

        // ================= closure (A): alpha held (== cases.cpp reference) =================
        State A;
        const double Pihat = ((gamma_mix - 1.0) / gamma_mix) * pre.rho * cp_mix * pre.T - pre.p;
        const double pr = 1.0 + (2.0 * gamma_mix / (gamma_mix + 1.0)) * (Ms * Ms - 1.0)
                                    * (1.0 + Pihat / pre.p);
        A.p = pr * pre.p;
        const double G = (gamma_mix + 1.0) / (gamma_mix - 1.0);
        const double pratio = (A.p + Pihat) / (pre.p + Pihat);
        A.rho = pre.rho * (G * pratio + 1.0) / (G + pratio);
        A.u = Vs * (1.0 - pre.rho / A.rho);
        A.alpha = pre.alpha;                                    // <-- the frozen assumption
        A.T = T_from_p_rho_alpha(A.p, A.rho, A.alpha, air, water);
        A.Y = A.alpha * rho_a_of(A.p, A.T, air) / A.rho;        // NOT equal to pre.Y

        // ================= closure (B): Y held, true NASG mixture RH =======================
        // shock frame, pre-shock stationary:  mdot = rho_pre*Vs
        //   Rayleigh : p_post = p_pre + mdot^2 (v_pre - v_post)
        //   Hugoniot : h_post - h_pre = 0.5 (p_post - p_pre)(v_pre + v_post)
        //   EOS      : v_post = Y/rho_a(p_post,T_post) + (1-Y)/rho_b(p_post,T_post)
        const double mdot = pre.rho * Vs;
        const double v_pre = 1.0 / pre.rho;
        const double h_pre = h_mix_Y(pre.p, pre.T, pre.Y, air, water);

        auto residual = [&](double v_post, State* out) {
            const double p_post = pre.p + mdot * mdot * (v_pre - v_post);
            const double T_post = T_from_p_v_Y(p_post, v_post, pre.Y, air, water);
            const double h_post = h_mix_Y(p_post, T_post, pre.Y, air, water);
            const double F = (h_post - h_pre) - 0.5 * (p_post - pre.p) * (v_pre + v_post);
            if (out) {
                out->p = p_post;
                out->T = T_post;
                out->rho = 1.0 / v_post;
                out->u = Vs * (1.0 - v_post / v_pre);
                out->Y = pre.Y;
                out->alpha = std::clamp(pre.Y * (1.0 / v_post) / rho_a_of(p_post, T_post, air),
                                        0.0, 1.0);
            }
            return F;
        };

        // bracket the compressive root  v_post in (0, v_pre)
        double lo = 0.0, hi = 0.0;
        bool found = false;
        const double vhi = v_pre * (1.0 - 1.0e-9);
        const double Fhi = residual(vhi, nullptr);
        const int NS = 400000;
        double vprev = vhi, Fprev = Fhi;
        for (int k = 1; k <= NS; ++k) {
            const double vk = v_pre * (1.0 - static_cast<double>(k) / (NS + 1));
            const double Fk = residual(vk, nullptr);
            if (Fprev * Fk < 0.0) { lo = vk; hi = vprev; found = true; break; }
            vprev = vk;
            Fprev = Fk;
        }
        State B;
        if (!found) {
            std::printf("case%s: NO Y-consistent root bracketed\n", cs.id);
            continue;
        }
        for (int k = 0; k < 200; ++k) {
            const double mid = 0.5 * (lo + hi);
            if (residual(mid, nullptr) * residual(hi, nullptr) < 0.0) lo = mid;
            else hi = mid;
        }
        residual(0.5 * (lo + hi), &B);

        // ---- how well does closure (A) satisfy the TRUE RH conditions? -------------------
        // (A) is built from an EQUIVALENT stiffened gas (gamma_mix, Pihat), not from the NASG
        // mixture EOS, so its own Rayleigh / Hugoniot residuals are a useful separate number.
        const double vA = 1.0 / A.rho;
        const double rayA = (A.p - pre.p) - mdot * mdot * (v_pre - vA);
        const double hA = h_mix_Y(A.p, A.T, A.Y, air, water);
        const double hugA = (hA - h_pre) - 0.5 * (A.p - pre.p) * (v_pre + vA);
        // mass-conservation defect of (A): Y is NOT preserved when alpha is frozen
        const double Ydef = (A.Y - pre.Y) / pre.Y;

        std::printf("\n================ case%s   alpha_air(pre) = %.2f ================\n",
                    cs.id, cs.alpha_air);
        std::printf("Ms = %.1f   a_mix(Eq.57) = %.4f m/s   Vs = %.6f m/s   (both closures)\n",
                    Ms, c_pre, Vs);
        std::printf("pre-shock : p=%.6e  u=%.4f  T=%.4f  rho=%.6f  alpha=%.6f  Y=%.6e\n",
                    pre.p, pre.u, pre.T, pre.rho, pre.alpha, pre.Y);
        std::printf("%-14s %14s %14s\n", "post-shock", "(A) alpha-held", "(B) Y-held");
        std::printf("%-14s %14.6e %14.6e\n", "p_post [Pa]", A.p, B.p);
        std::printf("%-14s %14.6f %14.6f\n", "rho_post", A.rho, B.rho);
        std::printf("%-14s %14.4f %14.4f\n", "u_post [m/s]", A.u, B.u);
        std::printf("%-14s %14.4f %14.4f\n", "T_post [K]", A.T, B.T);
        std::printf("%-14s %14.6e %14.6e\n", "alpha_post", A.alpha, B.alpha);
        std::printf("%-14s %14.6e %14.6e\n", "Y_post", A.Y, B.Y);
        std::printf("closure (A) diagnostics vs the TRUE NASG mixture RH:\n");
        std::printf("   Rayleigh residual  = %+.4e Pa   (rel %.3e)\n", rayA, rayA / A.p);
        std::printf("   Hugoniot residual  = %+.4e J/kg (rel %.3e)\n", hugA, hugA / std::abs(hA));
        std::printf("   mass-fraction drift (Y_post-Y_pre)/Y_pre = %+.4e\n", Ydef);
        std::printf("ratios (B)/(A):  p %.4f   rho %.4f   u %.4f   alpha %.4e\n",
                    B.p / A.p, B.rho / A.rho, B.u / A.u, B.alpha / A.alpha);
    }
    return 0;
}
