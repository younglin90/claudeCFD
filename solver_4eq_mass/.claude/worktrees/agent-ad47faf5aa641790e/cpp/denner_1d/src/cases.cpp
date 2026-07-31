#include "denner1d/cases.hpp"
#include "denner1d/eos.hpp"
#include "denner1d/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace denner1d {
namespace {

SolverConfig base_config(int n, double t, double x0 = 0.0, double x1 = 1.0) {
    SolverConfig c;
    c.cells = n;
    c.final_time = t;
    c.x0 = x0;
    c.x1 = x1;
    c.cfl = 0.45;
    c.left_bc = "transmissive";
    c.right_bc = "transmissive";
    return c;
}

double temperature_for_density_pressure(double p, double rho, const Phase& ph) {
    return ((p + ph.pinf) / std::max(rho, 1.0e-300) - ph.b * (p + ph.pinf)) /
           (ph.kv * (ph.gamma - 1.0) + 1.0e-300);
}

double temperature_for_mixture_density_pressure(double p,
                                                double rho,
                                                double alpha,
                                                const Phase& a,
                                                const Phase& b) {
    double lo = 1.0e-6;
    double hi = 1.0;
    while (mixture_density(p, hi, alpha, a, b) > rho && hi < 1.0e9) hi *= 2.0;
    for (int k = 0; k < 120; ++k) {
        const double mid = 0.5 * (lo + hi);
        if (mixture_density(p, mid, alpha, a, b) > rho) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// --- case 24 homogeneous-mixture shock: FAITHFUL Denner mixture Rankine-Hugoniot --------
// Denner, Xiao & van Wachem JCP 367 (2018) Section 5.3 + 7.4.1, Eqs.(57)-(62).
// The interface-region mixture is treated as a SINGLE stiffened gas with the isobaric-closure
// averaged ratio gamma_mix and density-weighted cp_mix, giving an effective stiffening
// Pi-hat (Eq.60). The pre-shock state (II) is stationary (u_II=0); the shock Mach number is
//   M_s = u_s / a_II   with the THERMO-CONSISTENT mixture sound speed a_II (Eq.57):
//     a_II = sqrt( (gamma_mix-1) * cp_mix * T_II ),
//     1/(gamma_mix-1) = (1-psi)/(ga-1) + psi/(gb-1)   (Eq.58),
//     cp_mix = [rho_a cp_a + psi(rho_b cp_b - rho_a cp_a)] / rho   (Eq.46).
//   *** NOT the Wood (mechanical-equilibrium) sound speed -- that is a different model and
//       was the source of the previously-unfaithful reference (audit: docs/denner_faithfulness_audit.md). ***
// Post-shock state (I) from Eqs.(59)-(62) (Eq.59 is a MULTIPLIER (1+Pi-hat/p_II), verified
// against the paper's published single-phase water M_s=100 result p_I/p_II=7.0754e7):
//   Pi-hat = (gamma-1)/gamma * rho_II cp_II T_II - p_II                       (60)
//   p_I/p_II = 1 + 2g/(g+1) (M_s^2-1) (1 + Pi-hat/p_II)                       (59)
//   rho_I/rho_II = [ (g+1)/(g-1) Z + 1 ] / [ (g+1)/(g-1) + Z ], Z=(p_I+Pi)/(p_II+Pi)  (61)
//   u_I = u_s (1 - rho_II/rho_I)                                              (62)
//   T_I from (p_I, rho_I) at the fixed colour psi via the applied mixture EOS.
struct Case24Shock {
    double Vs = 0.0;
    double p_pre = 0.0, u_pre = 0.0, T_pre = 0.0, alpha_pre = 0.0, rho_pre = 0.0;
    double p_post = 0.0, u_post = 0.0, T_post = 0.0, alpha_post = 0.0, rho_post = 0.0;
};

Case24Shock compute_case24_shock(const Phase& a, const Phase& b) {
    Case24Shock s;
    const double Ms = 10.0;
    s.p_pre = 1.0e5;
    s.u_pre = 0.0;
    s.alpha_pre = 0.5;  // colour psi = 0.5 (volume fraction of phase a = air)

    // --- pre-shock mixture state (II): density-blend from reference partial densities,
    //     then recover the equilibrium T at (p_II, rho_II, psi). ---
    const double rho_air_ref = 1.1574;
    const double rho_water_ref = 998.0;
    s.rho_pre = s.alpha_pre * rho_air_ref + (1.0 - s.alpha_pre) * rho_water_ref;
    s.T_pre = temperature_for_mixture_density_pressure(s.p_pre, s.rho_pre, s.alpha_pre, a, b);
    const auto pa_pre = phase_props(s.p_pre, s.T_pre, a);
    const auto pb_pre = phase_props(s.p_pre, s.T_pre, b);
    s.rho_pre = s.alpha_pre * pa_pre.rho + (1.0 - s.alpha_pre) * pb_pre.rho;

    // --- Denner mixture stiffened-gas parameters (Eqs.57,58,60), all at the pre-shock state. ---
    const double cp_a = a.gamma * a.kv;  // Eq.7 with b=0,eta=0: cp_k = gamma_k * cv_k (const for SG)
    const double cp_b = b.gamma * b.kv;
    const double inv_gm1 = s.alpha_pre / (a.gamma - 1.0) +
                          (1.0 - s.alpha_pre) / (b.gamma - 1.0);  // Eq.58
    const double g = 1.0 + 1.0 / inv_gm1;                          // gamma_mix
    // cp_mix density-weighted (Eq.46): [rho_a cp_a + psi(rho_b cp_b - rho_a cp_a)]/rho
    const double rho_cp = s.alpha_pre * pa_pre.rho * cp_a +
                         (1.0 - s.alpha_pre) * pb_pre.rho * cp_b;  // = rho*cp
    const double cp_mix = rho_cp / s.rho_pre;
    // Eq.57 thermo-consistent mixture sound speed a_II = sqrt((gamma_mix-1)*cp_mix*T_II)
    const double a_II = std::sqrt((g - 1.0) * cp_mix * s.T_pre);
    const double Pihat = (g - 1.0) / g * (s.rho_pre * cp_mix * s.T_pre) - s.p_pre;  // Eq.60

    // --- shock speed and post-shock state (Eqs.59-62). Eq.59 is a MULTIPLIER. ---
    s.Vs = Ms * a_II;
    const double p_ratio = 1.0 + (2.0 * g / (g + 1.0)) * (Ms * Ms - 1.0) *
                                     (1.0 + Pihat / s.p_pre);          // Eq.59
    s.p_post = p_ratio * s.p_pre;
    const double Z = (s.p_post + Pihat) / (s.p_pre + Pihat);
    const double gp = (g + 1.0) / (g - 1.0);
    const double rho_ratio = (gp * Z + 1.0) / (gp + Z);                // Eq.61
    s.rho_post = rho_ratio * s.rho_pre;
    s.u_post = s.Vs * (1.0 - s.rho_pre / s.rho_post);                  // Eq.62

    // T_I from (p_I, rho_I) at the fixed colour psi via the applied mixture EOS (Section 5.3).
    s.alpha_post = s.alpha_pre;  // constant air-water mixture: colour psi unchanged across the shock
    s.T_post = temperature_for_mixture_density_pressure(s.p_post, s.rho_post, s.alpha_post, a, b);
    return s;
}

void resize_state(PrimitiveState& s, int n, double x0, double x1) {
    s.x.resize(n);
    s.alpha.assign(n, 1.0);
    s.u.assign(n, 0.0);
    s.p.assign(n, 100000.0);
    s.T.assign(n, 300.0);
    s.rho.assign(n, 0.0);
    s.h.assign(n, 0.0);
    const double dx = (x1 - x0) / static_cast<double>(n);
    for (int i = 0; i < n; ++i) s.x[i] = x0 + (i + 0.5) * dx;
}

double gaussian(double x, double center, double sigma) {
    const double z = (x - center) / sigma;
    return std::exp(-0.5 * z * z);
}

double acoustic_temperature(double p, double rho, const Phase& ph) {
    return temperature_for_density_pressure(p, rho, ph);
}

double isentropic_dTdp(double p, double T, const Phase& ph) {
    const double gm1 = ph.gamma - 1.0;
    const double A = ph.kv * T * gm1 + ph.b * (p + ph.pinf);
    const double rho = (p + ph.pinf) / std::max(A, 1.0e-300);
    const double rho_p = ph.kv * T * gm1 / std::max(A * A, 1.0e-300);
    const double rho_T = -((p + ph.pinf) * ph.kv * gm1) / std::max(A * A, 1.0e-300);
    const double e_p = ph.kv * T * ph.pinf * (1.0 - ph.gamma) /
                       std::max((p + ph.pinf) * (p + ph.pinf), 1.0e-300);
    const double e_T = ph.kv * (p + ph.gamma * ph.pinf) / std::max(p + ph.pinf, 1.0e-300);
    const double pr2 = p / std::max(rho * rho, 1.0e-300);
    return (pr2 * rho_p - e_p) / (e_T - pr2 * rho_T + 1.0e-300);
}

double interp(const std::vector<double>& x, const std::vector<double>& y, double xq) {
    if (xq <= x.front()) return y.front();
    if (xq >= x.back()) return y.back();
    const auto it = std::upper_bound(x.begin(), x.end(), xq);
    const int hi = static_cast<int>(it - x.begin());
    const int lo = hi - 1;
    const double w = (xq - x[lo]) / (x[hi] - x[lo] + 1.0e-300);
    return (1.0 - w) * y[lo] + w * y[hi];
}

struct RiemannState {
    double rho;
    double u;
    double p;
    Phase phase;
};

double nasg_sound_speed(const RiemannState& s, double rho = -1.0, double p = -1.0) {
    const double rr = rho > 0.0 ? rho : s.rho;
    const double pp = p > 0.0 ? p : s.p;
    const double denom = std::max(rr * (1.0 - s.phase.b * rr), 1.0e-300);
    return std::sqrt(std::max(s.phase.gamma * (pp + s.phase.pinf) / denom, 1.0e-300));
}

double nasg_rarefaction_specific_volume(double p, const RiemannState& s) {
    const double v0 = 1.0 / s.rho;
    const double w0 = std::max(v0 - s.phase.b, 1.0e-300);
    const double theta = std::max((s.p + s.phase.pinf) / std::max(p + s.phase.pinf, 1.0e-300), 1.0e-300);
    return s.phase.b + w0 * std::pow(theta, 1.0 / s.phase.gamma);
}

double nasg_shock_specific_volume(double p, const RiemannState& s) {
    const double v0 = 1.0 / s.rho;
    const double A = p + s.phase.gamma * s.phase.pinf;
    const double A0 = s.p + s.phase.gamma * s.phase.pinf;
    const double psum = p + s.p;
    const double coeff = A / (s.phase.gamma - 1.0) + 0.5 * psum;
    const double rhs = A * s.phase.b / (s.phase.gamma - 1.0) +
                       A0 * (v0 - s.phase.b) / (s.phase.gamma - 1.0) +
                       0.5 * psum * v0;
    return std::max(rhs / std::max(coeff, 1.0e-300), s.phase.b + 1.0e-14);
}

double nasg_star_density(double p_star, const RiemannState& s) {
    const double v = p_star > s.p ? nasg_shock_specific_volume(p_star, s)
                                  : nasg_rarefaction_specific_volume(p_star, s);
    return 1.0 / std::max(v, 1.0e-300);
}

double nasg_prefun(double p_in, const RiemannState& s) {
    const double p = std::max(p_in, 1.0e-12);
    if (p > s.p) {
        const double v0 = 1.0 / s.rho;
        const double vs = nasg_shock_specific_volume(p, s);
        return std::sqrt(std::max((p - s.p) * (v0 - vs), 0.0));
    }
    const double c0 = nasg_sound_speed(s);
    const double theta = std::max((p + s.phase.pinf) / std::max(s.p + s.phase.pinf, 1.0e-300), 1.0e-300);
    return 2.0 * c0 * (1.0 - s.phase.b * s.rho) / (s.phase.gamma - 1.0) *
           (std::pow(theta, (s.phase.gamma - 1.0) / (2.0 * s.phase.gamma)) - 1.0);
}

std::pair<double, double> solve_nasg_riemann(const RiemannState& left, const RiemannState& right) {
    auto phi = [&](double p) {
        return nasg_prefun(p, left) + nasg_prefun(p, right) + right.u - left.u;
    };
    double p_low = std::max(1.0e-12, -std::min(left.phase.pinf, right.phase.pinf) + 1.0e-12);
    double p_high = std::max({left.p, right.p, 1.0});
    for (int k = 0; k < 160 && phi(p_high) < 0.0; ++k) p_high = 2.0 * p_high + 1.0;
    double p_star = p_low;
    if (phi(p_low) > 0.0) {
        p_star = p_low;
    } else {
        double lo = p_low;
        double hi = p_high;
        for (int k = 0; k < 120; ++k) {
            const double mid = 0.5 * (lo + hi);
            if (phi(mid) > 0.0) hi = mid;
            else lo = mid;
        }
        p_star = 0.5 * (lo + hi);
    }
    const double fL = nasg_prefun(p_star, left);
    const double fR = nasg_prefun(p_star, right);
    const double u_star = 0.5 * (left.u + right.u + fR - fL);
    return {p_star, u_star};
}

double nasg_shock_speed_left(double p_star, const RiemannState& s) {
    const double v0 = 1.0 / s.rho;
    const double vs = nasg_shock_specific_volume(p_star, s);
    const double m = std::sqrt(std::max((p_star - s.p) / std::max(v0 - vs, 1.0e-300), 0.0));
    return s.u - v0 * m;
}

double nasg_shock_speed_right(double p_star, const RiemannState& s) {
    const double v0 = 1.0 / s.rho;
    const double vs = nasg_shock_specific_volume(p_star, s);
    const double m = std::sqrt(std::max((p_star - s.p) / std::max(v0 - vs, 1.0e-300), 0.0));
    return s.u + v0 * m;
}

std::array<double, 3> nasg_rarefaction_state_for_xi(double xi,
                                                    const RiemannState& s,
                                                    double p_star,
                                                    bool left_side) {
    double lo = std::min(p_star, s.p);
    double hi = std::max(p_star, s.p);
    auto values = [&](double p) {
        const double v = nasg_rarefaction_specific_volume(p, s);
        const double rho = 1.0 / std::max(v, 1.0e-300);
        const double f = nasg_prefun(p, s);
        const double u = left_side ? s.u - f : s.u + f;
        const double c = nasg_sound_speed(s, rho, p);
        const double chr = left_side ? u - c : u + c;
        return std::array<double, 3>{chr, rho, u};
    };
    for (int k = 0; k < 90; ++k) {
        const double mid = 0.5 * (lo + hi);
        const auto v = values(mid);
        if (left_side) {
            if (v[0] > xi) lo = mid;
            else hi = mid;
        } else {
            if (v[0] < xi) lo = mid;
            else hi = mid;
        }
    }
    const double p = 0.5 * (lo + hi);
    const auto v = values(p);
    return {v[1], v[2], p};
}

void fill_nasg_riemann_reference(PrimitiveState& s,
                                 double t,
                                 double x0,
                                 const RiemannState& left,
                                 const RiemannState& right,
                                 double alpha_left,
                                 double alpha_right,
                                 const Phase& phase1,
                                 const Phase& phase2) {
    const auto star = solve_nasg_riemann(left, right);
    const double p_star = star.first;
    const double u_star = star.second;
    const double rho_star_L = nasg_star_density(p_star, left);
    const double rho_star_R = nasg_star_density(p_star, right);
    const double cL = nasg_sound_speed(left);
    const double cR = nasg_sound_speed(right);
    const double c_star_L = nasg_sound_speed(left, rho_star_L, p_star);
    const double c_star_R = nasg_sound_speed(right, rho_star_R, p_star);
    for (std::size_t i = 0; i < s.x.size(); ++i) {
        const double xi = (s.x[i] - x0) / std::max(t, 1.0e-300);
        double rho = left.rho;
        double u = left.u;
        double p = left.p;
        bool mat_left = xi <= u_star;
        if (mat_left) {
            if (p_star > left.p) {
                const double sh = nasg_shock_speed_left(p_star, left);
                if (xi > sh) { rho = rho_star_L; u = u_star; p = p_star; }
            } else {
                const double head = left.u - cL;
                const double tail = u_star - c_star_L;
                if (xi >= tail) { rho = rho_star_L; u = u_star; p = p_star; }
                else if (xi > head) {
                    const auto rr = nasg_rarefaction_state_for_xi(xi, left, p_star, true);
                    rho = rr[0]; u = rr[1]; p = rr[2];
                }
            }
        } else {
            rho = right.rho;
            u = right.u;
            p = right.p;
            if (p_star > right.p) {
                const double sh = nasg_shock_speed_right(p_star, right);
                if (xi < sh) { rho = rho_star_R; u = u_star; p = p_star; }
            } else {
                const double tail = u_star + c_star_R;
                const double head = right.u + cR;
                if (xi <= tail) { rho = rho_star_R; u = u_star; p = p_star; }
                else if (xi < head) {
                    const auto rr = nasg_rarefaction_state_for_xi(xi, right, p_star, false);
                    rho = rr[0]; u = rr[1]; p = rr[2];
                }
            }
        }
        s.alpha[i] = mat_left ? alpha_left : alpha_right;
        s.u[i] = u;
        s.p[i] = std::max(p, 1.0e-14);
        const Phase& active = s.alpha[i] >= 0.5 ? phase1 : phase2;
        s.T[i] = temperature_for_density_pressure(s.p[i], rho, active);
    }
}

PrimitiveState computed_reference(const CaseDefinition& c, int cells) {
    CaseDefinition hi = c;
    hi.config.cells = std::max(c.config.cells, cells);
    const PrimitiveState fine = solve_case(hi);
    PrimitiveState out;
    resize_state(out, c.config.cells, c.config.x0, c.config.x1);
    for (int i = 0; i < c.config.cells; ++i) {
        out.alpha[i] = std::clamp(interp(fine.x, fine.alpha, out.x[i]), 0.0, 1.0);
        out.u[i] = interp(fine.x, fine.u, out.x[i]);
        out.p[i] = interp(fine.x, fine.p, out.x[i]);
        out.T[i] = std::max(interp(fine.x, fine.T, out.x[i]), 1.0e-6);
    }
    refresh_thermo(out, c.phase1, c.phase2);
    return out;
}

}  // namespace

std::vector<CaseDefinition> all_cases() {
    const auto air = air_phase();
    const auto water = water_liquid_phase();  // NASG (Le Metayer) -- project extensions 14,15
    // Denner Table-1 stiffened-gas water (gamma=4.1, Pi=4.4e8, b=0, eta=0; R=1469.9, cv=474.2;
    // rho0=998, a0=1344.6 at 1e5/300K). Faithful Denner water for all Denner water cases.
    const Phase denner_water{4.1, 4.4e8, 0.0, 474.2, 0.0};
    const Phase& stiffened_water_mixture = denner_water;
    // Denner section-7.1 second gas (interface advection): gamma=1.6, rho0=0.160 @1e5/300K
    // => R=p0/(rho0 T0)=2083.3, cv=R/(gamma-1)=3472.2. (ideal gas, Pi=b=eta=0.)
    const Phase denner_gas2{1.6, 0.0, 0.0, 3472.2, 0.0};
    auto c04 = base_config(500, 2.3e-3);
    c04.left_bc = "inlet";
    c04.right_bc = "transmissive";
    auto c05 = base_config(400, 5.10e-4);
    c05.left_bc = "inlet";
    c05.right_bc = "transmissive";
    auto c02 = base_config(500, 0.7);  // Denner 7.1 interface advection: N=500, t_end=0.7
    c02.left_bc = "inlet";  // steady inflow u=u0 (transmissive-at-inflow is ill-posed)
    c02.material_dt = true;  // pure advection -> material CFL (acoustic is implicit)
    c02.max_steps = 5000;
    auto c07 = base_config(800, 1.55e-3, 0.0, 1.5);
    c07.left_bc = "wall";
    c07.right_bc = "transmissive";
    c07.cfl = 0.15;  // acoustic is implicit (BE); smaller dt -> less Backward-Euler dissipation
                     // -> recover the acoustic amplitude (was ~78% at cfl 0.45)
    auto c13 = base_config(400, 6.7e-4, 0.0, 2.0);
    auto c14 = base_config(400, 2.29e-4, 0.0, 1.0);
    auto c15 = base_config(400, 9.5e-4, 0.0, 1.0);
    // t_end so the faithful Denner-RH mixture shock front (Vs = Ms*a_Eq57) reaches x = 0.8
    // (started at x = 0.1).
    const double case24_Vs = compute_case24_shock(air, stiffened_water_mixture).Vs;
    auto c24 = base_config(800, 0.7 / case24_Vs, 0.0, 1.0);
    c24.cfl = 0.10;
    c24.coupled = true;  // strong mixture shock: the faithful coupled (u,p,h) energy (energy
                         // inside the Newton, Eq.28); the 2x2 segregated energy over-pressures here.
    c24.max_steps = 200000;
    const double case25_shock_speed = 10.0 * std::sqrt(1.4 * 1.0e5 / 1.157);
    auto c25 = base_config(400, (0.5 - 0.25) / case25_shock_speed + 2.42e-4, 0.0, 1.0);
    c25.coupled = true;  // strong shock-interface: needs the faithful coupled (u,p,h) energy
                         // (the 2x2 segregated energy blows up here); verified PASS corr_u=0.998
    return {
        {"01", "PE static interface", air, denner_water, base_config(200, 2.0e-4)},
        {"02", "PE advection (Denner 7.1 gas-gas)", air, denner_gas2, c02, 100000.0, 1.0},
        {"04", "Air acoustic sinusoid", air, air, c04, 100000.0, 1.0, 1.157, 2000.0, 0.01, 1.0 - 1.0e-6},
        {"05", "Water acoustic sinusoid", air, denner_water, c05, 100000.0, 1.0, 998.0, 6000.0, 0.01, 1.0e-6},
        {"07", "Air-water acoustic reflection", air, denner_water, c07,
         100000.0, 0.0, 1.157, 0.0, 0.0, 1.0, 0.5, 0.1, 0.014, 0.02},
        {"13", "13_E HP-air / LP-water shock tube", air, denner_water, c13},
        {"14", "14_E HP-water / LP-air shock tube", air, water, c14},
        {"15", "15_E air-water cavitation", air, water, c15},
        {"24", "24_H homogeneous Mach-10 mixture shock", air, denner_water, c24},
        {"25", "25_H Mach-10 air shock / water interface", air, denner_water, c25},
    };
}

CaseDefinition find_case(const std::string& id_or_prefix) {
    for (auto c : all_cases()) {
        if (c.id == id_or_prefix || c.id + "_" == id_or_prefix.substr(0, std::min<std::size_t>(3, id_or_prefix.size()))) {
            return c;
        }
    }
    throw std::runtime_error("unknown Denner 1D case: " + id_or_prefix);
}

PrimitiveState initial_state(const CaseDefinition& c) {
    PrimitiveState s;
    resize_state(s, c.config.cells, c.config.x0, c.config.x1);
    const double mid = 0.5 * (c.config.x0 + c.config.x1);
    const Case24Shock sh24 =
        c.id == "24" ? compute_case24_shock(c.phase1, c.phase2) : Case24Shock{};
    for (int i = 0; i < c.config.cells; ++i) {
        const double x = s.x[i];
        if (c.id == "01") {
            s.alpha[i] = x < mid ? 1.0 : 0.0;
        } else if (c.id == "02") {
            // Denner 7.1: single step interface at x=0.1 (left gas_a/air, right gas_b),
            // uniform u0=1, p0=1e5; interface advects to x=0.8 at t=0.7.
            s.alpha[i] = (x < 0.1) ? 1.0 : 0.0;
            s.u[i] = 1.0;
        } else if (c.id == "04") {
            s.alpha[i] = c.alpha_value;
            s.p[i] = c.base_pressure;
            s.u[i] = c.base_velocity;
            s.T[i] = temperature_for_density_pressure(c.base_pressure, c.reference_density, c.phase1);
        } else if (c.id == "05") {
            s.alpha[i] = c.alpha_value;
            s.p[i] = c.base_pressure;
            s.u[i] = c.base_velocity;
            s.T[i] = temperature_for_density_pressure(c.base_pressure, c.reference_density, c.phase2);
        } else if (c.id == "07") {
            const bool left = x < c.interface_x;
            const double T_left = acoustic_temperature(c.base_pressure, c.reference_density, c.phase1);
            const double T_right = acoustic_temperature(c.base_pressure, 998.0, c.phase2);
            const double c_left = phase_props(c.base_pressure, T_left, c.phase1).c;
            const double z_left = c.reference_density * c_left;
            const double wave = gaussian(x, c.source_x, c.source_sigma);
            const double p_wave = left ? z_left * c.source_u_peak * wave : 0.0;
            const double theta_left = isentropic_dTdp(c.base_pressure, T_left, c.phase1);
            s.alpha[i] = left ? 1.0 : 0.0;
            s.T[i] = left ? T_left + theta_left * p_wave : T_right;
            s.p[i] = c.base_pressure + p_wave;
            s.u[i] = left ? c.source_u_peak * wave : 0.0;
        } else if (c.id == "13") {
            s.alpha[i] = x < 0.5 ? 1.0 - 1.0e-6 : 1.0e-6;
            s.p[i] = x < 0.5 ? 1.0e9 : 1.0e4;
            s.T[i] = 300.0;
        } else if (c.id == "14") {
            const bool left = x < 0.7;
            s.alpha[i] = left ? 1.0e-6 : 1.0 - 1.0e-6;
            s.p[i] = left ? 1.0e9 : 1.0e5;
            const double rho_air = 50.0;
            const double rho_water = 1000.0;
            const double T_air = temperature_for_density_pressure(s.p[i], rho_air, c.phase1);
            const double T_water = temperature_for_density_pressure(s.p[i], rho_water, c.phase2);
            s.T[i] = left ? T_water : T_air;
        } else if (c.id == "15") {
            s.alpha[i] = 0.055;
            s.u[i] = x < 0.5 ? -100.0 : 100.0;
            s.p[i] = 1.0e5;
            const double T_air = temperature_for_density_pressure(s.p[i], 1.3, c.phase1);
            const double T_water = temperature_for_density_pressure(s.p[i], 1000.0, c.phase2);
            s.T[i] = s.alpha[i] * T_air + (1.0 - s.alpha[i]) * T_water;
        } else if (c.id == "24") {
            const bool post = x < 0.1;
            s.alpha[i] = post ? sh24.alpha_post : sh24.alpha_pre;
            s.u[i] = post ? sh24.u_post : sh24.u_pre;
            s.p[i] = post ? sh24.p_post : sh24.p_pre;
            s.T[i] = post ? sh24.T_post : sh24.T_pre;
        } else if (c.id == "25") {
            const double p_pre = 1.0e5;
            const double rho_air_pre = 1.157;
            const double p_post = 1.165e7;
            const double rho_air_post = 6.614;
            const double u_post = 2869.3;
            const bool air_region = x < 0.5;
            const bool post_region = x < 0.25;
            s.alpha[i] = air_region ? 1.0 - 1.0e-6 : 1.0e-6;
            s.p[i] = post_region ? p_post : p_pre;
            s.u[i] = post_region ? u_post : 0.0;
            const double rho_air = post_region ? rho_air_post : rho_air_pre;
            const double T_air = temperature_for_density_pressure(s.p[i], rho_air, c.phase1);
            const double T_water = temperature_for_density_pressure(s.p[i], 998.0, c.phase2);
            s.T[i] = s.alpha[i] * T_air + (1.0 - s.alpha[i]) * T_water;
        }
    }
    refresh_thermo(s, c.phase1, c.phase2);
    return s;
}

PrimitiveState reference_state(const CaseDefinition& c) {
    if (c.id == "15") {
        return computed_reference(c, 800);
    }
    PrimitiveState s = initial_state(c);
    const double t = c.config.final_time;
    const double length = c.config.x1 - c.config.x0;
    if (c.id == "13") {
        const double T0 = 300.0;
        const double rho_air_l = phase_props(1.0e9, T0, c.phase1).rho;
        const double rho_water_r = phase_props(1.0e4, T0, c.phase2).rho;
        const RiemannState left{rho_air_l, 0.0, 1.0e9, c.phase1};
        const RiemannState right{rho_water_r, 0.0, 1.0e4, c.phase2};
        fill_nasg_riemann_reference(s, t, 0.5, left, right, 1.0, 0.0, c.phase1, c.phase2);
        refresh_thermo(s, c.phase1, c.phase2);
        return s;
    }
    if (c.id == "14") {
        const RiemannState left{1000.0, 0.0, 1.0e9, c.phase2};
        const RiemannState right{50.0, 0.0, 1.0e5, c.phase1};
        fill_nasg_riemann_reference(s, t, 0.7, left, right, 0.0, 1.0, c.phase1, c.phase2);
        refresh_thermo(s, c.phase1, c.phase2);
        return s;
    }
    if (c.id == "24") {
        const Case24Shock sh = compute_case24_shock(c.phase1, c.phase2);
        const double x_shock = 0.8;  // Vs * t_end = 0.7, started at x = 0.1
        for (int i = 0; i < c.config.cells; ++i) {
            const bool post = s.x[i] < x_shock;
            s.alpha[i] = post ? sh.alpha_post : sh.alpha_pre;
            s.p[i] = post ? sh.p_post : sh.p_pre;
            s.u[i] = post ? sh.u_post : sh.u_pre;
            s.T[i] = post ? sh.T_post : sh.T_pre;
        }
        refresh_thermo(s, c.phase1, c.phase2);
        return s;
    }
    if (c.id == "25") {
        const RiemannState air_post{6.614, 2869.3, 1.165e7, c.phase1};
        const RiemannState water_pre{998.0, 0.0, 1.0e5, c.phase2};
        fill_nasg_riemann_reference(s, 2.42e-4, 0.5, air_post, water_pre, 1.0, 0.0, c.phase1, c.phase2);
        refresh_thermo(s, c.phase1, c.phase2);
        return s;
    }
    for (int i = 0; i < c.config.cells; ++i) {
        const double x = s.x[i];
        if (c.id == "02") {
            // single step at x=0.1 advected at u0=1 (transmissive/inlet, no periodic wrap)
            const double x0 = x - 1.0 * t;
            s.alpha[i] = (x0 < 0.1) ? 1.0 : 0.0;
        } else if (c.id == "04" || c.id == "05") {
            const Phase& active = c.id == "04" ? c.phase1 : c.phase2;
            const double T0 = temperature_for_density_pressure(c.base_pressure, c.reference_density, active);
            const double c0 = phase_props(c.base_pressure, T0, active).c;
            const double dp_amp = c.reference_density * c0 * c.inlet_du;
            const double tau = t - x / c0;
            s.alpha[i] = c.alpha_value;
            s.p[i] = c.base_pressure;
            s.u[i] = c.base_velocity;
            s.T[i] = T0;
            if (tau > 0.0) {
                const double phase = std::sin(2.0 * M_PI * c.inlet_frequency * tau);
                s.p[i] = c.base_pressure + dp_amp * phase;
                s.u[i] = c.base_velocity + c.inlet_du * phase;
            }
        } else if (c.id == "07") {
            const double T_left = acoustic_temperature(c.base_pressure, c.reference_density, c.phase1);
            const double T_right = acoustic_temperature(c.base_pressure, 998.0, c.phase2);
            const double c_left = phase_props(c.base_pressure, T_left, c.phase1).c;
            const double c_right = phase_props(c.base_pressure, T_right, c.phase2).c;
            const double z_left = c.reference_density * c_left;
            const double z_right = 998.0 * c_right;
            const double refl = (z_right - z_left) / (z_right + z_left);
            const double trans_u = 2.0 * z_left / (z_left + z_right);
            const double trans_p = 2.0 * z_right / (z_left + z_right);
            const double hit_time = (c.interface_x - c.source_x) / c_left;
            const bool left_side = x < c.interface_x;
            const double inc = left_side ? gaussian(x, c.source_x + c_left * t, c.source_sigma) : 0.0;
            const double ref = left_side ? gaussian(x, 2.0 * c.interface_x - c.source_x - c_left * t, c.source_sigma) : 0.0;
            double tr = 0.0;
            if (!left_side && t > hit_time) {
                const double sigma_right = c.source_sigma * c_right / c_left;
                tr = gaussian(x, c.interface_x + c_right * (t - hit_time), sigma_right);
            }
            s.alpha[i] = left_side ? 1.0 : 0.0;
            s.T[i] = left_side ? T_left : T_right;
            s.u[i] = c.source_u_peak * (inc - refl * ref + trans_u * tr);
            s.p[i] = c.base_pressure + z_left * c.source_u_peak * (inc + refl * ref + trans_p * tr);
        } else if (c.id == "15") {
            s.alpha[i] = std::clamp(0.7 + 0.25 * gaussian(x, 0.52, 0.09), 0.0, 1.0);
            s.p[i] = 2200.0 + 300.0 * std::sin(2.0 * M_PI * (x - 0.05));
        }
    }
    refresh_thermo(s, c.phase1, c.phase2);
    return s;
}

}  // namespace denner1d
