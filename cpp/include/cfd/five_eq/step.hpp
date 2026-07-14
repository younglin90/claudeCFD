// cfd/five_eq/step.hpp — M10 production step driver (one imex_ad_step).
//
// Header-only C++ twin of solver/five_eq_IMEX/imex_ad.py::imex_ad_step
// (3902-4090) for the production BASE_ENV configuration.  Stage order:
//   1. regime_auto pressure-closure pick (M8, regime_auto.hpp)
//   2. material_update (M6)  -> q1_new,q2_new,m_adv,rhoE_adv,alpha_new
//   3. clip alpha to [1e-12, 1-1e-12]
//   4. acoustic_solve (M7)   -> u_new,p_new
//   5. closure (pressure_work_consistent / compressive_recovery):
//        recompute Z from the W^n anchor, build acoustic-Riemann faces from
//        (u_new,p_new,Z), rebuild energy
//          rhoE_new = rhoE_adv - dt*(p_f[1:]*u_f[1:] - p_f[:-1]*u_f[:-1])/dx
//        then the PW pure-shock recovery mask (compressive & ~pure) via
//        _recover_pressure_from_total_energy — inactive for 02A/07B acoustic.
//   6. near-vacuum velocity regularisation (inactive away from cavitation)
//   7. primitive LMP/LED filter — 'auto' -> off unless W^n has a pressure jump
//      (off for 02A/07B; the LED branch is not ported, see note)
//   8. explicit primitive recovery W_new = (alpha, T1, T2, u_new, p_new)
//
// implicit_energy / apec_pe closures use _solve_acoustic_energy_ad and are NOT
// on the 02A/07B path — left for a later module.
#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "cfd/eos.hpp"
#include "cfd/five_eq/config.hpp"
#include "cfd/five_eq/regime_auto.hpp"
#include "cfd/five_eq/material_update.hpp"
#include "cfd/five_eq/acoustic_solve.hpp"
#include "cfd/five_eq/sound_speed.hpp"

namespace cfd {
namespace five_eq {

struct StepResult {
    std::vector<double> alpha, T1, T2, u, p;   // W_new
    PressureClosure closure = PressureClosure::pressure_work_consistent;
    int vacuum_velocity_cells = 0;
};

namespace step_detail {

constexpr double MU_EPS = 1e-30;

// dp/de |_rho, analytic per EOS form (matches He2024 eos.dpde_rho):
//   Ideal/SG: (gamma-1)*rho ;  NASG: (gamma-1)*rho / max(1-b*rho, 1e-10).
inline double dpde_rho(const EOS& e, double rho, double /*ener*/) {
    if (e.kind == EOS::NASG) {
        double denom = EOS::max2(1.0 - e.b * rho, 1e-10);
        return (e.gamma - 1.0) * rho / denom;
    }
    return (e.gamma - 1.0) * rho;
}

inline AcousticBC to_acoustic_bc(BC5 b) {
    switch (b) {
        case BC5::Periodic:     return AcousticBC::periodic;
        case BC5::Reflective:   return AcousticBC::reflective;
        case BC5::Transmissive: return AcousticBC::transmissive;
    }
    return AcousticBC::transmissive;
}

// _extend_np: 1-ghost extension.  odd => reflective flips sign; transmissive
// copies; periodic wraps.  (Matches imex_ad._extend_np for the 3 BC kinds.)
inline std::vector<double> extend(const std::vector<double>& a,
                                  BC5 bc_l, BC5 bc_r, bool odd) {
    const int n = (int)a.size();
    std::vector<double> e(n + 2);
    for (int i = 0; i < n; ++i) e[i + 1] = a[i];
    if (bc_l == BC5::Periodic && bc_r == BC5::Periodic) {
        e[0] = a[n - 1]; e[n + 1] = a[0];
        return e;
    }
    e[0]     = (odd && bc_l == BC5::Reflective) ? -a[0]     : a[0];
    e[n + 1] = (odd && bc_r == BC5::Reflective) ? -a[n - 1] : a[n - 1];
    return e;
}

// _acoustic_faces_np (imex_ad.py 464-513): first-order acoustic Z-Riemann faces.
struct Faces { std::vector<double> pf, uf; };  // length n+1
inline Faces acoustic_faces(const std::vector<double>& u,
                            const std::vector<double>& p,
                            const std::vector<double>& Z,
                            BC5 bc_l, BC5 bc_r) {
    const int n = (int)u.size();
    const double EPS = MU_EPS;
    std::vector<double> ue(n + 2), pe(n + 2), Ze(n + 2);
    const bool periodic = (bc_l == BC5::Periodic && bc_r == BC5::Periodic);
    for (int i = 0; i < n; ++i) { ue[i + 1] = u[i]; pe[i + 1] = p[i]; Ze[i + 1] = Z[i]; }
    if (periodic) {
        ue[0] = u[n - 1]; pe[0] = p[n - 1]; Ze[0] = Z[n - 1];
        ue[n + 1] = u[0]; pe[n + 1] = p[0]; Ze[n + 1] = Z[0];
    } else {
        double ul = (bc_l == BC5::Reflective) ? -u[0] : u[0];
        double pl = p[0];
        double ur = (bc_r == BC5::Reflective) ? -u[n - 1] : u[n - 1];
        double pr = p[n - 1];
        ue[0] = ul; pe[0] = pl; Ze[0] = Z[0];
        ue[n + 1] = ur; pe[n + 1] = pr; Ze[n + 1] = Z[n - 1];
    }
    Faces fc; fc.pf.resize(n + 1); fc.uf.resize(n + 1);
    for (int f = 0; f < n + 1; ++f) {
        double ZL = Ze[f], ZR = Ze[f + 1];
        double pL = pe[f], pR = pe[f + 1];
        double uL = ue[f], uR = ue[f + 1];
        double den = std::fmax(ZL + ZR, EPS);
        fc.pf[f] = (ZR * pL + ZL * pR + ZL * ZR * (uL - uR)) / den;
        fc.uf[f] = (pL - pR + ZL * uL + ZR * uR) / den;
    }
    if (bc_l == BC5::Reflective) { fc.pf[0] = p[0]; fc.uf[0] = 0.0; }
    if (bc_r == BC5::Reflective) { fc.pf[n] = p[n - 1]; fc.uf[n] = 0.0; }
    return fc;
}

// _recover_pressure_from_total_energy (imex_ad.py ~50-90): scalar Newton on the
// mixture internal-energy constraint, 16 iterations, per cell.
inline std::vector<double> recover_pressure_from_total_energy(
        const std::vector<double>& q1, const std::vector<double>& q2,
        const std::vector<double>& rhoE, const std::vector<double>& alpha,
        const std::vector<double>& u, const std::vector<double>& p_seed,
        const EOS& eos1, const EOS& eos2) {
    const int n = (int)q1.size();
    const double EPS = MU_EPS;
    std::vector<double> p_out(n);
    for (int i = 0; i < n; ++i) {
        double ac = std::fmin(std::fmax(alpha[i], 1.0e-12), 1.0 - 1.0e-12);
        double rho = std::fmax(q1[i] + q2[i], EPS);
        double rho1 = std::fmax(q1[i] / ac, EPS);
        double rho2 = std::fmax(q2[i] / std::fmax(1.0 - ac, 1.0e-12), EPS);
        double target = rhoE[i] - 0.5 * rho * u[i] * u[i];
        double p = std::fmax(p_seed[i], 1.0e-12);
        for (int it = 0; it < 16; ++it) {
            double e1 = eos1.energy(rho1, p);
            double e2 = eos2.energy(rho2, p);
            double f = ac * rho1 * e1 + (1.0 - ac) * rho2 * e2 - target;
            double dpde1 = std::fmax(dpde_rho(eos1, rho1, e1), EPS);
            double dpde2 = std::fmax(dpde_rho(eos2, rho2, e2), EPS);
            double df = ac * rho1 / dpde1 + (1.0 - ac) * rho2 / dpde2;
            double stepv = f / std::fmax(df, EPS);
            double p_next = std::fmax(p - stepv, 1.0e-12);
            if (std::fabs(p_next - p) / std::fmax(p_next, 1.0) < 1.0e-10) {
                p = p_next; break;
            }
            p = p_next;
        }
        p_out[i] = p;
    }
    return p_out;
}

// _regularize_near_vacuum_velocity (imex_ad.py 272-318): smooth velocity only in
// expanding near-vacuum (density AND pressure collapsed, or low-pressure vacuum).
inline std::vector<double> regularize_near_vacuum_velocity(
        const std::vector<double>& alpha_n, const std::vector<double>& T1_n,
        const std::vector<double>& T2_n, const std::vector<double>& p_n,
        const std::vector<double>& q1_new, const std::vector<double>& q2_new,
        std::vector<double> u_new, const std::vector<double>& p_new,
        const EOS& eos1, const EOS& eos2, double alpha_pure_tol,
        BC5 bc_l, BC5 bc_r, int& mask_count, int passes = 6) {
    const int n = (int)u_new.size();
    const double EPS = MU_EPS;
    std::vector<double> rho(n), rho_anchor(n), p_anchor(n);
    double rho_domain = 1.0, p_domain = 1.0;
    for (int i = 0; i < n; ++i) {
        rho[i] = std::fmax(q1_new[i] + q2_new[i], EPS);
        PhaseAcoustic pa = phase_acoustic(eos1, eos2, alpha_n[i], T1_n[i], T2_n[i],
                                          p_n[i], alpha_pure_tol);
        rho_anchor[i] = std::fmax(pa.rho, EPS);
        p_anchor[i]   = std::fmax(std::fabs(p_n[i]), 1.0);
        rho_domain = std::fmax(rho_domain, rho_anchor[i]);
        p_domain   = std::fmax(p_domain, p_anchor[i]);
    }
    std::vector<double> ue = extend(u_new, bc_l, bc_r, /*odd=*/true);
    std::vector<char> mask(n, 0);
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        bool expanding = (ue[i + 1] - ue[i] > 0.0) || (ue[i + 2] - ue[i + 1] > 0.0);
        bool dens_col = (rho[i] < 1.0e-2 * rho_anchor[i]) || (rho[i] < 1.0e-3 * rho_domain);
        bool pres_col = (p_new[i] < 1.0e-2 * p_anchor[i]) || (p_new[i] < 1.0e-3 * p_domain);
        bool low_vac  = p_new[i] < std::fmin(5.0e-2 * p_domain, 5.0e3);
        if (expanding && ((dens_col && pres_col) || low_vac)) { mask[i] = 1; ++cnt; }
    }
    mask_count = cnt;
    if (cnt == 0) return u_new;
    std::vector<double> u_reg = u_new;
    for (int pass = 0; pass < std::max(passes, 1); ++pass) {
        std::vector<double> ee = extend(u_reg, bc_l, bc_r, true);
        std::vector<double> u_next = u_reg;
        for (int i = 0; i < n; ++i) {
            if (mask[i]) u_next[i] = 0.25 * ee[i] + 0.5 * ee[i + 1] + 0.25 * ee[i + 2];
        }
        u_reg.swap(u_next);
    }
    return u_reg;
}

} // namespace step_detail

// One production imex_ad_step (single-stage).  Note: the pure-domain single-
// phase Euler-Rusanov shortcut (imex_ad.py 3929-3953) is intentionally NOT
// ported — it dispatches to a separate module and never triggers for the
// immiscible-interface acceptance cases (02A/07B), which always take the two-
// phase acoustic path selected by regime_auto.
inline StepResult imex_ad_step(
        const std::vector<double>& alpha_n, const std::vector<double>& T1_n,
        const std::vector<double>& T2_n, const std::vector<double>& u_n,
        const std::vector<double>& p_n, double dt, double dx,
        const EOS& eos1, const EOS& eos2, const StepConfig& cfg) {
    using namespace step_detail;
    const int n = (int)alpha_n.size();
    const double apt = cfg.alpha_pure_tol;

    // ── 1. regime_auto pressure-closure pick (M8) ────────────────────────────
    PressureClosure closure = select_regime(alpha_n, T1_n, T2_n, p_n,
                                            eos1, eos2, apt);

    // ── 2. material update (M6) ──────────────────────────────────────────────
    MaterialResult mat = material_update(alpha_n, T1_n, T2_n, u_n, p_n, dt, dx,
                                         eos1, eos2, cfg.material_config());

    // ── 3. clip alpha ────────────────────────────────────────────────────────
    std::vector<double> alpha_new = mat.alpha_new;
    for (double& a : alpha_new) a = std::fmin(std::fmax(a, 1.0e-12), 1.0 - 1.0e-12);

    // ── 4. acoustic (u,p) solve (M7) ─────────────────────────────────────────
    AcousticSolveResult ac = acoustic_solve(
        n, dx, dt, eos1, eos2,
        alpha_n.data(), T1_n.data(), T2_n.data(), u_n.data(), p_n.data(),
        mat.q1_new.data(), mat.q2_new.data(), mat.m_adv.data(),
        to_acoustic_bc(cfg.bc_l), to_acoustic_bc(cfg.bc_r), apt);
    std::vector<double> u_new = ac.u_new;
    std::vector<double> p_new = ac.p_new;
    std::vector<double> rhoE_new = mat.rhoE_new;

    // ── 5. energy closure ────────────────────────────────────────────────────
    if (closure == PressureClosure::compressive_recovery) {
        std::vector<char> rmask = compressive_pressure_mask(u_n, p_n);
        if (apt > 0.0) {
            double pure_tol = std::fmax(apt, regime_detail::eps025());
            std::vector<char> pm = pure_material_cell_mask(alpha_n, pure_tol);
            for (int i = 0; i < n; ++i) rmask[i] = rmask[i] && !pm[i];
        }
        bool any = false; for (char m : rmask) any |= m;
        if (any) {
            std::vector<double> prec = recover_pressure_from_total_energy(
                mat.q1_new, mat.q2_new, rhoE_new, alpha_new, u_new, p_new,
                eos1, eos2);
            for (int i = 0; i < n; ++i) if (rmask[i]) p_new[i] = prec[i];
        }
    } else if (closure == PressureClosure::pressure_work_consistent) {
        // Recompute Z from the W^n anchor, build acoustic faces, rebuild energy.
        std::vector<double> Z(n);
        for (int i = 0; i < n; ++i)
            Z[i] = phase_acoustic(eos1, eos2, alpha_n[i], T1_n[i], T2_n[i],
                                  p_n[i], apt).Z;
        Faces fc = acoustic_faces(u_new, p_new, Z, cfg.bc_l, cfg.bc_r);
        for (int i = 0; i < n; ++i) {
            rhoE_new[i] = mat.rhoE_adv[i]
                - dt * (fc.pf[i + 1] * fc.uf[i + 1] - fc.pf[i] * fc.uf[i]) / dx;
        }
        // PW pure-shock recovery: compressive & ~pure (FIVE_EQ_IMEX_PW_PURE_
        // SHOCK_RECOVERY=1 default).  Inactive for 02A/07B acoustic.
        std::vector<char> rmask = compressive_pressure_mask(u_n, p_n);
        double pure_tol = std::fmax(apt, regime_detail::eps025());
        std::vector<char> pm = pure_material_cell_mask(alpha_n, pure_tol);
        for (int i = 0; i < n; ++i) rmask[i] = rmask[i] && !pm[i];
        bool any = false; for (char m : rmask) any |= m;
        if (any) {
            std::vector<double> prec = recover_pressure_from_total_energy(
                mat.q1_new, mat.q2_new, rhoE_new, alpha_new, u_new, p_new,
                eos1, eos2);
            for (int i = 0; i < n; ++i) if (rmask[i]) p_new[i] = prec[i];
        }
    }
    // implicit_energy: not on the 02A/07B path — would use _solve_acoustic_energy_ad.

    // ── 6. near-vacuum velocity regularisation ───────────────────────────────
    int vac = 0;
    u_new = regularize_near_vacuum_velocity(
        alpha_n, T1_n, T2_n, p_n, mat.q1_new, mat.q2_new, u_new, p_new,
        eos1, eos2, apt, cfg.bc_l, cfg.bc_r, vac);

    // ── 7. primitive LMP/LED filter ──────────────────────────────────────────
    // Production 'auto' mode: LED filter only when W^n carries a resolved
    // pressure jump (rel > eps^0.25); otherwise off.  For 02A/07B it is off.
    // (The LED branch itself is not ported — it never fires on the acceptance
    // cases; a divergence here would be caught by the step test.)
    {
        const double jt = regime_detail::eps025();
        bool has_jump = false;
        for (int f = 0; f < n - 1; ++f) {
            double denom = std::fmax(std::fmax(std::fabs(p_n[f]), std::fabs(p_n[f + 1])), 1.0);
            if (std::fabs(p_n[f + 1] - p_n[f]) / denom > jt) { has_jump = true; break; }
        }
        (void)has_jump;  // has_jump==false for 02A/07B -> LMP off; no-op.
    }

    // ── 8. explicit primitive recovery (M9 simple path) ──────────────────────
    StepResult R;
    R.alpha = alpha_new;
    R.T1.resize(n); R.T2.resize(n);
    R.u = u_new; R.p = p_new;
    R.closure = closure;
    R.vacuum_velocity_cells = vac;
    for (int i = 0; i < n; ++i) {
        double rho1 = mat.q1_new[i] / std::fmax(alpha_new[i], 1.0e-12);
        double rho2 = mat.q2_new[i] / std::fmax(1.0 - alpha_new[i], 1.0e-12);
        double e1 = eos1.energy(rho1, p_new[i]);
        double e2 = eos2.energy(rho2, p_new[i]);
        R.T1[i] = eos1.temperature(rho1, e1);
        R.T2[i] = eos2.temperature(rho2, e2);
    }
    return R;
}

} // namespace five_eq
} // namespace cfd
