// cfd/primitive.hpp — primitive W = (alpha1, T1, T2, u, p) <-> conservative
// U = (alpha1*rho1, alpha2*rho2, rho*u, rho*E, alpha1) for the 5-equation model.
//
// C++ port of solver/He2024/primitive_W.py. Per-cell scalar functions (each cell
// is independent in the Python vectorised code), so the caller drives the cell
// loop with OpenMP/OpenACC. Newton, Cramer 3x3 solve, clamps and line-search are
// reproduced exactly for bit-comparable results.
#pragma once
#include "cfd/eos.hpp"
#include <cmath>
#include <limits>

namespace cfd {

constexpr double PRIM_EPS = 1e-30;

struct PrimW { double alpha1, T1, T2, u, p; };
struct ConsU { double m1, m2, mom, rhoE, a1; }; // (a1*rho1, a2*rho2, rho*u, rho*E, a1)

struct PrimAux { double rho1, rho2, e1, e2, rho; };

// ── W -> U ───────────────────────────────────────────────────────────────
CFD_ROUTINE_SEQ
inline ConsU prim_to_cons_W(const PrimW& W, const EOS& eos1, const EOS& eos2,
                            PrimAux* aux = nullptr) {
    double beta = 1.0 - W.alpha1;
    double rho1 = EOS::max2(eos1.density(W.p, W.T1), PRIM_EPS);
    double rho2 = EOS::max2(eos2.density(W.p, W.T2), PRIM_EPS);
    double e1 = eos1.energy(rho1, W.p);
    double e2 = eos2.energy(rho2, W.p);
    double rho = W.alpha1 * rho1 + beta * rho2;
    double q = 0.5 * W.u * W.u;
    double rhoE = W.alpha1 * rho1 * (e1 + q) + beta * rho2 * (e2 + q);
    if (aux) { aux->rho1 = rho1; aux->rho2 = rho2; aux->e1 = e1; aux->e2 = e2; aux->rho = rho; }
    return ConsU{ W.alpha1 * rho1, beta * rho2, rho * W.u, rhoE, W.alpha1 };
}

// ── U -> W (3x3 Newton on (p, T1, T2)) ─────────────────────────────────────
CFD_ROUTINE_SEQ
inline PrimW cons_to_prim_W_newton(const ConsU& U, const EOS& eos1, const EOS& eos2,
                                   double tol = 1e-9, int max_iter = 30,
                                   double T1_init = -1.0, double T2_init = -1.0) {
    double U1 = U.m1, U2 = U.m2, U3 = U.mom, U4 = U.rhoE, U5 = U.a1;
    double alpha1 = U5 < 0.0 ? 0.0 : (U5 > 1.0 ? 1.0 : U5);
    double rho = EOS::max2(U1 + U2, PRIM_EPS);
    double u = U3 / rho;
    double rho_e = U4 - 0.5 * U3 * u;

    double rho1_seed = EOS::max2(U1 / EOS::max2(alpha1, 1e-8), PRIM_EPS);
    double rho2_seed = EOS::max2(U2 / EOS::max2(1.0 - alpha1, 1e-8), PRIM_EPS);
    double e1_seed = EOS::max2(rho_e / EOS::max2(rho, PRIM_EPS) / 2.0, 1.0);
    double T1 = T1_init > 0.0 ? EOS::max2(T1_init, 1.0)
                               : EOS::max2(eos1.temperature(rho1_seed, e1_seed), 1.0);
    double T2 = T2_init > 0.0 ? EOS::max2(T2_init, 1.0)
                               : EOS::max2(eos2.temperature(rho2_seed, e1_seed), 1.0);

    double rho1_g = EOS::max2(U1 / EOS::max2(alpha1, 1e-8), PRIM_EPS);
    double rho2_g = EOS::max2(U2 / EOS::max2(1.0 - alpha1, 1e-8), PRIM_EPS);
    double e1_g = eos1.energy(rho1_g, eos1.pressure_from_rhoT(rho1_g, T1));
    double p_seed_1 = eos1.pressure(rho1_g, e1_g);
    double e2_g = eos2.energy(rho2_g, eos2.pressure_from_rhoT(rho2_g, T2));
    double p_seed_2 = eos2.pressure(rho2_g, e2_g);
    double p = (alpha1 >= 0.5) ? p_seed_1 : p_seed_2;
    p = EOS::max2(p, 1.0);

    for (int it = 0; it < max_iter; ++it) {
        double rho1 = EOS::max2(eos1.density(p, T1), PRIM_EPS);
        double rho2 = EOS::max2(eos2.density(p, T2), PRIM_EPS);
        double e1 = eos1.energy(rho1, p);
        double e2 = eos2.energy(rho2, p);

        double F1 = U1 - alpha1 * rho1;
        double F2 = U2 - (1.0 - alpha1) * rho2;
        double F3 = rho_e - U1 * e1 - U2 * e2;

        double scale = EOS::max2((rho_e < 0 ? -rho_e : rho_e), 1.0);
        double r1 = (F1 < 0 ? -F1 : F1) / EOS::max2((U1 < 0 ? -U1 : U1), 1.0);
        double r2 = (F2 < 0 ? -F2 : F2) / EOS::max2((U2 < 0 ? -U2 : U2), 1.0);
        double r3 = (F3 < 0 ? -F3 : F3) / scale;
        double res = r1; if (r2 > res) res = r2; if (r3 > res) res = r3;
        if (res < tol) break;

        double drho1_dp = eos1.drhodp_T(rho1, T1), drho1_dT = eos1.drhodT_p(rho1, T1);
        double drho2_dp = eos2.drhodp_T(rho2, T2), drho2_dT = eos2.drhodT_p(rho2, T2);
        double de1_dp = eos1.dedp_T(rho1, T1), de1_dT = eos1.dedT_p(rho1, T1);
        double de2_dp = eos2.dedp_T(rho2, T2), de2_dT = eos2.dedT_p(rho2, T2);

        double J11 = -alpha1 * drho1_dp, J12 = -alpha1 * drho1_dT, J13 = 0.0;
        double J21 = -(1.0 - alpha1) * drho2_dp, J22 = 0.0, J23 = -(1.0 - alpha1) * drho2_dT;
        double J31 = -U1 * de1_dp - U2 * de2_dp, J32 = -U1 * de1_dT, J33 = -U2 * de2_dT;

        double det = J11 * (J22 * J33 - J23 * J32)
                   - J12 * (J21 * J33 - J23 * J31)
                   + J13 * (J21 * J32 - J22 * J31);
        if ((det < 0 ? -det : det) < 1e-30) {
            double sgn = (det + 1e-60) >= 0 ? 1.0 : -1.0;
            det = 1e-30 * sgn;
        }
        double b1 = -F1, b2 = -F2, b3 = -F3;
        double d_p = (b1 * (J22 * J33 - J23 * J32)
                    - J12 * (b2 * J33 - J23 * b3)
                    + J13 * (b2 * J32 - J22 * b3)) / det;
        double d_T1 = (J11 * (b2 * J33 - J23 * b3)
                     - b1 * (J21 * J33 - J23 * J31)
                     + J13 * (J21 * b3 - b2 * J31)) / det;
        double d_T2 = (J11 * (J22 * b3 - b2 * J32)
                     - J12 * (J21 * b3 - b2 * J31)
                     + b1 * (J21 * J32 - J22 * J31)) / det;

        double damp = 1.0;
        double p_new = p + damp * d_p, T1_new = T1 + damp * d_T1, T2_new = T2 + damp * d_T2;
        for (int ls = 0; ls < 8; ++ls) {
            if (p_new > 1.0 && T1_new > 1.0 && T2_new > 1.0) break;
            damp *= 0.5;
            p_new = p + damp * d_p; T1_new = T1 + damp * d_T1; T2_new = T2 + damp * d_T2;
        }
        p = EOS::max2(p_new, 1.0);
        T1 = EOS::max2(T1_new, 1.0);
        T2 = EOS::max2(T2_new, 1.0);
    }
    return PrimW{ alpha1, T1, T2, u, p };
}

// Near-pure fallback from five_eq_IMEX/primitive.py::_recover_near_pure_cell.
// The regular 3x3 Newton becomes singular when a ghost phase carries almost
// no mass.  At fixed phase densities the energy equation is scalar in p.
CFD_ROUTINE_SEQ
inline PrimW cons_to_prim_W(const ConsU& U, const EOS& eos1, const EOS& eos2,
                            double tol = 1e-9, int max_iter = 30,
                            double T1_init = -1.0, double T2_init = -1.0,
                            double alpha_pure_tol = 0.0) {
    const PrimW base = cons_to_prim_W_newton(U, eos1, eos2, tol, max_iter,
                                             T1_init, T2_init);
    const bool bad = !std::isfinite(base.T1) || !std::isfinite(base.T2) ||
                     !std::isfinite(base.u) || !std::isfinite(base.p);
    if (alpha_pure_tol <= 0.0 ||
        (U.a1 > alpha_pure_tol && U.a1 < 1.0 - alpha_pure_tol && !bad))
        return base;

    const double alpha = std::fmin(std::fmax(U.a1, 0.0), 1.0);
    const double beta = 1.0 - alpha;
    const double q1 = std::fmax(U.m1, 0.0), q2 = std::fmax(U.m2, 0.0);
    const double rho = std::fmax(q1 + q2, PRIM_EPS);
    const double u = U.mom / rho;
    const double rho_e = U.rhoE - 0.5 * U.mom * u;
    const double mass_floor = 1.0e-14 * std::fmax(rho, 1.0);
    const bool use1 = q1 > mass_floor && alpha > 0.0;
    const bool use2 = q2 > mass_floor && beta > 0.0;
    if (!use1 && !use2) return base;

    const double rho1 = use1 ? q1 / std::fmax(alpha, PRIM_EPS) : 1.0;
    const double rho2 = use2 ? q2 / std::fmax(beta, PRIM_EPS) : 1.0;
    double p = base.p;
    if (!(std::isfinite(p) && p > 0.0)) {
        p = use1 ? eos1.pressure(rho1, rho_e / std::fmax(q1, PRIM_EPS))
                 : eos2.pressure(rho2, rho_e / std::fmax(q2, PRIM_EPS));
    }
    p = std::fmax(std::isfinite(p) ? p : 1.0e5, 1.0);

    auto residual = [&](double trial, double& dF) {
        double F = -rho_e; dF = 0.0;
        if (use1) {
            F += q1 * eos1.energy(rho1, trial);
            const double rt = std::fmax(std::isfinite(eos1.temperature(rho1, eos1.energy(rho1, trial)))
                                                ? eos1.temperature(rho1, eos1.energy(rho1, trial)) : 1.0, 1.0e-12);
            const double rp = eos1.drhodp_T(rho1, rt), rT = eos1.drhodT_p(rho1, rt);
            dF += q1 * (eos1.dedp_T(rho1, rt) - eos1.dedT_p(rho1, rt) * rp / (std::fabs(rT) > PRIM_EPS ? rT : PRIM_EPS));
        }
        if (use2) {
            F += q2 * eos2.energy(rho2, trial);
            const double rt = std::fmax(std::isfinite(eos2.temperature(rho2, eos2.energy(rho2, trial)))
                                                ? eos2.temperature(rho2, eos2.energy(rho2, trial)) : 1.0, 1.0e-12);
            const double rp = eos2.drhodp_T(rho2, rt), rT = eos2.drhodT_p(rho2, rt);
            dF += q2 * (eos2.dedp_T(rho2, rt) - eos2.dedT_p(rho2, rt) * rp / (std::fabs(rT) > PRIM_EPS ? rT : PRIM_EPS));
        }
        return F;
    };
    for (int it = 0; it < max_iter; ++it) {
        double dF = 0.0; const double F = residual(p, dF);
        if (!std::isfinite(F)) return base;
        if (std::fabs(F) <= tol * std::fmax(std::fabs(rho_e), 1.0)) break;
        if (!(std::isfinite(dF) && std::fabs(dF) > PRIM_EPS)) {
            const double h = std::fmax(std::fabs(p) * 1.0e-6, 1.0);
            double unused = 0.0;
            dF = (residual(p + h, unused) - residual(std::fmax(p - h, 1.0), unused)) /
                 (p + h - std::fmax(p - h, 1.0));
        }
        if (!(std::isfinite(dF) && std::fabs(dF) > PRIM_EPS)) return base;
        const double dp = -F / dF;
        double damp = 1.0, p_new = p;
        for (int line = 0; line < 12; ++line) {
            const double trial = p + damp * dp;
            if (std::isfinite(trial) && trial > 1.0) { p_new = trial; break; }
            damp *= 0.5;
        }
        if (std::fabs(p_new - p) <= 1.0e-13 * std::fmax(std::fabs(p), 1.0)) {
            p = std::fmax(p_new, 1.0); break;
        }
        p = std::fmax(p_new, 1.0);
    }
    const double e1 = use1 ? eos1.energy(rho1, p) : 0.0;
    const double e2 = use2 ? eos2.energy(rho2, p) : 0.0;
    double T1 = use1 ? eos1.temperature(rho1, e1) : std::numeric_limits<double>::quiet_NaN();
    double T2 = use2 ? eos2.temperature(rho2, e2) : std::numeric_limits<double>::quiet_NaN();
    if (!(std::isfinite(T1) && T1 > 0.0)) T1 = T1_init > 0.0 ? T1_init : (std::isfinite(T2) ? T2 : 300.0);
    if (!(std::isfinite(T2) && T2 > 0.0)) T2 = T2_init > 0.0 ? T2_init : (std::isfinite(T1) ? T1 : 300.0);
    return (std::isfinite(p) && std::isfinite(T1) && std::isfinite(T2))
        ? PrimW{alpha, T1, T2, u, p} : base;
}

// ── analytic dU/dW (5x5), J[i][j] = dU_i/dW_j ──────────────────────────────
CFD_ROUTINE_SEQ
inline void dUdW_analytic(const PrimW& W, const EOS& eos1, const EOS& eos2,
                          double J[5][5]) {
    double beta = 1.0 - W.alpha1;
    double q = 0.5 * W.u * W.u;
    double rho1 = EOS::max2(eos1.density(W.p, W.T1), PRIM_EPS);
    double rho2 = EOS::max2(eos2.density(W.p, W.T2), PRIM_EPS);
    double e1 = eos1.energy(rho1, W.p);
    double e2 = eos2.energy(rho2, W.p);
    double rho = W.alpha1 * rho1 + beta * rho2;

    double rho1_p = eos1.drhodp_T(rho1, W.T1), rho1_T = eos1.drhodT_p(rho1, W.T1);
    double rho2_p = eos2.drhodp_T(rho2, W.T2), rho2_T = eos2.drhodT_p(rho2, W.T2);
    double e1_p = eos1.dedp_T(rho1, W.T1), e1_T = eos1.dedT_p(rho1, W.T1);
    double e2_p = eos2.dedp_T(rho2, W.T2), e2_T = eos2.dedT_p(rho2, W.T2);

    for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) J[i][j] = 0.0;

    // Row 1: U1 = alpha*rho1
    J[0][0] = rho1;            J[0][1] = W.alpha1 * rho1_T; J[0][4] = W.alpha1 * rho1_p;
    // Row 2: U2 = (1-alpha)*rho2
    J[1][0] = -rho2;           J[1][2] = beta * rho2_T;     J[1][4] = beta * rho2_p;
    // Row 3: U3 = rho*u
    J[2][0] = W.u * (rho1 - rho2);
    J[2][1] = W.alpha1 * W.u * rho1_T;
    J[2][2] = beta * W.u * rho2_T;
    J[2][3] = rho;
    J[2][4] = W.u * (W.alpha1 * rho1_p + beta * rho2_p);
    // Row 4: U4 = rho*E
    double h1 = e1 + q, h2 = e2 + q;
    J[3][0] = rho1 * h1 - rho2 * h2;
    J[3][1] = W.alpha1 * (h1 * rho1_T + rho1 * e1_T);
    J[3][2] = beta * (h2 * rho2_T + rho2 * e2_T);
    J[3][3] = rho * W.u;
    J[3][4] = W.alpha1 * (h1 * rho1_p + rho1 * e1_p) + beta * (h2 * rho2_p + rho2 * e2_p);
    // Row 5: U5 = alpha
    J[4][0] = 1.0;
}

} // namespace cfd
