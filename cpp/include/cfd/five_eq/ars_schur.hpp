// Periodic Schur-Helmholtz Newton correction for one ARS acoustic stage.
#pragma once

#include "cfd/five_eq/ars_residual.hpp"
#include "cfd/five_eq/helmholtz.hpp"
#include "cfd/five_eq/schur_blocks.hpp"

namespace cfd::five_eq {

// One undamped correction.  The caller owns positivity checks, residual
// decrease checks, and repeated Newton iterations.
inline StepResult ars_schur_iteration(const StepResult& W,
                                      const std::vector<ConsU>& target,
                                      double gamma_dt,
                                      const EOS& eos1, const EOS& eos2,
                                      double dx, double dissipation = 0.0) {
    const int n = static_cast<int>(W.p.size());
    const auto residual = ars_stage_residual(
        W, target, gamma_dt, eos1, eos2, dx, BC5::Periodic, BC5::Periodic,
        dissipation, true);

    std::vector<double> rho_eff(n), sigma_pp(n), rhs_p(n), r_tilde_u(n),
        r_tilde_p(n), dp, du(n);
    std::vector<SchurBlocks> blocks(n);

    for (int i = 0; i < n; ++i) {
        blocks[i] = schur_blocks(
            {W.alpha[i], W.T1[i], W.T2[i], W.u[i], W.p[i]}, eos1, eos2);
        const double rho1 = eos1.density(W.p[i], W.T1[i]);
        const double rho2 = eos2.density(W.p[i], W.T2[i]);
        const double c_mix_sq = mixture_sound_speed_sq(
            W.alpha[i], rho1, phase_sound_speed_sq(eos1, rho1, W.T1[i]),
            rho2, phase_sound_speed_sq(eos2, rho2, W.T2[i]));
        rho_eff[i] = 1.0 / std::fmax(c_mix_sq, 1.e-30);
        sigma_pp[i] = std::fabs(blocks[i].sigma_pp) > 1.e-30
            ? blocks[i].sigma_pp : 1.e-30;

        const std::array<double, 3> r_a{{
            residual.m1[i], residual.m2[i], residual.alpha[i]}};
        std::array<double, 3> correction{};
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                correction[row] += blocks[i].a_inv[row][col] * r_a[col];
        for (int k = 0; k < 3; ++k) {
            r_tilde_u[i] -= blocks[i].ua[k] * correction[k];
            r_tilde_p[i] -= blocks[i].pa[k] * correction[k];
        }
        r_tilde_u[i] += residual.mom[i];
        r_tilde_p[i] += residual.rhoE[i];
        rhs_p[i] = -(r_tilde_p[i] - blocks[i].pu /
            std::fmax(blocks[i].uu, 1.e-30) * r_tilde_u[i]);
    }

    dp = solve_helmholtz_periodic(sigma_pp, rho_eff, gamma_dt, dx, rhs_p);
    for (int i = 0; i < n; ++i) {
        double grad_dp;
        if (dissipation > 0.0) {
            const auto face_pressure = [&](int left) {
                const int im1 = (left + n - 1) % n;
                const int ip1 = (left + 1) % n;
                const int ip2 = (left + 2) % n;
                return 0.5 * (dp[left] + dp[ip1]) - dissipation *
                    (-dp[im1] + 3.0 * dp[left] - 3.0 * dp[ip1] + dp[ip2]) / 8.0;
            };
            grad_dp = (face_pressure(i) - face_pressure((i + n - 1) % n)) / dx;
        } else {
            grad_dp = (dp[(i + 1) % n] - dp[(i + n - 1) % n]) / (2.0 * dx);
        }
        du[i] = (-r_tilde_u[i] - blocks[i].up / gamma_dt * dp[i] - grad_dp) /
            std::fmax(blocks[i].uu / gamma_dt, 1.e-30);
    }

    StepResult out = W;
    for (int i = 0; i < n; ++i) {
        const std::array<double, 3> r_a{{
            residual.m1[i], residual.m2[i], residual.alpha[i]}};
        std::array<double, 3> da{};
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                da[row] -= blocks[i].a_inv[row][col] *
                    (gamma_dt * r_a[col] + blocks[i].au[col] * du[i] +
                     blocks[i].ap[col] * dp[i]);
        out.alpha[i] += da[0];
        out.T1[i] += da[1];
        out.T2[i] += da[2];
        out.u[i] += du[i];
        out.p[i] += dp[i];
    }
    return out;
}

} // namespace cfd::five_eq
