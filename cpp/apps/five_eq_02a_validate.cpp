// Full 02-A NASG pressure-equilibrium validation for the C++ production path.
#include "cfd/five_eq/solver.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using cfd::BC5;
using cfd::EOS;
using cfd::five_eq::RunConfig;
using cfd::five_eq::RunTermination;

namespace {

constexpr double P0 = 1.0e5;
constexpr int N = 100;

double range_ratio(const std::vector<double>& num, const std::vector<double>& exact) {
    double nmin = num[0], nmax = num[0], emin = exact[0], emax = exact[0];
    for (int i = 0; i < N; ++i) {
        nmin = std::fmin(nmin, num[i]); nmax = std::fmax(nmax, num[i]);
        emin = std::fmin(emin, exact[i]); emax = std::fmax(emax, exact[i]);
    }
    const double den = emax - emin;
    return den > 1.0e-300 ? (nmax - nmin) / den : 1.0;
}

double l1_ratio(const std::vector<double>& num, const std::vector<double>& exact) {
    double emin = exact[0], emax = exact[0], sum = 0.0;
    for (int i = 0; i < N; ++i) {
        emin = std::fmin(emin, exact[i]); emax = std::fmax(emax, exact[i]);
        sum += std::fabs(num[i] - exact[i]);
    }
    const double den = emax - emin;
    return den > 1.0e-300 ? sum / (N * den) : sum / N;
}

double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0.0, mb = 0.0;
    for (int i = 0; i < N; ++i) { ma += a[i]; mb += b[i]; }
    ma /= N; mb /= N;
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (int i = 0; i < N; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        dot += da * db; aa += da * da; bb += db * db;
    }
    return aa * bb > 1.0e-300 ? dot / std::sqrt(aa * bb) : 1.0;
}

bool finite_positive(const cfd::five_eq::StepResult& W,
                     const EOS& air, const EOS& water,
                     std::vector<double>& rho) {
    rho.resize(N);
    for (int i = 0; i < N; ++i) {
        const double r1 = air.density(W.p[i], W.T1[i]);
        const double r2 = water.density(W.p[i], W.T2[i]);
        rho[i] = W.alpha[i] * r1 + (1.0 - W.alpha[i]) * r2;
        if (!std::isfinite(W.alpha[i]) || !std::isfinite(W.T1[i]) ||
            !std::isfinite(W.T2[i]) || !std::isfinite(W.u[i]) ||
            !std::isfinite(W.p[i]) || W.alpha[i] < -1.0e-10 ||
            W.alpha[i] > 1.0 + 1.0e-10 || !(rho[i] > 0.0) || !(W.p[i] > 0.0)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const EOS air = EOS::ideal(1.4, 717.5);
    const EOS water = EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    constexpr double dx = 1.0 / N;
    std::vector<double> alpha0, T10(N, 300.0), T20(N, 300.0), u0(N, 1.0), p0(N, P0);
    alpha0.reserve(N);
    for (int i = 0; i < N; ++i) {
        const double x = (i + 0.5) * dx;
        alpha0.push_back((x >= 0.4 && x < 0.6) ? 1.0e-3 : 1.0 - 1.0e-3);
    }

    std::vector<double> rho0(N);
    for (int i = 0; i < N; ++i) {
        rho0[i] = alpha0[i] * air.density(P0, 300.0) +
                  (1.0 - alpha0[i]) * water.density(P0, 300.0);
    }

    RunConfig cfg;
    cfg.dt_fixed = 0.01;
    cfg.max_steps = 50000;
    cfg.step_config.alpha_pure_tol = 1.0e-3;
    cfg.step_config.bc_l = BC5::Periodic;
    cfg.step_config.bc_r = BC5::Periodic;
    const auto out = cfd::five_eq::solve_imex_ad(
        alpha0, T10, T20, u0, p0, dx, 1.0, air, water, cfg);

    std::vector<double> rho;
    const bool admissible = finite_positive(out.W, air, water, rho);
    double p_rel = 0.0, u_abs = 0.0;
    for (int i = 0; i < N; ++i) {
        p_rel = std::fmax(p_rel, std::fabs(out.W.p[i] - P0) / P0);
        u_abs = std::fmax(u_abs, std::fabs(out.W.u[i] - 1.0));
    }
    const double a_range = range_ratio(out.W.alpha, alpha0);
    const double r_range = range_ratio(rho, rho0);
    const double a_corr = pearson(out.W.alpha, alpha0);
    const double r_corr = pearson(rho, rho0);
    const double a_l1 = l1_ratio(out.W.alpha, alpha0);
    const double r_l1 = l1_ratio(rho, rho0);
    const bool pass = out.termination == RunTermination::completed && out.steps == 100 &&
                      admissible && p_rel <= 1.0e-10 && u_abs <= 1.0e-10 &&
                      a_range >= 0.85 && r_range >= 0.85 && a_corr >= 0.90 &&
                      r_corr >= 0.90 && a_l1 <= 0.20 && r_l1 <= 0.20;
    std::printf("CXX 02_A NASG steps=%d t=%.17g p_rel=%.3e u_abs=%.3e "
                "alpha_range=%.3f rho_range=%.3f corr_alpha=%.3f corr_rho=%.3f "
                "alpha_l1=%.3f rho_l1=%.3f admissible=%d %s\n",
                out.steps, out.t_final, p_rel, u_abs, a_range, r_range, a_corr, r_corr,
                a_l1, r_l1, admissible ? 1 : 0, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
