// cfd/five_eq/energy_flux.hpp — M2 EOS-consistent face energy flux.
//
// C++ port of solver/five_eq_IMEX/energy_flux.py (42-204).
//
//   total_energy_flux = F_rho_e(form) + 0.5*u_f^2 * F_rho
//
// with three internal-energy face-flux forms:
//   * "allaire"      : F_rho_e = e1_f*F_q1 + e2_f*F_q2   (the simple baseline).
//   * "differential" : APEC differential chi_k = e_k + rho_k*e_T/rho_T with a
//                      GLOBAL relative floor on rho_T (fallback_eps=1e-3).
//   * "secant"       : path-consistent secant chi_bar such that the discrete
//                      Delta g identity is byte-exact (energy_flux.py::_secant_chi).
// A pure-face collapse (alpha_pure_tol>0) reduces the flux to the single active
// phase e_k*F_qk. On the PRODUCTION 02_A/07_B path the material update uses the
// direct rE_f*u_star advective energy flux (material_update.hpp), NOT this
// module; total_energy_flux is only on the apec_pe pressure closure — so the
// APEC/secant branch is a correct but secondary port (lower priority).
//
// The "secant" per-face core is a portable CFD_ROUTINE_SEQ scalar kernel; the
// "differential" global floor needs a host reduction, so it is an array driver.
// Every clamp reproduces the Python guard EXACTLY. Bit-comparable (rel <= 1e-12)
// vs tests/5eq_ref/energy_flux_ref.txt.
#pragma once
#include "cfd/eos.hpp"
#include <cmath>
#include <vector>

namespace cfd {

enum class EnergyForm { Allaire, Differential, Secant };

// Per-face L/R + midpoint face state consumed by the energy flux (a subset of
// energy_flux.py's `face` dict): the fields each form actually reads.
struct FaceEnergy {
    std::vector<double> alpha, p, u;             // face mid alpha, p_star, u_star
    std::vector<double> a_L, a_R;                // reconstructed L/R volume fractions
    std::vector<double> rho1, rho2;              // face mid phase densities
    std::vector<double> rho1_L, rho1_R, rho2_L, rho2_R;
    std::vector<double> T1, T2, e1, e2;          // face mid phase temps / energies
};

namespace detail {
constexpr double EF_EPS = 1e-30;
inline double ef_max(double a, double b) { return a > b ? a : b; }
inline double ef_npsign(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }
} // namespace detail

// Secant chi triple for one face (energy_flux.py::_secant_chi 42-125).
CFD_ROUTINE_SEQ
inline void secant_chi(const EOS& eos1, const EOS& eos2,
                       double p_f, double a_L, double a_R,
                       double rho1_L, double rho2_L, double rho1_R, double rho2_R,
                       double rho1_f, double rho2_f, double T1_f, double T2_f,
                       double e1_f, double e2_f,
                       double& chi1, double& chi2, double& chia,
                       double increment_floor = 1e-12) {
    using detail::ef_max;
    double q1_L = a_L * rho1_L, q1_R = a_R * rho1_R;
    double q2_L = (1.0 - a_L) * rho2_L, q2_R = (1.0 - a_R) * rho2_R;
    double da = a_R - a_L, dq1 = q1_R - q1_L, dq2 = q2_R - q2_L;

    double rho1_T_f = eos1.drhodT_p(rho1_f, T1_f), e1_T_f = eos1.dedT_p(rho1_f, T1_f);
    double rho2_T_f = eos2.drhodT_p(rho2_f, T2_f), e2_T_f = eos2.dedT_p(rho2_f, T2_f);
    double r1Td = (std::fabs(rho1_T_f) > 1e-30) ? rho1_T_f : 1e-30;
    double r2Td = (std::fabs(rho2_T_f) > 1e-30) ? rho2_T_f : 1e-30;

    // step (a): vary q1
    double rho1_aL_qR = q1_R / ef_max(a_L, 1e-12);
    double e1_at_aL_qL = eos1.energy(rho1_L, p_f);
    double e1_at_aL_qR = eos1.energy(rho1_aL_qR, p_f);
    double g_a_L = q1_L * e1_at_aL_qL + q2_L * eos2.energy(rho2_L, p_f);
    double g_a_R = q1_R * e1_at_aL_qR + q2_L * eos2.energy(rho2_L, p_f);
    double safe_dq1 = (std::fabs(dq1) > increment_floor) ? dq1 : 1.0;
    chi1 = (g_a_R - g_a_L) / safe_dq1;
    double chi1_diff = e1_f + rho1_f * e1_T_f / r1Td;
    if (!(std::fabs(dq1) > increment_floor)) chi1 = chi1_diff;

    // step (b): vary q2
    double inv_b = 1.0 / ef_max(1.0 - a_L, 1e-12);
    double e2_at_aL_qL = eos2.energy(q2_L * inv_b, p_f);
    double e2_at_aL_qR = eos2.energy(q2_R * inv_b, p_f);
    double g_b_L = q1_R * e1_at_aL_qR + q2_L * e2_at_aL_qL;
    double g_b_R = q1_R * e1_at_aL_qR + q2_R * e2_at_aL_qR;
    double safe_dq2 = (std::fabs(dq2) > increment_floor) ? dq2 : 1.0;
    chi2 = (g_b_R - g_b_L) / safe_dq2;
    double chi2_diff = e2_f + rho2_f * e2_T_f / r2Td;
    if (!(std::fabs(dq2) > increment_floor)) chi2 = chi2_diff;

    // step (c): vary alpha
    double rho1_aL = q1_R / ef_max(a_L, 1e-12);
    double rho2_aL = q2_R / ef_max(1.0 - a_L, 1e-12);
    double rho1_aR = q1_R / ef_max(a_R, 1e-12);
    double rho2_aR = q2_R / ef_max(1.0 - a_R, 1e-12);
    double g_c_L = q1_R * eos1.energy(rho1_aL, p_f) + q2_R * eos2.energy(rho2_aL, p_f);
    double g_c_R = q1_R * eos1.energy(rho1_aR, p_f) + q2_R * eos2.energy(rho2_aR, p_f);
    double safe_da = (std::fabs(da) > increment_floor) ? da : 1.0;
    chia = (g_c_R - g_c_L) / safe_da;
    double chia_diff = -rho1_f * rho1_f * e1_T_f / r1Td + rho2_f * rho2_f * e2_T_f / r2Td;
    if (!(std::fabs(da) > increment_floor)) chia = chia_diff;
}

// total_energy_flux (energy_flux.py::total_energy_flux 170-204). Array driver:
// F_rE[f] for every face. F_q1/F_q2/F_alpha/F_rho are the conservative face
// fluxes the caller already assembled. alpha_pure_tol>0 enables pure collapse.
inline void total_energy_flux(const FaceEnergy& face, const EOS& eos1, const EOS& eos2,
                              const std::vector<double>& F_q1,
                              const std::vector<double>& F_q2,
                              const std::vector<double>& F_alpha,
                              const std::vector<double>& F_rho,
                              EnergyForm form, double alpha_pure_tol,
                              std::vector<double>& F_rE) {
    using namespace detail;
    int nf = (int)face.alpha.size();
    F_rE.resize(nf);
    std::vector<double> F_rho_e(nf);

    if (form == EnergyForm::Differential) {
        std::vector<double> rho1_T(nf), rho2_T(nf), e1_T(nf), e2_T(nf);
        double m1 = 0.0, m2 = 0.0;
        for (int f = 0; f < nf; ++f) {
            rho1_T[f] = eos1.drhodT_p(face.rho1[f], face.T1[f]);
            rho2_T[f] = eos2.drhodT_p(face.rho2[f], face.T2[f]);
            e1_T[f]   = eos1.dedT_p(face.rho1[f], face.T1[f]);
            e2_T[f]   = eos2.dedT_p(face.rho2[f], face.T2[f]);
            m1 = ef_max(m1, std::fabs(rho1_T[f]));
            m2 = ef_max(m2, std::fabs(rho2_T[f]));
        }
        double floor1 = ef_max(1.0e-3 * m1, 1e-30);
        double floor2 = ef_max(1.0e-3 * m2, 1e-30);
        for (int f = 0; f < nf; ++f) {
            double r1s = (std::fabs(rho1_T[f]) > floor1) ? rho1_T[f] : ef_npsign(rho1_T[f] + 1e-300) * floor1;
            double r2s = (std::fabs(rho2_T[f]) > floor2) ? rho2_T[f] : ef_npsign(rho2_T[f] + 1e-300) * floor2;
            double chi1 = face.e1[f] + face.rho1[f] * e1_T[f] / r1s;
            double chi2 = face.e2[f] + face.rho2[f] * e2_T[f] / r2s;
            double chia = -face.rho1[f] * face.rho1[f] * e1_T[f] / r1s
                        +  face.rho2[f] * face.rho2[f] * e2_T[f] / r2s;
            bool bad1 = std::fabs(rho1_T[f]) <= floor1;
            bool bad2 = std::fabs(rho2_T[f]) <= floor2;
            if (bad1) chi1 = face.e1[f];
            if (bad2) chi2 = face.e2[f];
            if (bad1 || bad2) chia = 0.0;
            F_rho_e[f] = chi1 * F_q1[f] + chi2 * F_q2[f] + chia * F_alpha[f];
        }
    } else if (form == EnergyForm::Secant) {
        for (int f = 0; f < nf; ++f) {
            double chi1, chi2, chia;
            secant_chi(eos1, eos2, face.p[f], face.a_L[f], face.a_R[f],
                       face.rho1_L[f], face.rho2_L[f], face.rho1_R[f], face.rho2_R[f],
                       face.rho1[f], face.rho2[f], face.T1[f], face.T2[f],
                       face.e1[f], face.e2[f], chi1, chi2, chia);
            F_rho_e[f] = chi1 * F_q1[f] + chi2 * F_q2[f] + chia * F_alpha[f];
        }
    } else { // Allaire
        for (int f = 0; f < nf; ++f)
            F_rho_e[f] = face.e1[f] * F_q1[f] + face.e2[f] * F_q2[f];
    }

    for (int f = 0; f < nf; ++f) {
        if (alpha_pure_tol > 0.0) {
            double a = face.alpha[f];
            if (a >= 1.0 - alpha_pure_tol) F_rho_e[f] = face.e1[f] * F_q1[f];  // pure1
            if (a <= alpha_pure_tol)       F_rho_e[f] = face.e2[f] * F_q2[f];  // pure2
        }
        double u = face.u[f];
        F_rE[f] = F_rho_e[f] + 0.5 * u * u * F_rho[f];
    }
}

} // namespace cfd
