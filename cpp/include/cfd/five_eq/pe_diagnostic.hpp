// Pressure-equilibrium face and update diagnostics.
#pragma once

#include <cmath>

#include "cfd/primitive.hpp"

namespace cfd::five_eq {

struct FacePEConsistency { double phase1; double phase2; };

inline FacePEConsistency pe_face_consistency(double rho1_face, double rho2_face,
                                             double u_face, double F_q1,
                                             double F_q2, double F_alpha) {
    return {F_q1 - rho1_face * F_alpha, F_q2 - rho2_face * (u_face - F_alpha)};
}

inline double pe_update_residual(const PrimW& W_n, const PrimW& W_new,
                                 const EOS& eos1, const EOS& eos2) {
    const auto internal_energy = [](const PrimW& W, const EOS& e1, const EOS& e2) {
        const double rho1 = e1.density(W.p, W.T1);
        const double rho2 = e2.density(W.p, W.T2);
        return W.alpha1 * rho1 * e1.energy(rho1, W.p) +
            (1.0 - W.alpha1) * rho2 * e2.energy(rho2, W.p);
    };
    const double rho1_n = eos1.density(W_n.p, W_n.T1);
    const double rho2_n = eos2.density(W_n.p, W_n.T2);
    const double rho1_e = eos1.density(W_new.p, W_new.T1);
    const double rho2_e = eos2.density(W_new.p, W_new.T2);
    const double d_rhoe = internal_energy(W_new, eos1, eos2) - internal_energy(W_n, eos1, eos2);
    const double d_q1 = W_new.alpha1 * rho1_e - W_n.alpha1 * rho1_n;
    const double d_q2 = (1.0 - W_new.alpha1) * rho2_e - (1.0 - W_n.alpha1) * rho2_n;
    const double d_alpha = W_new.alpha1 - W_n.alpha1;
    const double alpha = 0.5 * (W_n.alpha1 + W_new.alpha1);
    const double T1 = 0.5 * (W_n.T1 + W_new.T1);
    const double T2 = 0.5 * (W_n.T2 + W_new.T2);
    const double p = 0.5 * (W_n.p + W_new.p);
    const double rho1 = eos1.density(p, T1);
    const double rho2 = eos2.density(p, T2);
    const double e1 = eos1.energy(rho1, p);
    const double e2 = eos2.energy(rho2, p);
    const double rho1_T = eos1.drhodT_p(rho1, T1);
    const double rho2_T = eos2.drhodT_p(rho2, T2);
    const double e1_T = eos1.dedT_p(rho1, T1);
    const double e2_T = eos2.dedT_p(rho2, T2);
    const double safe_rho1_T = std::fabs(rho1_T) > 1.e-30 ? rho1_T : 1.e-30;
    const double safe_rho2_T = std::fabs(rho2_T) > 1.e-30 ? rho2_T : 1.e-30;
    const double chi1 = e1 + rho1 * e1_T / safe_rho1_T;
    const double chi2 = e2 + rho2 * e2_T / safe_rho2_T;
    const double chi_alpha = -rho1 * rho1 * e1_T / safe_rho1_T +
        rho2 * rho2 * e2_T / safe_rho2_T;
    (void)alpha;
    return d_rhoe - chi1 * d_q1 - chi2 * d_q2 - chi_alpha * d_alpha;
}

} // namespace cfd::five_eq
