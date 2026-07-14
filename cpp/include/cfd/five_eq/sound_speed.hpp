// cfd/five_eq/sound_speed.hpp — mixture sound speed + phase acoustic impedance.
//
// C++ port of the 5-equation production sound-speed path:
//   solver/five_eq_IMEX/sound_speed.py::phase_sound_speed_sq (21-33)
//   solver/five_eq_IMEX/sound_speed.py::mixture_sound_speed_sq (kind='kapila', 36-48)
//   solver/five_eq_IMEX/explicit.py::_phase_acoustic (22-40)
//
// Design mirrors eos.hpp: header-only, POD, per-cell scalar leaf functions marked
// CFD_ROUTINE_SEQ (caller drives the OpenMP/OpenACC loop over cells+ghosts). Every
// clamp reproduces the Python np.maximum guard EXACTLY (_EPS = 1e-30) so results
// are bit-comparable (rel <= 1e-12) against tests/5eq_ref/sound_speed_ref.txt.
#pragma once
#include "cfd/eos.hpp"
#include <cmath>

namespace cfd {

// Per-phase c_k^2 from analytic (p,T) derivatives (phase_sound_speed_sq 21-33).
CFD_ROUTINE_SEQ
inline double phase_sound_speed_sq(const EOS& eos, double rho, double T) {
    const double EPS = 1e-30;
    double p     = eos.pressure_from_rhoT(rho, T);
    double rho_p = eos.drhodp_T(rho, T);
    double rho_T = eos.drhodT_p(rho, T);
    double e_p   = eos.dedp_T(rho, T);
    double e_T   = eos.dedT_p(rho, T);
    double pr2 = p / EOS::max2(rho * rho, EPS);
    double num = pr2 * rho_p - e_p;
    double den = e_T - pr2 * rho_T;
    double den_used = (std::fabs(den) > EPS) ? den : EPS;
    double Theta = num / den_used;
    double K = rho_p + rho_T * Theta;
    return 1.0 / EOS::max2(K, EPS);
}

// Kapila (Wood) mixture c^2 (mixture_sound_speed_sq, kind='kapila', 42-46):
//   1/(rho c^2) = alpha1/(rho1 c1^2) + alpha2/(rho2 c2^2).
CFD_ROUTINE_SEQ
inline double mixture_sound_speed_sq(double alpha1, double rho1, double c1_sq,
                                     double rho2, double c2_sq) {
    const double EPS = 1e-30;
    double a2  = 1.0 - alpha1;
    double rho = alpha1 * rho1 + a2 * rho2;
    double inv = alpha1 / EOS::max2(rho1 * c1_sq, EPS)
               + a2     / EOS::max2(rho2 * c2_sq, EPS);
    return 1.0 / EOS::max2(rho * inv, EPS);
}

// Result of the production _phase_acoustic per-cell call.
struct PhaseAcoustic {
    double rho;       // mixture (or pure-branch) density
    double c_mix_sq;  // mixture (or pure-branch-overridden) sound speed^2
    double Z;         // acoustic impedance max(rho*sqrt(max(c_mix_sq,eps)),eps)
};

// Production _phase_acoustic (explicit.py 22-40): mixture Wood/Kapila c^2 with the
// pure-branch override, then impedance Z. rho1/rho2 = max(eos.density(p,T), eps).
CFD_ROUTINE_SEQ
inline PhaseAcoustic phase_acoustic(const EOS& eos1, const EOS& eos2,
                                    double alpha, double T1, double T2, double p,
                                    double alpha_pure_tol) {
    const double EPS = 1e-30;
    double rho1  = EOS::max2(eos1.density(p, T1), EPS);
    double rho2  = EOS::max2(eos2.density(p, T2), EPS);
    double c1_sq = phase_sound_speed_sq(eos1, rho1, T1);
    double c2_sq = phase_sound_speed_sq(eos2, rho2, T2);
    double rho = EOS::max2(alpha * rho1 + (1.0 - alpha) * rho2, EPS);
    double c_mix_sq = mixture_sound_speed_sq(alpha, rho1, c1_sq, rho2, c2_sq);
    if (alpha_pure_tol > 0.0) {
        // np.where applied sequentially: pure1 then pure2 (pure2 wins if both).
        bool pure1 = alpha >= 1.0 - alpha_pure_tol;
        bool pure2 = alpha <= alpha_pure_tol;
        if (pure1) { rho = rho1; c_mix_sq = c1_sq; }
        if (pure2) { rho = rho2; c_mix_sq = c2_sq; }
    }
    double Z = EOS::max2(rho * std::sqrt(EOS::max2(c_mix_sq, EPS)), EPS);
    return {rho, c_mix_sq, Z};
}

} // namespace cfd
