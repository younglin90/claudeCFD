// cfd/five_eq/regime_auto.hpp — M8 regime_auto energy-closure automaton.
//
// Header-only C++ twin of the pressure-closure selection in
// solver/five_eq_IMEX/imex_ad.py::imex_ad_step (3912-3927) plus the helpers
// _pressure_jump_stiff_to_soft_material (174-236, aliased
// _pressure_jump_high_to_low_impedance), _compressive_pressure_mask (93-114)
// and _pure_material_cell_mask (122-142).
//
// The selection is TOPOLOGY-KEYED (never a case id):
//   pure_tol_auto = max(alpha_pure_tol, eps^0.25)
//   has_immiscible_interface = (min(alpha) <= pure_tol_auto)
//                            & (max(alpha) >= 1 - pure_tol_auto)
//   if has_immiscible_interface:
//       stiff_to_soft ? compressive_recovery : pressure_work_consistent
//   else:
//       implicit_energy
//
// For 02_A and 07_B this resolves to pressure_work_consistent (immiscible
// interface present, no stiff->soft pressure jump at a pure material face).
#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "cfd/eos.hpp"
#include "cfd/five_eq/sound_speed.hpp"   // phase_acoustic, PhaseAcoustic

namespace cfd {
namespace five_eq {

// Energy-closure pressure-work regime.  Ordered as the Python string values.
enum class PressureClosure {
    pressure_work_consistent,
    compressive_recovery,
    implicit_energy,
    implicit_energy_momentum,
    no_recovery,
    path_kapila,
    dual_entropy,
    apec_pe,
};

namespace regime_detail {

inline double eps025() {
    return std::pow(std::numeric_limits<double>::epsilon(), 0.25);
}

} // namespace regime_detail

// _pressure_jump_stiff_to_soft_material (imex_ad.py 174-236).
// True when the strongest alpha-jump-at-pure-face pressure discontinuity is
// stiff -> soft.  Stiffness of the high-pressure side uses |pinf| (+|q|, but the
// He2024 EOS objects carry q=0, so only pinf contributes); acoustic impedance Z
// breaks ties when the two pure-side stiffnesses are equal.
inline bool pressure_jump_stiff_to_soft(
        const std::vector<double>& alpha, const std::vector<double>& T1,
        const std::vector<double>& T2, const std::vector<double>& p,
        const EOS& eos1, const EOS& eos2, double alpha_pure_tol,
        MixtureSoundSpeedKind mixture_kind = MixtureSoundSpeedKind::Kapila) {
    const int n = (int)alpha.size();
    if (n < 2) return false;
    const double jump_tol = regime_detail::eps025();
    const double pure_tol = std::fmax(alpha_pure_tol, jump_tol);

    // Find the pure-face alpha-jump face with the largest relative pressure jump.
    int idx = -1;
    double best_rel = -1.0;
    for (int f = 0; f < n - 1; ++f) {
        double aL = alpha[f], aR = alpha[f + 1];
        bool alpha_jump = std::fabs(aR - aL) > jump_tol;
        bool pure_face = (aL <= pure_tol) || (aL >= 1.0 - pure_tol)
                       || (aR <= pure_tol) || (aR >= 1.0 - pure_tol);
        if (!(alpha_jump && pure_face)) continue;
        double pL = p[f], pR = p[f + 1];
        double p_scale = std::fmax(std::fmax(std::fabs(pL), std::fabs(pR)), 1.0);
        double rel = std::fabs(pR - pL) / p_scale;
        if (rel > jump_tol && rel > best_rel) { best_rel = rel; idx = f; }
    }
    if (idx < 0) return false;

    auto side_stiffness = [&](double alpha_value) -> double {
        const EOS& e = (alpha_value >= 0.5) ? eos1 : eos2;
        return std::fabs(e.pinf) + std::fabs(e.q);   // q == 0 for these EOS.
    };
    auto Zcell = [&](int i) -> double {
        PhaseAcoustic pa = phase_acoustic(eos1, eos2, alpha[i], T1[i], T2[i],
                                          p[i], alpha_pure_tol, mixture_kind);
        return pa.Z;
    };

    double pL = p[idx], pR = p[idx + 1];
    double z_hi, z_lo, s_hi, s_lo;
    if (pL >= pR) {
        z_hi = Zcell(idx);     z_lo = Zcell(idx + 1);
        s_hi = side_stiffness(alpha[idx]);   s_lo = side_stiffness(alpha[idx + 1]);
    } else {
        z_hi = Zcell(idx + 1); z_lo = Zcell(idx);
        s_hi = side_stiffness(alpha[idx + 1]); s_lo = side_stiffness(alpha[idx]);
    }
    if (std::fabs(s_hi - s_lo) > 0.0) return s_hi > s_lo;
    return std::isfinite(z_hi) && std::isfinite(z_lo) && (z_hi > z_lo);
}

// imex_ad.py 3912-3927: topology-keyed regime selection.
inline PressureClosure select_regime(
        const std::vector<double>& alpha, const std::vector<double>& T1,
        const std::vector<double>& T2, const std::vector<double>& p,
        const EOS& eos1, const EOS& eos2, double alpha_pure_tol,
        MixtureSoundSpeedKind mixture_kind = MixtureSoundSpeedKind::Kapila) {
    double amin = alpha[0], amax = alpha[0];
    for (double a : alpha) { amin = std::fmin(amin, a); amax = std::fmax(amax, a); }
    double pure_tol_auto = std::fmax(alpha_pure_tol, regime_detail::eps025());
    bool immiscible = (amin <= pure_tol_auto) && (amax >= 1.0 - pure_tol_auto);
    if (!immiscible) return PressureClosure::implicit_energy;
    if (pressure_jump_stiff_to_soft(alpha, T1, T2, p, eos1, eos2, alpha_pure_tol,
                                    mixture_kind))
        return PressureClosure::compressive_recovery;
    return PressureClosure::pressure_work_consistent;
}

// _compressive_pressure_mask (imex_ad.py 93-114): cells adjacent to a resolved
// compressive pressure wave (rel jump > eps^0.25 AND u_L > u_R).
inline std::vector<char> compressive_pressure_mask(
        const std::vector<double>& u, const std::vector<double>& p) {
    const int n = (int)p.size();
    std::vector<char> mask(n, 0);
    if (n < 2) return mask;
    const double jt = regime_detail::eps025();
    for (int f = 0; f < n - 1; ++f) {
        double denom = std::fmax(std::fmax(std::fabs(p[f]), std::fabs(p[f + 1])), 1.0);
        double rel = std::fabs(p[f + 1] - p[f]) / denom;
        bool compression = u[f] > u[f + 1];
        if (rel > jt && compression) { mask[f] = 1; mask[f + 1] = 1; }
    }
    return mask;
}

// _pure_material_cell_mask (imex_ad.py 122-142): cells adjacent to a material
// jump with at least one pure-side state.
inline std::vector<char> pure_material_cell_mask(
        const std::vector<double>& alpha, double pure_tol) {
    const int n = (int)alpha.size();
    std::vector<char> mask(n, 0);
    if (n < 2) return mask;
    const double jt = regime_detail::eps025();
    for (int f = 0; f < n - 1; ++f) {
        double aL = alpha[f], aR = alpha[f + 1];
        bool jump = std::fabs(aR - aL) > jt;
        bool pure = (aL <= pure_tol) || (aL >= 1.0 - pure_tol)
                  || (aR <= pure_tol) || (aR >= 1.0 - pure_tol);
        if (jump && pure) { mask[f] = 1; mask[f + 1] = 1; }
    }
    return mask;
}

} // namespace five_eq
} // namespace cfd
