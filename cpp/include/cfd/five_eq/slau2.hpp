// cfd/five_eq/slau2.hpp — SLAU2 pressure-free material face velocity.
//
// C++ port of the core of solver/five_eq_IMEX/imex_ad.py::_slau2_faces_np
// (599-653): the all-speed, pressure-free face velocity used by the IMEX split.
// This header ports the per-face scalar KERNEL (averaging + (1-Mhat)^2 pressure
// coupling + valid mask). The L/R reconstructed states (rho,u,p) and per-cell
// mixture c^2 are produced upstream (M5 reconstruction + M1 sound_speed) and
// passed in by the caller, which also drives the face loop and applies the
// reflective-wall override (p_face[0]=p_ext[1], u_face[0]=0 at a reflective
// boundary; mirror at the right). Header-only, POD, CFD_ROUTINE_SEQ. Every clamp
// reproduces the Python np.maximum guard EXACTLY (_EPS = 1e-30) so the result is
// bit-comparable (rel <= 1e-12) vs tests/5eq_ref/slau2_faces_ref.txt.
#pragma once
#include "cfd/eos.hpp"   // CFD_ROUTINE_SEQ, EOS::max2
#include <cmath>

namespace cfd {

struct Slau2Face {
    double p_face;
    double u_face;
    bool   valid;
};

// SLAU2 material face velocity for one face. Inputs are the L/R reconstructed
// mixture density/velocity/pressure and the L/R mixture sound speed SQUARED
// (c_mix_sq on the two adjacent cells). Reproduces _slau2_faces_np 630-644.
CFD_ROUTINE_SEQ
inline Slau2Face slau2_face(double rho_L, double rho_R,
                            double u_L, double u_R,
                            double p_L, double p_R,
                            double c_L_sq, double c_R_sq) {
    const double EPS = 1e-30;
    double c_L = std::sqrt(EOS::max2(c_L_sq, EPS));
    double c_R = std::sqrt(EOS::max2(c_R_sq, EPS));
    double c_avg = EOS::max2(0.5 * (c_L + c_R), EPS);
    double u_rms = std::sqrt(0.5 * (u_L * u_L + u_R * u_R));
    double ratio = u_rms / c_avg;
    double mach_hat = ratio < 1.0 ? ratio : 1.0;      // np.minimum(1.0, u_rms/c_avg)
    double one_m = 1.0 - mach_hat;
    double chi = one_m * one_m;                        // (1 - Mhat)^2
    double rho_avg = EOS::max2(0.5 * (rho_L + rho_R), EPS);
    double sqrt_rho_L = std::sqrt(EOS::max2(rho_L, EPS));
    double sqrt_rho_R = std::sqrt(EOS::max2(rho_R, EPS));
    double v_avg = (sqrt_rho_L * u_L + sqrt_rho_R * u_R)
                 / EOS::max2(sqrt_rho_L + sqrt_rho_R, EPS);
    double u_face = v_avg - chi * (p_R - p_L) / EOS::max2(rho_avg * c_avg, EPS);
    double p_face = 0.5 * (p_L + p_R);
    bool valid = std::isfinite(u_face) && std::isfinite(p_face)
               && (rho_L > EPS) && (rho_R > EPS);
    return {p_face, u_face, valid};
}

} // namespace cfd
