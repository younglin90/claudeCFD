// cfd/five_eq/weno.hpp — WENO5-JS face reconstruction for the acoustic solve.
//
// C++ port of solver/five_eq_IMEX/imex_ad.py::_weno5_face_left_np (1396-1418) and
// _weno3_face_left_np (1373-1393). Header-only, POD, per-face scalar leaf function
// marked CFD_ROUTINE_SEQ. Reproduces the Jiang-Shu betas, the relative-eps JS
// weight form, and the absolute scale floor EXACTLY (bit-comparable, rel <= 1e-12
// vs tests/5eq_ref/weno5_face_ref.txt). No monotone clip is applied (the Python
// returns the raw convex WENO value).
#pragma once
#include "cfd/eos.hpp"   // CFD_ROUTINE_SEQ
#include <cmath>

namespace cfd {

// _WENO_EPS_REL = 1e-6 (dimensionless JS floor); _WENO_TINY = 1e-300 (absolute
// floor on `scale` so a perfectly constant stencil yields w_k = d_k).
constexpr double WENO_EPS_REL = 1.0e-6;
constexpr double WENO_TINY    = 1.0e-300;

// WENO5-JS value at the RIGHT face of cell q0. Linear weights d=(1/10,6/10,3/10).
// Mirror (left state of the cell right of a face) = weno5_face_left(qp3,qp2,qp1,q0,qm1).
CFD_ROUTINE_SEQ
inline double weno5_face_left(double qmm, double qm, double q0,
                              double qp, double qpp) {
    double c0 = (1.0 / 3.0) * qmm - (7.0 / 6.0) * qm + (11.0 / 6.0) * q0;
    double c1 = (-1.0 / 6.0) * qm + (5.0 / 6.0) * q0 + (1.0 / 3.0) * qp;
    double c2 = (1.0 / 3.0) * q0 + (5.0 / 6.0) * qp - (1.0 / 6.0) * qpp;
    double s0a = qmm - 2.0 * qm + q0,  s0b = qmm - 4.0 * qm + 3.0 * q0;
    double s1a = qm - 2.0 * q0 + qp,   s1b = qm - qp;
    double s2a = q0 - 2.0 * qp + qpp,  s2b = 3.0 * q0 - 4.0 * qp + qpp;
    double b0 = (13.0 / 12.0) * s0a * s0a + 0.25 * s0b * s0b;
    double b1 = (13.0 / 12.0) * s1a * s1a + 0.25 * s1b * s1b;
    double b2 = (13.0 / 12.0) * s2a * s2a + 0.25 * s2b * s2b;
    double scale = EOS::max2(EOS::max2(b0, b1), EOS::max2(b2, WENO_TINY));
    double t0 = WENO_EPS_REL + b0 / scale;
    double t1 = WENO_EPS_REL + b1 / scale;
    double t2 = WENO_EPS_REL + b2 / scale;
    double a0 = (1.0 / 10.0) / (t0 * t0);
    double a1 = (6.0 / 10.0) / (t1 * t1);
    double a2 = (3.0 / 10.0) / (t2 * t2);
    double s = a0 + a1 + a2;
    return (a0 * c0 + a1 * c1 + a2 * c2) / s;
}

// WENO3-JS value at the RIGHT face of cell q0. Linear weights d=(1/3,2/3).
// Mirror (left state of the cell right of a face) = weno3_face_left(qp2,qp1,q0).
CFD_ROUTINE_SEQ
inline double weno3_face_left(double qm1, double q0, double qp1) {
    double cand0 = -0.5 * qm1 + 1.5 * q0;
    double cand1 = 0.5 * q0 + 0.5 * qp1;
    double b0 = (q0 - qm1) * (q0 - qm1);
    double b1 = (qp1 - q0) * (qp1 - q0);
    double scale = EOS::max2(EOS::max2(b0, b1), WENO_TINY);
    double t0 = WENO_EPS_REL + b0 / scale;
    double t1 = WENO_EPS_REL + b1 / scale;
    double a0 = (1.0 / 3.0) / (t0 * t0);
    double a1 = (2.0 / 3.0) / (t1 * t1);
    double s = a0 + a1;
    return (a0 * cand0 + a1 * cand1) / s;
}

} // namespace cfd
