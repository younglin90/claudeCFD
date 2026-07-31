// cfd/eos.hpp — General EOS (Ideal / Stiffened-Gas / Noble-Abel-Stiffened-Gas).
//
// C++ port of solver/He2024/eos_general.py. Design: a single POD struct with an
// enum tag instead of virtual dispatch, so every method is a leaf scalar
// function that runs unmodified on the host (OpenMP) and the device (OpenACC).
// Array kernels live in the caller's parallel loops; the EOS only provides
// per-cell thermodynamics. Clamps (1e-30, 1e-10) mirror the Python np.maximum
// guards exactly so results are bit-comparable.
#pragma once
#include <algorithm>

namespace cfd {

#if defined(_OPENACC)
#  define CFD_ROUTINE_SEQ _Pragma("acc routine seq")
#else
#  define CFD_ROUTINE_SEQ
#endif

struct EOS {
    enum Kind { Ideal, SG, NASG };

    Kind   kind  = Ideal;
    double gamma = 1.4;
    double pinf  = 0.0;     // P_infinity (stiffness)
    double kv    = 717.5;   // c_v
    double b     = 0.0;     // co-volume (NASG)
    double eta   = 0.0;     // reference energy q' (NASG)
    double q     = 0.0;     // legacy

    // ── factories ────────────────────────────────────────────────────────
    static EOS ideal(double gamma = 1.4, double kv = 717.5) {
        EOS e; e.kind = Ideal; e.gamma = gamma; e.kv = kv; e.pinf = 0.0;
        e.b = 0.0; e.eta = 0.0; return e;
    }
    static EOS sg(double gamma, double pinf, double kv) {
        EOS e; e.kind = SG; e.gamma = gamma; e.pinf = pinf; e.kv = kv;
        e.b = 0.0; e.eta = 0.0; return e;
    }
    static EOS nasg(double gamma, double pinf, double kv, double b, double eta) {
        EOS e; e.kind = NASG; e.gamma = gamma; e.pinf = pinf; e.kv = kv;
        e.b = b; e.eta = eta; return e;
    }

    CFD_ROUTINE_SEQ static double max2(double a, double b_) { return a > b_ ? a : b_; }

    // ── p(rho, e) ────────────────────────────────────────────────────────
    CFD_ROUTINE_SEQ double pressure(double rho, double e) const {
        switch (kind) {
        case Ideal: return (gamma - 1.0) * rho * e;
        case SG:    return (gamma - 1.0) * rho * e - gamma * pinf;
        default: { // NASG
            double denom = max2(1.0 - b * rho, 1e-10);
            return (gamma - 1.0) * rho * (e - eta) / denom - gamma * pinf;
        }
        }
    }

    // ── e(rho, p) ────────────────────────────────────────────────────────
    CFD_ROUTINE_SEQ double energy(double rho, double p) const {
        switch (kind) {
        case Ideal: return p / ((gamma - 1.0) * max2(rho, 1e-30));
        case SG:    return (p + gamma * pinf) / ((gamma - 1.0) * max2(rho, 1e-30));
        default: {
            double denom = (gamma - 1.0) * max2(rho, 1e-30);
            return (p + gamma * pinf) * (1.0 - b * rho) / denom + eta;
        }
        }
    }

    // ── T(rho, e) ────────────────────────────────────────────────────────
    CFD_ROUTINE_SEQ double temperature(double rho, double e) const {
        switch (kind) {
        case Ideal: return e / kv;
        case SG:    return (e - pinf / max2(rho, 1e-30)) / kv;
        default: {
            double v = 1.0 / max2(rho, 1e-30);
            return (e - eta - pinf * (v - b)) / kv;
        }
        }
    }

    // ── c^2(rho, e, p) ───────────────────────────────────────────────────
    CFD_ROUTINE_SEQ double sound_speed_sq(double rho, double /*e*/, double p) const {
        switch (kind) {
        case Ideal: return gamma * p / max2(rho, 1e-30);
        case SG:    return gamma * (p + pinf) / max2(rho, 1e-30);
        default: {
            double denom = max2(rho * (1.0 - b * rho), 1e-30);
            return gamma * (p + pinf) / denom;
        }
        }
    }

    // ── p(rho, T) ────────────────────────────────────────────────────────
    CFD_ROUTINE_SEQ double pressure_from_rhoT(double rho, double T) const {
        switch (kind) {
        case Ideal: return (gamma - 1.0) * rho * kv * T;
        case SG:    return (gamma - 1.0) * rho * kv * T - pinf;
        default: {
            double denom = max2(1.0 - b * rho, 1e-10);
            return (gamma - 1.0) * rho * kv * T / denom - pinf;
        }
        }
    }

    // ── rho(p, T) — closed form for all three ────────────────────────────
    CFD_ROUTINE_SEQ double density(double p, double T) const {
        switch (kind) {
        case Ideal: return p / ((gamma - 1.0) * kv * max2(T, 1.0));
        case SG:    return (p + pinf) / ((gamma - 1.0) * kv * max2(T, 1.0));
        default: {
            double pp = p + pinf;
            double denom = (gamma - 1.0) * kv * max2(T, 1.0) + b * pp;
            return pp / max2(denom, 1e-30);
        }
        }
    }

    // ── (p,T)-anchored derivatives (for dU/dW assembly) ──────────────────
    CFD_ROUTINE_SEQ double drhodp_T(double rho, double T) const {
        switch (kind) {
        case Ideal: return 1.0 / ((gamma - 1.0) * kv * max2(T, 1.0));
        case SG:    return 1.0 / ((gamma - 1.0) * kv * max2(T, 1.0));
        default: {
            double p  = pressure_from_rhoT(rho, T);
            double pp = max2(p + pinf, 1e-30);
            return rho * rho * (gamma - 1.0) * kv * T / (pp * pp);
        }
        }
    }

    CFD_ROUTINE_SEQ double drhodT_p(double rho, double T) const {
        switch (kind) {
        case Ideal: return -rho / max2(T, 1.0);
        case SG:    return -rho / max2(T, 1.0);
        default: {
            double p  = pressure_from_rhoT(rho, T);
            double pp = max2(p + pinf, 1e-30);
            return -rho * rho * (gamma - 1.0) * kv / pp;
        }
        }
    }

    CFD_ROUTINE_SEQ double dedp_T(double rho, double T) const {
        switch (kind) {
        case Ideal: return 0.0;
        case SG: {
            double p  = pressure_from_rhoT(rho, T);
            double pp = max2(p + pinf, 1e-30);
            return -pinf * (gamma - 1.0) * kv * T / (pp * pp);
        }
        default: {
            double p  = pressure_from_rhoT(rho, T);
            double pp = max2(p + pinf, 1e-30);
            return -pinf * (gamma - 1.0) * kv * T / (pp * pp);
        }
        }
    }

    CFD_ROUTINE_SEQ double dedT_p(double rho, double T) const {
        switch (kind) {
        case Ideal: return kv;
        case SG: {
            double p  = pressure_from_rhoT(rho, T);
            double pp = max2(p + pinf, 1e-30);
            return kv + pinf * (gamma - 1.0) * kv / pp;
        }
        default: {
            double p  = pressure_from_rhoT(rho, T);
            double pp = max2(p + pinf, 1e-30);
            return kv * (p + gamma * pinf) / pp;
        }
        }
    }

    CFD_ROUTINE_SEQ bool is_admissible(double rho) const { return rho > 0.0; }
};

} // namespace cfd
