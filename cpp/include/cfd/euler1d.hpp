// cfd/euler1d.hpp — 1D compressible Euler (gamma-law) + LLF flux.
//
// C++ port of Euler1D in solver/solve_T-MLP-u/equations.py and llf() in flux.py.
//   U = (rho, rho*u, rho*E),  W = (rho, u, p),  e = p/((g-1)rho),  E = e + u^2/2.
// Scalar per-face/per-cell functions; the caller drives the loop with OpenMP.
#pragma once
#include "cfd/eos.hpp"   // for CFD_ROUTINE_SEQ
#include <cmath>

namespace cfd {

constexpr double EULER_EPS = 1e-30;

struct Euler1D {
    double gamma = 1.4;
    static constexpr int nvar = 3;

    CFD_ROUTINE_SEQ void prim_to_cons(const double W[3], double U[3]) const {
        double rho = W[0] > EULER_EPS ? W[0] : EULER_EPS;
        double u = W[1], p = W[2];
        double e = p / ((gamma - 1.0) * rho);
        double E = e + 0.5 * u * u;
        U[0] = rho; U[1] = rho * u; U[2] = rho * E;
    }
    CFD_ROUTINE_SEQ void cons_to_prim(const double U[3], double W[3]) const {
        double rho = U[0] > EULER_EPS ? U[0] : EULER_EPS;
        double u = U[1] / rho, E = U[2] / rho;
        double p = (gamma - 1.0) * rho * (E - 0.5 * u * u);
        W[0] = rho; W[1] = u; W[2] = p;
    }
    // F.n at conserved state U with face normal n (1D: n=+-1).
    CFD_ROUTINE_SEQ void physical_flux(const double U[3], double n, double F[3]) const {
        double rho = U[0] > EULER_EPS ? U[0] : EULER_EPS;
        double u = U[1] / rho, E = U[2] / rho;
        double p = (gamma - 1.0) * rho * (E - 0.5 * u * u);
        F[0] = rho * u * n;
        F[1] = (rho * u * u + p) * n;
        F[2] = (U[2] + p) * u * n;
    }
    CFD_ROUTINE_SEQ double max_wave_speed(const double U[3]) const {
        double rho = U[0] > EULER_EPS ? U[0] : EULER_EPS;
        double u = U[1] / rho, E = U[2] / rho;
        double p = (gamma - 1.0) * rho * (E - 0.5 * u * u);
        double c2 = gamma * p / rho; if (c2 < EULER_EPS) c2 = EULER_EPS;
        double c = std::sqrt(c2);
        return std::fabs(u) + c;
    }
};

// Local Lax-Friedrichs (Rusanov) numerical flux from primitive L/R states.
CFD_ROUTINE_SEQ
inline void llf_euler1d(const Euler1D& eq, const double WL[3], const double WR[3],
                        double n, double F[3]) {
    double UL[3], UR[3], FL[3], FR[3];
    eq.prim_to_cons(WL, UL);
    eq.prim_to_cons(WR, UR);
    eq.physical_flux(UL, n, FL);
    eq.physical_flux(UR, n, FR);
    double lamL = eq.max_wave_speed(UL), lamR = eq.max_wave_speed(UR);
    double lam = lamL > lamR ? lamL : lamR;
    for (int v = 0; v < 3; ++v)
        F[v] = 0.5 * (FL[v] + FR[v]) - 0.5 * lam * (UR[v] - UL[v]);
}

} // namespace cfd
