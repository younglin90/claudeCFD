// cfd/advection.hpp — scalar linear advection (1 variable) with a possibly
// space-varying, divergence-free velocity field, plus the pure-upwind flux.
// Port of Advection + upwind_advection (equations.py / flux.py).
//   d u/dt + a(x).grad u = 0,  F.n = (a.n) u,  upwind picks the upstream cell.
#pragma once
#include <functional>

namespace cfd {

// velocity(x,y) -> (ax, ay). For constant fields, ignore args.
using VelocityField = std::function<void(double, double, double&, double&)>;

struct Advection2D {
    static constexpr int nvar = 1;
    VelocityField velocity;

    // a.n at face point (fx,fy) with normal (nx,ny).
    double a_dot_n(double fx, double fy, double nx, double ny) const {
        double ax, ay; velocity(fx, fy, ax, ay);
        return ax * nx + ay * ny;
    }
};

// upwind advection flux at a face: F = (a.n) * (a.n>=0 ? uL : uR).
inline double upwind_advection(double adn, double uL, double uR) {
    return adn * (adn >= 0.0 ? uL : uR);
}

} // namespace cfd
