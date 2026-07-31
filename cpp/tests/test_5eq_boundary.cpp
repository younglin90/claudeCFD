#include "cfd/five_eq/boundary.hpp"
#include "cfd/five_eq/material_update.hpp"
#include "cfd/five_eq/acoustic_solve.hpp"
#include "cfd/five_eq/sound_speed.hpp"
#include <cmath>
#include <cstdio>
int main() {
    using cfd::BC5;
    const std::vector<double> q{1.,2.,3.};
    const auto inlet = cfd::five_eq::extend_component(q, BC5::Inlet, BC5::Outlet, true, 7., 9.);
    const auto wall = cfd::five_eq::extend_component(q, BC5::Reflective, BC5::Reflective, true);
    if (inlet[0] != 7. || inlet[4] != 9. || wall[0] != -1. || wall[4] != -3.) return 1;

    // The production stage must consume values, not merely expose a ghost helper.
    const cfd::EOS air = cfd::EOS::ideal(1.4, 287.0);
    const cfd::EOS gas = cfd::EOS::ideal(1.4, 300.0);
    const std::vector<double> a{0.5, 0.5, 0.5}, T1{300., 300., 300.},
                              T2{300., 300., 300.}, u{0., 0., 0.}, p{1.e5, 1.e5, 1.e5};
    const cfd::MaterialConfig base{1.e-8, BC5::Transmissive, BC5::Transmissive};
    cfd::MaterialConfig injected{1.e-8, BC5::Inlet, BC5::Transmissive};
    injected.u_inlet_l = 20.0;
    injected.p_inlet_l = 1.e5;
    const auto m0 = cfd::material_update(a, T1, T2, u, p, 1.e-5, 0.1, air, gas, base);
    const auto m1 = cfd::material_update(a, T1, T2, u, p, 1.e-5, 0.1, air, gas, injected);
    if (!(std::fabs(m1.q1_new[0] - m0.q1_new[0]) > 1.e-10)) return 2;

    std::vector<double> q1(3), q2(3), madv(3, 0.0);
    for (int i = 0; i < 3; ++i) {
        const cfd::ConsU U = cfd::prim_to_cons_W(cfd::PrimW{a[i], T1[i], T2[i], u[i], p[i]}, air, gas);
        q1[i] = U.m1; q2[i] = U.m2;
    }
    const auto ac0 = cfd::acoustic_solve(3, 0.1, 1.e-5, air, gas,
        a.data(), T1.data(), T2.data(), u.data(), p.data(), q1.data(), q2.data(), madv.data(),
        cfd::AcousticBC::transmissive, cfd::AcousticBC::transmissive, 1.e-8);
    const auto ac1 = cfd::acoustic_solve(3, 0.1, 1.e-5, air, gas,
        a.data(), T1.data(), T2.data(), u.data(), p.data(), q1.data(), q2.data(), madv.data(),
        cfd::AcousticBC::inlet, cfd::AcousticBC::outlet, 1.e-8,
        0.5, 1.e-8, 20.0, 1.e5, 1.01e5);
    if (!std::isfinite(ac1.u_new[0]) || !std::isfinite(ac1.p_new[2])) return 3;
    if (!(std::fabs(ac1.u_new[0] - ac0.u_new[0]) > 1.e-8)) return 4;
    std::puts("boundary passed"); return 0;
}
