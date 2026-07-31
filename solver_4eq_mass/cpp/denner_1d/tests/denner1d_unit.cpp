#include "denner1d/cases.hpp"
#include "denner1d/eos.hpp"
#include "denner1d/numerics.hpp"
#include "denner1d/solver.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "denner1d_unit FAIL: " << msg << "\n";
        ++failures;
    }
}
}  // namespace

int main() {
    // --- EOS sanity + conservative recovery round-trip -----------------------
    const auto air = denner1d::air_phase();
    const auto props = denner1d::phase_props(100000.0, 300.0, air);
    check(std::isfinite(props.rho) && props.rho > 0.0 && props.c > 0.0, "air phase_props");
    check(std::abs(denner1d::van_leer_phi(1.0) - 1.0) < 1.0e-12, "van_leer_phi(1)=1");

    const auto water = denner1d::water_liquid_phase();
    for (double alpha : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const double p0 = 8.0e6;
        const double T0 = 360.0;
        const double rho0 = denner1d::mixture_density(p0, T0, alpha, air, water);
        const double e0 = denner1d::mixture_internal_energy_density(p0, T0, alpha, air, water);
        double p = 1.1 * p0;
        double T = 0.9 * T0;
        const bool ok = denner1d::recover_pressure_temperature_from_density_energy(
            rho0, e0, alpha, air, water, p, T);
        check(ok, "recover converged");
        check(std::abs(p - p0) / p0 < 1.0e-9, "recover p");
        check(std::abs(T - T0) / T0 < 1.0e-9, "recover T");
    }

    // --- ACID_YADV: volume fraction <-> mass fraction round-trip -------------------
    // alpha -> Y -> alpha at a fixed (p,T), both phase orders. Two properties are asserted:
    //   (1) EXACT at the pure ends alpha in {0,1} (the off-phase term is a multiply by 0.0),
    //       which is what a sharp interface with pure cells needs;
    //   (2) in between, the error is machine eps times the CONDITION NUMBER of the map,
    //       kappa = max(rho_a/rho_b, rho_b/rho_a). This is not slop: Y = alpha*rho_a/rho
    //       compresses the whole alpha range into a sliver of the Y range when one phase is
    //       ~10^4x denser, so (1-Y) loses relative precision and the inverse magnifies it.
    //       Measured (scripts/yadv_cond.cpp): worst |d alpha| tracks eps*kappa to within 2x
    //       over p in [1e4,1e9], T in [250,1200], air/water/vapor pairs -- 1.7e-16 at
    //       kappa~1.5 up to 2.9e-12 at kappa~1.9e4 (water|air at p=1e4, T=1200).
    {
        double worst_ratio = 0.0;
        const denner1d::Phase pairs[][2] = {{air, water}, {water, air}};
        for (const auto& pr : pairs) {
            for (const double p0 : {1.0e4, 1.0e5, 8.0e6, 1.0e9}) {
                for (const double T0 : {250.0, 300.0, 360.0, 1200.0}) {
                    const double ra = denner1d::phase_props(p0, T0, pr[0]).rho;
                    const double rb = denner1d::phase_props(p0, T0, pr[1]).rho;
                    const double kappa = std::max(ra / rb, rb / ra);
                    const double tol = 8.0 * 2.220446049250313e-16 * std::max(kappa, 1.0);
                    for (int k = 0; k <= 1000; ++k) {
                        const double al = static_cast<double>(k) / 1000.0;
                        const double Y = denner1d::mass_fraction_from_alpha(al, ra, rb);
                        const double al2 = denner1d::alpha_from_mass_fraction(Y, ra, rb);
                        const double err = std::abs(al2 - al);
                        worst_ratio = std::max(worst_ratio, err / tol);
                        if (k == 0) check(Y == 0.0 && al2 == 0.0, "Y round-trip exact at alpha=0");
                        if (k == 1000) check(Y == 1.0 && al2 == 1.0, "Y round-trip exact at alpha=1");
                        check(err <= tol, "alpha->Y->alpha round-trip within 8*eps*kappa");
                        // the mass-fraction (specific-volume) blend 1/rho = Y/ra + (1-Y)/rb must
                        // reproduce the volume-fraction blend rho = al*ra + (1-al)*rb exactly.
                        const double rho_v = al * ra + (1.0 - al) * rb;
                        const double rho_y = ra * rb / (ra * (1.0 - Y) + Y * rb);
                        check(std::abs(rho_y - rho_v) <= tol * rho_v, "Y/alpha mixture density agree");
                    }
                }
            }
        }
        if (worst_ratio > 1.0)
            std::cerr << "  round-trip worst err / (8 eps kappa) = " << worst_ratio << "\n";
    }

    std::vector<double> a{1.0, 2.0, 3.0};
    auto g = denner1d::apply_ghost(a, "transmissive", "wall", 2, false);
    check(g.size() == 7, "ghost size");
    check(g[0] == 1.0 && g[1] == 1.0 && g[5] == 3.0 && g[6] == 2.0, "ghost values");

    // --- conservative operator invariant: static air-water interface stays static
    // (interface-equilibrium / Collis IEC property). This is the invariant the
    // double-flux recovery broke; the production scheme must hold it to roundoff.
    {
        const auto c01 = denner1d::find_case("01");
        const auto s = denner1d::solve_case(c01);
        double p0 = s.p.empty() ? 0.0 : s.p.front();
        double max_dp = 0.0, max_u = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < s.x.size(); ++i) {
            max_dp = std::max(max_dp, std::abs(s.p[i] - p0));
            max_u = std::max(max_u, std::abs(s.u[i]));
            finite = finite && std::isfinite(s.p[i]) && std::isfinite(s.u[i]);
        }
        check(finite, "case01 finite");
        check(max_dp < 1.0, "case01 interface-equilibrium pressure (|dp| < 1 Pa)");
        check(max_u < 1.0e-4, "case01 interface-equilibrium velocity (|u| < 1e-4)");
    }

    if (failures == 0) {
        std::cout << "denner1d_unit ok\n";
        return 0;
    }
    std::cerr << "denner1d_unit: " << failures << " failure(s)\n";
    return 1;
}
