// Diagnostic: conditioning of the alpha <-> Y round trip vs the phase density ratio.
#include "denner1d/eos.hpp"
#include <cmath>
#include <cstdio>
#include <string>
using namespace denner1d;

int main() {
    const Phase air = air_phase();
    const Phase wat = water_liquid_phase();
    const Phase vap = water_vapor_phase();
    struct PP { const char* nm; Phase a, b; };
    PP pairs[] = {{"air|water", air, wat}, {"water|air", wat, air}, {"air|vapor", air, vap}};
    for (const auto& pr : pairs) {
        for (double p0 : {1.0e4, 1.0e5, 8.0e6, 1.0e9}) {
            for (double T0 : {250.0, 300.0, 360.0, 1200.0}) {
                const double ra = phase_props(p0, T0, pr.a).rho;
                const double rb = phase_props(p0, T0, pr.b).rho;
                double wabs = 0.0, wal = 0.0;
                for (int k = 0; k <= 100000; ++k) {
                    const double al = double(k) / 100000.0;
                    const double Y = mass_fraction_from_alpha(al, ra, rb);
                    const double a2 = alpha_from_mass_fraction(Y, ra, rb);
                    const double e = std::abs(a2 - al);
                    if (e > wabs) { wabs = e; wal = al; }
                }
                std::printf("%-10s p=%-8.3g T=%-6.0f ra=%-11.4g rb=%-11.4g ratio=%-10.3g "
                            "worst|da|=%-11.4g at alpha=%.5f  eps*ratio=%.3g\n",
                            pr.nm, p0, T0, ra, rb, ra / rb, wabs, wal,
                            2.22e-16 * std::max(ra / rb, rb / ra));
            }
        }
    }
    return 0;
}
