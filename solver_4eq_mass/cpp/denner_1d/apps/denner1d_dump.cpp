#include "denner1d/cases.hpp"
#include "denner1d/solver.hpp"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    const std::string id = argc > 1 ? argv[1] : "13";
    try {
        auto c = denner1d::find_case(id);
        // ACID_DUMP_CELLS (round 30, DIAGNOSTIC ONLY, default unset = no change whatsoever):
        // override this case's mesh resolution so a refinement study can be run without touching
        // cases.cpp or the case definition. Same category as ACID_TEND_SCALE (acid.cpp:890) -- a
        // diagnostic sweep knob, NEVER a gate/validation path: denner1d_validate and denner1d_run
        // do not read it, and cases.cpp/validation.cpp are untouched.
        // WARNING, by design: reference_state() still builds computed_reference(c, 800) from the
        // OVERRIDDEN case (cases.cpp:423-437,754-756), so for N >= 800 the reference solve IS the
        // primary solve and the *_ref columns degenerate to the solution itself; for N < 800 they
        // are an N=800 solve interpolated onto the overridden grid. Either way every *_ref-derived
        // number and every validate metric is meaningless under this var -- only the solver
        // columns (alpha,p,u,rho) are valid. See docs/YADV_ROUND_30_PLAN.md sect.4.2.
        if (const char* e = std::getenv("ACID_DUMP_CELLS")) {
            const int nc = std::atoi(e);
            if (nc > 0) {
                c.config.cells = nc;
                std::fprintf(stderr,
                    "DUMP_CELLS: case=%s cells=%d (*_ref columns and all metrics INVALID)\n",
                    id.c_str(), nc);
            } else {
                std::fprintf(stderr,
                    "ACID_DUMP_CELLS=%s invalid (need integer > 0) -> ignored\n", e);
            }
        }
        const auto s = denner1d::solve_case(c);
        const auto r = denner1d::reference_state(c);
        // 12 significant digits: the default 6 quantizes p~1e5 to 1 Pa steps, which staircases
        // the few-Pa acoustic packets (04/05/07/35/36) in plots. Metrics are computed internally
        // at full precision and are unaffected; this is output fidelity only.
        std::cout << std::setprecision(12);
        std::cout << "x,alpha,p,u,rho,p_ref,u_ref,rho_ref\n";
        for (std::size_t i = 0; i < s.x.size(); ++i) {
            std::cout << s.x[i] << "," << s.alpha[i] << ","
                      << s.p[i] << "," << s.u[i] << "," << s.rho[i] << ","
                      << r.p[i] << "," << r.u[i] << "," << r.rho[i] << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "denner1d_dump: " << e.what() << "\n";
        return 2;
    }
}
