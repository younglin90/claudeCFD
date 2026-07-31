#include "cfd/primitive.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifndef PRIMITIVE_NEAR_PURE_REF
#define PRIMITIVE_NEAR_PURE_REF "primitive_near_pure_ref.txt"
#endif

int main() {
    const cfd::EOS eos1 = cfd::EOS::ideal(1.4, 717.5);
    const cfd::EOS eos2 = cfd::EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    std::ifstream in(PRIMITIVE_NEAR_PURE_REF);
    if (!in) return 1;
    int fail = 0, rows = 0;
    double max_rel = 0.0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        cfd::ConsU U{}; double t1, t2, a, T1, T2, u, p;
        if (!(s >> U.m1 >> U.m2 >> U.mom >> U.rhoE >> U.a1 >> t1 >> t2
                >> a >> T1 >> T2 >> u >> p)) continue;
        const cfd::PrimW got = cfd::cons_to_prim_W(U, eos1, eos2, 1.e-9, 30, t1, t2, 1.e-8);
        const double gotv[] = {got.alpha1, got.T1, got.T2, got.u, got.p};
        const double refv[] = {a, T1, T2, u, p};
        for (int k = 0; k < 5; ++k) {
            const double rel = std::fabs(gotv[k] - refv[k]) / std::fmax(std::fabs(refv[k]), 1.0);
            max_rel = std::fmax(max_rel, rel);
            if (rel > 1.e-10) ++fail;
        }
        ++rows;
    }
    std::printf("primitive near-pure: rows=%d max_rel=%.3e %s\n", rows, max_rel,
                fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
