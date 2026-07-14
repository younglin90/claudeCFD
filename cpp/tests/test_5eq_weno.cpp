// test_5eq_weno.cpp — validate cfd/five_eq/weno.hpp::weno5_face_left against the
// Python oracle tests/5eq_ref/weno5_face_ref.txt (from cpp/tools/gen_5eq_oracle.py).
// Reproduces _weno5_face_left_np exactly (bit-comparable, rel <= 1e-12).
#include "cfd/five_eq/weno.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#ifndef WENO_REF
#  define WENO_REF "weno5_face_ref.txt"
#endif

int main() {
    std::ifstream fin(WENO_REF);
    if (!fin) { std::printf("cannot open ref %s\n", WENO_REF); return 1; }

    int fail = 0, nrows = 0;
    double max_rel = 0.0;
    std::string line;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double idx, qmm, qm, q0, qp, qpp, ref;
        if (!(iss >> idx >> qmm >> qm >> q0 >> qp >> qpp >> ref)) continue;

        double got = cfd::weno5_face_left(qmm, qm, q0, qp, qpp);
        double denom = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-12) {
            std::printf("  [FAIL] case %d got=%.17g ref=%.17g rel=%.3e\n",
                        (int)idx, got, ref, rel);
            ++fail;
        }
        ++nrows;
    }

    if (fail == 0) {
        std::printf("test_5eq_weno: ALL PASS (%d rows, max_rel=%.3e)\n",
                    nrows, max_rel);
        return 0;
    }
    std::printf("test_5eq_weno: %d FAILURES (max_rel=%.3e)\n", fail, max_rel);
    return 1;
}
