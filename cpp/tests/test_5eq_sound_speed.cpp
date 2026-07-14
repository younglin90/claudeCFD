// test_5eq_sound_speed.cpp — validate cfd/five_eq/sound_speed.hpp against the
// Python oracle tests/5eq_ref/sound_speed_ref.txt (from cpp/tools/gen_5eq_oracle.py).
// Reproduces _phase_acoustic + mixture_sound_speed_sq + phase_sound_speed_sq
// exactly (bit-comparable, rel <= 1e-12). EOS: air(ideal 1.4,717.5) / NASG-water.
#include "cfd/five_eq/sound_speed.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::EOS;

#ifndef SS_REF
#  define SS_REF "sound_speed_ref.txt"
#endif

static int g_fail = 0;
static double g_max_rel = 0.0;

static void check(const char* field, int row, double got, double ref) {
    double denom = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
    double rel = std::fabs(got - ref) / denom;
    if (rel > g_max_rel) g_max_rel = rel;
    if (rel > 1e-12) {
        std::printf("  [FAIL] row %d %-9s got=%.17g ref=%.17g rel=%.3e\n",
                    row, field, got, ref, rel);
        ++g_fail;
    }
}

int main() {
    EOS eos1 = EOS::ideal(1.4, 717.5);
    EOS eos2 = EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    const double alpha_pure_tol = 1e-8;
    const double EPS = 1e-30;

    std::ifstream fin(SS_REF);
    if (!fin) { std::printf("cannot open ref %s\n", SS_REF); return 1; }

    std::string line;
    int row = 0, nrows = 0;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double alpha1, T1, T2, p, rho1, rho2, c1_sq, c2_sq, rho, c_mix_sq, Z, c_mix_raw;
        if (!(iss >> alpha1 >> T1 >> T2 >> p >> rho1 >> rho2 >> c1_sq >> c2_sq
                  >> rho >> c_mix_sq >> Z >> c_mix_raw))
            continue;

        double rho1_c  = EOS::max2(eos1.density(p, T1), EPS);
        double rho2_c  = EOS::max2(eos2.density(p, T2), EPS);
        double c1_sq_c = cfd::phase_sound_speed_sq(eos1, rho1_c, T1);
        double c2_sq_c = cfd::phase_sound_speed_sq(eos2, rho2_c, T2);
        double c_raw_c = cfd::mixture_sound_speed_sq(alpha1, rho1_c, c1_sq_c,
                                                     rho2_c, c2_sq_c);
        cfd::PhaseAcoustic pa = cfd::phase_acoustic(
            eos1, eos2, alpha1, T1, T2, p, alpha_pure_tol);

        check("rho1",      row, rho1_c,      rho1);
        check("rho2",      row, rho2_c,      rho2);
        check("c1_sq",     row, c1_sq_c,     c1_sq);
        check("c2_sq",     row, c2_sq_c,     c2_sq);
        check("c_mix_raw", row, c_raw_c,     c_mix_raw);
        check("rho",       row, pa.rho,      rho);
        check("c_mix_sq",  row, pa.c_mix_sq, c_mix_sq);
        check("Z",         row, pa.Z,        Z);
        ++row; ++nrows;
    }

    if (g_fail == 0) {
        std::printf("test_5eq_sound_speed: ALL PASS (%d rows, max_rel=%.3e)\n",
                    nrows, g_max_rel);
        return 0;
    }
    std::printf("test_5eq_sound_speed: %d FAILURES (max_rel=%.3e)\n",
                g_fail, g_max_rel);
    return 1;
}
