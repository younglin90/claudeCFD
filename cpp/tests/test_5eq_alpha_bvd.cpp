// test_5eq_alpha_bvd.cpp — validate cfd/five_eq/alpha_bvd.hpp (M4) against the
// Python oracle tests/5eq_ref/alpha_bvd_ref.txt (from gen_5eq_oracle.py::gen_alpha_bvd).
//
// The oracle runs _adaptive_bvd_alpha_face on two profiles with tvd_kind='umist',
// alpha_pure_tol=1e-8, dx=1/16, dt=0.4*dx/2000:
//   case_smooth=0 : sharp pure-jump (alpha 1-1e-8 | 1e-8)  -> CICSAM branch
//   case_smooth=1 : smooth sinusoid 0.5+0.3 sin(2 pi x)    -> MUSCL-Hancock branch
// This test rebuilds both extended profiles + face velocities exactly and
// asserts the reconstructed alpha faces bit-comparably (rel <= 1e-12).
#include "cfd/five_eq/alpha_bvd.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::AlphaTvd;

#ifndef ALPHA_BVD_REF
#  define ALPHA_BVD_REF "alpha_bvd_ref.txt"
#endif

static const double PI = 3.14159265358979311599796346854418516159057617187500;

static int g_fail = 0;
static double g_max_rel = 0.0;

static void check(const char* field, int cs, int f, double got, double ref) {
    double denom = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
    double rel = std::fabs(got - ref) / denom;
    if (rel > g_max_rel) g_max_rel = rel;
    if (rel > 1e-12) {
        std::printf("  [FAIL] case %d face %d %-9s got=%.17g ref=%.17g rel=%.3e\n",
                    cs, f, field, got, ref, rel);
        ++g_fail;
    }
}

int main() {
    const int n = 12;
    const double alpha_pure_tol = 1e-8;
    const double dx = 1.0 / 16.0;
    const double dt = 0.4 * dx / 2000.0;
    const int n_ext = n + 2;      // 14
    const int n_face = n + 1;     // 13

    // ── case 0: sharp pure-jump, constant u_face = 5 ─────────────────────────
    std::vector<double> a0_ext(n_ext), u0(n_face, 5.0);
    {
        std::vector<double> a_pure(n);
        for (int i = 0; i < n; ++i)
            a_pure[i] = (i < n / 2) ? (1.0 - alpha_pure_tol) : alpha_pure_tol;
        a0_ext[0] = a_pure[0];
        for (int i = 0; i < n; ++i) a0_ext[i + 1] = a_pure[i];
        a0_ext[n_ext - 1] = a_pure[n - 1];
    }

    // ── case 1: smooth sinusoid, u_face = 3 cos(2 pi f/n) ────────────────────
    std::vector<double> a1_ext(n_ext), u1(n_face);
    {
        std::vector<double> a_smooth(n);
        for (int i = 0; i < n; ++i) {
            double x = (i + 0.5) / n;
            a_smooth[i] = 0.5 + 0.3 * std::sin(2.0 * PI * x);
        }
        a1_ext[0] = a_smooth[0];
        for (int i = 0; i < n; ++i) a1_ext[i + 1] = a_smooth[i];
        a1_ext[n_ext - 1] = a_smooth[n - 1];
        for (int f = 0; f < n_face; ++f)
            u1[f] = 3.0 * std::cos(2.0 * PI * ((double)f / n));
    }

    std::vector<double> face0(n_face), face1(n_face);
    cfd::adaptive_bvd_alpha_face(a0_ext.data(), n_ext, u0.data(), n_face,
                                 dt, dx, AlphaTvd::Umist, alpha_pure_tol, face0.data());
    cfd::adaptive_bvd_alpha_face(a1_ext.data(), n_ext, u1.data(), n_face,
                                 dt, dx, AlphaTvd::Umist, alpha_pure_tol, face1.data());

    std::ifstream fin(ALPHA_BVD_REF);
    if (!fin) { std::printf("cannot open ref %s\n", ALPHA_BVD_REF); return 1; }
    std::string line;
    int nrows = 0;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double cs, idx, uf, af;
        if (!(iss >> cs >> idx >> uf >> af)) continue;
        int c = (int)cs, f = (int)idx;
        const std::vector<double>& u = c ? u1 : u0;
        const std::vector<double>& face = c ? face1 : face0;
        check("u_face", c, f, u[f], uf);
        check("alpha_face", c, f, face[f], af);
        ++nrows;
    }

    if (g_fail == 0) {
        std::printf("test_5eq_alpha_bvd: ALL PASS (%d rows, max_rel=%.3e)\n",
                    nrows, g_max_rel);
        return 0;
    }
    std::printf("test_5eq_alpha_bvd: %d FAILURES (max_rel=%.3e)\n", g_fail, g_max_rel);
    return 1;
}
