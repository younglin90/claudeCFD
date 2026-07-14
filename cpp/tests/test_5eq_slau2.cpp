// test_5eq_slau2.cpp — validate cfd/five_eq/slau2.hpp against the Python oracle
// tests/5eq_ref/slau2_faces_ref.txt (from cpp/tools/gen_5eq_oracle.py).
//
// The oracle calls _slau2_faces_np on an 8-cell air|water jump + Gaussian u/p
// bump with primitive_scheme='tmlpu', bc=(reflective,transmissive). To reach the
// SLAU2 face kernel bit-exactly this test reproduces the exact upstream chain the
// Python driver runs:
//   1. extend_W (reflective L / transmissive R, ng=1)                boundary.py
//   2. per-cell c_mix_sq (phase_acoustic, alpha_pure_tol=1e-8)       -> sound_speed.hpp
//   3. mixture-primitive vanleer MUSCL-Hancock L/R states           imex_ad.py 1771-1835
//        (_mixture_primitive_recon_enabled True: pure interface + p-jump;
//         _mixture_hancock_enabled default on; tvd_kind='vanleer')
//   4. slau2_face kernel                                            -> slau2.hpp
//   5. reflective-wall override (p_face[0]=p_ext[1], u_face[0]=0)
// Bit-comparable, rel <= 1e-12.
#include "cfd/five_eq/sound_speed.hpp"
#include "cfd/five_eq/slau2.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::EOS;

#ifndef SLAU2_REF
#  define SLAU2_REF "slau2_faces_ref.txt"
#endif

static const double EPS = 1e-30;

// van Leer TVD pair slope (imex_ad.py::_vanleer_pair).
static double vanleer_pair(double a, double b) {
    bool same = (a * b) > 0.0;
    double den = a + b;
    double den_safe = std::fabs(den) > EPS ? den : 1.0;
    double slope = 2.0 * a * b / den_safe;
    return same ? slope : 0.0;
}

// interior slopes: out[0]=out[n-1]=0; out[k]=vanleer(phi[k]-phi[k-1], phi[k+1]-phi[k]).
static std::vector<double> slopes(const std::vector<double>& phi) {
    int n = (int)phi.size();
    std::vector<double> out(n, 0.0);
    if (n >= 3)
        for (int k = 1; k < n - 1; ++k)
            out[k] = vanleer_pair(phi[k] - phi[k - 1], phi[k + 1] - phi[k]);
    return out;
}

int main() {
    EOS eos1 = EOS::ideal(1.4, 717.5);
    EOS eos2 = EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    const int n = 8;
    const double P0 = 1.0e5;
    const double alpha_pure_tol = 1e-8;
    const double dx = 1.0 / n;
    const double dt = 0.4 * dx / 2000.0;

    double T1v = eos1.temperature(1.157, eos1.energy(1.157, P0));
    double T2v = eos2.temperature(998.0, eos2.energy(998.0, P0));

    // ── cell-centred primitive state (matches gen_slau2) ─────────────────────
    std::vector<double> a1(n), uu(n), pp(n), T1(n, T1v), T2(n, T2v);
    for (int i = 0; i < n; ++i) {
        double x = (i + 0.5) / n;
        a1[i] = (x < 0.5) ? (1.0 - alpha_pure_tol) : alpha_pure_tol;
        double g = std::exp(-((x - 0.25) * (x - 0.25)) / (2.0 * 0.08 * 0.08));
        uu[i] = 0.02 * g;
        pp[i] = P0 + 1.0e3 * g;
    }

    // ── extend_W: reflective L (u odd), transmissive R; ng=1 → length n+2 ─────
    auto extend = [&](const std::vector<double>& arr, bool odd) {
        std::vector<double> e(n + 2);
        for (int i = 0; i < n; ++i) e[i + 1] = arr[i];
        e[0]     = odd ? -arr[0] : arr[0];   // reflective (even scalar / odd u)
        e[n + 1] = arr[n - 1];               // transmissive
        return e;
    };
    std::vector<double> a_ext  = extend(a1, false);
    std::vector<double> T1_ext = extend(T1, false);
    std::vector<double> T2_ext = extend(T2, false);
    std::vector<double> u_ext  = extend(uu, true);
    std::vector<double> p_ext  = extend(pp, false);
    const int ne = n + 2;

    // ── per-cell mixture c^2 (production _phase_acoustic on the extended state) ─
    std::vector<double> c_mix_sq(ne);
    for (int k = 0; k < ne; ++k)
        c_mix_sq[k] = cfd::phase_acoustic(eos1, eos2, a_ext[k], T1_ext[k],
                                          T2_ext[k], p_ext[k], alpha_pure_tol).c_mix_sq;

    // ── mixture-primitive base (rho, y1); energy unused by SLAU2 ──────────────
    std::vector<double> rho_ext(ne), y1_ext(ne);
    for (int k = 0; k < ne; ++k) {
        double rho1 = EOS::max2(eos1.density(p_ext[k], T1_ext[k]), EPS);
        double rho2 = EOS::max2(eos2.density(p_ext[k], T2_ext[k]), EPS);
        double q1 = EOS::max2(a_ext[k] * rho1, 0.0);
        double q2 = EOS::max2((1.0 - a_ext[k]) * rho2, 0.0);
        double rho = EOS::max2(q1 + q2, EPS);
        rho_ext[k] = rho;
        y1_ext[k]  = rho > 0.0 ? std::fmin(std::fmax(q1 / rho, 0.0), 1.0) : 0.0;
    }

    // ── vanleer MUSCL-Hancock predictor (imex_ad.py 1778-1835) ────────────────
    std::vector<double> drho = slopes(rho_ext);
    std::vector<double> du   = slopes(u_ext);
    std::vector<double> dp   = slopes(p_ext);
    const double inv_dx = 1.0 / dx;
    std::vector<double> rho_t(ne), u_t(ne), p_t(ne);
    for (int k = 0; k < ne; ++k) {
        double rho_x = drho[k] * inv_dx;
        double u_x   = du[k]   * inv_dx;
        double p_x   = dp[k]   * inv_dx;
        double c2       = EOS::max2(c_mix_sq[k], EPS);
        double rho_safe = EOS::max2(rho_ext[k], EPS);
        rho_t[k] = -u_ext[k] * rho_x - rho_safe * u_x;
        u_t[k]   = -u_ext[k] * u_x   - p_x / rho_safe;
        p_t[k]   = -u_ext[k] * p_x   - rho_safe * c2 * u_x;
    }

    const int nf = ne - 1;   // n+1 = 9 faces
    std::vector<double> p_face(nf), u_face(nf);
    std::vector<int>    valid(nf);
    for (int f = 0; f < nf; ++f) {
        double rho_L = rho_ext[f]     + 0.5 * drho[f]     + 0.5 * dt * rho_t[f];
        double rho_R = rho_ext[f + 1] - 0.5 * drho[f + 1] + 0.5 * dt * rho_t[f + 1];
        double u_L   = u_ext[f]       + 0.5 * du[f]       + 0.5 * dt * u_t[f];
        double u_R   = u_ext[f + 1]   - 0.5 * du[f + 1]   + 0.5 * dt * u_t[f + 1];
        double p_L   = p_ext[f]       + 0.5 * dp[f]       + 0.5 * dt * p_t[f];
        double p_R   = p_ext[f + 1]   - 0.5 * dp[f + 1]   + 0.5 * dt * p_t[f + 1];
        rho_L = EOS::max2(rho_L, EPS);
        rho_R = EOS::max2(rho_R, EPS);
        p_L   = EOS::max2(p_L, 1.0e-12);
        p_R   = EOS::max2(p_R, 1.0e-12);

        cfd::Slau2Face fc = cfd::slau2_face(rho_L, rho_R, u_L, u_R, p_L, p_R,
                                            c_mix_sq[f], c_mix_sq[f + 1]);
        p_face[f] = fc.p_face;
        u_face[f] = fc.u_face;
        valid[f]  = fc.valid ? 1 : 0;
    }
    // reflective-wall override at the left boundary (bc_l='reflective').
    p_face[0] = p_ext[1];
    u_face[0] = 0.0;
    valid[0]  = 1;

    // ── compare vs oracle ────────────────────────────────────────────────────
    std::ifstream fin(SLAU2_REF);
    if (!fin) { std::printf("cannot open ref %s\n", SLAU2_REF); return 1; }
    int fail = 0, nrows = 0;
    double max_rel = 0.0;
    std::string line;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double idx, pf, uf, vd;
        if (!(iss >> idx >> pf >> uf >> vd)) continue;
        int f = (int)idx;
        auto chk = [&](const char* fld, double got, double ref) {
            double denom = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
            double rel = std::fabs(got - ref) / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-12) {
                std::printf("  [FAIL] face %d %-6s got=%.17g ref=%.17g rel=%.3e\n",
                            f, fld, got, ref, rel);
                ++fail;
            }
        };
        chk("p_face", p_face[f], pf);
        chk("u_face", u_face[f], uf);
        if (valid[f] != (int)vd) {
            std::printf("  [FAIL] face %d valid got=%d ref=%d\n",
                        f, valid[f], (int)vd);
            ++fail;
        }
        ++nrows;
    }

    if (fail == 0) {
        std::printf("test_5eq_slau2: ALL PASS (%d faces, max_rel=%.3e)\n",
                    nrows, max_rel);
        return 0;
    }
    std::printf("test_5eq_slau2: %d FAILURES (max_rel=%.3e)\n", fail, max_rel);
    return 1;
}
