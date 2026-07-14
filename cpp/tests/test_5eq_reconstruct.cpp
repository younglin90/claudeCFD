// test_5eq_reconstruct.cpp — validate cfd/five_eq/reconstruct.hpp (M5).
//
// Two validations:
//  (A) Hand-verified superbee unit cases of limited_value (the closed form is
//      small enough to check by hand — no oracle needed).
//  (B) Reproduce reconstruct_upwind_faces under the PRODUCTION superbee limiter
//      vs tests/5eq_ref/reconstruct_ref.txt (from gen_5eq_oracle.py::gen_reconstruct
//      with FIVE_EQ_IMEX_TMLPU_TVD=superbee). Covers upwind-side selection, the
//      MUSCL-Hancock courant factor, the no-courant path, and the floor.
// Bit-comparable, rel <= 1e-12.
#include "cfd/five_eq/reconstruct.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::TvdLimiter;

#ifndef RECON_REF
#  define RECON_REF "reconstruct_ref.txt"
#endif

static int g_fail = 0;
static double g_max_rel = 0.0;

static void check(const char* field, int row, double got, double ref) {
    double denom = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
    double rel = std::fabs(got - ref) / denom;
    if (rel > g_max_rel) g_max_rel = rel;
    if (rel > 1e-12) {
        std::printf("  [FAIL] row %d %-13s got=%.17g ref=%.17g rel=%.3e\n",
                    row, field, got, ref, rel);
        ++g_fail;
    }
}

int main() {
    // ── (A) hand-verified superbee unit cases ────────────────────────────────
    // r=2 -> superbee psi=2. no courant: delta=0.5*(3-1)=1, psi_bound=2,
    //   val=1+2*1=3, clip[0,3]=3.
    check("hand_nocour", -1, cfd::limited_value(0.0, 1.0, 3.0,
                                                TvdLimiter::Superbee, 0.0, false), 3.0);
    // same with courant=0.5: delta=1*(1-0.5)=0.5, psi_bound=(3-1)/0.5=4,
    //   psi=min(2,2,4)=2, val=1+2*0.5=2, clip[0,3]=2.
    check("hand_cour",   -1, cfd::limited_value(0.0, 1.0, 3.0,
                                                TvdLimiter::Superbee, 0.5, true), 2.0);
    // r<=0 (phi_R<phi_L, phi_L>phi_LL): num=-1, den=+1, r=-1 -> psi=0 -> phi_L.
    check("hand_r_neg",  -1, cfd::limited_value(0.0, 1.0, 0.0,
                                                TvdLimiter::Superbee, 0.0, false), 1.0);

    // ── (B) reconstruct_upwind_faces vs the superbee oracle ──────────────────
    const std::vector<double> phi_ext = {
        1.0, 1.0, 1.2, 1.8, 3.0, 3.1, 3.05, 2.0, 0.5, 0.5, 0.6, 0.6};
    const std::vector<double> u_face = {
        2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0};
    const int n_ext = (int)phi_ext.size();
    const int n_face = (int)u_face.size();
    const double dx = 0.1, dt = 0.02;

    std::vector<double> vc(n_face), vn(n_face), vf(n_face);
    cfd::reconstruct_upwind_faces(phi_ext.data(), n_ext, u_face.data(),
                                  TvdLimiter::Superbee, dt, dx,
                                  /*has_courant*/ true, /*floor*/ -1.0, vc.data());
    cfd::reconstruct_upwind_faces(phi_ext.data(), n_ext, u_face.data(),
                                  TvdLimiter::Superbee, 0.0, 0.0,
                                  /*has_courant*/ false, /*floor*/ -1.0, vn.data());
    cfd::reconstruct_upwind_faces(phi_ext.data(), n_ext, u_face.data(),
                                  TvdLimiter::Superbee, dt, dx,
                                  /*has_courant*/ true, /*floor*/ 2.0, vf.data());

    std::ifstream fin(RECON_REF);
    if (!fin) { std::printf("cannot open ref %s\n", RECON_REF); return 1; }
    std::string line;
    int nrows = 0;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double idx, uf, rc, rn, rf;
        if (!(iss >> idx >> uf >> rc >> rn >> rf)) continue;
        int f = (int)idx;
        check("val_courant",   f, vc[f], rc);
        check("val_nocourant", f, vn[f], rn);
        check("val_floored",   f, vf[f], rf);
        ++nrows;
    }

    if (g_fail == 0) {
        std::printf("test_5eq_reconstruct: ALL PASS (%d faces + 3 unit, max_rel=%.3e)\n",
                    nrows, g_max_rel);
        return 0;
    }
    std::printf("test_5eq_reconstruct: %d FAILURES (max_rel=%.3e)\n", g_fail, g_max_rel);
    return 1;
}
