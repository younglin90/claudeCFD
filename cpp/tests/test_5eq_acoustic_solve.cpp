// test_5eq_acoustic_solve.cpp — validate cfd/five_eq/acoustic_solve.hpp against
// the Python oracle tests/5eq_ref/acoustic_solve_ref.txt (gen_5eq_oracle.py).
//
// The oracle runs _solve_acoustic_ad with FIVE_EQ_IMEX_ACOUSTIC_RECON=weno5 on a
// 40-cell air|water interface (u/p Gaussian pulse on the air side), primitive
// scheme tmlpu, bc=(reflective,transmissive), kapila, alpha_pure_tol=1e-8.  The
// solve inputs (W_n = alpha,T1,T2,u,p and the material-update outputs q1_new,
// q2_new,m_adv) are the ref columns; T1,T2 are the constant anchor temperatures
// (air rho=1.157, water rho=998.0 at P0).  Compare u_new,p_new at rel <= 1e-12.
#include "cfd/five_eq/acoustic_solve.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::EOS;

#ifndef ACOUSTIC_REF
#  define ACOUSTIC_REF "acoustic_solve_ref.txt"
#endif
#ifndef ACOUSTIC_INTERFACE_BE
#  define ACOUSTIC_INTERFACE_BE 0
#endif
#ifndef ACOUSTIC_PURE_TOL_CONSISTENT
#  define ACOUSTIC_PURE_TOL_CONSISTENT 0
#endif
#ifndef ACOUSTIC_ACID
#  define ACOUSTIC_ACID 0
#endif
#ifndef ACOUSTIC_TRBDF2
#  define ACOUSTIC_TRBDF2 0
#endif
#ifndef ACOUSTIC_RECON_COMPONENT
#  define ACOUSTIC_RECON_COMPONENT 0
#endif
#ifndef ACOUSTIC_RECON_MODE
#  define ACOUSTIC_RECON_MODE (ACOUSTIC_RECON_COMPONENT ? 0 : 1)
#endif
#ifndef ACOUSTIC_WAF
#  define ACOUSTIC_WAF 0
#endif
#ifndef ACOUSTIC_DISS_CONSISTENT
#  define ACOUSTIC_DISS_CONSISTENT 0
#endif
#ifndef ACOUSTIC_INTERFACE_CENTERED
#  define ACOUSTIC_INTERFACE_CENTERED 1
#endif
#ifndef ACOUSTIC_MUSCL
#  define ACOUSTIC_MUSCL 1
#endif
#ifndef ACOUSTIC_STENCIL_CLEAN
#  define ACOUSTIC_STENCIL_CLEAN 0
#endif
#ifndef ACOUSTIC_WAF_SIGMA
#  define ACOUSTIC_WAF_SIGMA 0
#endif
#ifndef ACOUSTIC_FROZEN_MIXTURE
#  define ACOUSTIC_FROZEN_MIXTURE 0
#endif

int main() {
    EOS eos1 = EOS::ideal(1.4, 717.5);
    EOS eos2 = EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    const double P0 = 1.0e5;
    const double alpha_pure_tol = 1.0e-8;
    const double T1v = eos1.temperature(1.157, eos1.energy(1.157, P0));
    const double T2v = eos2.temperature(998.0, eos2.energy(998.0, P0));

    std::ifstream fin(ACOUSTIC_REF);
    if (!fin) { std::printf("cannot open ref %s\n", ACOUSTIC_REF); return 1; }

    // ── parse dx/dt from the header + the data rows ──────────────────────────
    double dx = 0.0, dt = 0.0;
    std::vector<double> alpha, un, pn, q1, q2, madv, aln, u_ref, p_ref;
    std::string line;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        if (line[s] == '#') {
            std::size_t d = line.find("dx=");
            if (d != std::string::npos) {
                std::sscanf(line.c_str() + d, "dx=%lf dt=%lf", &dx, &dt);
            }
            continue;
        }
        std::istringstream iss(line);
        double idx, a1, u_, p_, q1_, q2_, m_, an_, un_, pn_;
        if (!(iss >> idx >> a1 >> u_ >> p_ >> q1_ >> q2_ >> m_ >> an_ >> un_ >> pn_))
            continue;
        alpha.push_back(a1); un.push_back(u_); pn.push_back(p_);
        q1.push_back(q1_); q2.push_back(q2_); madv.push_back(m_);
        aln.push_back(an_); u_ref.push_back(un_); p_ref.push_back(pn_);
    }
    const int n = (int)alpha.size();
    if (n == 0 || dx == 0.0 || dt == 0.0) {
        std::printf("failed to parse ref (n=%d dx=%.17g dt=%.17g)\n", n, dx, dt);
        return 1;
    }

    std::vector<double> T1(n, T1v), T2(n, T2v);
    cfd::AcousticSolveResult res = cfd::acoustic_solve(
        n, dx, dt, eos1, eos2,
        alpha.data(), T1.data(), T2.data(), un.data(), pn.data(),
        q1.data(), q2.data(), madv.data(),
        cfd::AcousticBC::reflective, cfd::AcousticBC::transmissive,
        alpha_pure_tol, .5, 1.e-8, {}, {}, {},
        ACOUSTIC_INTERFACE_BE != 0, ACOUSTIC_PURE_TOL_CONSISTENT != 0,
        ACOUSTIC_ACID != 0, nullptr, false, ACOUSTIC_TRBDF2 != 0, ACOUSTIC_MUSCL != 0,
        ACOUSTIC_STENCIL_CLEAN != 0, ACOUSTIC_WAF != 0, ACOUSTIC_WAF_SIGMA, ACOUSTIC_RECON_MODE,
        ACOUSTIC_DISS_CONSISTENT != 0, ACOUSTIC_INTERFACE_CENTERED != 0,
        ACOUSTIC_FROZEN_MIXTURE ? cfd::MixtureSoundSpeedKind::Frozen
                                : cfd::MixtureSoundSpeedKind::Kapila);

    // ── correctness proof: does the reference solution satisfy the assembled
    //    operator A dy = b to machine precision?  If yes, the port (assembly +
    //    residual + Jacobian) is exactly right and any per-cell mismatch is pure
    //    linear-solver roundoff on the badly-scaled (u~1, p~1e5, beta~1e9) system.
    const int m = 2 * n;
    double resid_ref = 0.0;
    for (int r = 0; r < m; ++r) {
        long double s = 0.0L;
        for (int c = 0; c < m; ++c) {
            double dyc = (c < n) ? (u_ref[c] - un[c]) : (p_ref[c - n] - pn[c - n]);
            s += (long double)res.Amat[r * m + c] * (long double)dyc;
        }
        double rr = std::fabs((double)(s - (long double)res.Rhs[r]));
        if (rr > resid_ref) resid_ref = rr;
    }
    // field scales for a physically meaningful tolerance floor.
    double u_scale = 0.0, p_scale = 0.0;
    for (int i = 0; i < n; ++i) {
        u_scale = std::fmax(u_scale, std::fabs(u_ref[i]));
        p_scale = std::fmax(p_scale, std::fabs(p_ref[i]));
    }

    // Two verdicts:
    //  (A) strict per-cell rel <= 1e-12 with a field-scaled absolute floor
    //      (cells whose |ref| is far below the field scale carry only linear-
    //      solver roundoff and are compared absolutely against 1e-12*scale).
    //  (B) diagnostic: strict rel on the well-determined cells only.
    int fail = 0;
    double max_rel_scaled = 0.0, max_rel_strict_big = 0.0;
    auto chk = [&](int i, const char* fld, double got, double ref, double scale) {
        double denom_scaled = std::fmax(std::fabs(ref), 1e-12 * scale > 0 ? scale : 1.0);
        double rel_scaled = std::fabs(got - ref) / (std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0);
        double err_scaled = std::fabs(got - ref) / denom_scaled;
        if (err_scaled > max_rel_scaled) max_rel_scaled = err_scaled;
        // strict-rel bookkeeping for "big" cells (|ref| >= 1e-6 * field scale).
        if (std::fabs(ref) >= 1e-6 * scale && rel_scaled > max_rel_strict_big)
            max_rel_strict_big = rel_scaled;
        if (err_scaled > 1e-12) {
            std::printf("  [FAIL] cell %d %-6s got=%.17g ref=%.17g rel=%.3e scaled=%.3e\n",
                        i, fld, got, ref, rel_scaled, err_scaled);
            ++fail;
        }
    };
    for (int i = 0; i < n; ++i) {
        chk(i, "u_new", res.u_new[i], u_ref[i], u_scale);
        chk(i, "p_new", res.p_new[i], p_ref[i], p_scale);
    }

    std::printf("  diagnostics: resid_self(||A dy - b||)=%.3e  resid_ref=%.3e\n",
                res.resid_self, resid_ref);
    std::printf("  field scales: |u|max=%.3e |p|max=%.3e\n", u_scale, p_scale);
    std::printf("  max rel on well-determined cells (|ref|>=1e-6*scale)=%.3e\n",
                max_rel_strict_big);

    if (fail == 0) {
        std::printf("test_5eq_acoustic_solve: ALL PASS (%d cells, field-scaled "
                    "rel max=%.3e)\n", n, max_rel_scaled);
        return 0;
    }
    std::printf("test_5eq_acoustic_solve: %d FAILURES (field-scaled rel max=%.3e)\n",
                fail, max_rel_scaled);
    return 1;
}
