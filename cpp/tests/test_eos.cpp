// test_eos.cpp — validate cfd::EOS against Python solver/He2024/eos_general.py
// reference values (generated 2026-06-16). Bit-comparable: same closed forms.
#include "cfd/eos.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

using cfd::EOS;

static int g_fail = 0;

static void check(const char* eos_name, const char* field,
                  double got, double ref, double rtol = 1e-12) {
    double denom = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
    double rel = std::fabs(got - ref) / denom;
    bool ok = rel <= rtol;
    if (!ok) {
        std::printf("  [FAIL] %-6s %-8s got=%.17g ref=%.17g rel=%.3e\n",
                    eos_name, field, got, ref, rel);
        ++g_fail;
    }
}

// One reference row from the Python generator.
struct Ref {
    const char* name;
    EOS eos;
    double rho, e;
    double p, T, c2, e_back, drhodp, drhodT, dedp, dedT, dens;
};

static void run(const Ref& r) {
    double p   = r.eos.pressure(r.rho, r.e);
    double T   = r.eos.temperature(r.rho, r.e);
    double c2  = r.eos.sound_speed_sq(r.rho, r.e, p);
    double eb  = r.eos.energy(r.rho, p);
    double dpT = r.eos.drhodp_T(r.rho, T);
    double dTp = r.eos.drhodT_p(r.rho, T);
    double edp = r.eos.dedp_T(r.rho, T);
    double edT = r.eos.dedT_p(r.rho, T);
    double dn  = r.eos.density(p, T);
    check(r.name, "p",      p,   r.p);
    check(r.name, "T",      T,   r.T);
    check(r.name, "c2",     c2,  r.c2);
    check(r.name, "e_back", eb,  r.e_back);
    check(r.name, "drhodp", dpT, r.drhodp);
    check(r.name, "drhodT", dTp, r.drhodT);
    check(r.name, "dedp",   edp, r.dedp);
    check(r.name, "dedT",   edT, r.dedT);
    check(r.name, "dens",   dn,  r.dens);
}

int main() {
    std::vector<Ref> refs = {
        {"ideal", EOS::ideal(1.4, 717.5), 1.2, 2.5e5,
         119999.99999999997, 348.4320557491289, 139999.99999999997, 250000.0,
         1.0000000000000003e-05, -0.003444, 0.0, 717.5, 1.2},
        {"ideal", EOS::ideal(1.4, 717.5), 0.5, 1.0e5,
         19999.999999999996, 139.37282229965157, 55999.999999999985, 100000.0,
         2.5000000000000005e-05, -0.0035875, 0.0, 717.5, 0.5},
        {"sg", EOS::sg(4.1, 4.4e8, 474.2), 1000.0, 1.0e6,
         1295999999.9999998, 1180.9363137916491, 7117599.999999998, 1000000.0,
         5.760368663594471e-07, -0.8467857142857143, -0.00025345622119815674,
         846.7857142857142, 1000.0},
        {"nasg", EOS::nasg(2.35, 1e9, 943.8, 6.61e-4, -1167e3), 1100.0, 2.0e6,
         14883400513.008427, 3092.7199522240844, 124341221.24511075, 2000000.0,
         1.8899605267407686e-08, -0.097063427868444, -1.5619508485460898e-05,
         1024.017708982185, 1099.9999999999998},
    };
    for (const auto& r : refs) run(r);

    // Exercise an OpenMP array kernel: pressure over N cells, reduction.
    const int N = 1 << 20;
    std::vector<double> rho(N), e(N), pr(N);
    EOS air = EOS::ideal(1.4, 717.5);
    for (int i = 0; i < N; ++i) { rho[i] = 1.0 + 1e-6 * i; e[i] = 2.0e5; }
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) { pr[i] = air.pressure(rho[i], e[i]); sum += pr[i]; }
    // expected analytic sum: (gamma-1)*e*sum(rho)
    double sum_rho = 0.0;
    for (int i = 0; i < N; ++i) sum_rho += rho[i];
    double expect = 0.4 * 2.0e5 * sum_rho;
    check("omp", "psum", sum, expect, 1e-12);

    if (g_fail == 0) {
        std::printf("test_eos: ALL PASS (%zu rows, OpenMP kernel ok)\n", refs.size());
        return 0;
    }
    std::printf("test_eos: %d FAILURES\n", g_fail);
    return 1;
}
