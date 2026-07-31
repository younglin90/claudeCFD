// test_primitive.cpp — validate cfd::primitive against Python
// solver/He2024/primitive_W.py reference (water SG + air Ideal, 02-A-like state).
#include "cfd/primitive.hpp"
#include <cstdio>
#include <cmath>

using namespace cfd;

static int g_fail = 0;
static void chk(const char* w, double got, double ref, double rtol) {
    double d = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
    double rel = std::fabs(got - ref) / d;
    if (rel > rtol) {
        std::printf("  [FAIL] %-14s got=%.17g ref=%.17g rel=%.3e\n", w, got, ref, rel);
        ++g_fail;
    }
}

int main() {
    EOS eos1 = EOS::sg(4.4, 6e8, 474.2);   // water
    EOS eos2 = EOS::ideal(1.4, 717.5);     // air
    PrimW W{0.7, 300.0, 305.0, 10.0, 1.0e5};

    // ── W -> U ──
    PrimAux aux;
    ConsU U = prim_to_cons_W(W, eos1, eos2, &aux);
    const double Uref[5] = {868.4802474342752, 0.34272005483520884,
                            8688.229674891103, 543668441.1483744, 0.7};
    chk("U.m1", U.m1, Uref[0], 1e-12);
    chk("U.m2", U.m2, Uref[1], 1e-12);
    chk("U.mom", U.mom, Uref[2], 1e-12);
    chk("U.rhoE", U.rhoE, Uref[3], 1e-12);
    chk("U.a1", U.a1, Uref[4], 1e-12);
    chk("aux.rho1", aux.rho1, 1240.6860677632503, 1e-12);
    chk("aux.rho2", aux.rho2, 1.1424001827840293, 1e-12);

    // ── dU/dW (5x5) ──
    double J[5][5];
    dUdW_analytic(W, eos1, eos2, J);
    const double Jref[5][5] = {
        {1240.6860677632503, -2.8949341581142507, 0.0, 0.0, 1.4472258747446677e-06},
        {-1.1424001827840293, 0.0, -0.001123672310935111, 0.0, 3.4272005483520886e-06},
        {12395.436675804664, -28.949341581142505, -0.011236723109351109, 868.8229674891104, 4.874426423096757e-05},
        {776311977.1833789, -144.74670790554956, -0.05618361554679724, 8688.229674891103, 0.9561260742623312},
        {1.0, 0.0, 0.0, 0.0, 0.0}};
    char lbl[16];
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) {
            std::snprintf(lbl, sizeof(lbl), "J[%d][%d]", i, j);
            chk(lbl, J[i][j], Jref[i][j], 1e-12);
        }

    // ── U -> W round trip (Newton path matches Python) ──
    PrimW Wb = cons_to_prim_W(U, eos1, eos2);
    const double Wbref[5] = {0.7, 300.00000000000006, 305.00000000039773,
                             9.999999999999998, 100000.00000012873};
    chk("Wb.alpha1", Wb.alpha1, Wbref[0], 1e-12);
    chk("Wb.T1", Wb.T1, Wbref[1], 1e-10);
    chk("Wb.T2", Wb.T2, Wbref[2], 1e-10);
    chk("Wb.u", Wb.u, Wbref[3], 1e-12);
    chk("Wb.p", Wb.p, Wbref[4], 1e-10);
    // and it must recover the original W to Newton tolerance
    chk("recover.T1", Wb.T1, W.T1, 1e-8);
    chk("recover.T2", Wb.T2, W.T2, 1e-8);
    chk("recover.p", Wb.p, W.p, 1e-8);

    if (g_fail == 0) { std::printf("test_primitive: ALL PASS\n"); return 0; }
    std::printf("test_primitive: %d FAILURES\n", g_fail);
    return 1;
}
