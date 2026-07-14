// test_5eq_material_update.cpp — validate cfd/five_eq/material_update.hpp against
// the Python oracle tests/5eq_ref/material_update_ref.txt (gen_5eq_oracle.py::
// gen_material_update). Two production-config cases:
//   case_07B=0 : 02_A IC (n=100, periodic, apure=1e-3) — density-recon + alpha-FCT.
//   case_07B=1 : small 07_B-like air|water pulse (n=40, reflective/transmissive,
//                apure=1e-8) — mixture-hancock recon + primitive-FCT + alpha-FCT.
// Bit-comparable, rel <= 1e-12.
#include "cfd/five_eq/material_update.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::EOS;
using cfd::BC5;

#ifndef MATERIAL_REF
#  define MATERIAL_REF "material_update_ref.txt"
#endif

static const double P0 = 1.0e5;

struct Case {
    std::vector<double> a1, T1, T2, u, p;
    double dt, dx;
    cfd::MaterialConfig cfg;
};

static Case make_02A(const EOS& eos1, const EOS& eos2) {
    (void)eos1; (void)eos2;
    int n = 100;
    double dx = 1.0 / n;
    Case c;
    c.dt = 0.01; c.dx = dx;
    c.cfg = {1.0e-3, BC5::Periodic, BC5::Periodic};
    double af = 1.0e-3;
    for (int i = 0; i < n; ++i) {
        double x = (i + 0.5) * dx;
        c.a1.push_back((x >= 0.4 && x < 0.6) ? af : (1.0 - af));
        c.T1.push_back(300.0); c.T2.push_back(300.0);
        c.u.push_back(1.0); c.p.push_back(P0);
    }
    return c;
}

static Case make_07B(const EOS& eos1, const EOS& eos2) {
    int n = 40;
    double dx = 1.5 / n;
    Case c;
    c.dt = 0.4 * dx / 1600.0; c.dx = dx;
    c.cfg = {1.0e-8, BC5::Reflective, BC5::Transmissive};
    double apt = 1.0e-8;
    double T1v = eos1.temperature(1.157, eos1.energy(1.157, P0));
    double T2v = eos2.temperature(998.0, eos2.energy(998.0, P0));
    double amp = 1.157 * 347.8 * 0.02;
    for (int i = 0; i < n; ++i) {
        double x = (i + 0.5) / n;
        double g = std::exp(-((x - 0.25) * (x - 0.25)) / (2.0 * 0.05 * 0.05));
        double side = (x < 0.5) ? 1.0 : 0.0;
        c.a1.push_back(x < 0.5 ? (1.0 - apt) : apt);
        c.T1.push_back(T1v); c.T2.push_back(T2v);
        c.u.push_back(0.02 * g * side);
        c.p.push_back(P0 + amp * g * side);
    }
    return c;
}

int main() {
    EOS eos1 = EOS::ideal(1.4, 717.5);
    EOS eos2 = EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);

    Case c0 = make_02A(eos1, eos2);
    Case c1 = make_07B(eos1, eos2);
    cfd::MaterialResult r0 = cfd::material_update(
        c0.a1, c0.T1, c0.T2, c0.u, c0.p, c0.dt, c0.dx, eos1, eos2, c0.cfg);
    cfd::MaterialResult r1 = cfd::material_update(
        c1.a1, c1.T1, c1.T2, c1.u, c1.p, c1.dt, c1.dx, eos1, eos2, c1.cfg);

    std::ifstream fin(MATERIAL_REF);
    if (!fin) { std::printf("cannot open ref %s\n", MATERIAL_REF); return 1; }
    int fail = 0, nrows = 0;
    double max_rel = 0.0;
    std::string line;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double cs, idx, q1, q2, mm, re_new, re_adv, an;
        if (!(iss >> cs >> idx >> q1 >> q2 >> mm >> re_new >> re_adv >> an)) continue;
        const cfd::MaterialResult& r = (cs > 0.5) ? r1 : r0;
        int i = (int)idx;
        auto chk = [&](const char* fld, double got, double ref) {
            double denom = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
            double rel = std::fabs(got - ref) / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-12) {
                std::printf("  [FAIL] case%d cell %d %-9s got=%.17g ref=%.17g rel=%.3e\n",
                            (int)cs, i, fld, got, ref, rel);
                ++fail;
            }
        };
        chk("q1_new", r.q1_new[i], q1);
        chk("q2_new", r.q2_new[i], q2);
        chk("m_adv", r.m_adv[i], mm);
        chk("rhoE_new", r.rhoE_new[i], re_new);
        chk("rhoE_adv", r.rhoE_adv[i], re_adv);
        chk("alpha_new", r.alpha_new[i], an);
        ++nrows;
    }
    if (fail == 0) {
        std::printf("test_5eq_material_update: ALL PASS (%d cells, max_rel=%.3e)\n",
                    nrows, max_rel);
        return 0;
    }
    std::printf("test_5eq_material_update: %d FAILURES (max_rel=%.3e)\n", fail, max_rel);
    return 1;
}
