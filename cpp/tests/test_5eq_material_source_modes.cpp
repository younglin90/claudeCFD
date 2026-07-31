// Validate every Python Kapila source branch in material_update.
#include "cfd/five_eq/material_update.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::BC5;
using cfd::EOS;

#ifndef MATERIAL_SOURCE_MODES_REF
#  define MATERIAL_SOURCE_MODES_REF "material_source_modes_ref.txt"
#endif

struct Case {
    std::vector<double> a1, T1, T2, u, p;
    double dt, dx;
    BC5 left, right;
    double alpha_pure_tol;
};

static Case mixed_case(const EOS& eos1, const EOS& eos2) {
    const double p0 = 1.0e5;
    const double T1 = eos1.temperature(1.157, eos1.energy(1.157, p0));
    const double T2 = eos2.temperature(998.0, eos2.energy(998.0, p0));
    return {{.5,.5,.5,.5,.5}, {T1-.2,T1+.3,T1-.1,T1+.4,T1},
            {T2+.1,T2-.2,T2+.3,T2-.1,T2+.2}, {.05,.07,.04,.06,.05},
            {p0+40.,p0-25.,p0+70.,p0-35.,p0+20.}, .002, .1,
            BC5::Periodic, BC5::Periodic, 1.e-8};
}

static Case interface_case(const EOS& eos1, const EOS& eos2) {
    const double p0 = 1.0e5, apure = 1.e-8;
    const double T1 = eos1.temperature(1.157, eos1.energy(1.157, p0));
    const double T2 = eos2.temperature(998.0, eos2.energy(998.0, p0));
    Case c; c.dt = .4 * (1.5 / 40.) / 1600.; c.dx = 1.5 / 40.;
    c.left = BC5::Reflective; c.right = BC5::Transmissive; c.alpha_pure_tol = apure;
    for (int i = 0; i < 40; ++i) {
        const double x = (i + .5) / 40.;
        const double g = std::exp(-((x-.25)*(x-.25)) / (2.*.05*.05));
        const double side = x < .5 ? 1. : 0.;
        c.a1.push_back(x < .5 ? 1.-apure : apure); c.T1.push_back(T1); c.T2.push_back(T2);
        c.u.push_back(.02*g*side); c.p.push_back(p0 + 1.157*347.8*.02*g*side);
    }
    return c;
}

int main() {
    const EOS eos1 = EOS::ideal(1.4, 717.5);
    const EOS eos2 = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    const Case cases[] = {mixed_case(eos1, eos2), interface_case(eos1, eos2)};
    cfd::MaterialResult got[8][2];
    for (int mode = 0; mode < 8; ++mode) for (int cs = 0; cs < 2; ++cs) {
        const Case& c = cases[cs];
        cfd::MaterialConfig cfg{c.alpha_pure_tol, c.left, c.right};
        cfg.kapila_closure = mode != 0;
        if (mode) cfg.kapila_source_mode = static_cast<cfd::KapilaSourceMode>(mode - 1);
        got[mode][cs] = cfd::material_update(c.a1, c.T1, c.T2, c.u, c.p,
                                               c.dt, c.dx, eos1, eos2, cfg);
    }
    std::ifstream in(MATERIAL_SOURCE_MODES_REF);
    if (!in) { std::printf("cannot open ref %s\n", MATERIAL_SOURCE_MODES_REF); return 1; }
    int fail = 0, rows = 0; double max_rel = 0.; std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream s(line); int mode, cs, i;
        double q1, q2, m, re, rea, a;
        if (!(s >> mode >> cs >> i >> q1 >> q2 >> m >> re >> rea >> a)) continue;
        const auto& r = got[mode][cs];
        auto check = [&](const char* name, double value, double ref) {
            const double rel = std::fabs(value-ref) / (std::fabs(ref) > 1.e-300 ? std::fabs(ref) : 1.);
            if (rel > max_rel) max_rel = rel;
            if (rel > 1.e-12) { std::printf("[FAIL] mode%d case%d cell%d %s %.17g %.17g %.3e\n", mode, cs, i, name, value, ref, rel); ++fail; }
        };
        check("q1", r.q1_new[i], q1); check("q2", r.q2_new[i], q2); check("m", r.m_adv[i], m);
        check("rhoE", r.rhoE_new[i], re); check("rhoE_adv", r.rhoE_adv[i], rea); check("alpha", r.alpha_new[i], a);
        ++rows;
    }
    std::printf("test_5eq_material_source_modes: %s (%d cells, max_rel=%.3e)\n", fail ? "FAIL" : "ALL PASS", rows, max_rel);
    return fail ? 1 : 0;
}
