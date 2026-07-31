#include "cfd/five_eq/material_update.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef MATERIAL_OPTIONS_REF
#define MATERIAL_OPTIONS_REF "material_options_ref.txt"
#endif

int main() {
    using cfd::BC5;
    using cfd::EOS;
    const EOS eos1 = EOS::ideal(1.4, 717.5);
    const EOS eos2 = EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    const double p0 = 1.e5;
    const double t1 = eos1.temperature(1.157, eos1.energy(1.157, p0));
    const double t2 = eos2.temperature(998.0, eos2.energy(998.0, p0));
    const std::vector<double> a(5, .5), T1{t1-.2,t1+.3,t1-.1,t1+.4,t1},
        T2{t2+.1,t2-.2,t2+.3,t2-.1,t2+.2}, u{.05,.07,.04,.06,.05},
        p{p0+40.,p0-25.,p0+70.,p0-35.,p0+20.};
    cfd::MaterialConfig hllc{1.e-8, BC5::Periodic, BC5::Periodic};
    hllc.material_flux = cfd::MaterialFlux::HllcContact;
    cfd::MaterialConfig characteristic{1.e-8, BC5::Periodic, BC5::Periodic};
    characteristic.characteristic_reconstruction = true;
    const auto h = cfd::material_update(a,T1,T2,u,p,.002,.1,eos1,eos2,hllc);
    const auto c = cfd::material_update(a,T1,T2,u,p,.002,.1,eos1,eos2,characteristic);

    std::ifstream in(MATERIAL_OPTIONS_REF);
    if (!in) { std::printf("cannot open %s\n", MATERIAL_OPTIONS_REF); return 1; }
    int rows = 0, fail = 0;
    double max_rel = 0.0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        int kind, i; double q1,q2,m,re,rea,an;
        if (!(s >> kind >> i >> q1 >> q2 >> m >> re >> rea >> an)) continue;
        const auto& r = kind == 0 ? h : c;
        const double got[] = {r.q1_new[i],r.q2_new[i],r.m_adv[i],r.rhoE_new[i],r.rhoE_adv[i],r.alpha_new[i]};
        const double ref[] = {q1,q2,m,re,rea,an};
        for (int k=0;k<6;++k) {
            const double rel = std::fabs(got[k]-ref[k]) / (std::fabs(ref[k]) > 1.e-300 ? std::fabs(ref[k]) : 1.0);
            max_rel = std::fmax(max_rel, rel);
            if (rel > 1.e-11) { ++fail; std::printf("kind=%d cell=%d field=%d rel=%.3e\n",kind,i,k,rel); }
        }
        ++rows;
    }
    std::printf("test_5eq_material_options: %s (%d rows, max_rel=%.3e)\n",
                fail ? "FAIL" : "PASS", rows, max_rel);
    return fail ? 1 : 0;
}
