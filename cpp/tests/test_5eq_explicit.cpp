#include "cfd/five_eq/explicit.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef EXPLICIT_REF
#define EXPLICIT_REF "explicit_step_ref.txt"
#endif

int main() {
    std::ifstream in(EXPLICIT_REF);
    if (!in) return 1;
    std::vector<double> a, T1, T2, u, p, ar, T1r, T2r, ur, pr;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line); int i;
        double x;
        if (!(s >> i >> x)) return 2; a.push_back(x);
        s >> x; T1.push_back(x); s >> x; T2.push_back(x); s >> x; u.push_back(x); s >> x; p.push_back(x);
        s >> x; ar.push_back(x); s >> x; T1r.push_back(x); s >> x; T2r.push_back(x); s >> x; ur.push_back(x); s >> x; pr.push_back(x);
    }
    cfd::EOS e1 = cfd::EOS::ideal(1.4, 717.5);
    cfd::EOS e2 = cfd::EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    cfd::five_eq::StepConfig cfg;
    cfg.alpha_pure_tol = 1.e-8; cfg.bc_l = cfd::BC5::Inlet; cfg.bc_r = cfd::BC5::Dirichlet;
    cfg.u_inlet_l = .25; cfg.p_inlet_l = 100320.; cfg.p_outlet_r = 99750.;
    cfg.alpha_inlet_l = .97; cfg.T1_inlet_l = T1[0]; cfg.T2_inlet_l = T2[0];
    const auto got = cfd::five_eq::explicit_rusanov_step(a,T1,T2,u,p,1.e-7,.125,e1,e2,cfg);
    double worst = 0.0;
    for (size_t i=0;i<a.size();++i) for (auto pair : {std::pair{got.alpha[i],ar[i]}, {got.T1[i],T1r[i]}, {got.T2[i],T2r[i]}, {got.u[i],ur[i]}, {got.p[i],pr[i]}}) {
        worst = std::fmax(worst, std::fabs(pair.first-pair.second)/std::fmax(std::fabs(pair.second),1.0));
    }
    std::printf("explicit oracle relative max %.3e\n", worst);
    return worst <= 2.e-11 ? 0 : 3;
}
