#include "cfd/five_eq/source_terms.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifndef SOURCE_TERMS_REF
#  define SOURCE_TERMS_REF "source_terms_ref.txt"
#endif

using cfd::EOS;
using cfd::five_eq::HeatConductionConfig;
using cfd::five_eq::PhaseChangeConfig;
using cfd::five_eq::StepResult;

int main() {
    const EOS air = EOS::ideal(1.4, 717.5);
    const EOS water = EOS::nasg(1.187, 7.028e8, 3610., 6.61e-4, -1.177788e6);
    const StepResult W{{.7,.5,.2}, {380.,390.,400.}, {360.,370.,380.},
                       {2.,-.5,.25}, {9.99e4,1.001e5,1.0005e5}};
    PhaseChangeConfig phase; phase.enabled = true; phase.tau = 100.; phase.T_sat = 373.15;
    phase.p_sat = 1.e5; phase.latent_heat = 2.257e6;
    PhaseChangeConfig phase_iso = phase; phase_iso.isothermal = true; phase_iso.saturation_pressure_target = true;
    HeatConductionConfig heat; heat.enabled = true; heat.k_liquid = .6; heat.k_vapor = .025;
    heat.T_left = 350.; heat.T_right = 410.;
    HeatConductionConfig heat_primitive = heat; heat_primitive.primitive_temperature = true;
    StepResult got[7];
    got[0] = cfd::five_eq::apply_source_terms(W,.002,.1,air,water,-9.81,{},{});
    got[1] = cfd::five_eq::apply_source_terms(W,.002,.1,air,water,0.,phase,{});
    got[2] = cfd::five_eq::apply_source_terms(W,.002,.1,air,water,0.,phase_iso,{});
    got[3] = cfd::five_eq::apply_source_terms(W,.002,.1,air,water,0.,{},heat);
    got[4] = cfd::five_eq::apply_source_terms(W,.002,.1,air,water,0.,{},heat_primitive);
    got[5] = cfd::five_eq::apply_source_terms(W,.002,.1,air,water,-9.81,phase_iso,{});
    got[6] = cfd::five_eq::apply_source_terms(W,.002,.1,air,water,-9.81,phase_iso,heat_primitive);
    std::ifstream in(SOURCE_TERMS_REF);
    if (!in) { std::printf("cannot open ref %s\n", SOURCE_TERMS_REF); return 1; }
    int fail=0, rows=0; double worst=0.; std::string line;
    while (std::getline(in,line)) {
        const auto first=line.find_first_not_of(" \t\r\n");
        if (first==std::string::npos || line[first]=='#') continue;
        std::istringstream s(line); int kind,i; double a,t1,t2,u,p;
        if (!(s>>kind>>i>>a>>t1>>t2>>u>>p)) continue;
        const auto check=[&](const char* name,double value,double ref) {
            const double rel=std::fabs(value-ref)/(std::fabs(ref)>1.e-300?std::fabs(ref):1.);
            worst=std::fmax(worst,rel);
            // Python performs this 3x3 recovery vectorised across the full
            // state; the scalar C++ Newton reaches the same source branch
            // within its 1e-9 residual tolerance, with a 2.5e-7 state delta.
            if(rel>1.e-6){std::printf("[FAIL] kind%d cell%d %s %.17g %.17g %.3e\n",kind,i,name,value,ref,rel);++fail;}
        };
        check("alpha",got[kind].alpha[i],a); check("T1",got[kind].T1[i],t1);
        check("T2",got[kind].T2[i],t2); check("u",got[kind].u[i],u); check("p",got[kind].p[i],p); ++rows;
    }
    std::printf("test_5eq_source_terms_oracle: %s (%d cells, max_rel=%.3e)\n",fail?"FAIL":"ALL PASS",rows,worst);
    return fail?1:0;
}
