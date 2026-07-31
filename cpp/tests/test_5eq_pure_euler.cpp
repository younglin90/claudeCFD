#include "cfd/five_eq/step.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifndef PURE_EULER_REF
#define PURE_EULER_REF "pure_euler_ref.txt"
#endif

int main() {
    using cfd::EOS;
    using cfd::BC5;
    const EOS eos=EOS::ideal(1.4,717.5);
    const std::vector<double> a(5,.999), T(5,300.), u{.12,.18,.15,.10,.14}, p{1e5,100005.,99996.,100003.,1e5};
    cfd::five_eq::StepConfig cfg; cfg.alpha_pure_tol=.01; cfg.bc_l=BC5::Periodic; cfg.bc_r=BC5::Periodic;
    const auto hlle=cfd::five_eq::imex_ad_step(a,T,T,u,p,2e-5,.1,eos,eos,cfg);
    cfg.pure_euler_flux=cfd::five_eq::PureEulerFlux::Hllc;
    const auto hllc=cfd::five_eq::imex_ad_step(a,T,T,u,p,2e-5,.1,eos,eos,cfg);
    const std::vector<double> ul{.25,.08,-.04,.06,.18}, pl{1.20e5,1.05e5,.95e5,1.00e5,1.10e5};
    cfg.pure_euler_flux=cfd::five_eq::PureEulerFlux::Hlle;
    const auto led_hlle=cfd::five_eq::imex_ad_step(a,T,T,ul,pl,2e-5,.1,eos,eos,cfg);
    cfg.pure_euler_flux=cfd::five_eq::PureEulerFlux::Hllc;
    const auto led_hllc=cfd::five_eq::imex_ad_step(a,T,T,ul,pl,2e-5,.1,eos,eos,cfg);
    cfg.pure_euler_flux=cfd::five_eq::PureEulerFlux::Hlle;
    cfg.pure_euler_characteristic_reconstruction=true;
    const auto characteristic=cfd::five_eq::imex_ad_step(a,T,T,u,p,2e-5,.1,eos,eos,cfg);
    cfg.pure_euler_characteristic_reconstruction=false;
    cfg.pure_euler_hancock=false;
    const auto no_hancock=cfd::five_eq::imex_ad_step(a,T,T,u,p,2e-5,.1,eos,eos,cfg);
    std::ifstream in(PURE_EULER_REF); if(!in) return 1; std::string line; int rows=0,fail=0; double worst=0.;
    while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;std::istringstream s(line);int kind,i;double ref[5];if(!(s>>kind>>i>>ref[0]>>ref[1]>>ref[2]>>ref[3]>>ref[4]))continue;const auto& got=kind==0?hlle:(kind==1?hllc:(kind==2?led_hlle:(kind==3?led_hllc:(kind==4?characteristic:no_hancock))));double val[5]={got.alpha[i],got.T1[i],got.T2[i],got.u[i],got.p[i]};for(int k=0;k<5;++k){double rel=std::fabs(val[k]-ref[k])/std::fmax(std::fabs(ref[k]),1.);worst=std::fmax(worst,rel);if(rel>2e-11){++fail;std::printf("kind=%d cell=%d field=%d rel=%.3e\n",kind,i,k,rel);}}++rows;}
    std::printf("test_5eq_pure_euler: %s (%d rows, max_rel=%.3e)\n",fail?"FAIL":"PASS",rows,worst);return fail?1:0;
}
