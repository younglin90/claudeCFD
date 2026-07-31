#include "cfd/five_eq/ars_residual.hpp"
#include <cmath>
#include <cstdio>
int main(){
 cfd::five_eq::StepResult W; W.alpha={.9,.9,.1,.1}; W.T1={300,300,300,300}; W.T2=W.T1; W.u={1,1,1,1}; W.p={1e5,1e5,1e5,1e5};
 auto e1=cfd::EOS::ideal(1.4,717.5), e2=cfd::EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
 auto d=cfd::five_eq::implicit_divergences(W,e1,e2,.1,cfd::BC5::Periodic,cfd::BC5::Periodic,.02,true);
 for(double x:d.grad_p) if(!std::isfinite(x)||std::fabs(x)>1e-8) return 1;
 for(double x:d.div_u) if(!std::isfinite(x)||std::fabs(x)>1e-8) return 2;
 std::vector<cfd::ConsU> target;
 for(size_t i=0;i<W.alpha.size();++i) target.push_back(cfd::prim_to_cons_W({W.alpha[i],W.T1[i],W.T2[i],W.u[i],W.p[i]},e1,e2));
 auto R=cfd::five_eq::ars_stage_residual(W,target,.01,e1,e2,.1,cfd::BC5::Periodic,cfd::BC5::Periodic,.02,true);
 for(double x:R.mom) if(std::fabs(x)>1e-8) return 4;
 W.u[1]=2.; auto f=cfd::five_eq::implicit_face_pu(W,e1,e2,cfd::BC5::Periodic,cfd::BC5::Periodic,.02,true);
 for(double x:f.p) if(!std::isfinite(x)) return 3;
 std::puts("implicit faces passed"); return 0;
}
