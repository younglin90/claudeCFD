#include "cfd/five_eq/face_state.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace cfd; using namespace cfd::five_eq;
int main(){std::ifstream in(LOW_FLUX_REF);if(!in)return 1;std::string l;double worst=0.;const auto a=EOS::ideal(1.4,717.5),b=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);double t1=a.temperature(1.157,a.energy(1.157,1e5)),t2=b.temperature(998.,b.energy(998.,1e5));StepResult W{{.2,.5,.8},{t1,t1+.2,t1-.1},{t2-.2,t2+.3,t2},{.03,-.01,.02},{100030.,99980.,100010.}};auto f=acid_face_state(W,a,b,BC5::Periodic,BC5::Periodic);auto r=rusanov_advective_fluxes(f,a,b),p=pe_preserving_low_flux(f);while(std::getline(in,l)){if(l.empty()||l[0]=='#')continue;std::istringstream s(l);int k,i;double x[5];s>>k>>i>>x[0]>>x[1]>>x[2]>>x[3]>>x[4];auto&q=k?p:r;double y[5]={q.m1[i],q.m2[i],q.mom[i],q.rhoE[i],q.alpha[i]};for(int j=0;j<5;++j)worst=std::fmax(worst,std::fabs(y[j]-x[j])/std::fmax(std::fabs(x[j]),1.));}std::printf("low flux max %.3e\n",worst);return worst<=2e-12?0:2;}
