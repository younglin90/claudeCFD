#include "cfd/five_eq/pe_correction.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace cfd; using namespace cfd::five_eq;
int main(){std::ifstream in(PE_MODES_REF);if(!in)return 1;const auto a=EOS::ideal(1.4,717.5),b=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);double t1=a.temperature(1.157,a.energy(1.157,1e5)),t2=b.temperature(998.,b.energy(998.,1e5));StepResult W{{.08,.72,.23,.91,.44},{t1,t1+.2,t1-.1,t1+.4,t1-.3},{t2-.2,t2+.3,t2+.1,t2-.4,t2+.2},{0,0,0,0,0},{1e5,1e5,1e5,1e5,1e5}};Residual5 raw{{.3,-.1,.2,-.25,.15},{-.2,.15,-.05,.1,-.12},{.4,-.3,.1,.2,-.35},{50,-30,20,-40,35},{.01,-.02,.03,-.015,.025},{}};std::string l;double worst=0.;while(std::getline(in,l)){if(l.empty()||l[0]=='#')continue;std::istringstream s(l);int mode,i;double ref[6];s>>mode>>i;for(double&x:ref)s>>x;Residual5 q=raw;std::vector<double> normal;apply_pe_tangent_projection(q,W,a,b,&normal,static_cast<PEProjectionMode>(mode));double got[6]={normal[i],q.m1[i],q.m2[i],q.mom[i],q.rhoE[i],q.alpha[i]};for(int k=0;k<6;++k)worst=std::fmax(worst,std::fabs(got[k]-ref[k])/std::fmax(std::fabs(ref[k]),1.));}std::printf("pe modes max %.3e\n",worst);return worst<=4e-7?0:2;}
