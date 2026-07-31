#include "cfd/five_eq/schur_blocks.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
using namespace cfd; using namespace cfd::five_eq;
int main(){std::ifstream in(SCHUR_REF);std::string line;while(std::getline(in,line)&&(line.empty()||line[0]=='#')){}std::istringstream q(line);double e[5];for(double&x:e)q>>x;auto a=EOS::ideal(1.4,717.5),b=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);const double T1=(1e5/((1.4-1)*717.5))/1.157+1.;const double r2=998.;const double e2=b.energy(r2,1e5);const double T2=b.temperature(r2,e2)-2.;auto s=schur_blocks({.31,T1,T2,.04,100025.},a,b);double g[5]={s.uu,s.up,s.pu,s.pp,s.sigma_pp},w=0.;for(int i=0;i<5;++i)w=std::fmax(w,std::fabs(g[i]-e[i])/std::fmax(std::fabs(e[i]),1.));std::printf("schur %.3e\n",w);return w<=1e-7?0:1;}
