#include "cfd/five_eq/helmholtz.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace cfd::five_eq;
int main(){std::ifstream in(HELMHOLTZ_REF);if(!in)return 1;std::vector<double>s,r,b,d,x;std::string line;while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;std::istringstream q(line);int i;double si,ri,bi,di,xi;q>>i>>si>>ri>>bi>>di>>xi;s.push_back(si);r.push_back(ri);b.push_back(bi);d.push_back(di);x.push_back(xi);}const auto A=assemble_helmholtz_periodic(s,r,.05,.1);const auto got=solve_helmholtz_periodic(s,r,.05,.1,b);double worst=0.;for(size_t i=0;i<s.size();++i){worst=std::fmax(worst,std::fabs(A.diagonal[i]-d[i])/std::fmax(std::fabs(d[i]),1.));worst=std::fmax(worst,std::fabs(got[i]-x[i])/std::fmax(std::fabs(x[i]),1.));}std::printf("helmholtz oracle max %.3e\n",worst);return worst<=1.e-12?0:2;}
