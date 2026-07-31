#include "cfd/five_eq/positivity.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
using namespace cfd; using namespace cfd::five_eq;
int main(){std::ifstream in(POSITIVITY_REF);if(!in)return 1;std::vector<double> ref;std::string l;while(std::getline(in,l)){if(l.empty()||l[0]=='#')continue;std::istringstream s(l);int i;double x;s>>i>>x;ref.push_back(x);}AdvectiveFlux5 h{{0,2,-2,0},{0,1.8,-1.8,0},{0,.2,-.2,0},{0,3,-3,0},{0,1.2,-1.2,0}},z;z.m1=z.m2=z.mom=z.rhoE=z.alpha=std::vector<double>(4);std::vector<ConsU>u(3,{.2,.2,0,1,.5});auto got=positivity_blend_theta(h,z,u,1.,.2);double w=0;for(size_t i=0;i<ref.size();++i)w=std::fmax(w,std::fabs(got[i]-ref[i]));std::printf("positivity theta max %.3e\n",w);return w<=1e-15?0:2;}
