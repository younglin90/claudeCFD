#include "cfd/five_eq/nd_transport.hpp"
#include <cmath>
#include <cstdio>
int main(){ using namespace cfd::five_eq; AlphaGrid2 a{8,6,std::vector<double>(48)}; for(int i=0;i<8;++i)for(int j=0;j<6;++j)a.value[alpha_idx(i,j,6)]=(i<4?.2:.8); double mass=0.;for(double x:a.value)mass+=x;
 auto vel=[](double){FaceVelocity2 f;f.ux.assign(9*6,.25);f.uy.assign(8*7,-.1);return f;}; AlphaTransportInfo info;auto b=solve_alpha_transport_2d(a,.125,1./6.,.1,vel,.4,-1.,true,0.,1.,&info); double sum=0.;for(double x:b.value){if(!std::isfinite(x)||x<0.||x>1.)return 1;sum+=x;} if(std::fabs(sum-mass)>1.e-10||info.steps<1)return 2; std::puts("nd transport passed");}
