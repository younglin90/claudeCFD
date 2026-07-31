#include "cfd/five_eq/ars_solver.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
int main() {
 using namespace cfd; using namespace cfd::five_eq;
 const auto air=EOS::ideal(1.4,717.5), water=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
 StepResult W; W.alpha={.5,.5,.5,.5}; W.T1={300,300,300,300}; W.T2={300,300,300,300}; W.u={0,0,0,0}; W.p={1e5,1e5,1e5,1e5};
 StepConfig c; c.bc_l=BC5::Periodic; c.bc_r=BC5::Periodic;
 NewtonInfo a,b; const auto U=ars222_step(W,1.e-5,.1,air,water,c,&a,&b);
 for(size_t i=0;i<W.alpha.size();++i) if(std::fabs(U.p[i]-W.p[i])>1.e-6 || std::fabs(U.u[i])>1.e-10) return 1;
 W.p[1]=100100.; const auto V=ars222_step(W,1.e-7,.1,air,water,c,&a,&b);
 for(double x:V.p) if(!std::isfinite(x)||x<=1.) return 2;
 std::ifstream in(ARS_REF); if(!in) return 3;
 std::vector<double> aa,t1,t2,uu,pp,ar,t1r,t2r,ur,pr; std::string line;
 while(std::getline(in,line)) { if(line.empty()||line[0]=='#') continue; std::istringstream s(line); int i; double x;
   s>>i>>x; aa.push_back(x); s>>x; t1.push_back(x); s>>x; t2.push_back(x); s>>x; uu.push_back(x); s>>x; pp.push_back(x);
   s>>x; ar.push_back(x); s>>x; t1r.push_back(x); s>>x; t2r.push_back(x); s>>x; ur.push_back(x); s>>x; pr.push_back(x); }
 StepResult O{aa,t1,t2,uu,pp}; const auto Q=ars222_step(O,1.e-8,.1,air,water,c,&a,&b); double worst=0.;
 for(size_t i=0;i<aa.size();++i) for(auto z: {std::pair{Q.alpha[i],ar[i]},std::pair{Q.T1[i],t1r[i]},std::pair{Q.T2[i],t2r[i]},std::pair{Q.u[i],ur[i]},std::pair{Q.p[i],pr[i]}})
   worst=std::fmax(worst,std::fabs(z.first-z.second)/std::fmax(std::fabs(z.second),1.));
 std::printf("ars222 oracle relative max %.3e; Newton=(%d,%.3e) (%d,%.3e)\\n",worst,a.iterations,a.norm,b.iterations,b.norm); if(worst>2.e-5) return 4;
 StepConfig schur=c; schur.ars_linear_solver=ARSLinearSolver::schur_helmholtz;
 NewtonInfo sa,sb; const auto S=ars222_step(O,1.e-8,.1,air,water,schur,&sa,&sb); double schur_worst=0.;
 for(size_t i=0;i<aa.size();++i) for(auto z: {std::pair{S.alpha[i],Q.alpha[i]},std::pair{S.T1[i],Q.T1[i]},std::pair{S.T2[i],Q.T2[i]},std::pair{S.u[i],Q.u[i]},std::pair{S.p[i],Q.p[i]}})
   schur_worst=std::fmax(schur_worst,std::fabs(z.first-z.second)/std::fmax(std::fabs(z.second),1.));
 std::printf("ars222 schur relative to dense %.3e; Newton=(%d,%d) (%d,%d)\\n",schur_worst,sa.converged?1:0,sa.iterations,sb.converged?1:0,sb.iterations);
 if(!sa.converged || !sb.converged || schur_worst>2.e-4) return 5;
 StepConfig upwind=c; upwind.ars_upwind_dissipation=true; upwind.ars_implicit_dissipation=.2;
 NewtonInfo ua,ub; const auto UW=ars222_step(O,1.e-8,.1,air,water,upwind,&ua,&ub);
 for(double value:UW.p) if(!std::isfinite(value)||value<=1.) return 6;
 if(!ua.converged || !ub.converged) return 7;
 StepConfig wall=c; wall.bc_l=BC5::Reflective; wall.bc_r=BC5::Transmissive;
 NewtonInfo da,db,fa,fb; const auto D=ars222_step(O,1.e-8,.1,air,water,wall,&da,&db);
 wall.ars_linear_solver=ARSLinearSolver::schur_helmholtz;
 const auto F=ars222_step(O,1.e-8,.1,air,water,wall,&fa,&fb); double fallback_worst=0.;
 for(size_t i=0;i<aa.size();++i) for(auto z: {std::pair{D.alpha[i],F.alpha[i]},std::pair{D.T1[i],F.T1[i]},std::pair{D.T2[i],F.T2[i]},std::pair{D.u[i],F.u[i]},std::pair{D.p[i],F.p[i]}})
   fallback_worst=std::fmax(fallback_worst,std::fabs(z.first-z.second)/std::fmax(std::fabs(z.second),1.));
 if(!da.converged || !db.converged || !fa.converged || !fb.converged || fallback_worst>1.e-14) return 8;
 StepConfig limited=c; limited.ars_explicit_positivity=true;
 NewtonInfo la,lb; const auto L=ars222_step(O,1.e-8,.1,air,water,limited,&la,&lb);
 for(double value:L.p) if(!std::isfinite(value)||value<=1.) return 9;
 if(!la.converged || !lb.converged) return 10;
 std::puts("ars222 passed"); return 0;
}
