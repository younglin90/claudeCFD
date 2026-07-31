// cfd/five_eq/ars_residual.hpp -- ARS stage residual, port of residual.py::residual.
#pragma once
#include "cfd/five_eq/implicit_faces.hpp"
namespace cfd::five_eq {
struct Residual5 { std::vector<double> m1,m2,mom,rhoE,alpha; ImplicitDivergences implicit; };
inline Residual5 ars_stage_residual(const StepResult& W, const std::vector<ConsU>& target,
                                   double gamma_dt, const EOS& e1,const EOS& e2,double dx,BC5 l,BC5 r,
                                   double dissipation=.0,bool biharmonic=true,
                                   const std::vector<double>* kapila_implicit=nullptr,
                                   bool rhie_chow=false,
                                   bool acoustic_riemann=false,
                                   bool upwind_dissipation=false,
                                   double compact_lap_coeff=0.0) {
 const int n=(int)W.alpha.size(); Residual5 R; R.m1.resize(n);R.m2.resize(n);R.mom.resize(n);R.rhoE.resize(n);R.alpha.resize(n);
 R.implicit=implicit_divergences(
     W,e1,e2,dx,l,r,dissipation,biharmonic,rhie_chow,gamma_dt,
     acoustic_riemann,upwind_dissipation,compact_lap_coeff);
 for(int i=0;i<n;++i){ const ConsU u=prim_to_cons_W(PrimW{W.alpha[i],W.T1[i],W.T2[i],W.u[i],W.p[i]},e1,e2);
  R.m1[i]=(u.m1-target[i].m1)/gamma_dt; R.m2[i]=(u.m2-target[i].m2)/gamma_dt;
  R.mom[i]=(u.mom-target[i].mom)/gamma_dt+R.implicit.grad_p[i];
  R.rhoE[i]=(u.rhoE-target[i].rhoE)/gamma_dt+R.implicit.div_pu[i];
  R.alpha[i]=(u.a1-target[i].a1)/gamma_dt-(kapila_implicit?(*kapila_implicit)[i]:0.0)*R.implicit.div_u[i]; }
 return R;
}
} // namespace cfd::five_eq
