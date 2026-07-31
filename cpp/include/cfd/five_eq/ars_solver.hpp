// Dense-Newton ARS(2,2,2) integrator.  This is the correctness reference for
// the C++ ARS path; large grids can later replace its linear solve by Schur.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "cfd/five_eq/ars_residual.hpp"
#include "cfd/five_eq/ars_schur.hpp"
#include "cfd/five_eq/face_state.hpp"
#include "cfd/five_eq/relaxation.hpp"

namespace cfd::five_eq {
constexpr double ARS_GAMMA = 1.0 - 1.0 / 1.4142135623730950488;

struct NewtonInfo { bool converged=false; int iterations=0; double norm=0.; };

inline std::vector<ConsU> conservative_cells(const StepResult& W, const EOS& e1, const EOS& e2) {
    std::vector<ConsU> U(W.alpha.size());
    for(std::size_t i=0;i<U.size();++i) U[i]=prim_to_cons_W({W.alpha[i],W.T1[i],W.T2[i],W.u[i],W.p[i]},e1,e2);
    return U;
}
inline StepResult primitive_cells(const std::vector<ConsU>& U, const StepResult& guess,
                                  const EOS& e1, const EOS& e2) {
    StepResult W; const int n=static_cast<int>(U.size());
    W.alpha.resize(n); W.T1.resize(n); W.T2.resize(n); W.u.resize(n); W.p.resize(n);
    for(int i=0;i<n;++i) { const auto w=cons_to_prim_W(U[i],e1,e2,1.e-13,50,guess.T1[i],guess.T2[i]);
        W.alpha[i]=w.alpha1; W.T1[i]=w.T1; W.T2[i]=w.T2; W.u[i]=w.u; W.p[i]=w.p; }
    return W;
}
inline std::vector<double> flatten_residual(const Residual5& R) {
    const int n=static_cast<int>(R.m1.size()); std::vector<double> out(5*n);
    for(int i=0;i<n;++i) { out[5*i]=R.m1[i];out[5*i+1]=R.m2[i];out[5*i+2]=R.mom[i];out[5*i+3]=R.rhoE[i];out[5*i+4]=R.alpha[i]; }
    return out;
}
inline double residual_norm(const std::vector<double>& r, const std::vector<double>& scale) {
    double m=0.; for(std::size_t i=0;i<r.size();++i) m=std::fmax(m,std::fabs(r[i])/scale[i%5]); return m;
}
inline std::vector<double> residual_scales(const StepResult& W, const EOS& e1, const EOS& e2) {
    std::vector<double> s(5,1.); for(const auto& u:conservative_cells(W,e1,e2)) {
        s[0]=std::fmax(s[0],std::fabs(u.m1)); s[1]=std::fmax(s[1],std::fabs(u.m2));
        s[2]=std::fmax(s[2],std::fabs(u.mom)); s[3]=std::fmax(s[3],std::fabs(u.rhoE)); }
    return s;
}
inline bool admissible(const StepResult& W) {
    for(std::size_t i=0;i<W.alpha.size();++i) if(!(W.alpha[i]>1.e-12 && W.alpha[i]<1.-1.e-12 && W.T1[i]>1. && W.T2[i]>1. && W.p[i]>1.)) return false;
    return true;
}
inline bool dense_solve(std::vector<double> A, std::vector<double>& b, int n) {
    for(int c=0;c<n;++c) { int p=c; for(int i=c+1;i<n;++i) if(std::fabs(A[i*n+c])>std::fabs(A[p*n+c])) p=i;
        if(std::fabs(A[p*n+c])<1.e-30) return false;
        if(p!=c) { for(int j=c;j<n;++j) std::swap(A[c*n+j],A[p*n+j]); std::swap(b[c],b[p]); }
        for(int i=c+1;i<n;++i) { const double q=A[i*n+c]/A[c*n+c]; A[i*n+c]=0.; for(int j=c+1;j<n;++j) A[i*n+j]-=q*A[c*n+j]; b[i]-=q*b[c]; }
    }
    for(int i=n-1;i>=0;--i) { for(int j=i+1;j<n;++j) b[i]-=A[i*n+j]*b[j]; b[i]/=A[i*n+i]; }
    return true;
}
inline void add_primitive_delta(StepResult& W, const std::vector<double>& d, double lambda) {
    for(std::size_t i=0;i<W.alpha.size();++i) { W.alpha[i]+=lambda*d[5*i]; W.T1[i]+=lambda*d[5*i+1]; W.T2[i]+=lambda*d[5*i+2]; W.u[i]+=lambda*d[5*i+3]; W.p[i]+=lambda*d[5*i+4]; }
}

inline StepResult ars_newton_solve(const StepResult& initial, const std::vector<ConsU>& target,
                                   double gamma_dt, const EOS& e1, const EOS& e2, double dx, BC5 l, BC5 r,
                                   NewtonInfo& info, int max_iter=10, double rtol=1.e-6, double atol=1.e-10,
                                   double dissipation=1., bool biharmonic=true,
                                   bool rhie_chow=false, bool acoustic_riemann=false,
                                   bool upwind_dissipation=false,
                                   double compact_lap_coeff=0.0,
                                   const std::vector<double>* kapila_implicit=nullptr,
                                   int line_search_max=8, double eta=1.e-4) {
    StepResult W=initial; const int n=static_cast<int>(W.alpha.size()), m=5*n;
    const auto scale=residual_scales(W,e1,e2);
    auto eval=[&](const StepResult& x) { return flatten_residual(
        ars_stage_residual(x,target,gamma_dt,e1,e2,dx,l,r,dissipation,biharmonic,
                           kapila_implicit,rhie_chow,acoustic_riemann,upwind_dissipation,
                           compact_lap_coeff)); };
    std::vector<double> R=eval(W); double norm=residual_norm(R,scale), norm0=std::fmax(norm,atol);
    if(norm==0. || norm<=atol+rtol*norm0) { info={true,0,norm}; return W; }
    for(int it=0;it<max_iter;++it) {
        std::vector<double> J(m*m), minusR(m); for(int q=0;q<m;++q) minusR[q]=-R[q];
        for(int c=0;c<m;++c) {
            StepResult P=W; const int i=c/5,k=c%5; double* x=k==0?&P.alpha[i]:k==1?&P.T1[i]:k==2?&P.T2[i]:k==3?&P.u[i]:&P.p[i];
            const double h=k==0?1.e-6:std::fmax(std::fabs(*x)*1.e-6,1.e-6); *x+=h;
            const auto Rp=eval(P); for(int row=0;row<m;++row) J[row*m+c]=(Rp[row]-R[row])/h;
        }
        const double diag_floor=1.e-12; for(int i=0;i<m;++i) J[i*m+i]+=diag_floor;
        if(!dense_solve(std::move(J),minusR,m)) { info={false,it,norm}; return W; }
        bool accepted=false; double lambda=1.;
        for(int ls=0;ls<line_search_max;++ls) { StepResult T=W; add_primitive_delta(T,minusR,lambda);
            if(admissible(T)) { const auto Rt=eval(T); const double nt=residual_norm(Rt,scale);
                if(nt<=(1.-eta*lambda)*norm) { W=std::move(T); R=Rt; norm=nt; accepted=true; break; } }
            lambda*=.5;
        }
        if(!accepted) { info={false,it,norm}; return W; }
        if(norm<=atol+rtol*norm0) { info={true,it+1,norm}; return W; }
    }
    info={false,max_iter,norm}; return W;
}

inline StepResult ars_schur_solve(const StepResult& initial, const std::vector<ConsU>& target,
                                  double gamma_dt, const EOS& e1, const EOS& e2, double dx,
                                  BC5 l, BC5 r, NewtonInfo& info, int max_iter=10,
                                  double rtol=1.e-6, double atol=1.e-10,
                                  double dissipation=1., int line_search_max=8,
                                  double eta=1.e-4) {
    StepResult W=initial;
    if (l != BC5::Periodic || r != BC5::Periodic)
        return ars_newton_solve(initial,target,gamma_dt,e1,e2,dx,l,r,info,
                                max_iter,rtol,atol,dissipation,true,false,false,
                                false,0.0,nullptr,line_search_max,eta);
    const auto scale=residual_scales(W,e1,e2);
    auto eval=[&](const StepResult& x) {
        return flatten_residual(ars_stage_residual(
            x,target,gamma_dt,e1,e2,dx,l,r,dissipation,true));
    };
    std::vector<double> R=eval(W);
    double norm=residual_norm(R,scale), norm0=std::fmax(norm,atol);
    if (norm == 0. || norm <= atol + rtol * norm0) { info={true,0,norm}; return W; }
    for (int it=0; it<max_iter; ++it) {
        const StepResult correction=ars_schur_iteration(W,target,gamma_dt,e1,e2,dx,dissipation);
        StepResult delta=correction;
        for (std::size_t i=0; i<W.alpha.size(); ++i) {
            delta.alpha[i]-=W.alpha[i]; delta.T1[i]-=W.T1[i]; delta.T2[i]-=W.T2[i];
            delta.u[i]-=W.u[i]; delta.p[i]-=W.p[i];
        }
        bool accepted=false;
        double lambda=1.;
        for (int ls=0; ls<line_search_max; ++ls) {
            StepResult trial=W;
            for (std::size_t i=0; i<W.alpha.size(); ++i) {
                trial.alpha[i]+=lambda*delta.alpha[i]; trial.T1[i]+=lambda*delta.T1[i];
                trial.T2[i]+=lambda*delta.T2[i]; trial.u[i]+=lambda*delta.u[i];
                trial.p[i]+=lambda*delta.p[i];
            }
            if (admissible(trial)) {
                const auto Rt=eval(trial);
                const double nt=residual_norm(Rt,scale);
                if (nt <= (1.-eta*lambda)*norm) {
                    W=std::move(trial); R=Rt; norm=nt; accepted=true; break;
                }
            }
            lambda*=.5;
        }
        if (!accepted) { info={false,it,norm}; return W; }
        if (norm <= atol + rtol * norm0) { info={true,it+1,norm}; return W; }
    }
    info={false,max_iter,norm}; return W;
}

inline std::vector<ConsU> advance_target(std::vector<ConsU> U, const ExplicitResidual* E, double ce,
                                         const ImplicitDivergences* I, double ci, double dt) {
    for(std::size_t k=0;k<U.size();++k) {
        if(E && ce!=0.) { U[k].m1-=dt*ce*E->m1[k]; U[k].m2-=dt*ce*E->m2[k]; U[k].mom-=dt*ce*E->mom[k]; U[k].rhoE-=dt*ce*E->rhoE[k]; U[k].a1-=dt*ce*E->alpha[k]; }
        if(I && ci!=0.) { U[k].mom-=dt*ci*I->grad_p[k]; U[k].rhoE-=dt*ci*I->div_pu[k]; }
    } return U;
}

inline StepResult ars222_step(const StepResult& Wn, double dt, double dx, const EOS& e1, const EOS& e2,
                              const StepConfig& cfg, NewtonInfo* stage2_info=nullptr, NewtonInfo* stage3_info=nullptr) {
    const auto Un=conservative_cells(Wn,e1,e2);
    FaceStateOptions face_options=cfg.ars_face_options;
    face_options.dt=dt; face_options.dx=dx; face_options.has_dt_dx=true;
    const auto E0=explicit_residual(Wn,e1,e2,dx,cfg.bc_l,cfg.bc_r,cfg.kapila_closure,false,EnergyForm::Differential,
                                    cfg.u_inlet_l,cfg.p_inlet_l,cfg.alpha_inlet_l,cfg.T1_inlet_l,cfg.T2_inlet_l,
                                    cfg.ars_explicit_positivity,cfg.ars_explicit_force_low,
                                    cfg.ars_explicit_rusanov_low,dt,face_options);
    const bool biharmonic = cfg.ars_biharmonic();
    const bool acoustic_riemann = cfg.ars_effective_acoustic_riemann();
    const bool upwind_dissipation = cfg.ars_effective_upwind_dissipation();
    const auto I0=implicit_divergences(Wn,e1,e2,dx,cfg.bc_l,cfg.bc_r,
                                       cfg.ars_implicit_dissipation,biharmonic,
                                       cfg.ars_rhie_chow,ARS_GAMMA*dt,
                                       acoustic_riemann,upwind_dissipation,
                                       cfg.ars_implicit_compact_lap_coeff);
    const auto U2=advance_target(Un,&E0,ARS_GAMMA,nullptr,0.,dt);
    NewtonInfo ni2;
    const StepResult W2 = cfg.ars_linear_solver == ARSLinearSolver::schur_helmholtz &&
                          !cfg.ars_rhie_chow && biharmonic && !acoustic_riemann
                          && !upwind_dissipation && cfg.ars_implicit_compact_lap_coeff == 0.0 && cfg.bc_l==BC5::Periodic
                          && cfg.bc_r==BC5::Periodic
        ? ars_schur_solve(Wn,U2,ARS_GAMMA*dt,e1,e2,dx,cfg.bc_l,cfg.bc_r,ni2,
                          cfg.newton_max_iter,cfg.newton_rtol,cfg.newton_atol,cfg.ars_implicit_dissipation,
                          cfg.newton_line_search_max,cfg.newton_eta)
        : ars_newton_solve(Wn,U2,ARS_GAMMA*dt,e1,e2,dx,cfg.bc_l,cfg.bc_r,ni2,
                           cfg.newton_max_iter,cfg.newton_rtol,cfg.newton_atol,cfg.ars_implicit_dissipation,biharmonic,
                           cfg.ars_rhie_chow,acoustic_riemann,
                           upwind_dissipation,cfg.ars_implicit_compact_lap_coeff,nullptr,
                           cfg.newton_line_search_max,cfg.newton_eta);
    const auto E2=explicit_residual(W2,e1,e2,dx,cfg.bc_l,cfg.bc_r,cfg.kapila_closure,false,EnergyForm::Differential,
                                    cfg.u_inlet_l,cfg.p_inlet_l,cfg.alpha_inlet_l,cfg.T1_inlet_l,cfg.T2_inlet_l,
                                    cfg.ars_explicit_positivity,cfg.ars_explicit_force_low,
                                    cfg.ars_explicit_rusanov_low,dt,face_options);
    const auto I2=implicit_divergences(W2,e1,e2,dx,cfg.bc_l,cfg.bc_r,
                                       cfg.ars_implicit_dissipation,biharmonic,
                                       cfg.ars_rhie_chow,ARS_GAMMA*dt,
                                       acoustic_riemann,upwind_dissipation,
                                       cfg.ars_implicit_compact_lap_coeff);
    auto U3=advance_target(Un,nullptr,0.,&I0,0.,dt); U3=advance_target(std::move(U3),&E2,1.,&I2,1.-ARS_GAMMA,dt);
    NewtonInfo ni3;
    const StepResult W3 = cfg.ars_linear_solver == ARSLinearSolver::schur_helmholtz &&
                          !cfg.ars_rhie_chow && biharmonic && !acoustic_riemann
                          && !upwind_dissipation && cfg.ars_implicit_compact_lap_coeff == 0.0 && cfg.bc_l==BC5::Periodic
                          && cfg.bc_r==BC5::Periodic
        ? ars_schur_solve(W2,U3,ARS_GAMMA*dt,e1,e2,dx,cfg.bc_l,cfg.bc_r,ni3,
                          cfg.newton_max_iter,cfg.newton_rtol,cfg.newton_atol,cfg.ars_implicit_dissipation,
                          cfg.newton_line_search_max,cfg.newton_eta)
        : ars_newton_solve(W2,U3,ARS_GAMMA*dt,e1,e2,dx,cfg.bc_l,cfg.bc_r,ni3,
                           cfg.newton_max_iter,cfg.newton_rtol,cfg.newton_atol,cfg.ars_implicit_dissipation,biharmonic,
                           cfg.ars_rhie_chow,acoustic_riemann,
                           upwind_dissipation,cfg.ars_implicit_compact_lap_coeff,nullptr,
                           cfg.newton_line_search_max,cfg.newton_eta);
    const auto I3=implicit_divergences(W3,e1,e2,dx,cfg.bc_l,cfg.bc_r,
                                       cfg.ars_implicit_dissipation,biharmonic,
                                       cfg.ars_rhie_chow,ARS_GAMMA*dt,
                                       acoustic_riemann,upwind_dissipation,
                                       cfg.ars_implicit_compact_lap_coeff);
    auto Unext=advance_target(Un,&E2,1.,&I2,1.-ARS_GAMMA,dt); Unext=advance_target(std::move(Unext),nullptr,0.,&I3,ARS_GAMMA,dt);
    StepResult out=relax_pressure(primitive_cells(Unext,W3,e1,e2),e1,e2);
    if(stage2_info) *stage2_info=ni2; if(stage3_info) *stage3_info=ni3;
    return out;
}
} // namespace cfd::five_eq
