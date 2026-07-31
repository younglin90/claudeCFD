// cfd/reconstruct3d_thinc_tanh.hpp — 3D tanh THINC-QQ BASELINE (for the efficiency
// comparison against deg3t). SAME additive-shift formulation as the deg3t core
// (sigma(kk·P + D)) so the ONLY difference is the sigmoid + the solve method:
//   * cell-D : NEWTON iteration, each iter evaluates <tanh(kk P + D)> over the hex by the
//              THINC/QQ PAPER REFERENCE quadrature (Xie & Xiao 2017 JCP): 3^3=27 pts for a
//              hexahedron (tanh has no closed-form cell-average inverse).
//   * face   : <tanh(kk P + D)> over the quad face by the THINC/QQ reference 3x3=9 pts.
//   (THINC/QQ also specifies tri 6 / quad 9 / tet 11 for simplex/hybrid grids — this hex
//    solver uses 27/9; the simplex counts are exercised in /tmp/mbq/tet_prof.cpp.)
// deg3t replaces BOTH with closed forms (Cardano on analytic moments + analytic face),
// so the wall-time delta isolates the Newton+quadrature cost. Dispatch via env
// THINCQQ_SIG_TANH: thinc3d_cellD/thinc3d_face_avg pick tanh (baseline) vs deg3t.
#pragma once
#include "cfd/reconstruct3d_bvd_core.hpp"   // D3Poly, deg3t3d_cellD, deg3t3d_face_avg
#include <cmath>
#include <cstdlib>

namespace cfd {

// Gauss-Legendre on [-1,1], orders 2/3/4 (points per axis). LOCKED to the THINC/QQ PAPER
// REFERENCE (Xie & Xiao 2017 JCP, "THINC method with quadratic surface representation and
// Gaussian quadrature"): a hexahedron uses 3^3=27 cell pts / 3^2=9 quad-face pts ⇒ NQ=3.
// (The paper specifies tri 6 / quad 9 / tet 11 / hex 27.) env THINCQQ_NQ overrides for
// SENSITIVITY studies only; default = the reference lock.
inline int thinc_nq(){ static const int n=[]{ const char* e=std::getenv("THINCQQ_NQ"); int v=e?std::atoi(e):3; return (v<2?2:(v>4?4:v)); }(); return n; }
inline const double* gauss_x(int nq){
    static const double X2[2]={-0.5773502691896257,0.5773502691896257};
    static const double X3[3]={-0.7745966692414834,0.0,0.7745966692414834};
    static const double X4[4]={-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526};
    return nq==2?X2:nq==3?X3:X4;
}
inline const double* gauss_w(int nq){
    static const double W2[2]={1.0,1.0};
    static const double W3[3]={0.5555555555555556,0.8888888888888888,0.5555555555555556};
    static const double W4[4]={0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538};
    return nq==2?W2:nq==3?W3:W4;
}
// 4-pt Gauss-Legendre on [-1,1] (legacy alias, kept for reference).
inline constexpr double TQ4_X[4] = {-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526};
inline constexpr double TQ4_W[4] = { 0.3478548451374538, 0.6521451548625461,0.6521451548625461,0.3478548451374538};

inline double tanh3d_Pval(const D3Poly& P, double dx, double dy, double dz){
    return P.c[0]*dx+P.c[1]*dy+P.c[2]*dz+P.c[3]*dx*dx+P.c[4]*dy*dy+P.c[5]*dz*dz
          +P.c[6]*dx*dy+P.c[7]*dx*dz+P.c[8]*dy*dz;
}

// OPT: the NQ^3 P-values + weights are geometry-only (kk/D-independent) ⇒ compute ONCE
// per cell/var, reuse across betas. Returns the point count np.
inline int tanh3d_cell_pvals(const D3Poly& P, double hx, double hy, double hz, double* Pv, double* wt){
    int nq=thinc_nq(); const double* GX=gauss_x(nq); const double* GW=gauss_w(nq);
    int idx=0;
    for(int a=0;a<nq;++a) for(int b=0;b<nq;++b) for(int c=0;c<nq;++c){
        double dx=0.5*hx*GX[a], dy=0.5*hy*GX[b], dz=0.5*hz*GX[c];
        Pv[idx]=tanh3d_Pval(P,dx,dy,dz); wt[idx]=GW[a]*GW[b]*GW[c]*0.125; ++idx;
    }
    return idx;
}
// cell-D from precomputed P-values (per beta kk): Newton on <tanh(kk Pv + D)> = Q.
inline double tanh3d_cellD_fromP(const double* Pv, const double* wt, int np, double kk, double Q){
    double xi[64]; for(int p=0;p<np;++p) xi[p]=kk*Pv[p];   // kk-scale once per beta
    double D=Q;                                            // initial guess (tanh ~ id near 0)
    for(int it=0;it<25;++it){
        double f=0.0, fp=0.0;
        for(int p=0;p<np;++p){ double th=std::tanh(xi[p]+D); f += wt[p]*th; fp += wt[p]*(1.0-th*th); }
        f -= Q;
        if(std::fabs(fp)<1e-30) break;
        double dD=f/fp; D-=dD;
        if(std::fabs(dD)<1e-12) break;
    }
    return D;
}
inline double tanh3d_cellD(const D3Poly& P, double kk, double hx, double hy, double hz, double Q){
    double Pv[64],wt[64]; int np=tanh3d_cell_pvals(P,hx,hy,hz,Pv,wt);
    return tanh3d_cellD_fromP(Pv,wt,np,kk,Q);
}

// OPT: the NQ^2 face P-values + weights are geometry-only ⇒ compute ONCE per face, reuse across betas.
inline int tanh3d_face_pvals(const D3Poly& P, int axis, double off, double h0, double h1, double* Pv, double* wt){
    int nq=thinc_nq(); const double* GX=gauss_x(nq); const double* GW=gauss_w(nq);
    int t0=(axis==0)?1:0, t1=(axis==2)?1:2; int idx=0;
    for(int a=0;a<nq;++a) for(int b=0;b<nq;++b){
        double d[3]; d[axis]=off; d[t0]=0.5*h0*GX[a]; d[t1]=0.5*h1*GX[b];
        Pv[idx]=tanh3d_Pval(P,d[0],d[1],d[2]); wt[idx]=GW[a]*GW[b]*0.25; ++idx;
    }
    return idx;
}
inline double tanh3d_face_avg_fromP(const double* Pv, const double* wt, int np, double kk, double D){
    double s=0.0; for(int p=0;p<np;++p) s += wt[p]*std::tanh(kk*Pv[p]+D); return s;
}
inline double tanh3d_face_avg(const D3Poly& P, double D, double kk, int axis, double off,
                              double h0, double h1){
    double Pv[16],wt[16]; int np=tanh3d_face_pvals(P,axis,off,h0,h1,Pv,wt);
    return tanh3d_face_avg_fromP(Pv,wt,np,kk,D);
}

// dispatch: env THINCQQ_SIG_TANH selects the tanh BASELINE (Newton+quadrature); default
// = our deg3t (closed-form Cardano + closed-form face). Same call signature for both.
inline bool thinc3d_use_tanh(){ static const bool t = std::getenv("THINCQQ_SIG_TANH")!=nullptr; return t; }

inline double thinc3d_cellD(const D3Poly& P, double kk, double hx, double hy, double hz, double Q){
    return thinc3d_use_tanh() ? tanh3d_cellD(P,kk,hx,hy,hz,Q) : deg3t3d_cellD(P,kk,hx,hy,hz,Q);
}
inline double thinc3d_face_avg(const D3Poly& P, double D, double kk, int axis, double off,
                               double h0, double h1){
    return thinc3d_use_tanh() ? tanh3d_face_avg(P,D,kk,axis,off,h0,h1)
                              : deg3t3d_face_avg(P,D,kk,axis,off,h0,h1);
}

// BETA-SHARED dispatch: solve both betas while building the geometry (moments / P-values)
// ONCE. This is the solver-side beta-sharing win (the per-beta combined calls recompute it).
inline void thinc3d_cellD_both(const D3Poly& P, double kk_l, double kk_s,
                               double hx, double hy, double hz, double Q,
                               double& Dl, double& Ds){
    if(thinc3d_use_tanh()){
        double Pv[64],wt[64]; int np=tanh3d_cell_pvals(P,hx,hy,hz,Pv,wt);
        Dl=tanh3d_cellD_fromP(Pv,wt,np,kk_l,Q); Ds=tanh3d_cellD_fromP(Pv,wt,np,kk_s,Q);
    } else {
        double M1,M2,M3; d3_cell_moments(P,hx,hy,hz,M1,M2,M3);
        Dl=deg3t3d_cellD_fromM(M1,M2,M3,kk_l,Q); Ds=deg3t3d_cellD_fromM(M1,M2,M3,kk_s,Q);
    }
}
inline void thinc3d_face_avg_both(const D3Poly& P, double D_l, double D_s,
                                  double kk_l, double kk_s, int axis, double off,
                                  double h0, double h1, double& fa_l, double& fa_s){
    if(thinc3d_use_tanh()){
        double Pv[16],wt[16]; int np=tanh3d_face_pvals(P,axis,off,h0,h1,Pv,wt);
        fa_l=tanh3d_face_avg_fromP(Pv,wt,np,kk_l,D_l); fa_s=tanh3d_face_avg_fromP(Pv,wt,np,kk_s,D_s);
    } else {
        D3FaceMom fm=deg3t3d_face_moments(P,axis,off,h0,h1);
        fa_l=deg3t3d_face_avg_fromM(fm,D_l,kk_l); fa_s=deg3t3d_face_avg_fromM(fm,D_s,kk_s);
    }
}

} // namespace cfd
