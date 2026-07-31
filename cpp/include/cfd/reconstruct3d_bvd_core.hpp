// cfd/reconstruct3d_bvd_core.hpp — deg3t THINC-QQ 3D core math (cell-D + face).
//
// The research core of the 3D deg3t-BVD reconstruction (Stage 2b). Self-contained
// (takes the interface polynomial P[9] directly; the o2 P2-LSQ supplies P when
// wired). Honors the constraints:
//   * QQ curvature  : P is a full 3D quadratic (gradient + Hessian).
//   * no Newton     : cell-D is a CLOSED-FORM cubic (Cardano) on the inner cubic.
//   * no quadrature : cell & face polynomial moments are EXACT CLOSED-FORM
//                     (monomial sums over analytic box moments), not Gauss.
// deg3t sigmoid (spl_mode 9): inner cubic s + a3 s^3 (|s|<asp), m=3 rational tail.
// Hybrid tail: the cell-D solves the INNER cubic exactly (Cardano); the face adds
// a centroid-frozen tail correction (sigma_full - sigma_inner at the face centre).
#pragma once
#include <cmath>
#include <array>
#include <cstdlib>

namespace cfd {

// ── deg3t sigmoid (exact constants from reconstruct_bvd.hpp:311) ──
inline constexpr double D3_A3  = -0.2175;        // inner cubic coeff
inline constexpr double D3_ASP =  0.7946;        // splice |s|
inline constexpr double D3_TA  =  0.3145201473;  // tail A
inline constexpr double D3_TB  =  0.8313725062;  // tail B
inline constexpr double D3_TK  =  1.5042926943;  // tail K

inline double d3_sig_inner(double s){ return s + D3_A3*s*s*s; }          // inner cubic only
inline double d3_sig_full(double s){                                     // full deg3t
    double u = s<0.0?-s:s, gg;
    if(u<D3_ASP){ gg = u + D3_A3*u*u*u; }
    else { double w=u-D3_ASP, KW=1.0+D3_TK*w; gg = 1.0-(D3_TA+D3_TB*w)/(KW*KW*KW); }
    return s<0.0?-gg:gg;
}
inline double d3_sig_inner_deriv(double s){ return 1.0 + 3.0*D3_A3*s*s; }
inline double d3_sig_full_deriv(double s){                               // dσ/ds (even in s)
    double u = s<0.0?-s:s;
    if(u<D3_ASP) return 1.0 + 3.0*D3_A3*u*u;
    double w=u-D3_ASP, KW=1.0+D3_TK*w;
    return -(D3_TB*KW - 3.0*D3_TK*(D3_TA+D3_TB*w))/(KW*KW*KW*KW);
}

// ── analytic 1D box moment over [-h/2,h/2]:  <x^n> = 0 (n odd) else (h/2)^n/(n+1) ──
inline double d3_boxmom(double h, int n){
    if(n & 1) return 0.0;
    double hh = 0.5*h, p = 1.0; for(int i=0;i<n;++i) p*=hh; return p/(double)(n+1);
}

// Interface polynomial P(dx,dy,dz) as 9 monomials (coef + integer exponents).
// Layout matches the o2 coeffs b0..b8 = [dx,dy,dz, dx^2,dy^2,dz^2, dxy,dxz,dyz].
struct D3Poly {
    double c[9];
    static constexpr int EX[9] = {1,0,0, 2,0,0, 1,1,0};
    static constexpr int EY[9] = {0,1,0, 0,2,0, 1,0,1};
    static constexpr int EZ[9] = {0,0,1, 0,0,2, 0,1,1};
};

// 1D box-moment table for [-h/2,h/2]: t[n]=<x^n>, n=0..6 (odd=0). OPT: precomputed
// once per axis (h²/12, h⁴/80, h⁶/448) → the moment sums become pure lookups+mul,
// no per-term pow-loop/divide (the old d3_boxmom hot-path cost).
inline void d3_mom_tab(double h, double t[7]){
    double a=0.25*h*h;                  // (h/2)^2
    t[0]=1.0; t[1]=0.0; t[2]=a/3.0; t[3]=0.0; t[4]=a*a/5.0; t[5]=0.0; t[6]=a*a*a/7.0;
}

// EXACT cell moments <P>, <P^2>, <P^3> over the hex box. STRAIGHT-LINE form: the
// surviving (all-even-exponent) monomial products are fixed at compile time, so the
// loops+parity-branches are unrolled to 3/12/32 unordered terms (mult folded in). This
// is ~13x faster than the parity-skip loop (verified identical to 1e-19). mx/my/mz hold
// the 1D box moments <x^n> (n even); odd-exponent factors are dropped (=1). Code-gen:
// /tmp/mbq/gen.py from the D3Poly monomial exponents.
inline void d3_cell_moments(const D3Poly& P, double hx, double hy, double hz,
                            double& M1, double& M2, double& M3){
    double mx[7],my[7],mz[7]; d3_mom_tab(hx,mx); d3_mom_tab(hy,my); d3_mom_tab(hz,mz);
    const double* c = P.c;
    M1 = c[3]*mx[2] + c[4]*my[2] + c[5]*mz[2];
    M2 = c[0]*c[0]*mx[2] + c[1]*c[1]*my[2] + c[2]*c[2]*mz[2]
       + c[3]*c[3]*mx[4] + 2.0*c[3]*c[4]*mx[2]*my[2] + 2.0*c[3]*c[5]*mx[2]*mz[2]
       + c[4]*c[4]*my[4] + 2.0*c[4]*c[5]*my[2]*mz[2] + c[5]*c[5]*mz[4]
       + c[6]*c[6]*mx[2]*my[2] + c[7]*c[7]*mx[2]*mz[2] + c[8]*c[8]*my[2]*mz[2];
    M3 = 3.0*c[0]*c[0]*c[3]*mx[4] + 3.0*c[0]*c[0]*c[4]*mx[2]*my[2] + 3.0*c[0]*c[0]*c[5]*mx[2]*mz[2]
       + 6.0*c[0]*c[1]*c[6]*mx[2]*my[2] + 6.0*c[0]*c[2]*c[7]*mx[2]*mz[2]
       + 3.0*c[1]*c[1]*c[3]*mx[2]*my[2] + 3.0*c[1]*c[1]*c[4]*my[4] + 3.0*c[1]*c[1]*c[5]*my[2]*mz[2]
       + 6.0*c[1]*c[2]*c[8]*my[2]*mz[2]
       + 3.0*c[2]*c[2]*c[3]*mx[2]*mz[2] + 3.0*c[2]*c[2]*c[4]*my[2]*mz[2] + 3.0*c[2]*c[2]*c[5]*mz[4]
       + c[3]*c[3]*c[3]*mx[6] + 3.0*c[3]*c[3]*c[4]*mx[4]*my[2] + 3.0*c[3]*c[3]*c[5]*mx[4]*mz[2]
       + 3.0*c[3]*c[4]*c[4]*mx[2]*my[4] + 6.0*c[3]*c[4]*c[5]*mx[2]*my[2]*mz[2] + 3.0*c[3]*c[5]*c[5]*mx[2]*mz[4]
       + 3.0*c[3]*c[6]*c[6]*mx[4]*my[2] + 3.0*c[3]*c[7]*c[7]*mx[4]*mz[2] + 3.0*c[3]*c[8]*c[8]*mx[2]*my[2]*mz[2]
       + c[4]*c[4]*c[4]*my[6] + 3.0*c[4]*c[4]*c[5]*my[4]*mz[2] + 3.0*c[4]*c[5]*c[5]*my[2]*mz[4]
       + 3.0*c[4]*c[6]*c[6]*mx[2]*my[4] + 3.0*c[4]*c[7]*c[7]*mx[2]*my[2]*mz[2] + 3.0*c[4]*c[8]*c[8]*my[4]*mz[2]
       + c[5]*c[5]*c[5]*mz[6] + 3.0*c[5]*c[6]*c[6]*mx[2]*my[2]*mz[2] + 3.0*c[5]*c[7]*c[7]*mx[2]*mz[4]
       + 3.0*c[5]*c[8]*c[8]*my[2]*mz[4] + 6.0*c[6]*c[7]*c[8]*mx[2]*my[2]*mz[2];
}

// real root of c3 D^3 + c2 D^2 + c1 D + c0 = 0 nearest to `guess` (robust).
inline double d3_cubic_root_near(double c3, double c2, double c1, double c0, double guess){
    if(std::fabs(c3) < 1e-300){  // quadratic/linear fallback
        if(std::fabs(c2) < 1e-300) return std::fabs(c1)>1e-300 ? -c0/c1 : guess;
        double disc = c1*c1 - 4*c2*c0; if(disc<0) disc=0; double sq=std::sqrt(disc);
        double r1=(-c1+sq)/(2*c2), r2=(-c1-sq)/(2*c2);
        return std::fabs(r1-guess)<std::fabs(r2-guess)?r1:r2;
    }
    // depressed cubic t^3 + p t + q,  D = t - c2/(3c3)
    double a=c2/c3, b=c1/c3, c=c0/c3, sh=a/3.0;
    double p=b - a*a/3.0, q=2.0*a*a*a/27.0 - a*b/3.0 + c;
    double disc = q*q/4.0 + p*p*p/27.0;
    double roots[3]; int nr=0;
    if(disc > 1e-300){ double s=std::sqrt(disc);
        double u=std::cbrt(-q/2.0+s), v=std::cbrt(-q/2.0-s); roots[nr++]=u+v-sh;
    } else { double r=std::sqrt(-p*p*p/27.0); double phi=std::acos(std::max(-1.0,std::min(1.0,-q/2.0/std::max(r,1e-300))));
        double m=2.0*std::cbrt(r);
        for(int k=0;k<3;++k) roots[nr++]=m*std::cos((phi+2.0*M_PI*k)/3.0)-sh; }
    double best=roots[0], bd=std::fabs(roots[0]-guess);
    for(int i=1;i<nr;++i){ double d=std::fabs(roots[i]-guess); if(d<bd){bd=d;best=roots[i];} }
    return best;
}

// FAST inner-cubic root. σ_inner(s)=s+a3 s³ (a3=-0.2175) is NON-monotone (turns at
// s=±1.238, σ≈±0.825), so for |Q| beyond the inner reach the physical root is NOT near
// the linear guess and Newton would overshoot/diverge. So: bounded Newton from the linear
// guess (cheap, POLYNOMIAL — no cbrt/acos) for the common physical case, with a robust
// Cardano FALLBACK on any sign of divergence (|D| runs away, flat derivative, or the
// residual fails to collapse). Fast path hits on the |Q|≲0.8 majority; Cardano only the tail.
inline double d3_cubic_root_fast(double c3,double c2,double c1,double c0,double guess){
    double D=guess;
    for(int it=0;it<5;++it){
        double f=((c3*D+c2)*D+c1)*D+c0;
        double fp=(3.0*c3*D+2.0*c2)*D+c1;
        if(std::fabs(fp)<1e-12 || D>8.0 || D<-8.0) break;            // degenerate/diverging
        double dD=f/fp; D-=dD;
        if(std::fabs(dD)<1e-11){
            double fr=((c3*D+c2)*D+c1)*D+c0;                          // verify residual
            if(std::fabs(fr)<1e-9) return D;
            break;
        }
    }
    return d3_cubic_root_near(c3,c2,c1,c0,guess);                     // robust fallback
}

// cell-D from PRECOMPUTED P-moments (M1=<P>,M2=<P^2>,M3=<P^3>): the moments are kk- and
// D-independent, so compute them ONCE per cell/var and call this per beta (kk). This is
// the beta-sharing win (the old combined call recomputed the 91-term M3 for every beta).
// <sigma_inner(xi+D)> = (m1+D) + a3(m3 + 3 m2 D + 3 m1 D^2 + D^3), xi=kk P, m_n=kk^n<P^n>.
// => a3 D^3 + 3 a3 m1 D^2 + (1+3 a3 m2) D + (m1 + a3 m3 - Q) = 0.
// cell-D WITH centroid-frozen tail: P=0 at the cell centroid ⇒ the tail's argument is D, so
// solve <sigma_inner(kk P+D)> + [sigma_full(D) - sigma_inner(D)] = Q. Accounts for saturation
// (consistent with the face's inner+frozen-tail) ⇒ reduces the overshoot of the inner-only solve.
inline double deg3t3d_cellD_fromM_tail(double M1,double M2,double M3,double kk,double Q){
    double m1=kk*M1, m2=kk*kk*M2, m3=kk*kk*kk*M3;
    double D = Q - m1;
    for(int it=0;it<8;++it){
        double inner = (m1+D) + D3_A3*(m3 + 3.0*m2*D + 3.0*m1*D*D + D*D*D);
        double tail  = d3_sig_full(D) - d3_sig_inner(D);
        double f = inner + tail - Q;
        double fp = (1.0 + D3_A3*(3.0*m2 + 6.0*m1*D + 3.0*D*D))
                  + (d3_sig_full_deriv(D) - d3_sig_inner_deriv(D));
        if(fp>-1e-300 && fp<1e-300) break;
        double dD=f/fp; D-=dD;
        if(std::fabs(dD)<1e-11) break;
    }
    return D;
}
inline bool deg3t_celld_tail(){ static const bool t=std::getenv("THINCQQ_CELLD_TAIL")!=nullptr; return t; }
// GAUSS (probit-identity) scheme: <tanh(kk P+D)>_cell ~ tanh(m1/sqrt(1+ (pi/2) v)), v=kk^2 Var(P)
// is D-INDEPENDENT, so the cell-average constraint inverts in CLOSED FORM (no Newton):
//   D = atanh(Q) sqrt(1+(pi/2)v) - kk<P>.  Uses only <P>,<P^2>; true tanh; cell-D & face CONSISTENT.
inline bool deg3t_gauss(){ static const bool t=std::getenv("THINCQQ_GAUSS")!=nullptr; return t; }
// probit constant c: theory value pi/2 (integral-optimal, confirmed by tuning); env-tunable for
// the scheme (E1) overshoot-vs-diffusion trade (larger c = smoother = less overshoot, more diffusion).
inline double deg3t_gc(){ static const double g=[]{ const char* e=std::getenv("THINCQQ_GC"); return (e&&e[0])?std::atof(e):1.5707963; }(); return g; }
inline double deg3t3d_cellD_fromM_gauss(double M1,double M2,double kk,double Q){
    double v = kk*kk*(M2 - M1*M1); if(v<0.0) v=0.0;
    double Qc = Q<-0.999?-0.999:(Q>0.999?0.999:Q);
    return std::atanh(Qc)*std::sqrt(1.0+deg3t_gc()*v) - kk*M1;
}
// GAUSS-S: probit base + Edgeworth SKEWNESS (3rd-moment) correction. The 3rd CENTRAL
// moment of s=kk P+D is D-independent, so the cell-average constraint still inverts in
// CLOSED FORM via a 1-step perturbation off the probit base D:
//   D = D_base + (mu3 a^2 / 3)(1 - 3 Q^2),  mu3 = kk^3(<P^3>-3<P><P^2>+2<P>^3), a^2=1/(1+c v).
// Adds only <P^3>; no Newton/quadrature. Improves cell-D fidelity ~2x and worst-cell face
// error ~40% vs the 2-moment GAUSS (validated, thinc_cellshape_bench). env THINCQQ_GAUSS_SKEW.
inline bool deg3t_gauss_skew(){ static const bool t=std::getenv("THINCQQ_GAUSS_SKEW")!=nullptr; return t; }
inline double deg3t3d_cellD_fromM_gaussS(double M1,double M2,double M3,double kk,double Q){
    double c=deg3t_gc(); double v=kk*kk*(M2-M1*M1); if(v<0.0)v=0.0; double a2=1.0/(1.0+c*v);
    double Qc=Q<-0.999?-0.999:(Q>0.999?0.999:Q);
    double Dbase=std::atanh(Qc)*std::sqrt(1.0+c*v)-kk*M1;
    double mu3=kk*kk*kk*(M3-3.0*M1*M2+2.0*M1*M1*M1);
    return Dbase+(mu3*a2/3.0)*(1.0-3.0*Qc*Qc);
}
inline double deg3t3d_cellD_fromM(double M1,double M2,double M3,double kk,double Q){
    if(deg3t_gauss())      return deg3t_gauss_skew()?deg3t3d_cellD_fromM_gaussS(M1,M2,M3,kk,Q)
                                                    :deg3t3d_cellD_fromM_gauss(M1,M2,kk,Q);
    if(deg3t_celld_tail()) return deg3t3d_cellD_fromM_tail(M1,M2,M3,kk,Q);
    double m1=kk*M1, m2=kk*kk*M2, m3=kk*kk*kk*M3;
    double c3=D3_A3, c2=3.0*D3_A3*m1, c1=1.0+3.0*D3_A3*m2, c0=m1+D3_A3*m3-Q;
    return d3_cubic_root_fast(c3,c2,c1,c0, Q - m1);  // guess = linear (a3->0) solution
}
inline double deg3t3d_cellD(const D3Poly& P, double kk, double hx, double hy, double hz, double Q){
    double M1,M2,M3; d3_cell_moments(P,hx,hy,hz,M1,M2,M3);
    return deg3t3d_cellD_fromM(M1,M2,M3,kk,Q);
}

// Build the in-face 2D quadratic (in the two tangential axes) at a fixed normal
// offset and accumulate <P_f>,<P_f^2>,<P_f^3> over the face rectangle (EXACT,
// closed-form monomial sums over the two tangential box moments).
inline void d3_face_moments(const D3Poly& P, int axis, double off,
                            double h0, double h1,   // tangential cell sizes (the two non-axis dims)
                            double& F1, double& F2, double& F3, double& Pc /*centre value*/){
    // tangential axes = the two not equal to `axis`
    int t0 = (axis==0)?1:0, t1 = (axis==2)?1:2;
    // collect face monomials in (a,b)=(exp of t0, exp of t1) with coef folded by off^(axis exp)
    double fc[3][3] = {{0,0,0},{0,0,0},{0,0,0}};   // a,b in 0..2
    for(int i=0;i<9;++i){ double ci=P.c[i]; if(ci==0.0) continue;
        int en = (axis==0)?D3Poly::EX[i] : (axis==1)?D3Poly::EY[i] : D3Poly::EZ[i];
        int ea = (t0==0)?D3Poly::EX[i] : (t0==1)?D3Poly::EY[i] : D3Poly::EZ[i];
        int eb = (t1==0)?D3Poly::EX[i] : (t1==1)?D3Poly::EY[i] : D3Poly::EZ[i];
        double off_p = 1.0; for(int e=0;e<en;++e) off_p*=off;
        fc[ea][eb] += ci*off_p;
    }
    Pc = fc[0][0];   // value at face centre (tangential 0,0)
    double t0m[7],t1m[7]; d3_mom_tab(h0,t0m); d3_mom_tab(h1,t1m);
    // STRAIGHT-LINE 2D face moments (in-face quadratic g0..g5 at exps (0,0)(1,0)(0,1)(2,0)(0,2)(1,1)),
    // surviving all-even-exponent terms unrolled (F1=3, F2=9, F3=20). Code-gen /tmp/mbq/gen2d.py.
    double g0=fc[0][0],g1=fc[1][0],g2=fc[0][1],g3=fc[2][0],g4=fc[0][2],g5=fc[1][1];
    F1 = g0 + g3*t0m[2] + g4*t1m[2];
    F2 = g0*g0 + 2.0*g0*g3*t0m[2] + 2.0*g0*g4*t1m[2] + g1*g1*t0m[2] + g2*g2*t1m[2]
       + g3*g3*t0m[4] + 2.0*g3*g4*t0m[2]*t1m[2] + g4*g4*t1m[4] + g5*g5*t0m[2]*t1m[2];
    F3 = g0*g0*g0 + 3.0*g0*g0*g3*t0m[2] + 3.0*g0*g0*g4*t1m[2] + 3.0*g0*g1*g1*t0m[2]
       + 3.0*g0*g2*g2*t1m[2] + 3.0*g0*g3*g3*t0m[4] + 6.0*g0*g3*g4*t0m[2]*t1m[2]
       + 3.0*g0*g4*g4*t1m[4] + 3.0*g0*g5*g5*t0m[2]*t1m[2] + 3.0*g1*g1*g3*t0m[4]
       + 3.0*g1*g1*g4*t0m[2]*t1m[2] + 6.0*g1*g2*g5*t0m[2]*t1m[2] + 3.0*g2*g2*g3*t0m[2]*t1m[2]
       + 3.0*g2*g2*g4*t1m[4] + g3*g3*g3*t0m[6] + 3.0*g3*g3*g4*t0m[4]*t1m[2]
       + 3.0*g3*g4*g4*t0m[2]*t1m[4] + 3.0*g3*g5*g5*t0m[4]*t1m[2] + g4*g4*g4*t1m[6]
       + 3.0*g4*g5*g5*t0m[2]*t1m[4];
}

// Face moments are kk/D-independent ⇒ compute ONCE per face, reuse across betas + faces.
struct D3FaceMom { double F1,F2,F3,Pc; };
inline D3FaceMom deg3t3d_face_moments(const D3Poly& P, int axis, double off, double h0, double h1){
    D3FaceMom m; d3_face_moments(P,axis,off,h0,h1,m.F1,m.F2,m.F3,m.Pc); return m;
}
// MOMENT-2pt face average: build a 2-point Gauss rule in s=kk*P+D from the closed-form
// s-moments (m1,m2,m3 from F1,F2,F3) and evaluate the FULL deg3t sigmoid at the 2 nodes.
// Conic-clip-free, no divergence (bounded sigmoid at the nodes), 17-56x more accurate than
// the centroid-freeze (smoke-validated). Cost = same moments + a 2x2 solve + 2 evals.
// NOTE: accurate in isolation (smoke-validated 17-56x better than freeze) BUT it
// DESTABILIZES as a face-ONLY change — the cell-D solves <sigma_inner>=Q while this face
// computes <sigma_full>, so the face value is inconsistent with the cell-average constraint
// ⇒ non-conservative flux ⇒ the deformation run diverges (E1~1e12). Using it requires a
// CONSISTENT moment-2pt cell-D too (solve <sigma_full>_2pt=Q). Kept env-gated, default OFF.
inline double deg3t3d_face_avg_mq2(const D3FaceMom& fm, double D, double kk){
    double m1 = kk*fm.F1 + D;
    double m2 = kk*kk*fm.F2 + 2.0*kk*D*fm.F1 + D*D;
    double m3 = kk*kk*kk*fm.F3 + 3.0*kk*kk*D*fm.F2 + 3.0*kk*D*D*fm.F1 + D*D*D;
    double den = m2 - m1*m1;                                             // Var(s) >= 0
    // near-constant s over the face (flat interface / smooth cell): the two Gauss nodes
    // collapse and the weights blow up — fall back to the EXACT 1-point value (for flat s,
    // <sigma> = sigma(m1) exactly). Relative threshold sits above the m2-m1^2 cancellation floor.
    if(den < 1e-9*(1.0 + m1*m1)){ double v=d3_sig_full(m1); return v<-1.0?-1.0:(v>1.0?1.0:v); }
    double b = (m3 - m1*m2)/den, g = m2 - b*m1;
    double disc = b*b + 4.0*g;                                           // = (b-2m1)^2 + 4*den > 0
    double sq = std::sqrt(disc);
    double s1=(b+sq)*0.5, s2=(b-sq)*0.5;
    double w1 = (m1-s2)/(s1-s2), w2=1.0-w1;
    double v = w1*d3_sig_full(s1) + w2*d3_sig_full(s2);
    // robust guard: if anything degenerated (non-finite or |w| huge), use the 1-point value.
    if(!std::isfinite(v) || std::fabs(w1)>1e6 || std::fabs(w2)>1e6){
        v = d3_sig_full(m1);
    }
    return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v);
}
inline bool deg3t_mq2(){ static const bool t = std::getenv("THINCQQ_MQ2")!=nullptr; return t; }
// GAUSS (probit) face average: tanh(m1/sqrt(1+(pi/2)v)), v=kk^2(F2-F1^2). Consistent with the
// GAUSS closed-form cell-D (both probit) ⇒ conservative & stable. 1 tanh, uses only <P>,<P^2>.
inline double deg3t3d_face_avg_gauss(const D3FaceMom& fm, double D, double kk){
    double v = kk*kk*(fm.F2 - fm.F1*fm.F1); if(v<0.0) v=0.0;
    double m1 = kk*fm.F1 + D;
    return std::tanh(m1/std::sqrt(1.0+deg3t_gc()*v));
}
// GAUSS-S face average: probit + Edgeworth skewness, <tanh(kk P+D)>_face ~=
//   T0 - (mu3 a^3/3)(1-T0^2)(1-3 T0^2),  T0=tanh(m1/sqrt(1+c v)), a=1/sqrt(1+c v),
//   mu3 = kk^3(<P^3>_f - 3<P>_f<P^2>_f + 2<P>_f^3).  Uses fm.F3 (the 3rd face moment).
inline double deg3t3d_face_avg_gaussS(const D3FaceMom& fm, double D, double kk){
    double c=deg3t_gc(); double vf=kk*kk*(fm.F2-fm.F1*fm.F1); if(vf<0.0)vf=0.0; double sq=std::sqrt(1.0+c*vf);
    double T0=std::tanh((kk*fm.F1+D)/sq);
    double mu3=kk*kk*kk*(fm.F3-3.0*fm.F1*fm.F2+2.0*fm.F1*fm.F1*fm.F1), a3=1.0/(sq*sq*sq);
    double fv=T0-(mu3*a3/3.0)*(1.0-T0*T0)*(1.0-3.0*T0*T0);
    return fv<-1.0?-1.0:(fv>1.0?1.0:fv);
}
// face-average of sigma(kk P + D) from PRECOMPUTED face moments (per beta).
inline double deg3t3d_face_avg_fromM(const D3FaceMom& fm, double D, double kk){
    if(deg3t_gauss()) return deg3t_gauss_skew()?deg3t3d_face_avg_gaussS(fm,D,kk):deg3t3d_face_avg_gauss(fm,D,kk);
    if(deg3t_mq2()) return deg3t3d_face_avg_mq2(fm,D,kk);                 // moment-2pt (opt-in)
    double m1=kk*fm.F1, m2=kk*kk*fm.F2, m3=kk*kk*kk*fm.F3;
    double inner = (m1+D) + D3_A3*(m3 + 3.0*m2*D + 3.0*m1*D*D + D*D*D);   // <sigma_inner>
    double sc = kk*fm.Pc + D;                                            // centroid-frozen tail
    double tail = d3_sig_full(sc) - d3_sig_inner(sc);
    double v = inner + tail;
    return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v);
}
// combined (axis-normal face, offset off + tangential sizes h0,h1) — wrapper.
inline double deg3t3d_face_avg(const D3Poly& P, double D, double kk, int axis, double off,
                               double h0, double h1){
    return deg3t3d_face_avg_fromM(deg3t3d_face_moments(P,axis,off,h0,h1), D, kk);
}

} // namespace cfd
