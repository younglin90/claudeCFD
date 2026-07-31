// cfd/reconstruct_bvd.hpp — Boundary-Variation-Diminishing (BVD) reconstruction.
// Two candidates per cell: SMOOTH = BJ-vertex MLP-u (reconstruct_bj_vertex),
// SHARP = order-2 quadratic MLP (reconstruct_o2_limited). Per cell pick the
// candidate with the LOWER total boundary variation (sum of |W_L-W_R| over the
// cell's interior faces, measured on a chosen variable -> density for Euler).
// BVD selects the sharp candidate at discontinuities (lower TBV there) and the
// smooth one in smooth regions. Port of TMLPUSmoothSharpBVD's TBV selection.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/reconstruct2d.hpp"
#include "cfd/reconstruct2d_o2.hpp"
#include "cfd/io_vtk.hpp"   // BVD_CANDFLAG diagnostic buffers (bvd_cand_flag); io_vtk pulls only mesh.hpp (no cycle)
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>

// Fast tanh (Pade [7/8] approximant; |err| < 3e-8 for |x| < 4.6, ~2-3x faster than libm tanh).
// The THINC sigmoid is a sub-cell shape model, so this does not change the reconstruction scheme.
inline double fast_tanh(double x){
    if(x<-4.6) return -1.0; if(x>4.6) return 1.0;
    double x2=x*x;
    return x*(135135.0+x2*(17325.0+x2*(378.0+x2)))
          /(135135.0+x2*(62370.0+x2*(3150.0+x2*28.0)));
}

// In-code component profiler. env CHENG3_PROF=1 -> at program exit prints accumulated wall time
// of the MUSCL candidate vs the THINC/QQ passes vs the BVD selection, across ALL recon calls.
#include <chrono>
#include <cstdio>
struct Cheng3Prof { double muscl=0,thinc=0,sel=0,geom=0,face=0; long ncall=0;
    ~Cheng3Prof(){ if(std::getenv("CHENG3_PROF") && ncall>0)
        std::fprintf(stderr,"[CHENG3_PROF] recon_calls=%ld  MUSCL=%.3fs  THINC=%.3fs (geom=%.3fs face=%.3fs)  BVD_sel=%.3fs  (THINC/MUSCL=%.2fx)\n",
            ncall,muscl,thinc,geom,face,sel, muscl>0?thinc/muscl:0.0); } };
inline Cheng3Prof& cheng3_prof(){ static Cheng3Prof p; return p; }
inline double prof_now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ML-free predictive-BVD feasibility (env CHENG3_PREDICT): can the CHEAP MUSCL total-boundary-
// variation tM_norm (computed anyway) predict the actual TBV selection (THINC vs MUSCL)? If a
// threshold on tM_norm agrees with the real pick >~90%, predictive BVD (compute only the
// predicted candidate) is feasible. Reports agreement at several thresholds at program exit.
struct Cheng3Predict { static const int NT=7,NC=6;
    long n=0,n_thinc=0,agree_tm[NT]={0},agree_cc[NC]={0},agree_2f[NT][NC]={{0}};
    double tmthr[NT]={0.02,0.04,0.06,0.10,0.15,0.25,0.40}, ccthr[NC]={0.40,0.50,0.60,0.70,0.80,0.90};
    double sct=0,scm=0;  // mean conc for THINC/MUSCL
    ~Cheng3Predict(){ if(!std::getenv("CHENG3_PREDICT")||n==0)return;
        std::fprintf(stderr,"[PREDICT] n=%ld THINC=%.1f%%  mean conc(T/M)=%.3f/%.3f\n",
            n,100.0*n_thinc/n, n_thinc>0?sct/n_thinc:0, (n-n_thinc)>0?scm/(n-n_thinc):0);
        long bt=0,bc=0,b2=0; for(int k=0;k<NT;++k)bt=std::max(bt,agree_tm[k]);
        for(int k=0;k<NC;++k)bc=std::max(bc,agree_cc[k]);
        for(int i=0;i<NT;++i)for(int j=0;j<NC;++j)b2=std::max(b2,agree_2f[i][j]);
        std::fprintf(stderr,"[PREDICT] best agreement: tM-only=%.1f%%  conc-only=%.1f%%  tM+conc(2feat)=%.1f%%\n",
            100.0*bt/n,100.0*bc/n,100.0*b2/n); } };
inline Cheng3Predict& cheng3_predict(){ static Cheng3Predict p; return p; }
#include <string>

namespace cfd {

// ---- POLYGON FAN DECOMPOSITION (arbitrary convex n-gon cell) --------------------------------
// BUGFIX 2026-07-27: every THINC/QQ CELL integral in this file used to read ONLY the first three
// nodes (vs[0..2]), i.e. a quadrilateral cell was integrated over the sub-triangle (v0,v1,v2)
// instead of the whole cell, while kk=beta/H used the TRUE quad area+perimeter. Measured damage on
// random convex quads: |D_true - D_subtri| RMS 2.5e-1, p99 1.0, max 2.0 -- on a quantity bounded by
// |D|<1. All paper meshes are triangular so the bug was dormant, but any quad/polygon mesh silently
// got a badly wrong THINC interface.
// FIX: fan-triangulate about the CELL CENTROID (cx,cy): sub-triangle k = (centroid, v_k, v_{k+1}).
// All sub-triangle moments/quadratures are taken about the SAME origin (cx,cy) and combined by
// AREA-WEIGHTED average (the barycentric/quadrature formulas return AREA AVERAGES, not raw
// integrals):   <f>_cell = ( sum_k Area_k <f>_k ) / ( sum_k Area_k ).
// nv==3 returns the ORIGINAL triangle as the single "fan" member with normalised weight exactly
// 1.0 -> the triangle path is BIT-IDENTICAL to the legacy code (no regression).
// Validity: the fan areas are SIGNED (the divergence-theorem/shoelace convention), so the
// decomposition is exact for ANY simple polygon with consistently ordered nodes -- if the centroid
// happens to fall outside a re-entrant cell, the flipped sub-triangles carry a negative weight and
// the over-counted region cancels exactly. For convex cells (every real mesh cell) all signs are
// positive and this is identical to using |area|. Verified: over 160k random convex + 76k random
// non-convex quads the fan areas sum to the shoelace area to <=1e-14 relative in both classes.
// NOTE: FACES are untouched -- a 2D face is always a straight 2-node segment, so the edge
// quadrature / closed-form edge moments were correct all along.
inline constexpr int C3_FAN_MAX = 12;              // max fan sub-triangles (= max nodes per cell)
inline constexpr int C3_NQMAX   = C3_FAN_MAX * 6;  // max cell quadrature points (fan x 6-pt rule)
// tri[k] = {x0,y0,x1,y1,x2,y2} (vertex 0 = fan apex), ar[k] = SIGNED area. Returns #sub-triangles.
inline int c3_cell_fan(const double* nodes, const int* vs, int nv,
                       double cx, double cy, double (*tri)[6], double* ar) {
    if(nv<3) return 0;
    if(nv==3){   // legacy path, verbatim: the cell IS the triangle
        for(int j=0;j<3;++j){ int vid=vs[j]; tri[0][2*j]=nodes[vid*2]; tri[0][2*j+1]=nodes[vid*2+1]; }
        ar[0]=1.0;                       // single member -> area normalisation is an exact no-op
        return 1; }
    if(nv>C3_FAN_MAX) nv=C3_FAN_MAX;     // never hit by real 2D meshes (tri/quad)
    for(int k=0;k<nv;++k){ int k2=(k+1)%nv; int va=vs[k], vb=vs[k2];
        double xa=nodes[va*2], ya=nodes[va*2+1], xb=nodes[vb*2], yb=nodes[vb*2+1];
        tri[k][0]=cx; tri[k][1]=cy; tri[k][2]=xa; tri[k][3]=ya; tri[k][4]=xb; tri[k][5]=yb;
        ar[k]=0.5*((xa-cx)*(yb-cy)-(xb-cx)*(ya-cy)); }   // SIGNED (see note above)
    return nv;
}
// Cell POINT quadrature over the WHOLE polygon: P2 surface values Pg[] + area-normalised weights
// wq[] (sum=1). Triangle -> exactly the legacy points and weights (wq[q]==TQp[q][0] bitwise).
//   P = A0 dx + A1 dy + A2 dx^2 + A3 dy^2 + A4 dx dy   (production form)
inline int c3_cell_quad_P(const double (*TQp)[4], int NQC, const double* nodes,
                          const int* vs, int nv, double cx, double cy,
                          const double* A, double* Pg, double* wq) {
    double tri[C3_FAN_MAX][6], ar[C3_FAN_MAX];
    const int nt=c3_cell_fan(nodes,vs,nv,cx,cy,tri,ar);
    double at=0.0; for(int k=0;k<nt;++k) at+=ar[k]; if(!(std::fabs(at)>1e-300)) at=1.0;   // signed total area
    int nq=0;
    for(int k=0;k<nt;++k){ const double fw=ar[k]/at;
        for(int q=0;q<NQC;++q){
            double x=TQp[q][1]*tri[k][0]+TQp[q][2]*tri[k][2]+TQp[q][3]*tri[k][4];
            double y=TQp[q][1]*tri[k][1]+TQp[q][2]*tri[k][3]+TQp[q][3]*tri[k][5];
            double dx=x-cx, dy=y-cy;
            Pg[nq]=A[0]*dx+A[1]*dy+A[2]*dx*dx+A[3]*dy*dy+A[4]*dx*dy;
            wq[nq]=TQp[q][0]*fw; ++nq; } }
    return nq;
}

// T-MLP-u downwind compressive reconstruction (the BVD SHARP candidate, CHEAP =
// linear gradient + t* downwind blend + neighbour-bound clip; NO quadratic). For
// face f, owner o: W_L = clip( W_o + grad_o.(m_f-c_o) + t*(W_n - W_o), [min_o,max_o] ).
// The t*(W_n-W_o) term is the T-MLP-u tangent/downwind compression (pulls the face
// value toward the downwind neighbour = sharpens contacts); the clip is the psi in
// [0,1] (LMP/positivity). BVD/TBV gates WHERE this compressive candidate is used,
// fixing the r-ratio Flaw of a standalone TVD T-MLP-u.
inline void reconstruct_tmlpu_dw(const Mesh& m, const ReconCtx& c,
                                 const std::vector<double>& W, int nvar,
                                 std::vector<double>& WLf, std::vector<double>& WRf,
                                 double tstar) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const double* cc = m.cell_centers.data();
    std::vector<double> grad((size_t)nvar*N*2, 0.0), mn((size_t)nvar*N), mx((size_t)nvar*N);
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci)
        for (int v = 0; v < nvar; ++v) {
            double r0=0, r1=0, wc=W[(size_t)v*N+ci], lo=wc, hi=wc;
            for (int k = 0; k < c.max_nb; ++k) {
                int nb = c.nb[(size_t)ci*c.max_nb+k]; if (nb < 0) continue;
                double dphi = W[(size_t)v*N+nb] - wc, wk = c.w[(size_t)ci*c.max_nb+k];
                r0 += wk*c.d[((size_t)ci*c.max_nb+k)*2+0]*dphi;
                r1 += wk*c.d[((size_t)ci*c.max_nb+k)*2+1]*dphi;
                if (W[(size_t)v*N+nb] < lo) lo = W[(size_t)v*N+nb];
                if (W[(size_t)v*N+nb] > hi) hi = W[(size_t)v*N+nb];
            }
            grad[((size_t)v*N+ci)*2+0] = c.ATA_inv[ci*4+0]*r0 + c.ATA_inv[ci*4+1]*r1;
            grad[((size_t)v*N+ci)*2+1] = c.ATA_inv[ci*4+2]*r0 + c.ATA_inv[ci*4+3]*r1;
            mn[(size_t)v*N+ci] = lo; mx[(size_t)v*N+ci] = hi;
        }
    WLf.assign((size_t)nvar*Nf, 0.0); WRf.assign((size_t)nvar*Nf, 0.0);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double fx = m.face_centers[f*2+0], fy = m.face_centers[f*2+1];
        for (int v = 0; v < nvar; ++v) {
            double wo = W[(size_t)v*N+o];
            double inc = grad[((size_t)v*N+o)*2+0]*(fx-cc[o*2+0]) + grad[((size_t)v*N+o)*2+1]*(fy-cc[o*2+1]);
            double wl = wo + inc + (n>=0 ? tstar*(W[(size_t)v*N+n]-wo) : 0.0);
            double lo=mn[(size_t)v*N+o], hi=mx[(size_t)v*N+o];
            WLf[(size_t)v*Nf+f] = wl<lo?lo:(wl>hi?hi:wl);
            if (n >= 0) {
                double wn = W[(size_t)v*N+n];
                double incn = grad[((size_t)v*N+n)*2+0]*(fx-cc[n*2+0]) + grad[((size_t)v*N+n)*2+1]*(fy-cc[n*2+1]);
                double wr = wn + incn + tstar*(wo-wn);
                double lon=mn[(size_t)v*N+n], hin=mx[(size_t)v*N+n];
                WRf[(size_t)v*Nf+f] = wr<lon?lon:(wr>hin?hin:wr);
            } else WRf[(size_t)v*Nf+f] = WLf[(size_t)v*Nf+f];
        }
    }
}

// THINC (tangent-hyperbola interface capturing) reconstruction — the BVD SHARP
// candidate. Bounded-by-construction tanh profile that steepens contacts to ~1-2
// cells; NO ratcheting (no FCT needed), LINEAR (no quadrature). Per face f, owner
// o, neighbour n: c_bar = (W_o-qmin)/(qmax-qmin) over o's neighbours, theta =
// sign(W_n-W_o); face value from the tanh profile, clipped to local [min,max].
// Port of solver/five_eq_IMEX/explicit.py::_thinc_alpha_face (beta=1.6).
inline void reconstruct_thinc(const Mesh& m, const ReconCtx& c,
                              const std::vector<double>& W, int nvar,
                              std::vector<double>& WLf, std::vector<double>& WRf,
                              double beta) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const double tb = std::tanh(beta), cb = std::cosh(beta);
    std::vector<double> mn((size_t)nvar*N), mx((size_t)nvar*N);
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) for (int v = 0; v < nvar; ++v) {
        double wc=W[(size_t)v*N+ci], lo=wc, hi=wc;
        for (int k = 0; k < c.max_nb; ++k) { int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
            double w=W[(size_t)v*N+nb]; if(w<lo)lo=w; if(w>hi)hi=w; }
        mn[(size_t)v*N+ci]=lo; mx[(size_t)v*N+ci]=hi;
    }
    // THINC face value for cell ci toward a neighbour with value wn.
    auto thinc = [&](double wc, double qmin, double qmax, double wn) -> double {
        double rng = qmax - qmin;
        if (rng <= 1e-14) return wc;
        double cbar = (wc - qmin) / rng;
        if (cbar <= 1e-9 || cbar >= 1.0-1e-9) return wc;     // no interface in cell
        double theta = (wn >= wc) ? 1.0 : -1.0;
        double b = std::exp(theta*beta*(2.0*cbar - 1.0));
        double a = (b/cb - 1.0) / tb;
        double qf = qmin + 0.5*rng*(1.0 + theta*(tb + a)/std::max(1.0 + a*tb, 1e-14));
        double clo = std::min(wc, wn), chi = std::max(wc, wn);   // local monotone bound
        return qf<clo?clo:(qf>chi?chi:qf);
    };
    WLf.assign((size_t)nvar*Nf, 0.0); WRf.assign((size_t)nvar*Nf, 0.0);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o=m.face_owner[f], n=m.face_neighbour[f];
        for (int v = 0; v < nvar; ++v) {
            double wo=W[(size_t)v*N+o], wn=(n>=0)?W[(size_t)v*N+n]:wo;
            WLf[(size_t)v*Nf+f] = thinc(wo, mn[(size_t)v*N+o], mx[(size_t)v*N+o], wn);
            WRf[(size_t)v*Nf+f] = (n>=0) ? thinc(wn, mn[(size_t)v*N+n], mx[(size_t)v*N+n], wo)
                                         : WLf[(size_t)v*Nf+f];
        }
    }
}

// THINC/QQ (Cheng 2021 / Xie & Xiao 2017, JCP 349:415-440) — multi-dim THINC with Quadratic
// surface representation + Gaussian Quadrature, faithful to [45]:
//  - P_i is the UNIT-NORMAL interface surface: a_st from the NORMALIZED gradient n=grad q/|grad q|
//    and its derivatives (curvature), NOT the raw quadratic LSQ. This makes beta control the
//    transition thickness uniformly regardless of field-gradient magnitude -> NO shock
//    over-sharpening. (Raw-LSQ P over-steepens strong shocks -> the earlier divergence.)
//  - d from conservation via the rational form in D=tanh(k d) (Eq 20-23):
//    sum_g w_g (A_g+D)/(1+A_g D) = 2 cbar-1, A_g=tanh(k P_g), Newton-Raphson.
//  - 6-pt triangle Gauss quadrature (cell integral), 4-pt Gauss-Legendre (edge flux integral).
//  k=beta/H, H=hydraulic diameter 4|i|/perim. qmin/qmax over VERTEX-sharing cells. n=grad q/|grad q|
//  points toward increasing q so the qmax side aligns automatically. Face: (A_face+D)/(1+A_face D).
inline void reconstruct_thinc_qq(const Mesh& m, const ReconCtxO2& c,
                                 const std::vector<double>& W, int nvar,
                                 std::vector<double>& WLf, std::vector<double>& WRf,
                                 double beta) {
    const int N=m.n_cells(), Nf=m.n_faces(); const double* cc=m.cell_centers.data();
    std::vector<double> g((size_t)nvar*N*5, 0.0);   // g0=qx g1=qy g2=qxx g3=qyy g4=qxy
    #pragma omp parallel for
    for(int ci=0;ci<N;++ci) for(int v=0;v<nvar;++v){ double wc=W[(size_t)v*N+ci]; double co[5]={0,0,0,0,0};
        for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue; double dW=W[(size_t)v*N+nb]-wc;
            for(int i=0;i<5;++i) co[i]+=c.M[((size_t)ci*5+i)*c.max_nb+k]*dW; }
        for(int i=0;i<5;++i) g[((size_t)v*N+ci)*5+i]=co[i]; }
    std::vector<double> vmn((size_t)nvar*c.Nn), vmx((size_t)nvar*c.Nn);
    #pragma omp parallel for
    for(int vv=0;vv<c.Nn;++vv) for(int v=0;v<nvar;++v){ double mn=1e300,mx=-1e300;
        for(int k=0;k<c.max_v2c;++k){ int ci=c.v2c[(size_t)vv*c.max_v2c+k]; if(ci<0)continue; double val=W[(size_t)v*N+ci]; if(val<mn)mn=val; if(val>mx)mx=val; }
        if(mn>mx){mn=0;mx=0;} vmn[(size_t)v*c.Nn+vv]=mn; vmx[(size_t)v*c.Nn+vv]=mx; }
    // P_i(dx,dy)=a0 dx+a1 dy+a2 dx^2+a3 dy^2+a4 dx dy  (a from unit normal + curvature)
    auto Pval=[&](const double* A,double dx,double dy){ return A[0]*dx+A[1]*dy+A[2]*dx*dx+A[3]*dy*dy+A[4]*dx*dy; };
    static const double TQ[6][4]={ // 6-pt triangle Gauss (Dunavant deg4): w,b0,b1,b2
        {0.109951743655322,0.816847572980459,0.091576213509771,0.091576213509771},
        {0.109951743655322,0.091576213509771,0.816847572980459,0.091576213509771},
        {0.109951743655322,0.091576213509771,0.091576213509771,0.816847572980459},
        {0.223381589678011,0.108103018168070,0.445948490915965,0.445948490915965},
        {0.223381589678011,0.445948490915965,0.108103018168070,0.445948490915965},
        {0.223381589678011,0.445948490915965,0.445948490915965,0.108103018168070}};
    static const double EQ[4][2]={ // 4-pt Gauss-Legendre on [0,1]: t,w
        {0.0694318442029737,0.1739274225687269},{0.3300094782075719,0.3260725774312731},
        {0.6699905217924281,0.3260725774312731},{0.9305681557970263,0.1739274225687269}};
    std::vector<double> acoef((size_t)nvar*N*5,0.0), Dsol((size_t)nvar*N,0.0),
                        qmnc((size_t)nvar*N), qmxc((size_t)nvar*N), kc(N);
    std::vector<char> hasint((size_t)nvar*N,0);
    #pragma omp parallel for schedule(dynamic,16)
    for(int ci=0;ci<N;++ci){ const auto& vs=m.cell_nodes[ci]; int nv=(int)vs.size();
        double perim=0; for(int f:m.cell_faces[ci]) perim+=m.face_areas[f]; double H=4.0*m.cell_volumes[ci]/std::max(perim,1e-30);
        double kk=beta/std::max(H,1e-30); kc[ci]=kk;
        for(int v=0;v<nvar;++v){ double qbar=W[(size_t)v*N+ci], qmn=1e300,qmx=-1e300;
            for(int vid:vs){ double a=vmn[(size_t)v*c.Nn+vid],b=vmx[(size_t)v*c.Nn+vid]; if(a<qmn)qmn=a; if(b>qmx)qmx=b; }
            qmnc[(size_t)v*N+ci]=qmn; qmxc[(size_t)v*N+ci]=qmx; double rng=qmx-qmn;
            double cbar = rng>1e-14 ? (qbar-qmn)/rng : 0.5;
            if(cbar<=1e-6||cbar>=1.0-1e-6||rng<=1e-14||nv<3){ hasint[(size_t)v*N+ci]=0; continue; }
            const double* G=&g[((size_t)v*N+ci)*5];
            double g0=G[0],g1=G[1],g2=G[2],g3=G[3],g4=G[4], Gm=std::sqrt(g0*g0+g1*g1);
            if(Gm<1e-30){ hasint[(size_t)v*N+ci]=0; continue; }
            double nx=g0/Gm, ny=g1/Gm, G3=Gm*Gm*Gm;
            double nxx=g2/Gm - g0*(g0*g2+g1*g4)/G3;   // d n_x/dx
            double nxy=g4/Gm - g0*(g0*g4+g1*g3)/G3;   // d n_x/dy
            double nyx=g4/Gm - g1*(g0*g2+g1*g4)/G3;   // d n_y/dx
            double nyy=g3/Gm - g1*(g0*g4+g1*g3)/G3;   // d n_y/dy
            double* A=&acoef[((size_t)v*N+ci)*5];
            A[0]=nx; A[1]=ny; A[2]=0.5*nxx; A[3]=0.5*nyy; A[4]=0.5*(nxy+nyx);
            // polygon fan cell quadrature (triangle -> bit-identical to the legacy 6-pt loop)
            double Pg_[C3_NQMAX], cwq[C3_NQMAX], Ag[C3_NQMAX];
            const int NQ=c3_cell_quad_P(TQ,6,m.nodes.data(),vs.data(),nv,cc[ci*2],cc[ci*2+1],A,Pg_,cwq);
            for(int q=0;q<NQ;++q) Ag[q]=std::tanh(kk*Pg_[q]);
            double Q=2.0*cbar-1.0, D=0.0;   // Newton on rational eq for D=tanh(k d)
            for(int it=0;it<25;++it){ double f=-Q,fp=0.0;
                for(int q=0;q<NQ;++q){ double den=1.0+Ag[q]*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12);
                    f+=cwq[q]*(Ag[q]+D)/den; fp+=cwq[q]*(1.0-Ag[q]*Ag[q])/(den*den); }
                if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD;
                if(D>0.999999)D=0.999999; else if(D<-0.999999)D=-0.999999;
                if(std::fabs(dD)<1e-12)break; }
            Dsol[(size_t)v*N+ci]=D; hasint[(size_t)v*N+ci]=1; }
    }
    WLf.assign((size_t)nvar*Nf,0.0); WRf.assign((size_t)nvar*Nf,0.0);
    #pragma omp parallel for
    for(int f=0;f<Nf;++f){ int o=m.face_owner[f],n=m.face_neighbour[f]; const auto& fn=m.face_nodes[f];
        for(int v=0;v<nvar;++v){ auto fv=[&](int ci)->double{ double qbar=W[(size_t)v*N+ci];
                if(!hasint[(size_t)v*N+ci]) return qbar;
                double qmn=qmnc[(size_t)v*N+ci],qmx=qmxc[(size_t)v*N+ci],rng=qmx-qmn;
                const double* A=&acoef[((size_t)v*N+ci)*5]; double kk=kc[ci], D=Dsol[(size_t)v*N+ci], th=0.0;
                if(fn.size()>=2){ double ax=m.nodes[fn[0]*2],ay=m.nodes[fn[0]*2+1],bx=m.nodes[fn[1]*2],by=m.nodes[fn[1]*2+1];
                    for(int q=0;q<4;++q){ double t=EQ[q][0],x=ax+t*(bx-ax),y=ay+t*(by-ay);
                        double Af=std::tanh(kk*Pval(A,x-cc[ci*2],y-cc[ci*2+1])); double den=1.0+Af*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12);
                        th+=EQ[q][1]*(Af+D)/den; } }
                else { double Af=std::tanh(kk*Pval(A,m.face_centers[f*2]-cc[ci*2],m.face_centers[f*2+1]-cc[ci*2+1])); double den=1.0+Af*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12); th=(Af+D)/den; }
                // NO clamp (Xie2017-faithful): th is a convex combination of tanh-addition values,
                // each strictly in (-1,1), so qf lies strictly inside (qmn,qmx) by construction.
                double qf=qmn+0.5*rng*(1.0+th); return qf; };
            WLf[(size_t)v*Nf+f]=fv(o); WRf[(size_t)v*Nf+f]=(n>=0)?fv(n):WLf[(size_t)v*Nf+f]; }
    }
}

// Optimized EXACT Cheng 2021 three-member MUSCL-THINC/QQ-BVD. MUSCL face values (WLs,WRs) come
// in precomputed. FUSED over the two THINC/QQ beta candidates: shared geometry (o2 LSQ g[],
// unit-normal P_i a_st, qmin/qmax, cell-quad P_g) is computed ONCE per interface cell (cbar in
// (eps,1-eps)) -> skips the o2 matvec for ~80% non-interface cells; only the beta-dependent
// Newton D-solve and edge tanh differ. Loop-separated for cache locality: cell loops touch
// per-cell arrays, face loops touch per-face arrays (no monolithic loop). Per-variable min-TBV
// selection (Deng 2018). Results identical to two separate reconstruct_thinc_qq calls.
// Fast atan (range-reduced degree-9 minimax, |err|<2e-6, ~3x faster than libm atan).
inline double fast_atan(double x){
    double ax=std::fabs(x), z=ax>1.0?1.0/ax:ax, z2=z*z;
    double r=z*(0.99997726+z2*(-0.33262347+z2*(0.19354346+z2*(-0.11643287+z2*(0.05265332-z2*0.01172120)))));
    if(ax>1.0) r=1.5707963267948966-r;
    return x<0.0?-r:r;
}
// Cheaper atan (Rajan |z|<=1 form, |err|~1.5e-3). Used (opt-in) ONLY on the large-disc J-path
// where the (2/s) prefactor does not amplify the error; small-disc keeps the accurate fast_atan.
inline double fast_atan_cheap(double x){
    double ax=std::fabs(x), z=ax>1.0?1.0/ax:ax;
    double r=0.78539816339744831*z + z*(1.0-z)*(0.2447+0.0663*z);
    if(ax>1.0) r=1.5707963267948966-r;
    return x<0.0?-r:r;
}
// Fast log (frexp exponent split + degree-5 minimax on mantissa, |err|<1e-6, ~2-3x faster).
inline double fast_log(double x){
    if(x<=0.0) return -1e30;
    int e; double m=std::frexp(x,&e);            // x = m*2^e, m in [0.5,1)
    if(m<0.7071067811865476){ m*=2.0; --e; }     // center mantissa around 1
    double f=m-1.0;
    double r=f*(0.9999964+f*(-0.4998741+f*(0.3317990+f*(-0.2407338+f*0.1676540))));
    return r+(double)e*0.6931471805599453;       // + e*ln2
}

// SPLICED sigmoid: tanh-hugging, closed-form, asymptotic tail. Inner |u|<=a: Pade[3/2] of tanh
// R(u)=u(15+u^2)/(15+6u^2) (slope 1 at 0, hugs tanh to O(u^7)). Outer |u|>a: asymptotic rational tail
// 1-A/(1+k(u-a)) (->1, never reaches = high-beta stable), C1-matched at a=1.5. Odd: sign(s)*g(|s|).
// QUINTIC-HERMITE spline of tanh: knots u=0,1,2,3,4 matching tanh value+1st+2nd deriv (C2, 4 quintic
// pieces) + C2 asymptotic rational tail for u>4. C2 => curvature continuous at knots (no carbuncle seed).
// Polynomial pieces -> polynomial QQ integral (no arctan), quintic inverse -> small scalar solve. Odd.
// Shared per-segment POWER coeffs of the quintic-Hermite spliced sigmoid (built ONCE). On |xi| in
// [k,k+1] (k=0..3), tau=|xi|-k: value = Horner(cv[k],tau) (~10 ops); derivative = Horner(cd[k],tau).
// cd = d/dtau of cv. Replaces the 6-Hermite-basis evaluation (~36 ops) in the hot Newton loop.
struct SplCoef { double cv[4][6]; double cd[4][5];
    SplCoef(){ const double Y[5]={0.0,0.76159416,0.96402758,0.99505475,0.99932930};
        const double M[5]={1.0,0.41997434,0.07065082,0.00986604,0.00134095};
        const double S[5]={0.0,-0.63969420,-0.13621900,-0.01963450,-0.00268010};
        const double H[6][6]={ {1,0,0,-10,15,-6},{0,1,0,-6,8,-3},{0,0,0.5,-1.5,1.5,-0.5},
                               {0,0,0,10,-15,6},{0,0,0,-4,7,-3},{0,0,0,0.5,-1,0.5} };  // Hermite basis power coeffs
        for(int k=0;k<4;++k){ for(int i=0;i<6;++i)
                cv[k][i]=Y[k]*H[0][i]+M[k]*H[1][i]+S[k]*H[2][i]+Y[k+1]*H[3][i]+M[k+1]*H[4][i]+S[k+1]*H[5][i];
            for(int i=0;i<5;++i) cd[k][i]=(double)(i+1)*cv[k][i+1]; } } };
inline const SplCoef& spl_coef(){ static const SplCoef sc; return sc; }
// Sigmoid SELECTOR for the comparison study (default=quintic-Hermite). Each is slope-1 at origin (same beta).
inline int spl_mode(){ static int m=[](){
    if(std::getenv("THINCQQ_SIG_PADETAIL")) return 1;   // Pade[5/4]-of-tanh inner + rational tail (2-piece splice)
    if(std::getenv("THINCQQ_SIG_OPTRAT"))   return 2;   // coefficient-optimized rational deg-6 (single formula)
    if(std::getenv("THINCQQ_SIG_ARCTAN"))   return 3;   // arctan-quintic (2/pi)atan(P5), slope-1 normalized
    if(std::getenv("THINCQQ_SIG_COMPACT"))  return 4;   // compact rational tanh R5 (Pade[5/4], C2 clamp, L=5)
    if(std::getenv("THINCQQ_SIG_PADE32"))   return 5;   // Pade[3/2]-of-tanh inner (deg-2 denom, cheap) + rational tail, a=2
    if(std::getenv("THINCQQ_SIG_PADEU"))     return 6;   // USER spec: Pade[3/2] inner + G/(quadratic) tail, splice a=1.5 (C2, tail~1/s^2, clean Psi primitives)
    if(std::getenv("THINCQQ_SIG_POLY9"))     return 7;   // USER spec: compact deg-9 polynomial smoothstep (C2 clamp at S=2.682), NO div/log/atan/exp -> pure-poly edge moment
    if(std::getenv("THINCQQ_SIG_DEG3T"))     return 9;   // deg3t: cubic inner + rational m=3 tail (gentle, C2, no hard clamp): centroid-D Cardano + CF face (poly inner + atan/log tail)
    if(std::getenv("THINCQQ_SIG_DEG3"))      return 8;   // deg-3 cubic smoothstep (C1, S=1.5): centroid-D Cardano (NO Newton) + pure-poly CF face (NO quadrature)
    return 0; }(); return m; }
// deg3t (mode 9) coefficients -- SINGLE SOURCE OF TRUTH (C2-matched, tanh err 0.025): cubic inner u + a3 u^3 (|u|<asp),
// rational m=3 tail 1-(A+Bw)/(1+kk w)^3 (w=|u|-asp). Used by spl_sig_and_d case9, deg3t_edge_avg, deg3t_inv.
inline constexpr double DEG3T_A3=-0.2175, DEG3T_ASP=0.7946, DEG3T_A=0.3145201473, DEG3T_B=0.8313725062, DEG3T_K=1.5042926943;  // A,B,K = EXACT C2-match (splice jumps -> machine 0; was 4-digit rounded: C0 2e-5/C1 2.1e-4 kink)
inline void spl_sig_and_d(double s, double& g, double& gd){   // value g (ODD), derivative gd (EVEN). All modes slope-1 at 0.
    switch(spl_mode()){
    case 1:{ double u=s<0.0?-s:s, gg,gp;                 // Pade[5/4] inner + rational tail
        if(u<3.0){ double u2=u*u,u3=u2*u; double N=945.0*u+105.0*u3+u2*u3, D=945.0+420.0*u2+15.0*u2*u2;
            double Np=945.0+315.0*u2+5.0*u2*u2, Dp=840.0*u+60.0*u3; gg=N/D; gp=(Np*D-N*Dp)/(D*D); }
        else { const double A=0.004550,B=0.027970,kk=4.25860; double w=u-3.0,kw=1.0+kk*w,kw2=kw*kw;
            gg=1.0-(A+B*w)/kw2; gp=-(B*kw-2.0*kk*(A+B*w))/(kw2*kw); }
        g=s<0.0?-gg:gg; gd=gp; return; }
    case 2:{ double u=s<0.0?-s:s, u2=u*u,u3=u2*u,u4=u2*u2,u5=u4*u,u6=u3*u3;   // opt rational deg-6
        const double a3=0.521955,a4=0.862627,a5=-0.476668,a6=0.286677;
        double D=1.0+u+u2+a3*u3+a4*u4+a5*u5+a6*u6;
        double Dp=1.0+2.0*u+3.0*a3*u2+4.0*a4*u3+5.0*a5*u4+6.0*a6*u5;
        double gg=1.0-1.0/D; g=s<0.0?-gg:gg; gd=Dp/(D*D); return; }
    case 3:{ const double TPI=0.63661977236758134, s0=1.1936620731892150;    // arctan-quintic, slope-1
        double z=s/s0, z2=z*z, P=z*(3.0*z2*z2+10.0*z2+15.0)/8.0, Pp=15.0*(z2+1.0)*(z2+1.0)/8.0;
        g=TPI*std::atan(P); gd=TPI*Pp/(1.0+P*P)/s0; return; }
    case 4:{ double a=s<0.0?-s:s; const double L=5.0;     // compact rational tanh R5 (Pade[5/4] C2 clamp)
        if(a>=L){ g=s<0.0?-1.0:1.0; gd=0.0; return; }
        double z=s/L, z2=z*z, z4=z2*z2, N=5.0*z+10.0*z*z2+z*z4, D=1.0+10.0*z2+5.0*z4;
        double Np=5.0+30.0*z2+5.0*z4, Dp=20.0*z+20.0*z*z2; g=N/D; gd=(Np*D-N*Dp)/(D*D)/L; return; }
    case 5:{ double u=s<0.0?-s:s, gg,gp;                 // Pade[3/2] inner (deg-2 denom) + rational tail, a=2
        if(u<2.0){ double u2=u*u; double N=15.0*u+u*u2, D=15.0+6.0*u2, Np=15.0+3.0*u2, Dp=12.0*u; gg=N/D; gp=(Np*D-N*Dp)/(D*D); }
        else { const double A=0.025641,B=0.262785,kk=6.932009; double w=u-2.0,kw=1.0+kk*w,kw2=kw*kw;
            gg=1.0-(A+B*w)/kw2; gp=-(B*kw-2.0*kk*(A+B*w))/(kw2*kw); }
        g=s<0.0?-gg:gg; gd=gp; return; }
    case 6:{ double u=s<0.0?-s:s, gg,gp;                 // USER spec: Pade[3/2] inner (a=1.5) + 1-G/(1+lam t+mu t^2) tail
        if(u<=1.5){ double u2=u*u; double N=15.0*u+u*u2, D=15.0+6.0*u2, Np=15.0+3.0*u2, Dp=12.0*u; gg=N/D; gp=(Np*D-N*Dp)/(D*D); }
        else { const double G=7.0/76.0,lam=274.0/133.0,mu=2404.0/931.0; double t=u-1.5,Dd=1.0+lam*t+mu*t*t,Dp2=lam+2.0*mu*t;
            gg=1.0-G/Dd; gp=G*Dp2/(Dd*Dd); }
        g=s<0.0?-gg:gg; gd=gp; return; }
    case 7:{ const double S=2.68179418,a3=-0.2790604680,a5=0.0571770650,a7=-0.00595679733,a9=0.000238602530;  // compact deg-9 poly smoothstep
        if(s>=S){ g=1.0; gd=0.0; return; } if(s<=-S){ g=-1.0; gd=0.0; return; }
        double s2=s*s; g=s*(1.0+s2*(a3+s2*(a5+s2*(a7+a9*s2))));
        gd=1.0+s2*(3.0*a3+s2*(5.0*a5+s2*(7.0*a7+9.0*a9*s2))); return; }
    case 8:{ const double a1=0.8508,S=3.0/(2.0*a1),a3=-0.5/(S*S*S);  // deg-3 SOFTENED (a1=origin slope<1; S,a3 from a1 -> psi(S)=1 exact)
        if(s>=S){ g=1.0; gd=0.0; return; } if(s<=-S){ g=-1.0; gd=0.0; return; }
        double s2=s*s; g=s*(a1+a3*s2); gd=a1+3.0*a3*s2; return; }
    case 9:{ constexpr double a3=DEG3T_A3,asp=DEG3T_ASP,A=DEG3T_A,B=DEG3T_B,kk=DEG3T_K;  // deg3t: cubic inner + rational m=3 tail (gentle)
        double u=s<0.0?-s:s, gg,gp;
        if(u<asp){ double u2=u*u; gg=u+a3*u2*u; gp=1.0+3.0*a3*u2; }
        else { double w=u-asp, KW=1.0+kk*w, KW2=KW*KW, KW4=KW2*KW2; gg=1.0-(A+B*w)/(KW2*KW);
            gp=(3.0*kk*(A+B*w)-B*KW)/KW4; }
        g=s<0.0?-gg:gg; gd=gp; return; }
    default:{ double u=s<0.0?-s:s;                        // quintic-Hermite (default)
        if(u<4.0){ int k=(int)u; double t=u-(double)k; const SplCoef& sc=spl_coef(); const double* c=sc.cv[k]; const double* d=sc.cd[k];
            double gg=((((c[5]*t+c[4])*t+c[3])*t+c[2])*t+c[1])*t+c[0];
            gd=(((d[4]*t+d[3])*t+d[2])*t+d[1])*t+d[0]; g=s<0.0?-gg:gg; return; }
        const double A=0.00067070,B=0.0032380,kt=3.4135; double w=u-4.0,kw=1.0+kt*w,kw2=kw*kw;
        double gg=1.0-(A+B*w)/kw2; g=s<0.0?-gg:gg; gd=-(B*kw-2.0*kt*(A+B*w))/(kw2*kw); return; }
    } }
inline double spl_sig(double s){   // VALUE-ONLY (face evals don't need derivative; avoids wasteful gd compute)
    switch(spl_mode()){
    case 5:{ double u=s<0.0?-s:s,gg; if(u<2.0){double u2=u*u;gg=(15.0*u+u*u2)/(15.0+6.0*u2);}
        else{const double A=0.025641,B=0.262785,kk=6.932009;double w=u-2.0,kw=1.0+kk*w;gg=1.0-(A+B*w)/(kw*kw);} return s<0.0?-gg:gg; }
    case 6:{ double u=s<0.0?-s:s,gg; if(u<=1.5){double u2=u*u;gg=(15.0*u+u*u2)/(15.0+6.0*u2);}
        else{const double G=7.0/76.0,lam=274.0/133.0,mu=2404.0/931.0;double t=u-1.5;gg=1.0-G/(1.0+lam*t+mu*t*t);} return s<0.0?-gg:gg; }
    case 7:{ const double S=2.68179418,a3=-0.2790604680,a5=0.0571770650,a7=-0.00595679733,a9=0.000238602530;
        if(s>=S)return 1.0; if(s<=-S)return -1.0; double s2=s*s; return s*(1.0+s2*(a3+s2*(a5+s2*(a7+a9*s2)))); }
    default:{ double g,gd; spl_sig_and_d(s,g,gd); return g; } } }
inline double spl_sig_d(double s){ double g,gd; spl_sig_and_d(s,g,gd); return gd; }

// SPL closed-form edge integral: EXACT edge-average of the quintic-Hermite spl_sig over xi(t)=a t^2+b t+g
// (t in [0,1]). NO quadrature. Split [0,1] at every t where xi(t)=K (integer knot K in -4..4); on each
// sub-interval xi stays in ONE Hermite segment -> spl_sig = sg*(quintic in |xi|) -> polynomial in t
// (deg<=10) -> EXACT integral (moment sum). Tail |xi|>=4 -> spl_sig ~ +-1 (val 0.99933 at 4; err<7e-4).
inline double spl_edge_avg(double a, double b, double g){
    const double (*D5)[6] = spl_coef().cv;   // shared per-segment value power coeffs (same table the sigmoid uses)
    // integrate sg*value(|xi|) over [t1,t2] in segment k, REMAPPED s=(t-t1)/dt in [0,1] (coeffs stay O(1) -> stable)
    auto seg_int=[&](double t1,double t2,double sg,int k)->double{
        double dt=t2-t1, al=sg*a, be=sg*b, ga=sg*g-(double)k;             // tau(t)=al t^2+be t+ga
        double Ap=al*dt*dt, Bp=dt*(2.0*al*t1+be), Gp=al*t1*t1+be*t1+ga;   // tau(t1+dt*s)=Ap s^2+Bp s+Gp
        double poly[11]={0,0,0,0,0,0,0,0,0,0,0}, cur[11]={1,0,0,0,0,0,0,0,0,0,0}; int cd=0;
        for(int i=0;i<=5;++i){ double di=D5[k][i];
            if(di!=0.0) for(int z=0;z<=cd;++z) poly[z]+=di*cur[z];
            if(i<5){ double nx[11]={0,0,0,0,0,0,0,0,0,0,0};
                for(int z=0;z<=cd;++z){ nx[z]+=Gp*cur[z]; nx[z+1]+=Bp*cur[z]; nx[z+2]+=Ap*cur[z]; }
                for(int z=0;z<11;++z)cur[z]=nx[z]; cd+=2; } }
        double seg=0.0; for(int z=0;z<=10;++z) seg+=poly[z]/(double)(z+1);   // int_0^1 s^z ds = 1/(z+1)
        return sg*seg*dt; };
    double xi0=g, xi1=a+b+g, xmn=xi0<xi1?xi0:xi1, xmx=xi0<xi1?xi1:xi0;   // parabola range over [0,1]
    double tv=-2.0; bool hasv=false;
    if(std::fabs(a)>1e-300){ tv=-b/(2.0*a); if(tv>1e-9&&tv<1.0-1e-9){ hasv=true;   // vertex -> each piece monotone in xi
        double xv=a*tv*tv+b*tv+g; if(xv<xmn)xmn=xv; if(xv>xmx)xmx=xv; } }
    int Klo=(int)std::ceil(xmn), Khi=(int)std::floor(xmx); if(Klo<-4)Klo=-4; if(Khi>4)Khi=4;   // only knots the parabola crosses
    if(Khi<Klo){   // FAST PATH (~70-80% of edges): no knot crossing -> ONE segment, one sign, integrate [0,1] directly
        double xm=0.25*a+0.5*b+g, axm=xm<0.0?-xm:xm, sg=(xm>=0.0?1.0:-1.0);
        if(axm>=4.0) return sg;                                  // tail: int over [0,1] = sg
        int k=(int)axm; if(k>3)k=3; return seg_int(0.0,1.0,sg,k); }
    double bp[26]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0; if(hasv)bp[nb++]=tv;
    for(int K=Klo;K<=Khi;++K){ double c=g-(double)K;          // xi(t)-K = a t^2 + b t + c
        if(std::fabs(a)<1e-13){ if(std::fabs(b)>1e-300){ double r=-c/b; if(r>1e-9&&r<1.0-1e-9&&nb<24)bp[nb++]=r; } }
        else { double d=b*b-4.0*a*c; if(d>0.0){ double s=std::sqrt(d),r1=(-b-s)/(2.0*a),r2=(-b+s)/(2.0*a);
            if(r1>1e-9&&r1<1.0-1e-9&&nb<24)bp[nb++]=r1; if(r2>1e-9&&r2<1.0-1e-9&&nb<24)bp[nb++]=r2; } } }
    for(int i=1;i<nb;++i){ double x=bp[i]; int j=i-1; while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;} bp[j+1]=x; }
    double I=0.0;
    for(int sgi=0;sgi<nb-1;++sgi){ double t1=bp[sgi],t2=bp[sgi+1]; if(t2-t1<1e-12)continue;
        double tm=0.5*(t1+t2), xm=a*tm*tm+b*tm+g, axm=xm<0.0?-xm:xm, sg=(xm>=0.0?1.0:-1.0);
        if(axm>=4.0){ I += sg*(t2-t1); continue; }            // tail -> +-1
        int k=(int)axm; if(k>3)k=3; I += seg_int(t1,t2,sg,k); }
    return I;
}
// ASIG closed-form edge integral: EXACT edge-average of the ALGEBRAIC-sigmoid profile
// th(xi)=xi/(1+|xi|), xi(t)=a t^2+b t+g (t in [0,1]). Split at roots of xi=0 (interface-center
// crossings); on each piece th=+-1 -/+ 1/(1+-xi) (rational) -> ELEMENTARY integral (partial
// fractions: int dt/(quadratic) = arctan or log). Quadrature-free, Newton-free. ASYMPTOTIC
// (non-compact) -> stable at high beta where the compact polynomial smoothstep (pst) diverges.
inline double asig_edge_avg(double a, double b, double g){
    static const bool AFA = std::getenv("THINCQQ_ASIG_FASTATAN") != nullptr;  // opt-in cheaper atan on large-disc path
    static const bool ADAPT = std::getenv("THINCQQ_ASIG_ADAPT") != nullptr;   // curvature-adaptive: drop near-straight parabola -> cheap log/linear branch, keep curve(arctan) on genuinely-curved edges (accuracy-exact when bow<tol)
    static const char* _atol_s = std::getenv("THINCQQ_ASIG_ADAPT_TOL");
    static const double ATOL = _atol_s ? std::atof(_atol_s) : 0.1;             // |a|<ATOL -> parabola bows < ATOL/4 in xi-units from its chord -> indistinguishable from planar
    if(ADAPT && std::fabs(a) < ATOL) a=0.0;
    auto J=[&](double A,double B,double C,double t1,double t2)->double{   // ∫ dt/(A t^2+B t+C)
        if(std::fabs(A)<1e-14){ if(std::fabs(B)<1e-300) return (t2-t1)/C;
            return (fast_log(std::fabs(B*t2+C))-fast_log(std::fabs(B*t1+C)))/B; }
        double disc=4.0*A*C-B*B;
        if(disc>1e-300){ double s=std::sqrt(disc);
            // atan(u2/s)-atan(u1/s) via the subtraction identity -> ONE atan (atan is the bottleneck).
            double u1=2.0*A*t1+B, u2=2.0*A*t2+B, den=s*s+u1*u2;
            if(std::fabs(den)<1e-300) return (2.0/s)*(fast_atan(u2/s)-fast_atan(u1/s));   // rare fallback
            double arg=(u2-u1)*s/den;
            // small disc -> (2/s) amplifies atan error -> keep accurate; else opt-in cheap atan.
            double v=(AFA && disc>1e-4)?fast_atan_cheap(arg):fast_atan(arg);
            if(den<0.0) v += (u2>=0.0? 3.14159265358979324 : -3.14159265358979324);       // branch correction
            return (2.0/s)*v; }
        if(disc<-1e-300){ double s=std::sqrt(-disc);
            auto L=[&](double t){ double u=2.0*A*t+B; return fast_log(std::fabs((u-s)/(u+s))); };
            return (L(t2)-L(t1))/s; }
        return -2.0/(2.0*A*t2+B)+2.0/(2.0*A*t1+B); };
    // FAST PATH: if xi(t) keeps one sign on [0,1] (interface center not crossing this edge = most
    // edges), it's ONE piece -> no root-finding/sqrt. Range of xi from endpoints + vertex.
    double xi0=g, xi1=a+b+g, xmn=std::min(xi0,xi1), xmx=std::max(xi0,xi1);
    if(std::fabs(a)>1e-300){ double tv=-b/(2.0*a); if(tv>0.0&&tv<1.0){ double xv=a*tv*tv+b*tv+g;
        if(xv<xmn)xmn=xv; if(xv>xmx)xmx=xv; } }
    if(xmn>=0.0) return 1.0 - J(a,b,1.0+g,0.0,1.0);          // xi>0 throughout: sigma=1-1/(1+xi)
    if(xmx<=0.0) return -1.0 + J(-a,-b,1.0-g,0.0,1.0);       // xi<0 throughout: sigma=-1+1/(1-xi)
    // SLOW PATH (rare): xi crosses 0 -> split.
    double bp[4]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0;
    if(std::fabs(a)<1e-13){ if(std::fabs(b)>1e-300){ double r=-g/b; if(r>1e-9&&r<1.0-1e-9)bp[nb++]=r; } }
    else { double d=b*b-4.0*a*g; if(d>0.0){ double s=std::sqrt(d),r1=(-b-s)/(2.0*a),r2=(-b+s)/(2.0*a);
        if(r1>1e-9&&r1<1.0-1e-9)bp[nb++]=r1; if(r2>1e-9&&r2<1.0-1e-9)bp[nb++]=r2; } }
    for(int i=1;i<nb;++i){ double x=bp[i]; int j=i-1; while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;} bp[j+1]=x; }
    double I=0.0;
    for(int s=0;s<nb-1;++s){ double t1=bp[s],t2=bp[s+1],dt=t2-t1; if(dt<1e-12)continue;
        double tm=0.5*(t1+t2), xm=a*tm*tm+b*tm+g;
        if(xm>=0.0) I += dt - J(a,b,1.0+g,t1,t2);       // xi>0: sigma = 1 - 1/(1+xi)
        else        I += -dt + J(-a,-b,1.0-g,t1,t2); }  // xi<0: sigma = -1 + 1/(1-xi)
    return I;
}

// ARAT closed-form edge integral: EXACT-to-curvature edge-average of the RATIONAL sigmoid
// sigma(xi)=sign(xi)(1-1/(1+|xi|)^2), xi(t)=a t^2+b t+g (t in [0,1]). The curve (quadratic xi) is
// kept via a PERTURBATIVE expansion of the integrand about the edge chord u_c=1+xi_chord (linear):
//   1/u^2 = 1/u_c^2 - 2(a t(t-1))/u_c^3 + 3(a t(t-1))^2/u_c^4 - ...   [a t(t-1) = parabola minus chord]
// First/second-order terms integrate to PURE RATIONAL (I0) + curvature correction (1 log + rational).
// => NO sqrt, NO arctan, NO quadrature loop, NO Newton. Flat edge (a~0) -> I0 only = ZERO transcendental.
inline double arat_edge_avg(double a, double b, double g){
    static const bool ADAPT = std::getenv("THINCQQ_ARAT_ADAPT") != nullptr;        // skip curvature corr when near-straight
    static const char* _t=std::getenv("THINCQQ_ARAT_ADAPT_TOL"); static const double ATOL=_t?std::atof(_t):0.0;
    static const bool O2 = std::getenv("THINCQQ_ARAT_O2")!=nullptr;                // 2nd-order curvature corr (opt-in; default 1st-order = robust, avoids 1/q^4 cancellation)
    // EApos: int_0^1 [1 - 1/(1+xi)^2] dt, perturbative in curvature a. Assumes 1+xi>0 on [0,1] (true on a one-signed positive piece).
    auto EApos=[&](double A,double B,double G)->double{
        double p=1.0+G, q=A+B, pe=p+q;                          // u_c = p + q t (chord of 1+xi)
        double I0,corr=0.0;
        if(std::fabs(q)<1e-12){                                 // flat chord: direct moments of t(t-1)^k
            double ip2=1.0/(p*p); I0=ip2;
            if(!(ADAPT&&std::fabs(A)<ATOL)&&std::fabs(A)>0.0){
                double ip3=ip2/p; corr = 2.0*A*(-1.0/6.0)*ip3;  // +2a I1, I1=int t(t-1)/p^3 = -1/(6 p^3)
                if(O2){ double ip4=ip3/p; corr -= 3.0*A*A*(1.0/30.0)*ip4; } } // -3a^2 I2c, I2c=int [t(t-1)]^2/p^4 = 1/(30 p^4)
            return 1.0 - I0 + corr; }
        double iq=1.0/q, ip=1.0/p, ipe=1.0/pe;
        I0 = iq*(ip-ipe);                                       // int 1/u_c^2 dt
        if(!(ADAPT&&std::fabs(A)<ATOL)&&std::fabs(A)>0.0){
            double Lq  = iq*fast_log(pe*ip);                    // int 1/u_c dt
            double I3  = iq*0.5*(ip*ip-ipe*ipe);                // int 1/u_c^3 dt
            double I1  = (iq*iq)*( Lq - (2.0*p+q)*I0 + (p*p+p*q)*I3 ); // int t(t-1)/u_c^3 dt
            corr = 2.0*A*I1;
            if(O2){ double I4=iq*(1.0/3.0)*(ip*ip*ip-ipe*ipe*ipe);    // int 1/u_c^4 dt
                // int [t(t-1)]^2/u_c^4 dt, [t(t-1)]^2=t^4-2t^3+t^2; reuse u_c=p+qt moments via (t^2-t)^2=(1/q^4)[u_c^2-(2p+q)u_c+(p^2+pq)]^2
                double r1=1.0, r2=-(2.0*p+q), r3=(p*p+p*q);     // (t^2-t)=(1/q^2)(r1 u^2+r2 u+r3)
                // [(t^2-t)]^2 = (1/q^4)( r1^2 u^4 + 2 r1 r2 u^3 + (r2^2+2 r1 r3) u^2 + 2 r2 r3 u + r3^2 )
                double c4=r1*r1, c3=2.0*r1*r2, c2=r2*r2+2.0*r1*r3, c1=2.0*r2*r3, c0=r3*r3;
                // /u_c^4 -> c4/u^0(=int dt) + c3/u + c2/u^2 + c1/u^3 + c0/u^4 ; int over [0,1]
                double Idt=1.0;                                 // int u^0 dt over [0,1] (in t) = 1
                double I2c=(1.0/(q*q*q*q))*( c4*Idt + c3*Lq + c2*I0 + c1*I3 + c0*I4 );
                corr -= 3.0*A*A*I2c; } }   // -3a^2 I2c
        return 1.0 - I0 + corr; };
    // fast path: xi one sign on [0,1]? (interface center not crossing this edge = most edges)
    double xi0=g, xi1=a+b+g, xmn=std::min(xi0,xi1), xmx=std::max(xi0,xi1);
    if(std::fabs(a)>1e-300){ double tv=-b/(2.0*a); if(tv>0.0&&tv<1.0){ double xv=a*tv*tv+b*tv+g; if(xv<xmn)xmn=xv; if(xv>xmx)xmx=xv; } }
    if(xmn>=0.0) return EApos(a,b,g);
    if(xmx<=0.0) return -EApos(-a,-b,-g);
    // crossing (rare): split at roots of xi=0; remap each piece [t1,t2]->[0,1] (a'=a*dt^2, b'=dt*(2 a t1+b), g'=xi(t1)).
    double bp[4]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0;
    if(std::fabs(a)<1e-13){ if(std::fabs(b)>1e-300){ double r=-g/b; if(r>1e-9&&r<1.0-1e-9)bp[nb++]=r; } }
    else { double d=b*b-4.0*a*g; if(d>0.0){ double s=std::sqrt(d),r1=(-b-s)/(2.0*a),r2=(-b+s)/(2.0*a);
        if(r1>1e-9&&r1<1.0-1e-9)bp[nb++]=r1; if(r2>1e-9&&r2<1.0-1e-9)bp[nb++]=r2; } }
    for(int i=1;i<nb;++i){ double x=bp[i]; int j=i-1; while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;} bp[j+1]=x; }
    double I=0.0;
    for(int s=0;s<nb-1;++s){ double t1=bp[s],t2=bp[s+1],dt=t2-t1; if(dt<1e-12)continue;
        double Ap=a*dt*dt, Bp=dt*(2.0*a*t1+b), Gp=a*t1*t1+b*t1+g; double tm=0.5*(t1+t2),xm=a*tm*tm+b*tm+g;
        if(xm>=0.0) I += dt*EApos(Ap,Bp,Gp);
        else        I += -dt*EApos(-Ap,-Bp,-Gp); }
    return I;
}

// pade32 closed-form edge-average: sigma(xi)=sign(xi)*[ inner (15u+u^3)/(15+6u^2), u<2 ; tail 1-(A+Bw)/(1+kk w)^2, u>=2 ].
// xi(t)=a t^2+b t+g over [0,1]. exact = int_0^1 sigma(xi) dt. Split at xi=+-2; inner via complex-J (bounded, safe),
// tail via real robust ∫dt/P, ∫dt/P^2. Degenerate (a~0 linear xi / P linear / double root) handled -> no NaN.
inline double p32_Jr(double A,double B,double C,double t1,double t2){   // int dt/(A t^2+B t+C)
    if(std::fabs(A)<1e-14){ if(std::fabs(B)<1e-300) return (t2-t1)/C;
        return (fast_log(std::fabs(B*t2+C))-fast_log(std::fabs(B*t1+C)))/B; }
    double disc=4.0*A*C-B*B;
    if(disc>1e-300){ double sq=std::sqrt(disc); double u1=2*A*t1+B,u2=2*A*t2+B,den=sq*sq+u1*u2,v;
        if(std::fabs(den)<1e-300) v=std::atan(u2/sq)-std::atan(u1/sq);
        else { v=std::atan((u2-u1)*sq/den); if(den<0) v+=(u2>=0?3.14159265358979324:-3.14159265358979324); }
        return 2.0/sq*v; }
    if(disc<-1e-300){ double sq=std::sqrt(-disc);
        auto L=[&](double t){ double u=2*A*t+B; return fast_log(std::fabs((u-sq)/(u+sq))); };
        return (L(t2)-L(t1))/sq; }
    return -2.0/(2*A*t2+B)+2.0/(2*A*t1+B); }
inline double p32_I2r(double A,double B,double C,double t1,double t2){  // int dt/(A t^2+B t+C)^2
    if(std::fabs(A)<1e-14){ if(std::fabs(B)<1e-300) return (t2-t1)/(C*C);
        return -1.0/(B*(B*t2+C)) + 1.0/(B*(B*t1+C)); }
    double disc=4.0*A*C-B*B;
    if(std::fabs(disc)<1e-300){ double r=-B/(2*A);
        return -1.0/(3*A*A)*(1.0/std::pow(t2-r,3)-1.0/std::pow(t1-r,3)); }
    auto P=[&](double t){ return A*t*t+B*t+C; };
    return (2*A*t2+B)/(disc*P(t2)) - (2*A*t1+B)/(disc*P(t1)) + (2*A/disc)*p32_Jr(A,B,C,t1,t2); }
inline double p32_I3r(double A,double B,double C,double t1,double t2){  // int dt/(A t^2+B t+C)^3 (for deg3t m=3 tail)
    if(std::fabs(A)<1e-14){ if(std::fabs(B)<1e-300) return (t2-t1)/(C*C*C);
        double d2=B*t2+C,d1=B*t1+C; return -1.0/(2*B*d2*d2)+1.0/(2*B*d1*d1); }
    double disc=4.0*A*C-B*B;
    if(std::fabs(disc)<1e-300){ double r=-B/(2*A);
        return -1.0/(5*A*A*A)*(1.0/std::pow(t2-r,5)-1.0/std::pow(t1-r,5)); }
    auto P=[&](double t){ return A*t*t+B*t+C; };
    return (2*A*t2+B)/(2*disc*P(t2)*P(t2)) - (2*A*t1+B)/(2*disc*P(t1)*P(t1)) + (3*A/disc)*p32_I2r(A,B,C,t1,t2); }
inline double p32_innerI(double a,double b,double g,double t1,double t2){  // int sigma_inner dt
    double poly=a/3.0*(t2*t2*t2-t1*t1*t1)+b/2.0*(t2*t2-t1*t1)+g*(t2-t1);
    const double c=2.5; double frac;
    if(std::fabs(a)<1e-13){ if(std::fabs(b)<1e-300) frac=g/(g*g+c)*(t2-t1);
        else { double x2=b*t2+g,x1=b*t1+g; frac=(fast_log(x2*x2+c)-fast_log(x1*x1+c))/(2*b); } }
    else { std::complex<double> bb=std::sqrt(std::complex<double>(g,-std::sqrt(c))/a - std::complex<double>(b*b/(4*a*a),0.0));
        double u1=t1+b/(2*a),u2=t2+b/(2*a);
        frac=((std::atan(std::complex<double>(u2,0)/bb)-std::atan(std::complex<double>(u1,0)/bb))/(a*bb)).real(); }
    return poly/6.0 + (25.0/12.0)*frac; }
inline double p32_tailI(double a,double b,double g,double sg,double t1,double t2){  // int sigma_tail dt (|xi|>2, sign sg)
    const double A=0.025641,B=0.262785,kk=6.932009;
    double PA=sg*kk*a, PB=sg*kk*b, PC=sg*kk*g+(1.0-2.0*kk);   // P=1+kk w
    double frac=(A-B/kk)*p32_I2r(PA,PB,PC,t1,t2) + (B/kk)*p32_Jr(PA,PB,PC,t1,t2);
    return sg*((t2-t1)-frac); }
inline double pade32_edge_avg(double a,double b,double g){
    double bp[8]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0;
    for(int sgn=-1;sgn<=1;sgn+=2){ double C=g-2.0*sgn;
        if(std::fabs(a)<1e-13){ if(std::fabs(b)>1e-300){ double r=-C/b; if(r>1e-9&&r<1-1e-9)bp[nb++]=r; } }
        else { double d=b*b-4*a*C; if(d>0){ double s=std::sqrt(d),r1=(-b-s)/(2*a),r2=(-b+s)/(2*a);
            if(r1>1e-9&&r1<1-1e-9)bp[nb++]=r1; if(r2>1e-9&&r2<1-1e-9)bp[nb++]=r2; } } }
    for(int i=1;i<nb;++i){ double x=bp[i];int j=i-1;while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;}bp[j+1]=x; }
    double I=0.0;
    for(int s=0;s<nb-1;++s){ double t1=bp[s],t2=bp[s+1],dt=t2-t1; if(dt<1e-12)continue;
        double tm=0.5*(t1+t2),xm=a*tm*tm+b*tm+g;
        if(std::fabs(xm)<=2.0) I+=p32_innerI(a,b,g,t1,t2);
        else I+=p32_tailI(a,b,g,(xm<0?-1.0:1.0),t1,t2); }
    return I<-1.0?-1.0:(I>1.0?1.0:I); }
// pade32 CENTROID-D inverse: D s.t. sigma_pade32(D)=Q, fully closed-form (NO Newton).
// inner |Q|<38/39: cubic u^3-6q u^2+15u-15q=0 (Cardano, p=15-12q^2>3>0 -> single real root). tail: quadratic in P=1+kk w.
inline double pade32_inv(double Q){
    double q=Q<0?-Q:Q; if(q>=1.0) return Q<0?-50.0:50.0; double u;
    if(q<38.0/39.0){ double p=15.0-12.0*q*q, r=15.0*q-16.0*q*q*q;
        double disc=0.25*r*r+p*p*p/27.0, sq=std::sqrt(disc);
        u=std::cbrt(-0.5*r+sq)+std::cbrt(-0.5*r-sq)+2.0*q; }
    else { const double A=0.025641,B=0.262785,kk=6.932009; double m=1.0-q, bc=B/kk, ac=A-B/kk;
        double P=(bc+std::sqrt(bc*bc+4.0*m*ac))/(2.0*m); u=2.0+(P-1.0)/kk; }
    return Q<0?-u:u; }

// USER-spec edge integral (mode 6): inner Pade[3/2] (reuse p32_innerI), splice a=1.5, tail 1-G/(1+lam t+mu t^2).
// tail int dt/D over parabola xi(t): linear xi -> real p32_Jr ; curved xi -> (2/sqrtDelta) Im[cJ] via D=mu(tau-tau1)(tau-tau2).
inline double p32u_tailI(double a,double b,double g,double sg,double t1,double t2){
    const double G=7.0/76.0,lam=274.0/133.0,mu=2404.0/931.0,A0=1.5;
    const double Delta=4.0*mu-lam*lam, sqD=std::sqrt(Delta);   // Delta>0 -> complex tau roots
    if(std::fabs(a)<1e-13){ double P=sg*b, Q=sg*g-A0;          // linear xi: tau=Pt+Q -> D quadratic in t (real)
        double DA=mu*P*P, DB=2.0*mu*P*Q+lam*P, DC=mu*Q*Q+lam*Q+1.0;
        return sg*(t2-t1) - sg*G*p32_Jr(DA,DB,DC,t1,t2); }
    std::complex<double> tau1(-lam/(2.0*mu), sqD/(2.0*mu)), gam1=sg*(A0+tau1);
    std::complex<double> bb=std::sqrt((std::complex<double>(g,0.0)-gam1)/a - std::complex<double>(b*b/(4*a*a),0.0));
    double u1=t1+b/(2*a),u2=t2+b/(2*a);
    double intD=(2.0*sg/sqD)*((std::atan(std::complex<double>(u2,0)/bb)-std::atan(std::complex<double>(u1,0)/bb))/(a*bb)).imag();
    return sg*(t2-t1) - sg*G*intD; }
inline double pade32u_edge_avg(double a,double b,double g){
    double bp[8]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0; const double A0=1.5;
    for(int sgn=-1;sgn<=1;sgn+=2){ double C=g-A0*sgn;
        if(std::fabs(a)<1e-13){ if(std::fabs(b)>1e-300){ double r=-C/b; if(r>1e-9&&r<1-1e-9)bp[nb++]=r; } }
        else { double d=b*b-4*a*C; if(d>0){ double s=std::sqrt(d),r1=(-b-s)/(2*a),r2=(-b+s)/(2*a);
            if(r1>1e-9&&r1<1-1e-9)bp[nb++]=r1; if(r2>1e-9&&r2<1-1e-9)bp[nb++]=r2; } } }
    for(int i=1;i<nb;++i){ double x=bp[i];int j=i-1;while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;}bp[j+1]=x; }
    double I=0.0;
    for(int s=0;s<nb-1;++s){ double t1=bp[s],t2=bp[s+1],dt=t2-t1; if(dt<1e-12)continue;
        double tm=0.5*(t1+t2),xm=a*tm*tm+b*tm+g;
        if(std::fabs(xm)<=A0) I+=p32_innerI(a,b,g,t1,t2);
        else I+=p32u_tailI(a,b,g,(xm<0?-1.0:1.0),t1,t2); }
    return I<-1.0?-1.0:(I>1.0?1.0:I); }
inline double pade32u_inv(double Q){   // CENTROID-D: inner cubic (Cardano, q<=69/76) + tail quadratic
    double q=Q<0?-Q:Q; const double pa=69.0/76.0; if(q>=1.0) return Q<0?-50.0:50.0; double u;
    if(q<=pa){ double p=15.0-12.0*q*q, r=15.0*q-16.0*q*q*q; double disc=0.25*r*r+p*p*p/27.0,sq=std::sqrt(disc);
        u=std::cbrt(-0.5*r+sq)+std::cbrt(-0.5*r-sq)+2.0*q; }
    else { const double G=7.0/76.0,lam=274.0/133.0,mu=2404.0/931.0;
        double tau=(-lam+std::sqrt(lam*lam+4.0*mu*(G/(1.0-q)-1.0)))/(2.0*mu); u=1.5+tau; }
    return Q<0?-u:u; }

// mode 7 (compact deg-9 polynomial smoothstep) closed-form edge integral: PURE polynomial moment, NO transcendental, NO division.
// inner |xi|<S: int P9(xi(t)) dt, xi(t) quadratic -> P9(xi) is deg-18 poly in t. saturated regions integrate to +-(t2-t1). split at xi=+-S.
inline double poly9_edge_avg(double a,double b,double g){
    const double S=2.68179418,a3=-0.2790604680,a5=0.0571770650,a7=-0.00595679733,a9=0.000238602530;
    double bp[8]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0;
    for(int sgn=-1;sgn<=1;sgn+=2){ double C=g-S*sgn;
        if(std::fabs(a)<1e-13){ if(std::fabs(b)>1e-300){ double r=-C/b; if(r>1e-9&&r<1-1e-9)bp[nb++]=r; } }
        else { double d=b*b-4*a*C; if(d>0){ double s=std::sqrt(d),r1=(-b-s)/(2*a),r2=(-b+s)/(2*a);
            if(r1>1e-9&&r1<1-1e-9)bp[nb++]=r1; if(r2>1e-9&&r2<1-1e-9)bp[nb++]=r2; } } }
    for(int i=1;i<nb;++i){ double x=bp[i];int j=i-1;while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;}bp[j+1]=x; }
    double I=0.0;
    for(int s=0;s<nb-1;++s){ double t1=bp[s],t2=bp[s+1],dt=t2-t1; if(dt<1e-12)continue;
        double tm=0.5*(t1+t2),xm=a*tm*tm+b*tm+g;
        if(xm>=S){ I+=dt; continue; } if(xm<=-S){ I-=dt; continue; }
        double xi[3]={g,b,a}, xi2[5]={0,0,0,0,0};                 // xi(t)=g+b t+a t^2
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)xi2[i+j]+=xi[i]*xi[j];
        double acc[17]={0}; int ad=0; acc[0]=a9;                  // Horner in xi^2: acc=((a9*x2+a7)*x2+a5)*x2+a3)*x2+1
        const double add[4]={a7,a5,a3,1.0};
        for(int st=0;st<4;++st){ double tmp[17]={0}; int nd=ad+4;
            for(int i=0;i<=ad;++i)for(int j=0;j<5;++j)tmp[i+j]+=acc[i]*xi2[j];
            tmp[0]+=add[st]; for(int k=0;k<=nd;++k)acc[k]=tmp[k]; ad=nd; }     // ad=16
        double Q[19]={0}; for(int i=0;i<3;++i)for(int j=0;j<=ad;++j)Q[i+j]+=xi[i]*acc[j];   // P9(xi)=xi*acc, deg18
        double p1[20],p2[20]; p1[0]=t1;p2[0]=t2; for(int k=1;k<20;++k){p1[k]=p1[k-1]*t1;p2[k]=p2[k-1]*t2;}
        for(int k=0;k<=18;++k)I+=Q[k]*(p2[k]-p1[k])/(k+1); }
    return I<-1.0?-1.0:(I>1.0?1.0:I); }
inline double poly9_inv(double Q){   // CENTROID-D: deg-9 inverse via bounded Newton (monotone, no closed form)
    const double S=2.68179418,a3=-0.2790604680,a5=0.0571770650,a7=-0.00595679733,a9=0.000238602530;
    double q=Q<0?-Q:Q; if(q>=1.0) return Q<0?-S:S;
    double u=q<0.9?q:2.0;
    for(int it=0;it<40;++it){ double s2=u*u;
        double P=u*(1.0+s2*(a3+s2*(a5+s2*(a7+a9*s2))))-q;
        double Pp=1.0+s2*(3*a3+s2*(5*a5+s2*(7*a7+9*a9*s2))); if(Pp<1e-7)Pp=1e-7;
        double du=P/Pp; u-=du; if(u<0)u=0; else if(u>S)u=S; if(std::fabs(du)<1e-13)break; }
    return Q<0?-u:u; }

// deg-3 cubic smoothstep (mode 8): pure-poly CF face + centroid-D Cardano (NO Newton, NO quadrature).
inline double deg3_edge_avg(double a,double b,double g){
    const double a1=0.8508, S=3.0/(2.0*a1), a3=-0.5/(S*S*S);   // computed from a1 -> psi(S)=1.5-0.5=1 exactly
    auto innerI=[&](double t1,double t2)->double{   // int (a1 xi + a3 xi^3) dt over [t1,t2], xi=a t^2+b t+g; CLOSED-FORM xi^3
        double g2=g*g,b2=b*b,a2=a*a;
        double Q0=a1*g+a3*g*g2, Q1=a1*b+a3*3.0*b*g2, Q2=a1*a+a3*(3.0*a*g2+3.0*b2*g),
               Q3=a3*(6.0*a*b*g+b*b2), Q4=a3*(3.0*a2*g+3.0*a*b2), Q5=a3*3.0*a2*b, Q6=a3*a*a2;
        double P1[8],P2[8]; P1[0]=t1;P2[0]=t2; for(int k=1;k<8;++k){P1[k]=P1[k-1]*t1;P2[k]=P2[k-1]*t2;}
        return Q0*(P2[0]-P1[0]) + Q1*(P2[1]-P1[1])*0.5 + Q2*(P2[2]-P1[2])/3.0 + Q3*(P2[3]-P1[3])*0.25
             + Q4*(P2[4]-P1[4])*0.2 + Q5*(P2[5]-P1[5])/6.0 + Q6*(P2[6]-P1[6])/7.0; };
    double xi0=g,xi1=a+b+g,xmn=xi0<xi1?xi0:xi1,xmx=xi0<xi1?xi1:xi0;   // FAST PATH: xi range (endpoints + vertex)
    if(a<-1e-300||a>1e-300){ double tv=-b/(2.0*a); if(tv>0.0&&tv<1.0){ double xv=a*tv*tv+b*tv+g; if(xv<xmn)xmn=xv; if(xv>xmx)xmx=xv; } }
    if(xmn>=S) return 1.0;                       // fully clamped +
    if(xmx<=-S) return -1.0;                     // fully clamped -
    if(xmn>=-S&&xmx<=S) return innerI(0.0,1.0);  // fully inner: ONE integral, no sort/roots
    double bp[8]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0;   // SLOW PATH: crosses +-S
    for(int sgn=-1;sgn<=1;sgn+=2){ double C=g-S*sgn;
        if(a>-1e-13&&a<1e-13){ if(b>1e-300||b<-1e-300){ double r=-C/b; if(r>1e-9&&r<1-1e-9)bp[nb++]=r; } }
        else { double d=b*b-4*a*C; if(d>0){ double s=std::sqrt(d),r1=(-b-s)/(2*a),r2=(-b+s)/(2*a);
            if(r1>1e-9&&r1<1-1e-9)bp[nb++]=r1; if(r2>1e-9&&r2<1-1e-9)bp[nb++]=r2; } } }
    for(int i=1;i<nb;++i){ double x=bp[i];int j=i-1;while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;}bp[j+1]=x; }
    double I=0.0;
    for(int s=0;s<nb-1;++s){ double t1=bp[s],t2=bp[s+1],dt=t2-t1; if(dt<1e-12)continue;
        double tm=0.5*(t1+t2),xm=a*tm*tm+b*tm+g;
        if(xm>=S) I+=dt; else if(xm<=-S) I-=dt; else I+=innerI(t1,t2); }
    return I<-1.0?-1.0:(I>1.0?1.0:I); }
inline double deg3_inv(double Q){   // CENTROID-D: psi(D)=Q, single-point cubic -> Cardano (trig form, NO Newton)
    const double a1=0.8508, S=3.0/(2.0*a1), a3=-0.5/(S*S*S);
    double q=Q<0?-Q:Q; if(q>=1.0) return Q<0?-S:S;
    double p=a1/a3, qd=-q/a3, m=2.0*std::sqrt(-p/3.0);    // a1 u + a3 u^3 = q -> u^3 + (a1/a3) u + qd = 0, physical = k=1
    double arg=3.0*qd/(2.0*p)*std::sqrt(-3.0/p); if(arg>1.0)arg=1.0; else if(arg<-1.0)arg=-1.0;
    double u=m*std::cos(std::acos(arg)/3.0 - 2.0943951023931953); if(u<0)u=0; else if(u>S)u=S;
    return Q<0?-u:u; }
// ALL real roots of c3 x^3+c2 x^2+c1 x+c0=0 into out[], returns count (1, 2, or 3). For the fast-path root-selection.
inline int cubic_real_roots(double c3,double c2,double c1,double c0,double* out){
    if(c3>-1e-13&&c3<1e-13){ if(c2>-1e-13&&c2<1e-13){ if(c1>-1e-300&&c1<1e-300)return 0; out[0]=-c0/c1; return 1; }
        double disc=c1*c1-4*c2*c0; if(disc<0)return 0; double sq=std::sqrt(disc);
        out[0]=(-c1+sq)/(2*c2); out[1]=(-c1-sq)/(2*c2); return 2; }
    double b=c2/c3,c=c1/c3,d=c0/c3; double p=c-b*b/3.0,q=2*b*b*b/27.0-b*c/3.0+d; double Di=q*q/4.0+p*p*p/27.0;
    if(Di>=0){ double sq=std::sqrt(Di); out[0]=std::cbrt(-q/2.0+sq)+std::cbrt(-q/2.0-sq)-b/3.0; return 1; }
    double r=std::sqrt(-p*p*p/27.0),ar=-q/2.0/r; if(ar>1.0)ar=1.0; else if(ar<-1.0)ar=-1.0;   // clamp acos arg (rounding can push |ar|>1 -> NaN)
    double phi=std::acos(ar),m=2.0*std::cbrt(r);
    for(int k=0;k<3;++k)out[k]=m*std::cos((phi+2.0*3.14159265358979324*k)/3.0)-b/3.0; return 3; }
// real cubic root of c3 x^3+c2 x^2+c1 x+c0=0 lying in [lo,hi] (unique by monotonicity of the bracket)
inline double cardano_in(double c3,double c2,double c1,double c0,double lo,double hi){
    if(std::fabs(c3)<1e-13){ if(std::fabs(c2)<1e-13){ if(std::fabs(c1)<1e-300)return 0.5*(lo+hi); return -c0/c1; }
        double disc=c1*c1-4*c2*c0; if(disc<0)return 0.5*(lo+hi); double sq=std::sqrt(disc);
        double r1=(-c1+sq)/(2*c2),r2=(-c1-sq)/(2*c2); return (r1>=lo-1e-9&&r1<=hi+1e-9)?r1:r2; }
    double b=c2/c3,c=c1/c3,d=c0/c3; double p=c-b*b/3.0,q=2*b*b*b/27.0-b*c/3.0+d; double Di=q*q/4.0+p*p*p/27.0;
    if(Di>=0){ double sq=std::sqrt(Di); return std::cbrt(-q/2.0+sq)+std::cbrt(-q/2.0-sq)-b/3.0; }
    double r=std::sqrt(-p*p*p/27.0),ar=-q/2.0/r; if(ar>1.0)ar=1.0; else if(ar<-1.0)ar=-1.0;   // clamp acos arg
    double phi=std::acos(ar),m=2.0*std::cbrt(r),mid=0.5*(lo+hi);
    for(int k=0;k<3;++k){ double x=m*std::cos((phi+2.0*3.14159265358979324*k)/3.0)-b/3.0; if(x>=lo-1e-7&&x<=hi+1e-7)return x; }
    double best=1e18,res=mid; for(int k=0;k<3;++k){ double x=m*std::cos((phi+2.0*3.14159265358979324*k)/3.0)-b/3.0; double dc=std::fabs(x-mid); if(dc<best){best=dc;res=x;} } return res; }
// deg-3 EXACT-cell-D, NO Newton: cell-avg(D)=Q is monotone piecewise-cubic. Clamp set changes only at breakpoints
// D=+-S-kP_q; in each interval the clamp set is fixed -> one cubic -> Cardano. Bracket via monotonicity (deterministic, robust).
inline double deg3_cellD(const double* kP,const double* w,int nq,double Q){
    const double a1=0.8508, S=3.0/(2.0*a1), a3=-0.5/(S*S*S);
    // FAST PATH: full-moment cubic (treat cell as unclamped) -> direct Cardano (NO binary search).
    // 3 real roots possible; 2 are spurious (they'd saturate to +-1). The clamp-check picks the physical root AND validates the fast path.
    double M0=0,M1=0,M2=0,M3=0; for(int q=0;q<nq;++q){ double k=kP[q],wq=w[q]; M0+=wq; M1+=wq*k; M2+=wq*k*k; M3+=wq*k*k*k; }
    double roots[3]; int nr=cubic_real_roots(a3*M0, 3.0*a3*M1, a1*M0+3.0*a3*M2, (a1*M1+a3*M3)-Q, roots);
    for(int r=0;r<nr;++r){ double D=roots[r]; bool ok=true;
        for(int q=0;q<nq;++q){ double s=kP[q]+D; if(!(s>=-S-1e-9 && s<=S+1e-9)){ok=false;break;} }  // !(in-range) also rejects NaN
        if(ok) return D; }     // all samples unclamped -> exact-D, physical root
    // SLOW PATH (some sample clamps): monotone piecewise-cubic -> binary-search bracket -> bracketed Cardano.
    double bp[16]; int nb=0; for(int q=0;q<nq;++q){ bp[nb++]=S-kP[q]; bp[nb++]=-S-kP[q]; }
    for(int i=1;i<nb;++i){ double x=bp[i];int j=i-1;while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;}bp[j+1]=x; }
    auto cav=[&](double D){ double f=0; for(int q=0;q<nq;++q){ double s=kP[q]+D,u=s<0?-s:s; f+=w[q]*(u>=S?(s<0?-1.0:1.0):(a1*s+a3*s*s*s)); } return f; };
    int L=0,R=nb-1,idx=-1;
    while(L<=R){ int mid=(L+R)>>1; if(cav(bp[mid])<=Q){idx=mid;L=mid+1;} else R=mid-1; }
    double Dlo=(idx<0)?-1e3:bp[idx], Dhi=(idx>=nb-1)?1e3:bp[idx+1], Dmid=0.5*(Dlo+Dhi);
    double Qp=Q,m0=0,m1=0,m2=0,m3=0;
    for(int q=0;q<nq;++q){ double s=kP[q]+Dmid;
        if(s>=S)Qp-=w[q]; else if(s<=-S)Qp+=w[q];
        else { m0+=w[q]; m1+=w[q]*kP[q]; m2+=w[q]*kP[q]*kP[q]; m3+=w[q]*kP[q]*kP[q]*kP[q]; } }
    if(m0<1e-12) return Dmid;
    return cardano_in(a3*m0, 3.0*a3*m1, a1*m0+3.0*a3*m2, (a1*m1+a3*m3)-Qp, Dlo, Dhi); }
// exact min/max of quadratic kP(X,Y)=b0 X+b1 Y+b2 X^2+b3 Y^2+b4 XY over a triangle (vertex offsets dx[],dy[]).
// candidates: 3 vertices + 3 edge-interior extrema + interior critical point (if inside).
inline void deg3_tri_minmax(double b0,double b1,double b2,double b3,double b4,
                            const double* dx,const double* dy,double& pmn,double& pmx){
    auto kp=[&](double X,double Y){ return b0*X+b1*Y+b2*X*X+b3*Y*Y+b4*X*Y; };
    pmn=1e300; pmx=-1e300;
    for(int k=0;k<3;++k){ double v=kp(dx[k],dy[k]); if(v<pmn)pmn=v; if(v>pmx)pmx=v; }
    const int E[3][2]={{0,1},{1,2},{2,0}};
    for(int e=0;e<3;++e){ int a=E[e][0],b=E[e][1]; double ex=dx[b]-dx[a],ey=dy[b]-dy[a];
        double A=b2*ex*ex+b3*ey*ey+b4*ex*ey;
        double gx=b0+2.0*b2*dx[a]+b4*dy[a], gy=b1+2.0*b3*dy[a]+b4*dx[a]; double B=gx*ex+gy*ey;
        if(A>1e-300||A<-1e-300){ double u=-B/(2.0*A); if(u>0.0&&u<1.0){ double v=kp(dx[a]+u*ex,dy[a]+u*ey); if(v<pmn)pmn=v; if(v>pmx)pmx=v; } } }
    double det=4.0*b2*b3-b4*b4;
    if(det>1e-300||det<-1e-300){ double X=(-b0*2.0*b3+b1*b4)/det, Y=(-b1*2.0*b2+b0*b4)/det;
        double e1x=dx[1]-dx[0],e1y=dy[1]-dy[0],e2x=dx[2]-dx[0],e2y=dy[2]-dy[0],ddx=X-dx[0],ddy=Y-dy[0];
        double de=e1x*e2y-e1y*e2x;
        if(de>1e-300||de<-1e-300){ double s=(ddx*e2y-ddy*e2x)/de,t=(e1x*ddy-e1y*ddx)/de;
            if(s>=-1e-9&&t>=-1e-9&&s+t<=1.0+1e-9){ double v=kp(X,Y); if(v<pmn)pmn=v; if(v>pmx)pmx=v; } } } }
// deg-3 EXACT-cell-D v2 (geometry-based, CONSISTENT with exact CF face -> high-res stable). NO Newton.
// FAST: 6-pt cubic Cardano accepted iff TRUE range (tri_minmax) unclamped -> EXACT for unclamped cells.
// SLOW (clamping): dense KxK sub-tri + bounded bisection on cell-avg(D)=Q (conservation<0.4%, no wild D).
// Replaces deg3_cellD: the old 6-pt clamp-set under-detected clamping (vertex-region) -> up to 17% cell-avg
// error inconsistent with the exact CF face -> divergence at high res. v2 fixes both detection and accuracy.
inline double deg3_cellD2(double b0,double b1,double b2,double b3,double b4,
                          const double* dx,const double* dy,double Q){
    const double a1=0.8508, S=3.0/(2.0*a1), a3=-0.5/(S*S*S);
    auto kp=[&](double X,double Y){ return b0*X+b1*Y+b2*X*X+b3*Y*Y+b4*X*Y; };
    double pmn,pmx; deg3_tri_minmax(b0,b1,b2,b3,b4,dx,dy,pmn,pmx);
    static const double DQ[6][4]={ {0.223381589678011,0.108103018168070,0.445948490915965,0.445948490915965},
        {0.223381589678011,0.445948490915965,0.108103018168070,0.445948490915965},
        {0.223381589678011,0.445948490915965,0.445948490915965,0.108103018168070},
        {0.109951743655322,0.816847572980459,0.091576213509771,0.091576213509771},
        {0.109951743655322,0.091576213509771,0.816847572980459,0.091576213509771},
        {0.109951743655322,0.091576213509771,0.091576213509771,0.816847572980459} };
    double M0=0,M1=0,M2=0,M3=0;
    for(int q=0;q<6;++q){ double X=DQ[q][1]*dx[0]+DQ[q][2]*dx[1]+DQ[q][3]*dx[2];
        double Y=DQ[q][1]*dy[0]+DQ[q][2]*dy[1]+DQ[q][3]*dy[2];
        double k=kp(X,Y),w=DQ[q][0]; M0+=w; M1+=w*k; M2+=w*k*k; M3+=w*k*k*k; }
    double roots[3]; int nr=cubic_real_roots(a3*M0,3.0*a3*M1,a1*M0+3.0*a3*M2,(a1*M1+a3*M3)-Q,roots);
    for(int r=0;r<nr;++r){ double D=roots[r]; if(pmn+D>=-S-1e-9 && pmx+D<=S+1e-9) return D; }   // truly unclamped -> EXACT
    // SLOW (clamping): dense KxK sub-tri + Illinois regula-falsi on cell-avg(D)-Qc (monotone, bracketed,
    // superlinear -> ~6x fewer evals than bisection). bracket [-S-pmx,S-pmn] => cav in [-1,+1]. conservation<0.5%.
    const int K=10; double kp_d[160]; int nd=0;
    for(int i=0;i<K;++i) for(int j=0;j<K-i;++j){
        double s=(i+1.0/3.0)/K,t=(j+1.0/3.0)/K;
        kp_d[nd++]=kp((1-s-t)*dx[0]+s*dx[1]+t*dx[2],(1-s-t)*dy[0]+s*dy[1]+t*dy[2]);
        if(i+j<K-1){ s=(i+2.0/3.0)/K;t=(j+2.0/3.0)/K;
            kp_d[nd++]=kp((1-s-t)*dx[0]+s*dx[1]+t*dx[2],(1-s-t)*dy[0]+s*dy[1]+t*dy[2]); } }
    double wd=1.0/nd, Qc=Q; if(Qc>1.0-1e-12)Qc=1.0-1e-12; else if(Qc<-1.0+1e-12)Qc=-1.0+1e-12;
    auto cav=[&](double D){ double f=0; for(int q=0;q<nd;++q){ double s=kp_d[q]+D; f+=(s>=S?1.0:(s<=-S?-1.0:(a1*s+a3*s*s*s))); } return f*wd-Qc; };
    double lo=-S-pmx, hi=S-pmn, flo=cav(lo), fhi=cav(hi), D=0.5*(lo+hi);
    for(int it=0;it<16;++it){ if(fhi<=flo) break; D=hi-fhi*(hi-lo)/(fhi-flo); double fD=cav(D);
        if(fD>-1e-10 && fD<1e-10) break;
        if(fD>0.0){ hi=D; fhi=fD; flo*=0.5; } else { lo=D; flo=fD; fhi*=0.5; } }
    return D; }

// ===== deg3t (mode 9): cubic inner + rational m=3 tail. centroid-D (Cardano) + CF face (poly inner + rational tail). =====
// OPTIMIZED/REFACTORED: single-source coeffs (DEG3T_*); tailI computes J->I2->I3 in ONE pass (one atan/log, vs old 2x);
// edge_avg single-region FAST-PATH (skip split/sort when edge stays inner or one-sided tail). Validated vs quad: inv 1.7e-15, edge_avg 1.8e-9.
inline double deg3t_innerI(double a,double b,double g,double t1,double t2){   // int (xi + a3 xi^3) dt
    constexpr double a3=DEG3T_A3; double g2=g*g,b2=b*b,a2=a*a;
    double Q0=g+a3*g*g2, Q1=b+a3*3.0*b*g2, Q2=a+a3*(3.0*a*g2+3.0*b2*g),
           Q3=a3*(6.0*a*b*g+b*b2), Q4=a3*(3.0*a2*g+3.0*a*b2), Q5=a3*3.0*a2*b, Q6=a3*a*a2;
    double P1[7],P2[7]; P1[0]=t1;P2[0]=t2; for(int k=1;k<7;++k){P1[k]=P1[k-1]*t1;P2[k]=P2[k-1]*t2;}
    return Q0*(P2[0]-P1[0]) + Q1*(P2[1]-P1[1])*0.5 + Q2*(P2[2]-P1[2])/3.0 + Q3*(P2[3]-P1[3])*0.25
         + Q4*(P2[4]-P1[4])*0.2 + Q5*(P2[5]-P1[5])/6.0 + Q6*(P2[6]-P1[6])/7.0; }
// deg3t-PRIVATE tail-integral helpers (std::log, NOT fast_log). p32_Jr/p32_I2r/p32_I3r are SHARED
// with pade32 (mode 5/6) which rely on fast_log for speed; deg3t's tail face integral is the ONLY caller
// that exposes fast_log's C0 seam jump (~8e-5 at x=1/sqrt(2)) and that breaks the deg3t edge symmetry
// (per-recon asym 8.9e-9 -> 1.8e-15, full-run sym 2.79e-2 -> 1.82e-2). Math identical to the shared
// p32_* forms; only the log is swapped to libm std::log. Keeps the fast_log fast-path untouched for
// every other sigmoid. (atan path here already uses std::atan, so no fast_atan change is needed.)
inline double deg3t_p32_Jr(double A,double B,double C,double t1,double t2){   // int dt/(A t^2+B t+C)
    if(std::fabs(A)<1e-14){ if(std::fabs(B)<1e-300) return (t2-t1)/C;
        return (std::log(std::fabs(B*t2+C))-std::log(std::fabs(B*t1+C)))/B; }
    double disc=4.0*A*C-B*B;
    if(disc>1e-300){ double sq=std::sqrt(disc); double u1=2*A*t1+B,u2=2*A*t2+B,den=sq*sq+u1*u2,v;
        if(std::fabs(den)<1e-300) v=std::atan(u2/sq)-std::atan(u1/sq);
        else { v=std::atan((u2-u1)*sq/den); if(den<0) v+=(u2>=0?3.14159265358979324:-3.14159265358979324); }
        return 2.0/sq*v; }
    if(disc<-1e-300){ double sq=std::sqrt(-disc);
        auto L=[&](double t){ double u=2*A*t+B; return std::log(std::fabs((u-sq)/(u+sq))); };
        return (L(t2)-L(t1))/sq; }
    return -2.0/(2*A*t2+B)+2.0/(2*A*t1+B); }
inline double deg3t_p32_I2r(double A,double B,double C,double t1,double t2){  // int dt/(A t^2+B t+C)^2
    if(std::fabs(A)<1e-14){ if(std::fabs(B)<1e-300) return (t2-t1)/(C*C);
        return -1.0/(B*(B*t2+C)) + 1.0/(B*(B*t1+C)); }
    double disc=4.0*A*C-B*B;
    if(std::fabs(disc)<1e-300){ double r=-B/(2*A);
        return -1.0/(3*A*A)*(1.0/std::pow(t2-r,3)-1.0/std::pow(t1-r,3)); }
    auto P=[&](double t){ return A*t*t+B*t+C; };
    return (2*A*t2+B)/(disc*P(t2)) - (2*A*t1+B)/(disc*P(t1)) + (2*A/disc)*deg3t_p32_Jr(A,B,C,t1,t2); }
inline double deg3t_p32_I3r(double A,double B,double C,double t1,double t2){  // int dt/(A t^2+B t+C)^3
    if(std::fabs(A)<1e-14){ if(std::fabs(B)<1e-300) return (t2-t1)/(C*C*C);
        double d2=B*t2+C,d1=B*t1+C; return -1.0/(2*B*d2*d2)+1.0/(2*B*d1*d1); }
    double disc=4.0*A*C-B*B;
    if(std::fabs(disc)<1e-300){ double r=-B/(2*A);
        return -1.0/(5*A*A*A)*(1.0/std::pow(t2-r,5)-1.0/std::pow(t1-r,5)); }
    auto P=[&](double t){ return A*t*t+B*t+C; };
    return (2*A*t2+B)/(2*disc*P(t2)*P(t2)) - (2*A*t1+B)/(2*disc*P(t1)*P(t1)) + (3*A/disc)*deg3t_p32_I2r(A,B,C,t1,t2); }
inline double deg3t_tailI(double a,double b,double g,double sg,double t1,double t2){   // int sigma_tail dt, |xi|>asp
    constexpr double A=DEG3T_A,B=DEG3T_B,kk=DEG3T_K,asp=DEG3T_ASP, bc=B/kk, ac=A-B/kk;
    double PA=sg*kk*a, PB=sg*kk*b, PC=sg*kk*g+(1.0-kk*asp);    // P=1+kk w (quadratic in t)
    double J,I2,I3;                                            // J=int dt/P, I2=int dt/P^2, I3=int dt/P^3 (one atan/log)
    if(std::fabs(PA)<1e-14){                                   // P linear: PB t + PC
        if(std::fabs(PB)<1e-300){ double iv=1.0/PC; J=(t2-t1)*iv; I2=J*iv; I3=I2*iv; }
        else { double d2=PB*t2+PC,d1=PB*t1+PC, ib=1.0/PB;
            J=(std::log(std::fabs(d2))-std::log(std::fabs(d1)))*ib; I2=(1.0/d1-1.0/d2)*ib; I3=(0.5/(d1*d1)-0.5/(d2*d2))*ib; } }   // std::log (deg3t-private, no fast_log seam)
    else { double disc=4.0*PA*PC-PB*PB, Pq2=PA*t2*t2+PB*t2+PC, Pq1=PA*t1*t1+PB*t1+PC; J=deg3t_p32_Jr(PA,PB,PC,t1,t2);
        if(std::fabs(disc)<1e-300){ I2=deg3t_p32_I2r(PA,PB,PC,t1,t2); I3=deg3t_p32_I3r(PA,PB,PC,t1,t2); }   // double-root -> robust forms
        else { I2=(2*PA*t2+PB)/(disc*Pq2)-(2*PA*t1+PB)/(disc*Pq1)+(2*PA/disc)*J;
            I3=(2*PA*t2+PB)/(2*disc*Pq2*Pq2)-(2*PA*t1+PB)/(2*disc*Pq1*Pq1)+(3*PA/disc)*I2; } }
    return sg*((t2-t1)-(ac*I3+bc*I2)); }
inline double deg3t_edge_avg(double a,double b,double g){
    constexpr double asp=DEG3T_ASP;
    double xi0=g,xi1=a+b+g, xmn=xi0<xi1?xi0:xi1, xmx=xi0<xi1?xi1:xi0;   // xi range = endpoints + vertex
    if(a<-1e-300||a>1e-300){ double tv=-b/(2.0*a); if(tv>0.0&&tv<1.0){ double xv=a*tv*tv+b*tv+g; if(xv<xmn)xmn=xv; if(xv>xmx)xmx=xv; } }
    if(xmn>=-asp&&xmx<=asp) return deg3t_innerI(a,b,g,0.0,1.0);          // FAST: fully inner
    if(xmn>=asp) return deg3t_tailI(a,b,g,1.0,0.0,1.0);                  // FAST: fully tail +
    if(xmx<=-asp) return deg3t_tailI(a,b,g,-1.0,0.0,1.0);               // FAST: fully tail -
    double bp[8]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0;                  // SLOW: crosses +-asp -> split
    for(int sgn=-1;sgn<=1;sgn+=2){ double C=g-asp*sgn;
        if(std::fabs(a)<1e-13){ if(std::fabs(b)>1e-300){ double r=-C/b; if(r>1e-9&&r<1-1e-9)bp[nb++]=r; } }
        else { double d=b*b-4*a*C; if(d>0){ double s=std::sqrt(d),r1=(-b-s)/(2*a),r2=(-b+s)/(2*a);
            if(r1>1e-9&&r1<1-1e-9)bp[nb++]=r1; if(r2>1e-9&&r2<1-1e-9)bp[nb++]=r2; } } }
    for(int i=1;i<nb;++i){ double x=bp[i];int j=i-1;while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;}bp[j+1]=x; }
    double I=0.0;
    for(int s=0;s<nb-1;++s){ double t1=bp[s],t2=bp[s+1]; if(t2-t1<1e-12)continue;
        double tm=0.5*(t1+t2),xm=a*tm*tm+b*tm+g;
        if(std::fabs(xm)<=asp) I+=deg3t_innerI(a,b,g,t1,t2);
        else I+=deg3t_tailI(a,b,g,(xm<0?-1.0:1.0),t1,t2); }
    return I<-1.0?-1.0:(I>1.0?1.0:I); }
inline double deg3t_inv(double Q){   // CENTROID-D: psi_deg3t(D)=Q closed-form (inner cubic Cardano / tail cubic Cardano, NO Newton)
    constexpr double a3=DEG3T_A3,asp=DEG3T_ASP,A=DEG3T_A,B=DEG3T_B,kk=DEG3T_K, psia=asp+a3*asp*asp*asp;
    double q=Q<0?-Q:Q; if(q>=0.999999) return Q<0?-12.0:12.0;
    double roots[3]; double u=-1.0;
    if(q<psia){ int nr=cubic_real_roots(a3,0.0,1.0,-q,roots);   // a3 u^3 + u - q = 0
        for(int i=0;i<nr;++i) if(roots[i]>=-1e-7 && roots[i]<=asp+1e-6){ u=roots[i]; break; }
        if(u<0){ u=asp; for(int i=0;i<nr;++i) if(roots[i]>=0.0 && roots[i]<u) u=roots[i]; } }
    else { double s=1.0-q; int nr=cubic_real_roots(s*kk*kk*kk, 3.0*s*kk*kk, 3.0*s*kk-B, s-A, roots);   // tail cubic in w
        double w=-1.0; for(int i=0;i<nr;++i) if(roots[i]>=-1e-7){ if(w<0.0||roots[i]<w) w=roots[i]; }
        if(w<0.0) w=0.0; u=asp+w; }
    if(u<0.0) u=0.0; return Q<0?-u:u; }
// deg3t sigmoid VALUE only (case 9 of spl_sig, inlined): cubic inner s+a3 s^3 (|s|<asp) / rational m=3 tail.
inline double deg3t_sig(double s){
    constexpr double a3=DEG3T_A3,asp=DEG3T_ASP,A=DEG3T_A,B=DEG3T_B,kk=DEG3T_K;
    double u=s<0.0?-s:s, gg;
    if(u<asp){ double u2=u*u; gg=u+a3*u2*u; }
    else { double w=u-asp, KW=1.0+kk*w; gg=1.0-(A+B*w)/(KW*KW*KW); }
    return s<0.0?-gg:gg; }
// deg3t sigmoid DERIVATIVE sigma'_deg3t(s) (case 9 gd, EVEN in s): inner 1+3 a3 s^2 (|s|<asp) / rational m=3 tail deriv.
// Used by the LINEARIZED hybrid (THINCQQ_DEG3T_HYBLIN) to linearize the tail correction g'(s)=sigma'(s)-(a1+3 a3 s^2).
inline double deg3t_sigd(double s){
    constexpr double a3=DEG3T_A3,asp=DEG3T_ASP,A=DEG3T_A,B=DEG3T_B,kk=DEG3T_K;
    double u=s<0.0?-s:s;
    if(u<asp){ return 1.0+3.0*a3*u*u; }                       // inner cubic slope (=cubic'_inner -> g'=0 in inner)
    double w=u-asp, KW=1.0+kk*w, KW2=KW*KW, KW4=KW2*KW2;      // tail: matches spl_sig_and_d case 9 gp (EVEN)
    return (3.0*kk*(A+B*w)-B*KW)/KW4; }
// HYBRID deg3t cell-D (env THINCQQ_DEG3T_HYB), FULLY no-Newton. Decompose sigma_deg3t(s)=cubic(s)+g(s),
// cubic(s)=s+a3 s^3 (the deg3t INNER, slope-1), g(s)=sigma_deg3t(s)-cubic(s) (=0 for |s|<asp; the tail-
// minus-cubic correction for |s|>=asp). Cell-avg constraint (1/V)int sigma(kP+D) dV = Q becomes
//   cubic_moment(D) + (1/V)int g(kP+D) dV = Q,   cubic_moment(D)=M1+M0 D + a3(M3+3M2 D+3M1 D^2+M0 D^3).
// EXACT for the cubic part via quadrature moments Mj=sum_q w_q (kP_q)^j (a1=1,a3=DEG3T_A3 -> direct Cardano,
// same machinery as deg3_cellD / deg3t_inv). The g-integral is APPROXIMATED by its centroid value (no Newton,
// like deg3t_inv): kP(centroid)=0 so s_centroid=D; 1-pass: estimate D0=deg3t_inv(Q), evaluate g0=g(D0), then
// Cardano-solve cubic_moment(D)=Q-g0. Pick the real root nearest D0 (cubic monotone in the inner region).
inline double deg3t_hyb_cellD(const double* kP,const double* w,int nq,double Q){
    constexpr double a1=1.0, a3=DEG3T_A3;
    double M0=0,M1=0,M2=0,M3=0;
    for(int q=0;q<nq;++q){ double k=kP[q],wq=w[q]; M0+=wq; M1+=wq*k; M2+=wq*k*k; M3+=wq*k*k*k; }
    double D0=deg3t_inv(Q);                                  // centroid-D estimate (closed-form, no Newton)
    double g0=deg3t_sig(D0)-(a1*D0+a3*D0*D0*D0);             // tail-minus-cubic correction at the centroid (0 if |D0|<asp)
    double Qc=Q-g0;                                          // corrected cubic target
    double roots[3]; int nr=cubic_real_roots(a3*M0, 3.0*a3*M1, a1*M0+3.0*a3*M2, (a1*M1+a3*M3)-Qc, roots);
    if(nr<=0) return D0;                                     // degenerate -> centroid-D fallback
    double D=roots[0], best=std::fabs(roots[0]-D0);          // pick real root nearest the centroid estimate
    for(int i=1;i<nr;++i){ double dc=std::fabs(roots[i]-D0); if(dc<best){best=dc; D=roots[i];} }
    return D; }
// FROZEN-QUADRATURE HYBRID deg3t cell-D (env THINCQQ_DEG3T_HYBQ), FULLY no-Newton. Identical to
// deg3t_hyb_cellD (inner cubic EXACT via Cardano moments) EXCEPT the tail correction g0 is the TRUE
// cell-average <g> at the FROZEN centroid D0 -- g0 = sum_q w_q*g(kP[q]+D0) -- instead of the 1-point
// centroid g(D0) (which is the Jensen-gap source: frozen g(D0) != <g>, per-cell conservation residual
// ~0.1 RMS that the symmetric KH amplifies, worse at high res). Evaluating g at ALL the existing kP[]
// quad points (already passed in) removes the Jensen gap WITHOUT any Newton iteration and WITHOUT g'
// (so it cannot diverge like the linearized HYBLIN). Cost: one extra cheap g-eval per quad point vs the
// centroid hybrid (no atan/log -- g is poly+rational eval). Cubic-in-D root selection unchanged.
inline double deg3t_hybq_cellD(const double* kP,const double* w,int nq,double Q){
    constexpr double a1=1.0, a3=DEG3T_A3;
    double M0=0,M1=0,M2=0,M3=0;
    for(int q=0;q<nq;++q){ double k=kP[q],wq=w[q]; M0+=wq; M1+=wq*k; M2+=wq*k*k; M3+=wq*k*k*k; }
    double D0=deg3t_inv(Q);                                  // centroid-D estimate (closed-form, no Newton)
    double g0=0.0;                                           // FROZEN-QUAD cell-avg of the tail-minus-cubic correction at D0
    for(int q=0;q<nq;++q){ double s=kP[q]+D0; g0 += w[q]*(deg3t_sig(s)-(a1*s+a3*s*s*s)); }  // <g(kP+D0)> (0 only if ALL |kP+D0|<asp)
    double Qc=Q-g0;                                          // corrected cubic target (Jensen-gap-free)
    double roots[3]; int nr=cubic_real_roots(a3*M0, 3.0*a3*M1, a1*M0+3.0*a3*M2, (a1*M1+a3*M3)-Qc, roots);
    if(nr<=0) return D0;                                     // degenerate -> centroid-D fallback
    double D=roots[0], best=std::fabs(roots[0]-D0);          // pick real root nearest the centroid estimate
    for(int i=1;i<nr;++i){ double dc=std::fabs(roots[i]-D0); if(dc<best){best=dc; D=roots[i];} }
    return D; }
// LINEARIZED-tail HYBRID deg3t cell-D (env THINCQQ_DEG3T_HYBLIN), FULLY no-Newton. Same decomposition as
// deg3t_hyb_cellD (inner cubic EXACT via Cardano moments) but the tail correction g is 1st-ORDER linearized
// in D about the centroid estimate D0 instead of frozen (0th order): <g(D)> ~ <g(D0)> + <g'(D0)>(D-D0), with
// both <g(D0)> and T'=<g'(D0)> evaluated at the SAME single centroid point s=D0 (kP(centroid)=0) the frozen
// hybrid uses (d/dD = d/ds for s=kk P+D). The T'*D term ADDS T' to the cubic LINEAR coeff; (g0 - T' D0)
// moves to the RHS -> STILL a cubic in D, Cardano-solved (same machinery). Inner stays exact (g=g'=0 for
// |D0|<asp); only the small tail correction is linearized -> O(g^2) error vs the frozen hybrid's O(g).
inline double deg3t_hyblin_cellD(const double* kP,const double* w,int nq,double Q){
    constexpr double a1=1.0, a3=DEG3T_A3;
    double M0=0,M1=0,M2=0,M3=0;
    for(int q=0;q<nq;++q){ double k=kP[q],wq=w[q]; M0+=wq; M1+=wq*k; M2+=wq*k*k; M3+=wq*k*k*k; }
    double D0=deg3t_inv(Q);                                  // centroid-D estimate (closed-form, no Newton)
    double g0=deg3t_sig(D0)-(a1*D0+a3*D0*D0*D0);             // tail-minus-cubic value at centroid (0 if |D0|<asp)
    double Tp=deg3t_sigd(D0)-(a1+3.0*a3*D0*D0);              // T'=<g'(D0)> = sigma'(D0)-cubic'_inner(D0) (0 if |D0|<asp)
    // cubic_moment(D) + g0 + T'(D-D0) = Q  ->  add T' to linear coeff, move (g0 - T' D0) into c0.
    double c1=a1*M0+3.0*a3*M2+Tp, c0=(a1*M1+a3*M3)-Q+g0-Tp*D0;
    double roots[3]; int nr=cubic_real_roots(a3*M0, 3.0*a3*M1, c1, c0, roots);
    if(nr<=0) return D0;                                     // degenerate -> centroid-D fallback
    double D=roots[0], best=std::fabs(roots[0]-D0);          // pick real root nearest the centroid estimate
    for(int i=1;i<nr;++i){ double dc=std::fabs(roots[i]-D0); if(dc<best){best=dc; D=roots[i];} }
    return D; }

// EXACT AST cell-average D (reviewer-proof): closed-form ridge-reduced cell integral + 1D scalar Newton,
// NO volume quadrature. For sigma=xi/(1+|xi|): Sigma=int sigma dxi=|xi|-ln(1+|xi|). Per cell edge (linear
// xi: xa->xb), the unit-interval integrals of sigma and Sigma (one sign; split at xi=0 if crossing):
inline double asig_seg_sig(double a,double b){   // int_0^1 sigma(xi) dt, xi a->b SAME sign (or a zero)
    double s=(a+b>=0.0)?1.0:-1.0, ua=1.0+std::fabs(a), ub=1.0+std::fabs(b), du=ub-ua;
    double inv=(std::fabs(du)>1e-14)?(fast_log(ub/ua)/du):(1.0/ua);            // int_0^1 dt/(1+|xi|)
    return s*(1.0-inv); }
inline double asig_seg_Sig(double a,double b){   // int_0^1 Sigma(xi) dt, xi a->b SAME sign
    double ua=1.0+std::fabs(a), ub=1.0+std::fabs(b), du=ub-ua;
    double lint=(std::fabs(du)>1e-14)?((ub*fast_log(ub)-ub-(ua*fast_log(ua)-ua))/du):fast_log(ua); // int ln(1+|xi|)
    return 0.5*(std::fabs(a)+std::fabs(b))-lint; }
inline double asig_J_sig(double xa,double xb){ if(xa*xb>=0.0) return asig_seg_sig(xa,xb);
    double ts=xa/(xa-xb); return ts*asig_seg_sig(xa,0.0)+(1.0-ts)*asig_seg_sig(0.0,xb); }
inline double asig_J_Sig(double xa,double xb){ if(xa*xb>=0.0) return asig_seg_Sig(xa,xb);
    double ts=xa/(xa-xb); return ts*asig_seg_Sig(xa,0.0)+(1.0-ts)*asig_seg_Sig(0.0,xb); }

// PST closed-form edge integral: EXACT edge-average of the cubic-smoothstep THINC profile
// th(xi)=2S((xi+1)/2)-1, S=3t^2-2t^3, along an edge where xi(t)=a t^2 + b t + g (t in [0,1]).
// Analytic band-clipping: th=+-1 outside |xi|<1. Roots of xi=+-1 (two quadratics) split [0,1];
// each piece is constant (+-1) or a degree-6 polynomial in t integrated by its EXPLICIT
// POLYNOMIAL ANTIDERIVATIVE (ZERO quadrature points). NO transcendental, NO Newton, NO Gauss.
// This is the genuinely quadrature-free closed-form THINC flux integral that tanh cannot provide.
inline double pst_edge_avg(double a, double b, double g){
    // exact in-band integral of th over [t1,t2]: th=-0.5 w^3+1.5 w^2-1, w=a t^2+b t+(g+1). NO Gauss.
    auto inband=[&](double t1,double t2)->double{
        double G=g+1.0, wc[3]={G,b,a}, w2[5]={0,0,0,0,0}, w3[7]={0,0,0,0,0,0,0};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) w2[i+j]+=wc[i]*wc[j];          // w^2 (deg 4)
        for(int i=0;i<5;++i) for(int j=0;j<3;++j) w3[i+j]+=w2[i]*wc[j];          // w^3 (deg 6)
        double thp[7]; for(int k=0;k<7;++k) thp[k]=-0.5*w3[k]+(k<5?1.5*w2[k]:0.0); thp[0]-=1.0;
        double t1p=t1,t2p=t2,acc=0.0;
        for(int k=0;k<7;++k){ acc+=thp[k]*(t2p-t1p)/(double)(k+1); t1p*=t1; t2p*=t2; }
        return acc; };
    // FAST PATH: range of xi(t)=a t^2+b t+g over [0,1] from the 2 endpoints + the vertex (no sqrt).
    // Most interface edges do NOT cross the band edge (band ~3.6 cells >> edge) -> one closed-form
    // poly, no root-finding. Only the rare partial-saturation edge falls to the slow band-clip path.
    double xi0=g, xi1=a+b+g, xmin=std::min(xi0,xi1), xmax=std::max(xi0,xi1);
    if(std::fabs(a)>1e-300){ double tv=-b/(2.0*a); if(tv>0.0&&tv<1.0){ double xv=a*tv*tv+b*tv+g;
        if(xv<xmin)xmin=xv; if(xv>xmax)xmax=xv; } }
    if(xmin>=1.0) return 1.0;                              // fully saturated high: th=+1
    if(xmax<=-1.0) return -1.0;                            // fully saturated low:  th=-1
    if(xmin>=-1.0 && xmax<=1.0) return inband(0.0,1.0);    // fully in band: ONE closed-form, no roots
    // SLOW PATH (rare): band edge crosses [0,1] -> clip at roots of xi=+-1.
    double bp[8]; int nb=0; bp[nb++]=0.0; bp[nb++]=1.0;
    auto roots=[&](double c){ double cc=g-c;
        if(std::fabs(a)<1e-13){ if(std::fabs(b)>1e-300){ double r=-cc/b; if(r>1e-9&&r<1.0-1e-9)bp[nb++]=r; } }
        else { double d=b*b-4.0*a*cc; if(d>0.0){ double s=std::sqrt(d),r1=(-b-s)/(2.0*a),r2=(-b+s)/(2.0*a);
            if(r1>1e-9&&r1<1.0-1e-9)bp[nb++]=r1; if(r2>1e-9&&r2<1.0-1e-9)bp[nb++]=r2; } } };
    roots(1.0); roots(-1.0);
    for(int i=1;i<nb;++i){ double x=bp[i]; int j=i-1; while(j>=0&&bp[j]>x){bp[j+1]=bp[j];--j;} bp[j+1]=x; }
    double I=0.0;
    for(int s=0;s<nb-1;++s){ double t1=bp[s],t2=bp[s+1]; if(t2-t1<1e-12)continue;
        double tm=0.5*(t1+t2), xm=a*tm*tm+b*tm+g;
        if(xm<=-1.0){ I-=(t2-t1); continue; }
        if(xm>=1.0){ I+=(t2-t1); continue; }
        I+=inband(t1,t2); }
    return I;
}

// ---- GAUSS (S2/S3) geometric moments of the cell -------------------------------------------
// <P> and <P^2> of the quadratic surface P = A0 dx + A1 dy + A2 dx^2 + A3 dy^2 + A4 dx dy are
// LINEAR / QUADRATIC forms in the coefficients A[] contracted with pure GEOMETRIC moments of the
// cell (<dx>, <dx^2>, ... up to 4th order). Those moments depend on the CELL SHAPE ONLY, so they
// are built ONCE per cell (shape-only) and shared across all variables and all beta -- matching
// the paper2 description (and the 3D U3Gmom/u3_build_gmom path) instead of re-running the 6-pt
// quadrature per variable and per beta.
struct C3Gmom {
    double Mx,My,Mxx,Myy,Mxy,Mxxx,Mxxy,Mxyy,Myyy,Mxxxx,Mxxxy,Mxxyy,Mxyyy,Myyyy;
};
// LEGACY build from the (deg-4 exact) triangle quadrature rule the point loop used.
// Kept ONLY as the A/B reference for the analytic formula below (env C3_GMOM_QUAD=1).
inline void c3_build_gmom_quad(const double (*TQp)[4], int NQC,
                          const double* nodes, const int* vs, int nv,
                          double cx, double cy, C3Gmom& G) {
    G = C3Gmom{0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double tri[C3_FAN_MAX][6], ar[C3_FAN_MAX];
    const int nt=c3_cell_fan(nodes,vs,nv,cx,cy,tri,ar);
    double at=0.0; for(int k=0;k<nt;++k) at+=ar[k]; if(!(std::fabs(at)>1e-300)) at=1.0;   // signed total area
    for(int k=0;k<nt;++k){ const double fw=ar[k]/at;
    for(int q=0;q<NQC;++q){
        double x=TQp[q][1]*tri[k][0]+TQp[q][2]*tri[k][2]+TQp[q][3]*tri[k][4];
        double y=TQp[q][1]*tri[k][1]+TQp[q][2]*tri[k][3]+TQp[q][3]*tri[k][5];
        double dx=x-cx, dy=y-cy, w=TQp[q][0]*fw;
        double dx2=dx*dx, dy2=dy*dy, dxy=dx*dy;
        G.Mx+=w*dx;      G.My+=w*dy;
        G.Mxx+=w*dx2;    G.Myy+=w*dy2;    G.Mxy+=w*dxy;
        G.Mxxx+=w*dx2*dx; G.Mxxy+=w*dx2*dy; G.Mxyy+=w*dx*dy2; G.Myyy+=w*dy2*dy;
        G.Mxxxx+=w*dx2*dx2; G.Mxxxy+=w*dx2*dxy; G.Mxxyy+=w*dx2*dy2;
        G.Mxyyy+=w*dxy*dy2; G.Myyyy+=w*dy2*dy2;
    } }
}
// ANALYTIC build (paper2 sec.2.3): CLOSED-FORM barycentric simplex integration, no quadrature.
//   (1/A) int_T lam1^p lam2^q lam3^r dA = 2 p! q! r! / (p+q+r+2)!
// With d_k = P_k - (cx,cy) and lam1+lam2+lam3 = 1, a point of the triangle satisfies
//   dx = sum_k lam_k a_k,  dy = sum_k lam_k b_k      (a_k,b_k = components of d_k)
// so <dx^i dy^j> is the product of two multinomial expansions contracted with the rule above:
//   <dx^i dy^j> = sum_{|p|=i} sum_{|q|=j} (i!/p!)(j!/q!) a^p b^q * 2 (p1+q1)!(p2+q2)!(p3+q3)! / (i+j+2)!
// Exact for every i+j (the 6-pt rule was only deg-4 exact) and evaluated ONCE per mesh.
// TRIANGLE-level analytic moments about the (external) origin the offsets a[],b[] are measured
// from. Split out of c3_build_gmom so the polygon fan can combine several sub-triangles that all
// share the CELL centroid as origin (a fan apex then simply has a=b=0, which the factorial formula
// handles without any special case).
inline void c3_build_gmom_tri(const double a[3], const double b[3], C3Gmom& G) {
    static const double F[12]={1.0,1.0,2.0,6.0,24.0,120.0,720.0,5040.0,40320.0,362880.0,3628800.0,39916800.0};
    auto ip=[](double x,int n){ double r=1.0; for(int t=0;t<n;++t) r*=x; return r; };
    auto mom=[&](int i,int j)->double{
        const double inv=2.0/F[i+j+2]; double s=0.0;
        for(int p0=0;p0<=i;++p0) for(int p1=0;p1<=i-p0;++p1){ int p2=i-p0-p1;
            const double ca=(F[i]/(F[p0]*F[p1]*F[p2]))*ip(a[0],p0)*ip(a[1],p1)*ip(a[2],p2);
            if(ca==0.0) continue;
            for(int q0=0;q0<=j;++q0) for(int q1=0;q1<=j-q0;++q1){ int q2=j-q0-q1;
                const double cb=(F[j]/(F[q0]*F[q1]*F[q2]))*ip(b[0],q0)*ip(b[1],q1)*ip(b[2],q2);
                if(cb==0.0) continue;
                s += ca*cb*F[p0+q0]*F[p1+q1]*F[p2+q2]; } }
        return s*inv; };
    G.Mx=mom(1,0);    G.My=mom(0,1);
    G.Mxx=mom(2,0);   G.Myy=mom(0,2);   G.Mxy=mom(1,1);
    G.Mxxx=mom(3,0);  G.Mxxy=mom(2,1);  G.Mxyy=mom(1,2);  G.Myyy=mom(0,3);
    G.Mxxxx=mom(4,0); G.Mxxxy=mom(3,1); G.Mxxyy=mom(2,2); G.Mxyyy=mom(1,3); G.Myyyy=mom(0,4);
}
// POLYGON build: fan about the cell centroid + AREA-WEIGHTED average of the sub-triangle moments.
// nv==3 -> single member with weight exactly 1.0 -> bit-identical to the legacy triangle formula.
inline void c3_build_gmom(const double* nodes, const int* vs, int nv,
                          double cx, double cy, C3Gmom& G) {
    G = C3Gmom{0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double tri[C3_FAN_MAX][6], ar[C3_FAN_MAX];
    const int nt=c3_cell_fan(nodes,vs,nv,cx,cy,tri,ar);
    if(nt<=0) return;
    double at=0.0; for(int k=0;k<nt;++k) at+=ar[k]; if(!(std::fabs(at)>1e-300)) at=1.0;   // signed total area
    double* gp=reinterpret_cast<double*>(&G);
    for(int k=0;k<nt;++k){
        double a[3],b[3];
        for(int j=0;j<3;++j){ a[j]=tri[k][2*j]-cx; b[j]=tri[k][2*j+1]-cy; }
        C3Gmom Gk; c3_build_gmom_tri(a,b,Gk);
        const double fw=ar[k]/at; const double* kp=reinterpret_cast<const double*>(&Gk);
        for(int i=0;i<14;++i) gp[i]+=fw*kp[i]; }
}
// Per-mesh cache of the shape-only moments (paper2: "computed once per grid and cached").
// Same pattern as the 3D U3Gmom/u3_build_gmom table in reconstruct3d_unstr.hpp.
struct C3GmomTab { const Mesh* mp=nullptr; int N=0; std::vector<C3Gmom> g; };
// algebraic contraction: mm1=<P>, mm2=<P^2>  (replaces the per-variable 6-pt point loop)
inline void c3_gmom_moments(const C3Gmom& G, const double* A, double& mm1, double& mm2) {
    const double A0=A[0],A1=A[1],A2=A[2],A3=A[3],A4=A[4];
    mm1 = A0*G.Mx + A1*G.My + A2*G.Mxx + A3*G.Myy + A4*G.Mxy;
    mm2 = A0*A0*G.Mxx + A1*A1*G.Myy + A2*A2*G.Mxxxx + A3*A3*G.Myyyy + A4*A4*G.Mxxyy
        + 2*A0*A1*G.Mxy + 2*A0*A2*G.Mxxx + 2*A0*A3*G.Mxyy + 2*A0*A4*G.Mxxy
        + 2*A1*A2*G.Mxxy + 2*A1*A3*G.Myyy + 2*A1*A4*G.Mxyy
        + 2*A2*A3*G.Mxxyy + 2*A2*A4*G.Mxxxy + 2*A3*A4*G.Mxyyy;
}

inline void reconstruct_cheng3(const Mesh& m, const ReconCtxO2& c,
                               const std::vector<double>& W, int nvar,
                               const std::vector<double>& WLs, const std::vector<double>& WRs,
                               std::vector<double>& W_L, std::vector<double>& W_R,
                               double beta_l, double beta_s) {
    const int N=m.n_cells(), Nf=m.n_faces(); const double* cc=m.cell_centers.data();
    static const double TQ[6][4]={
        {0.109951743655322,0.816847572980459,0.091576213509771,0.091576213509771},
        {0.109951743655322,0.091576213509771,0.816847572980459,0.091576213509771},
        {0.109951743655322,0.091576213509771,0.091576213509771,0.816847572980459},
        {0.223381589678011,0.108103018168070,0.445948490915965,0.445948490915965},
        {0.223381589678011,0.445948490915965,0.108103018168070,0.445948490915965},
        {0.223381589678011,0.445948490915965,0.445948490915965,0.108103018168070}};
    static const double EQ[4][2]={
        {0.0694318442029737,0.1739274225687269},{0.3300094782075719,0.3260725774312731},
        {0.6699905217924281,0.3260725774312731},{0.9305681557970263,0.1739274225687269}};
    // THINCQQ_LOWQUAD: 3-pt triangle (deg2) + 2-pt edge -> ~2x fewer tanh, near-identical for BVD.
    static const double TQ3[3][4]={{0.333333333333333,0.666666666666667,0.166666666666667,0.166666666666667},
        {0.333333333333333,0.166666666666667,0.666666666666667,0.166666666666667},
        {0.333333333333333,0.166666666666667,0.166666666666667,0.666666666666667}};
    static const double EQ2[2][2]={{0.211324865405187,0.5},{0.788675134594813,0.5}};
    static const double EQ1[1][2]={{0.5,1.0}};   // 1-point: edge midpoint (2nd-order FV)
    static const double EQ3[3][2]={{0.1127016653792583,0.2777777777777778},   // 3-pt Gauss-Legendre [0,1]
        {0.5,0.4444444444444444},{0.8872983346207417,0.2777777777777778}};    // for the AST branch-free quadrature variant
    static const bool LOWQ = std::getenv("THINCQQ_LOWQUAD") != nullptr;
    static const bool EDGE1 = std::getenv("THINCQQ_EDGE1") != nullptr;  // dominant edge cost -> 1pt
    static const bool ANALYTIC_D = std::getenv("THINCQQ_ANALYTIC_D") != nullptr;  // centroid model D=2cbar-1, no Newton/cell-quad
    static const bool RAMP = std::getenv("THINCQQ_RAMP") != nullptr;  // clamped-linear sigmoid S=clamp(x,-1,1), no tanh (use with ANALYTIC_D: D=kd)
    // THINC/QQ vertex-band [qmn,qmx] clamp on the face value. Cheng 2021 has NO such clamp.
    // NEW DEFAULT (2026-07-07, user): clamp OFF for the pure-tanh path (S1) — where it is a
    // mathematical no-op anyway (tanh in [-1,1] => qf in [qmn,qmx]) — but kept ON for the
    // closed-form / GAUSS-probit sigmoids (S2/S3) which CAN overshoot [-1,1] and NEED it.
    // Force OFF everywhere: THINCQQ_NOCLAMP=1.  Force ON everywhere: THINCQQ_CLAMP=1.
    static const bool NOCLAMP = []{
        if (std::getenv("THINCQQ_CLAMP"))   return false;   // force clamp ON (all paths)
        if (std::getenv("THINCQQ_NOCLAMP")) return true;    // force clamp OFF (all paths)
        bool cf = std::getenv("THINCQQ_PST") || std::getenv("THINCQQ_ASIG") ||
                  std::getenv("THINCQQ_SPL") || std::getenv("THINCQQ_GAUSS");
        return !cf;   // default: OFF for tanh (no-op), ON for overshoot-capable closed-form
    }();
    // PST = cubic-smoothstep THINC: th=2S(tau)-1, S=3tau^2-2tau^3, tau=(kP+D+1)/2, band-compact.
    // Replaces tanh's hard-kink-free S-curve WITH a POLYNOMIAL (closed-form integrable). D from the
    // ANALYTIC cubic-smoothstep inverse D=2 S^{-1}(cbar)-1 = -2 sin(asin(1-2cbar)/3) (NO Newton, NO
    // quadrature for D). Stage-1 reuses edge quadrature; Stage-2 (later) = closed-form edge integral.
    static const bool PST = std::getenv("THINCQQ_PST") != nullptr;
    static const double PST_CAP = std::getenv("THINCQQ_PST_CAP") ? std::atof(std::getenv("THINCQQ_PST_CAP")) : 0.0;  // saturate smoothstep to +-(1-eps): residual softness for high-beta stability, keeps the cheap polynomial integral
    static const bool PST_EXACT = std::getenv("THINCQQ_PST_EXACT") != nullptr;  // closed-form exact edge integral (vs point quadrature)
    static const bool PST_LINEAR = std::getenv("THINCQQ_PST_LINEAR") != nullptr; // planar interface (drop QQ curvature): xi linear in t -> cheapest exact integral + skip curvature compute
    // ASIG = algebraic sigmoid th=s/(1+|s|), s=kP+D. ASYMPTOTIC (->+-1 but never reaches at finite s)
    // = tanh-like residual softness -> HIGH-beta STABLE (unlike compact smoothstep). Analytic inverse
    // D=Q/(1-|Q|) (no Newton). Rational -> ITS edge integral is elementary closed-form (poly+arctan/log,
    // partial fractions) = quadrature-free (Stage 2). Stage 1 here = sig only (tests high-beta stability).
    static const bool ASIG = std::getenv("THINCQQ_ASIG") != nullptr;
    static const bool ASIG_EXACT = std::getenv("THINCQQ_ASIG_EXACT") != nullptr;  // elementary closed-form edge integral (arctan/log) instead of quadrature
    static const bool ASIG_LINEAR = std::getenv("THINCQQ_ASIG_LINEAR") != nullptr; // planar interface (drop QQ curvature): xi linear -> integral = log only (no arctan, no sqrt) = cheaper
    static const bool ASIG_QUAD = std::getenv("THINCQQ_ASIG_QUAD") != nullptr;     // FAST variant: branch-free 3-pt sigma quadrature (SIMD), faster than the exact arctan/log integral
    static const bool ARAT = std::getenv("THINCQQ_ARAT") != nullptr;               // ASIG sub-mode: RATIONAL sigmoid sign(s)(1-1/(1+|s|)^2) + perturbative curvature integral -> NO sqrt/arctan/quadrature/Newton, keeps the curve
    static const bool CELLEXACT = std::getenv("THINCQQ_ASIG_CELLEXACT") != nullptr; // EXACT cell-average D (reviewer-proof): closed-form ridge-reduced cell integral + 1D scalar Newton (NO volume quadrature) vs centroid collocation D=Q/(1-|Q|)
    static const bool ASIG_NEWTON = std::getenv("THINCQQ_ASIG_NEWTON") != nullptr;  // AST true exact-cell-avg D via CELL-QUADRATURE Newton (SAME D-method as tanh) -> isolates sigmoid vs D-method effect on KH
    static const bool SPL = std::getenv("THINCQQ_SPL") != nullptr;  // SPLICED sigmoid (Pade-of-tanh inner + asymptotic tail): tanh-hugging shape, closed-form, QQ. Test: cell-quad Newton D + quadrature face (shape isolation)
    static const bool SPL_CF = std::getenv("THINCQQ_SPL_CF") != nullptr;  // SPL closed-form FACE edge integral (spl_edge_avg, quadrature-free) instead of edge quadrature
    static const bool SPL_PADE32 = (spl_mode()==5);   // pade32 sigmoid (deg-2 denom): face CF uses pade32_edge_avg (own closed-form), not spl_edge_avg(quintic)
    static const bool SPL_PADEU = (spl_mode()==6);    // USER-spec sigmoid (splice 1.5, G/quadratic tail): face CF uses pade32u_edge_avg, centroid-D uses pade32u_inv
    static const bool SPL_POLY9 = (spl_mode()==7);    // compact deg-9 poly smoothstep: face CF uses poly9_edge_avg (pure-poly moment), centroid-D uses poly9_inv
    static const bool SPL_DEG3  = (spl_mode()==8);    // deg-3 cubic: face CF uses deg3_edge_avg (pure-poly), centroid-D uses deg3_inv (Cardano, no Newton)
    static const bool SPL_DEG3T = (spl_mode()==9);    // deg3t: cubic inner + rational m=3 tail (gentle): face CF deg3t_edge_avg (poly+atan/log), centroid-D deg3t_inv (Cardano)
    static const bool DEG3T_NEWTON = std::getenv("THINCQQ_DEG3T_NEWTON") != nullptr;  // deg3t opt-out: force old cell-quadrature Newton cell-D. DEFAULT (flag off) = HYBRID (inner cubic Cardano EXACT + tail centroid-D, fully no-Newton, deg3t_hyb_cellD)
    static const bool DEG3T_HYBLIN = std::getenv("THINCQQ_DEG3T_HYBLIN") != nullptr;  // deg3t LINEARIZED-tail HYBRID cell-D (no Newton): inner cubic EXACT + tail correction 1st-order linearized about centroid (deg3t_hyblin_cellD)
    static const bool DEG3T_HYBQ = std::getenv("THINCQQ_DEG3T_HYBQ") != nullptr;  // deg3t FROZEN-QUADRATURE HYBRID cell-D (no Newton): inner cubic EXACT + tail correction = TRUE cell-avg <g(kP+D0)> over the kP[] quad points (removes the centroid Jensen-gap non-conservation, deg3t_hybq_cellD)
    static const bool SPL_CENTROID = std::getenv("THINCQQ_SPL_CENTROID") != nullptr;  // CENTROID-D closed-form (D=atanh(Q), spl~tanh): NO Newton, NO cell-quad, beta-independent. WARNING: diffuses (KH 0.66 vs exact-cell 0.90)
    static const bool GAUSS = std::getenv("THINCQQ_GAUSS") != nullptr;  // probit-identity THINC: <tanh(s)>~=tanh(m1/sqrt(1+c v)) over a dist (mean m1,var v); v is D-independent => closed-form cell-D D=atanh(Q)sqrt(1+c v)-kk<P> (NO Newton) + closed-form edge moments (NO quadrature). uses <P>,<P^2> only, true tanh, consistent cell/face. [[paper-thesis-deg3t-efficiency]]
    static const double GC = []{ const char* e=std::getenv("THINCQQ_GC"); return (e&&e[0])?std::atof(e):1.5707963267948966; }();  // probit averaging constant c (=pi/2 for tanh)
    const double (*TQp)[4] = LOWQ?TQ3:TQ; const int NQC = LOWQ?3:6;
    const double (*EQp)[2] = EDGE1?EQ1:(LOWQ?EQ2:EQ); const int NQE = EDGE1?1:(LOWQ?2:4);
    // GAUSS (S2/S3) shape-only geometric moments: built ONCE PER MESH (paper2 sec.2.3 "cached per grid")
    // instead of per reconstruction call. Analytic barycentric closed form; C3_GMOM_QUAD=1 restores the
    // legacy 6-pt quadrature build (A/B verification only -- must agree to roundoff).
    static const bool GMOM_QUAD = std::getenv("C3_GMOM_QUAD") != nullptr;
    static C3GmomTab GT;
    if(GAUSS && (GT.mp!=&m || GT.N!=N)){
        GT.g.assign((size_t)N, C3Gmom{0,0,0,0,0,0,0,0,0,0,0,0,0,0});
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ const auto& vsg=m.cell_nodes[ci]; const int nvg=(int)vsg.size(); if(nvg<3) continue;
            if(GMOM_QUAD) c3_build_gmom_quad(TQp,NQC,m.nodes.data(),vsg.data(),nvg,cc[ci*2],cc[ci*2+1],GT.g[ci]);
            else          c3_build_gmom(m.nodes.data(),vsg.data(),nvg,cc[ci*2],cc[ci*2+1],GT.g[ci]); }
        GT.mp=&m; GT.N=N;
        if(std::getenv("CHENG3_PROF")) std::fprintf(stderr,"[C3GMOM] moment table (re)built: N=%d cells, %s\n",N,GMOM_QUAD?"quad":"analytic");
    }
    const bool twomem = (beta_s <= 0.0);   // beta_s<=0 -> 2-member (MUSCL + THINC beta_l only)
    static const bool SHEARBETA = std::getenv("THINCQQ_SHEARBETA")!=nullptr;  // single adaptive-beta THINC (needs ANALYTIC_D)
    static const bool BOPT = std::getenv("THINCQQ_BOPT")!=nullptr;  // BVD-optimal-beta AST: ONE candidate, per-cell beta*=[sigma^-1(Q_nb)-sigma^-1(Q_i)]/(dist/H) (jump-min, closed-form, no beta enumeration/tuning) + MUSCL safeguard
    static const double BOPT_BMIN = std::getenv("THINCQQ_BOPT_BMIN")?std::atof(std::getenv("THINCQQ_BOPT_BMIN")):0.8;
    static const double BOPT_BMAX = std::getenv("THINCQQ_BOPT_BMAX")?std::atof(std::getenv("THINCQQ_BOPT_BMAX")):1.6;
    // PAPER 3 (THINCQQ_BETASTAR): per-cell TBV-min beta* replaces the fixed 2-beta set.
    //   beta* interior (beta_s < beta* < beta_l)  -> candidate set {MUSCL, THINC(beta*)}  (drop beta_l,beta_s)
    //   beta* on boundary (TBV monotone -> beta* pinned) -> {MUSCL, THINC(beta_l), THINC(beta_s)} (=baseline)
    //   Selection is per cell/variable. Reuses the BOPT3 machinery (beff/binc/Dstar/jst/WLst/WRst)
    //   for the beta* candidate; only the min-TBV PICK below differs. FIRST CUT uses the closed-form
    //   weighted-median beta* (existing, an L1-TBV proxy); exact min-TBV Newton refinement is the
    //   next step (env THINCQQ_BSTAR_EXACT, TODO). Requires THINCQQ_GAUSS (closed-form D(beta)).
    static const bool BETASTAR = std::getenv("THINCQQ_BETASTAR")!=nullptr;
    static const bool BOPT3 = (std::getenv("THINCQQ_BOPT3")!=nullptr) || BETASTAR;  // BETASTAR reuses all BOPT3 beta* machinery
    const bool single = twomem || SHEARBETA || BOPT;   // one THINC candidate
    const bool adaptb = SHEARBETA || BOPT;             // per-cell beta from beff[]
    static std::vector<double> beff;  // per-cell beta: shear -> beta_s (gentle, KH), else beta_l (sharp). filled after pass1, interface cells only. (H2 persistent; re-.assign'd below on the active path)
    static std::vector<char> binc;    // BOPT3: per-cell flag, beta* candidate included (beta* in [beta_s,beta_l])
    static std::vector<double> bstv;  // BSTAR_PERVAR: per-cell per-variable beta*_v (nvar*N); THINC(beta*) uses this per var
    // ===== 4th BVD candidate: T-MLP-u-L (one-sided MLP-u psi in [0,2], IDW p=2). Env BVD_TMLPU4=1.
    // Adds the one-sided-MLP-u / IDW reconstruction (reconstruct_bj_vertex with the idw_p=2 ctx,
    // the SAME method the LeVeque bench's "T-MLP-u-L" line uses) to the {MUSCL, THINC beta_l,
    // THINC beta_s} candidate set. Per-variable min-TBV then picks the sharpest-but-bounded of 4
    // (THINC for contacts, T-MLP-u-L for shear/peaks, MUSCL smooth fallback). NO vortex/Ducros
    // detection -> pure min-TBV. Default OFF (flag unset) => exactly the 3-way (byte-identical). =====
    // BVD_TMLPU4  = old 4th cand = MLP-u1 + idw_p=2 (reconstruct_bj_vertex, psi<=1, NO compression).
    // BVD_TMLPU4G = NEW 4th cand = GATED compressive T-MLP-u (reconstruct_tmlpu_gated: LSQ-residual
    //   gated van_leer(smooth)/CICSAM(sharp) TVD + per-vertex-LMP, psi in [0,2]). This is the
    //   genuinely-compressive candidate (the idw version differs from MUSCL only by gradient weight;
    //   the gated version differs by the psi<=2 compression). Params via TMLPU_THR/CO/CAP (same as solver2d).
    static const bool TMLPU4G = std::getenv("BVD_TMLPU4G") != nullptr;
    static const bool TMLPU4 = TMLPU4G || (std::getenv("BVD_TMLPU4") != nullptr);
    static const double TG_THR = []{ const char* e=std::getenv("TMLPU_THR"); return e?std::atof(e):0.20; }();
    static const double TG_CO  = []{ const char* e=std::getenv("TMLPU_CO");  return e?std::atof(e):0.38; }();
    static const double TG_CAP = []{ const char* e=std::getenv("TMLPU_CAP"); return e?std::atof(e):2.0;  }();
    static const bool TMLPU4_DIAG = std::getenv("BVD_TMLPU4_DIAG") != nullptr;  // per-cell candidate-4 win count (density)
    std::vector<double> WLt4, WRt4, jt4;
    if(TMLPU4){
        // Build the idw_p=2 vertex-stencil ctx ONCE per mesh (cached across RK stages / calls);
        // build_recon_ctx does a std::set neighbour walk = one-time setup cost, NOT per-stage.
        static const Mesh* cached_mesh = nullptr; static ReconCtx idw_ctx;
        if(cached_mesh != &m){ idw_ctx = build_recon_ctx(m, 2.0); cached_mesh = &m; }
        if(TMLPU4G)  // gated compressive T-MLP-u (van_leer/CICSAM + LMP, psi in [0,2])
            reconstruct_tmlpu_gated(m, idw_ctx, W, nvar, WLt4, WRt4, TG_THR, TG_CO, TG_CAP);
        else         // old: T-MLP-u-L one-sided MLP-u (psi<=1, face_bound=false)
            reconstruct_bj_vertex(m, idw_ctx, W, nvar, WLt4, WRt4, false);
        jt4.assign((size_t)nvar*Nf, 0.0);
        #pragma omp parallel for
        for(int f=0;f<Nf;++f){ if(m.face_neighbour[f]<0) continue;
            for(int v=0;v<nvar;++v) jt4[(size_t)f*nvar+v]=std::fabs(WLt4[(size_t)v*Nf+f]-WRt4[(size_t)v*Nf+f]); }
    }
    double _t0=prof_now();
    // Pass 0 (vertex loop): vertex min/max, once.
    static std::vector<double> vmn, vmx; vmn.assign((size_t)nvar*c.Nn,0.0); vmx.assign((size_t)nvar*c.Nn,0.0);   // H2 persistent scratch
    #pragma omp parallel for
    for(int vv=0;vv<c.Nn;++vv) for(int v=0;v<nvar;++v){ double mn=1e300,mx=-1e300;
        for(int k=0;k<c.max_v2c;++k){ int ci=c.v2c[(size_t)vv*c.max_v2c+k]; if(ci<0)continue; double val=W[(size_t)v*N+ci]; if(val<mn)mn=val; if(val>mx)mx=val; }
        if(mn>mx){mn=0;mx=0;} vmn[(size_t)vv*nvar+v]=mn; vmx[(size_t)vv*nvar+v]=mx; }
    // Pass 1 (cell loop): interface cells -> a_st, D_l, D_s, qmin/qmax, 1/H. Shared by both beta.
    static std::vector<double> acoef, Dl, Ds, qmnc, qmxc, kbc;   // H2 persistent scratch (byte-identical re-zero)
    acoef.assign((size_t)nvar*N*5,0.0); Dl.assign((size_t)nvar*N,0.0); Ds.assign((size_t)nvar*N,0.0);
    qmnc.assign((size_t)nvar*N,0.0); qmxc.assign((size_t)nvar*N,0.0); kbc.assign(N,0.0);
    static std::vector<double> Dstar; if(BOPT3) Dstar.assign((size_t)nvar*N,0.0);   // BOPT3: D solved at per-cell beta* (4th candidate)
    // PAPER 3 exact beta* (THINCQQ_BSTAR_EXACT): store per-cell GAUSS cell moments <P>,<P^2> so
    // D_i(beta) = atanh(Q)*sqrt(1+GC*kk^2*(m2-m1^2)) - kk*m1 is closed-form at ANY trial beta.
    static const bool BSTAR_EXACT = std::getenv("THINCQQ_BSTAR_EXACT")!=nullptr;
    static const bool BSTAR_MAXB  = std::getenv("THINCQQ_BSTAR_MAXB")!=nullptr;   // option c: largest beta with TBV<=tol*min
    static const double BSTAR_TOL = []{ const char* e=std::getenv("THINCQQ_BSTAR_TOL"); return (e&&e[0])?std::atof(e):1.05; }();
    // option A (THINCQQ_BSTAR_WIDE): drop the 2-beta fallback entirely. Search beta* over a WIDE
    // range [WMIN,WMAX] (~[0,inf) clamp) and ALWAYS use {MUSCL, THINC(beta*)} (binc forced 1). The
    // beta* THINC uses Dstar solved consistently at beta* (closed-form GAUSS) -> stable (unlike BOPT,
    // which reused Dl=D(beta_l) at beta* -> divergence). min-TBV picks MUSCL vs THINC(beta*) per cell.
    static const bool BSTAR_WIDE  = std::getenv("THINCQQ_BSTAR_WIDE")!=nullptr;
    // beta* range [WMIN,WMAX]. DEFAULT = [0.8, 1.4] (= S1's beta_s..beta_l) — the WIDE [ln3,6] range
    // OVER-COMPRESSES at fine (paper) resolution -> spurious small-scale noise ("mush") in strong-shock
    // post-shock regions (mach3/DMR); capping to [0.8,1.4] removes the mush while keeping clean coherent
    // KH rolls (verified 2026-07-08 mach3/shockmixing paper-res). Set WMIN/WMAX env for the wide range.
    static const double BSTAR_WMIN = []{ const char* e=std::getenv("THINCQQ_BSTAR_WMIN"); return (e&&e[0])?std::atof(e):0.8; }();
    static const double BSTAR_WMAX = []{ const char* e=std::getenv("THINCQQ_BSTAR_WMAX"); return (e&&e[0])?std::atof(e):1.4; }();
    // Option B (DEFAULT-ON in beta* mode): GAUSS is used ONLY inside the beta*-SEARCH algorithm (the
    // closed-form probit TBV(beta) evaluated over the NB grid -> fast argmin). The ACTUAL beta*
    // reconstruction (Dstar cell-D + face average) then uses the EXACT tanh THINC (Newton cell-D +
    // quadrature face) at the GAUSS-found beta*. So "GAUSS for the search, tanh for everything else"
    // (user directive 2026-07-08). Only active when BOPT3 (beta* mode) so pure S2 (GAUSS 2-beta) stays
    // fully GAUSS. opt-out THINCQQ_BSTAR_GAUSSRECON=1 -> GAUSS recon at beta* too (Option A).
    static const bool BSTAR_TANHRECON = BOPT3 && (std::getenv("THINCQQ_BSTAR_GAUSSRECON")==nullptr);
    // 4-way union (THINCQQ_BSTAR_4WAY): per-var min-TBV over {MUSCL, THINC(beta_l), THINC(beta_s), THINC(beta*)}.
    // beta* (density argmin) dominates beta_l/beta_s on DENSITY, but for velocity the fixed 0.8/1.6 can beat
    // the density-derived beta* -> restores 2-beta shear sharpness on u,v while keeping beta* for rho/rolls.
    static const bool BSTAR_4WAY  = std::getenv("THINCQQ_BSTAR_4WAY")!=nullptr;
    // per-variable beta* (THINCQQ_BSTAR_PERVAR): each primitive var (rho,u,v,p) runs its OWN TBV-argmin
    // beta*_v (moments gm1/gm2 already per-var) instead of reusing the density beta*. The THINC(beta*)
    // candidate for var v then uses beta*_v -> velocity gets sharp beta* on KH-roll edges the uniform-
    // density beta* misses. binc/pick logic unchanged (already per-variable). bstv sized nvar*N.
    // DEFAULT-ON in the exact-beta* path (density beta* MUST NOT be reused for u,v,p — the shear/velocity
    // KH rolls need their own beta*_v; density-only diffuses them). opt-out THINCQQ_BSTAR_NOPV. Non-exact
    // BOPT paths (median/grid) keep it opt-in (their bstv is only sized when PERVAR under EXACT).
    static const bool BSTAR_PERVAR = (std::getenv("THINCQQ_BSTAR_EXACT")!=nullptr)
        ? (std::getenv("THINCQQ_BSTAR_NOPV")==nullptr)
        : (std::getenv("THINCQQ_BSTAR_PERVAR")!=nullptr);
    // option A closed-form beta* (THINCQQ_BSTAR_MEDIAN): replace the NB-grid TBV search with the
    // per-face linearized zero-jump beta_f=|(Dl_nb-Dl_i)/a| weighted MEDIAN (= L1-TBV-argmin proxy,
    // Xiao/BOPT). No grid -> ~16x cheaper per cell. Dstar still solved consistently at this beta*
    // (avoids the old BOPT divergence, which reused Dl at beta*). Approximate (ignores saturation).
    static const bool BSTAR_MEDIAN = std::getenv("THINCQQ_BSTAR_MEDIAN")!=nullptr;
    // option 3 (THINCQQ_BSTAR_FAST): hoisted beta-independent face moments + golden-section min of the
    // EXACT TBV(beta) (unimodal). Same argmin as the grid, ~exact accuracy, few evals. +THINCQQ_BSTAR_WARM:
    // warm-start the golden bracket from the previous recon call's beta* (temporal coherence), with a
    // periodic full re-search (every WARMK calls) safeguard. GITER = golden iterations (evals ~= GITER+2).
    static const bool BSTAR_FAST = std::getenv("THINCQQ_BSTAR_FAST")!=nullptr;
    // BSTAR_KINK: EXACT global argmin of the symmetric TBV(beta) via kink-enumeration + per-segment
    // stationary points (reuses BSTAR_FAST's hoisted face moments). Per-face kink beta_f = root of
    // Jf(beta)=q_owner(beta)-q_neighbor(beta)=0 by safeguarded Newton (NO double-squaring: 3-4 sqrt
    // radicals + tanh injectivity => solve Jf=0 directly). Then compare TBV at all kinks + segment
    // stationary points + endpoints -> global beta* (handles tanh non-convexity, unlike median/golden).
    // Floor beta* >= ln3 (2nd-order smooth accuracy). This is the S3 exact-beta* method (2026-07-07).
    static const bool BSTAR_KINK = std::getenv("THINCQQ_BSTAR_KINK")!=nullptr;
    // segment-interior stationary refinement is ON by default — it makes beta* the TRUE global argmin
    // (matches the winning S3kink+PV result). Measured cost is only ~18% (not the bottleneck), and
    // dropping it PERTURBS beta* (mean 2.34->2.47, sharper), i.e. no longer the exact accuracy. Opt OUT
    // with THINCQQ_BSTAR_NOSEG only for speed experiments. BSTAR_KNEWT caps the per-face kink-root Newton
    // (bracketed+safeguarded -> converges in <8; default 8 is beta*-identical to 24, safe speed).
    static const bool BSTAR_SEG = std::getenv("THINCQQ_BSTAR_NOSEG")==nullptr;
    static const int  BSTAR_KNEWT = std::getenv("THINCQQ_BSTAR_KNEWT")?std::max(2,std::min(24,std::atoi(std::getenv("THINCQQ_BSTAR_KNEWT")))):8;
    static const bool BSTAR_WARM = std::getenv("THINCQQ_BSTAR_WARM")!=nullptr;
    static const int  BSTAR_WARMK = std::getenv("THINCQQ_BSTAR_WARMK")?std::atoi(std::getenv("THINCQQ_BSTAR_WARMK")):8;
    static const int  BSTAR_GITER = std::getenv("THINCQQ_BSTAR_GITER")?std::atoi(std::getenv("THINCQQ_BSTAR_GITER")):8;
    // ---- FAST beta* alternatives (all reuse the closed-form Jf(beta),J'(beta); pick ONE) ----
    // L2:   minimize sum J^2 (smooth, no kinks) -> stationary G=sum J*J'=0 by bisect-in-coarse-bracket.
    //       C-infinity; L2 argmin != L1 argmin in general (mean vs median), test on ens.
    // HUBER: minimize sum sqrt(J^2+eps^2) -> smoothed-L1; eps=1e-3*rng -> converges to L1 argmin, smooth.
    // IRLS: 2-3 reweighted-L2 outer iters w=1/max(|J|,delta) -> converges to exact L1 argmin.
    // DOM:  Newton the kink of the single |J|-dominant face at beta_l (~3 evals; approx).
    // MODMED: linearize J at beta_l, per-face kink beta_f=b0-J/J', weighted (|J'|) median (cache-free KPW-lite).
    // S3 DEFAULT beta* method = L2GN (Gauss-Newton on the L2 objective) — CONFIRMED 2026-07-07: ~1.5x S1,
    // ens ties/beats KINK, positivity+symmetry+smooth-accuracy OK on leveque/config3/shockvortex draft
    // battery. Bare exact-beta* path (no explicit method flag) => L2GN. Opt into exact-L1 KINK reference
    // with THINCQQ_BSTAR_KINK=1 (or golden via THINCQQ_BSTAR_FAST, grid via THINCQQ_BSTAR_MEDIAN).
    static const int BSTAR_FASTMODE = []{ const char* s=std::getenv("THINCQQ_BSTAR_FASTMODE");
        if(s){ if(!std::strcmp(s,"L2"))return 1; if(!std::strcmp(s,"HUBER"))return 2;
            if(!std::strcmp(s,"IRLS"))return 3; if(!std::strcmp(s,"DOM"))return 4; if(!std::strcmp(s,"MODMED"))return 5;
            if(!std::strcmp(s,"L2GN"))return 6; if(!std::strcmp(s,"WMGN"))return 7; return std::atoi(s); }
        // no explicit FASTMODE: default to L2GN(6) UNLESS a competing method is explicitly selected
        if(std::getenv("THINCQQ_BSTAR_KINK")||std::getenv("THINCQQ_BSTAR_FAST")||std::getenv("THINCQQ_BSTAR_MEDIAN")) return 0;
        return 6; }();
    // (a) P1-prefilter: skip the beta* solve on cells where the MUSCL(P1) boundary jump sum is < PREFILT*range
    // (cell ~smooth -> MUSCL wins BVD anyway, beta* irrelevant). Heuristic (beta* could rescue P2); ens-gate it.
    static const double BSTAR_PREFILT = std::getenv("THINCQQ_BSTAR_PREFILT")?std::atof(std::getenv("THINCQQ_BSTAR_PREFILT")):0.0;
    static std::vector<double> gm1,gm2; if(BETASTAR&&BSTAR_EXACT){ gm1.assign((size_t)nvar*N,0.0); gm2.assign((size_t)nvar*N,0.0); }   // H2 persistent
    static std::vector<char> hasint; hasint.assign((size_t)nvar*N,0);   // H2 persistent (MUST re-zero: interface gate)
    #pragma omp parallel for
    for(int ci=0;ci<N;++ci){ const auto& vs=m.cell_nodes[ci]; int nv=(int)vs.size();
        double perim=0; for(int f:m.cell_faces[ci]) perim+=m.face_areas[f];
        double kb=perim/std::max(4.0*m.cell_volumes[ci],1e-30); kbc[ci]=kb;   // 1/H
        for(int v=0;v<nvar;++v){ double qbar=W[(size_t)v*N+ci], qmn=1e300,qmx=-1e300;
            for(int vid:vs){ double a=vmn[(size_t)vid*nvar+v],b=vmx[(size_t)vid*nvar+v]; if(a<qmn)qmn=a; if(b>qmx)qmx=b; }
            qmnc[(size_t)ci*nvar+v]=qmn; qmxc[(size_t)ci*nvar+v]=qmx; double rng=qmx-qmn;
            double cbar = rng>1e-14 ? (qbar-qmn)/rng : 0.5;
            if(cbar<=1e-6||cbar>=1.0-1e-6||rng<=1e-14||nv<3){ hasint[(size_t)ci*nvar+v]=0; continue; }
            double g0=0,g1=0,g2=0,g3=0,g4=0;   // o2 quadratic coeffs (interface cells only)
            for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                double dW=W[(size_t)v*N+nb]-qbar; const double* Mr=&c.M[((size_t)ci*5)*c.max_nb+k];
                g0+=Mr[0]*dW; g1+=Mr[c.max_nb]*dW; g2+=Mr[2*c.max_nb]*dW; g3+=Mr[3*c.max_nb]*dW; g4+=Mr[4*c.max_nb]*dW; }
            double Gm=std::sqrt(g0*g0+g1*g1);
            if(Gm<1e-30){ hasint[(size_t)ci*nvar+v]=0; continue; }
            double nx=g0/Gm, ny=g1/Gm;
            double* A=&acoef[((size_t)ci*nvar+v)*5];
            A[0]=nx; A[1]=ny;
            if((PST && PST_LINEAR)||(ASIG && ASIG_LINEAR)){ A[2]=0.0; A[3]=0.0; A[4]=0.0; }   // planar interface: skip curvature
            else { double G3=Gm*Gm*Gm;
                double nxx=g2/Gm-g0*(g0*g2+g1*g4)/G3, nxy=g4/Gm-g0*(g0*g4+g1*g3)/G3;
                double nyx=g4/Gm-g1*(g0*g2+g1*g4)/G3, nyy=g3/Gm-g1*(g0*g4+g1*g3)/G3;
                A[2]=0.5*nxx; A[3]=0.5*nyy; A[4]=0.5*(nxy+nyx); }
            double Q=2.0*cbar-1.0;
            if(GAUSS){   // probit closed-form cell-D: solve tanh((kk<P>+D)/sqrt(1+c v))=Q => D=atanh(Q)sqrt(1+c v)-kk<P>, v=kk^2(<P^2>-<P>^2). NO Newton, NO cell-quadrature beyond the deg-4-exact 6-pt moments.
                double mm1,mm2; c3_gmom_moments(GT.g[ci],A,mm1,mm2);   // <P>,<P^2> = algebraic contraction of A[] with the per-mesh cached geometric moments (analytic, exact)
                double Qc=Q; if(Qc>0.999)Qc=0.999; else if(Qc<-0.999)Qc=-0.999; double aQ=0.5*std::log((1.0+Qc)/(1.0-Qc));
                auto solveG=[&](double kk){ double vv=kk*kk*(mm2-mm1*mm1); if(vv<0)vv=0; return aQ*std::sqrt(1.0+GC*vv)-kk*mm1; };
                Dl[(size_t)ci*nvar+v]=solveG(beta_l*kb); if(!single) Ds[(size_t)ci*nvar+v]=solveG(beta_s*kb);
                if(BETASTAR&&BSTAR_EXACT){ gm1[(size_t)ci*nvar+v]=mm1; gm2[(size_t)ci*nvar+v]=mm2; }   // store for exact beta* TBV search
            }
            else if(SPL && SPL_CENTROID){   // closed-form CENTROID-D (no Newton, no cell-quadrature), beta-independent. WARNING: diffuses
                double D;
                if(SPL_DEG3) D=deg3_inv(Q);                              // deg-3 centroid-D Cardano (trig, no Newton)
                else if(SPL_DEG3T) D=deg3t_inv(Q);                       // deg3t centroid-D: inner cubic Cardano / tail cubic Cardano (no Newton)
                else if(SPL_POLY9) D=poly9_inv(Q);                       // deg-9 inverse (bounded Newton)
                else if(SPL_PADEU) D=pade32u_inv(Q);                     // exact user-spec inverse (Cardano inner + quadratic tail)
                else if(SPL_PADE32) D=pade32_inv(Q);                     // exact pade32 inverse (Cardano), NOT atanh
                else { double Qc=Q; if(Qc>0.999999)Qc=0.999999; else if(Qc<-0.999999)Qc=-0.999999; D=0.5*std::log((1.0+Qc)/(1.0-Qc)); }  // spl~tanh: D=atanh(Q)
                Dl[(size_t)ci*nvar+v]=D; if(!single) Ds[(size_t)ci*nvar+v]=D;
            }
            else if(SPL){
                // SPLICED sigmoid: cell-quadrature Newton (like tanh), solve Σ_q w_q spl_sig(kk·Pg[q]+D)=Q
                double Pg[C3_NQMAX], cwq[C3_NQMAX];   // polygon fan quadrature (tri: identical to legacy)
                const int NQ=c3_cell_quad_P(TQp,NQC,m.nodes.data(),vs.data(),nv,cc[ci*2],cc[ci*2+1],A,Pg,cwq);
                static const bool DEG3_NEWTON = std::getenv("THINCQQ_DEG3_NEWTON")!=nullptr;  // diagnostic: deg3 via generic per-iter Newton (isolate sigmoid shape from Cardano fast-path)
                auto solveSpl=[&](double kk)->double{ double kP[C3_NQMAX]; for(int q=0;q<NQ;++q)kP[q]=kk*Pg[q];   // hoist kk*Pg (invariant across Newton iters)
                    if(SPL_DEG3T && DEG3T_HYBQ){   // deg3t FROZEN-QUADRATURE HYBRID: inner cubic EXACT + tail correction = TRUE cell-avg <g(kP+D0)> (removes centroid Jensen-gap). NO Newton.
                        return deg3t_hybq_cellD(kP,cwq,NQ,Q); }
                    if(SPL_DEG3T && DEG3T_HYBLIN){   // deg3t LINEARIZED-tail HYBRID: inner cubic EXACT + tail correction 1st-order linearized about centroid. NO Newton.
                        return deg3t_hyblin_cellD(kP,cwq,NQ,Q); }
                    if(SPL_DEG3T && !DEG3T_NEWTON){   // deg3t HYBRID (DEFAULT): inner cubic EXACT via Cardano moments + tail correction via centroid-D. NO Newton. (opt-out to old Newton: THINCQQ_DEG3T_NEWTON)
                        return deg3t_hyb_cellD(kP,cwq,NQ,Q); }
                    if(SPL_DEG3 && !DEG3_NEWTON){   // deg-3: EXACT-cell-D v2 (geometry-based, consistent w/ exact CF face). NO Newton, NO wild D.
                        // NOTE: this closed form is TRIANGLE-ONLY (3-node geometry). Diagnostic path
                        // (THINCQQ_SPL + deg3), not used by S1/S2/S3 -> left un-generalised for quads.
                        double dxx[3],dyy[3]; for(int j=0;j<3;++j){ int vid=vs[j]; dxx[j]=m.nodes[vid*2]-cc[ci*2]; dyy[j]=m.nodes[vid*2+1]-cc[ci*2+1]; }
                        return deg3_cellD2(kk*A[0],kk*A[1],kk*A[2],kk*A[3],kk*A[4], dxx,dyy, Q); }
                    if(SPL_POLY9){   // poly9: cell-avg constraint is a deg-9 polynomial in D -> hoist q-loop into 10 coeffs (no per-iter q-loop). HUGE Newton speedup.
                        const double S=2.68179418,a3=-0.2790604680,a5=0.0571770650,a7=-0.00595679733,a9=0.000238602530;
                        double M[10]={0,0,0,0,0,0,0,0,0,0}; for(int q=0;q<NQ;++q){ double w=cwq[q],p=1.0; for(int m=0;m<10;++m){ M[m]+=w*p; p*=kP[q]; } }
                        static const double C1[2]={1,1},C3[4]={1,3,3,1},C5[6]={1,5,10,10,5,1},C7[8]={1,7,21,35,35,21,7,1},C9[10]={1,9,36,84,126,126,84,36,9,1};
                        double A[10]; for(int j=0;j<=9;++j){ double s=0;
                            if(j<=1)s+=C1[j]*M[1-j]; if(j<=3)s+=a3*C3[j]*M[3-j]; if(j<=5)s+=a5*C5[j]*M[5-j];
                            if(j<=7)s+=a7*C7[j]*M[7-j]; if(j<=9)s+=a9*C9[j]*M[9-j]; A[j]=s; }
                        double D=poly9_inv(Q);   // centroid inverse = near-exact init -> fast convergence
                        for(int it=0;it<12;++it){ double f=A[9],fp=0.0; for(int j=8;j>=0;--j){ fp=fp*D+f; f=f*D+A[j]; }
                            f-=Q; if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD; if(D>1e4)D=1e4; else if(D<-1e4)D=-1e4; if(std::fabs(dD)<1e-12)break; }
                        bool clamp=false; for(int q=0;q<NQ;++q) if(std::fabs(kP[q]+D)>S){clamp=true;break;}   // saturated q -> poly-in-D invalid
                        if(!clamp) return D;
                        for(int it=0;it<14;++it){ double f=-Q,fp=0.0;   // fallback: exact per-iter Newton (handles clamp), warm-started from D
                            for(int q=0;q<NQ;++q){ double sg,sgd; spl_sig_and_d(kP[q]+D,sg,sgd); f+=cwq[q]*sg; fp+=cwq[q]*sgd; }
                            if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD; if(D>1e4)D=1e4; else if(D<-1e4)D=-1e4; if(std::fabs(dD)<1e-11)break; }
                        return D; }
                    double D=Q;                  // generic SPL: per-iter cell-quadrature Newton
                    for(int it=0;it<14;++it){ double f=-Q,fp=0.0;
                        for(int q=0;q<NQ;++q){ double sg,sgd; spl_sig_and_d(kP[q]+D,sg,sgd); f+=cwq[q]*sg; fp+=cwq[q]*sgd; }
                        if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD; if(D>1e4)D=1e4; else if(D<-1e4)D=-1e4; if(std::fabs(dD)<1e-11)break; }
                    return D; };
                Dl[(size_t)ci*nvar+v]=solveSpl(beta_l*kb); if(!single) Ds[(size_t)ci*nvar+v]=solveSpl(beta_s*kb);
            }
            else if(ANALYTIC_D || PST || ASIG){
                // analytic D (no Newton). PST: cubic-smoothstep inverse. ASIG: algebraic-sigmoid inverse
                // D=Q/(1-|Q|). tanh/ramp centroid: D=Q.
                double Dv = Q;
                if(PST){ double s=-Q/(1.0-PST_CAP); if(s>1.0)s=1.0; else if(s<-1.0)s=-1.0; Dv=-2.0*std::sin(std::asin(s)/3.0); }  // cap: smoothstep_pm(D)=Q/(1-eps)
                else if(ASIG && ARAT){ double aq=std::fabs(Q); double r=1.0/std::sqrt(std::max(1.0-aq,1e-12))-1.0; Dv=(Q<0.0?-r:r); }  // rational sigmoid inverse (1 sqrt/cell, no loop)
                else if(ASIG){ double aq=std::fabs(Q); Dv=Q/std::max(1.0-aq,1e-9); }
                double Dvl=Dv, Dvs=Dv;
                if(ASIG_NEWTON && ASIG && !ARAT && !BOPT){
                    // AST TRUE cell-average D via CELL-QUADRATURE Newton (same machinery as tanh): solve Σ_q w_q σ(kk·Pg[q]+D)=Q, σ=s/(1+|s|), σ'=1/(1+|s|)^2
                    double Pg[C3_NQMAX], cwq[C3_NQMAX];   // polygon fan quadrature (tri: identical to legacy)
                    const int NQ=c3_cell_quad_P(TQp,NQC,m.nodes.data(),vs.data(),nv,cc[ci*2],cc[ci*2+1],A,Pg,cwq);
                    auto solveDn=[&](double kk,double D0)->double{ double D=D0;
                        for(int it=0;it<12;++it){ double f=-Q,fp=0.0;
                            for(int q=0;q<NQ;++q){ double s=kk*Pg[q]+D, den=1.0+std::fabs(s); f+=cwq[q]*(s/den); fp+=cwq[q]/(den*den); }
                            if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD; if(D>1e4)D=1e4; else if(D<-1e4)D=-1e4; if(std::fabs(dD)<1e-11)break; }
                        return D; };
                    Dvl=solveDn(beta_l*kb, Dv); if(!single) Dvs=solveDn(beta_s*kb, Dv);
                }
                else if(CELLEXACT && ASIG && !ARAT && !BOPT){
                    // EXACT cell-average D: ridge-reduced closed-form cell integral I(D)=(1/kk)Σ_e(n̂·N_e)∫_edge Σ(ξ) + 1D Newton (NO volume quadrature)
                    double nx=A[0], ny=A[1], xc=cc[ci*2], yc=cc[ci*2+1], Vol=m.cell_volumes[ci];
                    int nvv=(int)vs.size(); if(nvv>16)nvv=16; double gvv[16], wee[16];
                    for(int j=0;j<nvv;++j){ int vid=vs[j]; gvv[j]=nx*(m.nodes[vid*2]-xc)+ny*(m.nodes[vid*2+1]-yc); }
                    for(int j=0;j<nvv;++j){ int j2=(j+1)%nvv; int va=vs[j],vb=vs[j2];
                        double xa=m.nodes[va*2],ya=m.nodes[va*2+1],xb=m.nodes[vb*2],yb=m.nodes[vb*2+1];
                        double ex=xb-xa,ey=yb-ya, Nx=ey,Ny=-ex, mx=0.5*(xa+xb)-xc,my=0.5*(ya+yb)-yc;
                        if(Nx*mx+Ny*my<0.0){Nx=-Nx;Ny=-Ny;} wee[j]=nx*Nx+ny*Ny; }
                    auto solveD=[&](double kk,double D0)->double{ if(kk<1e-30)return D0; double D=D0;
                        for(int it=0;it<8;++it){ double I=0.0,Ip=0.0;
                            for(int j=0;j<nvv;++j){ int j2=(j+1)%nvv; double xia=kk*gvv[j]+D, xib=kk*gvv[j2]+D;
                                I+=wee[j]*asig_J_Sig(xia,xib); Ip+=wee[j]*asig_J_sig(xia,xib); }
                            double gp=(Ip/kk)/Vol; if(std::fabs(gp)<1e-30)break;
                            double dD=((I/kk)/Vol-Q)/gp; D-=dD; if(D>1e5)D=1e5; else if(D<-1e5)D=-1e5;
                            if(std::fabs(dD)<1e-10)break; }
                        return D; };
                    Dvl=solveD(beta_l*kb, Dv); if(!single) Dvs=solveD(beta_s*kb, Dv);
                }
                Dl[(size_t)ci*nvar+v]=Dvl; if(!single) Ds[(size_t)ci*nvar+v]=Dvs;
            }
            else {
            // PRODUCTION tanh path (S1/S3-optionB). Cell quadrature now covers the WHOLE polygon
            // (fan about the centroid); for a triangle the points/weights are bit-identical to the
            // legacy single-triangle loop.
            double Pg[C3_NQMAX], cwq[C3_NQMAX];
            const int NQ=c3_cell_quad_P(TQp,NQC,m.nodes.data(),vs.data(),nv,cc[ci*2],cc[ci*2+1],A,Pg,cwq);
            for(int pass=0;pass<(single?1:2);++pass){ double kk=(pass?beta_s:beta_l)*kb, Ag[C3_NQMAX];   // SHEARBETA needs ANALYTIC_D (skips here)
                for(int q=0;q<NQ;++q) Ag[q]=std::tanh(kk*Pg[q]);   // exact tanh (S1 high-accuracy: cell-D must match the exact-tanh face for conservation)
                double D=0.0;
                for(int it=0;it<10;++it){ double f=-Q,fp=0.0;
                    for(int q=0;q<NQ;++q){ double den=1.0+Ag[q]*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12);
                        f+=cwq[q]*(Ag[q]+D)/den; fp+=cwq[q]*(1.0-Ag[q]*Ag[q])/(den*den); }
                    if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD;
                    if(D>0.999999)D=0.999999; else if(D<-0.999999)D=-0.999999;
                    if(std::fabs(dD)<1e-11)break; }
                (pass?Ds:Dl)[(size_t)ci*nvar+v]=D; } }
            hasint[(size_t)ci*nvar+v]=1; }
    }
    cheng3_prof().geom += prof_now()-_t0; double _tg=prof_now();   // pass0+pass1 = geometry+D-solve
    if(SHEARBETA && nvar>=3){ beff.assign(N,beta_l);   // INTERFACE cells only (hasint) -> ~3x cheaper than all-cells
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ bool anyif=false; for(int v=0;v<nvar;++v) if(hasint[(size_t)ci*nvar+v]){anyif=true;break;}
            if(!anyif) continue; double ux=0,uy=0,vx=0,vy=0;
            for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue; const double* Mr=&c.M[((size_t)ci*5)*c.max_nb+k];
                double du=W[(size_t)1*N+nb]-W[(size_t)1*N+ci], dv=W[(size_t)2*N+nb]-W[(size_t)2*N+ci];
                ux+=Mr[0]*du; uy+=Mr[c.max_nb]*du; vx+=Mr[0]*dv; vy+=Mr[c.max_nb]*dv; }
            double om=vx-uy, dl=ux+vy, s=om*om/(om*om+dl*dl+1e-30);   // Ducros shear sensor
            beff[ci]=beta_l-(beta_l-beta_s)*s; } }
    // PAPER 3 EXACT beta*: minimize the TRUE TBV_i(beta)=Sum_f |q_i^f(beta)-q_j^f(beta)| over
    // beta in [beta_s,beta_l], both sides at the same trial beta, via a bracketed grid+refine
    // search (GAUSS closed-form D(beta) + closed-form edge moments -> each eval is O(faces)).
    // Option c (BSTAR_MAXB): pick the LARGEST beta with TBV(beta)<=TOL*min (defeats the mild-
    // takeover: the pure argmin is pulled to the diffusive end, this keeps the sharpest beta that
    // is still near-TBV-optimal). binc = beta* strictly interior -> {MUSCL,THINC(beta*)}, else 2beta.
    if(BETASTAR && BSTAR_EXACT && GAUSS){ beff.assign(N,beta_l); binc.assign(N,0);
        if(BSTAR_PERVAR) bstv.assign((size_t)nvar*N, beta_l);
        // exact GAUSS face value for cell ci at trial beta, on face edge (ax,ay)-(bx,by).
        auto qface=[&](int ci,int v,double beta,double D,double ax,double ay,double bx,double by)->double{
            double qmn=qmnc[(size_t)ci*nvar+v],qmx=qmxc[(size_t)ci*nvar+v],rng=qmx-qmn;
            const double* A=&acoef[((size_t)ci*nvar+v)*5]; double kk=beta*kbc[ci];
            double dx0=ax-cc[ci*2],dy0=ay-cc[ci*2+1],ex=bx-ax,ey=by-ay;
            double p2=A[2]*ex*ex+A[3]*ey*ey+A[4]*ex*ey;
            double p1=A[0]*ex+A[1]*ey+2.0*A[2]*dx0*ex+2.0*A[3]*dy0*ey+A[4]*(dx0*ey+dy0*ex);
            double p0=A[0]*dx0+A[1]*dy0+A[2]*dx0*dx0+A[3]*dy0*dy0+A[4]*dx0*dy0;
            double F1=p2/3.0+p1/2.0+p0, F2=p2*p2/5.0+p1*p2/2.0+(p1*p1+2.0*p0*p2)/3.0+p0*p1+p0*p0;
            double vv=kk*kk*(F2-F1*F1); if(vv<0)vv=0; double th=std::tanh((kk*F1+D)/std::sqrt(1.0+GC*vv));
            double qf=qmn+0.5*rng*(1.0+th); return NOCLAMP?qf:(qf<qmn?qmn:(qf>qmx?qmx:qf)); };
        auto Dbeta=[&](int ci,int v,double beta)->double{ double kk=beta*kbc[ci];
            double m1=gm1[(size_t)ci*nvar+v],m2=gm2[(size_t)ci*nvar+v];
            double qmn=qmnc[(size_t)ci*nvar+v],qmx=qmxc[(size_t)ci*nvar+v],rng=qmx-qmn;
            double Q=2.0*(W[(size_t)v*N+ci]-qmn)/rng-1.0; if(Q>0.999)Q=0.999; else if(Q<-0.999)Q=-0.999;
            double aQ=0.5*std::log((1.0+Q)/(1.0-Q)); double vv=kk*kk*(m2-m1*m1); if(vv<0)vv=0;
            return aQ*std::sqrt(1.0+GC*vv)-kk*m1; };
        const int NB = std::getenv("THINCQQ_BSTAR_NB")?std::max(3,std::min(16,std::atoi(std::getenv("THINCQQ_BSTAR_NB")))):(BSTAR_WIDE?16:9);   // beta grid pts over [blo,bhi]; coarser NB = faster + discreteness avoids the diffusive exact-argmin
        const double blo = BSTAR_WIDE?BSTAR_WMIN:beta_s, bhi = BSTAR_WIDE?BSTAR_WMAX:beta_l;
        // warm-start persistent state (across recon calls): previous beta* per (v,ci). Reset on size change;
        // full (cold) search every BSTAR_WARMK calls as a safeguard against drift.
        static std::vector<double> bstv_warm; static long bstar_warmcall=0; bool bstar_full=true;
        // warm window half-width: kink/golden search is restricted to [pc-WARMW, pc+WARMW] around the
        // cached beta* on warm calls (temporal coherence — beta* drifts little per step). Full re-search
        // every WARMK calls bounds drift, so beta* stays ~exact.
        static const double BSTAR_WARMW = std::getenv("THINCQQ_BSTAR_WARMW")?std::atof(std::getenv("THINCQQ_BSTAR_WARMW")):0.8;
        if((BSTAR_FAST||BSTAR_KINK) && BSTAR_WARM){
            if((long)bstv_warm.size()!=(long)((size_t)nvar*N)){ bstv_warm.assign((size_t)nvar*N,beta_l); bstar_full=true; }
            else bstar_full=((bstar_warmcall%BSTAR_WARMK)==0);
            ++bstar_warmcall; }
        #pragma omp parallel for schedule(dynamic,64)   // M2: interface-ONLY loop (most cells early-out) -> dynamic avoids clustered-front load imbalance; unique-index writes => bit-identical to static
        for(int ci=0;ci<N;++ci){ if(!hasint[(size_t)ci*nvar+0]) continue;
            const int NVS = BSTAR_PERVAR ? nvar : 1;   // per-variable beta*_v search (v=0 also drives beff/binc)
            for(int v=0; v<NVS; ++v){
                if(!hasint[(size_t)ci*nvar+v]){ if(BSTAR_PERVAR) bstv[(size_t)ci*nvar+v]=beta_l; continue; }
                double bstar;
                bool pf_skip=false;
                if(BSTAR_PREFILT>0.0){   // (a) P1-prefilter: MUSCL boundary-jump sum tiny vs range -> smooth cell, skip beta* solve
                    double t1=0; for(int f:m.cell_faces[ci]) t1+=std::fabs(WLs[(size_t)v*Nf+f]-WRs[(size_t)v*Nf+f]);
                    double rc=qmxc[(size_t)ci*nvar+v]-qmnc[(size_t)ci*nvar+v];
                    if(t1 < BSTAR_PREFILT*rc){ bstar=beta_l; pf_skip=true; } }
                if(pf_skip){ /* beta* set to beta_l; MUSCL wins the pick anyway */ }
                else if(BSTAR_MEDIAN){   // option A: closed-form weighted median of per-face zero-jump beta_f=|(Dl_nb-Dl_i)/a|
                    const double* A=&acoef[((size_t)ci*nvar+v)*5]; double nx=A[0], ny=A[1];
                    double Di=Dl[(size_t)ci*nvar+v]; double bf[16], wf[16]; int nf=0;
                    for(int f:m.cell_faces[ci]){ int o=m.face_owner[f],n=m.face_neighbour[f],nb=(o==ci)?n:o; if(nb<0)continue;
                        if(!hasint[(size_t)nb*nvar+v]) continue;
                        double a=((cc[nb*2]-cc[ci*2])*nx+(cc[nb*2+1]-cc[ci*2+1])*ny)*kbc[ci];
                        if(std::fabs(a)<1e-9) continue;
                        double b=(Dl[(size_t)nb*nvar+v]-Di)/a; if(b<0)b=-b;
                        if(nf<16){ bf[nf]=b; wf[nf]=std::fabs(a); nf++; } }
                    if(nf==0) bstar=beta_l;
                    else { for(int x=1;x<nf;++x){ double bb=bf[x],ww=wf[x]; int y=x-1; while(y>=0&&bf[y]>bb){bf[y+1]=bf[y];wf[y+1]=wf[y];--y;} bf[y+1]=bb;wf[y+1]=ww; }
                        double wt=0; for(int x=0;x<nf;++x)wt+=wf[x]; double half=0.5*wt, acc=0; bstar=bf[nf-1];
                        for(int x=0;x<nf;++x){ acc+=wf[x]; if(acc>=half){bstar=bf[x];break;} } }
                    if(bstar<blo)bstar=blo; else if(bstar>bhi)bstar=bhi;
                } else if(BSTAR_FAST || BSTAR_KINK || BSTAR_FASTMODE){   // HOIST beta-indep face moments; then golden/kink/fast-alt
                    const double* Ao=&acoef[((size_t)ci*nvar+v)*5];
                    double cxo=cc[ci*2], cyo=cc[ci*2+1], kbo=kbc[ci];
                    double m1o=gm1[(size_t)ci*nvar+v], m2o=gm2[(size_t)ci*nvar+v];
                    double qmno=qmnc[(size_t)ci*nvar+v], qmxo=qmxc[(size_t)ci*nvar+v], rngo=qmxo-qmno;
                    double Qo=2.0*(W[(size_t)v*N+ci]-qmno)/rngo-1.0; if(Qo>0.999)Qo=0.999; else if(Qo<-0.999)Qo=-0.999;
                    double aQo=0.5*std::log((1.0+Qo)/(1.0-Qo));
                    double sco2o=m2o-m1o*m1o; if(sco2o<0)sco2o=0;   // owner cell variance (beta-indep) hoisted ONCE per cell
                    struct FM{ double F1o,F1n,kbn,aQn,m1n,qmnn,rngn,seo2,scn2,sen2; };   // precomputed beta-indep variances (replace raw F2o,F2n,m2n)
                    FM fm[16]; int nf=0;
                    for(int f:m.cell_faces[ci]){ int o=m.face_owner[f],n=m.face_neighbour[f],nb=(o==ci)?n:o; if(nb<0)continue;
                        if(!hasint[(size_t)nb*nvar+v]) continue; const auto& fn=m.face_nodes[f]; if(fn.size()<2)continue; if(nf>=16)break;
                        double ax=m.nodes[fn[0]*2],ay=m.nodes[fn[0]*2+1],bx=m.nodes[fn[1]*2],by=m.nodes[fn[1]*2+1];
                        double ex=bx-ax,ey=by-ay, dxo=ax-cxo,dyo=ay-cyo;
                        double p2=Ao[2]*ex*ex+Ao[3]*ey*ey+Ao[4]*ex*ey;
                        double p1=Ao[0]*ex+Ao[1]*ey+2.0*Ao[2]*dxo*ex+2.0*Ao[3]*dyo*ey+Ao[4]*(dxo*ey+dyo*ex);
                        double p0=Ao[0]*dxo+Ao[1]*dyo+Ao[2]*dxo*dxo+Ao[3]*dyo*dyo+Ao[4]*dxo*dyo;
                        double F1o=p2/3.0+p1/2.0+p0, F2o=p2*p2/5.0+p1*p2/2.0+(p1*p1+2.0*p0*p2)/3.0+p0*p1+p0*p0;
                        const double* An=&acoef[((size_t)nb*nvar+v)*5];
                        double dxn=ax-cc[nb*2],dyn=ay-cc[nb*2+1];
                        double s2=An[2]*ex*ex+An[3]*ey*ey+An[4]*ex*ey;
                        double s1=An[0]*ex+An[1]*ey+2.0*An[2]*dxn*ex+2.0*An[3]*dyn*ey+An[4]*(dxn*ey+dyn*ex);
                        double s0=An[0]*dxn+An[1]*dyn+An[2]*dxn*dxn+An[3]*dyn*dyn+An[4]*dxn*dyn;
                        double F1n=s2/3.0+s1/2.0+s0, F2n=s2*s2/5.0+s1*s2/2.0+(s1*s1+2.0*s0*s2)/3.0+s0*s1+s0*s0;
                        double m1n=gm1[(size_t)nb*nvar+v],m2n=gm2[(size_t)nb*nvar+v];
                        double qmnn=qmnc[(size_t)nb*nvar+v],qmxn=qmxc[(size_t)nb*nvar+v],rngn=qmxn-qmnn;
                        double Qn=2.0*(W[(size_t)v*N+nb]-qmnn)/rngn-1.0; if(Qn>0.999)Qn=0.999; else if(Qn<-0.999)Qn=-0.999;
                        double aQn=0.5*std::log((1.0+Qn)/(1.0-Qn));
                        double seo2=F2o-F1o*F1o; if(seo2<0)seo2=0;   // beta-indep variances precomputed once per face
                        double scn2=m2n-m1n*m1n; if(scn2<0)scn2=0;
                        double sen2=F2n-F1n*F1n; if(sen2<0)sen2=0;
                        fm[nf]={F1o,F1n,kbc[nb],aQn,m1n,qmnn,rngn,seo2,scn2,sen2}; nf++; }
                    if(nf==0) bstar=beta_l;
                    else {
                        auto tbvh=[&](double beta)->double{ double kko=beta*kbo, vo0=kko*kko*sco2o;
                            double Do=aQo*std::sqrt(1.0+GC*vo0)-kko*m1o; double tb=0.0;
                            for(int j=0;j<nf;++j){ const FM&F=fm[j];
                                double vvo=kko*kko*F.seo2;
                                double tho=std::tanh((kko*F.F1o+Do)/std::sqrt(1.0+GC*vvo));
                                double qo=qmno+0.5*rngo*(1.0+tho); if(qo<qmno)qo=qmno; else if(qo>qmxo)qo=qmxo;
                                double kkn=beta*F.kbn, vn0=kkn*kkn*F.scn2;
                                double Dn=F.aQn*std::sqrt(1.0+GC*vn0)-kkn*F.m1n;
                                double vvn=kkn*kkn*F.sen2;
                                double thn=std::tanh((kkn*F.F1n+Dn)/std::sqrt(1.0+GC*vvn));
                                double qxn=F.qmnn+F.rngn, qn=F.qmnn+0.5*F.rngn*(1.0+thn); if(qn<F.qmnn)qn=F.qmnn; else if(qn>qxn)qn=qxn;
                                tb+=std::fabs(qo-qn); }
                            return tb; };
                        // Jf(j,beta)=q_owner-q_neighbour at face j (UNCLAMPED, smooth for Newton); *dJ=dJ/dbeta (closed-form).
                        auto Jf=[&](int j,double beta,double* dJ)->double{ const FM&F=fm[j];
                            double sco2=sco2o, scn2=F.scn2, seo2=F.seo2, sen2=F.sen2;   // beta-indep, precomputed (no recompute per call)
                            double ko=beta*kbo, rDo=std::sqrt(1.0+GC*ko*ko*sco2), Do=aQo*rDo-ko*m1o;
                            double rvo=std::sqrt(1.0+GC*ko*ko*seo2), tho=std::tanh((ko*F.F1o+Do)/rvo);
                            double qo=qmno+0.5*rngo*(1.0+tho);
                            double kn=beta*F.kbn, rDn=std::sqrt(1.0+GC*kn*kn*scn2), Dn=F.aQn*rDn-kn*F.m1n;
                            double rvn=std::sqrt(1.0+GC*kn*kn*sen2), thn=std::tanh((kn*F.F1n+Dn)/rvn);
                            double qn=F.qmnn+0.5*F.rngn*(1.0+thn);
                            if(dJ){ double dDo=aQo*(GC*ko*sco2*kbo)/rDo-m1o*kbo, drvo=(GC*ko*seo2*kbo)/rvo;
                                double dao=((kbo*F.F1o+dDo)*rvo-(ko*F.F1o+Do)*drvo)/(rvo*rvo);
                                double dqo=0.5*rngo*(1.0-tho*tho)*dao;
                                double dDn=F.aQn*(GC*kn*scn2*F.kbn)/rDn-F.m1n*F.kbn, drvn=(GC*kn*sen2*F.kbn)/rvn;
                                double dan=((F.kbn*F.F1n+dDn)*rvn-(kn*F.F1n+Dn)*drvn)/(rvn*rvn);
                                double dqn=0.5*F.rngn*(1.0-thn*thn)*dan; *dJ=dqo-dqn; }
                            return qo-qn; };
                        if(BSTAR_FASTMODE){   // FAST alternatives (L2/HUBER/IRLS/DOM/MODMED) — all via closed-form Jf
                            double b; const double eps=1e-3*(rngo+1e-30);
                            auto Sobj=[&](double bb)->double{ double s=0; for(int j=0;j<nf;++j){ double J=Jf(j,bb,nullptr);
                                if(BSTAR_FASTMODE==1) s+=J*J; else if(BSTAR_FASTMODE==2) s+=std::sqrt(J*J+eps*eps); else s+=std::fabs(J); } return s; };
                            auto Ggrad=[&](double bb,const double* w)->double{ double s=0; for(int j=0;j<nf;++j){ double dJ,J=Jf(j,bb,&dJ);
                                double we = w? w[j] : ((BSTAR_FASTMODE==2)? 1.0/std::sqrt(J*J+eps*eps) : 1.0); s+=we*J*dJ; } return s; };
                            if(BSTAR_FASTMODE==6){   // L2GN: Gauss-Newton on min sum J^2 -> step = (sum J*J')/(sum J'^2), no scan
                                b=beta_l;
                                for(int it=0;it<4;++it){ double num=0,den=0;
                                    for(int j=0;j<nf;++j){ double dJ,J=Jf(j,b,&dJ); num+=J*dJ; den+=dJ*dJ; }
                                    if(den<1e-30) break; double step=num/den; b-=step;
                                    if(b<blo)b=blo; else if(b>bhi)b=bhi;
                                    if(std::fabs(step)<1e-6) break; }
                            } else if(BSTAR_FASTMODE==7){   // WMGN: L2GN seed beta0, then linearized-L1 weighted median at beta0
                                double b0=beta_l;
                                for(int it=0;it<2;++it){ double num=0,den=0;
                                    for(int j=0;j<nf;++j){ double dJ,J=Jf(j,b0,&dJ); num+=J*dJ; den+=dJ*dJ; }
                                    if(den<1e-30) break; b0-=num/den; if(b0<blo)b0=blo; else if(b0>bhi)b0=bhi; }
                                double bf[16],wf[16]; int mf=0;   // r_f=b0-J/J', w=|J'| at the good seed b0
                                for(int j=0;j<nf;++j){ double dJ,J=Jf(j,b0,&dJ); if(std::fabs(dJ)<1e-12)continue;
                                    double rf=b0-J/dJ; if(rf<blo)rf=blo; else if(rf>bhi)rf=bhi; bf[mf]=rf; wf[mf]=std::fabs(dJ); mf++; }
                                if(mf==0) b=b0;
                                else { for(int x=1;x<mf;++x){ double bb2=bf[x],ww=wf[x]; int y=x-1; while(y>=0&&bf[y]>bb2){bf[y+1]=bf[y];wf[y+1]=wf[y];--y;} bf[y+1]=bb2;wf[y+1]=ww; }
                                    double wt=0; for(int x=0;x<mf;++x)wt+=wf[x]; double half=0.5*wt,acc=0; b=bf[mf-1];
                                    for(int x=0;x<mf;++x){ acc+=wf[x]; if(acc>=half){b=bf[x];break;} } }
                            } else if(BSTAR_FASTMODE==4){   // DOM: kink of the |J|-dominant face at beta_l
                                double bref=beta_l, jm=-1; int jd=0;
                                for(int j=0;j<nf;++j){ double J=Jf(j,bref,nullptr); if(std::fabs(J)>jm){jm=std::fabs(J);jd=j;} }
                                b=bref; for(int it=0;it<8;++it){ double dJ,J=Jf(jd,b,&dJ);
                                    if(std::fabs(J)<1e-11*(rngo+1e-30))break; if(std::fabs(dJ)<1e-30)break;
                                    double bn=b-J/dJ; b=(bn>blo&&bn<bhi)?bn:0.5*(blo+bhi); }
                            } else if(BSTAR_FASTMODE==5){   // MODMED: linearized per-face kink weighted median
                                double b0=beta_l, bf[16],wf[16]; int mf=0;
                                for(int j=0;j<nf;++j){ double dJ,J=Jf(j,b0,&dJ); if(std::fabs(dJ)<1e-30)continue;
                                    double bfj=b0-J/dJ; if(bfj<blo)bfj=blo; else if(bfj>bhi)bfj=bhi; bf[mf]=bfj; wf[mf]=std::fabs(dJ); mf++; }
                                if(mf==0) b=beta_l;
                                else { for(int x=1;x<mf;++x){ double bb2=bf[x],ww=wf[x]; int y=x-1; while(y>=0&&bf[y]>bb2){bf[y+1]=bf[y];wf[y+1]=wf[y];--y;} bf[y+1]=bb2;wf[y+1]=ww; }
                                    double wt=0; for(int x=0;x<mf;++x)wt+=wf[x]; double half=0.5*wt,acc=0; b=bf[mf-1];
                                    for(int x=0;x<mf;++x){ acc+=wf[x]; if(acc>=half){b=bf[x];break;} } }
                            } else if(BSTAR_FASTMODE==2){   // HUBER: coarse Sobj scan -> basin, then Newton-POLISH on G_h (not 20-iter bisect)
                                auto Gh=[&](double bb)->double{ double s=0; for(int j=0;j<nf;++j){ double dJ,J=Jf(j,bb,&dJ); s+=J*dJ/std::sqrt(J*J+eps*eps);} return s; };
                                const int NC=6; double bb=blo, best=Sobj(blo);
                                for(int k=1;k<NC;++k){ double x=blo+(bhi-blo)*k/(NC-1); double f=Sobj(x); if(f<best){best=f;bb=x;} }
                                double hw=(bhi-blo)/(NC-1);   // stay within the scan cell around the min (keeps Newton in the right basin)
                                double blo2=bb-hw; if(blo2<blo)blo2=blo; double bhi2=bb+hw; if(bhi2>bhi)bhi2=bhi;
                                b=bb; double db=1e-4*(bhi-blo+1e-30);
                                for(int it=0;it<5;++it){ double g=Gh(b); if(std::fabs(g)<1e-13){break;}
                                    double gp=(Gh(b+db)-g)/db; if(std::fabs(gp)<1e-30) break;
                                    double bn=b-g/gp; if(bn<blo2)bn=blo2; else if(bn>bhi2)bn=bhi2;
                                    if(std::fabs(bn-b)<1e-7){b=bn;break;} b=bn; }
                            } else {   // L2(1)/IRLS(3): coarse-bracket then bisect stationary G=0
                                int OUT=(BSTAR_FASTMODE==3)?3:1; double wI[16]; for(int j=0;j<nf;++j) wI[j]=1.0; b=beta_l;
                                for(int oit=0; oit<OUT; ++oit){ const double* wp=(BSTAR_FASTMODE==3)?wI:nullptr;
                                    const int NC=6; double bb0=blo,best=1e300;
                                    for(int k=0;k<NC;++k){ double x=blo+(bhi-blo)*k/(NC-1);
                                        double f; if(BSTAR_FASTMODE==3){ f=0; for(int j=0;j<nf;++j){ double J=Jf(j,x,nullptr); f+=wI[j]*J*J; } } else f=Sobj(x);
                                        if(f<best){best=f;bb0=x;} }
                                    double lo=bb0-(bhi-blo)/(NC-1); if(lo<blo)lo=blo; double hi=bb0+(bhi-blo)/(NC-1); if(hi>bhi)hi=bhi;
                                    double glo=Ggrad(lo,wp),ghi=Ggrad(hi,wp); b=bb0;
                                    if(glo*ghi<0){ for(int it=0;it<20;++it){ double mb=0.5*(lo+hi); double g=Ggrad(mb,wp);
                                        if(std::fabs(g)<1e-13){b=mb;break;} if(g*glo<0)hi=mb; else {lo=mb;glo=g;} b=0.5*(lo+hi); } }
                                    if(BSTAR_FASTMODE==3) for(int j=0;j<nf;++j){ double J=Jf(j,b,nullptr); wI[j]=1.0/std::max(std::fabs(J),1e-3*(rngo+1e-30)); } }
                            }
                            if(b<blo)b=blo; else if(b>bhi)b=bhi; bstar=b;   // lower bound = blo(=WMIN, the ln3 accuracy floor by default)
                        } else if(BSTAR_KINK){   // EXACT global argmin: kink-enumeration + per-segment stationary points
                            double LO=blo, HI=bhi, kf[16]; int nk=0; double pcw=-1.0;
                            if(BSTAR_WARM && !bstar_full){   // temporal warm-start: restrict enum to a narrow window around cached beta*
                                pcw=bstv_warm[(size_t)v*N+ci]; LO=pcw-BSTAR_WARMW; if(LO<blo)LO=blo;
                                HI=pcw+BSTAR_WARMW; if(HI>bhi)HI=bhi; if(HI-LO<1e-3){LO=blo;HI=bhi;pcw=-1.0;} }
                            for(int j=0;j<nf;++j){   // kink beta_f = root of Jf=0 (bracketed safeguarded Newton, direct, no squaring)
                                double jlo=Jf(j,LO,nullptr), jhi=Jf(j,HI,nullptr);
                                if(jlo*jhi>0.0) continue;   // no sign change -> saturated face (no kink in [LO,HI])
                                double lo2=LO,hi2=HI,b=0.5*(LO+HI);
                                for(int it=0;it<BSTAR_KNEWT;++it){ double dJ,jb=Jf(j,b,&dJ);
                                    if(std::fabs(jb)<1e-11*(rngo+1e-30)) break;
                                    if(jb*jlo<0.0) hi2=b; else { lo2=b; jlo=jb; }
                                    double bn=(std::fabs(dJ)>1e-30)? b-jb/dJ : 0.5*(lo2+hi2);
                                    b=(bn>lo2 && bn<hi2)? bn : 0.5*(lo2+hi2); }
                                if(nk<16) kf[nk++]=b; }
                            double cand[48]; int ncd=0; cand[ncd++]=LO; cand[ncd++]=HI;
                            if(pcw>LO && pcw<HI) cand[ncd++]=pcw;   // keep cached beta* as candidate on warm calls
                            for(int k=0;k<nk;++k) cand[ncd++]=kf[k];
                            for(int x=1;x<nk;++x){ double t=kf[x]; int y=x-1; while(y>=0&&kf[y]>t){kf[y+1]=kf[y];--y;} kf[y+1]=t; }
                            double brk[18]; int nbk=0; brk[nbk++]=LO; for(int k=0;k<nk;++k) brk[nbk++]=kf[k]; brk[nbk++]=HI;
                            if(BSTAR_SEG) for(int s=0;s<nbk-1 && ncd<47;++s){ double lo3=brk[s],hi3=brk[s+1]; if(hi3-lo3<1e-6) continue;
                                double b=0.5*(lo3+hi3);   // segment interior stationary: TBV'(beta)=sum sign(J)*J' = 0
                                for(int it=0;it<12;++it){ double g=0,g2=0,db=1e-5;
                                    for(int j=0;j<nf;++j){ double dJ; double J=Jf(j,b,&dJ); g+=((J>0)?1.0:-1.0)*dJ;
                                        double dJ2; double J2=Jf(j,b+db,&dJ2); g2+=((J2>0)?1.0:-1.0)*dJ2; }
                                    double gp=(g2-g)/db; if(std::fabs(gp)<1e-30) break;
                                    double bn=b-g/gp; if(bn>lo3 && bn<hi3) b=bn; else break;
                                    if(std::fabs(g)<1e-12) break; }
                                if(b>lo3 && b<hi3) cand[ncd++]=b; }
                            double best=1e300; bstar=LO;   // compare exact (clamped) TBV at all candidates -> global argmin
                            for(int c=0;c<ncd;++c){ double tb=tbvh(cand[c]); if(tb<best){best=tb; bstar=cand[c];} }
                            if(bstar<blo)bstar=blo; else if(bstar>bhi)bstar=bhi;   // lower bound = blo(=WMIN, ln3 floor by default)
                        } else {
                        double lo=blo, hi=bhi;
                        if(BSTAR_WARM && !bstar_full){ double pc=bstv_warm[(size_t)v*N+ci]; double w=0.7;
                            lo=pc-w; if(lo<blo)lo=blo; hi=pc+w; if(hi>bhi)hi=bhi; if(hi-lo<1e-3){lo=blo;hi=bhi;} }
                        else {   // coarse scan brackets the GLOBAL min (TBV not strictly unimodal -> pure golden lands in a local min)
                            const int NCO=6; double bstep=(bhi-blo)/(NCO-1); int im=0; double fmin=tbvh(blo);
                            for(int k=1;k<NCO;++k){ double fk=tbvh(blo+bstep*k); if(fk<fmin){fmin=fk;im=k;} }
                            double bc=blo+bstep*im; lo=bc-bstep; if(lo<blo)lo=blo; hi=bc+bstep; if(hi>bhi)hi=bhi; }
                        const double GR=0.6180339887498949;
                        double c1=hi-GR*(hi-lo), c2=lo+GR*(hi-lo), f1=tbvh(c1), f2=tbvh(c2);
                        for(int it=0; it<BSTAR_GITER && (hi-lo)>0.08; ++it){
                            if(f1<f2){ hi=c2; c2=c1; f2=f1; c1=hi-GR*(hi-lo); f1=tbvh(c1); }
                            else { lo=c1; c1=c2; f1=f2; c2=lo+GR*(hi-lo); f2=tbvh(c2); } }
                        bstar=0.5*(lo+hi);
                        }
                    }
                    if(BSTAR_WARM) bstv_warm[(size_t)v*N+ci]=bstar;
                } else {
                    // collect interface-neighbour faces once (endpoints)
                    double TB[16]; int gi;   // TBV at NB grid points
                    for(gi=0;gi<NB;++gi){ double beta=blo+(bhi-blo)*gi/(NB-1); double Di=Dbeta(ci,v,beta);
                        double tb=0.0;
                        for(int f:m.cell_faces[ci]){ int o=m.face_owner[f],n=m.face_neighbour[f],nb=(o==ci)?n:o; if(nb<0)continue;
                            if(!hasint[(size_t)nb*nvar+v]) continue; const auto& fn=m.face_nodes[f]; if(fn.size()<2)continue;
                            double axx=m.nodes[fn[0]*2],ayy=m.nodes[fn[0]*2+1],bxx=m.nodes[fn[1]*2],byy=m.nodes[fn[1]*2+1];
                            double Dj=Dbeta(nb,v,beta);
                            double qi=qface(ci,v,beta,Di,axx,ayy,bxx,byy), qj=qface(nb,v,beta,Dj,axx,ayy,bxx,byy);
                            tb+=std::fabs(qi-qj); }
                        TB[gi]=tb; }
                    int imn=0; for(gi=1;gi<NB;++gi) if(TB[gi]<TB[imn]) imn=gi;
                    if(BSTAR_MAXB){   // option c: largest beta with TBV<=TOL*min
                        double thr=BSTAR_TOL*TB[imn]; int ib=imn; for(gi=NB-1;gi>=imn;--gi) if(TB[gi]<=thr){ib=gi;break;}
                        bstar=blo+(bhi-blo)*ib/(NB-1);
                    } else bstar=blo+(bhi-blo)*imn/(NB-1);
                }
                if(v==0){   // density -> beff + binc (drives pick branching, export, DIAG)
                    // WIDE (option A): no 2-beta fallback -> always {MUSCL, THINC(beta*)} (binc=1).
                    if(BSTAR_WIDE) binc[ci]=1;
                    else { int ibsel=(int)std::lround((bstar-beta_s)/(beta_l-beta_s)*(NB-1));
                           binc[ci]=(ibsel>0 && ibsel<NB-1)?1:0; }
                    beff[ci]=bstar; }
                if(BSTAR_PERVAR) bstv[(size_t)ci*nvar+v]=bstar;
            } }
        static const bool DIAG=std::getenv("BOPT_DIAG")!=nullptr;
        if(DIAG){ static int cc2=0; if((cc2++%200)==0){ double bmn=1e30,bmx=-1e30,bs=0; int nb=0,nin=0;
            for(int ci=0;ci<N;++ci){ if(!hasint[(size_t)ci*nvar+0])continue; double b=beff[ci]; if(b<bmn)bmn=b; if(b>bmx)bmx=b; bs+=b; nb++; if(binc[ci])nin++; }
            std::fprintf(stderr,"BSTAR_EXACT beta*: min=%.2f mean=%.2f max=%.2f interior-frac=%.2f (n_if=%d)\n",bmn,bs/(nb>0?nb:1),bmx,(double)nin/(nb>0?nb:1),nb); } }
        // BSTAR_CURVE=x0,x1,y0,y1 diagnostic: at recon call #BSTAR_CURVE_AT (default 3000, mid-run),
        // dump TB(beta) rows for <=50 interface cells inside the box -> shows WHERE argmin sits
        // (flat curve? monotone to diffusive end?). One-shot, stderr, serial (50 cells, cheap).
        static const char* CRV = std::getenv("BSTAR_CURVE");
        if(CRV){ static const int CAT = std::getenv("BSTAR_CURVE_AT")?std::atoi(std::getenv("BSTAR_CURVE_AT")):3000;
            static int callc=0; static bool done=false; ++callc;
            if(!done && callc>=CAT){ done=true;
                double bx0,bx1,by0,by1;
                if(std::sscanf(CRV,"%lf,%lf,%lf,%lf",&bx0,&bx1,&by0,&by1)==4){ int nd=0;
                    for(int ci=0;ci<N && nd<50;++ci){ if(!hasint[(size_t)ci*nvar+0]) continue;
                        double x=cc[ci*2],y=cc[ci*2+1]; if(x<bx0||x>bx1||y<by0||y>by1) continue;
                        std::fprintf(stderr,"BSTAR_CURVE ci=%d x=%.4g y=%.4g beta*=%.3g TB:",ci,x,y,beff[ci]);
                        for(int gi2=0;gi2<NB;++gi2){ double beta=blo+(bhi-blo)*gi2/(NB-1); double Di=Dbeta(ci,0,beta); double tb=0.0;
                            for(int f:m.cell_faces[ci]){ int o=m.face_owner[f],n2=m.face_neighbour[f],nb2=(o==ci)?n2:o; if(nb2<0)continue;
                                if(!hasint[(size_t)nb2*nvar+0]) continue; const auto& fn=m.face_nodes[f]; if(fn.size()<2)continue;
                                double axx=m.nodes[fn[0]*2],ayy=m.nodes[fn[0]*2+1],bxx=m.nodes[fn[1]*2],byy=m.nodes[fn[1]*2+1];
                                double Dj=Dbeta(nb2,0,beta);
                                tb+=std::fabs(qface(ci,0,beta,Di,axx,ayy,bxx,byy)-qface(nb2,0,beta,Dj,axx,ayy,bxx,byy)); }
                            std::fprintf(stderr," %.4g",tb); }
                        std::fprintf(stderr,"\n"); ++nd; }
                    std::fprintf(stderr,"BSTAR_CURVE dumped %d cells at call %d (NB=%d beta[%.2g,%.2g])\n",nd,callc,NB,blo,bhi); } } }
    }
    else if(BOPT||BOPT3){ beff.assign(N, beta_l); if(BOPT3) binc.assign(N,0);   // BVD-optimal per-cell beta* (median approx); BOPT3 -> beta* is a 4th candidate (included only if in range)
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){
            if(!hasint[(size_t)ci*nvar+0]) continue;
            double qmn=qmnc[(size_t)ci*nvar+0], qmx=qmxc[(size_t)ci*nvar+0], rng=qmx-qmn; if(rng<1e-12) continue;
            const double* A=&acoef[((size_t)ci*nvar+0)*5]; double nx=A[0], ny=A[1];
            // TBV-min single-beta* (L1 = canonical BVD): weighted MEDIAN of per-face zero-jump beta_f*=(D_nb-D_i)/a.
            // EXACT-CELL-D from Pass1 (Dl solved at beta_l), NOT centroid. [[exact-cell-D-mandatory]]. clamp [BMIN,BMAX].
            double Di = Dl[(size_t)ci*nvar+0];
            double bf[16], wf[16]; int nf=0;
            for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                if(!hasint[(size_t)nb*nvar+0]) continue;          // exact-cell-D defined only on interface cells
                double a=((cc[nb*2]-cc[ci*2])*nx+(cc[nb*2+1]-cc[ci*2+1])*ny)*kbc[ci];
                if(std::fabs(a)<1e-6) continue;
                double b=(Dl[(size_t)nb*nvar+0]-Di)/a; if(b<0)b=-b;
                if(nf<16){ bf[nf]=b; wf[nf]=std::fabs(a); nf++; } }
            if(nf==0) continue;
            for(int x=1;x<nf;++x){ double bb=bf[x],ww=wf[x]; int y=x-1; while(y>=0&&bf[y]>bb){bf[y+1]=bf[y];wf[y+1]=wf[y];--y;} bf[y+1]=bb;wf[y+1]=ww; }
            double wt=0; for(int x=0;x<nf;++x)wt+=wf[x]; double half=0.5*wt, acc=0, bstar=bf[nf-1];
            for(int x=0;x<nf;++x){ acc+=wf[x]; if(acc>=half){bstar=bf[x];break;} }   // weighted median = L1 TBV-min
            if(BOPT3){ binc[ci] = (bstar>=beta_s && bstar<=beta_l)?1:0; }   // 4th candidate included only if beta* in [beta_s,beta_l]
            else { if(bstar<BOPT_BMIN)bstar=BOPT_BMIN; else if(bstar>BOPT_BMAX)bstar=BOPT_BMAX; }
            beff[ci]=bstar; }
        static const bool DIAG=std::getenv("BOPT_DIAG")!=nullptr;
        if(DIAG){ static int cc=0; if((cc++%200)==0){ double bmn=1e30,bmx=-1e30,bs=0; int nb=0,nlo=0;
            for(int ci=0;ci<N;++ci){ if(!hasint[(size_t)ci*nvar+0])continue; double b=beff[ci]; if(b<bmn)bmn=b; if(b>bmx)bmx=b; bs+=b; nb++; if(b<0.6)nlo++; }
            std::fprintf(stderr,"BOPT_DIAG beta*: min=%.2f mean=%.2f max=%.2f frac<0.6=%.2f (n_if=%d)\n",bmn,bs/(nb>0?nb:1),bmx,(double)nlo/(nb>0?nb:1),nb); } } }
    if(BOPT && ASIG_NEWTON){   // BOPT+trueNewton: re-solve D via AST cell-quadrature Newton at the per-cell beta*
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ const auto& vs=m.cell_nodes[ci]; if((int)vs.size()<3) continue;
            for(int v=0;v<nvar;++v){ if(!hasint[(size_t)ci*nvar+v]) continue;
                double kk=(BSTAR_PERVAR?bstv[(size_t)ci*nvar+v]:beff[ci])*kbc[ci];
                const double* A=&acoef[((size_t)ci*nvar+v)*5];
                double qmn=qmnc[(size_t)ci*nvar+v],qmx=qmxc[(size_t)ci*nvar+v],rng=qmx-qmn; if(rng<1e-12) continue;
                double Q=2.0*(W[(size_t)v*N+ci]-qmn)/rng-1.0;
                if(SPL_CENTROID){ double Qc=Q; if(Qc>0.999999)Qc=0.999999; else if(Qc<-0.999999)Qc=-0.999999;
                    Dstar[(size_t)ci*nvar+v]=0.5*std::log((1.0+Qc)/(1.0-Qc)); continue; }   // centroid-D beta* candidate (consistent with centroid Dl/Ds), no Newton
                double Pg[C3_NQMAX], cwq[C3_NQMAX];
                const int NQ=c3_cell_quad_P(TQp,NQC,m.nodes.data(),vs.data(),(int)vs.size(),cc[ci*2],cc[ci*2+1],A,Pg,cwq);
                double D=Q/std::max(1.0-std::fabs(Q),1e-9);
                for(int it=0;it<12;++it){ double f=-Q,fp=0.0;
                    for(int q=0;q<NQ;++q){ double s=kk*Pg[q]+D, den=1.0+std::fabs(s); f+=cwq[q]*(s/den); fp+=cwq[q]/(den*den); }
                    if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD; if(D>1e4)D=1e4; else if(D<-1e4)D=-1e4; if(std::fabs(dD)<1e-11)break; }
                Dl[(size_t)ci*nvar+v]=D; } } }
    if(BOPT && !SPL && !ASIG && !PST && !ANALYTIC_D){   // BOPT + tanh: re-solve EXACT-CELL-D via tanh cell-quad Newton at per-cell beta*
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ const auto& vs=m.cell_nodes[ci]; if((int)vs.size()<3) continue;
            for(int v=0;v<nvar;++v){ if(!hasint[(size_t)ci*nvar+v]) continue;
                double kk=(BSTAR_PERVAR?bstv[(size_t)ci*nvar+v]:beff[ci])*kbc[ci];
                const double* A=&acoef[((size_t)ci*nvar+v)*5];
                double qmn=qmnc[(size_t)ci*nvar+v],qmx=qmxc[(size_t)ci*nvar+v],rng=qmx-qmn; if(rng<1e-12) continue;
                double Q=2.0*(W[(size_t)v*N+ci]-qmn)/rng-1.0;
                if(SPL_CENTROID){ Dstar[(size_t)ci*nvar+v]=Dl[(size_t)ci*nvar+v]; continue; }   // centroid-D is beta-INDEPENDENT (sigma^-1(Q)) -> beta* candidate D = Dl (no Newton, consistent w/ beta_l/s)
                double Pg[C3_NQMAX], cwq[C3_NQMAX];
                const int NQ=c3_cell_quad_P(TQp,NQC,m.nodes.data(),vs.data(),(int)vs.size(),cc[ci*2],cc[ci*2+1],A,Pg,cwq);
                double Ag[C3_NQMAX]; for(int q=0;q<NQ;++q) Ag[q]=std::tanh(kk*Pg[q]);   // exact tanh (S1 high-accuracy: cell-D must match the exact-tanh face for conservation)
                double D=0.0;
                for(int it=0;it<12;++it){ double f=-Q,fp=0.0;
                    for(int q=0;q<NQ;++q){ double den=1.0+Ag[q]*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12);
                        f+=cwq[q]*(Ag[q]+D)/den; fp+=cwq[q]*(1.0-Ag[q]*Ag[q])/(den*den); }
                    if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD;
                    if(D>0.999999)D=0.999999; else if(D<-0.999999)D=-0.999999; if(std::fabs(dD)<1e-11)break; }
                Dl[(size_t)ci*nvar+v]=D; } } }
    if(BOPT && SPL){   // BOPT + spliced(quintic-Hermite): re-solve D via spl_sig cell-quad Newton at per-cell beta*
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ const auto& vs=m.cell_nodes[ci]; if((int)vs.size()<3) continue;
            for(int v=0;v<nvar;++v){ if(!hasint[(size_t)ci*nvar+v]) continue;
                double kk=(BSTAR_PERVAR?bstv[(size_t)ci*nvar+v]:beff[ci])*kbc[ci];
                const double* A=&acoef[((size_t)ci*nvar+v)*5];
                double qmn=qmnc[(size_t)ci*nvar+v],qmx=qmxc[(size_t)ci*nvar+v],rng=qmx-qmn; if(rng<1e-12) continue;
                double Q=2.0*(W[(size_t)v*N+ci]-qmn)/rng-1.0;
                double Pg[C3_NQMAX], cwq[C3_NQMAX];
                const int NQ=c3_cell_quad_P(TQp,NQC,m.nodes.data(),vs.data(),(int)vs.size(),cc[ci*2],cc[ci*2+1],A,Pg,cwq);
                double D=Q;
                for(int it=0;it<14;++it){ double f=-Q,fp=0.0;
                    for(int q=0;q<NQ;++q){ double sg,sgd; spl_sig_and_d(kk*Pg[q]+D,sg,sgd); f+=cwq[q]*sg; fp+=cwq[q]*sgd; }
                    if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD; if(D>1e4)D=1e4; else if(D<-1e4)D=-1e4; if(std::fabs(dD)<1e-11)break; }
                Dl[(size_t)ci*nvar+v]=D; } } }
    if(BOPT3){   // BOPT3: solve EXACT-CELL-D at per-cell beta* into Dstar (4th candidate), all interface cells (jstar needs both sides)
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ const auto& vs=m.cell_nodes[ci]; if((int)vs.size()<3) continue;
            for(int v=0;v<nvar;++v){ if(!hasint[(size_t)ci*nvar+v]) continue;
                double kk=(BSTAR_PERVAR?bstv[(size_t)ci*nvar+v]:beff[ci])*kbc[ci];
                const double* A=&acoef[((size_t)ci*nvar+v)*5];
                double qmn=qmnc[(size_t)ci*nvar+v],qmx=qmxc[(size_t)ci*nvar+v],rng=qmx-qmn; if(rng<1e-12) continue;
                double Q=2.0*(W[(size_t)v*N+ci]-qmn)/rng-1.0;
                double Pg[C3_NQMAX], cwq[C3_NQMAX]; int NQ=0;
                if(!(GAUSS && !BSTAR_TANHRECON))   // GAUSS path is moment-algebraic -> no point evaluation needed
                    NQ=c3_cell_quad_P(TQp,NQC,m.nodes.data(),vs.data(),(int)vs.size(),cc[ci*2],cc[ci*2+1],A,Pg,cwq);
                double D;
                if(GAUSS && !BSTAR_TANHRECON){   // probit closed-form D at beta* (consistent with the GAUSS beta_l/beta_s Dl/Ds); Option B skips this -> falls to tanh Newton cell-D below
                    double mm1,mm2; c3_gmom_moments(GT.g[ci],A,mm1,mm2);
                    double Qc=Q; if(Qc>0.999)Qc=0.999; else if(Qc<-0.999)Qc=-0.999; double aQ=0.5*std::log((1.0+Qc)/(1.0-Qc));
                    double vv=kk*kk*(mm2-mm1*mm1); if(vv<0)vv=0; D=aQ*std::sqrt(1.0+GC*vv)-kk*mm1; }
                else if(SPL){ D=Q; for(int it=0;it<14;++it){ double f=-Q,fp=0.0;
                        for(int q=0;q<NQ;++q){ double sg,sgd; spl_sig_and_d(kk*Pg[q]+D,sg,sgd); f+=cwq[q]*sg; fp+=cwq[q]*sgd; }
                        if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD; if(D>1e4)D=1e4; else if(D<-1e4)D=-1e4; if(std::fabs(dD)<1e-11)break; } }
                else { double Ag[C3_NQMAX]; for(int q=0;q<NQ;++q) Ag[q]=std::tanh(kk*Pg[q]); D=0.0;
                    for(int it=0;it<12;++it){ double f=-Q,fp=0.0;
                        for(int q=0;q<NQ;++q){ double den=1.0+Ag[q]*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12);
                            f+=cwq[q]*(Ag[q]+D)/den; fp+=cwq[q]*(1.0-Ag[q]*Ag[q])/(den*den); }
                        if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD;
                        if(D>0.999999)D=0.999999; else if(D<-0.999999)D=-0.999999; if(std::fabs(dD)<1e-11)break; } }
                Dstar[(size_t)ci*nvar+v]=D; } } }
    // Pass 2 (face loop): THINC_l, THINC_s candidate face values + per-candidate jumps.
    // H2 (perf): persistent scratch reused across calls -> avoids malloc/free + first-touch page faults every
    // recon (recon is called serially, so function-static shared workspace is safe). The .assign re-zeros so
    // values are byte-identical to the old per-call vectors; only the allocation/fault overhead is removed.
    static std::vector<double> WL1,WR1,WL0,WR0; WL1.assign((size_t)nvar*Nf,0.0);WR1.assign((size_t)nvar*Nf,0.0);WL0.assign((size_t)nvar*Nf,0.0);WR0.assign((size_t)nvar*Nf,0.0);
    static std::vector<double> WLst, WRst, jst; if(BOPT3){ WLst.assign((size_t)nvar*Nf,0.0); WRst.assign((size_t)nvar*Nf,0.0); jst.assign((size_t)nvar*Nf,0.0); }
    static std::vector<double> jM,j1,j0; jM.assign((size_t)nvar*Nf,0.0);j1.assign((size_t)nvar*Nf,0.0);j0.assign((size_t)nvar*Nf,0.0);
    #pragma omp parallel for schedule(dynamic,32)
    for(int f=0;f<Nf;++f){ int o=m.face_owner[f],n=m.face_neighbour[f]; const auto& fn=m.face_nodes[f];
        bool he=fn.size()>=2; double ax=0,ay=0,bx=0,by=0;
        if(he){ ax=m.nodes[fn[0]*2];ay=m.nodes[fn[0]*2+1];bx=m.nodes[fn[1]*2];by=m.nodes[fn[1]*2+1]; }
        double edx=bx-ax, edy=by-ay;   // edge vector (v-independent), hoisted out of the v-loop
        for(int v=0;v<nvar;++v){
            auto tf=[&](int ci,double beta,double D)->double{
                if(!hasint[(size_t)ci*nvar+v]) return W[(size_t)v*N+ci];
                double qmn=qmnc[(size_t)ci*nvar+v],qmx=qmxc[(size_t)ci*nvar+v],rng=qmx-qmn;
                const double* A=&acoef[((size_t)ci*nvar+v)*5]; double kk=beta*kbc[ci], th=0.0;
                auto sig=[&](double kP)->double{
                    if(GAUSS && !BSTAR_TANHRECON){ return std::tanh(kP+D); }   // point fallback (non-he boundary face); interior he-faces use the closed-form edge moments below. Option B skips this -> rational THINC sigmoid (Af+D)/(1+Af*D) below, MATCHING the tanh Newton cell-D
                    if(SPL){ return spl_sig(kP+D); }   // spliced tanh-hugging sigmoid
                    if(ASIG && ARAT){ double s=kP+D, as=std::fabs(s); double r=1.0-1.0/((1.0+as)*(1.0+as)); return s<0.0?-r:r; }  // rational sigmoid
                    if(ASIG){ double s=kP+D; return s/(1.0+std::fabs(s)); }   // algebraic sigmoid, asymptotic (high-beta stable)
                    if(PST){ double xi=kP+D, cs=1.0-PST_CAP; if(xi<=-1.0)return -cs; if(xi>=1.0)return cs;
                        double tau=0.5*(xi+1.0), S=tau*tau*(3.0-2.0*tau); return cs*(2.0*S-1.0); }  // capped cubic smoothstep: saturates to +-(1-eps)
                    if(RAMP){ double s=kP+D; return s<-1.0?-1.0:(s>1.0?1.0:s); }  // ramp: D=kd
                    double Af=std::tanh(kP); double den=1.0+Af*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12); return (Af+D)/den; };
                if(he && GAUSS && !BSTAR_TANHRECON){   // probit closed-form edge average: tanh((kk<P>_e+D)/sqrt(1+c v_e)); <P>_e,<P^2>_e are closed-form moments of P(t)=p2 t^2+p1 t+p0 over t in [0,1] (NO quadrature). Option B skips this -> tanh quadrature edge (he branch, sig()) below
                    double dx0=ax-cc[ci*2],dy0=ay-cc[ci*2+1],ex=bx-ax,ey=by-ay;
                    double p2=A[2]*ex*ex+A[3]*ey*ey+A[4]*ex*ey;
                    double p1=A[0]*ex+A[1]*ey+2.0*A[2]*dx0*ex+2.0*A[3]*dy0*ey+A[4]*(dx0*ey+dy0*ex);
                    double p0=A[0]*dx0+A[1]*dy0+A[2]*dx0*dx0+A[3]*dy0*dy0+A[4]*dx0*dy0;
                    double F1=p2/3.0+p1/2.0+p0;                                       // int_0^1 P dt
                    double F2=p2*p2/5.0+p1*p2/2.0+(p1*p1+2.0*p0*p2)/3.0+p0*p1+p0*p0;  // int_0^1 P^2 dt
                    double vv=kk*kk*(F2-F1*F1); if(vv<0)vv=0;
                    th=std::tanh((kk*F1+D)/std::sqrt(1.0+GC*vv)); }
                else if(he && ((PST&&PST_EXACT)||(ASIG&&ASIG_EXACT)||(SPL&&SPL_CF))){   // exact closed-form edge integral (xi(t)=kk*P(t)+D, P quadratic in t)
                    double dx0=ax-cc[ci*2],dy0=ay-cc[ci*2+1],ex=bx-ax,ey=by-ay;
                    double p2=A[2]*ex*ex+A[3]*ey*ey+A[4]*ex*ey;
                    double p1=A[0]*ex+A[1]*ey+2.0*A[2]*dx0*ex+2.0*A[3]*dy0*ey+A[4]*(dx0*ey+dy0*ex);
                    double p0=A[0]*dx0+A[1]*dy0+A[2]*dx0*dx0+A[3]*dy0*dy0+A[4]*dx0*dy0;
                    double Aa=kk*p2, Bb=kk*p1, Gg=kk*p0+D;
                    th = SPL ? (SPL_DEG3 ? deg3_edge_avg(Aa,Bb,Gg) : SPL_DEG3T ? deg3t_edge_avg(Aa,Bb,Gg) : SPL_POLY9 ? poly9_edge_avg(Aa,Bb,Gg) : SPL_PADEU ? pade32u_edge_avg(Aa,Bb,Gg) : SPL_PADE32 ? pade32_edge_avg(Aa,Bb,Gg) : spl_edge_avg(Aa,Bb,Gg))
                       : ASIG ? (ARAT ? arat_edge_avg(Aa,Bb,Gg) : asig_edge_avg(Aa,Bb,Gg))
                       : pst_edge_avg(Aa,Bb,Gg); }
                else if(he){ for(int q=0;q<NQE;++q){ double t=EQp[q][0],x=ax+t*(bx-ax),y=ay+t*(by-ay),dx=x-cc[ci*2],dy=y-cc[ci*2+1];
                    th+=EQp[q][1]*sig(kk*(A[0]*dx+A[1]*dy+A[2]*dx*dx+A[3]*dy*dy+A[4]*dx*dy)); } }
                else { double dx=m.face_centers[f*2]-cc[ci*2],dy=m.face_centers[f*2+1]-cc[ci*2+1];
                    th=sig(kk*(A[0]*dx+A[1]*dy+A[2]*dx*dx+A[3]*dy*dy+A[4]*dx*dy)); }
                double qf=qmn+0.5*rng*(1.0+th); return NOCLAMP?qf:(qf<qmn?qmn:(qf>qmx?qmx:qf)); };
            if(he && !BOPT3 && ((ASIG && (ASIG_EXACT||ASIG_QUAD)) || (SPL && SPL_CF))){   // AST/SPL fast lane: beta-INDEPENDENT edge coeffs ONCE per side, reused for both betas (closed-form face)
                auto coef=[&](int ci,double& p0,double& p1,double& p2){
                    const double* A=&acoef[((size_t)ci*nvar+v)*5];
                    double dx0=ax-cc[ci*2],dy0=ay-cc[ci*2+1];
                    p2=A[2]*edx*edx+A[3]*edy*edy+A[4]*edx*edy;
                    p1=A[0]*edx+A[1]*edy+2.0*A[2]*dx0*edx+2.0*A[3]*dy0*edy+A[4]*(dx0*edy+dy0*edx);
                    p0=A[0]*dx0+A[1]*dy0+A[2]*dx0*dx0+A[3]*dy0*dy0+A[4]*dx0*dy0; };
                auto fval=[&](int ci,double p0,double p1,double p2,double beta,double D)->double{
                    double qmn=qmnc[(size_t)ci*nvar+v],qmx=qmxc[(size_t)ci*nvar+v],rng=qmx-qmn;
                    double kk=beta*kbc[ci], th;
                    if(SPL) th=SPL_DEG3?deg3_edge_avg(kk*p2,kk*p1,kk*p0+D):SPL_DEG3T?deg3t_edge_avg(kk*p2,kk*p1,kk*p0+D):SPL_POLY9?poly9_edge_avg(kk*p2,kk*p1,kk*p0+D):SPL_PADEU?pade32u_edge_avg(kk*p2,kk*p1,kk*p0+D):SPL_PADE32?pade32_edge_avg(kk*p2,kk*p1,kk*p0+D):spl_edge_avg(kk*p2,kk*p1,kk*p0+D);   // deg3 / deg3t / poly9 / user-spec / pade32 / quintic CF edge integral
                    else if(ASIG_QUAD){                                  // branch-free 3-pt sigma quadrature (no sqrt/arctan); compiler unrolls fixed-3
                        double s0=kk*(p2*EQ3[0][0]*EQ3[0][0]+p1*EQ3[0][0]+p0)+D;
                        double s1=kk*(p2*EQ3[1][0]*EQ3[1][0]+p1*EQ3[1][0]+p0)+D;
                        double s2=kk*(p2*EQ3[2][0]*EQ3[2][0]+p1*EQ3[2][0]+p0)+D;
                        th=EQ3[0][1]*(s0/(1.0+std::fabs(s0)))+EQ3[1][1]*(s1/(1.0+std::fabs(s1)))+EQ3[2][1]*(s2/(1.0+std::fabs(s2))); }
                    else th=ARAT?arat_edge_avg(kk*p2,kk*p1,kk*p0+D)        // rational sigmoid, perturbative curvature (no sqrt/arctan)
                               :asig_edge_avg(kk*p2,kk*p1,kk*p0+D);        // exact closed-form (default; quadrature-free novelty)
                    double qf=qmn+0.5*rng*(1.0+th); return NOCLAMP?qf:(qf<qmn?qmn:(qf>qmx?qmx:qf)); };
                double bo=adaptb?beff[o]:beta_l, bn=(n>=0&&adaptb)?beff[n]:beta_l;
                double L1,L0=0.0;
                if(!hasint[(size_t)o*nvar+v]){ double wo=W[(size_t)v*N+o]; L1=wo; if(!single)L0=wo; }
                else { double cp0,cp1,cp2; coef(o,cp0,cp1,cp2);
                    L1=fval(o,cp0,cp1,cp2,bo,Dl[(size_t)o*nvar+v]);
                    if(!single)L0=fval(o,cp0,cp1,cp2,beta_s,Ds[(size_t)o*nvar+v]); }
                double R1=L1,R0=L0;
                if(n>=0){
                    if(!hasint[(size_t)n*nvar+v]){ double wn=W[(size_t)v*N+n]; R1=wn; if(!single)R0=wn; }
                    else { double cp0,cp1,cp2; coef(n,cp0,cp1,cp2);
                        R1=fval(n,cp0,cp1,cp2,bn,Dl[(size_t)n*nvar+v]);
                        if(!single)R0=fval(n,cp0,cp1,cp2,beta_s,Ds[(size_t)n*nvar+v]); }
                    jM[(size_t)f*nvar+v]=std::fabs(WLs[(size_t)v*Nf+f]-WRs[(size_t)v*Nf+f]);
                    j1[(size_t)f*nvar+v]=std::fabs(L1-R1);
                    if(!single) j0[(size_t)f*nvar+v]=std::fabs(L0-R0); }
                WL1[(size_t)f*nvar+v]=L1; WR1[(size_t)f*nvar+v]=R1; WL0[(size_t)f*nvar+v]=L0; WR0[(size_t)f*nvar+v]=R0;
                continue; }
            double bo=adaptb?beff[o]:beta_l, bn=(n>=0&&adaptb)?beff[n]:beta_l;   // adaptb = SHEARBETA||BOPT -> per-cell beta*
            // H1 (perf, result-preserving): compute ONLY the THINC candidates the pick actually consults.
            // BETASTAR pick: binc==1 cell -> {MUSCL, THINC(beta*)} (beta_l/beta_s dead); binc==0 cell ->
            // {MUSCL, beta_l, beta_s} (beta* dead). A candidate's face jump for face f is summed only by cells
            // o and n, so compute it iff o OR n consults it. 4WAY / non-BETASTAR keep the full set. No result change.
            bool o2b=(!BETASTAR||BSTAR_4WAY||binc[o]==0), n2b=(n>=0&&(!BETASTAR||BSTAR_4WAY||binc[n]==0));
            bool obs=(BOPT3&&(BSTAR_4WAY||binc[o]==1)), nbs=(n>=0&&BOPT3&&(BSTAR_4WAY||binc[n]==1));
            bool need2b=o2b||n2b, needBstar=obs||nbs;
            double L1=0.0,R1=0.0,L0=0.0,R0=0.0;
            if(need2b){ L1=tf(o,bo,Dl[(size_t)o*nvar+v]); R1=L1; if(!single){ L0=tf(o,beta_s,Ds[(size_t)o*nvar+v]); R0=L0; } }
            double Lst=0.0,Rst=0.0; if(needBstar){ Lst=tf(o,BSTAR_PERVAR?bstv[(size_t)o*nvar+v]:beff[o],Dstar[(size_t)o*nvar+v]); Rst=Lst; }   // 4th candidate THINC(beta*_v per-var when BSTAR_PERVAR)
            if(n>=0){
                jM[(size_t)f*nvar+v]=std::fabs(WLs[(size_t)v*Nf+f]-WRs[(size_t)v*Nf+f]);
                if(need2b){ R1=tf(n,bn,Dl[(size_t)n*nvar+v]); j1[(size_t)f*nvar+v]=std::fabs(L1-R1);
                    if(!single){ R0=tf(n,beta_s,Ds[(size_t)n*nvar+v]); j0[(size_t)f*nvar+v]=std::fabs(L0-R0); } }
                if(needBstar){ Rst=tf(n,BSTAR_PERVAR?bstv[(size_t)n*nvar+v]:beff[n],Dstar[(size_t)n*nvar+v]); jst[(size_t)f*nvar+v]=std::fabs(Lst-Rst); } }
            if(need2b){ WL1[(size_t)f*nvar+v]=L1; WR1[(size_t)f*nvar+v]=R1; WL0[(size_t)f*nvar+v]=L0; WR0[(size_t)f*nvar+v]=R0; }
            if(needBstar){ WLst[(size_t)f*nvar+v]=Lst; WRst[(size_t)f*nvar+v]=Rst; } }
    }
    cheng3_prof().face += prof_now()-_tg;   // pass2 = face values (candidate-count proportional)
    cheng3_prof().thinc += prof_now()-_t0; double _t1=prof_now();
    // Pass 3 (cell loop): per-variable min-TBV pick. Wave-appropriate PROXY (primitive):
    // DENSONLY -> THINC only on density (v=0, entropy/contact carrier); NOACOUSTIC -> no THINC
    // on pressure (v=nvar-1, acoustic). Tests whether sharpness lives in entropy not acoustic.
    static const bool W_DENS = std::getenv("THINCQQ_DENSONLY") != nullptr;
    static const bool W_NOAC = std::getenv("THINCQQ_NOACOUSTIC") != nullptr;
    static std::vector<char> pick; pick.assign((size_t)nvar*N,0);   // H2 persistent scratch
    #pragma omp parallel for
    for(int ci=0;ci<N;++ci) for(int v=0;v<nvar;++v){ double tM=0,t1=0,t0=0,tst=0,t4=0;
        for(int f:m.cell_faces[ci]){ if(m.face_neighbour[f]<0)continue;
            tM+=jM[(size_t)f*nvar+v]; t1+=j1[(size_t)f*nvar+v]; t0+=j0[(size_t)f*nvar+v]; if(BOPT3) tst+=jst[(size_t)f*nvar+v]; if(TMLPU4) t4+=jt4[(size_t)f*nvar+v]; }
        char p;
        if(BETASTAR){
            // PAPER 3: beta* interior -> candidate set {MUSCL, THINC(beta*)} (DROP beta_l,beta_s);
            //          beta* on boundary -> {MUSCL, THINC(beta_l), THINC(beta_s)} (= baseline).
            if(BSTAR_4WAY){ p=0; double best=tM; if(t1<best){best=t1;p=1;} if(!single && t0<best){best=t0;p=2;} if(BOPT3 && tst<best){best=tst;p=3;} } // {MUSCL, beta_l, beta_s, beta*} union
            else if(binc[ci]){ p=0; double best=tM; if(tst<best){best=tst;p=3;} }        // {MUSCL, beta*}
            else        { p=0; double best=tM; if(t1<best){best=t1;p=1;} if(!single && t0<best){best=t0;p=2;} } // {MUSCL, beta_l, beta_s}
        } else {
            p=0; double best=tM; if(t1<best){best=t1;p=1;} if(!single && t0<best){best=t0;p=2;}
            if(BOPT3 && binc[ci] && tst<best){best=tst;p=3;}   // 4th candidate THINC(beta*) has lowest TBV and beta* in [beta_s,beta_l]
            if(TMLPU4 && t4<best){best=t4;p=4;}   // 4th candidate T-MLP-u-L (one-sided MLP-u, idw p=2): lowest TBV -> shear/peak sharpening
        }
        if((W_DENS && v!=0) || (W_NOAC && v==nvar-1)) p=0;   // wave-appropriate proxy: force MUSCL
        pick[(size_t)ci*nvar+v]=p; }
    // BVD_CANDFLAG: export the DENSITY (v=0) per-cell candidate slot for the paper diagnostic
    // (0=MUSCL,1=beta_l,2=beta_s,3=beta*,4=TMLPU4). Overwrites each call -> LAST recon (final-time) wins.
    static const bool CANDFLAG = std::getenv("BVD_CANDFLAG")!=nullptr;
    if(CANDFLAG){ auto& cf=cfd::bvd_cand_flag(); cf.assign((size_t)N,-1); for(int ci=0;ci<N;++ci) cf[ci]=(signed char)pick[(size_t)ci*nvar+0];
        // beta* field (BETASTAR only): beff on interface cells, -1 elsewhere -> SCALARS bvd_bstar
        if(BETASTAR && (int)beff.size()==N){ auto& bsf=cfd::bvd_bstar_flag(); bsf.assign((size_t)N,-1.0);
            for(int ci=0;ci<N;++ci) if(hasint[(size_t)ci*nvar+0]) bsf[ci]=beff[ci]; } }
    // TBVDIAG (2026-07-06): THINCQQ_TBVDIAG=x0,x1,y0,y1 -> for cells in the box, per variable,
    // recompute tM/t1/t0 from jM/j1/j0 and report MUSCL/b_l/b_s pick% + mean margins
    // mean(t_bl - t_M), mean(t_bs - t_M). margin>0 => THINC higher TBV (loses); magnitude
    // = how decisively MUSCL wins (near-0 = tie, nudgeable; large = THINC genuinely can't
    // align on the oblique slip line). Writes "w" each call -> last (final-time) survives.
    static const char* TBVD = std::getenv("THINCQQ_TBVDIAG");
    if(TBVD){ double bx0,bx1,by0,by1;
        if(std::sscanf(TBVD,"%lf,%lf,%lf,%lf",&bx0,&bx1,&by0,&by1)==4){
            FILE* fd=std::fopen("/tmp/mbq/tbvdiag.txt","w");
            if(fd){ const char* vn[4]={"rho","u  ","v  ","p  "};
                // per-cell interface flag: max neighbour |d rho|/local-mean > 0.12 (true slip/shock cell)
                std::vector<char> iface((size_t)N,0);
                for(int ci=0;ci<N;++ci){ double rc=W[(size_t)0*N+ci],mj=0;
                    for(int f:m.cell_faces[ci]){ int nb=m.face_neighbour[f]; if(nb<0)continue;
                        double d=std::fabs(W[(size_t)0*N+nb]-rc)/(0.5*(std::fabs(W[(size_t)0*N+nb])+std::fabs(rc))+1e-30);
                        if(d>mj)mj=d; }
                    iface[ci]=(mj>0.12)?1:0; }
                for(int filt=0; filt<2; ++filt){   // filt 0 = all box cells, 1 = interface cells only
                    std::fprintf(fd, filt? "--- INTERFACE cells only (|d rho|/mean>0.12) ---\n" : "--- ALL box cells ---\n");
                    for(int v=0; v<nvar; ++v){ long nM=0,n1=0,n0=0,nc=0; double sm1=0,sm0=0,sml=0; long nif=0;
                        for(int ci=0;ci<N;++ci){ double x=cc[ci*2],y=cc[ci*2+1];
                            if(x<bx0||x>bx1||y<by0||y>by1) continue;
                            if(filt && !iface[ci]) continue;
                            double tM=0,t1=0,t0=0;
                            for(int f:m.cell_faces[ci]){ if(m.face_neighbour[f]<0)continue;
                                tM+=jM[(size_t)f*nvar+v]; t1+=j1[(size_t)f*nvar+v]; t0+=j0[(size_t)f*nvar+v]; }
                            char p=pick[(size_t)ci*nvar+v];
                            if(p==0)++nM; else if(p==1)++n1; else if(p==2)++n0;
                            sm1+=(t1-tM); sm0+=(t0-tM); sml+=(t1-t0); ++nc; }
                        if(nc>0) std::fprintf(fd,"%s cells=%ld MUSCL=%.0f%% bl=%.0f%% bs=%.0f%% | mean(t_bl-t_M)=%.3e mean(t_bs-t_M)=%.3e mean(t_bl-t_bs)=%.3e\n",
                            vn[v<4?v:0], nc, 100.0*nM/nc, 100.0*n1/nc, 100.0*n0/nc, sm1/nc, sm0/nc, sml/nc); } }
                std::fclose(fd); } } }
    static const bool PICK_DIAG = TMLPU4_DIAG || (std::getenv("BVD_PICK_DIAG") != nullptr);
    if(PICK_DIAG){ static int wc=0; if((wc++%200)==0){ long n0=0,n1=0,n2=0,n4=0;
        for(int ci=0;ci<N;++ci){ char p=pick[(size_t)ci*nvar+0]; if(p==0)n0++; else if(p==1)n1++; else if(p==2)n2++; else if(p==4)n4++; }
        std::fprintf(stderr,"BVD_TMLPU4_DIAG picks(density): MUSCL=%ld THINC_l=%ld THINC_s=%ld T-MLP-u-L(cand4)=%ld\n",n0,n1,n2,n4); } }
    static const bool WINDIAG=std::getenv("CHENG3_WINDIAG")!=nullptr;
    if(WINDIAG){ static int wc=0; if((wc++%200)==0){
        int nM=0,n1=0,n0=0; double bsT=0,bsMu=0; int cT=0,cMu=0,cTlt=0;  // cTlt = THINC cells with median-beta* < beta_l(1.6)
        for(int ci=0;ci<N;++ci){ if(!hasint[(size_t)ci*nvar+0])continue;
            char p=pick[(size_t)ci*nvar+0]; if(p==0)nM++; else if(p==1)n1++; else n0++;
            double qmn=qmnc[(size_t)ci*nvar+0],qmx=qmxc[(size_t)ci*nvar+0],rng=qmx-qmn; if(rng<1e-12)continue;
            const double* A=&acoef[((size_t)ci*nvar+0)*5]; double nx=A[0],ny=A[1];
            double Qi=2.0*(W[ci]-qmn)/rng-1.0; if(Qi>0.999)Qi=0.999; else if(Qi<-0.999)Qi=-0.999;
            double Di= SPL?0.5*std::log((1.0+Qi)/(1.0-Qi)):Qi/(1.0-std::fabs(Qi));
            // weighted MEDIAN of |beta_f*| = the beta TBV-min single-beta actually picks (robust, NOT mean)
            double bf[16],wf[16]; int nf=0;
            for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                double a=((cc[nb*2]-cc[ci*2])*nx+(cc[nb*2+1]-cc[ci*2+1])*ny)*kbc[ci]; if(std::fabs(a)<1e-6)continue;
                double Qnb=2.0*(W[nb]-qmn)/rng-1.0; if(Qnb>0.999)Qnb=0.999; else if(Qnb<-0.999)Qnb=-0.999;
                double Dnb= SPL?0.5*std::log((1.0+Qnb)/(1.0-Qnb)):Qnb/(1.0-std::fabs(Qnb));
                double b=(Dnb-Di)/a; if(b<0)b=-b;
                if(nf<16){bf[nf]=b; wf[nf]=(1.0-Qnb*Qnb)*std::fabs(a); nf++;} }
            if(nf==0)continue;
            for(int x=1;x<nf;++x){double bb=bf[x],ww=wf[x];int y=x-1;while(y>=0&&bf[y]>bb){bf[y+1]=bf[y];wf[y+1]=wf[y];--y;}bf[y+1]=bb;wf[y+1]=ww;}
            double wt=0;for(int x=0;x<nf;++x)wt+=wf[x];double half=0.5*wt,acc=0,bmed=bf[nf-1];
            for(int x=0;x<nf;++x){acc+=wf[x];if(acc>=half){bmed=bf[x];break;}}
            if(p>0){bsT+=bmed;cT++; if(bmed<beta_l)cTlt++;} else {bsMu+=bmed;cMu++;} }
        std::fprintf(stderr,"CHENG3_WINDIAG ifc picks: MUSCL=%d THINC_l(b1.6)=%d THINC_s(b0.8)=%d | MEDIAN beta*: THINC-cells=%.2f MUSCL-cells=%.2f | THINC-cells frac(median<1.6)=%.2f\n",
            nM,n1,n0,cT>0?bsT/cT:0.0,cMu>0?bsMu/cMu:0.0,cT>0?(double)cTlt/cT:0.0); } }
    if(BOPT){ static const bool DIAG=std::getenv("BOPT_DIAG")!=nullptr;
        if(DIAG){ static int c2=0; if((c2++%200)==0){ int nM=0,nT=0;
            for(int ci=0;ci<N;++ci){ if(!hasint[(size_t)ci*nvar+0])continue; if(pick[(size_t)ci*nvar+0]==0)nM++; else nT++; }
            std::fprintf(stderr,"BOPT_DIAG pick(density,interface cells): MUSCL=%d THINC=%d MUSCL-frac=%.2f\n",nM,nT,(double)nM/(nM+nT>0?nM+nT:1)); } } }
    if(BOPT3){ static const bool DIAG=std::getenv("BOPT_DIAG")!=nullptr;
        if(DIAG){ static int c3=0; if((c3++%200)==0){ int n0=0,n1=0,n2=0,n3=0,ninc=0;
            for(int ci=0;ci<N;++ci){ if(!hasint[(size_t)ci*nvar+0])continue; if(binc[ci])ninc++;
                char p=pick[(size_t)ci*nvar+0]; if(p==0)n0++; else if(p==1)n1++; else if(p==2)n2++; else n3++; }
            std::fprintf(stderr,"BOPT3_DIAG picks(density,ifc): MUSCL=%d THINC_l(1.6)=%d THINC_s(0.8)=%d THINC(beta*)=%d | beta*-included=%d\n",n0,n1,n2,n3,ninc); } } }
    static const bool PRED = std::getenv("CHENG3_PREDICT")!=nullptr;
    if(PRED){ auto& P=cheng3_predict();
        for(int ci=0;ci<N;++ci) for(int v=0;v<nvar && v<8;++v){ double tM=0,t1=0,t0=0,maxj=0; bool any=false;
            for(int f:m.cell_faces[ci]){ if(m.face_neighbour[f]<0)continue; any=true;
                double jm=jM[(size_t)f*nvar+v]; tM+=jm; if(jm>maxj)maxj=jm;
                t1+=j1[(size_t)f*nvar+v]; t0+=j0[(size_t)f*nvar+v]; }
            if(!any) continue; double rng=qmxc[(size_t)ci*nvar+v]-qmnc[(size_t)ci*nvar+v]; if(rng<1e-12||tM<1e-30) continue;
            double tMn=tM/rng, conc=maxj/tM, tT=twomem?t1:std::min(t1,t0); bool actual=(tT<tM);
            P.n++; if(actual){P.n_thinc++; P.sct+=conc;} else P.scm+=conc;
            for(int k=0;k<Cheng3Predict::NT;++k){ bool pt=(tMn>P.tmthr[k]); if(pt==actual)P.agree_tm[k]++;
                for(int j=0;j<Cheng3Predict::NC;++j){ bool p2=pt&&(conc>P.ccthr[j]); if(p2==actual)P.agree_2f[k][j]++; } }
            for(int j=0;j<Cheng3Predict::NC;++j){ if((conc>P.ccthr[j])==actual)P.agree_cc[j]++; } } }
    // Pass 4 (face loop): assemble owner/neighbour picks.
    W_L.assign((size_t)nvar*Nf,0.0); W_R.assign((size_t)nvar*Nf,0.0);
    #pragma omp parallel for
    for(int f=0;f<Nf;++f){ int o=m.face_owner[f],n=m.face_neighbour[f];
        for(int v=0;v<nvar;++v){ char po=pick[(size_t)o*nvar+v];
            W_L[(size_t)v*Nf+f] = po==4?WLt4[(size_t)v*Nf+f]:(po==3?WLst[(size_t)f*nvar+v]:(po==1?WL1[(size_t)f*nvar+v]:(po==2?WL0[(size_t)f*nvar+v]:WLs[(size_t)v*Nf+f])));
            if(n>=0){ char pn=pick[(size_t)n*nvar+v];
                W_R[(size_t)v*Nf+f] = pn==4?WRt4[(size_t)v*Nf+f]:(pn==3?WRst[(size_t)f*nvar+v]:(pn==1?WR1[(size_t)f*nvar+v]:(pn==2?WR0[(size_t)f*nvar+v]:WRs[(size_t)v*Nf+f]))); }
            else W_R[(size_t)v*Nf+f]=W_L[(size_t)v*Nf+f]; } }
    cheng3_prof().sel += prof_now()-_t1; cheng3_prof().ncall++;
}

// ===== ABVD adaptive linear candidate (unstructured TVD-BVD, Majima-Wakimura-Aoki-Xiao
// C&F 266 2023 Eq.21-28 extended to unstructured). Replaces BOTH linear candidates
// (diffusive MUSCL + compressive T-MLP-u) with ONE per-face zero-BV construction:
//   BV = |1/2*dq*(2-(phi_L+phi_R))|  =>  BV=0 iff phi_L+phi_R=2  (Eq.21-22).
//   A = phi_sb(rL)+phi_sb(rR) < 2  -> zero-BV unreachable in TVD -> superbee both sides
//       (BV>0 there, so the outer min-TBV hands discontinuities to THINC).
//   A >= 2 -> the two TVD-admissible face-value intervals [q_mm,q_sb] overlap; the shared
//       segment xi is found by the sign tests B,C (Table 1); pick q* = upwind endpoint
//       (Eq.26, face-normal velocity as the advection speed; primitive vars, NO
//       characteristic decomposition) and set qL=qR=q* -> face jump EXACTLY 0.
// Unstructured slope ratios via projected LSQ gradients (reduces to Eq.11-12 in 1D):
//   rL = 2*(grad_i . d_ij)/dq - 1,   rR = 2*(grad_j . d_ji)/(-dq) - 1.
// MLP fusion is BY CONSTRUCTION: the Delta-form q = qbar + 1/2*phi*dq with phi in [0,2]
// keeps every face value inside [qbar_i, qbar_j], a subset of the vertex-neighbourhood
// (MLP/LMP) admissible range -> the vertex-monotone bound is implied, NO vertex pass.
// Cost: one LSQ-gradient pass + one O(1) face loop (no Newton, no vertex loops)
// ~2-3x cheaper than the two full vertex-limited linear reconstructions it replaces.
// Env: BVD_ABVD=1 activates (in reconstruct_bvd, CHENG3 mode); BVD_ABVD_MID=1 -> midpoint
// of xi instead of the upwind endpoint (scalar advection always uses midpoint: the recon
// has no velocity field).
inline void reconstruct_abvd_linear(const Mesh& m, const ReconCtx& c,
                                    const std::vector<double>& W, int nvar,
                                    std::vector<double>& WL, std::vector<double>& WR,
                                    const double* face_a = nullptr) {
    // face_a: optional per-face advection speed (scalar solver's a.n) for the Eq.26 upwind
    // endpoint. CRITICAL: the midpoint fallback is NOT TVD (zero-dissipation central + nonlinear
    // switching amplifies, verified divergent on LeVeque); the upwind endpoint IS the stability
    // mechanism, not an optimization.
    const int N = m.n_cells(), Nf = m.n_faces();
    const double* cc = m.cell_centers.data();
    static const bool MID = std::getenv("BVD_ABVD_MID") != nullptr;
    // Unstructured-stabilization variants (the projected LSQ r is only an APPROXIMATION of the
    // directional ratio the 1D TVD proof needs; the most anti-diffusive branches are fragile):
    //  BVD_ABVD_SAFE : pattern (a) uses van Leer instead of superbee (multi-D-robust envelope).
    //  BVD_ABVD_EMM  : pattern (e) uses minmod both sides instead of the midpoint (the midpoint
    //                  forces phi=1 which may EXCEED the local TVD cap when intervals crossed).
    //  BVD_ABVD_PDIAG: count pattern occurrences (a/b/c/d/e) every 200 calls.
    static const bool SAFE  = std::getenv("BVD_ABVD_SAFE") != nullptr;
    static const bool EMM   = std::getenv("BVD_ABVD_EMM") != nullptr;
    static const bool PDIAG = std::getenv("BVD_ABVD_PDIAG") != nullptr;
    // MLP-LMP fusion cap (DEFAULT ON; BVD_ABVD_NOLMP for ablation). VERIFIED NECESSARY:
    // without it every variant diverges on LeVeque (L1 -> 1e63..1e119). Root cause: the
    // per-face zero-BV forcing abandons CELL-PROFILE COHERENCE -- a nascent extremum cell
    // receives independently max-compressed inflow from ALL faces (LED broken); the 1D TVD
    // proof that protects the structured scheme needs the exact directional r, which the
    // projected LSQ r only approximates. Fix = the T-MLP-u device: cap each side's increment
    // by the cell's own vertex-LMP-limited linear-profile reach at the face,
    //   delta = clamp(T - qbar, [min(0,dprof), max(0,dprof)]),  dprof = psi_LMP * grad.(xf-xc),
    // so zero-BV is achieved ONLY where both vertex-monotone profiles can reach the shared
    // value; at forming extrema psi_LMP -> 0 pins the face to the cell average (LED restored).
    static const bool LMP = std::getenv("BVD_ABVD_NOLMP") == nullptr;
    // BVD_ABVD_LMPFIRST (user 2026-07-03): fold the LMP reach into the zero-BV DECISION instead of
    // post-clamping each side independently. The zero-BV shared value qs is clamped to the
    // INTERSECTION of both sides' LMP reach boxes -> q^L=q^R PRESERVED (independent capv breaks it).
    // If the two clamped reaches do not overlap, zero-BV is unreachable under the clamp -> fall to
    // one-sided TVD (pattern-a). Self-consistent: decision made on the achievable (clamped) profile.
    static const bool LMPFIRST = std::getenv("BVD_ABVD_LMPFIRST") != nullptr;
    // BVD_ABVD_SGATE (user 2026-07-04): smoothness gate. The structured paper gets smooth-2nd-order
    // AUTOMATICALLY because its exact directional slope ratio r -> 1 on smooth flow (psi->1). On an
    // unstructured mesh the projected-LSQ r wobbles face-to-face on smooth curved profiles -> the
    // per-face zero-BV pattern pick becomes face-incoherent -> WAVY cone/hump. Recover the paper's
    // design goal by an unstructured-valid mechanism: where BOTH cells are smooth (vertex-MLP phi=1,
    // i.e. psi_LMP>=1 -> the linear reconstruction is within the vertex bounds, unclipped), emit the
    // COHERENT single-gradient limited-MUSCL (one gradient for all of the cell's faces -> smooth),
    // NOT the zero-BV shared value. ABVD's zero-BV compression is kept only where phi<1 (genuine
    // discontinuity / forming extremum). scalar-compatible (phi exists for scalars). env-tunable phi.
    static const bool SGATE = std::getenv("BVD_ABVD_SGATE") != nullptr;
    static const double SGATE_PHI = []{ const char* e=std::getenv("BVD_ABVD_SGATE_PHI"); return (e&&e[0])?std::atof(e):1.0; }();
    // BVD_ABVD_VA (user 2026-07-04): Van-Albada interval LOWER bound instead of minmod. The wobble
    // comes from the zero-BV interval [q_mm, q_sb] opening up and the pattern pick flipping when the
    // noisy unstructured r wobbles around 1 on smooth curved flow. Van-Albada Phi_va(r)=(r^2+r)/(r^2+1)
    // is a SMOOTH limiter that hugs superbee near r=1 -> the interval [q_va, q_sb] is NARROW on smooth
    // (robust to r noise, no wobble) yet still opens toward superbee at strong discontinuities (large r).
    // Phi_va in [phi_mm, phi_sb] for r>0, 0 for r<=0 (extremum). Recovers the structured r's smooth
    // property without any structured-only stencil / tuning. Keeps LMP clamp + upwind zero-BV pick.
    static const bool VA = std::getenv("BVD_ABVD_VA") != nullptr;
    const bool euler = (nvar >= 4) && !MID;   // primitive (rho,u,v,p): face-normal velocity known
    auto phi_mm = [](double r){ return std::max(0.0, std::min(r, 1.0)); };
    auto phi_sb = [](double r){ return std::max(0.0, std::max(std::min(2.0*r,1.0), std::min(r,2.0))); };
    auto phi_vl = [](double r){ return (r+std::fabs(r))/(1.0+std::fabs(r)); };
    auto phi_va = [](double r){ return r>0.0 ? (r*r+r)/(r*r+1.0) : 0.0; };   // Van-Albada (smooth minmod->superbee)
    auto phi_lo = [&](double r){ return VA ? phi_va(r) : phi_mm(r); };       // interval lower bound
    // BVD_ABVD_IDW (user 2026-07-04): inverse-distance-weighted LSQ gradient for the slope ratio.
    // Both deep-research agents flag the UNWEIGHTED cell-LSQ gradient (default idw_p=0) as the primary
    // face-to-face noise source that makes r wobble on irregular triangles (Mavriplis; Diskin-Thomas;
    // Darwish-Moukalled §2.3). Build a cached ctx with w_j=|x_j-x_i|^-p (p=BVD_ABVD_IDW, default 1) +
    // matching ATA_inv, and use it for the GRADIENT ONLY (vertex bounds / LMP stay on the passed ctx).
    static const bool IDW = std::getenv("BVD_ABVD_IDW") != nullptr;
    static const double IDWP = []{ const char* e=std::getenv("BVD_ABVD_IDW"); double p=(e&&e[0])?std::atof(e):0.0; return p>0.0?p:1.0; }();
    static const Mesh* abvd_idw_mesh = nullptr; static ReconCtx abvd_idw_ctx;
    if (IDW && abvd_idw_mesh != &m) { abvd_idw_ctx = build_recon_ctx(m, IDWP); abvd_idw_mesh = &m; }
    const ReconCtx& gc = IDW ? abvd_idw_ctx : c;   // gradient stencil/weights (IDW or passed ctx)
    static long pcnt[5] = {0,0,0,0,0};
    // Pass 0 (vertex loop, LMP only): vertex min/max of surrounding cell averages.
    std::vector<double> vmn, vmx;
    if (LMP) {
        vmn.assign((size_t)nvar*c.Nn, 0.0); vmx.assign((size_t)nvar*c.Nn, 0.0);
        #pragma omp parallel for
        for (int vn = 0; vn < c.Nn; ++vn) for (int v = 0; v < nvar; ++v) {
            double mn = 1e300, mx = -1e300;
            for (int k = 0; k < c.max_v2c; ++k) { int ci = c.v2c[(size_t)vn*c.max_v2c+k]; if (ci < 0) continue;
                double val = W[(size_t)v*N+ci]; if (val < mn) mn = val; if (val > mx) mx = val; }
            if (mn > mx) { mn = 0; mx = 0; }
            vmn[(size_t)v*c.Nn+vn] = mn; vmx[(size_t)v*c.Nn+vn] = mx;
        }
    }
    // Pass 1 (cell loop): LSQ gradients (all vars) + face-neighbour min/max (boundary BJ cap)
    // + per-cell vertex-LMP psi (one-sided, cap 2 -- the T-MLP-u room-to-vertex-bound).
    std::vector<double> grad((size_t)nvar*N*2);
    std::vector<double> qmn((size_t)nvar*N), qmx((size_t)nvar*N);
    std::vector<double> psil;
    if (LMP) psil.assign((size_t)nvar*N, 2.0);
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        for (int v = 0; v < nvar; ++v) {
            double qc = W[(size_t)v*N+ci], r0 = 0.0, r1 = 0.0, mn = qc, mx = qc;
            for (int k = 0; k < gc.max_nb; ++k) {   // gc = IDW-weighted ctx when BVD_ABVD_IDW, else the passed ctx
                int nb = gc.nb[(size_t)ci*gc.max_nb+k]; if (nb < 0) continue;
                double wk = gc.w[(size_t)ci*gc.max_nb+k];
                double dx = gc.d[((size_t)ci*gc.max_nb+k)*2+0], dy = gc.d[((size_t)ci*gc.max_nb+k)*2+1];
                double dq = W[(size_t)v*N+nb] - qc;
                r0 += wk*dx*dq; r1 += wk*dy*dq;
            }
            double gx = gc.ATA_inv[ci*4+0]*r0 + gc.ATA_inv[ci*4+1]*r1;
            double gy = gc.ATA_inv[ci*4+2]*r0 + gc.ATA_inv[ci*4+3]*r1;
            grad[((size_t)v*N+ci)*2+0] = gx; grad[((size_t)v*N+ci)*2+1] = gy;
            for (int f : m.cell_faces[ci]) {   // face-neighbour range (boundary-face BJ cap only)
                int o = m.face_owner[f], n2 = m.face_neighbour[f], nb = (o==ci)?n2:o; if (nb < 0) continue;
                double w2 = W[(size_t)v*N+nb]; if (w2 < mn) mn = w2; if (w2 > mx) mx = w2;
            }
            qmn[(size_t)v*N+ci] = mn; qmx[(size_t)v*N+ci] = mx;
            if (LMP) {   // vertex-LMP psi: min over sample vertices of room/|projection|
                // BVD_ABVD_A_VENK (autoresearch iter2, 2026-07-04): Venkatakrishnan-smooth psi. The hard
                // BJ ratio room/|proj| -> 0 at a SMOOTH extremum (cone apex / hump crest) -> the LMP box
                // collapses -> the increment is clipped to first order -> smooth bodies over-diffuse. Venk
                // with eps2=(K h)^3 keeps psi ~1 where the vertex oscillation is O(h^3)<eps2 (smooth), while
                // still limiting at a genuine discontinuity (slot edge). Caps at 1 (no >1 compression on the
                // venk path; the slot sharpness comes from the case-table band, not this reach).
                static const bool AVENK = std::getenv("BVD_ABVD_A_VENK") != nullptr;
                static const double AVENK_K = []{ const char* e=std::getenv("BVD_ABVD_A_VENK_K"); return (e&&e[0])?std::atof(e):3.0; }();
                double p = 2.0;
                if (AVENK) {
                    double eps2 = std::pow(AVENK_K*std::sqrt(2.0*m.cell_volumes[ci]), 3.0);
                    for (int k = 0; k < c.max_v; ++k) { int vn = c.sample_vid[(size_t)ci*c.max_v+k]; if (vn < 0) continue;
                        double dx = c.sample_off[((size_t)ci*c.max_v+k)*2+0], dy = c.sample_off[((size_t)ci*c.max_v+k)*2+1];
                        double proj = gx*dx + gy*dy;
                        double pk = venk_phi(proj, qc, vmn[(size_t)v*c.Nn+vn], vmx[(size_t)v*c.Nn+vn], eps2);
                        if (pk < p) p = pk; }
                } else {
                    // BVD_ABVD_DIRLMP (autoresearch iter4, 2026-07-04): DIRECTIONAL (anisotropic) vertex bound.
                    // The isotropic psi = min room/|proj| over ALL sample vertices collapses at a DISCONTINUITY
                    // EDGE (a cell straddling the slot/disk edge sees the across-edge vertex, room->0, psi->0)
                    // -> superbee clipped on EVERY face -> disk-edge over-diffusion (the dominant residual). The
                    // reconstruction only VARIES along the gradient direction; transverse vertices do not
                    // constrain it. Bound ONLY along the cell-gradient direction ghat (skip vertices with
                    // |off.ghat|/|off| < DIRLMP_COS) = dimension-splitting's per-axis 1D bound (Goodman-LeVeque
                    // admissible: a per-direction bound, not isotropic multi-D). Divergence-safe: still bounds
                    // the reconstruction along its own steepest direction.
                    static const bool DIRLMP = std::getenv("BVD_ABVD_DIRLMP") != nullptr;
                    static const double DIRLMP_COS = []{ const char* e=std::getenv("BVD_ABVD_DIRLMP_COS"); return (e&&e[0])?std::atof(e):0.5; }();
                    double gmag = std::sqrt(gx*gx+gy*gy);
                    double ghx = gmag>1e-30?gx/gmag:0.0, ghy = gmag>1e-30?gy/gmag:0.0;
                    for (int k = 0; k < c.max_v; ++k) { int vn = c.sample_vid[(size_t)ci*c.max_v+k]; if (vn < 0) continue;
                        double dx = c.sample_off[((size_t)ci*c.max_v+k)*2+0], dy = c.sample_off[((size_t)ci*c.max_v+k)*2+1];
                        if (DIRLMP && gmag>1e-30) { double L=std::sqrt(dx*dx+dy*dy);
                            if (L>1e-30 && std::fabs(dx*ghx+dy*ghy)/L < DIRLMP_COS) continue; }  // skip transverse vertices
                        double proj = gx*dx + gy*dy;
                        double allowed = proj >= 0.0 ? (vmx[(size_t)v*c.Nn+vn]-qc) : (qc-vmn[(size_t)v*c.Nn+vn]);
                        double pk = (std::fabs(proj) > 1e-30) ? std::max(allowed,0.0)/std::fabs(proj) : 2.0;
                        if (pk < p) p = pk; }
                }
                psil[(size_t)v*N+ci] = p < 0.0 ? 0.0 : (p > 2.0 ? 2.0 : p);
            }
        }
    }
    // Pass 2 (face loop): the TVD-BVD interval-intersection pick, O(1) per face per var.
    WL.assign((size_t)nvar*Nf, 0.0); WR.assign((size_t)nvar*Nf, 0.0);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int i = m.face_owner[f], j = m.face_neighbour[f];
        if (j < 0) {   // boundary: BJ-capped gradient projection (2nd order at walls, monotone)
            double fx = m.face_centers[f*2+0]-cc[i*2+0], fy = m.face_centers[f*2+1]-cc[i*2+1];
            for (int v = 0; v < nvar; ++v) {
                double qc = W[(size_t)v*N+i];
                double dlt = grad[((size_t)v*N+i)*2+0]*fx + grad[((size_t)v*N+i)*2+1]*fy;
                if (dlt > 0.0) dlt = std::min(dlt, qmx[(size_t)v*N+i]-qc);
                else           dlt = std::max(dlt, qmn[(size_t)v*N+i]-qc);
                WL[(size_t)v*Nf+f] = qc + dlt; WR[(size_t)v*Nf+f] = qc + dlt;
            }
            continue;
        }
        double dxc = cc[j*2+0]-cc[i*2+0], dyc = cc[j*2+1]-cc[i*2+1];
        double fxL = m.face_centers[f*2+0]-cc[i*2+0], fyL = m.face_centers[f*2+1]-cc[i*2+1];
        double fxR = m.face_centers[f*2+0]-cc[j*2+0], fyR = m.face_centers[f*2+1]-cc[j*2+1];
        double af = 0.0;   // face-normal advection speed (upwind endpoint pick)
        bool have_a = euler || (face_a != nullptr);
        if (face_a)    af = face_a[f];
        else if (euler) af = 0.5*((W[(size_t)1*N+i]+W[(size_t)1*N+j])*m.face_normals[f*2+0]
                          + (W[(size_t)2*N+i]+W[(size_t)2*N+j])*m.face_normals[f*2+1]);
        // SUPERSONIC-FACE GUARD (2026-07-03): the zero-BV shared state makes the face flux
        // CENTRAL; in a supersonic region central discretization carries parasitic modes with
        // UPSTREAM numerical group velocity (physically forbidden: all characteristics run one
        // way), which accumulate at inflow boundaries (verified: shock-vortex standing dipole,
        // vorticity 55x the interior level). Where |u.n| > c, keep the one-sided TVD values
        // (pattern-a treatment) so the upwind flux retains its dissipation. Opt-out BVD_ABVD_NOSS.
        static const bool SSON = std::getenv("BVD_ABVD_SS") != nullptr;   // guard OFF by default (user 2026-07-03); opt-in BVD_ABVD_SS
        bool ssonic = false;
        if (euler && SSON) {
            double rb = 0.5*(W[(size_t)0*N+i]+W[(size_t)0*N+j]);
            double pb = 0.5*(W[(size_t)(nvar-1)*N+i]+W[(size_t)(nvar-1)*N+j]);
            if (rb > 0.0 && pb > 0.0) { double cf = std::sqrt(1.4*pb/rb); if (std::fabs(af) > cf) ssonic = true; }
        }
        // LMP fusion: clamp each side's increment into the vertex-LMP-limited profile reach
        // [min(0,dprof), max(0,dprof)] -> zero-BV only where both monotone profiles reach T.
        // Phase-1 item3 (BVD_ABVD_MLP2): smooth Venkatakrishnan (MLP-u2) cap with eps2=(K h)^3 instead
        // of the hard box-clip -> relaxes the vertex bound at smooth extrema (cone apex/hump crest keep
        // amplitude). cap_eps2 is set per side by the NVD branch; 0 elsewhere -> hard clip unchanged.
        static const bool MLP2 = std::getenv("BVD_ABVD_MLP2") != nullptr;
        static const double KMLP = []{ const char* e=std::getenv("BVD_ABVD_MLP2_K"); return (e&&e[0])?std::atof(e):3.0; }();
        double hI = std::sqrt(2.0*m.cell_volumes[i]), hJ = std::sqrt(2.0*m.cell_volumes[j]);
        double eps2L = MLP2 ? std::pow(KMLP*hI, 3.0) : 0.0, eps2R = MLP2 ? std::pow(KMLP*hJ, 3.0) : 0.0;
        double cap_eps2 = 0.0;   // set by the NVD branch per side; 0 => hard clip
        auto capv = [&](double q0, double t, double dprof) -> double {
            if (!LMP) return t;
            double d = t - q0, lo = std::min(0.0,dprof), hi = std::max(0.0,dprof);
            if (MLP2 && cap_eps2 > 0.0) return q0 + venk_phi(d, 0.0, lo, hi, cap_eps2)*d;
            return q0 + (d < lo ? lo : (d > hi ? hi : d));
        };
        for (int v = 0; v < nvar; ++v) {
            double qi = W[(size_t)v*N+i], qj = W[(size_t)v*N+j], dq = qj - qi;
            if (std::fabs(dq) < 1e-13*(std::fabs(qi)+std::fabs(qj)+1e-100)) {
                WL[(size_t)v*Nf+f] = qi; WR[(size_t)v*Nf+f] = qj; continue; }
            // smoothness gate: both cells smooth (BJ phi = min(psi_LMP,1) = 1) -> coherent MUSCL,
            // skip the wavy face-incoherent zero-BV pick. Keeps ABVD only at phi<1 (discontinuity).
            static const bool SGATE_C = std::getenv("BVD_ABVD_SGATE_C") != nullptr;  // Phase-1 item2: continuous gate (NVD-only, smoothstep blend in the NVD branch instead of this hard switch)
            if (SGATE && !SGATE_C && LMP) {
                double pi = psil[(size_t)v*N+i], pj = psil[(size_t)v*N+j];
                if (pi >= SGATE_PHI && pj >= SGATE_PHI) {
                    double phL = pi < 1.0 ? pi : 1.0, phR = pj < 1.0 ? pj : 1.0;  // BJ phi in [0,1]
                    WL[(size_t)v*Nf+f] = qi + phL*(grad[((size_t)v*N+i)*2+0]*fxL + grad[((size_t)v*N+i)*2+1]*fyL);
                    WR[(size_t)v*Nf+f] = qj + phR*(grad[((size_t)v*N+j)*2+0]*fxR + grad[((size_t)v*N+j)*2+1]*fyR);
                    continue;
                }
            }
            double dpL = 0.0, dpR = 0.0;
            if (LMP) {
                dpL = psil[(size_t)v*N+i]*(grad[((size_t)v*N+i)*2+0]*fxL + grad[((size_t)v*N+i)*2+1]*fyL);
                dpR = psil[(size_t)v*N+j]*(grad[((size_t)v*N+j)*2+0]*fxR + grad[((size_t)v*N+j)*2+1]*fyR);
            }
            // ===== BVD_ABVD_A (user 2026-07-04, Architecture A): the ORIGINAL DISCRETE structured
            // case-table (phi_sb/phi_mm bands, A=sbL+sbR, B, C, patterns a-e, Eq.26 upwind pick) run
            // VERBATIM, fed by the NVD-DONOR-guarded slope ratio r, and emitted as a single shared
            // value with NO LMP clamp (the original structured scheme has no such clamp). The NVD
            // guard (phiC = 1 - dq/(2 grad.d); phiC outside (0,1) or flat gradient -> r=0 = upwind)
            // pins non-monotone/extremum faces to upwind BEFORE forming r, so r NEVER blows up through
            // dq->0 (the unstructured face-jump-denominator fragility), while on a Cartesian grid
            // phiC = the structured normalized variable and the guard = structured r<=0, so this
            // reduces EXACTLY to struct_leveque.py::recon_abvd. Boundedness mechanism = CBC-admissibility
            // (each band value in [qc,qd]) + the phiC guard, NOT a vertex-MLP cap.
            static const bool ARCH_A = std::getenv("BVD_ABVD_A") != nullptr;
            if (ARCH_A) {
                double epsn = 1e-12*(std::fabs(qi)+std::fabs(qj)+1e-30);
                double gLn =  grad[((size_t)v*N+i)*2+0]*dxc + grad[((size_t)v*N+i)*2+1]*dyc;   // ∇q_i·d
                double gRn = -(grad[((size_t)v*N+j)*2+0]*dxc + grad[((size_t)v*N+j)*2+1]*dyc); // ∇q_j·(-d)
                auto rNVD = [&](double qc, double qd, double g)->double{
                    double span = 2.0*g, dqs = qd - qc;
                    if (std::fabs(span) < epsn) return 0.0;             // flat gradient -> upwind
                    double phiC = 1.0 - dqs/span;                       // NVD donor (bounded through dq->0)
                    if (phiC <= 0.0 || phiC >= 1.0) return 0.0;         // extremum/nonmonotone -> upwind (= structured r<=0)
                    return phiC/(1.0 - phiC);                           // safe: phiC in (0,1) => r in (0,inf)
                };
                // BVD_ABVD_DIRR (autoresearch iter1, 2026-07-04): GRADIENT-FREE directional-stencil r.
                // The structured ASVL is gradient-free (r from aligned cell averages). Faithful unstructured
                // analog: for face C|D with unit direction dh, find the real UPWIND cell U of C (neighbour
                // most anti-parallel to dh) and DOWNWIND cell F of D (most parallel), and form r from REAL
                // cell-average directional slopes: sf=(q_D-q_C)/|d|, sb=(q_C-q_U)/proj_U, sff=(q_F-q_D)/proj_F,
                // r_L=sb/sf, r_R=sff/sf. EXACT structured on Cartesian (U=i-1, F=i+2); co-vanishes cleanly at
                // extrema (r<0 -> phi=0 naturally, NO NVD guard, NO LSQ-gradient noise). This attacks the ROOT
                // (gradient-based r) + removes the over-firing guard + decouples faces (per-face directional slope).
                static const bool DIRR = std::getenv("BVD_ABVD_DIRR") != nullptr;
                static const double DIRR_COS = []{ const char* e=std::getenv("BVD_ABVD_DIRR_COS"); return (e&&e[0])?std::atof(e):0.85; }();  // iter3: 0.85 (closest-collinear needs high cos)
                double rL, rR;
                if (DIRR) {
                    double dlen = std::sqrt(dxc*dxc + dyc*dyc);
                    double dhx = dxc/dlen, dhy = dyc/dlen, sf = (qj - qi)/dlen;
                    // iter3 (2026-07-04): among well-aligned (cos>=DIRR_COS) donors pick the one CLOSEST
                    // to the ideal mirror distance |d| (min |L-dlen|), not merely max-cos. On criss-cross the
                    // diagonal faces have a cos~1 donor at ~1.7|d| AND a closer one; the far donor is a WIDER
                    // stencil (more averaging -> more diffusive). Closest-collinear keeps s_b on the same O(h)
                    // baseline as s_f, matching the structured (q_i-q_{i-1}) tight difference.
                    auto find_aligned = [&](int cX, double dirx, double diry)->int{
                        double bestd = 1e300; int B = -1;
                        for (int k=0;k<gc.max_nb;++k){ int nb=gc.nb[(size_t)cX*gc.max_nb+k]; if(nb<0) continue;
                            double ox=cc[nb*2]-cc[cX*2], oy=cc[nb*2+1]-cc[cX*2+1];
                            double L=std::sqrt(ox*ox+oy*oy); if(L<1e-30) continue;
                            double al=(ox*dirx+oy*diry)/L;
                            if(al>=DIRR_COS){ double dd=std::fabs(L-dlen); if(dd<bestd){bestd=dd; B=nb;} } }
                        return B;
                    };
                    rL = 0.0; rR = 0.0;
                    if (std::fabs(sf) > 1e-30) {
                        int U = find_aligned(i, -dhx, -dhy);
                        if (U >= 0) { double qU=W[(size_t)v*N+U];
                            double gU=(cc[U*2]-cc[i*2])*dhx + (cc[U*2+1]-cc[i*2+1])*dhy;
                            if (gU < -1e-30) rL = ((qi - qU)/(-gU))/sf; }
                        int F = find_aligned(j, dhx, dhy);
                        if (F >= 0) { double qF=W[(size_t)v*N+F];
                            double gF=(cc[F*2]-cc[j*2])*dhx + (cc[F*2+1]-cc[j*2+1])*dhy;
                            if (gF > 1e-30) rR = ((qF - qj)/gF)/sf; }
                    }
                } else { rL = rNVD(qi, qj, gLn); rR = rNVD(qj, qi, gRn); }
                // BVD_ABVD_UPB (autoresearch iter7, user 2026-07-04): selectable UPPER (compressive) bound
                // of the band, replacing superbee. 0=superbee(default), 1=downwind(phi=2 const), 2=MSTACS
                // (min(superbee, 2r/Co) = superbee capped by the Hyper-C Courant cap, "less wrinkly").
                // The band is [minmod, upper]; a more aggressive upper -> sharper contact (test vs waviness).
                static const int ABUPB = []{ const char* e=std::getenv("BVD_ABVD_UPB"); return (e&&e[0])?std::atoi(e):0; }();
                static const double ABMS_CO = []{ const char* e=std::getenv("BVD_ABVD_MS_CO"); return (e&&e[0])?std::atof(e):0.4; }();
                auto phi_up = [&](double r)->double{
                    if (ABUPB==1) return r>0.0 ? 2.0 : 0.0;          // downwind (constant 2) - sharp but WIGGLY
                    if (ABUPB==2) return psi_mstacs(r, ABMS_CO);      // MSTACS (== superbee, cap never binds Co<=1)
                    if (ABUPB==3) return psi_cicsam(r, ABMS_CO);      // CICSAM min(2r(1-Co)/Co,2) - Courant-bounded, LESS wiggly
                    return phi_sb(r);                                 // superbee (default)
                };
                double sbL = phi_up(rL), sbR = phi_up(rR);
                double qLmm = qi + 0.5*phi_mm(rL)*dq, qLsb = qi + 0.5*sbL*dq;
                double qRmm = qj - 0.5*phi_mm(rR)*dq, qRsb = qj - 0.5*sbR*dq;
                // NO-CLAMP (BVD_ABVD_NOLMP): faithful to the original structured scheme, but on an
                // unstructured multi-D stencil this DIVERGES (Goodman-LeVeque: per-face zero-BV
                // max-compression from all of a cell's faces breaks LED; verified L1->1e93). With LMP
                // on, apply the cap to the SHARED value via the INTERSECTION of both cells' LMP reach
                // boxes (LMPFIRST) -> q^L=q^R is PRESERVED (single-valued novelty intact) AND the
                // multi-D max principle is restored. Empty intersection -> zero-BV unreachable under
                // the bound -> honest one-sided TVD fall (BV>0).
                double qLout, qRout;
                if (sbL + sbR < 2.0 || ssonic) {           // pattern (a): zero-BV unreachable -> one-sided
                    qLout = capv(qi, qi + 0.5*sbL*dq, dpL); // capv = no-op when !LMP (NOLMP path)
                    qRout = capv(qj, qj - 0.5*sbR*dq, dpR);
                } else {
                    double B = (qLsb - qRmm)*(qLsb - qRsb);
                    double C = (qRsb - qLmm)*(qRsb - qLsb);
                    double qs;
                    if (B >= 0.0 && C >= 0.0) {            // pattern (e): intervals crossed
                        // BVD_ABVD_ECOMP (iter7): at a SHARP jump (r>=2) both superbee/downwind cross ->
                        // the Majima default is the CENTRAL midpoint (NO compression) = why the disk EDGE
                        // loses to MUSCL. ECOMP instead picks the upwind-COMPRESSIVE endpoint of the crossed
                        // overlap -> genuine compression at the sharpest edges (bounded by the LMP clamp).
                        static const bool ECOMP = std::getenv("BVD_ABVD_ECOMP") != nullptr;
                        if (ECOMP && have_a) {
                            double loA=std::min(qLmm,qLsb), hiA=std::max(qLmm,qLsb);
                            double loB=std::min(qRmm,qRsb), hiB=std::max(qRmm,qRsb);
                            double lo=std::max(loA,loB), hi=std::min(hiA,hiB);
                            qs = (lo<=hi) ? ((af*(qi-qj)<0.0)?hi:lo) : 0.5*(qi+qj);
                        } else qs = 0.5*(qi + qj);
                    } else {
                        double e0, e1;
                        if      (B <= 0.0 && C <= 0.0) { e0 = qLsb; e1 = qRsb; }   // (b)
                        else if (B <= 0.0)             { e0 = qLmm; e1 = qLsb; }   // (c)
                        else                           { e0 = qRmm; e1 = qRsb; }   // (d)
                        double xmin = std::min(e0,e1), xmax = std::max(e0,e1);
                        qs = have_a ? ((af*(qi - qj) < 0.0) ? xmax : xmin) : 0.5*(xmin+xmax);  // Eq.26 upwind endpoint
                    }
                    // BVD_ABVD_A_DISS (autoresearch iter5, 2026-07-04): dispersion damping. The per-face
                    // zero-BV pick is face-incoherent on a CURVED discontinuity (disk rim) -> the shared
                    // central-ish flux is DISPERSIVE -> wavy contours (visible in the field). Blend the
                    // shared value a fraction eps toward the UPWIND cell average (still single-valued ->
                    // zero-BV preserved) to inject a little upwind dissipation that damps the wiggles.
                    static const bool ADISS = std::getenv("BVD_ABVD_A_DISS") != nullptr;
                    static const double ADISS_E = []{ const char* e=std::getenv("BVD_ABVD_A_DISS"); double x=(e&&e[0])?std::atof(e):0.0; return x; }();
                    if (ADISS && have_a && ADISS_E > 0.0) {
                        double qup = (af >= 0.0) ? qi : qj;   // upwind cell average
                        qs = (1.0 - ADISS_E)*qs + ADISS_E*qup;
                    }
                    if (LMP && LMPFIRST) {   // LMPFIRST: clamp SHARED value to INTERSECTION -> zero-BV preserved
                        double loL=qi+std::min(0.0,dpL), hiL=qi+std::max(0.0,dpL);
                        double loR=qj+std::min(0.0,dpR), hiR=qj+std::max(0.0,dpR);
                        double lo=std::max(loL,loR), hi=std::min(hiL,hiR);
                        if (lo <= hi) { double q = qs<lo?lo:(qs>hi?hi:qs); qLout = qRout = q; }
                        else { qLout = capv(qi, qi+0.5*sbL*dq, dpL); qRout = capv(qj, qj-0.5*sbR*dq, dpR); }
                    } else if (LMP) {        // per-side capv -> sharper but BREAKS zero-BV (q^L != q^R at clamped faces)
                        qLout = capv(qi, qs, dpL); qRout = capv(qj, qs, dpR);
                    } else qLout = qRout = qs;             // NO clamp (diverges on unstructured)
                    // BVD_ABVD_OUTCLAMP (iter10): outer discrete-max-principle safety net. The relaxed
                    // (DIRLMP) clamp lets the aggressive downwind band OVERSHOOT at curved fronts (disk rim
                    // -> range beyond [0,1]). Clamp the SHARED value to the face-neighbour cell-average
                    // envelope [min qmn, max qmx] (both cells): catches gross overshoot WITHOUT the
                    // transverse-vertex over-diffusion, and keeps q^L=q^R (zero-BV). Full compression to a
                    // neighbour value stays admissible (it is inside the envelope).
                    static const bool OUTCLAMP = std::getenv("BVD_ABVD_OUTCLAMP") != nullptr;
                    if (OUTCLAMP) {
                        double olo = std::min(qmn[(size_t)v*N+i], qmn[(size_t)v*N+j]);
                        double ohi = std::max(qmx[(size_t)v*N+i], qmx[(size_t)v*N+j]);
                        qLout = qLout<olo?olo:(qLout>ohi?ohi:qLout);
                        qRout = qRout<olo?olo:(qRout>ohi?ohi:qRout);
                    }
                }
                WL[(size_t)v*Nf+f] = qLout; WR[(size_t)v*Nf+f] = qRout;
                continue;
            }
            // ===== BVD_ABVD_NVD (user 2026-07-04, faithful recondition): NVD normalized-variable
            // zero-BV. Normalize by the GRADIENT PROJECTION (span=2 ∇q·d), NOT the face jump -> phiC is
            // BOUNDED and CONTINUOUS through dq->0 (no r=2∇q·d/Δq blow-up). Per side build the CBC-TVD
            // admissible face interval [central, superbee]; zero-BV = CONTINUOUS overlap-midpoint (no
            // discontinuous pattern switch). Keeps Φ^L+Φ^R=2 spirit (shared face value => BV=0). =====
            static const bool NVD = std::getenv("BVD_ABVD_NVD") != nullptr;
            if (NVD) {
                // compression knobs (all keep q_sb <= acceptor => overshoot-free, LMP capv is the closure):
                //   COMP_BETA = superbee-bound coefficient (2=classic superbee; higher saturates to the
                //     acceptor sooner => sharper, still bounded). COMP_LO/HI = smoothstep window on psi that
                //     gates the compressive-vs-central target (lower LO fires compression on more faces).
                static const double CBETA = []{ const char* e=std::getenv("BVD_ABVD_COMP_BETA"); return (e&&e[0])?std::atof(e):2.0; }();
                static const double CLO = []{ const char* e=std::getenv("BVD_ABVD_COMP_LO"); return (e&&e[0])?std::atof(e):0.8; }();
                static const double CHI = []{ const char* e=std::getenv("BVD_ABVD_COMP_HI"); return (e&&e[0])?std::atof(e):1.0; }();
                double epsn = 1e-12*(std::fabs(qi)+std::fabs(qj)+1e-30);
                double gLn = grad[((size_t)v*N+i)*2+0]*dxc + grad[((size_t)v*N+i)*2+1]*dyc;   // ∇q_i·d (owner donor->acceptor)
                double gRn = -(grad[((size_t)v*N+j)*2+0]*dxc + grad[((size_t)v*N+j)*2+1]*dyc); // ∇q_j·(-d) (neighbour donor->acceptor)
                // per side: TVD-admissible interval [upwind, superbee(CBETA)] from the NVD donor variable.
                auto nvd_int = [&](double qc, double qd, double g, double& lo, double& hi){
                    double span = 2.0*g;
                    if (std::fabs(span) < epsn) { lo=hi=qc; return; }               // flat gradient -> upwind
                    double phiC = 1.0 - (qd - qc)/span;                             // NVD donor (bounded through (qd-qc)->0)
                    if (phiC <= 0.0 || phiC >= 1.0) { lo=hi=qc; return; }           // outside CBC -> upwind (extremum/nonmonotone)
                    double qU = qd - span;
                    double q_sb = qU + std::min(CBETA*phiC,1.0)*span;               // compressive upper (<= acceptor)
                    lo = std::min(qc,q_sb); hi = std::max(qc,q_sb);                 // TVD interval: upwind(qc) .. compressive
                };
                double loL,hiL, loR,hiR;
                nvd_int(qi, qj, gLn, loL, hiL);
                nvd_int(qj, qi, gRn, loR, hiR);
                // DESIRED value per side = the cell's own gradient-limited MUSCL face value (accurate 2nd
                // order on smooth). zero-BV target = average of the two -> on smooth both ~equal so target is
                // the smooth face value (NO over-compression); at a discontinuity target=midpoint. Clamp each
                // side to its TVD interval -> minimal-BV, continuous, CBC-bounded.
                double qceL = qi + (grad[((size_t)v*N+i)*2+0]*fxL + grad[((size_t)v*N+i)*2+1]*fyL);   // P1 gradient face value (MUSCL = P1 only, user 2026-07-04: no P2+)
                double qceR = qj + (grad[((size_t)v*N+j)*2+0]*fxR + grad[((size_t)v*N+j)*2+1]*fyR);
                // Phase-1 item 1 (BVD_ABVD_UPW): upwind-biased zero-BV target instead of the symmetric
                // average. The average pulls the shared contact state toward the DOWNWIND cell (dispersive
                // on moving contacts; Majima Eq.26 picks the UPWIND endpoint). w = smooth-sign of the
                // face-normal speed af (continuous through af->0 -> central), weight on the owner (upwind if
                // af>0). af floor uses the local sound speed for Euler, else an absolute floor.
                static const bool UPW = std::getenv("BVD_ABVD_UPW") != nullptr;
                static const double UPW_A0 = []{ const char* e=std::getenv("BVD_ABVD_UPW_A0"); return (e&&e[0])?std::atof(e):0.05; }();
                double target;
                if (UPW && have_a) {
                    double a0 = UPW_A0;
                    if (euler) { double rb=0.5*(W[(size_t)0*N+i]+W[(size_t)0*N+j]), pb=0.5*(W[(size_t)(nvar-1)*N+i]+W[(size_t)(nvar-1)*N+j]);
                                 if (rb>0.0&&pb>0.0) a0 = UPW_A0*std::sqrt(1.4*pb/rb); }
                    double w = 0.5*(1.0 + af/(std::fabs(af)+a0));   // owner-weight, smooth sign
                    target = w*qceL + (1.0-w)*qceR;
                } else target = 0.5*(qceL + qceR);
                // COMPRESSION (BVD_ABVD_COMP): at a genuine (non-smooth) jump pull the zero-BV target toward
                // the COMPRESSIVE overlap endpoint (Majima Eq.26 upwind pick) instead of the gentle central
                // average -> activates ABVD's superbee-level slot compression (the paper's actual power).
                // superbee bound (q_sb<=acceptor) => overshoot-free; smoothstep(psi) keeps smooth extrema
                // on the central value; LMP capv still the closure.
                static const bool COMP = std::getenv("BVD_ABVD_COMP") != nullptr;
                if (COMP && LMP) {
                    double lo_ov = std::max(loL,loR), hi_ov = std::min(hiL,hiR);
                    double qcomp = (lo_ov<=hi_ov) ? ((have_a && af*(qi-qj)<0.0) ? hi_ov : lo_ov) : 0.5*(lo_ov+hi_ov);
                    double ws = smoothstep(CLO, CHI, std::min(psil[(size_t)v*N+i], psil[(size_t)v*N+j]));
                    target = (1.0-ws)*qcomp + ws*target;   // discontinuity -> compressive, smooth -> central
                }
                double qLf = std::min(std::max(target, loL), hiL);
                double qRf = std::min(std::max(target, loR), hiR);
                // Phase-1 item2 (BVD_ABVD_SGATE_C): CONTINUOUS smoothness gate. Blend the zero-BV values
                // toward the coherent single-gradient limited MUSCL by w=smoothstep(0.8,1.0,min(psi_i,psi_j))
                // (w=1 fully smooth -> coherent MUSCL with proper upwind dissipation; w=0 discontinuity ->
                // zero-BV compression). Replaces the hard psi>=0.9 binary switch -> no C0 gate seam.
                if (SGATE_C && LMP) {
                    double pi = psil[(size_t)v*N+i], pj = psil[(size_t)v*N+j];
                    double w = smoothstep(0.8, 1.0, std::min(pi, pj));
                    if (w > 0.0) {
                        double phL = pi<1.0?pi:1.0, phR = pj<1.0?pj:1.0;
                        double qL_ms = qi + phL*(grad[((size_t)v*N+i)*2+0]*fxL + grad[((size_t)v*N+i)*2+1]*fyL);
                        double qR_ms = qj + phR*(grad[((size_t)v*N+j)*2+0]*fxR + grad[((size_t)v*N+j)*2+1]*fyR);
                        qLf = (1.0-w)*qLf + w*qL_ms;
                        qRf = (1.0-w)*qRf + w*qR_ms;
                    }
                }
                cap_eps2 = eps2L; WL[(size_t)v*Nf+f] = capv(qi, qLf, dpL);
                cap_eps2 = eps2R; WR[(size_t)v*Nf+f] = capv(qj, qRf, dpR);
                continue;
            }
            // projected slope ratios (exactly Eq.11-12 on a uniform 1D line)
            double gLd = grad[((size_t)v*N+i)*2+0]*dxc + grad[((size_t)v*N+i)*2+1]*dyc;
            double gRd = -(grad[((size_t)v*N+j)*2+0]*dxc + grad[((size_t)v*N+j)*2+1]*dyc);
            double rL = 2.0*gLd/dq - 1.0, rR = 2.0*gRd/(-dq) - 1.0;
            double sbL = phi_sb(rL), sbR = phi_sb(rR);
            // emit a zero-BV SHARED value qs. LMPFIRST: clamp qs to the INTERSECTION of both LMP
            // reach boxes (q^L=q^R preserved); empty intersection -> zero-BV unreachable -> one-sided.
            // Default: independent per-side capv (breaks q^L=q^R at clamped cells).
            auto emitZB = [&](double qs){
                if (LMPFIRST && LMP) {
                    double loL=qi+std::min(0.0,dpL), hiL=qi+std::max(0.0,dpL);
                    double loR=qj+std::min(0.0,dpR), hiR=qj+std::max(0.0,dpR);
                    double lo=std::max(loL,loR), hi=std::min(hiL,hiR);
                    if (lo <= hi) { double q = qs<lo?lo:(qs>hi?hi:qs);
                        WL[(size_t)v*Nf+f]=q; WR[(size_t)v*Nf+f]=q; return; }
                    double pL=SAFE?phi_vl(rL):sbL, pR=SAFE?phi_vl(rR):sbR;  // unreachable under clamp -> one-sided TVD
                    WL[(size_t)v*Nf+f]=capv(qi, qi+0.5*pL*dq, dpL);
                    WR[(size_t)v*Nf+f]=capv(qj, qj-0.5*pR*dq, dpR); return;
                }
                WL[(size_t)v*Nf+f]=capv(qi, qs, dpL); WR[(size_t)v*Nf+f]=capv(qj, qs, dpR);
            };
            if (sbL + sbR < 2.0 || ssonic) {   // pattern (a): zero-BV unreachable (or supersonic guard)
                double pL = SAFE ? phi_vl(rL) : sbL, pR = SAFE ? phi_vl(rR) : sbR;
                WL[(size_t)v*Nf+f] = capv(qi, qi + 0.5*pL*dq, dpL);
                WR[(size_t)v*Nf+f] = capv(qj, qj - 0.5*pR*dq, dpR);
                if (PDIAG) pcnt[0]++;   // racy count, diagnostic only
                continue;
            }
            double qLmm = qi + 0.5*phi_lo(rL)*dq, qLsb = qi + 0.5*sbL*dq;   // qLmm = interval lower bound (minmod, or Van-Albada if VA)
            double qRmm = qj - 0.5*phi_lo(rR)*dq, qRsb = qj - 0.5*sbR*dq;
            double B = (qLsb - qRmm)*(qLsb - qRsb);
            double C = (qRsb - qLmm)*(qRsb - qLsb);
            if (B >= 0.0 && C >= 0.0) {                               // pattern (e): intervals crossed
                if (PDIAG) pcnt[4]++;
                if (EMM) {   // minmod both sides (NOT a shared zero-BV value -> keep independent capv)
                    WL[(size_t)v*Nf+f] = capv(qi, qLmm, dpL); WR[(size_t)v*Nf+f] = capv(qj, qRmm, dpR);
                } else {
                    emitZB(0.5*(qi + qj));
                }
                continue;
            }
            double e0, e1; int pat;                                   // xi endpoints (Table 1)
            if      (B <= 0.0 && C <= 0.0) { e0 = qLsb; e1 = qRsb; pat=1; }   // (b)
            else if (B <= 0.0)             { e0 = qLmm; e1 = qLsb; pat=2; }   // (c)
            else                           { e0 = qRmm; e1 = qRsb; pat=3; }   // (d)
            if (PDIAG) pcnt[pat]++;
            double xmin = std::min(e0,e1), xmax = std::max(e0,e1), qs;
            if (have_a) qs = (af*(qi - qj) < 0.0) ? xmax : xmin;      // Eq.26 upwind endpoint
            else        qs = 0.5*(xmin + xmax);                       // no velocity info: midpoint (NOT TVD - avoid)
            emitZB(qs);                                               // Eq.28 jump=0, LMP-consistent
        }
    }
    if (PDIAG) { static int wc=0; if((wc++%200)==0)
        std::fprintf(stderr,"ABVD patterns a=%ld b=%ld c=%ld d=%ld e=%ld\n",pcnt[0],pcnt[1],pcnt[2],pcnt[3],pcnt[4]); }
}

// W is nvar*N. sel_var = variable used for the TBV score (0 = density/scalar).
inline void reconstruct_bvd(const Mesh& m, const ReconCtx& ctx_smooth,
                            const ReconCtxO2& ctx_sharp,
                            const std::vector<double>& W, int nvar,
                            std::vector<double>& W_L, std::vector<double>& W_R,
                            bool face_bound = false, int sel_var = 0,
                            const double* shear = nullptr, double krelax = 0.0,
                            double tvbM = 0.0, double venkatK = 0.0, bool hier = false,
                            const double* face_a = nullptr) {
    const int N = m.n_cells(), Nf = m.n_faces();
    // smooth candidate (all vars) and sharp candidate (per var -> assemble).
    std::vector<double> WLs, WRs;
    double _tm=prof_now();
    // BVD_ABVD: replace the vertex-limited MUSCL smooth candidate with the adaptive
    // TVD-BVD linear (per-face zero-BV interval pick; subsumes diffusive+compressive).
    static const bool ABVD = std::getenv("BVD_ABVD") != nullptr;
    // face_bound clamp = the EXTRA a-priori clamp of the (BJ) MUSCL smooth candidate to the owner's
    // face-neighbour min/max. Cheng 2021 has NO such second clamp on the smooth member.
    // NEW DEFAULT (2026-07-07, user): face_bound clamp OFF (NOFB=true). Opt-in BVD_FACEBOUND=1
    // restores it. (Legacy BVD_NOFACEBOUND still recognized as OFF = now the default.) Only affects
    // the BJ MUSCL path (S1_MUSCL_BJ=1); the default van_leer T-MLP-u member never used face_bound.
    static const bool NOFB = std::getenv("BVD_FACEBOUND") == nullptr;
    // MUSCL smooth member. DEFAULT = reconstruct_bj_vertex (BJ, or MLP-u2 when MLP_U2 is set — the
    // DMR bench sets MLP_U2=0.001 to match Cheng 2021 / dln_paper). face_bound clamp default OFF (NOFB).
    // Opt-in S1_MUSCL_VL=1 -> van_leer-only T-MLP-u (psi in [0,2], compressive).
    static const bool MUSCL_VL = std::getenv("S1_MUSCL_VL") != nullptr;
    if (ABVD)          reconstruct_abvd_linear(m, ctx_smooth, W, nvar, WLs, WRs, face_a);
    else if (MUSCL_VL) reconstruct_tmlpu_gated(m, ctx_smooth, W, nvar, WLs, WRs, 1e30, 0.5, 2.0);
    else               reconstruct_bj_vertex(m, ctx_smooth, W, nvar, WLs, WRs, face_bound && !NOFB);
    cheng3_prof().muscl += prof_now()-_tm;
    // BVD_ABVD_ONLY (user 2026-07-04): PURE MUSCL(ABVD) — NO THINC/QQ, NO BVD selection. Isolates
    // the unstructured-ABVD reconstruction so it can be improved on its own (paper1's core goal =
    // structured ABVD -> unstructured). The ABVD linear reconstruction IS the final face state.
    static const bool ABVD_ONLY = std::getenv("BVD_ABVD_ONLY") != nullptr;
    if (ABVD_ONLY) { W_L = WLs; W_R = WRs; return; }   // ABVD set -> pure ABVD; unset -> pure BJ-MUSCL (reference)
    // ===== EXACT Cheng et al. 2021 three-member MUSCL-THINC/QQ-BVD =====
    // candidate union {MUSCL (WLs, MLP-u2), THINC/QQ(beta_l=1.4), THINC/QQ(beta_s=0.8)}.
    // PER-VARIABLE min-TBV selection (Deng 2018 [38]: TBV is per single primitive variable;
    // selection is per cell per variable). beta_s=0.8 captures shear/vortices the large-beta
    // THINC misses (where MUSCL would otherwise be chosen -> dissipative slip-line).
    static const bool CHENG3 = std::getenv("BVD_CHENG3") != nullptr;
    if (CHENG3) {
        static const char* BLs = std::getenv("BVD_BETA_L"); double bl = BLs?std::atof(BLs):1.4;  // Cheng2021 §4.2.4 value (default changed 1.6->1.4, user 2026-07-07)
        static const char* BSs = std::getenv("BVD_BETA_S"); double bs = BSs?std::atof(BSs):0.8;
        reconstruct_cheng3(m, ctx_sharp, W, nvar, WLs, WRs, W_L, W_R, bl, bs);  // fused, loop-separated
        // PP-floor (positivity-preserving, a-priori): clamp face rho (var 0) & p (var nvar-1)
        // to a fraction of the OWNING cell average so THINC over-sharpening cannot drive them
        // to vacuum. cell-avg>0 => floored faces>0 => HLLC/HLL+CFL keeps the cell-avg positive
        // (Zhang-Shu). Env BVD_PPFLOOR=<frac> (e.g. 0.2); unset/0 = off. Fixes config3/implosion/
        // noh/einfeldt123/sedov divergence (THINC->vacuum) without MOOD's recompute cost.
        static const double ppf = std::getenv("BVD_PPFLOOR") ? std::atof(std::getenv("BVD_PPFLOOR")) : 0.0;
        if (ppf > 0.0 && nvar >= 4) {
            const int pv = nvar-1;
            #pragma omp parallel for
            for (int f=0; f<Nf; ++f) {
                int o=m.face_owner[f], nb=m.face_neighbour[f];
                double rL=ppf*W[(size_t)0*N+o], pL=ppf*W[(size_t)pv*N+o];
                if (W_L[(size_t)0*Nf+f]  < rL) W_L[(size_t)0*Nf+f]  = rL;
                if (W_L[(size_t)pv*Nf+f] < pL) W_L[(size_t)pv*Nf+f] = pL;
                if (nb>=0) { double rR=ppf*W[(size_t)0*N+nb], pR=ppf*W[(size_t)pv*N+nb];
                    if (W_R[(size_t)0*Nf+f]  < rR) W_R[(size_t)0*Nf+f]  = rR;
                    if (W_R[(size_t)pv*Nf+f] < pR) W_R[(size_t)pv*Nf+f] = pR; }
            }
        }
        // Two-sided DMP clamp: bound each face value to the owner cell's face-neighbour
        // min/max (strict LMP). Stops THINC over-sharpening from creating NEW extrema (overshoot
        // jet OR vacuum) -> kills the runaway that the lower-only PP-floor can't. The sharp THINC
        // transition is kept WITHIN the physical [min,max] data range. Env BVD_DMPCLAMP=1.
        static const bool dmpc = std::getenv("BVD_DMPCLAMP") != nullptr;
        if (dmpc) {
            std::vector<double> lo((size_t)nvar*N), hi((size_t)nvar*N);
            #pragma omp parallel for
            for (int c=0;c<N;++c) for (int v=0;v<nvar;++v){ double mn=W[(size_t)v*N+c],mx=mn;
                for(int f2:m.cell_faces[c]){ int o2=m.face_owner[f2],n2=m.face_neighbour[f2],nb=(o2==c)?n2:o2; if(nb<0)continue;
                    double w=W[(size_t)v*N+nb]; if(w<mn)mn=w; if(w>mx)mx=w; }
                lo[(size_t)v*N+c]=mn; hi[(size_t)v*N+c]=mx; }
            #pragma omp parallel for
            for (int f=0;f<Nf;++f){ int o=m.face_owner[f], nb=m.face_neighbour[f];
                for(int v=0;v<nvar;++v){ double a=W_L[(size_t)v*Nf+f], L=lo[(size_t)v*N+o], H=hi[(size_t)v*N+o];
                    W_L[(size_t)v*Nf+f] = a<L?L:(a>H?H:a);
                    if(nb>=0){ double b=W_R[(size_t)v*Nf+f], L2=lo[(size_t)v*N+nb], H2=hi[(size_t)v*N+nb]; W_R[(size_t)v*Nf+f] = b<L2?L2:(b>H2?H2:b); } } }
        }
        return;
    }
    std::vector<double> WLh((size_t)nvar*Nf), WRh((size_t)nvar*Nf);
    // SHARP candidate: BVD_SHARP=tmlpu -> cheap T-MLP-u downwind compression (no
    // quadratic, fixes the order-2 bottleneck); default -> order-2 quadratic.
    static const char* SH = std::getenv("BVD_SHARP");
    static const bool sharp_tmlpu = SH && std::string(SH) == "tmlpu";
    static const bool sharp_thinc = SH && std::string(SH) == "thinc";
    static const bool sharp_thincqq = SH && std::string(SH) == "thincqq";
    static const char* TS = std::getenv("BVD_TSTAR");
    static const double tstar = TS ? std::atof(TS) : 1.0;
    static const char* BT = std::getenv("BVD_BETA");
    static const double beta = BT ? std::atof(BT) : 1.6;
    if (sharp_thincqq) {
        reconstruct_thinc_qq(m, ctx_sharp, W, nvar, WLh, WRh, beta);
    } else if (sharp_thinc) {
        reconstruct_thinc(m, ctx_smooth, W, nvar, WLh, WRh, beta);
    } else if (sharp_tmlpu) {
        reconstruct_tmlpu_dw(m, ctx_smooth, W, nvar, WLh, WRh, tstar);
    } else {
        std::vector<double> wl, wr;
        for (int v = 0; v < nvar; ++v) {
            reconstruct_o2_limited(m, ctx_sharp, W, nvar, v, wl, wr, shear, krelax, tvbM, venkatK, hier);  // parallel inside
            #pragma omp parallel for
            for (int f = 0; f < Nf; ++f) { WLh[(size_t)v*Nf+f]=wl[f]; WRh[(size_t)v*Nf+f]=wr[f]; }
        }
    }
    // per-face jumps for both candidates (parallel, no races). Default: single
    // sel_var (density). BVD_MULTIVAR=1: sum of per-variable jumps normalised by
    // each variable's range -> makes the shear-dominated slip line (velocity jump,
    // small density jump) trigger the sharp candidate instead of staying smooth.
    static const bool MV = std::getenv("BVD_MULTIVAR") != nullptr;
    std::vector<double> scale(nvar, 1.0);
    if (MV) {
        for (int v = 0; v < nvar; ++v) {
            double mn = 1e300, mx = -1e300;
            for (int i = 0; i < N; ++i) { double w = W[(size_t)v*N+i];
                if (w < mn) mn = w; if (w > mx) mx = w; }
            scale[v] = std::max(mx - mn, 1e-30);
        }
    }
    std::vector<double> js(Nf), jh(Nf);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        if (m.face_neighbour[f] < 0) { js[f]=jh[f]=0.0; continue; }
        if (MV) {
            double s = 0.0, h = 0.0;
            for (int v = 0; v < nvar; ++v) {
                s += std::fabs(WLs[(size_t)v*Nf+f]-WRs[(size_t)v*Nf+f]) / scale[v];
                h += std::fabs(WLh[(size_t)v*Nf+f]-WRh[(size_t)v*Nf+f]) / scale[v];
            }
            js[f] = s; jh[f] = h;
        } else {
            js[f] = std::fabs(WLs[(size_t)sel_var*Nf+f] - WRs[(size_t)sel_var*Nf+f]);
            jh[f] = std::fabs(WLh[(size_t)sel_var*Nf+f] - WRh[(size_t)sel_var*Nf+f]);
        }
    }
    // ===== 3-CANDIDATE BVD (Cheng2021 + high-order): smooth=P2(WLh, 3rd-order),
    // robust=P1/BJ(WLs), sharp=THINC(WLt). Per cell pick min-TBV. -> 3rd-order in smooth
    // (beats Cheng's P1-smooth), THINC sharp at contacts/slip-lines (3 cells, low-diss),
    // P1 robust fallback where P2 would oscillate (avoids P2's strong-shock divergence).
    // Requires WLh=P2 (do NOT set BVD_SHARP). =====
    static const bool C3 = std::getenv("BVD_3CAND") != nullptr;
    if (C3 && !sharp_tmlpu && !sharp_thinc) {
        std::vector<double> WLt, WRt;
        reconstruct_thinc(m, ctx_smooth, W, nvar, WLt, WRt, beta);
        std::vector<double> jt(Nf, 0.0);
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) { if (m.face_neighbour[f] < 0) { jt[f]=0; continue; }
            if (MV) { double t=0; for (int v=0;v<nvar;++v) t+=std::fabs(WLt[(size_t)v*Nf+f]-WRt[(size_t)v*Nf+f])/scale[v]; jt[f]=t; }
            else jt[f]=std::fabs(WLt[(size_t)sel_var*Nf+f]-WRt[(size_t)sel_var*Nf+f]); }
        // SHOCK VETO: THINC over-steepens the acoustic (pressure) jump at a genuine shock
        // -> negative pressure / divergence. P2 oscillates there. So where the relative
        // pressure jump is large (shock, genuinely nonlinear) force the robust P1/BJ; THINC
        // (sharp) is allowed ONLY at contacts/slip-lines (pressure ~continuous, lin. degen.).
        static const double C3_PJ = std::getenv("BVD_3CAND_PJ") ? std::atof(std::getenv("BVD_3CAND_PJ")) : 0.15;
        std::vector<char> pick(N, 0);  // 0=P1/BJ robust, 1=P2 smooth, 2=THINC sharp
        #pragma omp parallel for
        for (int c = 0; c < N; ++c) { double ts=0,th=0,tt=0;
            for (int f : m.cell_faces[c]) { ts+=js[f]; th+=jh[f]; tt+=jt[f]; }
            double pc=W[(size_t)(nvar-1)*N+c], mpj=0.0;
            for (int k=0;k<ctx_smooth.max_nb;++k){ int nb=ctx_smooth.nb[(size_t)c*ctx_smooth.max_nb+k]; if(nb<0)continue;
                double pj=std::fabs(W[(size_t)(nvar-1)*N+nb]-pc)/(std::fabs(pc)+1e-30); if(pj>mpj)mpj=pj; }
            if (mpj > C3_PJ) pick[c]=0;                                      // shock -> robust P1
            else if (tt<=ts && tt<=th) pick[c]=2;                           // contact -> THINC
            else if (th<=ts) pick[c]=1; else pick[c]=0; }                   // smooth -> P2, else P1
        W_L.assign((size_t)nvar*Nf, 0.0); W_R.assign((size_t)nvar*Nf, 0.0);
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) { int o=m.face_owner[f], n=m.face_neighbour[f];
            int po=pick[o], pn=(n>=0)?pick[n]:0;
            for (int v = 0; v < nvar; ++v) {
                const std::vector<double>& SL = po==2?WLt:(po==1?WLh:WLs);
                W_L[(size_t)v*Nf+f] = SL[(size_t)v*Nf+f];
                if (n>=0) { const std::vector<double>& SR = pn==2?WRt:(pn==1?WRh:WRs);
                    W_R[(size_t)v*Nf+f] = SR[(size_t)v*Nf+f]; }
                else W_R[(size_t)v*Nf+f] = W_L[(size_t)v*Nf+f]; }
        }
        return;
    }
    // ===== PG (physics-gated CONTINUOUS blend): instead of discrete TBV switch, blend
    // smooth(mlp_u1) <-> sharp(P2) by a per-cell gate g in [0,1] that is ~1 ONLY at a
    // genuine contact/slip-line (density/velocity jump at ~constant pressure) and 0 at
    // shocks (pressure jump) and in smooth flow. -> W_f = (1-g) W_mlp + g W_P2.
    // NEVER worse than mlp_u1 (g=0 reproduces it exactly); enhances only at contacts.
    // Cheap (one neighbour loop, no sorts). Hessian-free physical classifier. =====
    static const bool PG = std::getenv("BVD_PG") != nullptr;
    if (PG && nvar >= 4) {
        static const double PG_RJ = std::getenv("BVD_PG_RJ") ? std::atof(std::getenv("BVD_PG_RJ")) : 0.04; // contact sharpness on-scale (rel jump)
        static const double PG_PJ = std::getenv("BVD_PG_PJ") ? std::atof(std::getenv("BVD_PG_PJ")) : 0.20; // shock pressure-jump veto scale
        std::vector<double> gate(N, 0.0);
        double uref = 1e-30; for (int i=0;i<N;++i){ double u=W[(size_t)1*N+i],v=W[(size_t)2*N+i]; double s=std::sqrt(u*u+v*v); if(s>uref)uref=s; }
        #pragma omp parallel for
        for (int c = 0; c < N; ++c) {
            double pc=W[(size_t)(nvar-1)*N+c], rc=W[(size_t)0*N+c], uc=W[(size_t)1*N+c], vc=W[(size_t)2*N+c];
            double mpj=0.0, mrj=0.0, mvj=0.0;
            for (int k=0;k<ctx_smooth.max_nb;++k){ int nb=ctx_smooth.nb[(size_t)c*ctx_smooth.max_nb+k]; if(nb<0)continue;
                double pj=std::fabs(W[(size_t)(nvar-1)*N+nb]-pc)/(std::fabs(pc)+1e-30); if(pj>mpj)mpj=pj;
                double rj=std::fabs(W[(size_t)0*N+nb]-rc)/(std::fabs(rc)+1e-30); if(rj>mrj)mrj=rj;
                double du=W[(size_t)1*N+nb]-uc, dv=W[(size_t)2*N+nb]-vc; double vj=std::sqrt(du*du+dv*dv)/uref; if(vj>mvj)mvj=vj; }
            double sharp = std::max(smoothstep(0.5*PG_RJ, PG_RJ, mrj), smoothstep(0.5*PG_RJ, PG_RJ, mvj)); // density OR velocity jump
            double shock = smoothstep(0.5*PG_PJ, PG_PJ, mpj);                                              // pressure jump = shock veto
            gate[c] = sharp * (1.0 - shock);
        }
        W_L.assign((size_t)nvar*Nf, 0.0); W_R.assign((size_t)nvar*Nf, 0.0);
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            int o = m.face_owner[f], n = m.face_neighbour[f];
            double go = gate[o], gn = (n>=0) ? gate[n] : 0.0;
            for (int v = 0; v < nvar; ++v) {
                W_L[(size_t)v*Nf+f] = (1.0-go)*WLs[(size_t)v*Nf+f] + go*WLh[(size_t)v*Nf+f];
                if (n >= 0) W_R[(size_t)v*Nf+f] = (1.0-gn)*WRs[(size_t)v*Nf+f] + gn*WRh[(size_t)v*Nf+f];
                else        W_R[(size_t)v*Nf+f] = W_L[(size_t)v*Nf+f];
            }
        }
        return;
    }
    // per-cell TBV via gather (race-free): sharp wins where its TBV is lower.
    std::vector<char> use_sharp(N);
    #pragma omp parallel for
    for (int c = 0; c < N; ++c) {
        double ts = 0.0, th = 0.0;
        for (int f : m.cell_faces[c]) { ts += js[f]; th += jh[f]; }
        use_sharp[c] = (th < ts) ? 1 : 0;
    }
    // assemble (parallel over faces): owner's choice for W_L, neighbour's for W_R.
    W_L.assign((size_t)nvar*Nf, 0.0); W_R.assign((size_t)nvar*Nf, 0.0);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        bool so = use_sharp[o], sn = (n >= 0) && use_sharp[n];
        for (int v = 0; v < nvar; ++v) {
            W_L[(size_t)v*Nf+f] = so ? WLh[(size_t)v*Nf+f] : WLs[(size_t)v*Nf+f];
            if (n >= 0) W_R[(size_t)v*Nf+f] = sn ? WRh[(size_t)v*Nf+f] : WRs[(size_t)v*Nf+f];
            else        W_R[(size_t)v*Nf+f] = W_L[(size_t)v*Nf+f];
        }
    }
}

} // namespace cfd
