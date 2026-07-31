// cfd/reconstruct3d_unstr.hpp — UNSTRUCTURED MLP-limited deg-? GAUSS-THINC + min-TBV BVD.
//
// Genuine unstructured 3D reconstruction (mixed tetra/hexa/prism/pyramid cells). NO
// structured assumptions: cell/face polynomial moments are computed by EXACT-ish simplex
// integration over the actual mesh geometry (Duffy-Gauss, mesh-driven, no s3_h / no
// axis-aligned face), and the GAUSS closed-form sigmoid (reconstruct3d_bvd_core.hpp) is
// shape-agnostic given those moments. Smooth candidate = o2 P2-LSQ, MLP-limited to the
// node-ring stencil bound; THINC candidate = GAUSS (probit) interface, clamped; BVD picks
// min total-boundary-variation per cell/variable. GAUSS via THINCQQ_GAUSS (+_SKEW).
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/reconstruct3d_bvd_core.hpp"
#include "cfd/io_vtk.hpp"   // BVD_CANDFLAG diagnostic buffers (bvd_cand_flag); io_vtk pulls only mesh.hpp (no cycle)
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <atomic>

namespace cfd {

// recon sub-phase profiler (env E3D_PROF): split the BVD recon into o2-LSQ / cell-D / face.
struct U3Prof { double lsq=0, celld=0, facebvd=0; };
inline U3Prof& u3_prof(){ static U3Prof p; return p; }
// cell-D Newton iteration accountancy (diagnostic: total iters / total Newton solves -> avg).
inline std::atomic<long>& u3_celld_iters(){ static std::atomic<long> n{0}; return n; }
inline std::atomic<long>& u3_celld_solves(){ static std::atomic<long> n{0}; return n; }
inline bool u3_prof_on(){ static const bool b=std::getenv("E3D_PROF")!=nullptr; return b; }
inline double u3_prof_ms(){ return std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ── 2D S1 vertex-MLP limiters, VERBATIM from reconstruct2d.hpp:104-124 (local copies with a
// u3_ prefix so no cross-header ODR clash if a TU also pulls reconstruct2d.hpp). u3_bj_phi =
// Barth-Jespersen (MLP-u1); u3_venk_phi = Venkatakrishnan MLP-u2 (differentiable, less
// diffusive) on the SAME vertex neighbour min/max bound. Used by the SMOOTH candidate below so
// the 3D S1 smooth limiter is byte-faithful to 2D (only the eps2 length-scale is 3D geometry).
inline double u3_bj_phi(double delta,double center,double lo,double hi){
    const double eps=1e-30;
    double allowed = delta>=0.0?(hi-center):(center-lo);
    double phi=(std::fabs(delta)>eps)?std::max(allowed,0.0)/std::max(std::fabs(delta),eps):1.0;
    return phi<0.0?0.0:(phi>1.0?1.0:phi);
}
inline double u3_venk_phi(double delta,double center,double lo,double hi,double eps2){
    const double eps=1e-30;
    if(std::fabs(delta)<=eps) return 1.0;
    double dp = delta>=0.0?(hi-center):(lo-center);   // signed Delta+
    double dm = delta;                                // Delta-
    double phi=(dp*dp+2.0*dp*dm+eps2)/(dp*dp+dp*dm+2.0*dm*dm+eps2);
    return phi<0.0?0.0:(phi>1.0?1.0:phi);
}

// GL order for the geometric-moment (Duffy) quadrature. DEFAULT 4: the Duffy jacobian
// (1-xi)^2(1-eta) raises a degree-d monomial to effective degree d+2, so n=3 (deg-5 exact)
// is INEXACT for the deg-4 moments (effective 6) -- measured 1e-3 rel error on <P^2>, which
// perturbs the GAUSS/beta* (S2/S3) BVD decision input. n=4 (deg-7 exact) makes them exact at
// no measurable cost (gmom is built once per mesh). S1/tanh is unaffected (own quadrature).
// Env U3_GMOM_NQ overrides in [2,8]; =3 reproduces the pre-2026-07-26 results.
inline int u3_gmom_nq() {
    static const int nq = []{ const char* e=std::getenv("U3_GMOM_NQ");
        int v=(e&&e[0])?std::atoi(e):4; return (v>=2&&v<=8)?v:4; }();
    return nq;
}
// Gauss-Legendre on [0,1], n=u3_gmom_nq() (default 4). Thread-safe magic-static init
// (the recon evaluates this inside OpenMP parallel regions).
inline const std::vector<std::array<double,2>>& u3_gl() {
    static const std::vector<std::array<double,2>> g = []{
        const int n=u3_gmom_nq(); std::vector<std::array<double,2>> r;
        for (int i=0;i<n;++i){ double z=std::cos(M_PI*(i+0.75)/(n+0.5)),z1,pp;
            do{ double p1=1,p2=0; for(int j=0;j<n;++j){double p3=p2;p2=p1;p1=((2*j+1)*z*p2-j*p3)/(j+1);}
                pp=n*(z*p1-p2)/(z*z-1.0); z1=z; z=z1-p1/pp; }while(std::fabs(z-z1)>1e-14);
            r.push_back({0.5*(1.0-z), 1.0/((1.0-z*z)*pp*pp)}); }
        return r; }();
    return g;
}

// integrate f(x,y,z) over cell ci (sum over face-fan tets from the cell centroid).
template<class F>
inline double u3_int_cell(const Mesh& m, int ci, const F& f) {
    const auto& gl = u3_gl(); const double* X = m.nodes.data();
    double cx=m.cell_centers[3*ci],cy=m.cell_centers[3*ci+1],cz=m.cell_centers[3*ci+2];
    double s=0;
    for (int fc : m.cell_faces[ci]) {
        const auto& fn = m.face_nodes[fc];
        double fx=m.face_centers[3*fc],fy=m.face_centers[3*fc+1],fz=m.face_centers[3*fc+2];
        for (size_t i=0;i<fn.size();++i){ int va=fn[i], vb=fn[(i+1)%fn.size()];
            // tet (cell-centroid, face-centroid, va, vb)
            double ax=fx-cx,ay=fy-cy,az=fz-cz;          // edges from cell centroid
            double bx=X[3*va]-cx,by=X[3*va+1]-cy,bz=X[3*va+2]-cz;
            double dx=X[3*vb]-cx,dy=X[3*vb+1]-cy,dz=X[3*vb+2]-cz;
            double vol=std::fabs(ax*(by*dz-bz*dy)-ay*(bx*dz-bz*dx)+az*(bx*dy-by*dx))/6.0;
            for (auto& gi:gl) for(auto& gj:gl) for(auto& gk:gl){
                double xi=gi[0],eta=gj[0],ze=gk[0];
                double L1=xi,L2=(1-xi)*eta,L3=(1-xi)*(1-eta)*ze,L0=(1-xi)*(1-eta)*(1-ze);
                double px=cx+L1*ax+L2*bx+L3*dx, py=cy+L1*ay+L2*by+L3*dy, pz=cz+L1*az+L2*bz+L3*dz;
                s += gi[1]*gj[1]*gk[1]*(1-xi)*(1-xi)*(1-eta)*6.0*vol*f(px,py,pz);
            }
        }
    }
    return s;
}
// SELF-CONSISTENT FAN AREA of face fc: the sum of the SAME edge-fan sub-triangles (from the face
// centroid) that u3_int_face integrates over. For a PLANAR face this equals m.face_areas[fc]
// (which the mesh reader builds as a v0-fan, i.e. a fan from the FIRST NODE), but for a WARPED
// (non-planar) quad the two decompositions cover different surfaces and their areas DIFFER.
// Normalizing a fan-quadrature integral by the v0-fan area therefore leaves the moment weights
// un-normalized (measured |Sum w - 1| up to 1.7e-1 on warped quads). All moment/average
// normalization inside this file must divide by THIS area, never m.face_areas[fc].
// NOTE: m.face_areas[fc] itself is deliberately left untouched (flux uses it).
inline double u3_face_area_fan(const Mesh& m, int fc) {
    const double* X = m.nodes.data(); const auto& fn = m.face_nodes[fc];
    double fx=m.face_centers[3*fc],fy=m.face_centers[3*fc+1],fz=m.face_centers[3*fc+2];
    double A=0;
    for (size_t i=0;i<fn.size();++i){ int va=fn[i], vb=fn[(i+1)%fn.size()];
        double ax=X[3*va]-fx,ay=X[3*va+1]-fy,az=X[3*va+2]-fz;
        double bx=X[3*vb]-fx,by=X[3*vb+1]-fy,bz=X[3*vb+2]-fz;
        double cr0=ay*bz-az*by,cr1=az*bx-ax*bz,cr2=ax*by-ay*bx;
        A += 0.5*std::sqrt(cr0*cr0+cr1*cr1+cr2*cr2);
    }
    return A;
}
// integrate f over face fc (sum over edge-fan tris from the face centroid).
template<class F>
inline double u3_int_face(const Mesh& m, int fc, const F& f) {
    const auto& gl = u3_gl(); const double* X = m.nodes.data();
    const auto& fn = m.face_nodes[fc];
    double fx=m.face_centers[3*fc],fy=m.face_centers[3*fc+1],fz=m.face_centers[3*fc+2];
    double s=0;
    for (size_t i=0;i<fn.size();++i){ int va=fn[i], vb=fn[(i+1)%fn.size()];
        double ax=X[3*va]-fx,ay=X[3*va+1]-fy,az=X[3*va+2]-fz;
        double bx=X[3*vb]-fx,by=X[3*vb+1]-fy,bz=X[3*vb+2]-fz;
        double cr0=ay*bz-az*by,cr1=az*bx-ax*bz,cr2=ax*by-ay*bx;
        double area=0.5*std::sqrt(cr0*cr0+cr1*cr1+cr2*cr2);
        for(auto& gi:gl) for(auto& gj:gl){ double xi=gi[0],eta=gj[0],L1=xi,L2=(1-xi)*eta;
            double px=fx+L1*ax+L2*bx, py=fy+L1*ay+L2*by, pz=fz+L1*az+L2*bz;
            s += gi[1]*gj[1]*(1-xi)*2.0*area*f(px,py,pz);
        }
    }
    return s;
}

inline double u3_Peval(const D3Poly& P,double dx,double dy,double dz){
    return P.c[0]*dx+P.c[1]*dy+P.c[2]*dz+P.c[3]*dx*dx+P.c[4]*dy*dy+P.c[5]*dz*dz+P.c[6]*dx*dy+P.c[7]*dx*dz+P.c[8]*dy*dz;}

// ---- EXACT tanh THINC/QQ (numerical quad + Newton).  env THINCQQ_TANH.
// ORIGINAL scheme unchanged (true tanh, Newton cell-D, tensor Duffy-Gauss face/cell average).
// The ONLY knob is the number of Gauss points per axis NQ (env THINCQQ_TANH_NQ, default 2):
//   cell = NQ^3 pts/sub-tet, face = NQ^2 pts/sub-tri.  Original was NQ=5 (125/25); NQ=2 (8/4)
//   is the chosen reduction — degree-3-exact per Duffy axis, conservative, ~accuracy of NQ=5.
//   NQ=1 (1/1) is the cheapest but degree-1 only (non-conservative, ~5-15% cell-avg error).
inline int u3_tanh_nq(){ static const int n=[]{const char*e=std::getenv("THINCQQ_TANH_NQ");return (e&&e[0])?std::atoi(e):2;}(); return n; }
// NQ-point Gauss-Legendre nodes/weights on [0,1] (cached per NQ; built once).
inline const std::vector<std::array<double,2>>& u3_tanh_gl(){
    static const std::vector<std::array<double,2>> g=[]{
        int n=u3_tanh_nq(); std::vector<std::array<double,2>> r; r.reserve(n);
        for(int i=0;i<n;++i){ double z=std::cos(M_PI*(i+0.75)/(n+0.5)),z1,pp;
            do{ double p1=1,p2=0;
                for(int j=0;j<n;++j){ double p3=p2;p2=p1; p1=((2*j+1)*z*p2-j*p3)/(j+1); }
                pp=n*(z*p1-p2)/(z*z-1.0); z1=z; z=z1-p1/pp;
            }while(std::fabs(z-z1)>1e-15);
            r.push_back({0.5*(1.0-z), 1.0/((1.0-z*z)*pp*pp)}); }
        return r;
    }();
    return g;
}
inline bool u3_tanh_c1(){ return false; }  // legacy hook (centroid path removed; use NQ=1)

// fill Pv/wt in-place (caller passes thread_local refs — zero heap alloc after first call)
inline void u3_cell_pvals(const Mesh& m,int ci,const D3Poly& P,std::vector<double>& Pv,std::vector<double>& wt){
    const double* X=m.nodes.data();
    double cx=m.cell_centers[3*ci],cy=m.cell_centers[3*ci+1],cz=m.cell_centers[3*ci+2]; double V=m.cell_volumes[ci];
    Pv.clear(); wt.clear();
    const auto& gl=u3_tanh_gl();
    for(int fc:m.cell_faces[ci]){ const auto& fn=m.face_nodes[fc];
        double fx=m.face_centers[3*fc],fy=m.face_centers[3*fc+1],fz=m.face_centers[3*fc+2];
        for(size_t i=0;i<fn.size();++i){ int va=fn[i],vb=fn[(i+1)%fn.size()];
            double ax=fx-cx,ay=fy-cy,az=fz-cz, bx=X[3*va]-cx,by=X[3*va+1]-cy,bz=X[3*va+2]-cz, dx=X[3*vb]-cx,dy=X[3*vb+1]-cy,dz=X[3*vb+2]-cz;
            double vol=std::fabs(ax*(by*dz-bz*dy)-ay*(bx*dz-bz*dx)+az*(bx*dy-by*dx))/6.0;
            for(auto&gi:gl)for(auto&gj:gl)for(auto&gk:gl){ double xi=gi[0],eta=gj[0],ze=gk[0];
                double L1=xi,L2=(1-xi)*eta,L3=(1-xi)*(1-eta)*ze;
                double rx=L1*ax+L2*bx+L3*dx, ry=L1*ay+L2*by+L3*dy, rz=L1*az+L2*bz+L3*dz;
                Pv.push_back(u3_Peval(P,rx,ry,rz)); wt.push_back(gi[1]*gj[1]*gk[1]*(1-xi)*(1-xi)*(1-eta)*6.0*vol/V); } }
    }
}
inline void u3_face_pvals(const Mesh& m,int fc,const D3Poly& P,double cx,double cy,double cz,std::vector<double>& Pv,std::vector<double>& wt){
    const double* X=m.nodes.data(); const auto& fn=m.face_nodes[fc];
    double fx=m.face_centers[3*fc],fy=m.face_centers[3*fc+1],fz=m.face_centers[3*fc+2];
    Pv.clear(); wt.clear();
    const auto& gl=u3_tanh_gl();
    // A = SELF-CONSISTENT fan area (sum of the sub-triangles actually used below), NOT
    // m.face_areas[fc] (v0-fan). Accumulated in the same pass, then divided out at the end,
    // so Sum(wt)==1 to roundoff even on WARPED (non-planar) quad faces.
    double A=0;
    for(size_t i=0;i<fn.size();++i){ int va=fn[i],vb=fn[(i+1)%fn.size()];
        double ax=X[3*va]-fx,ay=X[3*va+1]-fy,az=X[3*va+2]-fz, bx=X[3*vb]-fx,by=X[3*vb+1]-fy,bz=X[3*vb+2]-fz;
        double cr0=ay*bz-az*by,cr1=az*bx-ax*bz,cr2=ax*by-ay*bx; double area=0.5*std::sqrt(cr0*cr0+cr1*cr1+cr2*cr2);
        A+=area;
        for(auto&gi:gl)for(auto&gj:gl){ double xi=gi[0],eta=gj[0],L1=xi,L2=(1-xi)*eta;
            double px=fx+L1*ax+L2*bx, py=fy+L1*ay+L2*by, pz=fz+L1*az+L2*bz;
            Pv.push_back(u3_Peval(P,px-cx,py-cy,pz-cz)); wt.push_back(gi[1]*gj[1]*(1-xi)*2.0*area); } }
    if(A>1e-300){ for(double& w:wt) w/=A; }   // division (not *1/A) => bit-identical to the old `...*area/A` form when A matches
}
// Newton for the cell-D constraint <tanh(kk P + D)>_cell = Q. The bottleneck (≈86% of recon)
// was a COLD start (D=0) that needed ~10-15 iters, each summing tanh over all ~192 cell quad
// points × 2 betas. Two fixes, no scheme change (still true tanh + exact quadrature):
//   (1) WARM START D0 from the GAUSS closed form (probit estimate of D, ~1-3% off) -> the
//       monotone tanh Newton then polishes to 1e-12 in ~2-3 iters (cap 8).
//   (2) HOIST s_q = kk·Pv[q] (D-independent) into a per-thread scratch -> computed once, not
//       re-multiplied every iter.
inline double u3_tanh_cellD(const std::vector<double>& Pv,const std::vector<double>& wt,double kk,double Q,double D0=0.0){
    // XIE2017 THINC/QQ Eq.21-23 tanh-addition acceleration: the cell-average constraint
    // Σ_q w_q tanh(kk·P_q + D) = Q is solved for the shift D. tanh(kk·P_q + D) =
    // (A_q + Dt)/(1 + A_q·Dt) with A_q=tanh(kk·P_q) (D-INDEPENDENT, precomputed ONCE) and Dt=tanh(D).
    // Newton then iterates on Dt over a RATIONAL function — NO tanh/exp inside the loop (pure
    // arithmetic, SIMD-vectorizable) — instead of the old form that recomputed tanh(kk·P_q+D) every
    // iteration × every quad point (the 69%-of-recon cost). D=atanh(Dt) once at the end. Bit-exact
    // (tanh addition is an identity). Opt back to the direct form with THINCQQ_CELLD_DIRECT.
    static const bool direct = std::getenv("THINCQQ_CELLD_DIRECT") != nullptr;
    size_t nq=Pv.size();
    static const int MAXIT = []{ const char* e=std::getenv("THINCQQ_TANH_MAXIT"); return (e&&e[0])?std::atoi(e):50; }();
    if(direct){
        thread_local std::vector<double> s; s.resize(nq);
        for(size_t q=0;q<nq;++q) s[q]=kk*Pv[q];
        double D=D0; int it=0;
        for(; it<MAXIT; ++it){ double f=-Q,fp=0;
            for(size_t q=0;q<nq;++q){ double th=std::tanh(s[q]+D); f+=wt[q]*th; fp+=wt[q]*(1.0-th*th); }
            if(fp<1e-14){ ++it; break; } double dD=f/fp; D-=dD; if(std::fabs(dD)<1e-12){ ++it; break; } }
        if(u3_prof_on()){ u3_celld_iters().fetch_add(it,std::memory_order_relaxed); u3_celld_solves().fetch_add(1,std::memory_order_relaxed); }
        return D;
    }
    thread_local std::vector<double> A, om;   // A_q=tanh(kk·P_q), om_q=1-A_q² (derivative numerator)
    A.resize(nq); om.resize(nq);
    for(size_t q=0;q<nq;++q){ double a=std::tanh(kk*Pv[q]); A[q]=a; om[q]=1.0-a*a; }   // ONLY tanh calls
    double Dt=std::tanh(D0); int it=0;
    for(; it<MAXIT; ++it){ double f=-Q, fp=0;
        for(size_t q=0;q<nq;++q){ double id=1.0/(1.0+A[q]*Dt); f+=wt[q]*((A[q]+Dt)*id); fp+=wt[q]*(om[q]*id*id); }
        if(fp<1e-14){ ++it; break; } double dDt=f/fp; Dt-=dDt;
        if(Dt<=-1.0) Dt=-1.0+1e-13; else if(Dt>=1.0) Dt=1.0-1e-13;   // keep Dt in (-1,1)
        if(std::fabs(dDt)<1e-12){ ++it; break; } }
    if(u3_prof_on()){ u3_celld_iters().fetch_add(it,std::memory_order_relaxed); u3_celld_solves().fetch_add(1,std::memory_order_relaxed); }
    return 0.5*std::log((1.0+Dt)/(1.0-Dt));   // D = atanh(Dt), single call
}
inline double u3_tanh_face_avg(const std::vector<double>& Pv,const std::vector<double>& wt,double kk,double D){
    double s=0; for(size_t q=0;q<Pv.size();++q) s+=wt[q]*std::tanh(kk*Pv[q]+D); return s;
}

// ===========================================================================================
// GEOMETRIC-MOMENT PRECOMPUTE (perf): the interface poly P(d)=Σ c_a·mon_a(d) is a quadratic in
// (dx,dy,dz), so its cell/face moments M1=<P>, M2=<P^2> are exact linear/bilinear combos of the
// GEOMETRY-ONLY monomial moments <dx^i dy^j dz^k> (i+j+k<=4). Precomputing those ONCE per mesh
// (35 per cell + 35 per incident cell-face, rel the CELL centroid, via the SAME Duffy-Gauss
// quadrature) removes ALL per-variable per-step integration from the hot loop — M1/M2 become
// ~20-FLOP dot products. Bit-identical to the on-the-fly path (only the summation order changes).
// Degree-4 table covers M1 (deg<=2) and M2 (deg<=4); the skew M3 (deg<=6) keeps the slow path.
inline const std::array<std::array<int,3>,35>& u3_mon_list(){
    static const std::array<std::array<int,3>,35> L = []{
        std::array<std::array<int,3>,35> a; int n=0;
        for(int d=0;d<=4;++d) for(int i=d;i>=0;--i) for(int j=d-i;j>=0;--j){ a[n++]={i,j,(d-i-j)}; }
        return a; }();
    return L;
}
inline int u3_midx(int i,int j,int k){
    static const std::array<int,125> M = []{
        std::array<int,125> m; m.fill(-1); const auto& L=u3_mon_list();
        for(int n=0;n<35;++n) m[L[n][0]*25+L[n][1]*5+L[n][2]]=n; return m; }();
    return M[i*25+j*5+k];
}
// the 9 monomials of P, in c[]-order: dx,dy,dz,dx2,dy2,dz2,dxy,dxz,dyz
inline const std::array<std::array<int,3>,9>& u3_Pmon(){
    static const std::array<std::array<int,3>,9> p = {{ {1,0,0},{0,1,0},{0,0,1},{2,0,0},{0,2,0},{0,0,2},{1,1,0},{1,0,1},{0,1,1} }};
    return p;
}
// map P-monomial a -> 35-index (for M1); product (a,b) -> 35-index (for M2). Built once.
inline const std::array<int,9>& u3_g1(){
    static const std::array<int,9> g = []{ std::array<int,9> r; const auto& p=u3_Pmon();
        for(int a=0;a<9;++a) r[a]=u3_midx(p[a][0],p[a][1],p[a][2]); return r; }();
    return g;
}
inline const std::array<std::array<int,9>,9>& u3_g2(){
    static const std::array<std::array<int,9>,9> g = []{ std::array<std::array<int,9>,9> r; const auto& p=u3_Pmon();
        for(int a=0;a<9;++a) for(int b=0;b<9;++b) r[a][b]=u3_midx(p[a][0]+p[b][0],p[a][1]+p[b][1],p[a][2]+p[b][2]);
        return r; }();
    return g;
}
struct U3Gmom {
    const Mesh* mp=nullptr; int N=0;
    std::vector<double> cellm;     // N*35  : <mon>_cell (÷V), rel cell centroid
    std::vector<double> facem;     // FLAT: face moments at face_off[ci] + e*35 + n (rel cell centroid)
    std::vector<size_t> face_off;  // N+1 prefix offsets into facem
};
// build all 35 cell + per-cell-face monomial moments (rel each cell's centroid). ONCE per mesh.
inline void u3_build_gmom(const Mesh& m, U3Gmom& G){
    const int N=m.n_cells(); G.mp=&m; G.N=N; G.cellm.assign((size_t)N*35,0.0);
    G.face_off.assign(N+1,0);
    for(int ci=0;ci<N;++ci) G.face_off[ci+1]=G.face_off[ci]+(size_t)m.cell_faces[ci].size()*35;
    G.facem.assign(G.face_off[N],0.0);
    const auto& L=u3_mon_list();
    #pragma omp parallel for schedule(dynamic,64)
    for(int ci=0;ci<N;++ci){
        double cx=m.cell_centers[3*ci],cy=m.cell_centers[3*ci+1],cz=m.cell_centers[3*ci+2];
        double V=m.cell_volumes[ci];
        for(int n=0;n<35;++n){ int i=L[n][0],j=L[n][1],k=L[n][2];
            G.cellm[(size_t)ci*35+n]=u3_int_cell(m,ci,[&](double x,double y,double z){
                double dx=x-cx,dy=y-cy,dz=z-cz; double r=1;
                for(int t=0;t<i;++t)r*=dx; for(int t=0;t<j;++t)r*=dy; for(int t=0;t<k;++t)r*=dz; return r;})/V; }
        int nf=m.cell_faces[ci].size();
        for(int e=0;e<nf;++e){ int fc=m.cell_faces[ci][e]; double A=u3_face_area_fan(m,fc);   // self-consistent w/ u3_int_face (NOT m.face_areas: v0-fan differs on warped quads)
            double* fm=&G.facem[G.face_off[ci]+(size_t)e*35];
            for(int n=0;n<35;++n){ int i=L[n][0],j=L[n][1],k=L[n][2];
                fm[n]=u3_int_face(m,fc,[&](double x,double y,double z){
                    double dx=x-cx,dy=y-cy,dz=z-cz; double r=1;
                    for(int t=0;t<i;++t)r*=dx; for(int t=0;t<j;++t)r*=dy; for(int t=0;t<k;++t)r*=dz; return r;})/A; }
        }
    }
}
// M1=<P>, M2=<P^2> from a 35-moment table mom[] and P coeffs c[] (closed-form, no integration).
inline void u3_M12(const double* mom, const double* c, double& M1, double& M2){
    const auto& g1=u3_g1(); const auto& g2=u3_g2();
    double m1=0; for(int a=0;a<9;++a) m1+=c[a]*mom[g1[a]];
    double m2=0; for(int a=0;a<9;++a){ double ca=c[a]; m2+=ca*ca*mom[g2[a][a]];
        for(int b=a+1;b<9;++b) m2+=2.0*ca*c[b]*mom[g2[a][b]]; }
    M1=m1; M2=m2;
}
// local index of global face fc within cell ci's cell_faces (for GMOM facem lookup; cell_faces is tiny).
inline int u3_face_lidx(const Mesh& m, int ci, int fc){
    const auto& cf=m.cell_faces[ci];
    for(int e=0;e<(int)cf.size();++e) if(cf[e]==fc) return e;
    return -1; }

// OPT2: O(1) face->local-slot table replacing the u3_face_lidx linear scan in the GMOM facem
// lookups. loc[2f+0] = local index of face f within its OWNER's cell_faces; loc[2f+1] = same
// within its NEIGHBOUR's cell_faces (-1 if boundary). Built once per mesh (geometry-only),
// cached like U3NodeAdj. u3_face_lidx_tab returns the SAME local index as u3_face_lidx (ci is
// always the owner or the neighbour of fc, since fc came from cell_faces[ci]).
struct U3FaceLoc{ const Mesh* mp=nullptr; int N=0, Nf=0; std::vector<int> loc; };
inline void u3_build_faceloc(const Mesh& m, U3FaceLoc& FL){
    int N=m.n_cells(), Nf=m.n_faces();
    FL.mp=&m; FL.N=N; FL.Nf=Nf; FL.loc.assign((size_t)2*Nf,-1);
    for(int ci=0;ci<N;++ci){ const auto& cf=m.cell_faces[ci];
        for(int e=0;e<(int)cf.size();++e){ int fc=cf[e];
            if(m.face_owner[fc]==ci) FL.loc[(size_t)2*fc+0]=e; else FL.loc[(size_t)2*fc+1]=e; } }
}
inline int u3_face_lidx_tab(const U3FaceLoc& FL, const Mesh& m, int ci, int fc){
    return (m.face_owner[fc]==ci) ? FL.loc[(size_t)2*fc+0] : FL.loc[(size_t)2*fc+1]; }

// Build the unit-normal interface quadratic P from the o2 P2 coeffs g (grad+Hessian).
inline bool u3_build_P(const double* g, D3Poly& P){
    double gx=g[0],gy=g[1],gz=g[2], gn=std::sqrt(gx*gx+gy*gy+gz*gz);
    if(gn<1e-30) return false;
    double inv=1.0/gn, nx=gx*inv,ny=gy*inv,nz=gz*inv;
    double Hxx=g[3],Hyy=g[4],Hzz=g[5],Hxy=g[6],Hxz=g[7],Hyz=g[8];
    double Hgx=Hxx*gx+Hxy*gy+Hxz*gz, Hgy=Hxy*gx+Hyy*gy+Hyz*gz, Hgz=Hxz*gx+Hyz*gy+Hzz*gz;
    double inv3=inv*inv*inv;
    double nxx=Hxx*inv-gx*Hgx*inv3, nyy=Hyy*inv-gy*Hgy*inv3, nzz=Hzz*inv-gz*Hgz*inv3;
    double nxy=Hxy*inv-gx*Hgy*inv3, nyx=Hxy*inv-gy*Hgx*inv3;
    double nxz=Hxz*inv-gx*Hgz*inv3, nzx=Hxz*inv-gz*Hgx*inv3;
    double nyz=Hyz*inv-gy*Hgz*inv3, nzy=Hyz*inv-gz*Hgy*inv3;
    P.c[0]=nx;P.c[1]=ny;P.c[2]=nz; P.c[3]=0.5*nxx;P.c[4]=0.5*nyy;P.c[5]=0.5*nzz;
    P.c[6]=0.5*(nxy+nyx);P.c[7]=0.5*(nxz+nzx);P.c[8]=0.5*(nyz+nzy);
    return true;
}

// ── node->cells adjacency (for the genuine vertex-MLP limiter, fix 2026-06-30) ──
// MLP-u (Park-Yoon-Kim): the o2 reconstruction must stay within the min/max of cell-averages
// over cells sharing EACH VERTEX, checked AT the vertex. The old 3D limiter was face-center
// Barth-Jespersen with a single cell-wide band -> diagonal/corner over-undershoot leaked.
// Built once per mesh (geometry-only, CSR node->cells), cached like U3Gmom.
struct U3NodeAdj {
    const Mesh* mp=nullptr; int N=0, nn=0;
    std::vector<int> off, cells;   // CSR: node p -> cells[off[p]..off[p+1])
};
inline void u3_build_nodeadj(const Mesh& m, U3NodeAdj& A){
    int N=m.n_cells(); int nn=(int)(m.nodes.size()/3);
    A.mp=&m; A.N=N; A.nn=nn;
    A.off.assign((size_t)nn+1,0);
    for(int ci=0;ci<N;++ci) for(int p:m.cell_nodes[ci]) if(p>=0&&p<nn) A.off[p+1]++;
    for(int p=0;p<nn;++p) A.off[p+1]+=A.off[p];
    A.cells.assign((size_t)A.off[nn],0);
    std::vector<int> cur(A.off.begin(),A.off.end());
    for(int ci=0;ci<N;++ci) for(int p:m.cell_nodes[ci]) if(p>=0&&p<nn) A.cells[cur[p]++]=ci;
}
inline bool u3_vmlp(){ static const bool b=(std::getenv("RECON_BJ_FACE")==nullptr); return b; } // default vertex-MLP; opt-out to face BJ

// Public entry: MLP-limited GAUSS-THINC + min-TBV BVD on an unstructured 3D mesh.
inline void reconstruct3d_bvd_gauss_unstr(const Mesh& m, const ReconCtx3DO2& o2c,
        const std::vector<double>& W, int nvar, std::vector<double>& WL, std::vector<double>& WR,
        double beta_l = 1.4, double beta_s = 0.8, const double* face_a = nullptr) {   // FIX3: beta_l 1.6->1.4 = 2D S1 value
    const int N=m.n_cells(), Nf=m.n_faces();
    WL.assign((size_t)nvar*Nf,0.0); WR.assign((size_t)nvar*Nf,0.0);
    static const double BL=[]{const char*e=std::getenv("BVD_BETA_L");return (e&&e[0])?std::atof(e):-1.0;}();
    static const double BS=[]{const char*e=std::getenv("BVD_BETA_S");return (e&&e[0])?std::atof(e):-1.0;}();
    if(BL>0) beta_l=BL; if(BS>0) beta_s=BS;
    // ===== ABVD adaptive TVD-BVD linear candidate, 3D port (env BVD_ABVD, same as 2D) =====
    // Replaces the SMOOTH candidate's face values with the per-face zero-BV interval pick
    // (Majima2023 Eq.21-28, projected LSQ slope ratios) FUSED with the one-sided vertex-LMP
    // clamp psi2 in [0,2] (mandatory stabilizer; see reconstruct_bvd.hpp 2D notes).
    // face_a = optional per-face advection speed (scalar advection upwind endpoint);
    // Euler (nvar>=5) derives the face-normal velocity internally from vars 1..3.
    static const bool ABVD3   = std::getenv("BVD_ABVD") != nullptr;
    static const bool A3_SAFE = std::getenv("BVD_ABVD_SAFE") != nullptr;
    static const bool A3_EMM  = std::getenv("BVD_ABVD_EMM") != nullptr;
    auto a3_mm = [](double r){ return std::max(0.0, std::min(r, 1.0)); };
    auto a3_sb = [](double r){ return std::max(0.0, std::max(std::min(2.0*r,1.0), std::min(r,2.0))); };
    auto a3_vl = [](double r){ return (r+std::fabs(r))/(1.0+std::fabs(r)); };
    static std::vector<double> g_all;      // OPT1: o2 P2 coeffs ALL vars (var-outermost, nvar*N*9); H2 persistent (fused reconstruct3d_o2_coeffs_allvars)

    // per-cell interface state (current var): TWO THINC candidates (sharp beta_l, gentle beta_s)
    struct Cint{ D3Poly P; double D_l=0,D_s=0,D_star=0,kk_l=0,kk_s=0,M1=0,M2=0,qmin=0,qmax=0; int hasint=0; double phi=1.0; double phi2=1.0; };
    static std::vector<Cint> ci_st; ci_st.assign(N, Cint{});   // H2 persistent scratch (byte-identical re-init)
    // per-FACE candidate face values (BVD face-jump metric, fix 2026-06-30): for each of the 3
    // candidates {o2-quad, THINC_l, THINC_s}, the OWNER-side (qoF) and NEIGHBOUR-side (qnF) face
    // value. TBV = Σ_faces |qoF − qnF| (true boundary variation), shared by both adjacent cells.
    // 4th candidate slot (index 3) = PAPER 3 per-cell TBV-min beta* (GAUSS closed-form D(beta)).
    static std::vector<double> qoF, qnF; qoF.assign((size_t)4*Nf,0.0); qnF.assign((size_t)4*Nf,0.0);   // H2 persistent; OPT3 layout = candidate-INNERMOST [f*4+cand] (4 cands per face = 1 cache line)
    static std::vector<int> pickv; pickv.assign(N,0);   // H2 persistent

    // a-priori strong-shock sensor: at big-pressure-jump COMPRESSIVE cells force FIRST-ORDER
    // (the MLP-bounded GAUSS still over-sharpens a Mach-strong shock into neg-p on the
    // viscous shock tube). env BVD_SHOCK=1 to enable, BVD_SHOCK_P = relative p-jump threshold.
    static const int    SHK  = []{const char*e=std::getenv("BVD_SHOCK");  return (e&&e[0])?std::atoi(e):0;}();
    static const double SHKP = []{const char*e=std::getenv("BVD_SHOCK_P");return (e&&e[0])?std::atof(e):0.5;}();
    // U2 smooth-extremum limiter spare (accuracy-preserving): at a smooth extremum the BJ limiter
    // spuriously clips (MMS: 3rd→1.46 order). Detect smoothness by per-axis curvature (Hessian diag)
    // sign-coherence over the node-ring (no Gibbs) and DO NOT limit there (φ=1); keep BJ where the
    // curvature flips sign (genuine discontinuity). MOOD(PAD) still backstops a-posteriori. DEFAULT ON;
    // opt-out RECON_BJ_HARD for the legacy hard-BJ smooth candidate.
    static const bool LIM_U2 = (std::getenv("RECON_BJ_HARD") == nullptr);
    // MLP-u2 (Venkatakrishnan) smooth-limiter knob — 2D S1 parity (reconstruct2d.hpp:1053-1055).
    // MLP_U2=K>0 (S1 sets 0.001) -> the SMOOTH candidate uses u3_venk_phi on the SAME vertex-MLP
    // bound as BJ (cost-identical, less diffusive at smooth extrema); K==0 -> hard u3_bj_phi.
    // eps2 = (K·(2V)^{1/3})^3 (3D geometric analog of 2D's (K·(2·area)^{1/2})^3 — only the length
    // power 1/2 -> 1/3 differs). When K>0 the venk relaxation IS the smooth-extremum treatment, so
    // the 3D-only U2 Hessian spare (below) is disabled (FIX1b) to match 2D exactly.
    static const char* U2ENV = std::getenv("MLP_U2");
    static const double U2K = U2ENV ? std::atof(U2ENV) : 0.0;
    std::vector<unsigned char> shockcell;
    if(SHK && nvar>=5){ const int ip=nvar-1; shockcell.assign((size_t)N,0);
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ double pc=W[(size_t)ip*N+ci]; double rp=0,divv=0;
            for(int fc:m.cell_faces[ci]){ bool ow=(m.face_owner[fc]==ci); int nb=ow?m.face_neighbour[fc]:m.face_owner[fc]; if(nb<0)continue;
                double pn=W[(size_t)ip*N+nb],mn=std::min(pc,pn); double jp=(mn>0.0)?std::fabs(pc-pn)/mn:1e300; if(jp>rp)rp=jp;
                const double* fn=&m.face_normals[3*fc]; double sg=ow?1.0:-1.0;
                for(int d=0;d<3;++d) divv+=0.5*(W[(size_t)(1+d)*N+nb]+W[(size_t)(1+d)*N+ci])*fn[d]*sg; }
            shockcell[ci]=(rp>SHKP && divv<0.0)?1:0; }
    }

    // GEOMETRIC-MOMENT FAST PATH (perf): precompute the deg<=4 monomial moments once per mesh so
    // M1/M2 + F1/F2 are closed-form combos (no per-step Duffy-Gauss integration). Bit-identical.
    // Opt-out RECON_NOGMOM; auto-off for the skew variant (deg-6 M3 stays on-the-fly). Built single-
    // threaded here (before the v-loop's OMP region); u3_build_gmom parallelises internally.
    static const bool USE_GMOM = (std::getenv("RECON_NOGMOM")==nullptr) && !deg3t_gauss_skew();
    static const bool USE_TANH = (std::getenv("THINCQQ_TANH")!=nullptr); // exact tanh THINC/QQ (numerical quad+Newton)
    // A-PRIORI THINC activation (env THINCQQ_APRIORI): the cell-D Newton + face quadrature
    // (≈92% of recon) only PAY OFF where BVD actually picks THINC — i.e. near a sharp,
    // under-resolved discontinuity, which is exactly where the MLP limiter activates (φ<1).
    // Where φ≈1 (smooth region, and u2-spared smooth extrema) the o2-quad is already
    // high-order + bounded and BVD keeps it, so the expensive THINC is wasted. So a-priori
    // SKIP THINC (hasint=0 → Pass-1 cell-D AND Pass-2 face-quad both skipped → o2-quad used)
    // for φ ≥ APHI. Cheap (reuses the already-computed φ), conservative default APHI=0.999
    // (only skip nearly-unlimited cells). Big win on smooth-dominated flows (vortex/low-Mach);
    // small on shock-saturated octant (most cells limited). env THINCQQ_APRIORI_PHI to tune.
    static const bool APRIORI = (std::getenv("THINCQQ_APRIORI")!=nullptr);
    static const double APHI = []{ const char* e=std::getenv("THINCQQ_APRIORI_PHI"); return (e&&e[0])?std::atof(e):0.999; }();
    static U3Gmom GM;
    if(USE_GMOM && (GM.mp!=&m || GM.N!=N)) u3_build_gmom(m, GM);
    // OPT2: O(1) face->local-slot table (only used inside the USE_GMOM facem lookups).
    static U3FaceLoc FL;
    if(USE_GMOM && (FL.mp!=&m || FL.N!=N || FL.Nf!=Nf)) u3_build_faceloc(m, FL);
    // vertex-MLP: node->cells adjacency (cached) + per-vertex bound arrays (per var).
    const bool VMLP=u3_vmlp();
    static U3NodeAdj NA;
    if(VMLP && (NA.mp!=&m || NA.N!=N)) u3_build_nodeadj(m, NA);
    static std::vector<double> vnmin, vnmax; if(VMLP){ vnmin.assign((size_t)NA.nn,0.0); vnmax.assign((size_t)NA.nn,0.0); }   // H2 persistent

    // ===== PAPER 3 (THINCQQ_BETASTAR + THINCQQ_BSTAR_EXACT): per-cell TBV-min beta* replaces the
    // fixed 2-beta set. beta* interior (beta_s<beta*<beta_l) -> candidate {smooth, THINC(beta*)};
    // beta* pinned to an endpoint -> fallback {smooth, THINC(beta_l), THINC(beta_s)}. Requires the
    // GAUSS closed-form D(beta) (not tanh/skew) so each grid eval is O(faces), no Newton. Selected
    // on density (v=0) once, reused for all vars (mirror of 2D reconstruct_cheng3). =====
    const bool BSTAR3 = (std::getenv("THINCQQ_BETASTAR")!=nullptr) && (std::getenv("THINCQQ_BSTAR_EXACT")!=nullptr)
                        && !USE_TANH && !deg3t_gauss_skew();
    // option c MAXB + option A WIDE (mirror of 2D reconstruct_bvd.hpp): largest beta with TBV<=TOL*min;
    // WIDE drops the 2-beta fallback -> search beta* over [WMIN,WMAX] (~[0,inf)) and always {smooth,THINC(beta*)}.
    const bool BSTAR3_MAXB = std::getenv("THINCQQ_BSTAR_MAXB")!=nullptr;
    const double BSTAR3_TOL = [](){ const char* e=std::getenv("THINCQQ_BSTAR_TOL"); return (e&&e[0])?std::atof(e):1.05; }();
    const bool BSTAR3_WIDE = std::getenv("THINCQQ_BSTAR_WIDE")!=nullptr;
    const double BSTAR3_WMIN = [](){ const char* e=std::getenv("THINCQQ_BSTAR_WMIN"); return (e&&e[0])?std::atof(e):0.8; }();   // DEFAULT [0.8,1.4] cap (wide range over-compresses at fine res)
    const double BSTAR3_WMAX = [](){ const char* e=std::getenv("THINCQQ_BSTAR_WMAX"); return (e&&e[0])?std::atof(e):1.4; }();   // DEFAULT [0.8,1.4] cap
    // beff[ci] = density(v==0) beta* (drives binc/pick branching); bstv3[v*N+ci] = per-variable beta*
    // (mirror of 2D reconstruct_bvd.hpp: density beta* MUST NOT be reused for u,v,w,p — s3-pervar rule).
    static std::vector<double> beff; static std::vector<char> binc; static std::vector<double> bstv3;   // H2 persistent
    if(BSTAR3){ beff.assign(N,beta_l); binc.assign(N,0); bstv3.assign((size_t)nvar*N,beta_l); }
    // 3D beta* method: DEFAULT = L2GN (Gauss-Newton on the L2 objective sum J^2, FD derivative, no grid)
    // — mirrors the finalized 2D S3. Opt into the NB-grid TBV-argmin with THINCQQ_BSTAR_GRID=1.
    const bool BSTAR3_GRID = std::getenv("THINCQQ_BSTAR_GRID")!=nullptr;
    // S3 option B (DEFAULT ON): beta* is SEARCHED with the cheap GAUSS closed-form (unchanged), but the
    // FINAL THINC(beta*) candidate D_star + face value are computed with the EXACT tanh (Newton cell-D +
    // quadrature face) — most-accurate reconstruction, GAUSS only as the beta*-search tool. Opt out to the
    // all-GAUSS recon (option A) with THINCQQ_BSTAR_GAUSSRECON=1.
    const bool BSTAR3_TANHRECON = (std::getenv("THINCQQ_BSTAR_GAUSSRECON")==nullptr);
    const bool PROF=u3_prof_on(); double _pt;
    // OPT1: fused P2-LSQ across all nvar (stream the big M operator table ONCE, not nvar x).
    // g_all layout = var-outermost: g_all[v*N*9 + ci*9 + i]. Byte-identical to per-var calls.
    if(PROF) _pt=u3_prof_ms();
    reconstruct3d_o2_coeffs_allvars(m,o2c,W,nvar,g_all);
    if(PROF) u3_prof().lsq += u3_prof_ms()-_pt;
    for(int v=0; v<nvar; ++v){
        const double* g = &g_all[(size_t)v*(size_t)N*9];   // this var's slice (replaces reconstruct3d_o2_coeffs per var)
        if(PROF) _pt=u3_prof_ms();
        // per-vertex bound: min/max of cell-averages over cells sharing each node (this var).
        if(VMLP){
            #pragma omp parallel for schedule(static)
            for(int p=0;p<NA.nn;++p){ int b=NA.off[p],e=NA.off[p+1];
                if(b==e){ vnmin[p]=0; vnmax[p]=0; continue; }
                double lo=W[(size_t)v*N+NA.cells[b]], hi=lo;
                for(int k=b+1;k<e;++k){ double q=W[(size_t)v*N+NA.cells[k]]; if(q<lo)lo=q; if(q>hi)hi=q; }
                vnmin[p]=lo; vnmax[p]=hi; }
        }
        // SMOOTH-candidate polynomial order. 2D's BVD smooth/MUSCL candidate is P1 (gradient-only,
        // vertex-MLP limited); 3D used P2 (grad+Hessian). The P2 Hessian term 0.5*dx^T H dx grows as
        // (centroid->face distance)^2 -> on a POINTY tet (face/vertex ~2-3x farther from the centroid
        // than on a compact hex) it overshoots, and the U2 spare un-limits it -> divergence (MUSCL-only
        // on tets blows up at step 10). P1 (HFAC=0) drops the Hessian from the SMOOTH candidate ONLY:
        // a linear plane can't swing past its vertex-MLP bound, so it stays bounded on any cell shape.
        // The THINC candidate keeps the full P2 interface (unchanged). This matches 2D. 2nd-order in
        // smooth regions instead of 3rd. Opt back to P2 with RECON_P2_SMOOTH.
        static const double HFAC = (std::getenv("RECON_P2_SMOOTH")!=nullptr) ? 1.0 : 0.0;
        // (1) per-cell: stencil bound, MLP phi, interface P + GAUSS cell-D
        #pragma omp parallel for schedule(dynamic,64)
        for(int ci=0; ci<N; ++ci){
            Cint st; double qbar=W[(size_t)v*N+ci];
            // FIX2: THINC normalization band = 2D S1's qmnc/qmxc (reconstruct_bvd.hpp:1246): min/max
            // of the per-node vertex-neighbour bounds over the cell's nodes (VMLP). cbar + the
            // interface gate + Q all use this SAME band (2D consistency). Fallback = old o2c.nb LSQ
            // stencil when VMLP is off (RECON_BJ_FACE, legacy).
            double qmin,qmax;
            if(VMLP){
                qmin=1e300; qmax=-1e300;
                for(int p:m.cell_nodes[ci]){ if(p<0||p>=NA.nn)continue; double a=vnmin[p],b=vnmax[p]; if(a<qmin)qmin=a; if(b>qmax)qmax=b; }
                if(qmin>qmax){ qmin=qbar; qmax=qbar; }
            } else {
                qmin=qbar; qmax=qbar;
                for(int k=0;k<o2c.max_nb;++k){int nb=o2c.nb[(size_t)ci*o2c.max_nb+k]; if(nb<0)continue;
                    double wn=W[(size_t)v*N+nb]; qmin=std::min(qmin,wn); qmax=std::max(qmax,wn);}
            }
            st.qmin=qmin; st.qmax=qmax;
            const double* gi=&g[(size_t)ci*9];
            double cx=m.cell_centers[3*ci],cy=m.cell_centers[3*ci+1],cz=m.cell_centers[3*ci+2];
            // limiter on the o2 gradient. DEFAULT = genuine vertex-MLP (Park-Yoon-Kim MLP-u):
            // check the o2 reconstruction AT EACH VERTEX against that vertex's neighbour-min/max
            // (multi-dimensional bound). Opt-out RECON_BJ_FACE = legacy face-center Barth-Jespersen.
            double phi=1.0;
            if(VMLP){
                // FIX1: 2D S1 smooth candidate (reconstruct2d.hpp:1100-1108). Per vertex p, project the
                // P1 gradient (HFAC=0) and take the MLP-u2 Venkatakrishnan limiter (MLP_U2>0) OR hard
                // Barth-Jespersen (K==0) on that vertex's neighbour min/max bound. phi = min over p.
                double eps2 = (U2K>0.0) ? std::pow(U2K*std::cbrt(2.0*m.cell_volumes[ci]),3) : 0.0;
                for(int p:m.cell_nodes[ci]){
                    double dx=m.nodes[3*p]-cx,dy=m.nodes[3*p+1]-cy,dz=m.nodes[3*p+2]-cz;
                    double dq=gi[0]*dx+gi[1]*dy+gi[2]*dz+HFAC*(0.5*gi[3]*dx*dx+0.5*gi[4]*dy*dy+0.5*gi[5]*dz*dz+gi[6]*dx*dy+gi[7]*dx*dz+gi[8]*dy*dz);
                    double pk = (U2K>0.0) ? u3_venk_phi(dq,qbar,vnmin[p],vnmax[p],eps2)
                                          : u3_bj_phi(dq,qbar,vnmin[p],vnmax[p]);
                    if(pk<phi)phi=pk; }
            } else {
                for(int fc:m.cell_faces[ci]){ double dx=m.face_centers[3*fc]-cx,dy=m.face_centers[3*fc+1]-cy,dz=m.face_centers[3*fc+2]-cz;
                    double dq=gi[0]*dx+gi[1]*dy+gi[2]*dz+HFAC*(0.5*gi[3]*dx*dx+0.5*gi[4]*dy*dy+0.5*gi[5]*dz*dz+gi[6]*dx*dy+gi[7]*dx*dz+gi[8]*dy*dz);
                    double allowed = dq>0?(qmax-qbar):(qmin-qbar);
                    double r = (std::fabs(dq)>1e-30)?allowed/dq:1.0; if(r<0)r=0; if(r<phi)phi=r; }
            }
            st.phi = phi<0?0:(phi>1?1:phi);
            if(ABVD3){   // one-sided vertex-LMP psi2 in [0,2] (P1 reach) for the adaptive candidate
                double p2=2.0;
                if(VMLP){
                    for(int p:m.cell_nodes[ci]){
                        double dx=m.nodes[3*p]-cx,dy=m.nodes[3*p+1]-cy,dz=m.nodes[3*p+2]-cz;
                        double proj=gi[0]*dx+gi[1]*dy+gi[2]*dz;
                        double allowed = proj>=0.0?(vnmax[p]-qbar):(qbar-vnmin[p]);
                        double pk=(std::fabs(proj)>1e-30)?std::max(allowed,0.0)/std::fabs(proj):2.0;
                        if(pk<p2)p2=pk; }
                } else {
                    for(int fc:m.cell_faces[ci]){
                        double dx=m.face_centers[3*fc]-cx,dy=m.face_centers[3*fc+1]-cy,dz=m.face_centers[3*fc+2]-cz;
                        double proj=gi[0]*dx+gi[1]*dy+gi[2]*dz;
                        double allowed = proj>=0.0?(qmax-qbar):(qbar-qmin);
                        double pk=(std::fabs(proj)>1e-30)?std::max(allowed,0.0)/std::fabs(proj):2.0;
                        if(pk<p2)p2=pk; }
                }
                st.phi2 = p2<0?0:(p2>2?2:p2);
            }
            // U2 smooth-extremum spare: sign-coherent per-axis curvature over the ring => smooth => φ=1
            // FIX1b: this is a 3D-ONLY hack absent in 2D — when MLP_U2>0 (S1) the venk relaxation IS
            // the smooth-extremum treatment, so skip the spare entirely (U2K<=0.0) to match 2D. Kept
            // active for the legacy hard-BJ path (MLP_U2 unset).
            if(LIM_U2 && U2K<=0.0 && st.phi<1.0){
                double du2=std::cbrt(m.cell_volumes[ci]); du2=du2*du2*du2;
                bool smooth=(qmax-qmin)<du2;
                if(!smooth){ smooth=true;
                    for(int ax=0;ax<3 && smooth;++ax){ int c=3+ax; double Hlo=gi[c],Hhi=gi[c];
                        for(int k=0;k<o2c.max_nb;++k){int nb=o2c.nb[(size_t)ci*o2c.max_nb+k]; if(nb<0)continue;
                            double Hn=g[(size_t)nb*9+c]; if(Hn<Hlo)Hlo=Hn; if(Hn>Hhi)Hhi=Hn;}
                        if(Hhi*Hlo< -du2) smooth=false; }
                }
                // U2 tests ONLY the axis-diagonal Hessian (gi[3..5]); on a TET whose interface normal
                // lies along a cube diagonal ~35% of the curvature is OFF-diagonal (gi[6..8]) that U2
                // never inspects, so it mislabels the (smeared) diagonal interface "smooth" and un-limits
                // the o2-quad -> overshoot. Gate: if the off-diagonal Hessian energy is comparable to the
                // diagonal, U2's premise is unreliable -> do NOT spare. Axis-aligned smooth extrema (small
                // off-diagonal) are unaffected -> accuracy preserved. Opt-out U2_NOGATE.
                static const bool U2_NOGATE = std::getenv("U2_NOGATE") != nullptr;
                if(smooth && !U2_NOGATE){
                    double hd=gi[3]*gi[3]+gi[4]*gi[4]+gi[5]*gi[5];
                    double ho=gi[6]*gi[6]+gi[7]*gi[7]+gi[8]*gi[8];
                    if(ho > 0.25*hd) smooth=false;
                }
                if(smooth) st.phi=1.0;
            }
            // interface cell? cbar interior + nonzero gradient (+ a-priori: only where φ<APHI,
            // i.e. the MLP limiter activated → genuine discontinuity; smooth φ≈1 cells skip THINC)
            double rng=qmax-qmin, cbar=(rng>1e-14)?(qbar-qmin)/rng:0.5;
            // FIX4: interface-cell cbar gate 1e-4 -> 1e-6 (2D reconstruct_bvd.hpp:1249 value).
            if(rng>1e-14 && cbar>1e-6 && cbar<1.0-1e-6 && (!APRIORI || st.phi<APHI) && u3_build_P(gi, st.P)){
                // THINC sharpness scale H. 2D uses the HYDRAULIC DIAMETER H=4V/perimeter (shape-aware:
                // shrinks for high-aspect cells). The 3D port used H=cbrt(V) (pure volume cube-root, NO
                // directional info): for a cube it equals the edge (matches 2D), but for a pointy/elongated
                // TET cbrt(V) is ~1.8x too SMALL vs the real extent -> kk=beta/H too large -> THINC sigmoid
                // TOO SHARP on tets -> face overshoot -> divergence (Langseth tet, not hex). FIX = the exact
                // 3D analog HYDRAULIC DIAMETER H=6V/A_surf (2D "perimeter"->3D "surface area", 4->6);
                // == edge for a cube (hex byte-unchanged), larger for a tet. Opt-out RECON_H_CBRT.
                static const bool H_CBRT = std::getenv("RECON_H_CBRT") != nullptr;
                double V=m.cell_volumes[ci];
                double H;
                if(H_CBRT){ H=std::cbrt(V); }
                else { double A=0; for(int fc:m.cell_faces[ci]) A+=m.face_areas[fc]; H=(A>1e-300)?6.0*V/A:std::cbrt(V); }
                st.kk_l=beta_l/H; st.kk_s=beta_s/H;
                D3Poly& P=st.P;
                // moments of the interface poly are beta-INDEPENDENT -> compute once, reuse for both kk
                double M1,M2,M3=0.0;
                if(USE_GMOM) u3_M12(&GM.cellm[(size_t)ci*35], P.c, M1, M2);     // closed-form (precomputed table)
                else {                                                          // on-the-fly (skew / opt-out)
                    M1=u3_int_cell(m,ci,[&](double x,double y,double z){return u3_Peval(P,x-cx,y-cy,z-cz);})/V;
                    M2=u3_int_cell(m,ci,[&](double x,double y,double z){double p=u3_Peval(P,x-cx,y-cy,z-cz);return p*p;})/V;
                    M3= deg3t_gauss_skew()? u3_int_cell(m,ci,[&](double x,double y,double z){double p=u3_Peval(P,x-cx,y-cy,z-cz);return p*p*p;})/V : 0.0;
                }
                double Q=2.0*cbar-1.0;
                if(USE_TANH){
                    thread_local std::vector<double> tl_Pv, tl_wt; // one per OMP thread; grows once, never reallocates after warmup
                    u3_cell_pvals(m,ci,P,tl_Pv,tl_wt);
                    // PURE Xie2017 THINC/QQ: cold Newton start (D0=0). NO GAUSS warm-start — the tanh
                    // baseline must be the original algorithm, independent of GAUSS, for a fair speed
                    // comparison (opt-in THINCQQ_TANH_WARM to restore the probit warm-start).
                    static const bool WARM = std::getenv("THINCQQ_TANH_WARM")!=nullptr;
                    double D0l=WARM?deg3t3d_cellD_fromM_gauss(M1,M2,st.kk_l,Q):0.0;
                    double D0s=WARM?deg3t3d_cellD_fromM_gauss(M1,M2,st.kk_s,Q):0.0;
                    st.D_l=u3_tanh_cellD(tl_Pv,tl_wt,st.kk_l,Q,D0l); st.D_s=u3_tanh_cellD(tl_Pv,tl_wt,st.kk_s,Q,D0s);
                } else if(deg3t_gauss_skew()){ st.D_l=deg3t3d_cellD_fromM_gaussS(M1,M2,M3,st.kk_l,Q); st.D_s=deg3t3d_cellD_fromM_gaussS(M1,M2,M3,st.kk_s,Q); }
                else { st.D_l=deg3t3d_cellD_fromM_gauss(M1,M2,st.kk_l,Q); st.D_s=deg3t3d_cellD_fromM_gauss(M1,M2,st.kk_s,Q); }
                st.M1=M1; st.M2=M2;   // stored for the beta* search / D_star (paper3); D_star set per-var in the search below
                st.hasint=1;
            }
            ci_st[ci]=st;
        }
        if(PROF){ u3_prof().celld += u3_prof_ms()-_pt; _pt=u3_prof_ms(); }
        // (1b) PAPER 3 exact beta*: per-cell TBV(beta) grid-min over [beta_s,beta_l], selected on
        // density (v==0), reused for all vars. GAUSS closed-form D(beta) (deg3t3d_cellD_fromM_gauss)
        // + closed-form face avg (deg3t3d_face_avg_gauss) => each of the NB grid evals is O(faces),
        // no Newton. Sets beff[ci]=beta*, binc[ci]=interior-flag, and ci_st[ci].D_star at beta*.
        if(BSTAR3){   // PAPER3 PER-VARIABLE beta* (runs every v; density v==0 also drives beff/binc/pick)
            const int NB = BSTAR3_WIDE?16:9;
            const double blo = BSTAR3_WIDE?BSTAR3_WMIN:beta_s, bhi = BSTAR3_WIDE?BSTAR3_WMAX:beta_l;
            #pragma omp parallel for schedule(dynamic,64)
            for(int ci=0; ci<N; ++ci){
                Cint& st=ci_st[ci]; if(!st.hasint) continue;
                double Hci=(st.kk_l>0.0)?beta_l/st.kk_l:std::cbrt(m.cell_volumes[ci]);
                double cx=m.cell_centers[3*ci],cy=m.cell_centers[3*ci+1],cz=m.cell_centers[3*ci+2];
                double qbar=W[(size_t)v*N+ci], rng=st.qmax-st.qmin;
                double Qc=(rng>1e-14)?(2.0*(qbar-st.qmin)/rng-1.0):0.0;
                // interface-interface faces w/ beta-INDEPENDENT owner/neighbour moments (this var)
                struct FM{ int nb; double F1o,F2o,F1n,F2n; double Hn,rn,qmn_n,qmx_n; double M1n,M2n,Qn; };
                FM fm[32]; int nf=0;
                for(int fc:m.cell_faces[ci]){ if(nf>=32) break;
                    int o=m.face_owner[fc], nn=m.face_neighbour[fc], nb=(o==ci)?nn:o; if(nb<0) continue;
                    const Cint& sn=ci_st[nb]; if(!sn.hasint) continue;
                    const D3Poly& Po=st.P; const D3Poly& Pn=sn.P;
                    double F1o,F2o,F1n,F2n;
                    if(USE_GMOM){   // closed-form owner/neighbour face moments from GMOM (no quadrature)
                        int eo=u3_face_lidx_tab(FL,m,ci,fc); u3_M12(&GM.facem[GM.face_off[ci]+(size_t)eo*35], Po.c, F1o, F2o);
                        int en=u3_face_lidx_tab(FL,m,nb,fc); u3_M12(&GM.facem[GM.face_off[nb]+(size_t)en*35], Pn.c, F1n, F2n);
                    } else {
                        double invA=1.0/u3_face_area_fan(m,fc);   // self-consistent fan area (see u3_face_area_fan)
                        F1o=u3_int_face(m,fc,[&](double x,double y,double z){return u3_Peval(Po,x-cx,y-cy,z-cz);})*invA;
                        F2o=u3_int_face(m,fc,[&](double x,double y,double z){double p=u3_Peval(Po,x-cx,y-cy,z-cz);return p*p;})*invA;
                        double nx=m.cell_centers[3*nb],ny=m.cell_centers[3*nb+1],nz=m.cell_centers[3*nb+2];
                        F1n=u3_int_face(m,fc,[&](double x,double y,double z){return u3_Peval(Pn,x-nx,y-ny,z-nz);})*invA;
                        F2n=u3_int_face(m,fc,[&](double x,double y,double z){double p=u3_Peval(Pn,x-nx,y-ny,z-nz);return p*p;})*invA;
                    }
                    double rn=sn.qmax-sn.qmin, qbn=W[(size_t)v*N+nb];
                    double Qn=(rn>1e-14)?(2.0*(qbn-sn.qmin)/rn-1.0):0.0;
                    fm[nf]={nb,F1o,F2o,F1n,F2n,(sn.kk_l>0.0)?beta_l/sn.kk_l:std::cbrt(m.cell_volumes[nb]),rn,sn.qmin,sn.qmax,sn.M1,sn.M2,Qn}; nf++;
                }
                double bstar;
                if(nf==0){ bstar=beta_l; }
                else {
                auto faceval=[&](double qmn,double r,double F1,double F2,double D,double kk)->double{
                    D3FaceMom fmm{F1,F2,0.0,0.0}; return qmn+0.5*r*(1.0+deg3t3d_face_avg_gauss(fmm,D,kk)); };
                if(BSTAR3_GRID){   // NB-grid TBV-argmin (legacy/reference)
                    double TB[16]; int gi;
                    for(gi=0; gi<NB; ++gi){ double beta=blo+(bhi-blo)*gi/(NB-1);
                        double kko=beta/Hci; double Do=deg3t3d_cellD_fromM_gauss(st.M1,st.M2,kko,Qc); double tb=0.0;
                        for(int j=0;j<nf;++j){ const FM& F=fm[j];
                            double kkn=beta/F.Hn; double Dn=deg3t3d_cellD_fromM_gauss(F.M1n,F.M2n,kkn,F.Qn);
                            tb+=std::fabs(faceval(st.qmin,rng,F.F1o,F.F2o,Do,kko)-faceval(F.qmn_n,F.rn,F.F1n,F.F2n,Dn,kkn)); }
                        TB[gi]=tb; }
                    int imn=0; for(gi=1;gi<NB;++gi) if(TB[gi]<TB[imn]) imn=gi;
                    if(BSTAR3_MAXB){ double thr=BSTAR3_TOL*TB[imn]; int ib=imn; for(gi=NB-1;gi>=imn;--gi) if(TB[gi]<=thr){ib=gi;break;}
                        bstar=blo+(bhi-blo)*ib/(NB-1);
                    } else bstar=blo+(bhi-blo)*imn/(NB-1);
                } else {   // DEFAULT L2GN: Gauss-Newton on min sum J^2, FD derivative, no grid (mirror of 2D S3)
                    double b=beta_l, db=1e-4*(bhi-blo+1e-30);
                    for(int it=0;it<4;++it){
                        double kko=b/Hci, Do=deg3t3d_cellD_fromM_gauss(st.M1,st.M2,kko,Qc);
                        double kko2=(b+db)/Hci, Do2=deg3t3d_cellD_fromM_gauss(st.M1,st.M2,kko2,Qc);
                        double num=0,den=0;
                        for(int j=0;j<nf;++j){ const FM& F=fm[j];
                            double kkn=b/F.Hn, Dn=deg3t3d_cellD_fromM_gauss(F.M1n,F.M2n,kkn,F.Qn);
                            double J=faceval(st.qmin,rng,F.F1o,F.F2o,Do,kko)-faceval(F.qmn_n,F.rn,F.F1n,F.F2n,Dn,kkn);
                            double kkn2=(b+db)/F.Hn, Dn2=deg3t3d_cellD_fromM_gauss(F.M1n,F.M2n,kkn2,F.Qn);
                            double J2=faceval(st.qmin,rng,F.F1o,F.F2o,Do2,kko2)-faceval(F.qmn_n,F.rn,F.F1n,F.F2n,Dn2,kkn2);
                            double Jp=(J2-J)/db; num+=J*Jp; den+=Jp*Jp; }
                        if(den<1e-30) break; double step=num/den; b-=step;
                        if(b<blo)b=blo; else if(b>bhi)b=bhi;
                        if(std::fabs(step)<1e-6) break; }
                    if(b<blo)b=blo; else if(b>bhi)b=bhi; bstar=b;   // lower bound = blo(=WMIN, ln3 by default)
                }
                }
                bstv3[(size_t)v*N+ci]=bstar;   // per-var beta* -> THINC(beta*) reconstruction value
                if(BSTAR3_TANHRECON){   // option B: EXACT tanh cell-D (Newton) at beta* (GAUSS was only the search)
                    thread_local std::vector<double> cpv,cwt; u3_cell_pvals(m,ci,st.P,cpv,cwt);
                    st.D_star=u3_tanh_cellD(cpv,cwt,bstar/Hci,Qc,deg3t3d_cellD_fromM_gauss(st.M1,st.M2,bstar/Hci,Qc));
                } else st.D_star=deg3t3d_cellD_fromM_gauss(st.M1,st.M2,bstar/Hci,Qc);
                if(v==0){ beff[ci]=bstar;   // density beta* drives binc/pick branching
                    if(BSTAR3_WIDE) binc[ci]=1;
                    else { int ibsel=(int)std::lround((bstar-beta_s)/(beta_l-beta_s)*(NB-1)); binc[ci]=(ibsel>0 && ibsel<NB-1)?1:0; }
                }
            }
            static const bool DIAG=std::getenv("BOPT_DIAG")!=nullptr;
            if(DIAG && v==0){ double bmn=1e30,bmx=-1e30,bs=0; int nb=0,nin=0;
                for(int ci=0;ci<N;++ci){ if(!ci_st[ci].hasint)continue; double b=beff[ci]; if(b<bmn)bmn=b; if(b>bmx)bmx=b; bs+=b; nb++; if(binc[ci])nin++; }
                std::fprintf(stderr,"3D BSTAR_EXACT beta*(density,L2GN): min=%.2f mean=%.2f max=%.2f interior-frac=%.2f (n_if=%d)\n",bmn,bs/(nb>0?nb:1),bmx,(double)nin/(nb>0?nb:1),nb); }
        }
        // (2) BVD with the CORRECT face-jump metric (3-stage, mirror of 2D reconstruct_cheng3).
        //   [FIX 2026-06-30] the old code used TBV = Σ|candidate − neighbour CELL-AVERAGE|, which is
        //   NOT boundary variation: it made the candidate pick non-monotone in beta + owner/neighbour
        //   asymmetric -> beta-chaotic enstrophy. Correct BVD = Σ|owner_RECON − neighbour_RECON| at faces.
        // helper: the 3 candidate face values {o2-quad, THINC_l, THINC_s} of cell ci at face fc.
        // H1 (perf, result-preserving): wantLS = the beta_l/beta_s (q1,q2) candidates are actually needed.
        // In WIDE the pick for a binc==1 cell uses only {smooth, THINC(beta*)} (L776), so for a face whose
        // BOTH cells are binc==1 the q1/q2 candidates (and their GAUSS moment assembly) are dead -> skip.
        auto cellFaceCands=[&](int ci,int fc,bool wantLS,bool wantBS,double& q0,double& q1,double& q2,double& q3){
            const Cint& st=ci_st[ci]; double qbar=W[(size_t)v*N+ci];
            const double* gi=&g[(size_t)ci*9];
            double cx=m.cell_centers[3*ci],cy=m.cell_centers[3*ci+1],cz=m.cell_centers[3*ci+2];
            double dx=m.face_centers[3*fc]-cx,dy=m.face_centers[3*fc+1]-cy,dz=m.face_centers[3*fc+2]-cz;
            double dq=gi[0]*dx+gi[1]*dy+gi[2]*dz+HFAC*(0.5*gi[3]*dx*dx+0.5*gi[4]*dy*dy+0.5*gi[5]*dz*dz+gi[6]*dx*dy+gi[7]*dx*dz+gi[8]*dy*dz);
            q0=qbar+st.phi*dq; q1=qbar; q2=qbar; q3=qbar;               // smooth candidate (P1 default, MLP-limited; P2 if RECON_P2_SMOOTH)
            if(st.hasint){ const D3Poly&P=st.P; double fal=0,fas=0;
                if(USE_TANH){
                    if(wantLS){ thread_local std::vector<double> pv,wt; u3_face_pvals(m,fc,P,cx,cy,cz,pv,wt);
                        fal=u3_tanh_face_avg(pv,wt,st.kk_l,st.D_l); fas=u3_tanh_face_avg(pv,wt,st.kk_s,st.D_s); }
                } else {   // GAUSS: closed-form face moments <P>,<P^2> from precomputed GMOM (no quadrature)
                    // fm is used by fal/fas (wantLS) and by the GAUSS-recon beta* face (option A, !TANHRECON, only when wantBS).
                    bool needFM = wantLS || (BSTAR3 && wantBS && !BSTAR3_TANHRECON);
                    D3FaceMom fm{0,0,0,0};
                    if(needFM){ double F1,F2;
                        // Afan = SELF-CONSISTENT fan area of fc (see u3_face_area_fan); NOT m.face_areas[fc],
                        // which is a v0-fan and differs on warped quads. Computed only on the paths that need it.
                        if(USE_GMOM){ int e=u3_face_lidx_tab(FL,m,ci,fc); u3_M12(&GM.facem[GM.face_off[ci]+(size_t)e*35], P.c, F1, F2); }
                        else { double Afan=u3_face_area_fan(m,fc);
                               F1=u3_int_face(m,fc,[&](double x,double y,double z){return u3_Peval(P,x-cx,y-cy,z-cz);})/Afan;
                               F2=u3_int_face(m,fc,[&](double x,double y,double z){double p=u3_Peval(P,x-cx,y-cy,z-cz);return p*p;})/Afan; }
                        double F3= deg3t_gauss_skew()? u3_int_face(m,fc,[&](double x,double y,double z){double p=u3_Peval(P,x-cx,y-cy,z-cz);return p*p*p;})/u3_face_area_fan(m,fc) : 0.0;
                        fm=D3FaceMom{F1,F2,F3,u3_Peval(P,dx,dy,dz)}; }
                    if(wantLS){
                        fal= deg3t_gauss_skew()? deg3t3d_face_avg_gaussS(fm,st.D_l,st.kk_l):deg3t3d_face_avg_gauss(fm,st.D_l,st.kk_l);
                        fas= deg3t_gauss_skew()? deg3t3d_face_avg_gaussS(fm,st.D_s,st.kk_s):deg3t3d_face_avg_gauss(fm,st.D_s,st.kk_s);
                    }
                    if(BSTAR3 && wantBS){   // PAPER 3 per-var beta* candidate face value (kk=beta*_v/H); skipped when no adjacent cell picks {beta*}
                        double kks=bstv3[(size_t)v*N+ci]*st.kk_l/beta_l;   // = beta*_v/H (kk_l = beta_l/H)
                        double fst;
                        if(BSTAR3_TANHRECON){   // option B: EXACT tanh face avg at beta* (quadrature)
                            thread_local std::vector<double> pv,wt; u3_face_pvals(m,fc,P,cx,cy,cz,pv,wt);
                            fst=u3_tanh_face_avg(pv,wt,kks,st.D_star);
                        } else fst=deg3t3d_face_avg_gauss(fm,st.D_star,kks);
                        q3=st.qmin+0.5*(st.qmax-st.qmin)*(1.0+fst);
                    }
                }
                // NO face-value clamp (Xie2017-faithful, user 2026-07-03): both face averages are
                // already strictly bounded in (-1,1) -- tanh: convex combination of tanh values;
                // GAUSS: tanh(m1/sqrt(1+cv)) -- so q1,q2 lie strictly inside (qmin,qmax) by
                // construction and the old clamp was a mathematically inactive safety line.
                if(wantLS){
                    q1=st.qmin+0.5*(st.qmax-st.qmin)*(1.0+fal);
                    q2=st.qmin+0.5*(st.qmax-st.qmin)*(1.0+fas);
                }
            }
        };
        // (2a) per-FACE: owner-side & neighbour-side candidate face values.
        #pragma omp parallel for schedule(dynamic,64)
        for(int f=0; f<Nf; ++f){
            int o=m.face_owner[f], nb=m.face_neighbour[f];
            // H1 (result-preserving): binc==1 cell picks {smooth,beta*} (beta_l/beta_s dead); binc==0 cell picks
            // {smooth,beta_l,beta_s} (beta* dead). Compute each candidate iff an adjacent cell consults it.
            bool wantLS = !BSTAR3 || binc[o]==0 || (nb>=0 && binc[nb]==0);
            bool wantBS = BSTAR3 && (binc[o]==1 || (nb>=0 && binc[nb]==1));
            double a0,a1,a2,a3; cellFaceCands(o,f,wantLS,wantBS,a0,a1,a2,a3);
            qoF[(size_t)f*4+0]=a0; qoF[(size_t)f*4+1]=a1; qoF[(size_t)f*4+2]=a2; qoF[(size_t)f*4+3]=a3;   // OPT3 cand-innermost
            if(nb>=0){ double b0,b1,b2,b3; cellFaceCands(nb,f,wantLS,wantBS,b0,b1,b2,b3);
                qnF[(size_t)f*4+0]=b0; qnF[(size_t)f*4+1]=b1; qnF[(size_t)f*4+2]=b2; qnF[(size_t)f*4+3]=b3;
            } else { qnF[(size_t)f*4+0]=a0; qnF[(size_t)f*4+1]=a1; qnF[(size_t)f*4+2]=a2; qnF[(size_t)f*4+3]=a3; } // boundary -> zero jump
        }
        // (2a') ABVD: overwrite candidate-0 (smooth slot) face values with the adaptive
        // TVD-BVD interval pick + vertex-LMP clamp. Boundary faces keep the MLP-P1 value.
        if(ABVD3){
            const bool eul5 = (nvar>=5);   // primitive (rho,u,v,w,p): face-normal velocity available
            #pragma omp parallel for schedule(static)
            for(int f=0; f<Nf; ++f){
                int o=m.face_owner[f], nb=m.face_neighbour[f];
                if(nb<0) continue;
                double qi=W[(size_t)v*N+o], qj=W[(size_t)v*N+nb], dq=qj-qi;
                if(std::fabs(dq)<1e-13*(std::fabs(qi)+std::fabs(qj)+1e-100)){
                    qoF[(size_t)f*4+0]=qi; qnF[(size_t)f*4+0]=qj; continue; }
                const double* gi=&g[(size_t)o*9]; const double* gj=&g[(size_t)nb*9];
                double cox=m.cell_centers[3*o],coy=m.cell_centers[3*o+1],coz=m.cell_centers[3*o+2];
                double cnx=m.cell_centers[3*nb],cny=m.cell_centers[3*nb+1],cnz=m.cell_centers[3*nb+2];
                double dxc=cnx-cox,dyc=cny-coy,dzc=cnz-coz;
                double fx=m.face_centers[3*f],fy=m.face_centers[3*f+1],fz=m.face_centers[3*f+2];
                // upwind speed (Eq.26 endpoint): scalar via face_a, Euler via face-normal velocity
                double af=0.0; bool have_a=false;
                if(face_a){ af=face_a[f]; have_a=true; }
                else if(eul5){ af=0.5*((W[(size_t)1*N+o]+W[(size_t)1*N+nb])*m.face_normals[3*f]
                                      +(W[(size_t)2*N+o]+W[(size_t)2*N+nb])*m.face_normals[3*f+1]
                                      +(W[(size_t)3*N+o]+W[(size_t)3*N+nb])*m.face_normals[3*f+2]); have_a=true; }
                // supersonic-face guard (see 2D reconstruct_abvd_linear note): where |u.n|>c the
                // zero-BV central state carries physically-forbidden upstream information; keep
                // one-sided TVD values there. Opt-out BVD_ABVD_NOSS.
                static const bool SSON3 = std::getenv("BVD_ABVD_SS") != nullptr;   // guard OFF by default (user 2026-07-03); opt-in BVD_ABVD_SS
                bool ssonic = false;
                if(eul5 && SSON3){
                    double rb=0.5*(W[(size_t)0*N+o]+W[(size_t)0*N+nb]);
                    double pb=0.5*(W[(size_t)(nvar-1)*N+o]+W[(size_t)(nvar-1)*N+nb]);
                    if(rb>0.0 && pb>0.0){ double cf=std::sqrt(1.4*pb/rb); if(std::fabs(af)>cf) ssonic=true; }
                }
                // vertex-LMP profile reach at the face (P1)
                double dpL=ci_st[o].phi2 *(gi[0]*(fx-cox)+gi[1]*(fy-coy)+gi[2]*(fz-coz));
                double dpR=ci_st[nb].phi2*(gj[0]*(fx-cnx)+gj[1]*(fy-cny)+gj[2]*(fz-cnz));
                auto capv=[&](double q0,double t,double dprof)->double{
                    double d=t-q0, lo=std::min(0.0,dprof), hi=std::max(0.0,dprof);
                    return q0 + (d<lo?lo:(d>hi?hi:d)); };
                // projected slope ratios
                double gLd=gi[0]*dxc+gi[1]*dyc+gi[2]*dzc;
                double gRd=-(gj[0]*dxc+gj[1]*dyc+gj[2]*dzc);
                double rL=2.0*gLd/dq-1.0, rR=2.0*gRd/(-dq)-1.0;
                double sbL=a3_sb(rL), sbR=a3_sb(rR);
                if(sbL+sbR<2.0 || ssonic){   // pattern (a): zero-BV unreachable (or supersonic guard)
                    double pL=A3_SAFE?a3_vl(rL):sbL, pR=A3_SAFE?a3_vl(rR):sbR;
                    qoF[(size_t)f*4+0]=capv(qi,qi+0.5*pL*dq,dpL);
                    qnF[(size_t)f*4+0]=capv(qj,qj-0.5*pR*dq,dpR);
                    continue; }
                double qLmm=qi+0.5*a3_mm(rL)*dq, qLsb=qi+0.5*sbL*dq;
                double qRmm=qj-0.5*a3_mm(rR)*dq, qRsb=qj-0.5*sbR*dq;
                double B=(qLsb-qRmm)*(qLsb-qRsb), Cc=(qRsb-qLmm)*(qRsb-qLsb);
                if(B>=0.0 && Cc>=0.0){   // pattern (e): intervals crossed
                    if(A3_EMM){ qoF[(size_t)f*4+0]=capv(qi,qLmm,dpL); qnF[(size_t)f*4+0]=capv(qj,qRmm,dpR); }
                    else { double qs=0.5*(qi+qj);
                        qoF[(size_t)f*4+0]=capv(qi,qs,dpL); qnF[(size_t)f*4+0]=capv(qj,qs,dpR); }
                    continue; }
                double e0,e1;
                if(B<=0.0 && Cc<=0.0){ e0=qLsb; e1=qRsb; }        // (b)
                else if(B<=0.0)      { e0=qLmm; e1=qLsb; }        // (c)
                else                 { e0=qRmm; e1=qRsb; }        // (d)
                double xmn=std::min(e0,e1), xmx=std::max(e0,e1), qs;
                if(have_a) qs=(af*(qi-qj)<0.0)?xmx:xmn;           // Eq.26 upwind endpoint
                else       qs=0.5*(xmn+xmx);                      // no velocity info (avoid: NOT TVD)
                qoF[(size_t)f*4+0]=capv(qi,qs,dpL);
                qnF[(size_t)f*4+0]=capv(qj,qs,dpR);
            }
        }
        // (2b) per-cell min-TBV over Σ_faces |qoF − qnF| (true face-jump; symmetric for both cells).
        #pragma omp parallel for schedule(dynamic,64)
        for(int ci=0; ci<N; ++ci){
            const Cint& st=ci_st[ci]; double t0=0,t1=0,t2=0,t3=0;
            for(int fc:m.cell_faces[ci]){
                t0+=std::fabs(qoF[(size_t)fc*4+0]-qnF[(size_t)fc*4+0]);
                t1+=std::fabs(qoF[(size_t)fc*4+1]-qnF[(size_t)fc*4+1]);
                t2+=std::fabs(qoF[(size_t)fc*4+2]-qnF[(size_t)fc*4+2]);
                if(BSTAR3) t3+=std::fabs(qoF[(size_t)fc*4+3]-qnF[(size_t)fc*4+3]);
            }
            int pick=0; double best=t0;                          // 0=o2-quad (smooth)
            if(BSTAR3){   // PAPER 3: interior beta* -> {smooth, THINC(beta*)}; else fallback {smooth,THINC_l,THINC_s}
                if(st.hasint){ if(binc[ci]){ if(t3<best){best=t3;pick=3;} }
                               else { if(t1<best){best=t1;pick=1;} if(t2<best){best=t2;pick=2;} } }
            } else if(st.hasint){ if(t1<best){best=t1;pick=1;} if(t2<best){best=t2;pick=2;} }
            pickv[ci]=pick;
        }
        // BVD_CANDFLAG: export DENSITY (v=0) per-cell candidate slot for the paper diagnostic.
        // 3D pickv convention already matches 2D: 0=o2-quad(MUSCL slot),1=THINC_l(beta_l),
        // 2=THINC_s(beta_s),3=THINC(beta*). Overwrites each call -> LAST recon (final-time) wins.
        if(v==0){ static const bool CANDFLAG3 = std::getenv("BVD_CANDFLAG")!=nullptr;
            if(CANDFLAG3){ auto& cf=cfd::bvd_cand_flag(); cf.assign((size_t)N,-1);
                for(int ci=0;ci<N;++ci) cf[ci]=(signed char)pickv[ci]; } }
        // (2c) assemble (face loop, race-free): owner->WL (its pick, owner-side), neighbour->WR (its pick, neighbour-side).
        //   shockcell -> first-order (qbar) on that cell's side.
        #pragma omp parallel for schedule(dynamic,64)
        for(int f=0; f<Nf; ++f){
            int o=m.face_owner[f], nb=m.face_neighbour[f];
            bool sho=!shockcell.empty() && shockcell[o];
            WL[(size_t)v*Nf+f] = sho? W[(size_t)v*N+o] : qoF[(size_t)f*4+pickv[o]];
            if(nb>=0){ bool shn=!shockcell.empty() && shockcell[nb];
                WR[(size_t)v*Nf+f] = shn? W[(size_t)v*N+nb] : qnF[(size_t)f*4+pickv[nb]]; }
        }
        if(PROF) u3_prof().facebvd += u3_prof_ms()-_pt;
    }
}

} // namespace cfd
