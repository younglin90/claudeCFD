// apps/thinc_cellshape_bench.cpp — 3D CELL-SHAPE unit testbed for the GAUSS THINC.
//
// THE NOVELTY TESTBED for the GAUSS closed-form interface reconstruction. Isolates the
// kernel  "QQ P2 polynomial input -> sigmoid cell-D (cell-average constraint) -> face
// integral"  and runs it on arbitrary 3D cells (tetra/hexa/prism/pyramid/octa-poly,
// plus skewed non-ortho variants), comparing:
//   (R) tanh-THINC  : the standard. cell-D by NEWTON, face & cell integrals by numerical
//                     QUADRATURE of the transcendental tanh. shape-dependent, expensive.
//   (G) GAUSS       : probit-identity closed form. The Gaussian-moment identity
//                     <tanh(kk P + D)>  ~=  tanh( (kk<P>+D) / sqrt(1 + c*kk^2 Var(P)) ),
//                     c = pi/2, makes the cell-average constraint invert in CLOSED FORM
//                       D = atanh(Q) sqrt(1 + c v) - kk<P>,   v = kk^2(<P^2>-<P>^2)
//                     (no Newton) and the face average a single tanh of the face moments
//                     (no quadrature). Uses ONLY the first two polynomial moments <P>,<P^2>.
// The closed form (reconstruct3d_bvd_core.hpp: deg3t3d_cellD_fromM_gauss / _face_avg_gauss)
// is SHAPE-AGNOSTIC given <P>,<P^2>; only those two moments depend on the cell, computed
// here for arbitrary polyhedra by EXACT simplex integration (Duffy-Gauss, degree-exact).
// Reports per cell: GAUSS face-average ERROR vs the fine-tanh truth + cell-average
// CONSERVATION error (does the closed-form D reproduce the target cell value under the
// TRUE tanh) + WALL time and speed-up vs the Newton+quadrature tanh-THINC.
#include "cfd/reconstruct3d_bvd_core.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <string>
#include <functional>

using namespace cfd;
using V3 = std::array<double,3>;
using Fn = std::function<double(const V3&)>;

static void gl01(int n, std::vector<double>& x, std::vector<double>& w){
    x.resize(n); w.resize(n);
    for(int i=0;i<n;++i){ double z=std::cos(M_PI*(i+0.75)/(n+0.5)),z1,pp;
        do{ double p1=1,p2=0;
            for(int j=0;j<n;++j){ double p3=p2;p2=p1; p1=((2*j+1)*z*p2-j*p3)/(j+1);}
            pp=n*(z*p1-p2)/(z*z-1.0); z1=z; z=z1-p1/pp;
        }while(std::fabs(z-z1)>1e-15);
        x[i]=0.5*(1.0-z); w[i]=1.0/((1.0-z*z)*pp*pp); }
}

struct Cell{ std::vector<V3> V; std::vector<std::vector<int>> F; std::string name; };
static V3 sub(const V3&a,const V3&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static V3 cross(const V3&a,const V3&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static double dot(const V3&a,const V3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static double tetVol(const V3&a,const V3&b,const V3&c,const V3&d){
    return std::fabs(dot(sub(b,a),cross(sub(c,a),sub(d,a))))/6.0;}

static void faceGeom(const Cell&C,const std::vector<int>&f,V3&cen,double&area){
    V3 v0=C.V[f[0]],acc{0,0,0}; double A=0;
    for(size_t i=1;i+1<f.size();++i){ V3 a=v0,b=C.V[f[i]],c=C.V[f[i+1]];
        V3 cr=cross(sub(b,a),sub(c,a)); double ta=0.5*std::sqrt(dot(cr,cr));
        acc[0]+=ta*(a[0]+b[0]+c[0])/3;acc[1]+=ta*(a[1]+b[1]+c[1])/3;acc[2]+=ta*(a[2]+b[2]+c[2])/3;A+=ta;}
    cen={acc[0]/A,acc[1]/A,acc[2]/A}; area=A;
}
static void cellGeom(const Cell&C,V3&cen,double&vol){
    V3 rp{0,0,0}; for(auto&v:C.V){rp[0]+=v[0];rp[1]+=v[1];rp[2]+=v[2];}
    rp[0]/=C.V.size();rp[1]/=C.V.size();rp[2]/=C.V.size();
    V3 acc{0,0,0}; double Vt=0;
    for(auto&f:C.F)for(size_t i=1;i+1<f.size();++i){
        V3 a=C.V[f[0]],b=C.V[f[i]],c=C.V[f[i+1]]; double tv=tetVol(rp,a,b,c);
        acc[0]+=tv*(rp[0]+a[0]+b[0]+c[0])/4;acc[1]+=tv*(rp[1]+a[1]+b[1]+c[1])/4;acc[2]+=tv*(rp[2]+a[2]+b[2]+c[2])/4;Vt+=tv;}
    cen={acc[0]/Vt,acc[1]/Vt,acc[2]/Vt}; vol=Vt;
}

static double intTri(const V3&a,const V3&b,const V3&c,const std::vector<double>&x,const std::vector<double>&w,const Fn&f){
    int n=x.size(); V3 cr=cross(sub(b,a),sub(c,a)); double area=0.5*std::sqrt(dot(cr,cr)),s=0;
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){ double xi=x[i],eta=x[j],L1=xi,L2=(1-xi)*eta,L0=(1-xi)*(1-eta);
        V3 p{L0*a[0]+L1*b[0]+L2*c[0],L0*a[1]+L1*b[1]+L2*c[1],L0*a[2]+L1*b[2]+L2*c[2]};
        s+=w[i]*w[j]*(1-xi)*f(p);}
    return 2.0*area*s;
}
static double intTet(const V3&a,const V3&b,const V3&c,const V3&d,const std::vector<double>&x,const std::vector<double>&w,const Fn&f){
    int n=x.size(); double vol=tetVol(a,b,c,d),s=0;
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<n;++k){ double xi=x[i],eta=x[j],ze=x[k];
        double L1=xi,L2=(1-xi)*eta,L3=(1-xi)*(1-eta)*ze,L0=(1-xi)*(1-eta)*(1-ze);
        V3 p{L0*a[0]+L1*b[0]+L2*c[0]+L3*d[0],L0*a[1]+L1*b[1]+L2*c[1]+L3*d[1],L0*a[2]+L1*b[2]+L2*c[2]+L3*d[2]};
        s+=w[i]*w[j]*w[k]*(1-xi)*(1-xi)*(1-eta)*f(p);}
    return 6.0*vol*s;
}
static double intFace(const Cell&C,const std::vector<int>&f,const V3&fc,const std::vector<double>&x,const std::vector<double>&w,const Fn&fn){
    double s=0; for(size_t i=0;i<f.size();++i) s+=intTri(fc,C.V[f[i]],C.V[f[(i+1)%f.size()]],x,w,fn); return s;
}
static double intCell(const Cell&C,const V3&cc,const std::vector<V3>&fcen,const std::vector<double>&x,const std::vector<double>&w,const Fn&fn){
    double s=0; for(size_t k=0;k<C.F.size();++k){const auto&f=C.F[k];
        for(size_t i=0;i<f.size();++i) s+=intTet(cc,fcen[k],C.V[f[i]],C.V[f[(i+1)%f.size()]],x,w,fn);} return s;
}

static double Peval(const D3Poly&P,double dx,double dy,double dz){
    return P.c[0]*dx+P.c[1]*dy+P.c[2]*dz+P.c[3]*dx*dx+P.c[4]*dy*dy+P.c[5]*dz*dz+P.c[6]*dx*dy+P.c[7]*dx*dz+P.c[8]*dy*dz;}

// GAUSS-S (probit + Edgeworth skewness) lives in reconstruct3d_bvd_core.hpp now:
//   deg3t3d_cellD_fromM_gaussS(M1,M2,M3,kk,Q) , deg3t3d_face_avg_gaussS(fm,D,kk).
// The testbed calls those core functions directly (verifies the port).

static Cell makeHex(){return{{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}},
    {{0,3,2,1},{4,5,6,7},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7}},"hexa"};}
static Cell makeTet(){return{{{0,0,0},{1,0,0},{0,1,0},{0,0,1}},{{0,2,1},{0,1,3},{1,2,3},{0,3,2}},"tetra"};}
static Cell makePrism(){return{{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}},
    {{0,2,1},{3,4,5},{0,1,4,3},{1,2,5,4},{2,0,3,5}},"prism"};}
static Cell makePyr(){return{{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0.5,0.5,1}},
    {{0,3,2,1},{0,1,4},{1,2,4},{2,3,4},{3,0,4}},"pyramid"};}
static Cell makeOcta(){return{{{0.5,0.5,1},{1,0.5,0.5},{0.5,1,0.5},{0,0.5,0.5},{0.5,0,0.5},{0.5,0.5,0}},
    {{0,1,2},{0,2,3},{0,3,4},{0,4,1},{5,2,1},{5,3,2},{5,4,3},{5,1,4}},"octa-poly"};}
static Cell skew(Cell C,double amp,unsigned seed){ std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-amp,amp);
    for(auto&v:C.V){v[0]+=u(g);v[1]+=u(g);v[2]+=u(g);} C.name+="-skew"; return C;}

int main(int argc,char**argv){
    const int Nsamp = argc>1?std::atoi(argv[1]):150;
    const int Nrep  = argc>2?std::atoi(argv[2]):800;
    const double curv = argc>3?std::atof(argv[3]):0.35;   // interface curvature scale (Hessian/H^2)
    const double beta = argc>4?std::atof(argv[4]):1.6;    // THINC sharpness
    const int NQ_REF=14, NQ_PR=7, NQ_MOM=6;
    const bool CSV = std::getenv("CSV")!=nullptr;         // machine-readable per-cell rows
    // per-face sample dump (variance v_face, GAUSS & GAUSS-S face errors) for the
    // error-vs-variance universal-collapse figure. env GAUSS_SAMPLES=<path>.
    FILE* sdump = std::getenv("GAUSS_SAMPLES")?std::fopen(std::getenv("GAUSS_SAMPLES"),"w"):nullptr;
    if(sdump) std::fprintf(sdump,"cell,beta,v,errG,errS,Dt,Dg,fvT,fvG,fvS\n");
    std::vector<double> xr,wr,xp,wp,xm,wm; gl01(NQ_REF,xr,wr); gl01(NQ_PR,xp,wp); gl01(NQ_MOM,xm,wm);

    std::vector<Cell> cells={makeHex(),makeTet(),makePrism(),makePyr(),makeOcta(),
                             skew(makeHex(),0.18,11),skew(makePrism(),0.15,22)};

    std::printf("GAUSS THINC cell-shape testbed  Nsamp=%d Nrep=%d  tanh-ref=%dpt(truth) tanh-practical=%dpt moments=%dpt(exact) beta=%.1f\n",
                Nsamp,Nrep,NQ_REF,NQ_PR,NQ_MOM,beta);
    std::printf("(faceErr rms/max, consErr, speedup-vs-tanh)  G=GAUSS(2-moment)  S=GAUSS-S(skew,3-moment)\n");
    std::printf("%-13s| tanh-wall ||  GAUSS rms/max  cons  spd  ||  GAUSS-S rms/max  cons  spd\n","cell");

    double sp_sum=0; int sp_n=0;
    for(auto&C:cells){
        V3 cc; double vol; cellGeom(C,cc,vol); double H=std::cbrt(vol),kk=beta/H;
        std::vector<V3> fcen(C.F.size()); std::vector<double> farea(C.F.size());
        for(size_t k=0;k<C.F.size();++k) faceGeom(C,C.F[k],fcen[k],farea[k]);
        auto Pl=[&](const D3Poly&P,const V3&p){return Peval(P,p[0]-cc[0],p[1]-cc[1],p[2]-cc[2]);};

        std::mt19937 g(777+(int)C.name.size()); std::normal_distribution<double> nd(0,1);
        std::vector<D3Poly> Ps(Nsamp); std::vector<double> Qs(Nsamp);
        for(int s=0;s<Nsamp;++s){ D3Poly P;
            double gx=nd(g),gy=nd(g),gz=nd(g),gn=std::sqrt(gx*gx+gy*gy+gz*gz)+1e-30;
            P.c[0]=gx/gn/H;P.c[1]=gy/gn/H;P.c[2]=gz/gn/H;
            for(int k=3;k<9;++k)P.c[k]=curv*nd(g)/(H*H);
            double Q=std::tanh(1.2*nd(g)); Q=Q>0.95?0.95:(Q<-0.95?-0.95:Q); Ps[s]=P;Qs[s]=Q;}

        auto tanhCellMean=[&](const D3Poly&P,double D,const std::vector<double>&x,const std::vector<double>&w){
            return intCell(C,cc,fcen,x,w,[&](const V3&p){return std::tanh(kk*Pl(P,p)+D);})/vol;};

        // ── accuracy vs fine-tanh truth ──  G = GAUSS(2-moment), S = GAUSS-S(skew, 3-moment)
        double gmx=0,grms=0,gc=0, smx=0,srms=0,sc=0; long nf=0;
        for(int s=0;s<Nsamp;++s){ const D3Poly&P=Ps[s]; double Q=Qs[s];
            double Dt=std::atanh(Q); for(int it=0;it<40;++it){ double f=tanhCellMean(P,Dt,xr,wr)-Q;
                double fp=intCell(C,cc,fcen,xr,wr,[&](const V3&p){double t=std::tanh(kk*Pl(P,p)+Dt);return 1-t*t;})/vol;
                double dD=f/fp; Dt-=dD; if(std::fabs(dD)<1e-13)break;}
            double M1=intCell(C,cc,fcen,xm,wm,[&](const V3&p){return Pl(P,p);})/vol;
            double M2=intCell(C,cc,fcen,xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v;})/vol;
            double M3=intCell(C,cc,fcen,xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v*v;})/vol;
            double Dg=deg3t3d_cellD_fromM_gauss(M1,M2,kk,Q);
            double Ds=deg3t3d_cellD_fromM_gaussS(M1,M2,M3,kk,Q);
            gc=std::max(gc,std::fabs(tanhCellMean(P,Dg,xr,wr)-Q));
            sc=std::max(sc,std::fabs(tanhCellMean(P,Ds,xr,wr)-Q));
            for(size_t k=0;k<C.F.size();++k){
                double fvT=intFace(C,C.F[k],fcen[k],xr,wr,[&](const V3&p){return std::tanh(kk*Pl(P,p)+Dt);})/farea[k];
                double F1=intFace(C,C.F[k],fcen[k],xm,wm,[&](const V3&p){return Pl(P,p);})/farea[k];
                double F2=intFace(C,C.F[k],fcen[k],xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v;})/farea[k];
                double F3=intFace(C,C.F[k],fcen[k],xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v*v;})/farea[k];
                D3FaceMom fm{F1,F2,0,Pl(P,fcen[k])};
                double fvG=deg3t3d_face_avg_gauss(fm,Dg,kk);
                double fvS=deg3t3d_face_avg_gaussS(D3FaceMom{F1,F2,F3,Pl(P,fcen[k])},Ds,kk);
                double eG=fvG-fvT, eS=fvS-fvT;
                gmx=std::max(gmx,std::fabs(eG));grms+=eG*eG;
                smx=std::max(smx,std::fabs(eS));srms+=eS*eS;
                if(sdump){ double vf=kk*kk*(F2-F1*F1); std::fprintf(sdump,"%s,%.3f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",C.name.c_str(),beta,vf,eG,eS,Dt,Dg,fvT,fvG,fvS); }}
            nf+=C.F.size();
        }
        grms=std::sqrt(grms/nf);srms=std::sqrt(srms/nf);

        // ── wall: full reconstruction (cell-D + all faces) per method ──
        volatile double sink=0; auto clk=std::chrono::steady_clock::now;
        auto t0=clk();
        for(int r=0;r<Nrep;++r){ const D3Poly&P=Ps[r%Nsamp]; double Q=Qs[r%Nsamp];
            double Dp=std::atanh(Q); for(int it=0;it<8;++it){ double f=tanhCellMean(P,Dp,xp,wp)-Q;
                double fp=intCell(C,cc,fcen,xp,wp,[&](const V3&p){double t=std::tanh(kk*Pl(P,p)+Dp);return 1-t*t;})/vol;
                double dD=f/fp; Dp-=dD; if(std::fabs(dD)<1e-10)break;}
            for(size_t k=0;k<C.F.size();++k) sink+=intFace(C,C.F[k],fcen[k],xp,wp,[&](const V3&p){return std::tanh(kk*Pl(P,p)+Dp);})/farea[k];
        }
        double wT=std::chrono::duration<double,std::milli>(clk()-t0).count();
        t0=clk();
        for(int r=0;r<Nrep;++r){ const D3Poly&P=Ps[r%Nsamp]; double Q=Qs[r%Nsamp];
            double M1=intCell(C,cc,fcen,xm,wm,[&](const V3&p){return Pl(P,p);})/vol;
            double M2=intCell(C,cc,fcen,xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v;})/vol;
            double Dg=deg3t3d_cellD_fromM_gauss(M1,M2,kk,Q);
            for(size_t k=0;k<C.F.size();++k){
                double F1=intFace(C,C.F[k],fcen[k],xm,wm,[&](const V3&p){return Pl(P,p);})/farea[k];
                double F2=intFace(C,C.F[k],fcen[k],xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v;})/farea[k];
                D3FaceMom fm{F1,F2,0,Pl(P,fcen[k])}; sink+=deg3t3d_face_avg_gauss(fm,Dg,kk);}
        }
        double wG=std::chrono::duration<double,std::milli>(clk()-t0).count();
        t0=clk();
        for(int r=0;r<Nrep;++r){ const D3Poly&P=Ps[r%Nsamp]; double Q=Qs[r%Nsamp];
            double M1=intCell(C,cc,fcen,xm,wm,[&](const V3&p){return Pl(P,p);})/vol;
            double M2=intCell(C,cc,fcen,xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v;})/vol;
            double M3=intCell(C,cc,fcen,xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v*v;})/vol;
            double Ds=deg3t3d_cellD_fromM_gaussS(M1,M2,M3,kk,Q);
            for(size_t k=0;k<C.F.size();++k){
                double F1=intFace(C,C.F[k],fcen[k],xm,wm,[&](const V3&p){return Pl(P,p);})/farea[k];
                double F2=intFace(C,C.F[k],fcen[k],xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v;})/farea[k];
                double F3=intFace(C,C.F[k],fcen[k],xm,wm,[&](const V3&p){double v=Pl(P,p);return v*v*v;})/farea[k];
                sink+=deg3t3d_face_avg_gaussS(D3FaceMom{F1,F2,F3,0.0},Ds,kk);}
        }
        double wS=std::chrono::duration<double,std::milli>(clk()-t0).count(); (void)sink;
        double spG=wT/wG, spS=wT/wS; sp_sum+=spS; sp_n++;
        std::printf("%-13s| %6.1fms || G %.2e/%.2e %.1e %4.1fx || S %.2e/%.2e %.1e %4.1fx\n",
            C.name.c_str(), wT, grms,gmx,gc,spG, srms,smx,sc,spS);
        if(CSV) std::printf("CSV,%s,%.3f,%.5f,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.3f,%.3f\n",
            C.name.c_str(), beta, deg3t_gc(), grms,gmx,gc, srms,smx,sc, spG,spS);
    }
    std::printf("# mean speed-up GAUSS vs tanh-THINC = %.1fx\n", sp_sum/sp_n);
    if(sdump) std::fclose(sdump);
    return 0;
}
