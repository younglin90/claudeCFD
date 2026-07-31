// apps/thinc_cellshape2d_bench.cpp — 2D CELL-SHAPE unit testbed for the GAUSS THINC.
//
// 2D analog of thinc_cellshape_bench.cpp (3D). Isolates the interface-reconstruction
// kernel  "QQ P2 polynomial input -> sigmoid cell-D (cell-average constraint) -> edge
// (face) integral"  and runs it on 2D cell shapes (triangle / quad + skewed & obtuse
// non-ortho variants — the 2D unstructured mesh cell set), comparing:
//   (R) tanh-THINC : the 2D standard (reconstruct_bvd.hpp reconstruct_cheng3). RATIONAL
//                    THINC  g=(tanh(kP)+D)/(1+D tanh(kP))  [ == tanh(kP+atanh(D)) ];
//                    cell-D by NEWTON on the cell-avg constraint over the 6-pt Dunavant
//                    deg-4 triangle rule TQ (a quad = 2 triangles); face value by 4-pt
//                    Gauss-Legendre EDGE quadrature EQ of the rational tanh. Shape-
//                    dependent, expensive (tanh at every quadrature point).
//   (G) GAUSS      : probit-identity closed form (reconstruct_bvd.hpp GAUSS path). The
//                    Gaussian-moment identity  <tanh(kk P+D)> ~= tanh((kk<P>+D)/sqrt(1+c v)),
//                    c=pi/2, v=kk^2 Var(P)  makes the cell-average constraint invert in
//                    CLOSED FORM  D = atanh(Q) sqrt(1+c v) - kk<P>  (no Newton), from the two
//                    cell moments <P>,<P^2> only; the face average is a single tanh of the
//                    CLOSED-FORM edge moments  F1=int_0^1 P dt, F2=int_0^1 P^2 dt  of the
//                    edge-restricted P(t)=p2 t^2+p1 t+p0  (no quadrature of tanh).
// Reports per shape: tanh wall time, GAUSS face-average error (rms/max) vs a fine-tanh
// truth, cell-average CONSERVATION error (does the closed-form D reproduce Q under the
// TRUE tanh), and WALL speed-up vs the Newton+quadrature tanh-THINC.
//
// NOTE: 2D reconstruct_cheng3 has NO skew (3-moment GAUSS-S) variant — the 2D GAUSS path
// uses only the 2 moments <P>,<P^2> (cell) and F1,F2 (face). The Edgeworth-skew GAUSS-S
// exists only in the 3D core (reconstruct3d_bvd_core.hpp). So (S) is omitted here.
//
// Self-contained kernel benchmark: the tanh-Newton + GAUSS closed-form formulas are
// replicated inline (matching reconstruct_cheng3 lines ~1265-1272 cell-D, ~1373-1381 tanh
// Newton, ~1820-1827 face). No solver / flux / mesh-load.
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <string>
#include <functional>

using V2 = std::array<double,2>;
using Fn = std::function<double(double,double)>;   // f(x,y)

// Gauss-Legendre nodes/weights on [0,1] (identical helper to the 3D bench).
static void gl01(int n, std::vector<double>& x, std::vector<double>& w){
    x.resize(n); w.resize(n);
    for(int i=0;i<n;++i){ double z=std::cos(M_PI*(i+0.75)/(n+0.5)),z1,pp;
        do{ double p1=1,p2=0;
            for(int j=0;j<n;++j){ double p3=p2;p2=p1; p1=((2*j+1)*z*p2-j*p3)/(j+1);}
            pp=n*(z*p1-p2)/(z*z-1.0); z1=z; z=z1-p1/pp;
        }while(std::fabs(z-z1)>1e-15);
        x[i]=0.5*(1.0-z); w[i]=1.0/((1.0-z*z)*pp*pp); }
}

// 6-pt Dunavant deg-4 triangle rule {w, b0,b1,b2} — the TQ rule reconstruct_cheng3 uses
// for the cell moments (exact for polynomials up to degree 4, so <P>,<P^2> are exact).
static const double TQ[6][4]={
    {0.109951743655322,0.816847572980459,0.091576213509771,0.091576213509771},
    {0.109951743655322,0.091576213509771,0.816847572980459,0.091576213509771},
    {0.109951743655322,0.091576213509771,0.091576213509771,0.816847572980459},
    {0.223381589678011,0.108103018168070,0.445948490915965,0.445948490915965},
    {0.223381589678011,0.445948490915965,0.108103018168070,0.445948490915965},
    {0.223381589678011,0.445948490915965,0.445948490915965,0.108103018168070}};
// 4-pt Gauss-Legendre on [0,1] {t,w} — the EQ edge rule reconstruct_cheng3 uses for the
// tanh face quadrature (the expensive shape-dependent path).
static const double EQ[4][2]={
    {0.0694318442029737,0.1739274225687269},{0.3300094782075719,0.3260725774312731},
    {0.6699905217924281,0.3260725774312731},{0.9305681557970263,0.1739274225687269}};

// 2D QQ-P2 interface polynomial:  P(dx,dy)=a0 dx + a1 dy + a2 dx^2 + a3 dy^2 + a4 dx dy
// (a0,a1 = unit normal linear part; a2,a3,a4 = curvature Hessian). Matches the 2D
// reconstruct_cheng3 monomial layout A[0..4].
struct P2 { double a[5]; };
static inline double Peval(const P2& P, double dx, double dy){
    return P.a[0]*dx + P.a[1]*dy + P.a[2]*dx*dx + P.a[3]*dy*dy + P.a[4]*dx*dy;
}

struct Cell{ std::vector<V2> V; std::string name; };   // vertices CCW; faces = consecutive edges
struct Edge{ double ax,ay,bx,by; };
struct CellQuad{ std::vector<double> qx,qy,qw; };       // TQ cell-quadrature points; qw normalized (sum=1)

static inline double triArea(const V2&a,const V2&b,const V2&c){
    return 0.5*std::fabs((b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1]));
}
// cell centroid / area / perimeter via fan triangulation from vertex 0 (all shapes convex).
static void cellGeom(const Cell&C,V2&cen,double&area,double&perim){
    int n=(int)C.V.size(); area=0; V2 acc{0,0};
    for(int i=1;i+1<n;++i){ const V2&a=C.V[0],&b=C.V[i],&c=C.V[i+1];
        double ta=triArea(a,b,c); area+=ta;
        acc[0]+=ta*(a[0]+b[0]+c[0])/3.0; acc[1]+=ta*(a[1]+b[1]+c[1])/3.0; }
    cen={acc[0]/area,acc[1]/area};
    perim=0; for(int i=0;i<n;++i){ const V2&a=C.V[i],&b=C.V[(i+1)%n]; perim+=std::hypot(b[0]-a[0],b[1]-a[1]); }
}
// build the TQ (6-pt Dunavant) cell quadrature (fan triangles), weights normalized so
// sum(qw)=1  =>  cell-average of f = sum_q qw[q] f(qx[q],qy[q])  (deg-4-exact).
static CellQuad buildTQ(const Cell&C,double area){
    CellQuad cq; int n=(int)C.V.size();
    for(int i=1;i+1<n;++i){ const V2&a=C.V[0],&b=C.V[i],&c=C.V[i+1]; double ta=triArea(a,b,c);
        for(int q=0;q<6;++q){ double L0=TQ[q][1],L1=TQ[q][2],L2=TQ[q][3];
            cq.qx.push_back(L0*a[0]+L1*b[0]+L2*c[0]);
            cq.qy.push_back(L0*a[1]+L1*b[1]+L2*c[1]);
            cq.qw.push_back(TQ[q][0]*ta/area); } }
    return cq;
}
// fine collapsed-Gauss (Duffy) integral of f over a triangle (2D analog of the 3D bench).
static double intTri(const V2&a,const V2&b,const V2&c,const std::vector<double>&x,const std::vector<double>&w,const Fn&f){
    int n=(int)x.size(); double area=triArea(a,b,c),s=0;
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){ double xi=x[i],eta=x[j];
        double L1=xi,L2=(1-xi)*eta,L0=(1-xi)*(1-eta);
        double px=L0*a[0]+L1*b[0]+L2*c[0], py=L0*a[1]+L1*b[1]+L2*c[1];
        s+=w[i]*w[j]*(1-xi)*f(px,py); }
    return 2.0*area*s;
}
static double intCell(const Cell&C,const std::vector<double>&x,const std::vector<double>&w,const Fn&f){
    int n=(int)C.V.size(); double s=0;
    for(int i=1;i+1<n;++i) s+=intTri(C.V[0],C.V[i],C.V[i+1],x,w,f);
    return s;
}

// ── GAUSS closed forms (inline, matching reconstruct_cheng3) ──────────────────────────
// cell-D:  D = atanh(Q) sqrt(1+c v) - kk<P>,  v = kk^2(<P^2>-<P>^2).   (reconstruct_bvd.hpp:1271)
static inline double gaussCellD(double mm1,double mm2,double kk,double Q,double c){
    double v=kk*kk*(mm2-mm1*mm1); if(v<0)v=0;
    double Qc=Q<-0.999?-0.999:(Q>0.999?0.999:Q);
    double aQ=0.5*std::log((1.0+Qc)/(1.0-Qc));          // atanh(Q)
    return aQ*std::sqrt(1.0+c*v)-kk*mm1;
}
// face:  tanh((kk F1+D)/sqrt(1+c(F2-F1^2)))  from the CLOSED-FORM edge moments of
// P(t)=p2 t^2+p1 t+p0 over t in [0,1].   (reconstruct_bvd.hpp:1820-1827)
static inline double gaussFace(const P2&P,const V2&cen,double D,double kk,const Edge&e,double c){
    double dx0=e.ax-cen[0],dy0=e.ay-cen[1],ex=e.bx-e.ax,ey=e.by-e.ay;
    double p2=P.a[2]*ex*ex+P.a[3]*ey*ey+P.a[4]*ex*ey;
    double p1=P.a[0]*ex+P.a[1]*ey+2.0*P.a[2]*dx0*ex+2.0*P.a[3]*dy0*ey+P.a[4]*(dx0*ey+dy0*ex);
    double p0=P.a[0]*dx0+P.a[1]*dy0+P.a[2]*dx0*dx0+P.a[3]*dy0*dy0+P.a[4]*dx0*dy0;
    double F1=p2/3.0+p1/2.0+p0;
    double F2=p2*p2/5.0+p1*p2/2.0+(p1*p1+2.0*p0*p2)/3.0+p0*p1+p0*p0;
    double vv=kk*kk*(F2-F1*F1); if(vv<0)vv=0;
    return std::tanh((kk*F1+D)/std::sqrt(1.0+c*vv));
}
// rational THINC sigmoid  (tanh(kP)+D)/(1+D tanh(kP))  (reconstruct_bvd.hpp:1818)
static inline double ratSig(double kP,double D){
    double A=std::tanh(kP); double den=1.0+A*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12);
    return (A+D)/den;
}

// ── cell shapes: 2D unstructured mesh cell set (triangle/quad) + non-ortho variants ──
static Cell makeTri(){      return {{{0,0},{1,0},{0.5,0.8660254}}, "triangle"}; }        // ~equilateral
static Cell makeQuad(){     return {{{0,0},{1,0},{1,1},{0,1}}, "quad"}; }                // unit square
static Cell makeSkewQuad(){ return {{{0,0},{1,0},{1.35,0.9},{0.35,0.9}}, "skew-quad"}; } // parallelogram (non-ortho)
static Cell makeObtuseTri(){return {{{0,0},{1.0,0},{-0.26,0.97}}, "obtuse-tri"}; }        // ~105 deg obtuse at v0 (compact, aspect ~1.3)
static Cell skew(Cell C,double amp,unsigned seed){ std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-amp,amp);
    for(auto&v:C.V){v[0]+=u(g);v[1]+=u(g);} C.name+="-skew"; return C; }

int main(int argc,char**argv){
    const int    Nsamp = argc>1?std::atoi(argv[1]):150;
    const int    Nrep  = argc>2?std::atoi(argv[2]):300000;   // 2D per-rep work is ~2000x cheaper than 3D -> need many reps for stable ms-scale walls
    const double curv  = argc>3?std::atof(argv[3]):0.35;   // interface curvature scale (Hessian/H^2)
    const double beta  = argc>4?std::atof(argv[4]):1.4;    // THINC sharpness (2D S1 value)
    const int    NQ_REF=14;                                // fine tanh truth (per-axis)
    const double GC = []{ const char* e=std::getenv("THINCQQ_GC"); return (e&&e[0])?std::atof(e):1.5707963267948966; }();  // probit c=pi/2
    std::vector<double> xr,wr; gl01(NQ_REF,xr,wr);

    std::vector<Cell> cells={makeTri(),makeQuad(),makeSkewQuad(),makeObtuseTri(),
                             skew(makeQuad(),0.12,11),skew(makeTri(),0.10,22)};

    std::printf("GAUSS THINC 2D cell-shape testbed  Nsamp=%d Nrep=%d  tanh-truth=%dpt  tanh-practical=TQ6/EQ4  moments=TQ6(deg4-exact) beta=%.1f\n",
                Nsamp,Nrep,NQ_REF,beta);
    std::printf("(faceErr rms/max, consErr, speedup-vs-tanh)  G=GAUSS(2-moment probit closed-form)  [2D reconstruct_cheng3 has NO GAUSS-S skew variant]\n");
    std::printf("%-13s| tanh-wall ||  GAUSS rms/max  cons  spd\n","cell");
    FILE* sdump = std::getenv("GAUSS_SAMPLES")?std::fopen(std::getenv("GAUSS_SAMPLES"),"w"):nullptr;
    if(sdump) std::fprintf(sdump,"cell,beta,v,errG,Dt,Dg,fvT,fvG\n");

    double sp_sum=0; int sp_n=0;
    for(auto&C:cells){
        V2 cc; double area,perim; cellGeom(C,cc,area,perim);
        double kb=perim/(4.0*area), H=1.0/kb, kk=beta*kb;   // kb=1/H exactly as reconstruct_cheng3 (perim/(4 vol))
        CellQuad cq=buildTQ(C,area);
        std::vector<Edge> edges; int nvv=(int)C.V.size();
        for(int i=0;i<nvv;++i) edges.push_back({C.V[i][0],C.V[i][1],C.V[(i+1)%nvv][0],C.V[(i+1)%nvv][1]});

        std::mt19937 g(777+(int)C.name.size()); std::normal_distribution<double> nd(0,1);
        std::vector<P2> Ps(Nsamp); std::vector<double> Qs(Nsamp);
        for(int s=0;s<Nsamp;++s){ P2 P;
            double gx=nd(g),gy=nd(g),gn=std::sqrt(gx*gx+gy*gy)+1e-30;
            P.a[0]=gx/gn/H; P.a[1]=gy/gn/H;
            P.a[2]=curv*nd(g)/(H*H); P.a[3]=curv*nd(g)/(H*H); P.a[4]=curv*nd(g)/(H*H);
            double Q=std::tanh(1.2*nd(g)); Q=Q>0.95?0.95:(Q<-0.95?-0.95:Q); Ps[s]=P; Qs[s]=Q; }

        // additive-tanh cell mean <tanh(kk P+D)> and its dD-derivative <1-tanh^2>, fine rule.
        auto tanhMeanAdd=[&](const P2&P,double D){
            return intCell(C,xr,wr,[&](double px,double py){return std::tanh(kk*Peval(P,px-cc[0],py-cc[1])+D);})/area; };
        auto tanhMeanDeriv=[&](const P2&P,double D){
            return intCell(C,xr,wr,[&](double px,double py){double t=std::tanh(kk*Peval(P,px-cc[0],py-cc[1])+D);return 1.0-t*t;})/area; };

        // ── accuracy vs fine-tanh truth ──
        double gmx=0,grms=0,gc=0; long nf=0;
        for(int s=0;s<Nsamp;++s){ const P2&P=Ps[s]; double Q=Qs[s];
            // truth additive cell-D (fine-rule Newton)
            double Qc=Q<-0.999999?-0.999999:(Q>0.999999?0.999999:Q); double Dt=0.5*std::log((1.0+Qc)/(1.0-Qc));
            for(int it=0;it<40;++it){ double f=tanhMeanAdd(P,Dt)-Q, fp=tanhMeanDeriv(P,Dt);
                if(std::fabs(fp)<1e-30)break; double dD=f/fp; Dt-=dD; if(std::fabs(dD)<1e-13)break; }
            // GAUSS cell-D from TQ 6-pt moments
            double mm1=0,mm2=0; for(size_t q=0;q<cq.qw.size();++q){ double p=Peval(P,cq.qx[q]-cc[0],cq.qy[q]-cc[1]); mm1+=cq.qw[q]*p; mm2+=cq.qw[q]*p*p; }
            double Dg=gaussCellD(mm1,mm2,kk,Q,GC);
            gc=std::max(gc,std::fabs(tanhMeanAdd(P,Dg)-Q));   // conservation: GAUSS D reproduces Q under TRUE tanh?
            for(auto&e:edges){
                double fvT=0; for(int q=0;q<NQ_REF;++q){ double t=xr[q],px=e.ax+t*(e.bx-e.ax),py=e.ay+t*(e.by-e.ay);
                    fvT+=wr[q]*std::tanh(kk*Peval(P,px-cc[0],py-cc[1])+Dt); }   // fine-tanh truth edge value
                double fvG=gaussFace(P,cc,Dg,kk,e,GC);
                double eG=fvG-fvT; gmx=std::max(gmx,std::fabs(eG)); grms+=eG*eG; nf++;
                if(sdump) std::fprintf(sdump,"%s,%.3f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",C.name.c_str(),beta,kk*kk*(mm2-mm1*mm1),eG,Dt,Dg,fvT,fvG);
            }
        }
        grms=std::sqrt(grms/nf);

        // ── wall: full reconstruction (cell-D + all edge faces) per method ──
        volatile double sink=0; auto clk=std::chrono::steady_clock::now;
        auto t0=clk();
        for(int r=0;r<Nrep;++r){ const P2&P=Ps[r%Nsamp]; double Q=Qs[r%Nsamp];
            // tanh cell-D: precompute Ag=tanh(kk Pg) once, rational Newton (reconstruct_cheng3 L1373-1381)
            double Ag[64]; int nq=(int)cq.qw.size();
            for(int q=0;q<nq;++q) Ag[q]=std::tanh(kk*Peval(P,cq.qx[q]-cc[0],cq.qy[q]-cc[1]));
            double D=0.0;
            for(int it=0;it<10;++it){ double f=-Q,fp=0.0;
                for(int q=0;q<nq;++q){ double den=1.0+Ag[q]*D; if(std::fabs(den)<1e-12)den=(den<0?-1e-12:1e-12);
                    f+=cq.qw[q]*(Ag[q]+D)/den; fp+=cq.qw[q]*(1.0-Ag[q]*Ag[q])/(den*den); }
                if(std::fabs(fp)<1e-30)break; double dD=f/fp; D-=dD;
                if(D>0.999999)D=0.999999; else if(D<-0.999999)D=-0.999999;
                if(std::fabs(dD)<1e-11)break; }
            for(auto&e:edges){ double th=0; for(int q=0;q<4;++q){ double t=EQ[q][0],px=e.ax+t*(e.bx-e.ax),py=e.ay+t*(e.by-e.ay);
                th+=EQ[q][1]*ratSig(kk*Peval(P,px-cc[0],py-cc[1]),D); } sink+=th; }
        }
        double wT=std::chrono::duration<double,std::milli>(clk()-t0).count();
        t0=clk();
        for(int r=0;r<Nrep;++r){ const P2&P=Ps[r%Nsamp]; double Q=Qs[r%Nsamp];
            double mm1=0,mm2=0; int nq=(int)cq.qw.size();
            for(int q=0;q<nq;++q){ double p=Peval(P,cq.qx[q]-cc[0],cq.qy[q]-cc[1]); mm1+=cq.qw[q]*p; mm2+=cq.qw[q]*p*p; }
            double Dg=gaussCellD(mm1,mm2,kk,Q,GC);
            for(auto&e:edges) sink+=gaussFace(P,cc,Dg,kk,e,GC);
        }
        double wG=std::chrono::duration<double,std::milli>(clk()-t0).count(); (void)sink;
        double spG=wT/wG; sp_sum+=spG; sp_n++;
        std::printf("%-13s| %6.1fms || G %.2e/%.2e %.1e %4.1fx\n",
            C.name.c_str(), wT, grms,gmx,gc, spG);
    }
    std::printf("# mean speed-up GAUSS vs tanh-THINC = %.1fx\n", sp_sum/sp_n);
    return 0;
}
