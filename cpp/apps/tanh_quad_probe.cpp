// tanh_quad_probe.cpp — ERROR vs #Gauss-points for the EXACT tanh THINC/QQ.
// Keeps the ORIGINAL tanh THINC kernel (Newton cell-D + quadrature face/cell average) and
// only varies the quadrature rule, to pick the minimal point count that holds error.
// Compares per cell shape, the cell-D CONSERVATION error (does the N-pt-derived D reproduce
// Q under the 14-pt TRUTH cell-average) and the face-average error vs the 14-pt truth, plus
// the relative wall time. Candidate rules (per sub-tet / per sub-tri):
//   C1   = 1-pt centroid                 (1 / 1)   exact deg-1
//   T4   = degree-2 symmetric simplex    (4 / 3)   exact deg-2 (= the quadratic P exactly)
//   G2   = 2-pt tensor Duffy-Gauss       (8 / 4)   current default
//   REF  = 14-pt tensor Duffy-Gauss      (2744/196) truth
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <string>
#include <functional>

using V3 = std::array<double,3>;
using Fn = std::function<double(const V3&)>;
static V3 sub(const V3&a,const V3&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static V3 cross(const V3&a,const V3&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static double dot(const V3&a,const V3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static double tetVol(const V3&a,const V3&b,const V3&c,const V3&d){
    return std::fabs(dot(sub(b,a),cross(sub(c,a),sub(d,a))))/6.0;}

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

// ---- quadrature accumulators returning (Pvals, weights) summing to 1 over cell/face ----
enum Rule { C1, T4, G2, REF };

// integrate over one sub-tet (a,b,c,d): push P-values + normalized weights
static void subTet(Rule rule, const V3&a,const V3&b,const V3&c,const V3&d,
                   const Fn& Pf, std::vector<double>& Pv,std::vector<double>& wt,
                   const std::vector<double>& gx,const std::vector<double>& gw){
    double vol=tetVol(a,b,c,d);
    auto bary=[&](double L0,double L1,double L2,double L3)->V3{
        return {L0*a[0]+L1*b[0]+L2*c[0]+L3*d[0],L0*a[1]+L1*b[1]+L2*c[1]+L3*d[1],L0*a[2]+L1*b[2]+L2*c[2]+L3*d[2]};};
    if(rule==C1){ Pv.push_back(Pf(bary(0.25,0.25,0.25,0.25))); wt.push_back(vol); }
    else if(rule==T4){ // degree-2 symmetric tet, 4 pts, equal weight
        const double aa=0.5854101966249685, bb=0.1381966011250105;
        Pv.push_back(Pf(bary(aa,bb,bb,bb))); wt.push_back(vol*0.25);
        Pv.push_back(Pf(bary(bb,aa,bb,bb))); wt.push_back(vol*0.25);
        Pv.push_back(Pf(bary(bb,bb,aa,bb))); wt.push_back(vol*0.25);
        Pv.push_back(Pf(bary(bb,bb,bb,aa))); wt.push_back(vol*0.25);
    } else { // tensor Duffy-Gauss (G2=2pt, REF=14pt)
        int n=gx.size();
        for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<n;++k){ double xi=gx[i],eta=gx[j],ze=gx[k];
            double L1=xi,L2=(1-xi)*eta,L3=(1-xi)*(1-eta)*ze,L0=1-L1-L2-L3;
            Pv.push_back(Pf(bary(L0,L1,L2,L3)));
            wt.push_back(gw[i]*gw[j]*gw[k]*(1-xi)*(1-xi)*(1-eta)*6.0*vol); }
    }
}
static void subTri(Rule rule, const V3&fc,const V3&a,const V3&b,
                   const Fn& Pf, std::vector<double>& Pv,std::vector<double>& wt,
                   const std::vector<double>& gx,const std::vector<double>& gw){
    V3 cr=cross(sub(a,fc),sub(b,fc)); double area=0.5*std::sqrt(dot(cr,cr));
    auto bary=[&](double L0,double L1,double L2)->V3{
        return {L0*fc[0]+L1*a[0]+L2*b[0],L0*fc[1]+L1*a[1]+L2*b[1],L0*fc[2]+L1*a[2]+L2*b[2]};};
    if(rule==C1){ Pv.push_back(Pf(bary(1.0/3,1.0/3,1.0/3))); wt.push_back(area); }
    else if(rule==T4){ // degree-2 tri, 3 pts (edge midpoints), equal weight
        Pv.push_back(Pf(bary(0.5,0.5,0.0))); wt.push_back(area/3.0);
        Pv.push_back(Pf(bary(0.0,0.5,0.5))); wt.push_back(area/3.0);
        Pv.push_back(Pf(bary(0.5,0.0,0.5))); wt.push_back(area/3.0);
    } else {
        int n=gx.size();
        for(int i=0;i<n;++i)for(int j=0;j<n;++j){ double xi=gx[i],eta=gx[j],L1=xi,L2=(1-xi)*eta,L0=1-L1-L2;
            Pv.push_back(Pf(bary(L0,L1,L2)));
            wt.push_back(gw[i]*gw[j]*(1-xi)*2.0*area); }
    }
}
static void cellQuad(Rule rule,const Cell&C,const V3&cc,const std::vector<V3>&fcen,double vol,
                     const Fn&Pf,std::vector<double>&Pv,std::vector<double>&wt,
                     const std::vector<double>&gx,const std::vector<double>&gw){
    Pv.clear(); wt.clear();
    for(size_t k=0;k<C.F.size();++k){const auto&f=C.F[k];
        for(size_t i=0;i<f.size();++i) subTet(rule,cc,fcen[k],C.V[f[i]],C.V[f[(i+1)%f.size()]],Pf,Pv,wt,gx,gw);}
    for(double& w:wt) w/=vol;
}
static void faceQuad(Rule rule,const Cell&C,const std::vector<int>&f,const V3&fc,double area,
                     const Fn&Pf,std::vector<double>&Pv,std::vector<double>&wt,
                     const std::vector<double>&gx,const std::vector<double>&gw){
    Pv.clear(); wt.clear();
    for(size_t i=0;i<f.size();++i) subTri(rule,fc,C.V[f[i]],C.V[f[(i+1)%f.size()]],Pf,Pv,wt,gx,gw);
    for(double& w:wt) w/=area;
}
static double cellD(const std::vector<double>&Pv,const std::vector<double>&wt,double kk,double Q){
    double D=0;
    for(int it=0;it<30;++it){ double f=-Q,fp=0;
        for(size_t q=0;q<Pv.size();++q){double th=std::tanh(kk*Pv[q]+D); f+=wt[q]*th; fp+=wt[q]*(1-th*th);}
        if(fp<1e-15)break; double dD=f/fp; D-=dD; if(std::fabs(dD)<1e-13)break;}
    return D;
}
static double faceAvg(const std::vector<double>&Pv,const std::vector<double>&wt,double kk,double D){
    double s=0; for(size_t q=0;q<Pv.size();++q) s+=wt[q]*std::tanh(kk*Pv[q]+D); return s;
}

static double Peval(const double*c,double dx,double dy,double dz){
    return c[0]*dx+c[1]*dy+c[2]*dz+c[3]*dx*dx+c[4]*dy*dy+c[5]*dz*dz+c[6]*dx*dy+c[7]*dx*dz+c[8]*dy*dz;}

static Cell makeHex(){return{{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}},
    {{0,3,2,1},{4,5,6,7},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7}},"hexa"};}
static Cell makeTet(){return{{{0,0,0},{1,0,0},{0,1,0},{0,0,1}},{{0,2,1},{0,1,3},{1,2,3},{0,3,2}},"tetra"};}
static Cell makePrism(){return{{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}},
    {{0,2,1},{3,4,5},{0,1,4,3},{1,2,5,4},{2,0,3,5}},"prism"};}
static Cell makePyr(){return{{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0.5,0.5,1}},
    {{0,3,2,1},{0,1,4},{1,2,4},{2,3,4},{3,0,4}},"pyramid"};}
static Cell skew(Cell C,double amp,unsigned seed){ std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-amp,amp);
    for(auto&v:C.V){v[0]+=u(g);v[1]+=u(g);v[2]+=u(g);} C.name+="-skew"; return C;}

int main(int argc,char**argv){
    int Nsamp=argc>1?std::atoi(argv[1]):200;
    int Nrep =argc>2?std::atoi(argv[2]):2000;
    double curv=argc>3?std::atof(argv[3]):0.35;
    double beta=argc>4?std::atof(argv[4]):1.6;
    std::vector<double> g2x,g2w,grx,grw; gl01(2,g2x,g2w); gl01(14,grx,grw);
    std::vector<double> empty;
    std::vector<Cell> cells={makeHex(),makeTet(),makePrism(),makePyr(),skew(makeHex(),0.18,11),skew(makePrism(),0.15,22)};
    std::printf("tanh THINC/QQ quad-point probe  Nsamp=%d Nrep=%d curv=%.2f beta=%.1f\n",Nsamp,Nrep,curv,beta);
    std::printf("(faceErr rms/max vs 14pt truth, consErr=|truth<tanh(.,D)> - Q|, wall, speedup vs G2)\n");
    std::printf("%-12s|   C1 face rms/max cons  wall  spd ||  T4 face rms/max cons  wall  spd ||  G2 face rms/max cons  wall\n","cell");
    for(auto&C:cells){
        V3 cc; double vol; cellGeom(C,cc,vol); double H=std::cbrt(vol),kk=beta/H;
        std::vector<V3> fcen(C.F.size()); std::vector<double> farea(C.F.size());
        for(size_t k=0;k<C.F.size();++k) faceGeom(C,C.F[k],fcen[k],farea[k]);
        std::mt19937 g(777+(int)C.name.size()); std::normal_distribution<double> nd(0,1);
        std::vector<std::array<double,9>> Ps(Nsamp); std::vector<double> Qs(Nsamp);
        for(int s=0;s<Nsamp;++s){ std::array<double,9> c{};
            double gx=nd(g),gy=nd(g),gz=nd(g),gn=std::sqrt(gx*gx+gy*gy+gz*gz)+1e-30;
            c[0]=gx/gn/H;c[1]=gy/gn/H;c[2]=gz/gn/H;
            for(int k=3;k<9;++k)c[k]=curv*nd(g)/(H*H);
            double Q=std::tanh(1.2*nd(g)); Q=Q>0.95?0.95:(Q<-0.95?-0.95:Q); Ps[s]=c; Qs[s]=Q; }

        auto runRule=[&](Rule rule,double& frms,double& fmax,double& cmx,double& wall){
            frms=0;fmax=0;cmx=0; long nf=0;
            const std::vector<double>&gx=(rule==G2)?g2x:(rule==REF?grx:empty);
            const std::vector<double>&gw=(rule==G2)?g2w:(rule==REF?grw:empty);
            std::vector<double> Pv,wt, Pvr,wtr, Pvf,wtf, Pvfr,wtfr;
            for(int s=0;s<Nsamp;++s){ const double* c=Ps[s].data(); double Q=Qs[s];
                auto Pf=[&](const V3&p){return Peval(c,p[0]-cc[0],p[1]-cc[1],p[2]-cc[2]);};
                cellQuad(rule,C,cc,fcen,vol,Pf,Pv,wt,gx,gw);
                double D=cellD(Pv,wt,kk,Q);
                cellQuad(REF,C,cc,fcen,vol,Pf,Pvr,wtr,grx,grw);
                double truthCell=0; for(size_t q=0;q<Pvr.size();++q) truthCell+=wtr[q]*std::tanh(kk*Pvr[q]+D);
                cmx=std::max(cmx,std::fabs(truthCell-Q));
                for(size_t k=0;k<C.F.size();++k){
                    faceQuad(rule,C,C.F[k],fcen[k],farea[k],Pf,Pvf,wtf,gx,gw);
                    double fv=faceAvg(Pvf,wtf,kk,D);
                    // truth face uses the SAME D but 14pt
                    double Dt; { std::vector<double> Pvt,wtt; cellQuad(REF,C,cc,fcen,vol,Pf,Pvt,wtt,grx,grw); Dt=cellD(Pvt,wtt,kk,Q); }
                    faceQuad(REF,C,C.F[k],fcen[k],farea[k],Pf,Pvfr,wtfr,grx,grw);
                    double fvt=faceAvg(Pvfr,wtfr,kk,Dt);
                    double e=fv-fvt; frms+=e*e; fmax=std::max(fmax,std::fabs(e)); nf++;
                }
            }
            frms=std::sqrt(frms/nf);
            // wall: cell-D + all faces, Nrep
            volatile double sink=0; auto t0=std::chrono::steady_clock::now();
            for(int r=0;r<Nrep;++r){ const double* c=Ps[r%Nsamp].data(); double Q=Qs[r%Nsamp];
                auto Pf=[&](const V3&p){return Peval(c,p[0]-cc[0],p[1]-cc[1],p[2]-cc[2]);};
                cellQuad(rule,C,cc,fcen,vol,Pf,Pv,wt,gx,gw);
                double D=cellD(Pv,wt,kk,Q);
                for(size_t k=0;k<C.F.size();++k){ faceQuad(rule,C,C.F[k],fcen[k],farea[k],Pf,Pvf,wtf,gx,gw); sink+=faceAvg(Pvf,wtf,kk,D);} }
            wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); (void)sink;
        };
        double c1r,c1m,c1c,c1w, t4r,t4m,t4c,t4w, g2r,g2m,g2c,g2w_;
        runRule(C1,c1r,c1m,c1c,c1w); runRule(T4,t4r,t4m,t4c,t4w); runRule(G2,g2r,g2m,g2c,g2w_);
        std::printf("%-12s| %.2e/%.2e %.1e %5.0f %4.1fx || %.2e/%.2e %.1e %5.0f %4.1fx || %.2e/%.2e %.1e %5.0f\n",
            C.name.c_str(), c1r,c1m,c1c,c1w,g2w_/c1w, t4r,t4m,t4c,t4w,g2w_/t4w, g2r,g2m,g2c,g2w_);
    }
    return 0;
}
