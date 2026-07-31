// cfd/mood2d.hpp — MOOD (Multi-dimensional Optimal Order Detection) for 2D Euler.
// Unlimited P2 (3rd-order) candidate everywhere; a-posteriori detect troubled cells
// (PAD: rho,p>0 & finite; relaxed DMP on density) and cascade their polynomial
// degree down 2->1->0 (P0 is always admissible), recompute, iterate. Smooth cells
// keep P2 (3rd order); only genuinely troubled cells drop order. SSP-RK3 in time,
// MOOD applied on the full step (detect on U^{n+1} vs U^n neighbourhood).
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/euler2d.hpp"
#include "cfd/reconstruct2d_o2.hpp"
#include "cfd/solver2d.hpp"
#include "cfd/solver_euler2d.hpp"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace cfd {

// UNLIMITED P2 prim reconstruction with per-cell degree (0/1/2) baked into faces.
inline void reconstruct_mood(const Mesh& m, const ReconCtxO2& c,
                             const std::vector<double>& W, int nvar,
                             const std::vector<char>& deg, std::vector<double>& g,
                             std::vector<double>& WL, std::vector<double>& WR) {
    const int N=m.n_cells(), Nf=m.n_faces();
    const double* cc=m.cell_centers.data();
    g.assign((size_t)nvar*N*5,0.0);
    #pragma omp parallel for
    for(int ci=0;ci<N;++ci) for(int vi=0;vi<nvar;++vi){
        double wc=W[(size_t)vi*N+ci]; double co[5]={0,0,0,0,0};
        for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
            double dW=W[(size_t)vi*N+nb]-wc;
            for(int i=0;i<5;++i) co[i]+=c.M[((size_t)ci*5+i)*c.max_nb+k]*dW; }
        for(int i=0;i<5;++i) g[(((size_t)vi*N+ci)*5)+i]=co[i]; }
    WL.assign((size_t)nvar*Nf,0.0); WR.assign((size_t)nvar*Nf,0.0);
    #pragma omp parallel for
    for(int f=0;f<Nf;++f){ int o=m.face_owner[f], n=m.face_neighbour[f];
        double fx=m.face_centers[f*2], fy=m.face_centers[f*2+1];
        for(int vi=0;vi<nvar;++vi){
            auto val=[&](int ci){ double dx=fx-cc[ci*2], dy=fy-cc[ci*2+1];
                const double* G=&g[(((size_t)vi*N+ci)*5)]; double inc=0; char d=deg[ci];
                if(d>=1) inc += G[0]*dx+G[1]*dy;
                if(d>=2) inc += 0.5*G[2]*dx*dx+0.5*G[3]*dy*dy+G[4]*dx*dy;
                return W[(size_t)vi*N+ci]+inc; };
            WL[(size_t)vi*Nf+f]=val(o);
            WR[(size_t)vi*Nf+f]=(n>=0)?val(n):val(o); } }
}

inline Solve2DResult solve_euler2d_mood(const Mesh& m, const Euler2D& eq,
        const std::vector<double>& U0, double t_end, double cfl,
        int max_steps, int flux, const std::vector<BC2D>* bcs,
        const ReconCtxO2& c2) {
    const int N=m.n_cells(), Nf=m.n_faces(), sz=4*N;
    const double floor = []{ const char* e=std::getenv("MOOD_FLOOR"); return e?std::atof(e):1e-10; }();
    const double drelax = []{ const char* e=std::getenv("MOOD_DMP"); return e?std::atof(e):0.0; }();
    const bool use_u2 = std::getenv("MOOD_U2") != nullptr;   // u2 smooth-extremum forgiveness (off=strict)
    std::vector<double> U=U0, Wc(sz), WL, WR, g, L(sz);
    std::vector<char> deg(N);
    double cur_t=0.0;
    auto rhs=[&](const std::vector<double>& s, std::vector<double>& out){
        #pragma omp parallel for
        for(int i=0;i<N;++i){ double u[4]={s[0*N+i],s[1*N+i],s[2*N+i],s[3*N+i]},w[4];
            eq.cons_to_prim(u,w); Wc[0*N+i]=w[0];Wc[1*N+i]=w[1];Wc[2*N+i]=w[2];Wc[3*N+i]=w[3]; }
        reconstruct_mood(m,c2,Wc,4,deg,g,WL,WR);
        std::vector<double> Fall((size_t)4*Nf);
        #pragma omp parallel for
        for(int f=0;f<Nf;++f){ int o=m.face_owner[f], n=m.face_neighbour[f];
            double nx=m.face_normals[f*2],ny=m.face_normals[f*2+1],area=m.face_areas[f];
            double wL[4],wR[4];
            for(int v=0;v<4;++v){ wL[v]=WL[(size_t)v*Nf+f]; if(n>=0) wR[v]=WR[(size_t)v*Nf+f]; }
            if(n<0){ // NO recon at ANY boundary face (2026-07-03): cell avg + BC ghost
                for(int v=0;v<4;++v) wL[v]=Wc[(size_t)v*N+o];
                int tag=m.face_bc_tag[f];
                const BC2D* bc=(bcs&&tag>0&&tag<(int)bcs->size())?&(*bcs)[tag]:nullptr;
                if(bc) apply_bc(*bc,wL,nx,ny,m.face_centers[f*2],m.face_centers[f*2+1],cur_t,wR);
                else for(int v=0;v<4;++v) wR[v]=wL[v]; }
            double F[4]; if(flux==FLUX_HLLC) hllc_euler2d(eq,wL,wR,nx,ny,F); else llf_euler2d(eq,wL,wR,nx,ny,F);
            for(int v=0;v<4;++v) Fall[(size_t)v*Nf+f]=F[v]*area; }
        #pragma omp parallel for
        for(int i=0;i<N;++i){ double acc[4]={0,0,0,0};
            for(int f:m.cell_faces[i]){ double s2=(m.face_owner[f]==i)?-1.0:1.0;
                for(int v=0;v<4;++v) acc[v]+=s2*Fall[(size_t)v*Nf+f]; }
            double inv=1.0/m.cell_volumes[i]; for(int v=0;v<4;++v) out[(size_t)v*N+i]=acc[v]*inv; }
    };
    double h_min=1e300;
    for(int i=0;i<N;++i){ double amax=0; for(int f:m.cell_faces[i]) amax=std::max(amax,m.face_areas[f]);
        h_min=std::min(h_min,m.cell_volumes[i]/std::max(amax,1e-30)); }
    // per-cell density min/max over self+face-neighbours (reference for DMP).
    auto nbminmax=[&](const std::vector<double>& s,int i,double&mn,double&mx){
        mn=mx=s[0*N+i];
        for(int f:m.cell_faces[i]){ int o=m.face_owner[f],n=m.face_neighbour[f],nb=(o==i)?n:o;
            if(nb<0)continue; double r=s[0*N+nb]; mn=std::min(mn,r); mx=std::max(mx,r); } };
    double t=0.0; int n=0;
    for(; n<max_steps && t<t_end; ++n){
        double wmax=0.0;
        #pragma omp parallel for reduction(max:wmax)
        for(int i=0;i<N;++i){ double u[4]={U[0*N+i],U[1*N+i],U[2*N+i],U[3*N+i]};
            double w=std::max(eq.max_wave_speed(u,1,0),eq.max_wave_speed(u,0,1)); if(w>wmax)wmax=w; }
        if(!std::isfinite(wmax)||wmax>1e12){ std::fprintf(stderr,"MOOD DIVERGED wmax=%.3e step %d t=%.4g\n",wmax,n,t); break; }
        double dt=cfl*h_min/std::max(wmax,1e-30); if(t+dt>t_end)dt=t_end-t; if(dt<=0)break;
        cur_t=t;
        // MOOD u2 smooth-extremum detector: density Hessian trace (curvature) over
        // self+neighbours; a cell is a SMOOTH extremum if the curvature has one sign
        // and varies mildly (min/max>0.5) -> such cells are NOT flagged for a DMP
        // violation (only genuine oscillations / PAD violations are). Recovers high
        // order at smooth extrema. (density = U[0], no prim conversion needed.)
        std::vector<double> densc(N); std::vector<char> u2smooth(N);
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ double rc=U[0*N+ci], cxx=0,cyy=0;
            for(int k=0;k<c2.max_nb;++k){ int nb=c2.nb[(size_t)ci*c2.max_nb+k]; if(nb<0)continue;
                double dW=U[0*N+nb]-rc;
                cxx+=c2.M[((size_t)ci*5+2)*c2.max_nb+k]*dW; cyy+=c2.M[((size_t)ci*5+3)*c2.max_nb+k]*dW; }
            densc[ci]=cxx+cyy; }
        #pragma omp parallel for
        for(int ci=0;ci<N;++ci){ double cmn=densc[ci],cmx=densc[ci];
            for(int f:m.cell_faces[ci]){ int o=m.face_owner[f],n=m.face_neighbour[f],nb=(o==ci)?n:o;
                if(nb<0)continue; cmn=std::min(cmn,densc[nb]); cmx=std::max(cmx,densc[nb]); }
            u2smooth[ci] = (cmn*cmx>0.0) &&
                (std::min(std::fabs(cmn),std::fabs(cmx))/std::max(std::fabs(cmx),1e-30) > 0.5); }
        std::fill(deg.begin(),deg.end(),(char)2);   // start every cell at P2
        std::vector<double> Unew(sz), U1(sz), U2(sz);
        int it=0; for(; it<8; ++it){
            rhs(U,L);  for(int i=0;i<sz;++i) U1[i]=U[i]+dt*L[i];
            rhs(U1,L); for(int i=0;i<sz;++i) U2[i]=0.75*U[i]+0.25*(U1[i]+dt*L[i]);
            rhs(U2,L); for(int i=0;i<sz;++i) Unew[i]=(1.0/3.0)*U[i]+(2.0/3.0)*(U2[i]+dt*L[i]);
            int troubled=0;
            for(int i=0;i<N;++i){ if(deg[i]==0) continue;
                double u[4]={Unew[0*N+i],Unew[1*N+i],Unew[2*N+i],Unew[3*N+i]},w[4];
                bool bad=false, padbad=false;
                if(!std::isfinite(u[0])||!std::isfinite(u[1])||!std::isfinite(u[2])||!std::isfinite(u[3])) { bad=true; padbad=true; }
                else { eq.cons_to_prim(u,w);
                    if(w[0]<floor||w[3]<floor) { bad=true; padbad=true; }
                    else { double mn,mx; nbminmax(U,i,mn,mx); double rl=drelax*(mx-mn)+1e-12;
                        if(w[0]<mn-rl||w[0]>mx+rl) bad=true; } }
                if(bad && !padbad && use_u2 && u2smooth[i]) bad=false;   // u2: forgive smooth-extremum DMP violation
                if(bad){ deg[i]--; troubled++; } }
            if(troubled==0) break;
        }
        U.swap(Unew); t+=dt;
    }
    return Solve2DResult{std::move(U),n,t};
}

} // namespace cfd
