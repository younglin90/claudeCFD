// cfd/bvd_mood2d.hpp — a-posteriori MOOD on the THINC-BVD reconstruction (BVD+MOOD,
// cf. Tann/Deng/Loubère/Xiao). Candidate = full THINC-BVD (reconstruct_bvd / cheng3);
// detect inadmissible cells (PAD: rho,p>0 & finite; optional relaxed DMP on density)
// and cascade them BVD(level1) -> constant/P0(level0, always admissible), recompute.
//
// OPTIMIZED: detection is PER SSP-RK3 STAGE (each stage output made admissible ->
// SSP convexity => final positive). The expensive cheng3 BVD is computed ONCE per
// stage (WL0/WR0 cached); the cascade only re-applies the P0 fallback + re-fluxes
// (cheap) + re-detects. All buffers pre-allocated. So clean steps cost ~1 base step;
// only troubled regions pay, and cheng3 never re-runs inside the cascade.
// Env: MOOD_FLOOR (PAD floor abs, def 1e-10), MOOD_DMP (>0 enables density DMP relax,
// def 0=PAD-only), MOOD_MAXIT (cascade iters/stage, def 4), BVD_MOOD_VERBOSE.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/euler2d.hpp"
#include "cfd/reconstruct2d.hpp"
#include "cfd/reconstruct_bvd.hpp"
#include "cfd/solver_euler2d.hpp"
#include "cfd/solver2d.hpp"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace cfd {

inline Solve2DResult solve_euler2d_bvd_mood(const Mesh& m, const Euler2D& eq,
        const std::vector<double>& U0, double t_end, double cfl,
        int max_steps, int flux, const std::vector<BC2D>* bcs,
        const ReconCtx& ctx, const ReconCtxO2& ctx_o2) {
    const int N=m.n_cells(), Nf=m.n_faces(), sz=4*N;
    const double floorv = []{ const char* e=std::getenv("MOOD_FLOOR"); return e?std::atof(e):1e-10; }();
    const double drelax = []{ const char* e=std::getenv("MOOD_DMP");   return e?std::atof(e):0.0;   }();
    const int    maxit  = []{ const char* e=std::getenv("MOOD_MAXIT"); return e?std::atoi(e):4;     }();
    const bool   do_dmp = drelax > 0.0;
    std::vector<double> U=U0, Wc(sz), WL0((size_t)4*Nf), WR0((size_t)4*Nf),   // P2 (BVD)
        WLm((size_t)4*Nf), WRm((size_t)4*Nf),                                  // P1 (MUSCL)
        WL((size_t)4*Nf), WR((size_t)4*Nf), Fall((size_t)4*Nf), L(sz), U1(sz), U2(sz), Unew(sz);
    std::vector<char> level(N);
    double cur_t=0.0; long long ncasc=0; int maxit_hit=0;
    double h_min=1e300;
    for(int i=0;i<N;++i){ double amax=0; for(int f:m.cell_faces[i]) amax=std::max(amax,m.face_areas[f]);
        h_min=std::min(h_min,m.cell_volumes[i]/std::max(amax,1e-30)); }
    auto nbminmax=[&](const std::vector<double>& s,int i,double&mn,double&mx){
        mn=mx=s[0*N+i];
        for(int f:m.cell_faces[i]){ int o=m.face_owner[f],nn=m.face_neighbour[f],nb=(o==i)?nn:o;
            if(nb<0)continue; double r=s[0*N+nb]; mn=std::min(mn,r); mx=std::max(mx,r); } };
    // one SSP-RK3 stage:  out = alpha*Ualpha + (1-alpha)*(s + dt*L(s)), with per-stage MOOD
    // cascade (cheng3 of `s` cached in WL0/WR0; only fallback+flux re-run). Uref = density
    // DMP reference (the step-start state). Returns false if a cell hits P0 and is STILL bad.
    auto do_stage=[&](const std::vector<double>& s, const std::vector<double>& Uref,
                      double alpha, const std::vector<double>& Ualpha, std::vector<double>& out, double dt){
        #pragma omp parallel for
        for(int i=0;i<N;++i){ double u[4]={s[0*N+i],s[1*N+i],s[2*N+i],s[3*N+i]},w[4];
            eq.cons_to_prim(u,w); for(int v=0;v<4;++v) Wc[(size_t)v*N+i]=w[v]; }
        reconstruct_bvd(m, ctx, ctx_o2, Wc, 4, WL0, WR0);          // P2 = cheng3 THINC-BVD, ONCE
        reconstruct_bj_vertex(m, ctx, Wc, 4, WLm, WRm, false);     // P1 = MLP-u MUSCL, ONCE
        std::fill(level.begin(),level.end(),(char)2);              // start every cell at P2
        int it=0;
        for(; it<=maxit; ++it){
            #pragma omp parallel for                               // cascade: 2=BVD, 1=MUSCL, 0=P0(cell avg)
            for(int f=0;f<Nf;++f){ int o=m.face_owner[f], nb=m.face_neighbour[f];
                int lo=level[o];
                for(int v=0;v<4;++v) WL[(size_t)v*Nf+f] = (lo==2)?WL0[(size_t)v*Nf+f] : (lo==1)?WLm[(size_t)v*Nf+f] : Wc[(size_t)v*N+o];
                if(nb>=0){ int ln=level[nb];
                    for(int v=0;v<4;++v) WR[(size_t)v*Nf+f] = (ln==2)?WR0[(size_t)v*Nf+f] : (ln==1)?WRm[(size_t)v*Nf+f] : Wc[(size_t)v*N+nb]; } }
            #pragma omp parallel for                               // flux (cheap re-run)
            for(int f=0;f<Nf;++f){ int o=m.face_owner[f], nb=m.face_neighbour[f];
                double nx=m.face_normals[f*2],ny=m.face_normals[f*2+1],area=m.face_areas[f];
                double wL[4],wR[4];
                for(int v=0;v<4;++v){ wL[v]=WL[(size_t)v*Nf+f]; if(nb>=0) wR[v]=WR[(size_t)v*Nf+f]; }
                if(nb<0){ // NO recon at ANY boundary face (2026-07-03): cell avg + BC ghost
                    for(int v=0;v<4;++v) wL[v]=Wc[(size_t)v*N+o];
                    int tag=m.face_bc_tag[f];
                    const BC2D* bc=(bcs&&tag>0&&tag<(int)bcs->size())?&(*bcs)[tag]:nullptr;
                    if(bc) apply_bc(*bc,wL,nx,ny,m.face_centers[f*2],m.face_centers[f*2+1],cur_t,wR);
                    else for(int v=0;v<4;++v) wR[v]=wL[v]; }
                double F[4];
                if(flux==FLUX_RROE)       rotated_roe_euler2d(eq,wL,wR,nx,ny,F);
                else if(flux==FLUX_RHLLC) rotated_hllc_euler2d(eq,wL,wR,nx,ny,F);
                else if(flux==FLUX_RSLAU2) rotated_slau2_euler2d(eq,wL,wR,nx,ny,F);
                else if(flux==FLUX_SLAU2) slau2_euler2d(eq,wL,wR,nx,ny,F);
                else if(flux==FLUX_HLLC)  hllc_euler2d(eq,wL,wR,nx,ny,F);
                else if(flux==FLUX_HLL)   hll_euler2d(eq,wL,wR,nx,ny,F);
                else                      llf_euler2d(eq,wL,wR,nx,ny,F);
                for(int v=0;v<4;++v) Fall[(size_t)v*Nf+f]=F[v]*area; }
            #pragma omp parallel for                               // L + stage combine
            for(int i=0;i<N;++i){ double acc[4]={0,0,0,0};
                for(int f:m.cell_faces[i]){ double s2=(m.face_owner[f]==i)?-1.0:1.0;
                    for(int v=0;v<4;++v) acc[v]+=s2*Fall[(size_t)v*Nf+f]; }
                double inv=1.0/m.cell_volumes[i];
                for(int v=0;v<4;++v) out[(size_t)v*N+i]=alpha*Ualpha[(size_t)v*N+i]
                    + (1.0-alpha)*(s[(size_t)v*N+i] + dt*acc[v]*inv); }
            int troubled=0;                                        // detect on stage output
            for(int i=0;i<N;++i){ if(level[i]==0) continue;
                double u[4]={out[0*N+i],out[1*N+i],out[2*N+i],out[3*N+i]},w[4];
                bool bad=false;
                if(!std::isfinite(u[0])||!std::isfinite(u[1])||!std::isfinite(u[2])||!std::isfinite(u[3])) bad=true;
                else { eq.cons_to_prim(u,w);
                    if(w[0]<floorv||w[3]<floorv) bad=true;
                    else if(do_dmp){ double mn,mx; nbminmax(Uref,i,mn,mx); double rl=drelax*(mx-mn)+1e-12;
                        if(w[0]<mn-rl||w[0]>mx+rl) bad=true; } }
                if(bad){ --level[i]; ++troubled; } }   // demote one level (2->1->0)
            ncasc+=troubled;
            if(troubled==0) break;
        }
        if(it>maxit_hit) maxit_hit=it;
    };
    double t=0.0; int n=0;
    if(const char*_e=std::getenv("CFD_MAXSTEP")){int _v=std::atoi(_e); if(_v>0&&_v<max_steps)max_steps=_v;}   // bench cap (per-cell timing)
    for(; n<max_steps && t<t_end; ++n){
        double wmax=0.0;
        #pragma omp parallel for reduction(max:wmax)
        for(int i=0;i<N;++i){ double u[4]={U[0*N+i],U[1*N+i],U[2*N+i],U[3*N+i]};
            double w=std::max(eq.max_wave_speed(u,1,0),eq.max_wave_speed(u,0,1)); if(w>wmax)wmax=w; }
        if(!std::isfinite(wmax)||wmax>1e12){ std::fprintf(stderr,"BVD-MOOD DIVERGED wmax=%.3e step %d t=%.4g\n",wmax,n,t); break; }
        double dt=cfl*h_min/std::max(wmax,1e-30); if(t+dt>t_end)dt=t_end-t; if(dt<=0)break;
        cur_t=t;
        do_stage(U,  U, 0.0,       U, U1,   dt);    // U1   = U + dt L(U)
        do_stage(U1, U, 0.75,      U, U2,   dt);    // U2   = 3/4 U + 1/4 (U1 + dt L(U1))
        do_stage(U2, U, 1.0/3.0,   U, Unew, dt);    // Unew = 1/3 U + 2/3 (U2 + dt L(U2))
        U.swap(Unew); t+=dt;
    }
    // BVD_CANDFLAG: export the last stage's final per-cell MOOD cascade level for the paper
    // diagnostic (2=BVD/P2 accepted, 1=demoted to MUSCL/P1, 0=demoted to P0/cell-avg).
    if(std::getenv("BVD_CANDFLAG")){ auto& ml=mood_level_flag(); ml.assign((size_t)N,2); for(int i=0;i<N;++i) ml[i]=(signed char)level[i]; }
    if(std::getenv("BVD_MOOD_VERBOSE")) std::fprintf(stderr,"BVD-MOOD: %d steps, %lld cascades-to-P0, max %d iters/stage\n", n, ncasc, maxit_hit);
    return Solve2DResult{std::move(U),n,t};
}

} // namespace cfd
