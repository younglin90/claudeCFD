// apps/octant3d_bench.cpp — Hoppe, Fleischmann, Biller, Adami, Adams (2024),
// Computers & Fluids 278: genuine 3D Riemann problems on the unit cube.
//
//   Unit cube [0,1]^3 split into 8 octants at x=y=z=0.5, each a constant state so
//   adjacent octants form a single elementary wave. gamma=1.4, t_end=0.2, all faces
//   transmissive. Reference: ALPACA, HLLC+WENO5, ~1024^3.
//   Octant numbering (Lax-Liu 2D extended): z>0.5 -> {1:(x>,y>),2:(x<,y>),3:(x<,y<),
//   4:(x>,y<)};  z<0.5 -> {5,6,7,8} same xy.  Table column order = (p, rho, u, v, w).
//
//   N3_CASE=1  : Case 1 (Table 1) — Config-11/15, mixed shocks+slips+rarefactions.
//   N3_CASE=5  : "alternative case 5" (Table 7) — all-12 polytropic slip lines
//                (uniform p=1), spiral roll-up; THE vortex/recon discriminator
//                (HLLC keeps the whirls, dissipative recon damps them).
//
// Env: N3_N (cube N^3, default 64), N3_RECON ("bj"|"bvd"; GAUSS via THINCQQ_GAUSS=1),
//      N3_CASE (1|5, default 5), N3_T (t_end, default 0.2), N3_CFL (0.4), N3_DUMP.
#include "cfd/mesh.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/solver_euler3d.hpp"
#include "cfd/io_vtk.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

using namespace cfd;

int main() {
    Euler3D eq; eq.gamma = 1.4;
    const int Ncube = std::getenv("N3_N") ? std::atoi(std::getenv("N3_N")) : 64;
    const double T = std::getenv("N3_T") ? std::atof(std::getenv("N3_T")) : 0.2;
    const double cfl = std::getenv("N3_CFL") ? std::atof(std::getenv("N3_CFL")) : 0.4;
    const int kase = std::getenv("N3_CASE") ? std::atoi(std::getenv("N3_CASE")) : 5;
    const char* rec = std::getenv("N3_RECON");
    const bool use_bvd = rec && std::strcmp(rec, "bvd") == 0;
    const int recon = use_bvd ? RECON3_BVD : RECON3_BJ_VERTEX;

    // octant initial states, column order (p, rho, u, v, w), index [octant-1].
    static const double CASE1[8][5] = {
        {0.4,      0.519705, 0.825864,  0.1,      -0.1},
        {1.0,      1.0,      0.1,       0.1,      -0.1},
        {0.4,      0.53125,  0.1,       0.827607, -0.1},
        {0.4,      0.622341, 0.1,       0.1,      -0.625862},
        {0.4,      0.622341, 0.1,      -0.625862, -0.1},
        {0.4,      0.53125,  0.1,       0.1,      -0.827607},
        {0.4,      0.622341, 0.825862,  0.1,      -0.1},
        {1.068254, 1.221896, 0.1,       0.1,      -0.1}};
    static const double CASE5[8][5] = {   // Table 7 "alternative case 5" — slip-line spiral (vortex)
        {1.0, 1.0,  0.5,   -0.25,  0.25},
        {1.0, 0.5,  0.5,    0.25, -0.25},
        {1.0, 2.0, -0.25,   0.25,  0.25},
        {1.0, 0.5, -0.25,  -0.25, -0.25},
        {1.0, 0.5,  0.375, -0.5,   0.25},
        {1.0, 2.0,  0.375,  0.5,  -0.25},
        {1.0, 0.5,  0.25,   0.5,   0.25},
        {1.0, 1.0,  0.25,  -0.5,  -0.25}};
    const double (*S)[5] = (kase == 1) ? CASE1 : CASE5;

    Mesh m = build_structured_3d(Ncube, Ncube, Ncube, 1.0, 1.0, 1.0, false, false, false);
    const int N = m.n_cells();
    std::vector<double> U0((size_t)5 * N);
    for (int c = 0; c < N; ++c) {
        double x = m.cell_centers[(size_t)c*3+0], y = m.cell_centers[(size_t)c*3+1],
               z = m.cell_centers[(size_t)c*3+2];
        bool hx = x>0.5, hy = y>0.5, hz = z>0.5; int oct;
        if (hz) oct = (hx&&hy)?1 : (!hx&&hy)?2 : (!hx&&!hy)?3 : 4;
        else    oct = (hx&&hy)?5 : (!hx&&hy)?6 : (!hx&&!hy)?7 : 8;
        const double* s = S[oct-1];                          // (p,rho,u,v,w)
        double W[5] = {s[1], s[2], s[3], s[4], s[0]}, Uc[5];  // -> (rho,u,v,w,p)
        eq.prim_to_cons(W, Uc);
        for (int v = 0; v < 5; ++v) U0[(size_t)v*N + c] = Uc[v];
    }
    std::vector<BC3D> bcs(7);   // all 6 faces transmissive (kind 0 default)

    ReconCtx3D ctx = build_recon_ctx_3d(m);
    ReconCtx3DO2 o2ctx; const ReconCtx3DO2* o2p = nullptr;
    if (use_bvd) { o2ctx = build_recon_ctx_3d_o2(m); o2p = &o2ctx; }

    // LLF (Rusanov) flux by default for octant: most dissipative -> most ROBUST (no
    // carbuncle / no rarefaction-shock), keeps these strong octant interactions stable.
    const int flux = std::getenv("N3_FLUX") ? std::atoi(std::getenv("N3_FLUX")) : (int)FLUX3_LLF;
    auto t0 = std::chrono::steady_clock::now();
    Solve3DResult R = solve_euler3d(m, eq, U0, T, /*integrator*/2, cfl, -1.0,
                                    100000000, recon, &ctx, flux, &bcs, o2p);
    auto t1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();

    auto idx = [&](int i,int j,int k){ return (size_t)((k*Ncube + j)*Ncube + i); };
    auto vel = [&](size_t c,double&u,double&v,double&w){ double r=R.U[(size_t)0*N+c];
        u=R.U[(size_t)1*N+c]/r; v=R.U[(size_t)2*N+c]/r; w=R.U[(size_t)3*N+c]/r; };
    double rmin=1e300,rmax=-1e300,pmin=1e300; bool finite=true; double ens=0.0;
    double h=1.0/Ncube;
    for (int c=0;c<N;++c){ double r=R.U[(size_t)0*N+c]; if(!std::isfinite(r))finite=false;
        double W[5],Uc[5]; for(int v=0;v<5;++v)Uc[v]=R.U[(size_t)v*N+c]; eq.cons_to_prim(Uc,W);
        if(r<rmin)rmin=r; if(r>rmax)rmax=r; if(W[4]<pmin)pmin=W[4]; }
    for(int k=1;k<Ncube-1;++k)for(int j=1;j<Ncube-1;++j)for(int i=1;i<Ncube-1;++i){
        double up,um,vp,vm,wp,wm;
        vel(idx(i+1,j,k),up,vp,wp); vel(idx(i-1,j,k),um,vm,wm); double dvdx=(vp-vm)/(2*h),dwdx=(wp-wm)/(2*h);
        vel(idx(i,j+1,k),up,vp,wp); vel(idx(i,j-1,k),um,vm,wm); double dudy=(up-um)/(2*h),dwdy=(wp-wm)/(2*h);
        vel(idx(i,j,k+1),up,vp,wp); vel(idx(i,j,k-1),um,vm,wm); double dudz=(up-um)/(2*h),dvdz=(vp-vm)/(2*h);
        double ox=dwdy-dvdz,oy=dudz-dwdx,oz=dvdx-dudy; ens+=(ox*ox+oy*oy+oz*oz)*(h*h*h); }
    std::printf("Octant3D case=%d recon=%s%s N=%d^3(%d) rho=[%.4f,%.4f] p_min=%.4f finite=%d "
                "enstrophy=%.4f steps=%d t=%.4f wall=%.1fs%s\n",
                kase, use_bvd?"bvd":"bj", (use_bvd&&std::getenv("THINCQQ_GAUSS"))?"+GAUSS":"",
                Ncube,N, rmin,rmax,pmin,(int)finite, ens, R.n_steps, R.t, wall,
                (pmin<=0||!finite)?"  [DIVERGED/neg-p]":"");

    // ── paper field output: VTK (full 3D volume) + CSV (z=0.5 pinwheel plane), primitives + |vorticity| ──
    std::vector<double> frho(N),fu(N),fv(N),fw(N),fp(N),fvort(N,0.0);
    for(int c=0;c<N;++c){ double W[5],Uc[5]; for(int v=0;v<5;++v)Uc[v]=R.U[(size_t)v*N+c];
        eq.cons_to_prim(Uc,W); frho[c]=W[0];fu[c]=W[1];fv[c]=W[2];fw[c]=W[3];fp[c]=W[4]; }
    for(int k=1;k<Ncube-1;++k)for(int j=1;j<Ncube-1;++j)for(int i=1;i<Ncube-1;++i){
        double up,um,vp,vm,wp,wm;
        vel(idx(i+1,j,k),up,vp,wp);vel(idx(i-1,j,k),um,vm,wm);double dvdx=(vp-vm)/(2*h),dwdx=(wp-wm)/(2*h);
        vel(idx(i,j+1,k),up,vp,wp);vel(idx(i,j-1,k),um,vm,wm);double dudy=(up-um)/(2*h),dwdy=(wp-wm)/(2*h);
        vel(idx(i,j,k+1),up,vp,wp);vel(idx(i,j,k-1),um,vm,wm);double dudz=(up-um)/(2*h),dvdz=(vp-vm)/(2*h);
        double ox=dwdy-dvdz,oy=dudz-dwdx,oz=dvdx-dudy; fvort[idx(i,j,k)]=std::sqrt(ox*ox+oy*oy+oz*oz);}
    std::vector<VtkField> flds={{"rho",frho.data()},{"u",fu.data()},{"v",fv.data()},{"w",fw.data()},{"p",fp.data()},{"vortmag",fvort.data()}};
    if(const char* vf=std::getenv("N3_VTK")){ write_vtk_image(vf,Ncube,Ncube,Ncube,h,h,h,flds); std::printf("VTK saved: %s\n",vf); }
    if(const char* cf=std::getenv("N3_CSV")){ FILE* fh=std::fopen(cf,"w"); if(fh){ int k=Ncube/2;
        std::fprintf(fh,"x,y,rho,u,v,w,p,vortmag\n");
        for(int j=0;j<Ncube;++j)for(int i=0;i<Ncube;++i){ size_t c=idx(i,j,k);
            std::fprintf(fh,"%.6g,%.6g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g\n",
                m.cell_centers[c*3+0],m.cell_centers[c*3+1],frho[c],fu[c],fv[c],fw[c],fp[c],fvort[c]); }
        std::fclose(fh); std::printf("CSV saved: %s (z=0.5 slice %dx%d)\n",cf,Ncube,Ncube); } }
    const char* df = std::getenv("N3_DUMP");
    if (df) { FILE* fh=std::fopen(df,"w"); if(fh){ int k=Ncube/2;   // z=0.5 mid-plane (x-y) density
        for(int j=0;j<Ncube;++j) for(int i=0;i<Ncube;++i){ size_t c=idx(i,j,k);
            std::fprintf(fh,"%.6g %.6g %.6g\n", m.cell_centers[c*3+0], m.cell_centers[c*3+1], R.U[(size_t)0*N+c]); }
        std::fclose(fh); } }
    return 0;
}
