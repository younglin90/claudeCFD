// apps/octant3d_unstr_bench.cpp — Hoppe 2024 octant Riemann (DEFAULT Case-5 six-identical spiral,
//   Table-7, Fig-9 = MOST whirls "multiple connection whirls near center"; paper2 3D vortex case per
//   user 2026-07-11 after visual survey of Hoppe Figs 3/6/7/8/9/10/11; flux HLLC to minimize dissipation)
//   on a GENUINE
// UNSTRUCTURED mixed-cell 3D mesh (tetra/hexa/prism/pyramid), GAUSS-THINC reconstruction.
//   Mesh from tools/gen_mesh3d*.py (.umsh, set N3_MESH). IC by cell centroid (mesh-agnostic).
// Env: N3_MESH=<file.umsh> (required), N3_T(0.2), N3_CFL(0.4), N3_FLUX(LLF), N3_VTK, N3_CSV.
//      GAUSS via THINCQQ_GAUSS=1 (+THINCQQ_GAUSS_SKEW). MOOD via MOOD_ON.
#include "cfd/mesh_unstructured3d.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/solver_euler3d.hpp"
#include "cfd/reconstruct3d_o2_unstr.hpp"
#include "cfd/io_vtk.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

using namespace cfd;

int main() {
    Euler3D eq; eq.gamma = 1.4;
    const char* mp = std::getenv("N3_MESH");
    if (!mp) { std::fprintf(stderr, "set N3_MESH=<file.umsh>\n"); return 1; }
    bool ok=false; Mesh m = load_umsh_3d(mp, &ok);
    if (!ok) { std::fprintf(stderr, "cannot load mesh %s\n", mp); return 1; }
    const int N = m.n_cells();
    const double T   = std::getenv("N3_T")   ? std::atof(std::getenv("N3_T"))   : 0.2;
    const double cfl = std::getenv("N3_CFL") ? std::atof(std::getenv("N3_CFL")) : 0.4;

    static const double T7[8][5] = {   // Table 7: case5 spiral slip-lines, Fig 9 (p,rho,u,v,w)
        {1.0,1.0,0.5,-0.25,0.25},{1.0,0.5,0.5,0.25,-0.25},{1.0,2.0,-0.25,0.25,0.25},{1.0,0.5,-0.25,-0.25,-0.25},
        {1.0,0.5,0.375,-0.5,0.25},{1.0,2.0,0.375,0.5,-0.25},{1.0,0.5,0.25,0.5,0.25},{1.0,1.0,0.25,-0.5,-0.25}};
    static const double T1[8][5] = {   // Table 1: case1 (Config 11+15), Fig 3 (p,rho,u,v,w)
        {0.4,0.519705,0.825864,0.1,-0.1},{1.0,1.0,0.1,0.1,-0.1},{0.4,0.53125,0.1,0.827607,-0.1},{0.4,0.622341,0.1,0.1,-0.625862},
        {0.4,0.622341,0.1,-0.625862,-0.1},{0.4,0.53125,0.1,0.1,-0.827607},{0.4,0.622341,0.825862,0.1,-0.1},{1.068254,1.221896,0.1,0.1,-0.1}};
    static const double T6[8][5] = {   // Table 6: case5 differing sides, Fig 8 (p,rho,u,v,w)
        {1.0,1.0,0.25,-0.25,-0.5},{1.0,0.5,0.25,0.25,-0.25},{1.0,2.0,-0.25,0.25,0.25},{1.0,0.5,-0.25,-0.25,-0.25},
        {1.0,0.5,-0.25,-0.5,-0.5},{1.0,2.0,-0.25,0.5,-0.25},{1.0,0.5,0.25,0.5,0.25},{1.0,1.0,0.25,-0.5,-0.25}};
    static const double T9[8][5] = {   // Table 9: CASE 7 (Hoppe), Fig 11 = slip lines (J) + shocks (S). (p,rho,u,v,w)
        {0.1,0.104918,-0.15,-0.1,2.815576},{0.1,0.262295,-0.15,0.1,1.290990},{0.1,0.524590,0.15,0.1,1.425},{0.1,2.098361,0.15,-0.1,0.2625},
        {1.0,0.4,-0.15,-0.1,0.3},{1.0,1.0,-0.15,0.1,-0.3},{1.0,2.0,0.15,0.1,0.3},{1.0,8.0,0.15,-0.1,-0.3}};
    const int icase = std::getenv("N3_CASE") ? std::atoi(std::getenv("N3_CASE")) : 7;   // DEFAULT 7=Table7 Hoppe Case-5 spiral (paper2 vortex); 9=Table9 Case-7(slip+shock); 1=Table1 Case-1(shock); 6=Table6 Case-5 differing
    const double (*CASE5)[5] = (icase==1)?T1:(icase==6)?T6:(icase==9)?T9:T7;
    std::vector<double> U0((size_t)5*N);
    for (int c=0;c<N;++c){ double x=m.cell_centers[3*c],y=m.cell_centers[3*c+1],z=m.cell_centers[3*c+2];
        bool hx=x>0.5,hy=y>0.5,hz=z>0.5; int oct;
        if(hz) oct=(hx&&hy)?1:(!hx&&hy)?2:(!hx&&!hy)?3:4; else oct=(hx&&hy)?5:(!hx&&hy)?6:(!hx&&!hy)?7:8;
        const double* s=CASE5[oct-1]; double W[5]={s[1],s[2],s[3],s[4],s[0]},Uc[5]; eq.prim_to_cons(W,Uc);
        for(int v=0;v<5;++v) U0[(size_t)v*N+c]=Uc[v]; }
    // Boundary conditions. DEFAULT = TRANSMISSIVE zero-gradient (kind 0) — the literature standard for
    // multi-D Riemann verification (Schulz-Rinne/Lax-Liu/Kurganov, ALPACA). Each cube face hosts its own
    // 2D Riemann problem (Hoppe Fig-2 "six 2D cases on each side"); transmissive lets that evolve and lets
    // waves exit. OCT_DIRICHLET pins each boundary face ghost to the constant octant IC state — this was
    // the old default but it REFLECTS the face 2D-Riemann and injects SPURIOUS boundary pressure/vorticity
    // (verified 2026-07-11 A/B, Nx=48 Case-7 isobaric: Dirichlet boundary p in [0.865,1.298] & ens 22.2 vs
    // transmissive p in [0.943,1.077] & ens 16.4; transmissive stays STABLE, no p->95 blowup with HLLC+PPFLOOR).
    std::vector<BC3D> bcs(9);   // tag 0 transmissive; tags 1-8 = per-octant Dirichlet (opt-in)
    static const bool dirichlet = std::getenv("OCT_DIRICHLET") != nullptr;   // DEFAULT OFF (transmissive); opt-in OCT_DIRICHLET
    if (dirichlet) {
        auto oct_of=[&](double x,double y,double z)->int{ bool hx=x>0.5,hy=y>0.5,hz=z>0.5;
            return hz?((hx&&hy)?1:(!hx&&hy)?2:(!hx&&!hy)?3:4):((hx&&hy)?5:(!hx&&hy)?6:(!hx&&!hy)?7:8); };
        for(int o=1;o<=8;++o){ const double* s=CASE5[o-1]; bcs[o].kind=2;
            bcs[o].state[0]=s[1];bcs[o].state[1]=s[2];bcs[o].state[2]=s[3];bcs[o].state[3]=s[4];bcs[o].state[4]=s[0]; }
        for(int f=0; f<m.n_faces(); ++f){ if(m.face_neighbour[f]>=0) continue;
            m.face_bc_tag[f]=oct_of(m.face_centers[3*f],m.face_centers[3*f+1],m.face_centers[3*f+2]); }
    }
    // DEFAULT HLLC (user 2026-07-11: minimize dissipation so slip-line whirls survive; Hoppe Fig-9e,f
    // shows LLF/HLL damp the whirls). Case-7 GAUSS+HLLC verified positive (p_min 0.77). Fallback N3_FLUX=0 (LLF)
    // for positivity-sensitive cases (stage-0 predictor already uses LLF regardless).
    const int flux = std::getenv("N3_FLUX") ? std::atoi(std::getenv("N3_FLUX")) : (int)FLUX3_HLLC;
    const int recon = std::getenv("N3_RECON") ? std::atoi(std::getenv("N3_RECON")) : (int)RECON3_BVD;

    auto t0=std::chrono::steady_clock::now();
    Solve3DResult R = solve_euler3d(m, eq, U0, T, /*integrator*/2, cfl, -1.0, 100000000,
                                    recon, nullptr, flux, &bcs, nullptr);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();

    // primitives + diagnostics
    std::vector<double> frho(N),fu(N),fv(N),fw(N),fp(N);
    double rmin=1e300,rmax=-1e300,pmin=1e300; bool finite=true;
    for(int c=0;c<N;++c){ double Uc[5],W[5]; for(int v=0;v<5;++v)Uc[v]=R.U[(size_t)v*N+c];
        if(!std::isfinite(Uc[0]))finite=false; eq.cons_to_prim(Uc,W);
        frho[c]=W[0];fu[c]=W[1];fv[c]=W[2];fw[c]=W[3];fp[c]=W[4];
        rmin=std::min(rmin,W[0]);rmax=std::max(rmax,W[0]);pmin=std::min(pmin,W[4]); }
    // vorticity via o2 LSQ velocity gradients
    ReconCtx3DO2 o2 = build_recon_ctx_3d_o2_unstr(m);
    std::vector<double> Wp((size_t)5*N); for(int v=0;v<5;++v)for(int c=0;c<N;++c) Wp[(size_t)v*N+c]= v==0?frho[c]:v==1?fu[c]:v==2?fv[c]:v==3?fw[c]:fp[c];
    std::vector<double> gu,gv,gw; reconstruct3d_o2_coeffs(m,o2,Wp,5,1,gu); reconstruct3d_o2_coeffs(m,o2,Wp,5,2,gv); reconstruct3d_o2_coeffs(m,o2,Wp,5,3,gw);
    std::vector<double> fvort(N); double ens=0;
    for(int c=0;c<N;++c){ double ox=gw[9*c+1]-gv[9*c+2], oy=gu[9*c+2]-gw[9*c+0], oz=gv[9*c+0]-gu[9*c+1];
        double w2=ox*ox+oy*oy+oz*oz; fvort[c]=std::sqrt(w2); ens+=w2*m.cell_volumes[c]; }

    std::printf("OctantUnstr recon=bvd%s mesh=%s cells=%d rho=[%.4f,%.4f] p_min=%.4f finite=%d "
                "enstrophy=%.4f steps=%d t=%.4f wall=%.1fs%s\n",
                std::getenv("THINCQQ_GAUSS")?"+GAUSS":"", mp, N, rmin,rmax,pmin,(int)finite,
                ens, R.n_steps, R.t, wall, (pmin<=0||!finite)?"  [DIVERGED/neg-p]":"");

    std::vector<VtkField> flds={{"rho",frho.data()},{"u",fu.data()},{"v",fv.data()},{"w",fw.data()},{"p",fp.data()},{"vortmag",fvort.data()}};
    if(const char* vf=std::getenv("N3_VTK")){ write_vtk_unstructured(vf,m,flds); std::printf("VTK saved: %s\n",vf); }
    if(const char* cf=std::getenv("N3_CSV")){ FILE* fh=std::fopen(cf,"w"); if(fh){   // z~0.5 slab
        std::fprintf(fh,"x,y,z,rho,u,v,w,p,vortmag\n");
        for(int c=0;c<N;++c){ double z=m.cell_centers[3*c+2]; if(std::fabs(z-0.5)>0.5/ std::cbrt((double)N)) continue;
            std::fprintf(fh,"%.6g,%.6g,%.6g,%.7g,%.7g,%.7g,%.7g,%.7g,%.7g\n",
                m.cell_centers[3*c],m.cell_centers[3*c+1],z,frho[c],fu[c],fv[c],fw[c],fp[c],fvort[c]); }
        std::fclose(fh); std::printf("CSV saved: %s (z~0.5 slab)\n",cf); } }
    return 0;
}
