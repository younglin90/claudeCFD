// apps/viscshock3d_unstr_bench.cpp — Daru-Tenaud VISCOUS SHOCK TUBE on a GENUINE
// UNSTRUCTURED 3D mesh (GAUSS-THINC + Navier-Stokes viscous flux). Domain [0,1]x[0,0.5]xthin;
// diaphragm at x=0.5: left rho=120,p=120/gamma ; right rho=1.2,p=1.2/gamma ; u=0. Strong
// shock (pressure ratio 100) + no-slip walls -> boundary-layer / shock-wall vortex.
//   BCs: x=0, x=1, y=0 no-slip adiabatic walls (kind 4); y=0.5 + z faces symmetry (kind 1).
//   Re=200 (mu=0.005), Pr=0.73, t_end=1.0. Env: N3_MESH(req), N3_T(1.0), N3_MU(0.005),
//   N3_CFL(0.3), N3_VTK, N3_CSV. GAUSS via THINCQQ_GAUSS=1, robustness BVD_BETA_L=0.8.
#include "cfd/mesh_unstructured3d.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/solver_euler3d.hpp"
#include "cfd/viscous3d.hpp"
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
    if (!mp) { std::fprintf(stderr, "set N3_MESH=<file.umsh on [1,0.5,thin]>\n"); return 1; }
    bool ok=false; Mesh m = load_umsh_3d(mp, &ok);
    if (!ok) { std::fprintf(stderr, "cannot load %s\n", mp); return 1; }
    const int N = m.n_cells();
    // domain extents from nodes
    double Lx=0,Ly=0,Lz=0; for (size_t i=0;i<m.nodes.size()/3;++i){ Lx=std::max(Lx,m.nodes[3*i]); Ly=std::max(Ly,m.nodes[3*i+1]); Lz=std::max(Lz,m.nodes[3*i+2]); }
    const double T   = std::getenv("N3_T")   ? std::atof(std::getenv("N3_T"))   : 1.0;
    const double cfl = std::getenv("N3_CFL") ? std::atof(std::getenv("N3_CFL")) : 0.3;
    const double mu  = std::getenv("N3_MU")  ? std::atof(std::getenv("N3_MU"))  : 0.005;

    // IC: Daru-Tenaud (default, rho 120/1.2, p ratio 100) or milder viscous Sod (N3_SOD=1).
    const bool sod = std::getenv("N3_SOD") != nullptr;
    std::vector<double> U0((size_t)5*N);
    for (int c=0;c<N;++c){ double x=m.cell_centers[3*c]; bool L=(x<0.5*Lx); double rho,p;
        if(sod){ rho=L?1.0:0.125; p=L?1.0:0.1; }
        else   { rho=L?120.0:1.2; p=(L?120.0:1.2)/eq.gamma; }
        double W[5]={rho,0,0,0,p},Uc[5]; eq.prim_to_cons(W,Uc);
        for(int v=0;v<5;++v) U0[(size_t)v*N+c]=Uc[v]; }

    // BC tags: x=0,x=1,y=0 -> no-slip adiabatic wall (4); y=Ly + z faces -> symmetry (1).
    for (int f=0; f<m.n_faces(); ++f){ if(m.face_neighbour[f]>=0) continue;
        double fx=m.face_centers[3*f],fy=m.face_centers[3*f+1],fz=m.face_centers[3*f+2]; int tag=2;
        if(fx<1e-6)tag=1; else if(fx>Lx-1e-6)tag=2; else if(fy<1e-6)tag=3; else if(fy>Ly-1e-6)tag=4;
        else if(fz<1e-6)tag=5; else if(fz>Lz-1e-6)tag=6; m.face_bc_tag[f]=(int8_t)tag; }
    std::vector<BC3D> bcs(7);
    bcs[1].kind=4; bcs[2].kind=4; bcs[3].kind=4;   // x=0, x=1, y=0  no-slip adiabatic walls
    bcs[4].kind=1; bcs[5].kind=1; bcs[6].kind=1;   // y=0.5, z faces  symmetry (slip)
    ViscousParams visc; visc.mu=mu; visc.Pr=0.73; visc.R=1.0;
    const int flux = std::getenv("N3_FLUX") ? std::atoi(std::getenv("N3_FLUX")) : (int)FLUX3_LLF;

    auto t0=std::chrono::steady_clock::now();
    Solve3DResult R = solve_euler3d(m, eq, U0, T, /*integrator*/2, cfl, -1.0, 100000000,
                                    RECON3_BVD, nullptr, flux, &bcs, nullptr, &visc);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();

    std::vector<double> frho(N),fu(N),fv(N),fw(N),fp(N);
    double rmin=1e300,rmax=-1e300,pmin=1e300; bool finite=true;
    for(int c=0;c<N;++c){ double Uc[5],W[5]; for(int v=0;v<5;++v)Uc[v]=R.U[(size_t)v*N+c];
        if(!std::isfinite(Uc[0]))finite=false; eq.cons_to_prim(Uc,W);
        frho[c]=W[0];fu[c]=W[1];fv[c]=W[2];fw[c]=W[3];fp[c]=W[4];
        rmin=std::min(rmin,W[0]);rmax=std::max(rmax,W[0]);pmin=std::min(pmin,W[4]); }

    std::printf("ViscShockUnstr recon=bvd%s mesh=%s cells=%d mu=%.4g rho=[%.3f,%.3f] p_min=%.4f finite=%d "
                "steps=%d t=%.4f wall=%.1fs%s\n",
                std::getenv("THINCQQ_GAUSS")?"+GAUSS":"", mp, N, mu, rmin,rmax,pmin,(int)finite,
                R.n_steps, R.t, wall, (pmin<=0||!finite)?"  [DIVERGED/neg-p]":"");

    std::vector<VtkField> flds={{"rho",frho.data()},{"u",fu.data()},{"v",fv.data()},{"w",fw.data()},{"p",fp.data()}};
    if(const char* vf=std::getenv("N3_VTK")){ write_vtk_unstructured(vf,m,flds); std::printf("VTK saved: %s\n",vf); }
    if(const char* cf=std::getenv("N3_CSV")){ FILE* fh=std::fopen(cf,"w"); if(fh){
        std::fprintf(fh,"x,y,rho,u,v,p\n");
        for(int c=0;c<N;++c){ double z=m.cell_centers[3*c+2]; if(std::fabs(z-0.5*Lz)>0.5*Lz/std::cbrt((double)N)+1e-9) continue;
            std::fprintf(fh,"%.6g,%.6g,%.7g,%.7g,%.7g,%.7g\n",m.cell_centers[3*c],m.cell_centers[3*c+1],frho[c],fu[c],fv[c],fp[c]); }
        std::fclose(fh); std::printf("CSV saved: %s (mid-z slab)\n",cf); } }
    return 0;
}
