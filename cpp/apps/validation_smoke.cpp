// validation_smoke.cpp — feasibility smoke of BVD-paper validation problems not yet done.
// 1D cases on a thin 2D strip (Ny=4), 2D cases directly; all via solve_euler2d + THINC-BVD
// (RECON_BVD). Small mesh + short time. Reports: completed to t_end? posOK (rho,p>0 finite)?
// wall. NOT full validation (transmissive BC everywhere, no reference compare) — just whether
// the scheme RUNS. Env: VS_ONLY=<id> (single case), VS_CFL (def 0.3). Run with the usual
// BVD_CHENG3/MLP_U2/THINCQQ_SIG_DEG3T... env for the deg3t scheme.
#include "cfd/solver_euler2d.hpp"
#include "cfd/reconstruct2d.hpp"
#include "cfd/reconstruct_bvd.hpp"
#include "cfd/mesh.hpp"
#include "cfd/euler2d.hpp"
#include "cfd/io_vtk.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
using namespace cfd;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Case { const char* name; int Nx, Ny; double Lx, Ly, t_end, gamma; };

static void ic(int cs, double x, double y, double Lx, double Ly, double g, double W[4]) {
    double cx=0.5*Lx, cy=0.5*Ly;
    auto set=[&](double r,double u,double v,double p){W[0]=r;W[1]=u;W[2]=v;W[3]=p;};
    switch(cs){
    case 0:  if(x<cx) set(1,0,0,1); else set(0.125,0,0,0.1); break;                                   // Sod
    case 1:  if(x<cx) set(0.445,0.698,0,3.528); else set(0.5,0,0,0.571); break;                        // Lax
    case 2:  if(x<1.0) set(3.857143,2.629369,0,10.3333); else set(1.0+0.2*std::sin(5.0*(x-5.0)),0,0,1.0); break; // Shu-Osher
    case 3:  { double p=(x<0.1)?1000.0:(x>0.9?100.0:0.01); set(1,0,0,p); } break;                       // two-blast (WC)
    case 4:  if(x<cx) set(1,-2,0,0.4); else set(1,2,0,0.4); break;                                       // 123/Einfeldt
    case 5:  if(x<3.0) set(1.0,0,0,(g-1.0)*1.0*0.1); else set(0.001,0,0,(g-1.0)*0.001*1e-7); break;     // Le Blanc
    case 6:  { double dx=Lx/200.0; if(std::fabs(x-cx)<1.5*dx) set(1,0,0,(g-1.0)*1.0/(3.0*dx)); else set(1,0,0,1e-6); } break; // Sedov-1D
    case 7:  if(x>0.4*Lx&&x<0.6*Lx) set(1.4,1,0,1); else set(1.0,1,0,1); break;                          // isolated contact
    case 8:  set(1.0+0.5*std::sin(2.0*M_PI*x/Lx),1,0,1); break;                                          // density wave (order)
    case 9:  { bool R=x>=cx,T=y>=cy; if(T&&R)set(1.1,0,0,1.1); else if(T)set(0.5065,0.8939,0,0.35); else if(!R)set(1.1,0.8939,0.8939,1.1); else set(0.5065,0,0.8939,0.35);} break; // 2D Riemann cfg4
    case 10: { bool R=x>=cx,T=y>=cy; if(T&&R)set(1,0.75,-0.5,1); else if(T)set(2,0.75,0.5,1); else if(!R)set(1,-0.75,0.5,1); else set(3,-0.75,-0.5,1);} break;                     // cfg6 (4 contacts/KH)
    case 11: { bool R=x>=cx,T=y>=cy; if(T&&R)set(0.5313,0,0,0.4); else if(T)set(1,0.7276,0,1); else if(!R)set(0.8,0,0,1); else set(1,0,0.7276,1);} break;                          // cfg12
    case 12: { double r=std::hypot(x-cx,y-cy); if(r<0.4)set(1,0,0,1); else set(0.125,0,0,0.1);} break;   // explosion (circular)
    case 13: { if(std::fabs(x-cx)+std::fabs(y-cy)<0.15)set(0.125,0,0,0.14); else set(1,0,0,1);} break;   // implosion
    case 14: { double dx=Lx/64.0; if(std::hypot(x-cx,y-cy)<1.5*dx) set(1,0,0,(g-1.0)*1.0/(M_PI*1.5*dx*1.5*dx)); else set(1,0,0,1e-5);} break; // Sedov-2D
    case 15: { double r=std::hypot(x-cx,y-cy); if(r<1e-6)set(1,0,0,1e-6); else set(1,-(x-cx)/r,-(y-cy)/r,1e-6);} break; // Noh
    case 16: { if(x<0.5){ double xv=0.25,yv=0.5,eps=0.3,rc=0.05,dx=x-xv,dy=y-yv,r2=(dx*dx+dy*dy)/(rc*rc),ex=eps*std::exp(0.5*(1-r2)),dT=-(g-1)*eps*eps*std::exp(1-r2)/(2*g); set(std::pow(1+dT,1.0/(g-1)),1.1*std::sqrt(g)+ex*dy/rc,-ex*dx/rc,std::pow(1+dT,g/(g-1))); } else set(1.169,1.114,0,1.245);} break; // shock-vortex
    default: set(1,0,0,1);
    }
}

int main(){
    static const Case cases[] = {
        {"sod_1d",200,4,1.0,0.05,0.20,1.4},        {"lax_1d",200,4,1.0,0.05,0.16,1.4},
        {"shuosher_1d",300,4,10.0,0.05,1.8,1.4},   {"twoblast_1d",200,4,1.0,0.05,0.038,1.4},
        {"einfeldt123_1d",200,4,1.0,0.05,0.15,1.4},{"leblanc_1d",300,4,9.0,0.05,6.0,5.0/3.0},
        {"sedov_1d",200,4,2.0,0.05,1.0,1.4},       {"contact_1d",200,4,1.0,0.05,0.5,1.4},
        {"denswave_1d",100,4,1.0,0.05,2.0,1.4},
        {"riemann_cfg4_2d",64,64,1.0,1.0,0.25,1.4},{"riemann_cfg6_2d",64,64,1.0,1.0,0.3,1.4},
        {"riemann_cfg12_2d",64,64,1.0,1.0,0.25,1.4},{"explosion_2d",64,64,1.5,1.5,0.2,1.4},
        {"implosion_2d",64,64,0.3,0.3,1.0,1.4},    {"sedov_2d",64,64,2.0,2.0,1.0,1.4},
        {"noh_2d",64,64,1.0,1.0,0.6,1.4},          {"shockvortex_2d",100,50,2.0,1.0,0.35,1.4},
    };
    const int NCASE = (int)(sizeof(cases)/sizeof(cases[0]));
    int only = std::getenv("VS_ONLY")?std::atoi(std::getenv("VS_ONLY")):-1;
    double cfl = std::getenv("VS_CFL")?std::atof(std::getenv("VS_CFL")):0.3;
    std::printf("%-18s %-9s %-6s %8s  %s\n","case","grid","status","wall(s)","diagnostics");
    for(int cs=0; cs<NCASE; ++cs){
        if(only>=0 && cs!=only) continue;
        const Case& C = cases[cs];
        int nsc = std::getenv("VS_NSCALE")?std::atoi(std::getenv("VS_NSCALE")):1; if(nsc<1)nsc=1;
        Mesh m = triangulate_box(C.Nx*nsc, C.Ny*nsc, C.Lx, C.Ly);
        Euler2D eq{C.gamma};
        int NC = m.n_cells();
        std::vector<double> U0((size_t)4*NC);
        for(int i=0;i<NC;++i){ double W[4]; ic(cs, m.cell_centers[i*2], m.cell_centers[i*2+1], C.Lx, C.Ly, C.gamma, W);
            double U[4]; eq.prim_to_cons(W,U); for(int v=0;v<4;++v) U0[(size_t)v*NC+i]=U[v]; }
        ReconCtx bj = build_recon_ctx(m, 0.0);
        ReconCtxO2 c2 = build_recon_ctx_o2(m);
        int recon = RECON_BVD;   // VS_RECON=mlp -> pure MLP-u MUSCL baseline (isolate BVD/THINC layer)
        if(const char* rs=std::getenv("VS_RECON")){ std::string s=rs; if(s=="mlp") recon=RECON_BJ_VERTEX; }
        int flux = FLUX_RHLLC;   // VS_FLUX=llf|hllc|hll|rroe|slau2|rslau2 (isolate flux as cause)
        if(const char* fs=std::getenv("VS_FLUX")){ std::string s=fs; if(s=="llf")flux=FLUX_LLF; else if(s=="hllc")flux=FLUX_HLLC; else if(s=="hll")flux=FLUX_HLL; else if(s=="rroe")flux=FLUX_RROE; else if(s=="slau2")flux=FLUX_SLAU2; else if(s=="rslau2")flux=FLUX_RSLAU2; }
        // Per-case BCs. shockvortex_2d: the LEFT boundary is a SUPERSONIC INFLOW (M=1.1) and
        // MUST be pinned by a Dirichlet BC — the old bcs=nullptr default made every boundary
        // transmissive (ghost = interior recon), so the inflow was held only by inertia and
        // low-dissipation schemes accumulated a spurious standing vorticity dipole at the
        // boundary (diagnosed 2026-07-03). Tags: 1=left(x_min), 2=right, 3=y_min, 4=y_max.
        std::vector<BC2D> case_bcs;
        const std::vector<BC2D>* bcp = nullptr;
        if (std::string(C.name) == "shockvortex_2d") {
            case_bcs.assign(5, BC2D{});                     // others stay kind 0 (transmissive)
            case_bcs[1].kind = 2;                           // supersonic inflow: fix upstream state
            case_bcs[1].state[0] = 1.0; case_bcs[1].state[1] = 1.1*std::sqrt(1.4);
            case_bcs[1].state[2] = 0.0; case_bcs[1].state[3] = 1.0;
            bcp = &case_bcs;
        }
        auto t0 = std::chrono::steady_clock::now();
        Solve2DResult r = solve_euler2d(m, eq, U0, C.t_end, 2, cfl, -1.0, 100000000, recon, &bj, flux, bcp, nullptr, &c2);
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        double rmin=1e30,pmin=1e30; bool fin=true;
        for(int i=0;i<NC;++i){ double U[4]={r.U[0*NC+i],r.U[1*NC+i],r.U[2*NC+i],r.U[3*NC+i]},W[4];
            if(!std::isfinite(U[0])||!std::isfinite(U[3])){fin=false;continue;} eq.cons_to_prim(U,W);
            if(W[0]<rmin)rmin=W[0]; if(W[3]<pmin)pmin=W[3]; }
        bool completed = r.t >= C.t_end*0.999;
        bool ok = fin && completed && rmin>0 && pmin>0;
        std::printf("%-18s %dx%-6d %-6s %8.1f  rho_min=%.2e p_min=%.2e steps=%d t=%.3f/%.3f\n",
            C.name, C.Nx, C.Ny, ok?"OK":(completed?"NEG":"DIVERGE"), wall, rmin, pmin, r.n_steps, r.t, C.t_end);
        std::printf("[WALL] %s wall=%.3fs steps=%d\n", C.name, wall, r.n_steps);
        std::fflush(stdout);
        if(const char* df=std::getenv("VS_DUMP")){ FILE* fh=std::fopen(df,"w");
            if(fh){ for(int i=0;i<NC;++i){ double U[4]={r.U[0*NC+i],r.U[1*NC+i],r.U[2*NC+i],r.U[3*NC+i]},W[4];
                eq.cons_to_prim(U,W); std::fprintf(fh,"%.6g %.6g %.6g\n", m.cell_centers[i*2], m.cell_centers[i*2+1], W[0]); }
                std::fclose(fh); std::printf("  dumped %s\n", df); std::fflush(stdout);
                cfd::write_vtk2d_euler(std::string(df)+".vtk", m, eq, r.U); } }
    }
    return 0;
}
