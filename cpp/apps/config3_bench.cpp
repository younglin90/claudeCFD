// config3_bench.cpp — 2D four-shock Riemann problem (Schulz-Rinne Config 3, Balsara setup:
// domain [0,2]², t_end=1.1, EXACT Rankine-Hugoniot fractions). 4 constant quadrants; all four interfaces are
// shocks that collide at the center -> a mushroom jet whose SLIP LINES (contacts born at the
// shock triple points) go Kelvin-Helmholtz unstable. Same physics class as the Mach-stem
// slip line, simpler geometry. The IC is EXACTLY symmetric under (x,y)->(y,x),(u,v)->(v,u),
// so the exact solution has ρ(x,y)=ρ(y,x): a numeric symmetry metric flags grid-bias /
// spurious (asymmetric) vortices = a free legitimacy check. No analytic solution -> judge by
// symmetry + enstrophy + grid convergence + fine reference. Compares mlp_u1 (RECON_BJ_VERTEX)
// vs T-MLP-u (RECON_BVD). Args: [N] [t_end].  γ=1.4, transmissive BC, default t=0.3.
#include "cfd/solver_euler2d.hpp"
#include "cfd/bvd_mood2d.hpp"
#include "cfd/diagnostics.hpp"
#include "cfd/io_vtk.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

using namespace cfd;
static const double GAMMA = 1.4;
static const std::vector<BC2D>* g_c3_bcs = nullptr;   // C3_BC=dirichlet -> hold IC at all boundaries (ref 4.3.1 "Dirichlet BC")

// Configuration 3 quadrant states: {rho,u,v,p}, EXACT Rankine-Hugoniot fractions (not the
// rounded 5-digit decimals, which create measurable artifacts). Domain [0,2]^2 about (1,1),
// t_end=1.1 (relevant region covers >1/4 of domain). (1)TR (2)TL (3)BL (4)BR.
static void config3_W(double x, double y, double W[4]) {
    static const double uS = 4.0/std::sqrt(11.0);   // = 1.20605...
    if (x >= 0.8 && y >= 0.8)      { W[0]=1.5;        W[1]=0.0; W[2]=0.0; W[3]=1.5;        }  // 1 TR
    else if (x < 0.8 && y >= 0.8)  { W[0]=33.0/62.0;  W[1]=uS;  W[2]=0.0; W[3]=0.3;        }  // 2 TL
    else if (x < 0.8 && y < 0.8)   { W[0]=77.0/558.0; W[1]=uS;  W[2]=uS;  W[3]=9.0/310.0;  }  // 3 BL
    else                           { W[0]=33.0/62.0;  W[1]=0.0; W[2]=uS;  W[3]=0.3;        }  // 4 BR
}

// Exact config-3 boundary with MOVING shocks: each boundary's shock translates at its
// Rankine-Hugoniot speed so the held boundary tracks the analytic solution (valid until a shock
// reaches a corner ~t=1.2). |S|: TR-adjacent shocks (TR|TL,TR|BR) = 0.66334, BL-adjacent
// (TL|BL,BR|BL) = 0.42211 (verified via RH). All four shocks travel toward the lower-left.
static void config3_W_moving(double x, double y, double t, double W[4]) {
    static const double uS = 4.0/std::sqrt(11.0);
    const double Sf = 0.66334, Ss = 0.42211;
    auto TR=[&]{W[0]=1.5;        W[1]=0.0; W[2]=0.0; W[3]=1.5;       };
    auto TL=[&]{W[0]=33.0/62.0;  W[1]=uS;  W[2]=0.0; W[3]=0.3;       };
    auto BL=[&]{W[0]=77.0/558.0; W[1]=uS;  W[2]=uS;  W[3]=9.0/310.0; };
    auto BR=[&]{W[0]=33.0/62.0;  W[1]=0.0; W[2]=uS;  W[3]=0.3;       };
    const double e = 1e-6;
    if      (y > 1.0 - e) { if (x > 0.8 - Sf*t) TR(); else TL(); }  // top:    TR|TL moves left  (Sf)
    else if (x > 1.0 - e) { if (y > 0.8 - Sf*t) TR(); else BR(); }  // right:  TR|BR moves down  (Sf)
    else if (x < e)       { if (y > 0.8 - Ss*t) TL(); else BL(); }  // left:   TL|BL moves down  (Ss)
    else                  { if (x > 0.8 - Ss*t) BR(); else BL(); }  // bottom: BR|BL moves left  (Ss)
}

struct Diag { double ens, sym, rho_min, p_min; int n_steps; };

// rising-diagonal triangulate_box(N,N): cell idx = 2*(j*N+i)+s. Diagonal reflection
// (i,j)->(j,i), sub-triangle s<->1-s. ρ at idx should equal ρ at reflected idx.
static Diag run(const Mesh& m, const Euler2D& eq, const std::vector<double>& U0, int N,
                double t_end, int recon, const ReconCtx* ctx, const ReconCtxO2* ctx2,
                const ReconCtx& mc, FluxKind flux, const char* dump=nullptr,
                const char* label=nullptr) {
    static const double c3cfl = std::getenv("C3_CFL") ? std::atof(std::getenv("C3_CFL")) : 0.4;
    static const bool c3mood = std::getenv("C3_MOOD") != nullptr;   // a-posteriori MOOD-on-BVD
    auto t0_ = std::chrono::steady_clock::now();
    Solve2DResult r = (c3mood && recon==RECON_BVD && ctx && ctx2)
        ? solve_euler2d_bvd_mood(m, eq, U0, t_end, c3cfl, 100000000, (int)flux, g_c3_bcs, *ctx, *ctx2)
        : solve_euler2d(m, eq, U0, t_end, 2, c3cfl, -1.0, 100000000,
                        recon, ctx, flux, g_c3_bcs, nullptr, ctx2);
    double wall_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    std::printf("[WALL] %s wall=%.3fs steps=%d\n", label ? label : "run", wall_, r.n_steps);
    std::fflush(stdout);
    const int NC = m.n_cells();
    Diag d{0,0,1e9,1e9,r.n_steps};
    std::vector<double> rho(NC), u(NC), v(NC);
    for (int i=0;i<NC;++i){ double U[4]={r.U[0*NC+i],r.U[1*NC+i],r.U[2*NC+i],r.U[3*NC+i]},W[4];
        eq.cons_to_prim(U,W); rho[i]=W[0]; u[i]=W[1]; v[i]=W[2];
        d.rho_min=std::min(d.rho_min,W[0]); d.p_min=std::min(d.p_min,W[3]); }
    // enstrophy = sum 0.5*omega^2*vol via LSQ gradient
    for (int ci=0;ci<NC;++ci){ double ru0=0,ru1=0,rv0=0,rv1=0;
        for(int k=0;k<mc.max_nb;++k){ int nb=mc.nb[(size_t)ci*mc.max_nb+k]; if(nb<0)continue;
            double wk=mc.w[(size_t)ci*mc.max_nb+k];
            double dx=mc.d[((size_t)ci*mc.max_nb+k)*2+0], dy=mc.d[((size_t)ci*mc.max_nb+k)*2+1];
            ru0+=wk*dx*(u[nb]-u[ci]); ru1+=wk*dy*(u[nb]-u[ci]);
            rv0+=wk*dx*(v[nb]-v[ci]); rv1+=wk*dy*(v[nb]-v[ci]); }
        double dudy=mc.ATA_inv[ci*4+2]*ru0+mc.ATA_inv[ci*4+3]*ru1;
        double dvdx=mc.ATA_inv[ci*4+0]*rv0+mc.ATA_inv[ci*4+1]*rv1;
        double om=dvdx-dudy; d.ens += 0.5*om*om*m.cell_volumes[ci]; }
    // diagonal-symmetry RMS of density (mesh-induced baseline affects both schemes equally)
    double se=0, sr=0; long cnt=0;
    for (int idx=0; idx<NC; ++idx){ int q=idx/2, s=idx%2, i=q%N, j=q/N;
        int ridx = 2*(i*N + j) + (1-s);
        if (ridx>=0 && ridx<NC){ double dd=rho[idx]-rho[ridx]; se+=dd*dd; sr+=rho[idx]*rho[idx]; ++cnt; } }
    d.sym = cnt>0 ? std::sqrt(se/std::max(sr,1e-30)) : 0.0;
    if (dump){ FILE* fh=std::fopen(dump,"w");
        for(int i=0;i<NC;++i) std::fprintf(fh,"%.6g %.6g %.6g\n",
            m.cell_centers[i*2],m.cell_centers[i*2+1],rho[i]); std::fclose(fh);
        cfd::write_vtk2d_euler(std::string(dump)+".vtk", m, eq, r.U); }
    return d;
}

int main(int argc, char** argv) {
    int N = argc>1 ? std::atoi(argv[1]) : 200;
    double t_end = argc>2 ? std::atof(argv[2]) : 0.8;   // standard Lax-Liu/Schulz-Rinne config3: [0,1]^2, interface 0.8, t=0.8
    FluxKind flux = FLUX_HLLC;
    if (const char* f = std::getenv("C3_FLUX")) { std::string s=f; if(s=="roe"||s=="rroe") flux=FLUX_RROE; else if(s=="rhllc") flux=FLUX_RHLLC; else if(s=="hll") flux=FLUX_HLL; else if(s=="llf") flux=FLUX_LLF; else if(s=="slau2") flux=FLUX_SLAU2; else if(s=="rslau2") flux=FLUX_RSLAU2; else if(s=="roeef"||s=="roe_ef") flux=FLUX_ROE_EF; else if(s=="hllcm") flux=FLUX_HLLCM; else if(s=="rhllcm") flux=FLUX_RHLLCM; }
    Mesh m = triangulate_box(N, N, 1.0, 1.0);   // domain [0,1]x[0,1] (standard config3, interface 0.8)
    Euler2D eq{GAMMA};
    const int NC = m.n_cells();
    std::vector<double> U0(4*NC);
    for (int i=0;i<NC;++i){ double W[4]; config3_W(m.cell_centers[i*2],m.cell_centers[i*2+1],W);
        double U[4]; eq.prim_to_cons(W,U); for(int v=0;v<4;++v) U0[(size_t)v*NC+i]=U[v]; }
    static std::vector<BC2D> c3bcs;
    if (const char* bc = std::getenv("C3_BC")) { std::string bcm = bc;
      if (bcm == "dirichlet" || bcm == "moving") {
        c3bcs.assign(5, BC2D{});
        for (int tg = 1; tg <= 4; ++tg) {
            c3bcs[tg].kind = 3;
            if (bcm == "moving") c3bcs[tg].func = [](double x, double y, double t, double* W){ config3_W_moving(x, y, t, W); };
            else                 c3bcs[tg].func = [](double x, double y, double, double* W){ config3_W(x, y, W); };
        }
        g_c3_bcs = &c3bcs;
        std::printf("  [C3_BC=%s active]\n", bc);
      }
    }
    ReconCtx bj = build_recon_ctx(m, 0.0);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    std::printf("Config3 four-shock: N=%d cells=%d t_end=%.3f flux=%d\n", N, NC, t_end, (int)flux);
    bool skip_mlp = std::getenv("C3_SKIP_MLP")!=nullptr, skip2 = std::getenv("C3_SKIP_2ND")!=nullptr;
    auto t_all_ = std::chrono::steady_clock::now();
    if(!skip_mlp){ Diag a = run(m,eq,U0,N,t_end,RECON_BJ_VERTEX,&bj,nullptr,bj,flux,"config3_mlpu1.txt","mlp_u1");
        std::printf("  mlp_u1   : enstrophy=%.5f sym_rms=%.2e rho_min=%.4f p_min=%.4f steps=%d\n",
                    a.ens,a.sym,a.rho_min,a.p_min,a.n_steps); }
    int c3recon = RECON_BVD;   // C3_RECON=first -> 1st-order upwind (no reconstruction)
    if(const char* rc=std::getenv("C3_RECON")){ std::string s=rc; if(s=="first") c3recon=RECON_FIRST; }
    if(!skip2){ Diag b = run(m,eq,U0,N,t_end,c3recon,&bj,&c2,bj,flux,"config3_tmlpu.txt","T-MLP-u");
        std::printf("  T-MLP-u  : enstrophy=%.5f sym_rms=%.2e rho_min=%.4f p_min=%.4f steps=%d\n",
                    b.ens,b.sym,b.rho_min,b.p_min,b.n_steps); }
    std::printf("[WALL] TOTAL wall=%.3fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all_).count());
    return 0;
}
