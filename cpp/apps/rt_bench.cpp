// rt_bench.cpp — Rayleigh-Taylor instability (Shi-Zhang-Shu). Heavy fluid (ρ=2) on TOP of
// light (ρ=1) in downward gravity g=1 -> unstable: spikes fall, bubbles rise, mushroom +
// secondary KH roll-up. The RICHNESS of vortical structure measures numerical dissipation
// (less dissipation -> more fine structure). Single-mode cos(8πx) perturbation -> the exact
// solution is LEFT-RIGHT symmetric about x=1/8; loss of symmetry is the signature of REDUCED
// dissipation (TENO paper §5.1). Domain [0,1/4]×[0,1], γ=5/3, hydrostatic-balanced p,
// reflective walls, t_end=1.95. Triangle (unstructured) mesh. Compares mlp_u1 vs T-MLP-u.
// Args: [Nx] [Ny] [t_end].  Env: RT_FLUX (hllc|roe).
#include "cfd/solver_euler2d.hpp"
#include "cfd/diagnostics.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

using namespace cfd;
static const double GAMMA = 5.0/3.0, LX = 0.25, LY = 1.0, GRAV = -1.0;  // gravity points -y

// hydrostatic RT state: ρ=2 (y>1/2) over ρ=1 (y<1/2); dp/dy=ρ·g_y -> p decreases upward.
static void rt_W(double x, double y, double W[4]) {
    double rho = (y > 0.5) ? 2.0 : 1.0;
    double p   = (y > 0.5) ? (3.0 - 2.0*y) : (2.5 - y);   // continuous at y=0.5 (p=2), >0 everywhere
    double c   = std::sqrt(GAMMA*p/rho);
    double v   = -0.025 * c * std::cos(8.0*M_PI*x);       // single-mode seed (one wavelength in [0,1/4])
    W[0]=rho; W[1]=0.0; W[2]=v; W[3]=p;
}

struct Diag { double ens, sym, rho_min, rho_max; int n_steps; };

// x-mirror symmetry about x=1/4·½ : rising-diagonal triangulate_box cell (i,j,s) mirrors to
// (Nx-1-i, j, 1-s). ρ should equal its mirror (single-mode IC is symmetric about x=1/8).
static Diag run(const Mesh& m, const Euler2D& eq, const std::vector<double>& U0, int Nx,
                double t_end, int recon, const ReconCtx* ctx, const ReconCtxO2* ctx2,
                const ReconCtx& mc, FluxKind flux, const std::vector<BC2D>* bcs, const char* dump=nullptr) {
    Solve2DResult r = solve_euler2d(m, eq, U0, t_end, 3, 0.4, -1.0, 100000000,
                                    recon, ctx, flux, bcs, nullptr, ctx2, GRAV);
    const int NC = m.n_cells();
    Diag d{0,0,1e9,-1e9,r.n_steps};
    std::vector<double> rho(NC), u(NC), v(NC);
    for (int i=0;i<NC;++i){ double U[4]={r.U[0*NC+i],r.U[1*NC+i],r.U[2*NC+i],r.U[3*NC+i]},W[4];
        eq.cons_to_prim(U,W); rho[i]=W[0]; u[i]=W[1]; v[i]=W[2];
        d.rho_min=std::min(d.rho_min,W[0]); d.rho_max=std::max(d.rho_max,W[0]); }
    for (int ci=0;ci<NC;++ci){ double ru0=0,ru1=0,rv0=0,rv1=0;
        for(int k=0;k<mc.max_nb;++k){ int nb=mc.nb[(size_t)ci*mc.max_nb+k]; if(nb<0)continue;
            double wk=mc.w[(size_t)ci*mc.max_nb+k];
            double dx=mc.d[((size_t)ci*mc.max_nb+k)*2+0], dy=mc.d[((size_t)ci*mc.max_nb+k)*2+1];
            ru0+=wk*dx*(u[nb]-u[ci]); ru1+=wk*dy*(u[nb]-u[ci]);
            rv0+=wk*dx*(v[nb]-v[ci]); rv1+=wk*dy*(v[nb]-v[ci]); }
        double dudy=mc.ATA_inv[ci*4+2]*ru0+mc.ATA_inv[ci*4+3]*ru1;
        double dvdx=mc.ATA_inv[ci*4+0]*rv0+mc.ATA_inv[ci*4+1]*rv1;
        double om=dvdx-dudy; d.ens += 0.5*om*om*m.cell_volumes[ci]; }
    double se=0,sr=0; long cnt=0;
    for (int idx=0; idx<NC; ++idx){ int q=idx/2, s=idx%2, i=q%Nx, j=q/Nx;
        int mi=2*(j*Nx+(Nx-1-i))+(1-s);
        if (mi>=0&&mi<NC){ double dd=rho[idx]-rho[mi]; se+=dd*dd; sr+=rho[idx]*rho[idx]; ++cnt; } }
    d.sym = cnt>0 ? std::sqrt(se/std::max(sr,1e-30)) : 0.0;
    if (dump){ FILE* fh=std::fopen(dump,"w");
        for(int i=0;i<NC;++i) std::fprintf(fh,"%.6g %.6g %.6g\n",m.cell_centers[i*2],m.cell_centers[i*2+1],rho[i]);
        std::fclose(fh); }
    return d;
}

int main(int argc, char** argv) {
    int Nx = argc>1 ? std::atoi(argv[1]) : 32;
    int Ny = argc>2 ? std::atoi(argv[2]) : 128;
    double t_end = argc>3 ? std::atof(argv[3]) : 1.95;
    FluxKind flux = FLUX_HLLC;
    if (const char* f = std::getenv("RT_FLUX")) { std::string s=f; if(s=="roe"||s=="rroe") flux=FLUX_RROE; }
    Mesh m = triangulate_box(Nx, Ny, LX, LY);
    Euler2D eq{GAMMA};
    const int NC = m.n_cells();
    std::vector<double> U0(4*NC);
    for (int i=0;i<NC;++i){ double W[4]; rt_W(m.cell_centers[i*2],m.cell_centers[i*2+1],W);
        double U[4]; eq.prim_to_cons(W,U); for(int v=0;v<4;++v) U0[(size_t)v*NC+i]=U[v]; }
    std::vector<BC2D> bcs(5);
    for (int t=1;t<=4;++t) bcs[t].kind = 1;   // reflective slip walls on all 4 sides (tags 1-4)
    ReconCtx bj = build_recon_ctx(m, 0.0);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    std::printf("Rayleigh-Taylor: grid=%dx%d cells=%d t_end=%.3f gamma=%.3f g=%.1f flux=%d\n",
                Nx, Ny, NC, t_end, GAMMA, GRAV, (int)flux);
    bool skip_mlp = std::getenv("RT_SKIP_MLP")!=nullptr, skip2 = std::getenv("RT_SKIP_2ND")!=nullptr;
    if(!skip_mlp){ Diag a = run(m,eq,U0,Nx,t_end,RECON_BJ_VERTEX,&bj,nullptr,bj,flux,&bcs,"rt_mlpu1.txt");
        std::printf("  mlp_u1   : enstrophy=%.4f sym_rms=%.2e rho[%.4f,%.4f] steps=%d\n",a.ens,a.sym,a.rho_min,a.rho_max,a.n_steps); }
    if(!skip2){ Diag b = run(m,eq,U0,Nx,t_end,RECON_BVD,&bj,&c2,bj,flux,&bcs,"rt_tmlpu.txt");
        std::printf("  T-MLP-u  : enstrophy=%.4f sym_rms=%.2e rho[%.4f,%.4f] steps=%d\n",b.ens,b.sym,b.rho_min,b.rho_max,b.n_steps); }
    return 0;
}
