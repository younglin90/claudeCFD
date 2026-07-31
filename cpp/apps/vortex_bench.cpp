// vortex_bench.cpp — stationary isentropic vortex (Shu). A smooth, exact steady
// solution: the vortex sits still, so exact(t)=initial. Numerical diffusion decays
// the core; a low-diffusion / high-order reconstruction preserves peak vorticity &
// minimum density. Cheap (smooth, large dt) -> fast dev loop to tune the limiter.
// Compares pure mlp_u1 (BJ-vertex) vs T-MLP-u+BVD (and env-gated enhancements).
// Args: [N] [t_end]. Env: BVD_MULTIVAR, SHEAR_RELAX (passed through to BVD).
#include "cfd/solver_euler2d.hpp"
#include "cfd/mood2d.hpp"
#include "cfd/diagnostics.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

using namespace cfd;
static const double GAMMA = 1.4, BETA = 5.0, XC = 5.0, YC = 5.0, L = 10.0;

// isentropic vortex primitive state at (x,y), mean flow zero -> stationary.
static void vortex_W(double x, double y, double W[4]) {
    double dx = x - XC, dy = y - YC, r2 = dx*dx + dy*dy;
    double ex = std::exp((1.0 - r2) * 0.5);
    double du = -(BETA / (2.0*M_PI)) * dy * ex;
    double dv =  (BETA / (2.0*M_PI)) * dx * ex;
    double dT = -(GAMMA - 1.0) * BETA*BETA / (8.0*GAMMA*M_PI*M_PI) * std::exp(1.0 - r2);
    double T = 1.0 + dT, rho = std::pow(T, 1.0/(GAMMA-1.0));
    W[0] = rho; W[1] = du; W[2] = dv; W[3] = rho * T;   // p = rho*T
}

struct Diag { double l2_rho, rho_min, omega_peak; int n_steps; };

static Diag run(const Mesh& m, const Euler2D& eq, const std::vector<double>& U0,
                const std::vector<double>& rho0, double t_end, int recon,
                const ReconCtx* ctx, const ReconCtxO2* ctx_o2, const ReconCtx& mc,
                const char* dump = nullptr) {
    static const int vflux = []{ const char* f=std::getenv("VTX_FLUX"); if(!f)return (int)FLUX_HLLC; std::string s=f; if(s=="hll")return (int)FLUX_HLL; if(s=="rhllc")return (int)FLUX_RHLLC; if(s=="llf")return (int)FLUX_LLF; return (int)FLUX_HLLC; }();
    Solve2DResult r = (std::getenv("MOOD") && ctx_o2)
        ? solve_euler2d_mood(m, eq, U0, t_end, 0.4, 100000000, vflux, nullptr, *ctx_o2)
        : solve_euler2d(m, eq, U0, t_end, 2, 0.4, -1.0, 100000000,
                        recon, ctx, vflux, nullptr, nullptr, ctx_o2);
    const int N = m.n_cells();
    Diag d{0,1e9,0,r.n_steps};
    std::vector<double> rho(N), u(N), v(N);
    double l2=0, tv=0;
    for (int i=0;i<N;++i){ double U[4]={r.U[0*N+i],r.U[1*N+i],r.U[2*N+i],r.U[3*N+i]},W[4];
        eq.cons_to_prim(U,W); rho[i]=W[0]; u[i]=W[1]; v[i]=W[2];
        double e=(W[0]-rho0[i]); l2 += e*e*m.cell_volumes[i]; tv += m.cell_volumes[i];
        d.rho_min=std::min(d.rho_min,W[0]); }
    d.l2_rho = std::sqrt(l2/tv);
    // peak |vorticity| via LSQ gradient (reuse metric ctx)
    for (int ci=0;ci<N;++ci){ double ru0=0,ru1=0,rv0=0,rv1=0;
        for(int k=0;k<mc.max_nb;++k){ int nb=mc.nb[(size_t)ci*mc.max_nb+k]; if(nb<0)continue;
            double wk=mc.w[(size_t)ci*mc.max_nb+k];
            double dx=mc.d[((size_t)ci*mc.max_nb+k)*2+0], dy=mc.d[((size_t)ci*mc.max_nb+k)*2+1];
            ru0+=wk*dx*(u[nb]-u[ci]); ru1+=wk*dy*(u[nb]-u[ci]);
            rv0+=wk*dx*(v[nb]-v[ci]); rv1+=wk*dy*(v[nb]-v[ci]); }
        double dudy=mc.ATA_inv[ci*4+2]*ru0+mc.ATA_inv[ci*4+3]*ru1;
        double dvdx=mc.ATA_inv[ci*4+0]*rv0+mc.ATA_inv[ci*4+1]*rv1;
        d.omega_peak=std::max(d.omega_peak,std::fabs(dvdx-dudy)); }
    if (dump){ FILE* fh=std::fopen(dump,"w");
        for(int i=0;i<N;++i) std::fprintf(fh,"%.6g %.6g %.6g\n",
            m.cell_centers[i*2],m.cell_centers[i*2+1],rho[i]); std::fclose(fh); }
    return d;
}

int main(int argc, char** argv) {
    int N = argc>1 ? std::atoi(argv[1]) : 80;
    double t_end = argc>2 ? std::atof(argv[2]) : 5.0;
    Mesh m = triangulate_box(N, N, L, L);
    Euler2D eq{GAMMA};
    const int NC = m.n_cells();
    std::vector<double> U0(4*NC), rho0(NC);
    for (int i=0;i<NC;++i){ double W[4]; vortex_W(m.cell_centers[i*2],m.cell_centers[i*2+1],W);
        rho0[i]=W[0]; double U[4]; eq.prim_to_cons(W,U);
        for(int v=0;v<4;++v) U0[(size_t)v*NC+i]=U[v]; }
    ReconCtx bj = build_recon_ctx(m, 0.0);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    // exact peak vorticity (from IC) for reference
    Diag ic = run(m, eq, U0, rho0, 0.0, RECON_FIRST, &bj, nullptr, bj);
    std::printf("Isentropic vortex: N=%d cells=%d t_end=%.2f | IC: rho_min=%.4f omega_peak=%.4f\n",
                N, NC, t_end, ic.rho_min, ic.omega_peak);
    const char* vr = std::getenv("VTX_RECON");        // "o2" -> compare RECON_O2 instead of BVD
    int recon2 = (vr && std::string(vr)=="o2") ? RECON_O2 : RECON_BVD;
    bool skip_mlp = std::getenv("VTX_SKIP_MLP")!=nullptr;   // time 2nd scheme alone
    bool skip_2nd = std::getenv("VTX_SKIP_2ND")!=nullptr;   // time mlp_u1 alone
    Diag a{}, bv{};
    if(!skip_mlp) a  = run(m, eq, U0, rho0, t_end, RECON_BJ_VERTEX, &bj, nullptr, bj, "vtx_mlpu1.txt");
    if(!skip_2nd) bv = run(m, eq, U0, rho0, t_end, recon2, &bj, &c2, bj, "vtx_bvd.txt");
    std::printf("  mlp_u1       : L2(rho)=%.4e rho_min=%.4f omega_peak=%.4f (%.1f%% of IC) steps=%d\n",
                a.l2_rho, a.rho_min, a.omega_peak, 100*a.omega_peak/ic.omega_peak, a.n_steps);
    std::printf("  T-MLP-u+BVD  : L2(rho)=%.4e rho_min=%.4f omega_peak=%.4f (%.1f%% of IC) steps=%d\n",
                bv.l2_rho, bv.rho_min, bv.omega_peak, 100*bv.omega_peak/ic.omega_peak, bv.n_steps);
    std::printf("  => BVD vs mlp_u1: L2 %.3fx (lower=better), omega %.1f%% vs %.1f%% (higher=less diffusion)\n",
                bv.l2_rho/std::max(a.l2_rho,1e-30),
                100*bv.omega_peak/ic.omega_peak, 100*a.omega_peak/ic.omega_peak);
    return 0;
}
