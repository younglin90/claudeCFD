// gresho_bench.cpp — Gresho rotating vortex (low-Mach, steady). ρ=1, azimuthal velocity
// is a piecewise-linear "tent": u_φ=5r (r<0.2), 2−5r (0.2≤r<0.4), 0 (r≥0.4). Pressure
// balances the centrifugal force -> the axisymmetric field is TIME-INDEPENDENT, so
// exact(t)=initial for all t. On [0,1]² centered (0.5,0.5) the whole boundary has
// r≥0.5>0.4 (u=0, quiescent) -> periodic ≡ transmissive (no boundary flux). Triangle mesh.
// Tests: low-Mach vortex preservation (Roe/HLLC dissipation ∝1/M spins it down) and the
// C0 velocity kinks at r=0.2,0.4 (limiter must preserve, not clip into spurious vortices).
// Compares pure mlp_u1 (RECON_BJ_VERTEX) vs T-MLP-u (RECON_BVD = reconstruct_tmlpu_gated).
// Args: [N] [t_end].  Env: G_PINF (background pressure -> Mach), G_FLUX (hllc|roe|rroe).
#include "cfd/solver_euler2d.hpp"
#include "cfd/euler2d.hpp"
#include "cfd/diagnostics.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

using namespace cfd;
static const double GAMMA = 1.4, XC = 0.5, YC = 0.5, L = 1.0;
static double PINF = 71.7;   // background pressure (r≥0.4). M_max≈0.1 default. Override G_PINF.

// Gresho primitive state at (x,y): ρ=1, tent u_φ(r), centrifugal-balanced p(r).
static void gresho_W(double x, double y, double W[4]) {
    double dx = x - XC, dy = y - YC, r = std::sqrt(dx*dx + dy*dy);
    double uphi, p;
    if (r < 0.2) {
        uphi = 5.0 * r;
        p = PINF - 0.7726 + 12.5*r*r;                       // p(0)=PINF-0.7726
    } else if (r < 0.4) {
        uphi = 2.0 - 5.0*r;
        p = PINF + 4.0*std::log(r/0.4) - 20.0*r + 12.5*r*r + 6.0;
    } else {
        uphi = 0.0;
        p = PINF;
    }
    double inv = (r > 1e-12) ? 1.0/r : 0.0;
    double u = -uphi * dy * inv, v = uphi * dx * inv;       // tangential (−sinθ,cosθ)
    W[0] = 1.0; W[1] = u; W[2] = v; W[3] = p;               // ρ=1
}

struct Diag { double l2_vel, speed_peak, ke, n_steps; };

static Diag run(const Mesh& m, const Euler2D& eq, const std::vector<double>& U0,
                const std::vector<double>& u0, const std::vector<double>& v0,
                double t_end, int recon, const ReconCtx* ctx, const ReconCtxO2* ctx_o2,
                FluxKind flux, const char* dump = nullptr) {
    static const double gcfl = std::getenv("G_CFL") ? std::atof(std::getenv("G_CFL")) : 0.4;
    Solve2DResult r = solve_euler2d(m, eq, U0, t_end, 2, gcfl, -1.0, 100000000,
                                    recon, ctx, flux, nullptr, nullptr, ctx_o2);
    const int N = m.n_cells();
    Diag d{0,0,0,(double)r.n_steps};
    double l2=0, tv=0;
    for (int i=0;i<N;++i){ double U[4]={r.U[0*N+i],r.U[1*N+i],r.U[2*N+i],r.U[3*N+i]},W[4];
        eq.cons_to_prim(U,W);
        double sp=std::sqrt(W[1]*W[1]+W[2]*W[2]);
        double eu=W[1]-u0[i], ev=W[2]-v0[i];
        l2 += (eu*eu+ev*ev)*m.cell_volumes[i]; tv += m.cell_volumes[i];
        d.speed_peak=std::max(d.speed_peak,sp);
        d.ke += 0.5*W[0]*sp*sp*m.cell_volumes[i]; }
    d.l2_vel = std::sqrt(l2/tv);
    if (dump){ FILE* fh=std::fopen(dump,"w");
        for(int i=0;i<N;++i){ double U[4]={r.U[0*N+i],r.U[1*N+i],r.U[2*N+i],r.U[3*N+i]},W[4];
            eq.cons_to_prim(U,W); double sp=std::sqrt(W[1]*W[1]+W[2]*W[2]);
            std::fprintf(fh,"%.6g %.6g %.6g\n",m.cell_centers[i*2],m.cell_centers[i*2+1],sp); }
        std::fclose(fh); }
    return d;
}

int main(int argc, char** argv) {
    int N = argc>1 ? std::atoi(argv[1]) : 96;
    double t_end = argc>2 ? std::atof(argv[2]) : 1.2566370614;   // 1 core turnover T=2π/5
    if (std::getenv("G_PINF")) PINF = std::atof(std::getenv("G_PINF"));
    FluxKind flux = FLUX_HLLC;
    if (const char* gf = std::getenv("G_FLUX")) { std::string s=gf;
        if(s=="roe"||s=="rroe") flux=FLUX_RROE; else if(s=="hllc") flux=FLUX_HLLC; else if(s=="rhllc") flux=FLUX_RHLLC; else if(s=="hll") flux=FLUX_HLL; else if(s=="llf") flux=FLUX_LLF; }
    Mesh m = triangulate_box(N, N, L, L);
    Euler2D eq{GAMMA};
    const int NC = m.n_cells();
    std::vector<double> U0(4*NC), u0(NC), v0(NC);
    double speed_ic=0, ke_ic=0, tv=0;
    for (int i=0;i<NC;++i){ double W[4]; gresho_W(m.cell_centers[i*2],m.cell_centers[i*2+1],W);
        u0[i]=W[1]; v0[i]=W[2]; double sp=std::sqrt(W[1]*W[1]+W[2]*W[2]);
        speed_ic=std::max(speed_ic,sp); ke_ic+=0.5*W[0]*sp*sp*m.cell_volumes[i]; tv+=m.cell_volumes[i];
        double U[4]; eq.prim_to_cons(W,U); for(int v=0;v<4;++v) U0[(size_t)v*NC+i]=U[v]; }
    double c_at_peak = std::sqrt(GAMMA*(PINF-0.2726)/1.0), Mmax = 1.0/c_at_peak;
    ReconCtx bj = build_recon_ctx(m, 0.0);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    std::printf("Gresho: N=%d cells=%d t_end=%.4f PINF=%.2f M_max=%.3f flux=%d | IC speed_peak=%.4f KE=%.5f\n",
                N, NC, t_end, PINF, Mmax, (int)flux, speed_ic, ke_ic);
    bool skip_mlp = std::getenv("G_SKIP_MLP")!=nullptr, skip_2nd = std::getenv("G_SKIP_2ND")!=nullptr;
    Diag a{}, bv{};
    if(!skip_mlp) a  = run(m, eq, U0, u0, v0, t_end, RECON_BJ_VERTEX, &bj, nullptr, flux, "gresho_mlpu1.txt");
    if(!skip_2nd) bv = run(m, eq, U0, u0, v0, t_end, RECON_BVD, &bj, &c2, flux, "gresho_tmlpu.txt");
    std::printf("  mlp_u1   : L2(vel)=%.4e speed_peak=%.4f (%.1f%% of IC) KE=%.5f (%.1f%%) steps=%d\n",
                a.l2_vel, a.speed_peak, 100*a.speed_peak/speed_ic, a.ke, 100*a.ke/ke_ic, (int)a.n_steps);
    std::printf("  T-MLP-u  : L2(vel)=%.4e speed_peak=%.4f (%.1f%% of IC) KE=%.5f (%.1f%%) steps=%d\n",
                bv.l2_vel, bv.speed_peak, 100*bv.speed_peak/speed_ic, bv.ke, 100*bv.ke/ke_ic, (int)bv.n_steps);
    std::printf("  => T-MLP-u vs mlp_u1: L2 %.3fx (lower=better), KE %.1f%% vs %.1f%% (higher=less diffusion)\n",
                bv.l2_vel/std::max(a.l2_vel,1e-30), 100*bv.ke/ke_ic, 100*a.ke/ke_ic);
    return 0;
}
