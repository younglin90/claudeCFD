// kh_linear_bench.cpp — LINEAR Kelvin-Helmholtz growth rate vs Michalke (1964) theory.
// Single hyperbolic-tangent shear layer u=U·tanh((y−y0)/δ), UNIFORM ρ and p (single-fluid,
// low convective Mach -> incompressible-like). Seed ONE wavelength (k=2π/Lx) of v at tiny
// amplitude. The most-unstable mode is selected by setting δ so that kδ=0.4446 (Michalke).
// Measure the growth of perturbation energy E_v=∫½ρv² in the LINEAR window -> σ_meas, and
// compare to σ_theory=0.1897·U/δ. The perturbation sin(2πx/Lx) has nodes at x=0,Lx, so a
// transmissive x-boundary is benign (≈periodic). A scheme that reproduces σ neither manufactures
// (σ>theory => anti-diffusion) nor kills (σ<<theory => over-diffusion) the instability ->
// QUANTITATIVE legitimacy test. Compares mlp_u1 (RECON_BJ_VERTEX) vs T-MLP-u (RECON_BVD).
// Args: [N] [t_end].  Env: KHL_P (pressure->Mach), KHL_W0 (seed amp), KHL_FLUX (hllc|roe).
#include "cfd/solver_euler2d.hpp"
#include "cfd/diagnostics.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

using namespace cfd;
static const double GAMMA = 1.4, Lx = 1.0, Ly = 1.0, U = 1.0;

static double Ev(const Euler2D& eq, const Mesh& m, const std::vector<double>& Uc) {
    int N = m.n_cells(); double e = 0;
    for (int i = 0; i < N; ++i) { double Ui[4]={Uc[0*N+i],Uc[1*N+i],Uc[2*N+i],Uc[3*N+i]}, W[4];
        eq.cons_to_prim(Ui, W); e += 0.5*W[0]*W[2]*W[2]*m.cell_volumes[i]; }
    return e;
}

// Fourier mode projection of v onto the seeded mode sin(2πx/Lx): sign-meaningful amplitude,
// isolates the physical KH mode from grid noise. A(t) grows as exp(σt) -> σ = d(ln|A|)/dt.
static double Amode(const Euler2D& eq, const Mesh& m, const std::vector<double>& Uc, double Lx) {
    int N = m.n_cells(); double a = 0;
    for (int i = 0; i < N; ++i) { double Ui[4]={Uc[0*N+i],Uc[1*N+i],Uc[2*N+i],Uc[3*N+i]}, W[4];
        eq.cons_to_prim(Ui, W);
        a += W[2]*std::sin(2.0*M_PI*m.cell_centers[i*2]/Lx)*m.cell_volumes[i]; }
    return a;
}

// chain short solves, sample E_v(t), fit log(E_v) vs t in linear window -> σ = slope/2.
static double run_sigma(const Mesh& m, const Euler2D& eq, const std::vector<double>& U0,
                        double t_end, int recon, const ReconCtx* ctx, const ReconCtxO2* ctx2,
                        FluxKind flux, const char* dump=nullptr) {
    const int nsamp = 41;   // dense sampling averages acoustic ripple in the ln(E_v) fit
    std::vector<double> ts(nsamp), le(nsamp), Uc = U0;
    double prev = 0;
    for (int s = 0; s < nsamp; ++s) {
        double t = t_end * s / (nsamp - 1);
        if (s > 0) { double dt = t - prev;
            Solve2DResult r = solve_euler2d(m, eq, Uc, dt, 2, 0.4, -1.0, 100000000,
                                            recon, ctx, flux, nullptr, nullptr, ctx2);
            Uc = r.U; }
        ts[s] = t; le[s] = std::log(Ev(eq, m, Uc) + 1e-300); prev = t;  // E_v=∫½ρv² ~ exp(2σt), positive
    }
    if (dump){ int NC=m.n_cells(); FILE* fh=std::fopen(dump,"w");
        for(int i=0;i<NC;++i){ double Ui[4]={Uc[0*NC+i],Uc[1*NC+i],Uc[2*NC+i],Uc[3*NC+i]},W[4];
            eq.cons_to_prim(Ui,W); std::fprintf(fh,"%.6g %.6g %.6g\n",m.cell_centers[i*2],m.cell_centers[i*2+1],W[2]); }
        std::fclose(fh); }
    double t0 = t_end*0.35, t1 = t_end*0.85, sx=0,sy=0,sxx=0,sxy=0; int n=0;
    for (int s = 0; s < nsamp; ++s) if (ts[s]>=t0 && ts[s]<=t1) {
        sx+=ts[s]; sy+=le[s]; sxx+=ts[s]*ts[s]; sxy+=ts[s]*le[s]; ++n; }
    double slope = (n*sxy - sx*sy) / (n*sxx - sx*sx + 1e-300);
    return slope * 0.5;   // E_v ~ exp(2σt)  ->  σ = slope/2
}

int main(int argc, char** argv) {
    int N = argc>1 ? std::atoi(argv[1]) : 96;
    double t_end = argc>2 ? std::atof(argv[2]) : 1.4;
    double P0 = std::getenv("KHL_P")  ? std::atof(std::getenv("KHL_P"))  : 100.0;
    double W0 = std::getenv("KHL_W0") ? std::atof(std::getenv("KHL_W0")) : 1e-3;
    FluxKind flux = FLUX_HLLC;
    if (const char* f = std::getenv("KHL_FLUX")) { std::string s=f; if(s=="roe"||s=="rroe") flux=FLUX_RROE; }
    double delta = 0.4446 * Lx / (2.0*M_PI);     // kδ=0.4446, k=2π/Lx
    double sigma_th = 0.1897 * U / delta;        // Michalke max temporal growth rate
    double c = std::sqrt(GAMMA*P0/1.0), Mc = U/c;
    Mesh m = triangulate_box(N, N, Lx, Ly);
    Euler2D eq{GAMMA};
    const int NC = m.n_cells();
    std::vector<double> U0(4*NC);
    // DIVERGENCE-FREE perturbation via streamfunction ψ'=W0·cos(2πx/Lx)·env(y):
    // u'=∂ψ'/∂y, v'=−∂ψ'/∂x  ->  ∇·u'=0  ->  does NOT excite acoustic waves -> clean σ.
    const double sg = 4.0*delta;
    for (int i = 0; i < NC; ++i) { double x=m.cell_centers[i*2], y=m.cell_centers[i*2+1];
        double env = std::exp(-((y-0.5)*(y-0.5))/(2.0*sg*sg));
        double envp = -((y-0.5)/(sg*sg))*env;                       // d(env)/dy
        double up = W0*std::cos(2.0*M_PI*x/Lx)*envp;               // ∂ψ'/∂y
        double vp = W0*(2.0*M_PI/Lx)*std::sin(2.0*M_PI*x/Lx)*env;  // −∂ψ'/∂x
        double u = U*std::tanh((y-0.5)/delta) + up, v = vp;
        double W[4] = {1.0, u, v, P0}, Uc[4]; eq.prim_to_cons(W, Uc);
        for (int vv=0; vv<4; ++vv) U0[(size_t)vv*NC+i] = Uc[vv]; }
    std::printf("KH-linear: N=%d cells=%d delta=%.4f Mc=%.3f flux=%d | sigma_theory=%.4f (Michalke 0.1897·U/δ)\n",
                N, NC, delta, Mc, (int)flux, sigma_th);
    bool skip_mlp = std::getenv("KHL_SKIP_MLP")!=nullptr, skip2 = std::getenv("KHL_SKIP_2ND")!=nullptr;
    ReconCtx bj = build_recon_ctx(m, 0.0);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    const char* dmp = std::getenv("KHL_DUMP");
    if (!skip_mlp) { double s = run_sigma(m, eq, U0, t_end, RECON_BJ_VERTEX, &bj, nullptr, flux, dmp?"khl_mlpu1.txt":nullptr);
        std::printf("  mlp_u1   : sigma=%.4f  (%.1f%% of theory)\n", s, 100*s/sigma_th); }
    if (!skip2)   { double s = run_sigma(m, eq, U0, t_end, RECON_BVD, &bj, &c2, flux, dmp?"khl_tmlpu.txt":nullptr);
        std::printf("  T-MLP-u  : sigma=%.4f  (%.1f%% of theory)\n", s, 100*s/sigma_th); }
    return 0;
}
