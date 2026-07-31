// shock_mixing_bench.cpp — Oblique Shock / Mixing-Layer Interaction
// (Kim & Kim, "Accurate, efficient and monotonic numerical methods for multi-
//  dimensional compressible flows, Part II / MLP", and the MLP/CPR validation
//  figure in AIAA 2015-3199). A spatially-developing compressible mixing layer:
//  an oblique shock launched from the upper-left corner impinges on the shear
//  layer, deflects, reflects off the bottom slip wall, then interacts with the
//  KH vortices downstream. Pressure contours at t=120.
//
//  IC (mean):  rho=1, u(y)=2.5+0.5*tanh(2y), v=0, p≈0.496, gamma=1.4.
//    Convective Mach M_c=(u1-u2)/(c1+c2)=0.6 with u1=3,u2=2 (Δu=1) =>
//    c1+c2=1.667; isothermal c1=c2=0.8333 => p=rho*c^2/gamma=0.8333^2/1.4≈0.496.
//
//  BCs:
//    left  (x=0):    time-dependent Dirichlet = mean state with v'(y,t) forcing
//                    v' = Σ_{k=1,2} a_k cos(2πk t/T + φ_k) exp(-y^2/b),
//                    T=λ/u_c, λ=30, u_c=2.68 (T≈11.19), a1=a2=0.05,
//                    φ1=0, φ2=π/2, b=10. (u,rho,p held at mean; only v perturbed.)
//    top   (y=20):   fixed Dirichlet = POST-SHOCK state of a weak oblique shock
//                    for the upper stream M1=3.6 deflected by θ=12°. Imposing the
//                    deflected post-shock state at the top generates the oblique
//                    shock that propagates down-right into the domain.
//    bottom(y=-20):  slip wall (reflective).
//    right (x=200):  outflow (transmissive).
//
//  NOTE: our solver is INVISCID (Euler). The paper's Re=500 physical viscosity
//  is OMITTED — only numerical dissipation acts. The inflow v' perturbation still
//  seeds the Kelvin-Helmholtz rolls; the shock dynamics are unaffected by Re.
//
//  Scheme: FLUX_RHLLC (rotated-HLLC) + recon=RECON_BVD. The deg3t-BVD variant is
//  selected at RUNTIME via env, e.g.:
//    BVD_CHENG3=1 MLP_U2=1 THINCQQ_SPL=1 THINCQQ_SIG_DEG3T=1 THINCQQ_SPL_CF=1 \
//      ./shock_mixing_bench
//
//  Domain [0,200]x[-20,20], uniform triangular mesh h≈0.75 (Nx≈267, Ny≈53).
//  Overridable: env SM_NX/SM_NY/SM_T or argv [Nx] [Ny] [t_end].
//  Pressure dump: env SM_DUMP (default shock_mixing.txt), "x y p" per cell.
#include "cfd/solver_euler2d.hpp"
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

static const double GAMMA = 1.4;

// ── Domain / mean-flow constants ───────────────────────────────────────────
static const double LX = 200.0, LY = 40.0, Y0 = -20.0;   // [0,200] x [-20,20]
static const double RHO_MEAN = 1.0;
static const double P_MEAN   = 0.496;                      // = rho*c^2/gamma, c≈0.8333
static double u_mean(double y) { return 2.5 + 0.5 * std::tanh(2.0 * y); }

// ── Inflow forcing (left boundary) ─────────────────────────────────────────
static const double FORCE_LAMBDA = 30.0, FORCE_UC = 2.68;
static const double FORCE_T = FORCE_LAMBDA / FORCE_UC;     // ≈ 11.194
static const double A1 = 0.05, A2 = 0.05;
static const double PHI1 = 0.0, PHI2 = M_PI / 2.0;
static const double FORCE_B = 10.0;
static double vprime(double y, double t) {
    double env = std::exp(-y * y / FORCE_B);
    double s = A1 * std::cos(2.0 * M_PI * 1.0 * t / FORCE_T + PHI1)
             + A2 * std::cos(2.0 * M_PI * 2.0 * t / FORCE_T + PHI2);
    return s * env;
}

// ── Weak oblique shock (θ-β-M) for the upper stream, deflection θ ───────────
// Returns post-shock primitive {rho2,u2,v2,p2} and (out) the weak shock angle.
// Incoming upper stream: (rho1,u1,0,p1) with u1>0; flow deflected DOWNWARD by θ
// (toward the shear layer) -> v2<0. Tangential velocity is conserved across the
// shock; normal-shock relations act on the normal Mach M1n=M1 sin(β_s).
static void oblique_shock_post(double rho1, double u1, double p1, double theta,
                               double out[4], double* beta_out) {
    const double g = GAMMA;
    const double c1 = std::sqrt(g * p1 / rho1);
    const double M1 = u1 / c1;
    auto theta_of_beta = [&](double b) {
        double s = std::sin(b);
        return std::atan(2.0 / std::tan(b) * (M1 * M1 * s * s - 1.0)
                         / (M1 * M1 * (g + std::cos(2.0 * b)) + 2.0));
    };
    const double mu = std::asin(1.0 / M1);                 // Mach angle (β lower bound)
    // locate β at maximum deflection (separates weak/strong roots)
    double beta_max = 0.5 * (mu + M_PI / 2.0), th_max = -1e9;
    const int NS = 20000;
    for (int i = 0; i <= NS; ++i) {
        double b = mu + (M_PI / 2.0 - mu) * (double)i / NS;
        double th = theta_of_beta(b);
        if (th > th_max) { th_max = th; beta_max = b; }
    }
    // bisection for the WEAK root: β ∈ [mu, beta_max], θ_of_β increasing there
    double lo = mu + 1e-12, hi = beta_max;
    for (int it = 0; it < 200; ++it) {
        double mid = 0.5 * (lo + hi);
        if (theta_of_beta(mid) < theta) lo = mid; else hi = mid;
    }
    double beta = 0.5 * (lo + hi);
    *beta_out = beta;

    double M1n = M1 * std::sin(beta);
    double p2_p1   = 1.0 + 2.0 * g / (g + 1.0) * (M1n * M1n - 1.0);
    double rho2_r1 = ((g + 1.0) * M1n * M1n) / ((g - 1.0) * M1n * M1n + 2.0);
    double M2n2 = (1.0 + 0.5 * (g - 1.0) * M1n * M1n) / (g * M1n * M1n - 0.5 * (g - 1.0));
    double M2n = std::sqrt(M2n2);
    double M2  = M2n / std::sin(beta - theta);

    double rho2 = rho1 * rho2_r1;
    double p2   = p1 * p2_p1;
    double c2   = std::sqrt(g * p2 / rho2);
    double V2   = M2 * c2;
    out[0] = rho2;
    out[1] = V2 * std::cos(theta);     // turned downward by θ
    out[2] = -V2 * std::sin(theta);
    out[3] = p2;
}

int main(int argc, char** argv) {
    auto envi = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
    auto envd = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };

    // h≈0.75: Nx=200/0.75≈267, Ny=40/0.75≈53. Override via argv then env.
    int Nx = argc > 1 ? std::atoi(argv[1]) : envi("SM_NX", 267);
    int Ny = argc > 2 ? std::atoi(argv[2]) : envi("SM_NY", 53);
    double t_end = argc > 3 ? std::atof(argv[3]) : envd("SM_T", 120.0);

    // ── post-shock top state for M1=3.6 (u1=3, p1=0.496, rho1=1), θ=12° ──
    double POST[4]; double beta_s;
    oblique_shock_post(RHO_MEAN, 3.0, P_MEAN, 12.0 * M_PI / 180.0, POST, &beta_s);
    std::printf("Oblique shock (upper stream M1=u1/c1=%.4f, deflection theta=12 deg):\n",
                3.0 / std::sqrt(GAMMA * P_MEAN / RHO_MEAN));
    std::printf("  weak shock angle beta_s = %.4f deg\n", beta_s * 180.0 / M_PI);
    std::printf("  post-shock (rho2,u2,v2,p2) = (%.5f, %.5f, %.5f, %.5f)\n",
                POST[0], POST[1], POST[2], POST[3]);

    // ── mesh ──
    Mesh m = triangulate_box(Nx, Ny, LX, LY, 0.0, Y0);
    Euler2D eq{GAMMA};
    const int NC = m.n_cells();

    // ── IC: mean shear layer everywhere ──
    std::vector<double> U0(4 * NC);
    for (int i = 0; i < NC; ++i) {
        double y = m.cell_centers[i * 2 + 1];
        double W[4] = {RHO_MEAN, u_mean(y), 0.0, P_MEAN};
        double U[4]; eq.prim_to_cons(W, U);
        for (int v = 0; v < 4; ++v) U0[(size_t)v * NC + i] = U[v];
    }

    // ── BCs (triangulate_box tags: 1=left, 2=right, 3=bottom, 4=top) ──
    // static so the capturing lambdas outlive the solve call.
    static std::vector<BC2D> bcs;
    static double s_post[4];
    for (int k = 0; k < 4; ++k) s_post[k] = POST[k];
    bcs.assign(5, BC2D{});
    bcs[1].kind = 3;   // left: time-dependent inflow (mean + v')
    bcs[1].func = [](double, double y, double t, double* W) {
        W[0] = RHO_MEAN; W[1] = u_mean(y); W[2] = vprime(y, t); W[3] = P_MEAN;
    };
    bcs[2].kind = 0;   // right: outflow (transmissive)
    bcs[3].kind = 1;   // bottom: slip wall (reflective)
    bcs[4].kind = 3;   // top: fixed post-shock Dirichlet (launches the oblique shock)
    bcs[4].func = [](double, double, double, double* W) {
        for (int k = 0; k < 4; ++k) W[k] = s_post[k];
    };

    ReconCtx   bj = build_recon_ctx(m, 0.0);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);

    double cfl = envd("SM_CFL", 0.4);
    const char* sm_recon = std::getenv("SM_RECON");
    bool use_mlp = sm_recon && std::string(sm_recon) == "mlp";
    ReconMode rk = use_mlp ? RECON_BJ_VERTEX : RECON_BVD;
    // SM_FLUX: flux selection (default RHLLC). hllc/hll/rhllc/llf/roe.
    static const int sm_flux = []{ const char* f=std::getenv("SM_FLUX"); if(!f)return (int)FLUX_RHLLC;
        std::string s=f; if(s=="hllc")return (int)FLUX_HLLC; if(s=="hll")return (int)FLUX_HLL;
        if(s=="rhllc")return (int)FLUX_RHLLC; if(s=="llf")return (int)FLUX_LLF; if(s=="roe")return (int)FLUX_ROE_EF;
        return (int)FLUX_RHLLC; }();
    std::printf("Shock-mixing layer: cells=%d grid=%dx%d  h=(%.3f,%.3f)  t_end=%.3f  flux=%d recon=%s cfl=%.2f\n",
                NC, Nx, Ny, LX / Nx, LY / Ny, t_end, sm_flux, use_mlp ? "MLP_u1(BJ-vertex)" : "BVD", cfl);
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();
    Solve2DResult r = solve_euler2d(m, eq, U0, t_end, 2, cfl, -1.0, 100000000,
                                    rk, &bj, sm_flux, &bcs, nullptr, &c2);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ── diagnostics: positivity + pressure dump ──
    double rmin = 1e30, rmax = -1e30, pmin = 1e30, pmax = -1e30;
    bool fin = true;
    for (int i = 0; i < NC; ++i) {
        double U[4] = {r.U[0*NC+i], r.U[1*NC+i], r.U[2*NC+i], r.U[3*NC+i]}, W[4];
        if (!std::isfinite(U[0]) || !std::isfinite(U[3])) { fin = false; continue; }
        eq.cons_to_prim(U, W);
        rmin = std::min(rmin, W[0]); rmax = std::max(rmax, W[0]);
        pmin = std::min(pmin, W[3]); pmax = std::max(pmax, W[3]);
    }
    bool completed = r.t >= t_end * 0.999;
    bool ok = fin && completed && rmin > 0 && pmin > 0;
    std::printf("  status=%s  wall=%.1fs steps=%d t=%.4f/%.4f  rho[%.4f,%.4f] p[%.4f,%.4f]\n",
                ok ? "OK" : (completed ? "NEG" : (fin ? "INCOMPLETE" : "DIVERGE")),
                wall, r.n_steps, r.t, t_end, rmin, rmax, pmin, pmax);
    std::fflush(stdout);

    const char* dump = std::getenv("SM_DUMP");
    char defbuf[] = "shock_mixing.txt";
    if (!dump) dump = defbuf;
    FILE* fh = std::fopen(dump, "w");
    if (fh) {
        for (int i = 0; i < NC; ++i) {
            double U[4] = {r.U[0*NC+i], r.U[1*NC+i], r.U[2*NC+i], r.U[3*NC+i]}, W[4];
            eq.cons_to_prim(U, W);
            std::fprintf(fh, "%.6g %.6g %.6g\n", m.cell_centers[i*2], m.cell_centers[i*2+1], W[3]);
        }
        cfd::write_vtk2d_euler(std::string(dump)+".vtk", m, eq, r.U);
        std::fclose(fh);
        std::printf("  dumped pressure -> %s\n", dump);
    }
    return ok ? 0 : 1;
}
