// kh_bench.cpp — transonic Kelvin-Helmholtz double shear layer = a SHEAR+SHOCK
// discriminator. Two counter-streaming layers with a density jump (slip lines) at a
// convective Mach ~1, so the roll-ups generate compressions/shocklets. Tests whether
// a reconstruction preserves KH roll-ups (omega / enstrophy / Q-vortices) WHILE
// keeping positivity (rho,p>0) at the compressions — the exact regime where MLP_S
// (limiter fully OFF in shear) risks negativity but T-MLP-u-D (normal box KEPT)
// should not. Runs ONE scheme (RECON_BJ_VERTEX); env MLP_S / TMLPU_DIR_LIMIT pick the
// variant (static getenv => one variant per process; run the binary 3x to compare).
// Args: [N] [t_end]. Env: KH_U0 (shear speed), KH_P (pressure -> Mach), KH_W0 (pert).
#include "cfd/solver_euler2d.hpp"
#include "cfd/reconstruct2d_o2.hpp"
#include "cfd/diagnostics.hpp"
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace cfd;
static const double GAMMA = 1.4;

int main(int argc, char** argv) {
    int N = argc>1 ? std::atoi(argv[1]) : 128;
    double t_end = argc>2 ? std::atof(argv[2]) : 0.8;
    double U0 = std::getenv("KH_U0") ? std::atof(std::getenv("KH_U0")) : 1.0;
    double P0 = std::getenv("KH_P")  ? std::atof(std::getenv("KH_P"))  : 1.0;
    double W0 = std::getenv("KH_W0") ? std::atof(std::getenv("KH_W0")) : 0.1;
    Mesh m = triangulate_box(N, N, 1.0, 1.0);
    Euler2D eq{GAMMA};
    const int NC = m.n_cells();
    // SMOOTH (tanh-ramped) double shear layer — McNally-style, avoids the grid-scale
    // discontinuity of a sharp step (which spuriously seeds instability for any
    // anti-diffusive scheme and unfairly punishes it). Ramp width delta ~ 2-3 cells.
    double delta = std::getenv("KH_DELTA") ? std::atof(std::getenv("KH_DELTA")) : 0.02;
    std::vector<double> U(4*NC);
    for (int i = 0; i < NC; ++i) {
        double x = m.cell_centers[i*2], y = m.cell_centers[i*2+1];
        double sft = 0.5*(std::tanh((y-0.25)/delta) - std::tanh((y-0.75)/delta)); // ~1 mid, ~0 out
        double rho = 1.0 + sft;            // 1 -> 2
        double u   = -U0 + 2.0*U0*sft;     // -U0 -> +U0
        double v   = W0*std::sin(4.0*M_PI*x)
                     *(std::exp(-(y-0.25)*(y-0.25)/(2*0.05*0.05))
                      +std::exp(-(y-0.75)*(y-0.75)/(2*0.05*0.05)));
        double W[4] = {rho,u,v,P0}, Uc[4]; eq.prim_to_cons(W, Uc);
        for (int vv = 0; vv < 4; ++vv) U[(size_t)vv*NC+i] = Uc[vv];
    }
    double cmid = std::sqrt(GAMMA*P0/2.0), cout = std::sqrt(GAMMA*P0/1.0);
    double Mc = 2.0*U0/(cmid+cout);   // convective Mach of the shear
    double cfl = std::getenv("KH_CFL") ? std::atof(std::getenv("KH_CFL")) : 0.4;
    ReconCtx bj = build_recon_ctx(m, 0.0);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    // KH_RECON=bvd -> RECON_BVD (BJ-vertex smooth <-> BVD_SHARP candidate, TBV select).
    const char* rk = std::getenv("KH_RECON");
    bool use_bvd = rk && std::string(rk) == "bvd";
    int recon = use_bvd ? RECON_BVD : RECON_BJ_VERTEX;
    FluxKind khflux = FLUX_HLLC;
    if (const char* kf = std::getenv("KH_FLUX")) { std::string s=kf; if(s=="rhllc") khflux=FLUX_RHLLC; else if(s=="roe"||s=="rroe") khflux=FLUX_RROE; else if(s=="hll") khflux=FLUX_HLL; else if(s=="llf") khflux=FLUX_LLF; }
    Solve2DResult r = solve_euler2d(m, eq, U, t_end, 2, cfl, -1.0, 100000000,
                                    recon, &bj, khflux, nullptr, nullptr, &c2);
    double rmin = 1e9, pmin = 1e9, omax = 0, ens = 0;
    std::vector<double> u(NC), v(NC);
    for (int i = 0; i < NC; ++i) {
        double Uc[4] = {r.U[0*NC+i],r.U[1*NC+i],r.U[2*NC+i],r.U[3*NC+i]}, W[4];
        eq.cons_to_prim(Uc, W); rmin=std::min(rmin,W[0]); pmin=std::min(pmin,W[3]);
        u[i]=W[1]; v[i]=W[2];
    }
    for (int ci = 0; ci < NC; ++ci) {
        double ru0=0,ru1=0,rv0=0,rv1=0;
        for (int k = 0; k < bj.max_nb; ++k) { int nb=bj.nb[(size_t)ci*bj.max_nb+k]; if(nb<0)continue;
            double wk=bj.w[(size_t)ci*bj.max_nb+k];
            double dx=bj.d[((size_t)ci*bj.max_nb+k)*2+0], dy=bj.d[((size_t)ci*bj.max_nb+k)*2+1];
            ru0+=wk*dx*(u[nb]-u[ci]); ru1+=wk*dy*(u[nb]-u[ci]);
            rv0+=wk*dx*(v[nb]-v[ci]); rv1+=wk*dy*(v[nb]-v[ci]); }
        double dudy=bj.ATA_inv[ci*4+2]*ru0+bj.ATA_inv[ci*4+3]*ru1;
        double dvdx=bj.ATA_inv[ci*4+0]*rv0+bj.ATA_inv[ci*4+1]*rv1;
        double om=dvdx-dudy; omax=std::max(omax,std::fabs(om)); ens+=om*om*m.cell_volumes[ci];
    }
    const char* kd = std::getenv("KH_DUMP");
    if (kd) { FILE* fh = std::fopen(kd, "w");
        for (int i = 0; i < NC; ++i) {
            double Uc[4]={r.U[0*NC+i],r.U[1*NC+i],r.U[2*NC+i],r.U[3*NC+i]}, W[4];
            eq.cons_to_prim(Uc, W);
            std::fprintf(fh, "%.6g %.6g %.6g\n", m.cell_centers[i*2], m.cell_centers[i*2+1], W[0]);
        }
        std::fclose(fh);
    }
    QDiag q = q_criterion_roi(m, eq, bj, r.U, 0.0,1.0,0.0,1.0, 0.10);
    const char* tag = use_bvd ? "BVD      "
                    : (std::getenv("TMLPU_DIR_LIMIT") ? "T-MLP-u-D"
                    : (std::getenv("MLP_S") ? "T-MLP-u-S" : "mlp_u1   "));
    std::printf("KH N=%d cells=%d t=%.2f Mc=%.2f U0=%.2f P=%.2f | %s: posOK=%d rho_min=%.4f "
                "p_min=%.4f omega_peak=%.2f enstrophy=%.3f Qvort=%d(S%d/M%d/L%d) steps=%d\n",
        N,NC,t_end,Mc,U0,P0, tag, (rmin>0&&pmin>0), rmin,pmin, omax,ens,
        q.n_vortices,q.n_small,q.n_mid,q.n_large, r.n_steps);
    return 0;
}
