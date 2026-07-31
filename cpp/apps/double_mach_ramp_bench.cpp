// double_mach_ramp_bench.cpp — Double-Mach reflection in the RAMP PHYSICAL frame
// (Dumbser 2007 JCP 226 sec.7.5 / Cheng 2021 JCP 428 sec.4.2.4). Reads an unstructured
// triangular mesh of the pentagon flow region above a 30-deg ramp (apex x=0.2, top wall
// y=2, right outflow x=3). Ramp-frame post-shock IC (8.0,8.25,0.0,116.5) for x<0.1 (u
// horizontal, v=0). Left = post-shock inflow, right = outflow, all walls (flat bottom
// x<0.2 + 30-deg ramp + top y=2) = reflective slip (face-normal mirror). Scheme (S1 =
// MUSCL-THINC/QQ-BVD) + flux via env (BVD_CHENG3, BVD_BETA_L/S, MLP_U2, DM_FLUX default HLL).
// Args: [mesh2d] [t_end]. Env: DM_DUMP, DM_FLUX(hll), DM_CFL, DM_BJ_IDW.
#include "cfd/solver_euler2d.hpp"
#include "cfd/io_vtk.hpp"
#include "cfd/mesh.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
using namespace cfd;
static const double POST[4] = {8.0, 8.25, 0.0, 116.5};   // ramp-frame post-shock (v=0, horizontal)
static const double PRE[4]  = {1.4, 0.0, 0.0, 1.0};

int main(int argc, char** argv) {
    const char* meshfile = argc > 1 ? argv[1] : "meshes/dmr_ramp.mesh2d";
    double t_end = argc > 2 ? std::atof(argv[2]) : 0.2;
    std::ifstream in(meshfile);
    if (!in) { std::fprintf(stderr, "cannot open mesh %s\n", meshfile); return 1; }
    int Nn = 0, Ne = 0; in >> Nn >> Ne;
    std::vector<double> nodes(2 * Nn);
    for (int i = 0; i < Nn; ++i) in >> nodes[2*i] >> nodes[2*i+1];
    std::vector<std::vector<int>> elems(Ne, std::vector<int>(3));
    for (int i = 0; i < Ne; ++i) in >> elems[i][0] >> elems[i][1] >> elems[i][2];
    in.close();
    // Boundary classifier by face centroid: left x=0 -> inflow(1), right x=3 -> outflow(2),
    // everything else (flat bottom y=0, 30-deg ramp, top y=2) -> reflective wall(3).
    auto classify = [](double cx, double cy, double nx, double ny) -> int {
        (void)cy; (void)nx; (void)ny;
        if (cx < 1e-4)        return 1;   // left inflow  (x=0)
        if (cx > 3.0 - 1e-4)  return 2;   // right outflow (x=3)
        return 3;                          // wall: flat bottom + ramp + top
    };
    Mesh m = build_unstructured_2d(nodes, elems, classify, {"inflow", "outflow", "wall"});
    Euler2D eq{1.4};
    const int N = m.n_cells();
    // IC: ramp-frame post-shock for x<0.1, pre-shock elsewhere.
    std::vector<double> U0(4 * N);
    for (int i = 0; i < N; ++i) {
        double x = m.cell_centers[i*2];
        const double* s = (x < 0.1) ? POST : PRE;
        double U[4]; eq.prim_to_cons(s, U);
        for (int v = 0; v < 4; ++v) U0[(size_t)v*N + i] = U[v];
    }
    std::vector<BC2D> bcs(4);
    bcs[1].kind = 2; for (int v = 0; v < 4; ++v) bcs[1].state[v] = POST[v];  // inflow post-shock
    bcs[2].kind = 0;                                                          // outflow
    bcs[3].kind = 1;                                                          // reflective wall (mirror face-normal vel)
    static const double bj_idw = []{ const char* e = std::getenv("DM_BJ_IDW"); return e ? std::atof(e) : 2.0; }();
    ReconCtx bj = build_recon_ctx(m, bj_idw);
    ReconCtxO2 c2 = build_recon_ctx_o2(m);
    static const int dmflux = []{ const char* f = std::getenv("DM_FLUX"); if (!f) return (int)FLUX_HLL;
        std::string s = f; if (s=="hll") return (int)FLUX_HLL; if (s=="hllc") return (int)FLUX_HLLC;
        if (s=="rhllc") return (int)FLUX_RHLLC; if (s=="llf") return (int)FLUX_LLF;
        if (s=="roe") return (int)FLUX_ROE_EF; if (s=="rroe") return (int)FLUX_RROE; return (int)FLUX_HLL; }();
    static const double cfl = []{ const char* e = std::getenv("DM_CFL"); return e ? std::atof(e) : 0.3; }();
    std::printf("ramp mesh: cells=%d nodes=%d bj_idw=%.1f flux=%d cfl=%.2f\n", N, Nn, bj_idw, dmflux, cfl);
    Solve2DResult r = solve_euler2d(m, eq, U0, t_end, 2, cfl, -1.0, 100000000,
                                    RECON_BVD, &bj, dmflux, &bcs, nullptr, &c2);
    double rmin = 1e9, rmax = -1e9, pmin = 1e9;
    for (int i = 0; i < N; ++i) { double U[4] = {r.U[0*N+i],r.U[1*N+i],r.U[2*N+i],r.U[3*N+i]}, W[4];
        eq.cons_to_prim(U, W); rmin = std::min(rmin, W[0]); rmax = std::max(rmax, W[0]); pmin = std::min(pmin, W[3]); }
    std::printf("  RAMP-BVD: posOK=%d rho[%.3f,%.3f] p_min=%.4f steps=%d t=%.4f\n",
                (rmin > 0 && pmin > 0), rmin, rmax, pmin, r.n_steps, r.t);
    const char* dump = std::getenv("DM_DUMP");
    if (dump) { FILE* fh = std::fopen(dump, "w");
        for (int i = 0; i < N; ++i) { double U[4] = {r.U[0*N+i],r.U[1*N+i],r.U[2*N+i],r.U[3*N+i]}, W[4];
            eq.cons_to_prim(U, W); std::fprintf(fh, "%.6g %.6g %.6g\n", m.cell_centers[i*2], m.cell_centers[i*2+1], W[0]); }
        std::fclose(fh);
        cfd::write_vtk2d_euler(std::string(dump) + ".vtk", m, eq, r.U); }
    return 0;
}
