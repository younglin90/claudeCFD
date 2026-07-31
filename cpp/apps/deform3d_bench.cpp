// apps/deform3d_bench.cpp — the famous Enright/LeVeque 3D DEFORMATION FIELD test.
//
// A sphere is advected by a prescribed, divergence-free, time-reversing velocity
// field on [0,1]^3 (the velocity vanishes on the walls so the box is effectively
// closed). Forward to t=T/2 the sphere deforms into a thin spiralling filament
// (the dramatic max-deformation state, paper Fig 5a); the cos(pi t/T) factor
// reverses the flow so at t=T the sphere should RETURN to its initial shape.
// This validates the 3D reconstruction in pure scalar-advection mode (the 3D
// analogue of the 2D leveque_bench). Plain MUSCL recon now; deg3t-BVD swaps in
// later. New file: does not modify any existing header or 2D bench.
//
// Metrics at t=T: L1 shape-return error, volume(mass) conservation VT/V0, and
// the g range (boundedness). Dumps z=0.35 slices at t=0, T/2, T.
//
// Env: DEF_N (mesh N^3, default 64), DEF_T (period, default 3.0),
//      DEF_RECON ("bj"/MUSCL default | "bvd" -> deg3t-THINC-QQ + min-TBV BVD).
#include "cfd/mesh.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/solver_advect3d.hpp"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

using namespace cfd;

// Enright/LeVeque deformation velocity (period T): divergence-free, vanishes on
// the cube boundary, reverses sign at t=T/2 via cos(pi t/T).
static double g_T = 3.0;
static void deform_vel(double x, double y, double z, double t, double* uvw) {
    const double pi = M_PI;
    double ct = std::cos(pi * t / g_T);
    double sx = std::sin(pi * x), sy = std::sin(pi * y), sz = std::sin(pi * z);
    uvw[0] =  2.0 * sx * sx * std::sin(2.0 * pi * y) * std::sin(2.0 * pi * z) * ct;
    uvw[1] = -       sy * sy * std::sin(2.0 * pi * x) * std::sin(2.0 * pi * z) * ct;
    uvw[2] = -       sz * sz * std::sin(2.0 * pi * x) * std::sin(2.0 * pi * y) * ct;
}

// Dump the k-slice (z ~ z_plane) as "x y g" rows for tricontourf.
static void dump_slice(const Mesh& m, const std::vector<double>& g,
                       int Nx, int Ny, int Nz, int k, const char* path) {
    FILE* fh = std::fopen(path, "w");
    if (!fh) { std::fprintf(stderr, "could not open %s\n", path); return; }
    for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i) {
            int c = (k * Ny + j) * Nx + i;
            std::fprintf(fh, "%.6g %.6g %.6g\n",
                         m.cell_centers[(size_t)c * 3 + 0],
                         m.cell_centers[(size_t)c * 3 + 1], g[c]);
        }
    std::fclose(fh);
}

int main() {
    const int N  = std::getenv("DEF_N") ? std::atoi(std::getenv("DEF_N")) : 64;
    const double T = std::getenv("DEF_T") ? std::atof(std::getenv("DEF_T")) : 3.0;
    g_T = T;
    const int Nx = N, Ny = N, Nz = N;

    const char* rec_env = std::getenv("DEF_RECON");
    const bool use_bvd = rec_env && std::strcmp(rec_env, "bvd") == 0;
    const int recon = use_bvd ? ADV3_BVD : ADV3_BJ_VERTEX;
    const double beta_l = 1.6, beta_s = 0.8;

    // Domain [0,1]^3, all non-periodic (velocity vanishes on walls => closed box).
    Mesh m = build_structured_3d(Nx, Ny, Nz, 1.0, 1.0, 1.0,
                                 /*px*/false, /*py*/false, /*pz*/false);
    const int NC = m.n_cells();
    ReconCtx3D ctx = build_recon_ctx_3d(m);
    // o2 (P2-LSQ) context for the BVD interface polynomial (only when needed).
    ReconCtx3DO2 o2ctx;
    if (use_bvd) o2ctx = build_recon_ctx_3d_o2(m);
    const ReconCtx3DO2* o2p = use_bvd ? &o2ctx : nullptr;

    // Initial scalar: sharp VOF indicator of a sphere R=0.15 at (0.35,0.35,0.35).
    const double R = 0.15, cx = 0.35, cy = 0.35, cz = 0.35;
    std::vector<double> g0(NC);
    for (int c = 0; c < NC; ++c) {
        double dx = m.cell_centers[(size_t)c * 3 + 0] - cx;
        double dy = m.cell_centers[(size_t)c * 3 + 1] - cy;
        double dz = m.cell_centers[(size_t)c * 3 + 2] - cz;
        g0[c] = (std::sqrt(dx*dx + dy*dy + dz*dz) < R) ? 1.0 : 0.0;
    }

    // z = 0.35 plane: k = round(0.35*Nz/1.0 - 0.5).
    int kslice = (int)std::lround(0.35 * Nz - 0.5);
    if (kslice < 0) kslice = 0; if (kslice >= Nz) kslice = Nz - 1;

    // ---- Full run to t=T (return) ----
    int steps = 0; double tend = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> gT = solve_advect3d(m, g0, T, &deform_vel,
                                            /*cfl*/0.5, /*dt_fixed*/-1.0,
                                            /*integrator*/2, recon, &ctx,
                                            &steps, &tend, o2p, beta_l, beta_s);
    auto t1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();

    // ---- Second run to t=T/2 (max-deformation filament) ----
    int hsteps = 0; double htend = 0.0;
    std::vector<double> gHalf = solve_advect3d(m, g0, 0.5 * T, &deform_vel,
                                               /*cfl*/0.5, /*dt_fixed*/-1.0,
                                               /*integrator*/2, recon, &ctx,
                                               &hsteps, &htend, o2p, beta_l, beta_s);

    // ---- Metrics at t=T ----
    double E1 = 0.0, V0 = 0.0, VT = 0.0, gmin = 1e300, gmax = -1e300;
    for (int c = 0; c < NC; ++c) {
        double V = m.cell_volumes[c];
        E1 += std::fabs(gT[c] - g0[c]) * V;
        V0 += g0[c] * V;
        VT += gT[c] * V;
        if (gT[c] < gmin) gmin = gT[c];
        if (gT[c] > gmax) gmax = gT[c];
    }
    double volRatio = (V0 != 0.0) ? VT / V0 : 0.0;

    // ---- Dumps (z=0.35 slice). BVD writes a distinct _bvd suffix so the MUSCL
    //      baseline slices are preserved for the 2-row comparison figure.
    //      Canonical results dir (validation battery); falls back to /tmp/mbq. ----
    const char* dir_env = std::getenv("DEF_OUTDIR");
    const std::string dir = dir_env ? dir_env : "/tmp/mbq/deform3d";
    const std::string sfx = use_bvd ? "_bvd" : "";
    std::string fi = dir + "/def3d_init" + sfx + ".txt";
    std::string fh = dir + "/def3d_half" + sfx + ".txt";
    std::string ff = dir + "/def3d_final" + sfx + ".txt";
    dump_slice(m, g0,    Nx, Ny, Nz, kslice, fi.c_str());
    dump_slice(m, gHalf, Nx, Ny, Nz, kslice, fh.c_str());
    dump_slice(m, gT,    Nx, Ny, Nz, kslice, ff.c_str());

    std::printf("deform3d recon=%s N=%d E1=%.6e volRatio=%.6f g_range=[%.4f,%.4f] "
                "steps=%d t=%.4f wall=%.1fs\n",
                use_bvd ? "bvd" : "bj", N, E1, volRatio, gmin, gmax, steps, tend, wall);
    std::printf("  (kslice=%d half_steps=%d half_t=%.4f)\n", kslice, hsteps, htend);
    return 0;
}
