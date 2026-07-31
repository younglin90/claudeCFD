// apps/euler3d_bench.cpp — Stage-1 sanity for the 3D Cartesian-hex Euler solver.
//   SANITY-A: 3D Sod shock tube, extruded in y,z (periodic) -> must stay
//             y,z-invariant and reproduce the 1D Sod density range.
//   SANITY-B: uniform flow on a fully periodic box -> exact steady state.
#include "cfd/mesh.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/solver_euler3d.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace cfd;

int main() {
    Euler3D eq; eq.gamma = 1.4;

    // ===== SANITY-A: 3D Sod, extruded in y,z =====
    {
        const int Nx = 200, Ny = 4, Nz = 4;
        Mesh m = build_structured_3d(Nx, Ny, Nz, 1.0, 0.02, 0.02,
                                     /*px*/false, /*py*/true, /*pz*/true);
        const int N = m.n_cells();
        std::vector<double> U0((size_t)5 * N);
        for (int c = 0; c < N; ++c) {
            double x = m.cell_centers[(size_t)c * 3 + 0];
            double W[5];
            if (x < 0.5) { W[0]=1.0;   W[1]=0; W[2]=0; W[3]=0; W[4]=1.0; }
            else         { W[0]=0.125; W[1]=0; W[2]=0; W[3]=0; W[4]=0.1; }
            double Uc[5]; eq.prim_to_cons(W, Uc);
            for (int v = 0; v < 5; ++v) U0[(size_t)v * N + c] = Uc[v];
        }
        ReconCtx3D ctx = build_recon_ctx_3d(m);
        // x_min tag1, x_max tag2 -> transmissive; y,z periodic (no bdry faces).
        std::vector<BC3D> bcs(3);
        bcs[1].kind = 0; bcs[2].kind = 0;

        Solve3DResult R = solve_euler3d(m, eq, U0, /*t_end*/0.2, /*integrator*/2,
                                        /*cfl*/0.4, /*dt_fixed*/-1.0, /*max_steps*/100000000,
                                        RECON3_BJ_VERTEX, &ctx, FLUX3_RHLLC, &bcs);

        // (i) y,z-invariance: per i-column, max deviation of rho across the Ny*Nz
        // cells sharing that i. c(i,j,k) = (k*Ny+j)*Nx+i.
        double yz_inv = 0.0;
        for (int i = 0; i < Nx; ++i) {
            double rmn = 1e300, rmx = -1e300;
            for (int k = 0; k < Nz; ++k) for (int j = 0; j < Ny; ++j) {
                int c = (k * Ny + j) * Nx + i;
                double rho = R.U[(size_t)0 * N + c];
                if (rho < rmn) rmn = rho; if (rho > rmx) rmx = rho;
            }
            yz_inv = std::max(yz_inv, rmx - rmn);
        }
        // (ii) rho range + finiteness.
        double rmin = 1e300, rmax = -1e300; bool finite = true;
        for (int c = 0; c < N; ++c) {
            double rho = R.U[(size_t)0 * N + c];
            if (!std::isfinite(rho)) finite = false;
            if (rho < rmin) rmin = rho; if (rho > rmax) rmax = rho;
        }
        std::printf("SANITY-A Sod: yz_invariance=%.3e rho=[%.6f,%.6f] finite=%d steps=%d t=%.6f\n",
                    yz_inv, rmin, rmax, (int)finite, R.n_steps, R.t);
        bool passA = (yz_inv < 1e-9) && finite
                   && std::fabs(rmin - 0.125) < 1e-3 && std::fabs(rmax - 1.0) < 1e-3;
        std::printf("SANITY-A %s\n", passA ? "PASS" : "FAIL");
    }

    // ===== SANITY-B: uniform flow, fully periodic =====
    {
        const int Nx = 8, Ny = 8, Nz = 8;
        Mesh m = build_structured_3d(Nx, Ny, Nz, 1.0, 1.0, 1.0,
                                     /*px*/true, /*py*/true, /*pz*/true);
        const int N = m.n_cells();
        double W[5] = {1.0, 2.0, 1.0, -0.5, 1.0};
        double Uc[5]; eq.prim_to_cons(W, Uc);
        std::vector<double> U0((size_t)5 * N);
        for (int c = 0; c < N; ++c)
            for (int v = 0; v < 5; ++v) U0[(size_t)v * N + c] = Uc[v];

        ReconCtx3D ctx = build_recon_ctx_3d(m);
        Solve3DResult R = solve_euler3d(m, eq, U0, /*t_end*/1e9, /*integrator*/2,
                                        /*cfl*/0.4, /*dt_fixed*/0.005, /*max_steps*/20,
                                        RECON3_BJ_VERTEX, &ctx, FLUX3_RHLLC, /*bcs*/nullptr);
        double maxdev = 0.0;
        for (size_t i = 0; i < U0.size(); ++i)
            maxdev = std::max(maxdev, std::fabs(R.U[i] - U0[i]));
        std::printf("SANITY-B uniform: max_dev=%.3e steps=%d t=%.6f\n", maxdev, R.n_steps, R.t);
        std::printf("SANITY-B %s\n", (maxdev < 1e-10) ? "PASS" : "FAIL");
    }

    return 0;
}
