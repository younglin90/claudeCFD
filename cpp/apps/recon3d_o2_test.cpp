// recon3d_o2_test.cpp — order-2 (P2 quadratic) 3D WLSQ reconstruction must be
// EXACT on a quadratic field (its defining property). Gate for the high-order 3D
// base feeding a later deg3t-THINC-QQ-BVD reconstruction.
//
// On a uniform 16^3 mesh of [0,1]^3 (non-periodic) sample the exact quadratic
//   f = 1 + 2x - 3y + 0.5z + 1.1x^2 - 0.7y^2 + 0.3z^2 + 0.9xy - 0.4xz + 0.6yz
// at the cell centres, recover the 9 coeffs, and compare to the analytic
// derivatives at every INTERIOR (full-26-stencil) cell. PASS: max err < 1e-9.
#include "cfd/reconstruct3d_o2.hpp"
#include <cstdio>
#include <cmath>
using namespace cfd;

static double F(double x, double y, double z) {
    return 1.0 + 2.0*x - 3.0*y + 0.5*z
         + 1.1*x*x - 0.7*y*y + 0.3*z*z
         + 0.9*x*y - 0.4*x*z + 0.6*y*z;
}

int main() {
    const int Nx = 16, Ny = 16, Nz = 16;
    Mesh m = build_structured_3d(Nx, Ny, Nz, 1.0, 1.0, 1.0,
                                 /*px*/false, /*py*/false, /*pz*/false);
    ReconCtx3DO2 c = build_recon_ctx_3d_o2(m);
    const int N = m.n_cells();

    std::vector<double> W(N);
    for (int i = 0; i < N; ++i)
        W[i] = F(m.cell_centers[(size_t)i*3+0],
                 m.cell_centers[(size_t)i*3+1],
                 m.cell_centers[(size_t)i*3+2]);

    std::vector<double> g;
    reconstruct3d_o2_coeffs(m, c, W, 1, 0, g);

    // analytic derivatives (gradients position-dependent, Hessian constant):
    //   gx = 2 + 2*1.1*x - 0.4*z + 0.9*y
    //   gy = -3 - 2*0.7*y + 0.9*x + 0.6*z
    //   gz = 0.5 + 2*0.3*z - 0.4*x + 0.6*y
    //   hxx=2.2, hyy=-1.4, hzz=0.6, hxy=0.9, hxz=-0.4, hyz=0.6
    const char* names[9] = {"gx","gy","gz","hxx","hyy","hzz","hxy","hxz","hyz"};
    double maxerr[9] = {0,0,0,0,0,0,0,0,0};
    int checked = 0;

    auto cidx = [Nx, Ny](int i, int j, int k) { return (k * Ny + j) * Nx + i; };
    for (int k = 1; k < Nz-1; ++k)
      for (int j = 1; j < Ny-1; ++j)
        for (int i = 1; i < Nx-1; ++i) {   // interior cells: full 26 stencil
            int ci = cidx(i, j, k);
            double x = m.cell_centers[(size_t)ci*3+0];
            double y = m.cell_centers[(size_t)ci*3+1];
            double z = m.cell_centers[(size_t)ci*3+2];
            double ref[9] = {
                2.0 + 2.0*1.1*x - 0.4*z + 0.9*y,    // gx
               -3.0 - 2.0*0.7*y + 0.9*x + 0.6*z,    // gy
                0.5 + 2.0*0.3*z - 0.4*x + 0.6*y,    // gz
                2.2, -1.4, 0.6,                      // hxx,hyy,hzz
                0.9, -0.4, 0.6 };                    // hxy,hxz,hyz
            const double* G = &g[(size_t)ci*9];
            for (int q = 0; q < 9; ++q) {
                double e = std::fabs(G[q] - ref[q]);
                if (e > maxerr[q]) maxerr[q] = e;
            }
            ++checked;
        }

    double overall = 0;
    for (int q = 0; q < 9; ++q) overall = std::max(overall, maxerr[q]);

    std::printf("recon3d_o2 per-coeff max_err over %d interior cells:\n", checked);
    for (int q = 0; q < 9; ++q)
        std::printf("  %-4s : %.3e\n", names[q], maxerr[q]);

    bool pass = (overall < 1e-9) && (checked > 0);
    std::printf("recon3d_o2 quadratic-recovery: max_err=%.3e %s\n",
                overall, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
