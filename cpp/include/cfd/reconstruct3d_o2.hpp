// cfd/reconstruct3d_o2.hpp — order-2 (quadratic / P2) WLSQ reconstruction in 3D.
//
// 3D mechanical mirror of reconstruct2d_o2.hpp, extended from 5 to 9 quadratic
// coefficients. Per cell fit
//   q(d) = W_c + gx*dx + gy*dy + gz*dz
//        + 0.5*hxx*dx^2 + 0.5*hyy*dy^2 + 0.5*hzz*dz^2
//        + hxy*dx*dy + hxz*dx*dz + hyz*dy*dz
// over the structured-hex vertex-26 neighbour stencil (all (i±1,j±1,k±1) minus
// the centre). The LSQ operator M = (A^T A)^-1 A^T (9 x maxnb) is geometry-only
// -> precomputed once; coeffs co[i] = sum_k M[i,k] * (W_nb - W_c).
//
// Defining property (unit-tested): EXACT on any quadratic field at every cell
// with the full 26-neighbour stencil. This is the high-order base for a later
// deg3t-THINC-QQ-BVD interface reconstruction (Stage 2b).
//
// Neighbour offsets reuse the reconstruct3d.hpp minimum-image periodic wrap +
// grid-snap (round(d/h)*h) so the LSQ matrix is bit-identical seam<->interior.
#pragma once
#include "cfd/mesh.hpp"
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>

namespace cfd {

struct ReconCtx3DO2 {
    int N = 0, max_nb = 0;
    std::vector<int>    nb;     // N*max_nb (-1 pad) — the vertex-26 stencil
    std::vector<double> M;      // N*9*max_nb : operator, coeff i = Σ_k M[(ci*9+i)*max_nb+k]*dW_k
};

// invert a 9x9 (Gauss-Jordan, partial pivot); returns false if singular.
inline bool inv9(double A[9][9], double Inv[9][9]) {
    double M[9][18];
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) { M[i][j] = A[i][j]; M[i][9 + j] = (i == j) ? 1.0 : 0.0; }
    }
    for (int col = 0; col < 9; ++col) {
        int piv = col; double best = std::fabs(M[col][col]);
        for (int r = col + 1; r < 9; ++r) { if (std::fabs(M[r][col]) > best) { best = std::fabs(M[r][col]); piv = r; } }
        if (best < 1e-300) return false;
        if (piv != col) for (int j = 0; j < 18; ++j) std::swap(M[col][j], M[piv][j]);
        double d = M[col][col]; for (int j = 0; j < 18; ++j) M[col][j] /= d;
        for (int r = 0; r < 9; ++r) { if (r == col) continue; double fct = M[r][col];
            for (int j = 0; j < 18; ++j) M[r][j] -= fct * M[col][j]; }
    }
    for (int i = 0; i < 9; ++i) for (int j = 0; j < 9; ++j) Inv[i][j] = M[i][9 + j];
    return true;
}

inline ReconCtx3DO2 build_recon_ctx_3d_o2(const Mesh& m) {
    ReconCtx3DO2 c; const int N = m.n_cells(); c.N = N;
    const double* cc = m.cell_centers.data();
    const int Nx = m.s3_nx, Ny = m.s3_ny, Nz = m.s3_nz;
    const bool px = m.s3_px, py = m.s3_py, pz = m.s3_pz;
    const double hx = m.s3_h[0], hy = m.s3_h[1], hz = m.s3_h[2];
    auto cidx = [Nx, Ny](int i, int j, int k) { return (k * Ny + j) * Nx + i; };

    // ── vertex-26 stencil from the (i,j,k) decomposition. Periodic wrap (mod N)
    //    per axis where the flag is set; drop out-of-range neighbours otherwise.
    std::vector<std::vector<int>> nbl(N);
    for (int k = 0; k < Nz; ++k)
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                int ci = cidx(i, j, k);
                auto& lst = nbl[ci];
                for (int dk = -1; dk <= 1; ++dk)
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int di = -1; di <= 1; ++di) {
                            if (di == 0 && dj == 0 && dk == 0) continue;
                            int ii = i + di, jj = j + dj, kk = k + dk;
                            if (px) ii = (ii % Nx + Nx) % Nx; else if (ii < 0 || ii >= Nx) continue;
                            if (py) jj = (jj % Ny + Ny) % Ny; else if (jj < 0 || jj >= Ny) continue;
                            if (pz) kk = (kk % Nz + Nz) % Nz; else if (kk < 0 || kk >= Nz) continue;
                            lst.push_back(cidx(ii, jj, kk));
                        }
                c.max_nb = std::max(c.max_nb, (int)lst.size());
            }
    c.max_nb = std::max(c.max_nb, 9);
    c.nb.assign((size_t)N * c.max_nb, -1);
    c.M.assign((size_t)N * 9 * c.max_nb, 0.0);

    // minimum-image periodic wrap then snap to round(d/h)*h per axis (mirror of
    // reconstruct3d.hpp). Lwrap[a] = Na*h. Bit-identical offsets seam<->interior.
    const double Lwrap[3] = {Nx * hx, Ny * hy, Nz * hz};
    const double hcan[3]  = {hx, hy, hz};
    auto wrap_snap = [](double d, double L, double h) {
        if (L > 0.0) { if (d > 0.5 * L) d -= L; else if (d < -0.5 * L) d += L; }
        if (h > 0.0) { double k = std::round(d / h); if (std::fabs(k) < 1e15) d = k * h; }
        return d;
    };

    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        int K = (int)nbl[ci].size();
        // weighted rows a_k = [dx,dy,dz, .5dx^2,.5dy^2,.5dz^2, dxy,dxz,dyz]*sqrt(w);
        // w = 1/dist^2.
        std::vector<std::array<double,9>> a(K); std::vector<double> sw(K);
        for (int k = 0; k < K; ++k) {
            int nb = nbl[ci][k]; c.nb[(size_t)ci * c.max_nb + k] = nb;
            double dx = wrap_snap(cc[(size_t)nb*3+0] - cc[(size_t)ci*3+0], Lwrap[0], hcan[0]);
            double dy = wrap_snap(cc[(size_t)nb*3+1] - cc[(size_t)ci*3+1], Lwrap[1], hcan[1]);
            double dz = wrap_snap(cc[(size_t)nb*3+2] - cc[(size_t)ci*3+2], Lwrap[2], hcan[2]);
            double w = 1.0 / std::max(dx*dx + dy*dy + dz*dz, 1e-30);
            double s = std::sqrt(w); sw[k] = s;
            a[k] = {dx*s, dy*s, dz*s,
                    0.5*dx*dx*s, 0.5*dy*dy*s, 0.5*dz*dz*s,
                    dx*dy*s, dx*dz*s, dy*dz*s};
        }
        double ATA[9][9] = {{0}}, Inv[9][9];
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) ATA[i][j] += a[k][i] * a[k][j];
        if (!inv9(ATA, Inv)) continue;   // leave M=0 (degenerate -> first-order)
        // store per neighbour: M[i,k] = (sum_j Inv[i][j]*a[k][j]) * sw[k]
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < 9; ++i) {
                double v = 0; for (int j = 0; j < 9; ++j) v += Inv[i][j] * a[k][j];
                c.M[((size_t)ci * 9 + i) * c.max_nb + k] = v * sw[k];
            }
    }
    return c;
}

// reconstruct the 9 P2 coefficients of variable 'v' (of nvar) per cell into g
// (size N*9, layout [gx,gy,gz, hxx,hyy,hzz, hxy,hxz,hyz]).
inline void reconstruct3d_o2_coeffs(const Mesh& m, const ReconCtx3DO2& c,
                                    const std::vector<double>& W, int nvar, int v,
                                    std::vector<double>& g /*N*9*/) {
    (void)nvar;
    const int N = m.n_cells();
    g.assign((size_t)N * 9, 0.0);
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        double wc = W[(size_t)v * N + ci];
        double co[9] = {0,0,0,0,0,0,0,0,0};
        for (int k = 0; k < c.max_nb; ++k) {
            int nb = c.nb[(size_t)ci * c.max_nb + k]; if (nb < 0) continue;
            double dW = W[(size_t)v * N + nb] - wc;
            for (int i = 0; i < 9; ++i) co[i] += c.M[((size_t)ci * 9 + i) * c.max_nb + k] * dW;
        }
        for (int i = 0; i < 9; ++i) g[(size_t)ci * 9 + i] = co[i];
    }
}

// FUSED all-variable variant (perf, byte-identical to nvar separate reconstruct3d_o2_coeffs
// calls): reconstruct every variable's 9 P2 coeffs in ONE sweep of the M operator table
// (SpMV -> SpMM). The M table (N*9*max_nb doubles) is the biggest, bandwidth-bound geometry
// array; the per-var routine re-streams it nvar times. Here each cell's M row is read ONCE
// per neighbour and reused across all nvar variables -> nvar x fewer DRAM reads of M.
// Layout of g_all = VAR-OUTERMOST:  g_all[(size_t)v*N*9 + ci*9 + i] == the per-var g[ci*9+i].
// Byte-exact: for each variable the neighbour (k) accumulation order and every M[..]*dW
// product are identical to reconstruct3d_o2_coeffs(...,v,g).
inline void reconstruct3d_o2_coeffs_allvars(const Mesh& m, const ReconCtx3DO2& c,
                                    const std::vector<double>& W, int nvar,
                                    std::vector<double>& g_all /*nvar*N*9, var-outermost*/) {
    const int N = m.n_cells();
    const int mnb = c.max_nb;
    const size_t per_var = (size_t)N * 9;
    g_all.resize((size_t)nvar * per_var);
    #pragma omp parallel
    {
        std::vector<double> co((size_t)nvar * 9), wc((size_t)nvar);
        #pragma omp for schedule(static)
        for (int ci = 0; ci < N; ++ci) {
            for (int v = 0; v < nvar; ++v) wc[v] = W[(size_t)v * N + ci];
            std::fill(co.begin(), co.end(), 0.0);
            const size_t nbbase = (size_t)ci * mnb;
            const size_t mrow0  = (size_t)ci * 9;            // M[(ci*9+i)*mnb + k]
            for (int k = 0; k < mnb; ++k) {
                int nb = c.nb[nbbase + k]; if (nb < 0) continue;
                double Mk[9];                                 // stream this cell's M column ONCE
                for (int i = 0; i < 9; ++i) Mk[i] = c.M[(mrow0 + i) * (size_t)mnb + k];
                for (int v = 0; v < nvar; ++v) {              // ...reuse it for every variable
                    double dW = W[(size_t)v * N + nb] - wc[v];
                    double* cov = &co[(size_t)v * 9];
                    for (int i = 0; i < 9; ++i) cov[i] += Mk[i] * dW;
                }
            }
            for (int v = 0; v < nvar; ++v) {
                double* dst = &g_all[(size_t)v * per_var + (size_t)ci * 9];
                const double* cov = &co[(size_t)v * 9];
                for (int i = 0; i < 9; ++i) dst[i] = cov[i];
            }
        }
    }
}

} // namespace cfd
