// cfd/reconstruct3d_o2_unstr.hpp — UNSTRUCTURED P2 (order-2) WLSQ context builder.
//
// Same P2 LSQ operator as reconstruct3d_o2.hpp, but the neighbour stencil is the
// NODE-SHARING ring (all cells sharing >=1 vertex with ci — the unstructured analog of
// the structured vertex-26 stencil) and offsets are raw centre-to-centre (no s3_h /
// no periodic wrap-snap). reconstruct3d_o2_coeffs() consumes the resulting ReconCtx3DO2
// unchanged (it is geometry-agnostic). Needs m.cell_nodes populated (build_unstructured_3d).
#pragma once
#include "cfd/reconstruct3d_o2.hpp"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace cfd {

inline ReconCtx3DO2 build_recon_ctx_3d_o2_unstr(const Mesh& m) {
    ReconCtx3DO2 c; const int N = m.n_cells(); c.N = N;
    const double* cc = m.cell_centers.data();

    // node -> incident cells
    int maxnode = -1;
    for (auto& cn : m.cell_nodes) for (int v : cn) maxnode = std::max(maxnode, v);
    std::vector<std::vector<int>> cells_of_node(maxnode + 1);
    for (int ci = 0; ci < N; ++ci) for (int v : m.cell_nodes[ci]) cells_of_node[v].push_back(ci);

    // node-sharing stencil per cell (sorted unique, excluding self)
    std::vector<std::vector<int>> nbl(N);
    for (int ci = 0; ci < N; ++ci) {
        std::vector<int>& s = nbl[ci];
        for (int v : m.cell_nodes[ci]) for (int cj : cells_of_node[v]) if (cj != ci) s.push_back(cj);
        std::sort(s.begin(), s.end()); s.erase(std::unique(s.begin(), s.end()), s.end());
        c.max_nb = std::max(c.max_nb, (int)s.size());
    }
    c.max_nb = std::max(c.max_nb, 9);
    c.nb.assign((size_t)N * c.max_nb, -1);
    c.M.assign((size_t)N * 9 * c.max_nb, 0.0);

    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        const int K = (int)nbl[ci].size();
        std::vector<std::array<double,9>> a(K); std::vector<double> sw(K);
        for (int k = 0; k < K; ++k) {
            int nb = nbl[ci][k]; c.nb[(size_t)ci * c.max_nb + k] = nb;
            double dx = cc[(size_t)nb*3+0]-cc[(size_t)ci*3+0];
            double dy = cc[(size_t)nb*3+1]-cc[(size_t)ci*3+1];
            double dz = cc[(size_t)nb*3+2]-cc[(size_t)ci*3+2];
            double w = 1.0/std::max(dx*dx+dy*dy+dz*dz, 1e-30); double s = std::sqrt(w); sw[k]=s;
            a[k] = {dx*s,dy*s,dz*s, 0.5*dx*dx*s,0.5*dy*dy*s,0.5*dz*dz*s, dx*dy*s,dx*dz*s,dy*dz*s};
        }
        double ATA[9][9]={{0}}, Inv[9][9];
        for (int k=0;k<K;++k) for(int i=0;i<9;++i) for(int j=0;j<9;++j) ATA[i][j]+=a[k][i]*a[k][j];
        // COLUMN NORMALIZATION (Jacobi precond of normal equations): the P2 basis columns
        // have wildly different scales (linear ~dx, quadratic ~0.5 dx^2) -> cond(ATA) ~1e6-1e11
        // on anisotropic/varied unstructured cells -> inv9 amplifies garbage in the ill-
        // conditioned directions (spurious gradients -> spurious velocity). Scale each column
        // by 1/||col|| so ATA_s has unit diagonal (a correlation matrix, cond = geometry only),
        // add a small ridge so residually-degenerate directions damp to ~0 (graceful loss of
        // curvature) instead of blowing up. Un-scale the inverse afterwards: ATA=D ATA_s D =>
        // Inv[i][j] = Inv_s[i][j]/(d_i d_j).  opt-out: RECON_O2_NONORM.
        static const bool nonorm = std::getenv("RECON_O2_NONORM") != nullptr;
        static const double ridge = []{ const char* e=std::getenv("RECON_O2_RIDGE"); return (e&&e[0])?std::atof(e):1e-6; }();
        double d[9];
        for (int j=0;j<9;++j){ d[j]=std::sqrt(ATA[j][j]); if(!(d[j]>0.0)) d[j]=1.0; }
        if (!nonorm) {
            double ATAs[9][9];
            for(int i=0;i<9;++i) for(int j=0;j<9;++j) ATAs[i][j]=ATA[i][j]/(d[i]*d[j]);
            for(int j=0;j<9;++j) ATAs[j][j]+=ridge;                 // unit-diagonal -> ridge is relative
            double Invs[9][9];
            if (!inv9(ATAs, Invs)) continue;                       // truly singular -> first-order
            for(int i=0;i<9;++i) for(int j=0;j<9;++j) Inv[i][j]=Invs[i][j]/(d[i]*d[j]);
        } else {
            if (!inv9(ATA, Inv)) continue;   // degenerate (too few nbrs) -> M=0 -> first-order
        }
        for (int k=0;k<K;++k) for(int i=0;i<9;++i){ double v=0; for(int j=0;j<9;++j) v+=Inv[i][j]*a[k][j];
            c.M[((size_t)ci*9+i)*c.max_nb+k]=v*sw[k]; }
    }
    return c;
}

} // namespace cfd
