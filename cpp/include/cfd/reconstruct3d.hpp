// cfd/reconstruct3d.hpp — minimal 3D limited-linear (MUSCL) reconstruction.
//
// Face-neighbour stencil (the cells across each face): per cell a weighted (w=1)
// least-squares gradient g=(gx,gy,gz)=(A^T A)^{-1} (Σ w d (W_nb - W_c)), a
// Barth-Jespersen limiter phi against the stencil neighbour min/max, and
// extrapolation of the owner/neighbour cell value to each face centre. This is
// the Stage-1 MUSCL baseline (the 3D analogue of reconstruct_bj_vertex in
// reconstruct2d.hpp, but on the face stencil rather than the vertex 1-ring).
//
// Boundary faces fall back to first-order (owner value), exactly as the 2D path.
#pragma once
#include "cfd/mesh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace cfd {

struct ReconCtx3D {
    int N = 0, max_nb = 0;
    std::vector<int>    nb;        // N*max_nb (-1 pad): face-neighbour cell ids
    std::vector<double> d;         // N*max_nb*3 (offsets, zero where invalid)
    std::vector<double> ATA_inv;   // N*9 (row-major 3x3 inverse of A^T A)
};

// Build the face-neighbour LSQ reconstruction context. Stencil = cells across
// each face (m.cell_neighbours[ci], skip -1). A^T A is the 3x3 symmetric normal
// matrix Σ d d^T (weight 1); its inverse is stored row-major (guard det~0 -> 0).
inline ReconCtx3D build_recon_ctx_3d(const Mesh& m) {
    ReconCtx3D c;
    const int N = m.n_cells();
    c.N = N;
    const double* cc = m.cell_centers.data();
    for (int ci = 0; ci < N; ++ci)
        c.max_nb = std::max(c.max_nb, (int)m.cell_neighbours[ci].size());
    c.max_nb = std::max(c.max_nb, 1);
    c.nb.assign((size_t)N * c.max_nb, -1);
    c.d.assign((size_t)N * c.max_nb * 3, 0.0);
    c.ATA_inv.assign((size_t)N * 9, 0.0);

    // ── Minimum-image periodic wrap. On a periodic structured mesh the raw centre
    // difference to a wrapped neighbour is the long way round (e.g. +3 dy for a
    // 4-cell column) instead of the adjacent −dy. That makes the LSQ normal matrix
    // (and its determinant) differ between seam and interior rows by round-off,
    // which seeds a transverse instability that contact-resolving fluxes amplify.
    // Per axis: domain length L = (max−min centre) + h, h = smallest positive
    // |offset| on that axis; wrap any offset with |d| > L/2 into [−L/2, L/2].
    // No-op for non-periodic meshes (offsets never exceed half the domain).
    double Lwrap[3] = {0,0,0};
    {
        double cmin[3] = {1e300,1e300,1e300}, cmax[3] = {-1e300,-1e300,-1e300};
        double hmin[3] = {1e300,1e300,1e300};
        for (int ci = 0; ci < N; ++ci)
            for (int a = 0; a < 3; ++a) {
                double v = cc[(size_t)ci * 3 + a];
                if (v < cmin[a]) cmin[a] = v; if (v > cmax[a]) cmax[a] = v;
            }
        for (int ci = 0; ci < N; ++ci)
            for (int nb : m.cell_neighbours[ci]) {
                if (nb < 0) continue;
                for (int a = 0; a < 3; ++a) {
                    double dd = std::fabs(cc[(size_t)nb * 3 + a] - cc[(size_t)ci * 3 + a]);
                    if (dd > 1e-12 && dd < hmin[a]) hmin[a] = dd;
                }
            }
        for (int a = 0; a < 3; ++a)
            Lwrap[a] = (hmin[a] < 1e299) ? (cmax[a] - cmin[a] + hmin[a]) : 0.0;
    }
    // hmin[a] doubles as the canonical axis spacing for snapping (below).
    double hcan[3] = {0,0,0};
    {
        double hmin2[3] = {1e300,1e300,1e300};
        for (int ci = 0; ci < N; ++ci)
            for (int nb : m.cell_neighbours[ci]) {
                if (nb < 0) continue;
                for (int a = 0; a < 3; ++a) {
                    double dd = std::fabs(cc[(size_t)nb * 3 + a] - cc[(size_t)ci * 3 + a]);
                    if (dd > 1e-12 && dd < hmin2[a]) hmin2[a] = dd;
                }
            }
        for (int a = 0; a < 3; ++a) hcan[a] = (hmin2[a] < 1e299) ? hmin2[a] : 0.0;
    }
    // wrap (minimum image) then snap to the axis grid (k*h). On a periodic mesh
    // the arithmetic wrap d-L is NOT bit-identical to the native −h (round-off),
    // which leaves a ULP-level asymmetry in the LSQ matrix between seam and
    // interior cells. Snapping to round(d/h)*h makes every uniform-mesh offset the
    // SAME double regardless of how it was reached -> bit-identical stencils ->
    // no round-off seed for transverse instabilities. A no-op (k*h reproduces d)
    // on a uniform mesh in the non-wrapped directions.
    auto wrap_snap = [](double d, double L, double h) {
        if (L > 0.0) { if (d > 0.5 * L) d -= L; else if (d < -0.5 * L) d += L; }
        if (h > 0.0) { double k = std::round(d / h); if (std::fabs(k) < 1e15) d = k * h; }
        return d;
    };

    for (int ci = 0; ci < N; ++ci) {
        const auto& nbl = m.cell_neighbours[ci];
        double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
        for (int k = 0; k < (int)nbl.size(); ++k) {
            int nb = nbl[k];
            c.nb[(size_t)ci * c.max_nb + k] = nb;
            if (nb < 0) continue;
            double dx = wrap_snap(cc[(size_t)nb * 3 + 0] - cc[(size_t)ci * 3 + 0], Lwrap[0], hcan[0]);
            double dy = wrap_snap(cc[(size_t)nb * 3 + 1] - cc[(size_t)ci * 3 + 1], Lwrap[1], hcan[1]);
            double dz = wrap_snap(cc[(size_t)nb * 3 + 2] - cc[(size_t)ci * 3 + 2], Lwrap[2], hcan[2]);
            c.d[((size_t)ci * c.max_nb + k) * 3 + 0] = dx;
            c.d[((size_t)ci * c.max_nb + k) * 3 + 1] = dy;
            c.d[((size_t)ci * c.max_nb + k) * 3 + 2] = dz;
            a00 += dx*dx; a01 += dx*dy; a02 += dx*dz;
            a11 += dy*dy; a12 += dy*dz; a22 += dz*dz;
        }
        // 3x3 symmetric inverse via cofactors.
        double c00 = a11*a22 - a12*a12;
        double c01 = a12*a02 - a01*a22;   // = -(a01*a22 - a02*a12)
        double c02 = a01*a12 - a11*a02;
        double det = a00*c00 + a01*c01 + a02*c02;
        if (std::fabs(det) > 1e-30) {
            double inv = 1.0 / det;
            double c11 = a00*a22 - a02*a02;
            double c12 = a02*a01 - a00*a12;   // = -(a00*a12 - a01*a02)
            double c22 = a00*a11 - a01*a01;
            double* M = &c.ATA_inv[(size_t)ci * 9];
            M[0] = c00*inv; M[1] = c01*inv; M[2] = c02*inv;
            M[3] = c01*inv; M[4] = c11*inv; M[5] = c12*inv;
            M[6] = c02*inv; M[7] = c12*inv; M[8] = c22*inv;
        }
    }
    return c;
}

// BJ admissible-ratio limiter (mirror of bj_phi in reconstruct2d.hpp:103).
inline double bj_phi_3d(double delta, double center, double lo, double hi) {
    const double eps = 1e-30;
    double allowed = delta >= 0.0 ? (hi - center) : (center - lo);
    double phi = (std::fabs(delta) > eps)
               ? std::max(allowed, 0.0) / std::max(std::fabs(delta), eps) : 1.0;
    return phi < 0.0 ? 0.0 : (phi > 1.0 ? 1.0 : phi);
}

// Limited-linear MUSCL reconstruction. Per cell, per variable: LSQ gradient,
// BJ limiter phi vs the stencil neighbour min/max, then extrapolate the cell
// value to each incident face centre. W_L holds the owner-side state at the
// face, W_R the neighbour-side; both sized nvar*Nf, initialised first-order,
// higher-order filled for interior faces. Boundary faces keep the owner value.
inline void reconstruct_bj_vertex_3d(const Mesh& m, const ReconCtx3D& c,
                                     const std::vector<double>& W, int nvar,
                                     std::vector<double>& W_L, std::vector<double>& W_R) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const double* cc = m.cell_centers.data();
    // grad[(v*N+ci)*3 + {0,1,2}], phi[v*N+ci]
    std::vector<double> grad((size_t)nvar * N * 3, 0.0);
    std::vector<double> phi((size_t)nvar * N, 1.0);
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        const double* M = &c.ATA_inv[(size_t)ci * 9];
        for (int v = 0; v < nvar; ++v) {
            double wc = W[(size_t)v * N + ci];
            double r0 = 0, r1 = 0, r2 = 0, mn = wc, mx = wc;
            for (int k = 0; k < c.max_nb; ++k) {
                int nb = c.nb[(size_t)ci * c.max_nb + k];
                if (nb < 0) continue;
                double dphi = W[(size_t)v * N + nb] - wc;
                double dx = c.d[((size_t)ci * c.max_nb + k) * 3 + 0];
                double dy = c.d[((size_t)ci * c.max_nb + k) * 3 + 1];
                double dz = c.d[((size_t)ci * c.max_nb + k) * 3 + 2];
                r0 += dx * dphi; r1 += dy * dphi; r2 += dz * dphi;
                double wn = W[(size_t)v * N + nb];
                if (wn < mn) mn = wn; if (wn > mx) mx = wn;
            }
            double gx = M[0]*r0 + M[1]*r1 + M[2]*r2;
            double gy = M[3]*r0 + M[4]*r1 + M[5]*r2;
            double gz = M[6]*r0 + M[7]*r1 + M[8]*r2;
            grad[((size_t)v * N + ci) * 3 + 0] = gx;
            grad[((size_t)v * N + ci) * 3 + 1] = gy;
            grad[((size_t)v * N + ci) * 3 + 2] = gz;
            // BJ limiter: minimum admissible ratio over the face midpoints (the
            // reconstruction sample points for this MUSCL stencil).
            double p = 1.0;
            for (int k = 0; k < c.max_nb; ++k) {
                int nb = c.nb[(size_t)ci * c.max_nb + k];
                if (nb < 0) continue;
                double dx = c.d[((size_t)ci * c.max_nb + k) * 3 + 0];
                double dy = c.d[((size_t)ci * c.max_nb + k) * 3 + 1];
                double dz = c.d[((size_t)ci * c.max_nb + k) * 3 + 2];
                // sample at the face between ci and nb ~ midpoint of the cell
                // centres -> projection 0.5*g.d (consistent with the 2D vertex
                // sampling; bounds the face-extrapolated value).
                double proj = 0.5 * (gx*dx + gy*dy + gz*dz);
                double pk = bj_phi_3d(proj, wc, mn, mx);
                if (pk < p) p = pk;
            }
            phi[(size_t)v * N + ci] = p;
        }
    }
    W_L.assign((size_t)nvar * Nf, 0.0);
    W_R.assign((size_t)nvar * Nf, 0.0);
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double fx = m.face_centers[(size_t)f * 3 + 0];
        double fy = m.face_centers[(size_t)f * 3 + 1];
        double fz = m.face_centers[(size_t)f * 3 + 2];
        double dofx = fx - cc[(size_t)o * 3 + 0];
        double dofy = fy - cc[(size_t)o * 3 + 1];
        double dofz = fz - cc[(size_t)o * 3 + 2];
        for (int v = 0; v < nvar; ++v) {
            double wo = W[(size_t)v * N + o];
            if (n < 0) { W_L[(size_t)v * Nf + f] = wo; W_R[(size_t)v * Nf + f] = wo; continue; }
            double go0 = grad[((size_t)v * N + o) * 3 + 0];
            double go1 = grad[((size_t)v * N + o) * 3 + 1];
            double go2 = grad[((size_t)v * N + o) * 3 + 2];
            double po = phi[(size_t)v * N + o];
            W_L[(size_t)v * Nf + f] = wo + po * (go0*dofx + go1*dofy + go2*dofz);
            double wn = W[(size_t)v * N + n];
            double gn0 = grad[((size_t)v * N + n) * 3 + 0];
            double gn1 = grad[((size_t)v * N + n) * 3 + 1];
            double gn2 = grad[((size_t)v * N + n) * 3 + 2];
            double pn = phi[(size_t)v * N + n];
            double dnfx = fx - cc[(size_t)n * 3 + 0];
            double dnfy = fy - cc[(size_t)n * 3 + 1];
            double dnfz = fz - cc[(size_t)n * 3 + 2];
            W_R[(size_t)v * Nf + f] = wn + pn * (gn0*dnfx + gn1*dnfy + gn2*dnfz);
        }
    }
}

} // namespace cfd
