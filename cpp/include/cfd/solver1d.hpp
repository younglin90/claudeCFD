// cfd/solver1d.hpp — 1D finite-volume time march for Euler1D.
//
// Port of the rhs / time loop in solver/solve_T-MLP-u/solver.py for the 1D
// first-order path: midpoint face quadrature, LLF flux, transmissive BCs,
// forward-Euler or SSP-RK2 in time. State layout U[v*N + i] (var-major, matches
// the Python (nvar, N) arrays). Validated against the Python solver via dt_fixed.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/euler1d.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace cfd {

struct Solve1DResult {
    std::vector<double> U;  // 3*N var-major
    int n_steps = 0;
    double t = 0.0;
};

// Symmetric two-argument minmod (limiters.minmod2): sign-agreement, smaller mag.
CFD_ROUTINE_SEQ inline double minmod2(double a, double b) {
    double sa = (a > 0) - (a < 0), sb = (b > 0) - (b < 0);
    double am = a < 0 ? -a : a, bm = b < 0 ? -b : b;
    return 0.5 * (sa + sb) * (am < bm ? am : bm);
}

// rhs: dUdt = -(1/V) sum_f F_f*area  (scatter to owner/neighbour). Transmissive BC.
// recon: 0 = first-order, 1 = minmod MUSCL (MinmodTVD1D).
inline void euler1d_rhs(const Mesh& m, const Euler1D& eq,
                        const std::vector<double>& U, std::vector<double>& dUdt,
                        int recon = 0) {
    const int N = m.n_cells();
    const int Nf = m.n_faces();
    std::vector<double> W(3 * N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double u[3] = {U[0 * N + i], U[1 * N + i], U[2 * N + i]};
        double w[3]; eq.cons_to_prim(u, w);
        W[0 * N + i] = w[0]; W[1 * N + i] = w[1]; W[2 * N + i] = w[2];
    }
    // Per-cell TVD-limited slopes (minmod) — zero at boundary/degenerate cells.
    std::vector<double> slope(recon == 1 ? 3 * N : 0, 0.0);
    if (recon == 1) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            const auto& nb = m.cell_neighbours[i];
            int left = -1, right = -1; double xl = 0, xr = 0; int nv = 0;
            for (int n : nb) {
                if (n < 0) continue;
                double x = m.cell_centers[n];
                if (nv == 0) { left = right = n; xl = xr = x; }
                else { if (x < xl) { left = n; xl = x; } if (x > xr) { right = n; xr = x; } }
                ++nv;
            }
            if (nv < 2 || left == right) continue;
            for (int v = 0; v < 3; ++v) {
                double dL = W[v * N + i] - W[v * N + left];
                double dR = W[v * N + right] - W[v * N + i];
                slope[v * N + i] = minmod2(dL, dR);
            }
        }
    }
    std::fill(dUdt.begin(), dUdt.end(), 0.0);
    // Serial face accumulation (scatter has owner/neighbour write races).
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double nf = m.face_normals[f];
        double area = m.face_areas[f];
        double WL[3], WR[3];
        for (int v = 0; v < 3; ++v) {
            double so = recon == 1 ? 0.5 * nf * slope[v * N + o] : 0.0;
            WL[v] = W[v * N + o] + so;
            if (n >= 0) WR[v] = W[v * N + n] - (recon == 1 ? 0.5 * nf * slope[v * N + n] : 0.0);
            else        WR[v] = WL[v]; // transmissive (post-limited owner value)
        }
        double F[3];
        llf_euler1d(eq, WL, WR, nf, F);
        for (int v = 0; v < 3; ++v) {
            dUdt[v * N + o] -= F[v] * area;
            if (n >= 0) dUdt[v * N + n] += F[v] * area;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double inv = 1.0 / m.cell_volumes[i];
        dUdt[0 * N + i] *= inv; dUdt[1 * N + i] *= inv; dUdt[2 * N + i] *= inv;
    }
}

inline double euler1d_global_max_wave(const Mesh& m, const Euler1D& eq,
                                      const std::vector<double>& U) {
    const int N = m.n_cells();
    double wmax = 0.0;
    #pragma omp parallel for reduction(max:wmax)
    for (int i = 0; i < N; ++i) {
        double u[3] = {U[0 * N + i], U[1 * N + i], U[2 * N + i]};
        double w = eq.max_wave_speed(u);
        if (w > wmax) wmax = w;
    }
    return wmax;
}

// integrator: 0 = forward_euler, 1 = ssp_rk2
inline Solve1DResult solve_euler1d(const Mesh& m, const Euler1D& eq,
                                   const std::vector<double>& U0,
                                   double t_end, int integrator = 1,
                                   double cfl = 0.4, double dt_fixed = -1.0,
                                   int max_steps = 200000, int recon = 0) {
    const int N = m.n_cells();
    std::vector<double> U = U0;
    std::vector<double> k1(3 * N), Utmp(3 * N), k2(3 * N);
    double h_min = *std::min_element(m.cell_volumes.begin(), m.cell_volumes.end());
    double t = 0.0; int n = 0;
    for (; n < max_steps && t < t_end; ++n) {
        double dt;
        if (dt_fixed > 0.0) dt = dt_fixed;
        else {
            double wmax = euler1d_global_max_wave(m, eq, U);
            dt = cfl * h_min / wmax;
        }
        if (t + dt > t_end) dt = t_end - t;
        if (dt <= 0.0) break;

        euler1d_rhs(m, eq, U, k1, recon);
        if (integrator == 0) {
            #pragma omp parallel for
            for (int i = 0; i < 3 * N; ++i) U[i] += dt * k1[i];
        } else { // ssp_rk2
            #pragma omp parallel for
            for (int i = 0; i < 3 * N; ++i) Utmp[i] = U[i] + dt * k1[i];
            euler1d_rhs(m, eq, Utmp, k2, recon);
            #pragma omp parallel for
            for (int i = 0; i < 3 * N; ++i)
                U[i] = 0.5 * U[i] + 0.5 * (Utmp[i] + dt * k2[i]);
        }
        t += dt;
    }
    return Solve1DResult{std::move(U), n, t};
}

} // namespace cfd
