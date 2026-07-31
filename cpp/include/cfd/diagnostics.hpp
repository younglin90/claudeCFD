// cfd/diagnostics.hpp — post-solution vorticity diagnostics for the Euler
// benchmark gates (Double-Mach slip-line rollups, Mach-3 upper rollups).
// omega = dv/dx - du/dy via the LSQ vertex-stencil gradient (reuse ReconCtx);
// inside an ROI box, count coherent vortices (local |omega| maxima above a
// fraction of the peak), accumulate enstrophy and peak. Sharper / less-diffusive
// reconstruction => more rollups + higher enstrophy.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/euler2d.hpp"
#include "cfd/reconstruct2d.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace cfd {

struct VortDiag { int vortex_count = 0; double enstrophy = 0.0, omega_peak = 0.0; };

inline VortDiag vorticity_roi(const Mesh& m, const Euler2D& eq, const ReconCtx& c,
                              const std::vector<double>& U,
                              double x0, double x1, double y0, double y1,
                              double peak_frac = 0.25) {
    const int N = m.n_cells();
    std::vector<double> u(N), v(N), om(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double Uc[4] = {U[0*N+i],U[1*N+i],U[2*N+i],U[3*N+i]}, W[4];
        eq.cons_to_prim(Uc, W); u[i] = W[1]; v[i] = W[2];
    }
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        double ru0=0,ru1=0,rv0=0,rv1=0;
        for (int k = 0; k < c.max_nb; ++k) {
            int nb = c.nb[(size_t)ci*c.max_nb+k]; if (nb < 0) continue;
            double wk = c.w[(size_t)ci*c.max_nb+k];
            double dx = c.d[((size_t)ci*c.max_nb+k)*2+0], dy = c.d[((size_t)ci*c.max_nb+k)*2+1];
            ru0 += wk*dx*(u[nb]-u[ci]); ru1 += wk*dy*(u[nb]-u[ci]);
            rv0 += wk*dx*(v[nb]-v[ci]); rv1 += wk*dy*(v[nb]-v[ci]);
        }
        double dudy = c.ATA_inv[ci*4+2]*ru0 + c.ATA_inv[ci*4+3]*ru1;
        double dvdx = c.ATA_inv[ci*4+0]*rv0 + c.ATA_inv[ci*4+1]*rv1;
        om[ci] = dvdx - dudy;
    }
    auto inroi = [&](int i){ double x=m.cell_centers[i*2], y=m.cell_centers[i*2+1];
                             return x>=x0 && x<=x1 && y>=y0 && y<=y1; };
    VortDiag d;
    for (int i = 0; i < N; ++i) if (inroi(i)) {
        d.enstrophy += om[i]*om[i]*m.cell_volumes[i];
        d.omega_peak = std::max(d.omega_peak, std::fabs(om[i]));
    }
    double thr = peak_frac * d.omega_peak;
    for (int ci = 0; ci < N; ++ci) {
        if (!inroi(ci) || std::fabs(om[ci]) < thr) continue;
        bool ismax = true;
        for (int k = 0; k < c.max_nb; ++k) { int nb=c.nb[(size_t)ci*c.max_nb+k];
            if (nb>=0 && std::fabs(om[nb]) > std::fabs(om[ci])) { ismax=false; break; } }
        if (ismax) ++d.vortex_count;
    }
    return d;
}

// ── Q-criterion vortex detection (robust; excludes shear/shock). ──
// Q = du/dx*dv/dy - du/dy*dv/dx (2D); Q>0 = rotation-dominated (true vortex).
// Connected Q>thr regions in the ROI are counted as coherent vortices and
// binned by area (small/mid/large) -> detects "various-sized vortices".
// Shock cells (large |grad p|) are excluded so shocks aren't mis-counted.
struct QDiag {
    int n_vortices = 0, n_small = 0, n_mid = 0, n_large = 0;
    int n_genuine = 0, n_stable_genuine = 0;
    int n_active_cells = 0, n_shock_cells = 0, n_contact_rejected = 0;
    double q_integral = 0.0, q_peak = 0.0, lambda_ci_peak = 0.0;
};

inline QDiag q_criterion_roi(const Mesh& m, const Euler2D& eq, const ReconCtx& c,
                             const std::vector<double>& U,
                             double x0, double x1, double y0, double y1,
                             double q_frac = 0.10) {
    const int N = m.n_cells();
    std::vector<double> u(N), v(N), p(N), Q(N, 0.0), lambda2(N, 0.0);
    std::vector<char> contact_ok(N, 1);
    for (int i = 0; i < N; ++i) {
        double Uc[4] = {U[0*N+i],U[1*N+i],U[2*N+i],U[3*N+i]}, W[4];
        eq.cons_to_prim(Uc, W); u[i]=W[1]; v[i]=W[2]; p[i]=W[3];
    }
    auto grad = [&](const std::vector<double>& f, int ci, double& gx, double& gy) {
        double r0=0,r1=0;
        for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
            double wk=c.w[(size_t)ci*c.max_nb+k];
            r0+=wk*c.d[((size_t)ci*c.max_nb+k)*2+0]*(f[nb]-f[ci]);
            r1+=wk*c.d[((size_t)ci*c.max_nb+k)*2+1]*(f[nb]-f[ci]); }
        gx=c.ATA_inv[ci*4+0]*r0+c.ATA_inv[ci*4+1]*r1;
        gy=c.ATA_inv[ci*4+2]*r0+c.ATA_inv[ci*4+3]*r1;
    };
    // slip-line ISOLATION: exclude shock-region. Q_SHOCKFAC = |grad p|/p factor (lower=stricter),
    // Q_SHOCKDIL = dilate shock mask by N cell layers (remove shock-adjacent vortex noise).
    static const double SF = std::getenv("Q_SHOCKFAC") ? std::atof(std::getenv("Q_SHOCKFAC")) : 5.0;
    static const int SD = std::getenv("Q_SHOCKDIL") ? std::atoi(std::getenv("Q_SHOCKDIL")) : 0;
    static const char* QTV = std::getenv("Q_TRUE_VORTEX");
    static const bool TRUE_VORTEX = QTV && std::atof(QTV) != 0.0;
    static const double PJMAX = std::getenv("Q_CONTACT_MAX") ? std::atof(std::getenv("Q_CONTACT_MAX")) : 0.20;
    static const double TSPAN = std::getenv("Q_THRESH_SPAN") ? std::atof(std::getenv("Q_THRESH_SPAN")) : 0.25;
    std::vector<char> shock(N, 0);
    #pragma omp parallel for
    for (int ci=0; ci<N; ++ci) {
        double ux,uy,vx,vy,px,py; grad(u,ci,ux,uy); grad(v,ci,vx,vy); grad(p,ci,px,py);
        Q[ci] = ux*vy - uy*vx;
        double disc = (ux + vy) * (ux + vy) - 4.0 * Q[ci];
        lambda2[ci] = disc < 0.0 ? -0.25 * disc : 0.0;
        double pjump = 0.0;
        for (int nb : m.cell_neighbours[ci]) if (nb >= 0) {
            double rel = std::fabs(p[nb] - p[ci]) / (std::fabs(p[nb]) + std::fabs(p[ci]) + 1e-30);
            if (rel > pjump) pjump = rel;
        }
        contact_ok[ci] = (!TRUE_VORTEX || pjump <= PJMAX) ? 1 : 0;
        double pg = std::sqrt(px*px+py*py);
        if (pg > SF * std::max(p[ci],1.0)) shock[ci] = 1; // strong pressure grad = shock
    }
    for (int it=0; it<SD; ++it) { std::vector<char> sh2=shock;
        for (int ci=0;ci<N;++ci) if(!shock[ci]) for(int nb:m.cell_neighbours[ci]) if(nb>=0&&shock[nb]){sh2[ci]=1;break;}
        shock.swap(sh2); }
    auto inroi=[&](int i){ double x=m.cell_centers[i*2],y=m.cell_centers[i*2+1];
                           return x>=x0&&x<=x1&&y>=y0&&y<=y1; };
    QDiag d;
    double roi_area = 0.0; int roi_cells = 0;
    for (int i=0;i<N;++i) if (inroi(i)) { roi_area += m.cell_volumes[i]; ++roi_cells; }
    const double ref_area = 4.0 * (roi_cells > 0 ? roi_area / roi_cells : 0.0);
    auto active_cell = [&](int i, double thr) {
        if (!inroi(i) || shock[i] || Q[i] <= thr) return false;
        if (TRUE_VORTEX && (lambda2[i] <= 0.0 || !contact_ok[i])) return false;
        return true;
    };
    for (int i=0;i<N;++i) if (inroi(i)) {
        if (shock[i]) ++d.n_shock_cells;
        else if (Q[i] > 0.0 && TRUE_VORTEX && !contact_ok[i]) ++d.n_contact_rejected;
        if (active_cell(i, 0.0)) {
            ++d.n_active_cells;
            d.q_integral += Q[i] * m.cell_volumes[i];
            d.q_peak = std::max(d.q_peak, Q[i]);
            d.lambda_ci_peak = std::max(d.lambda_ci_peak, std::sqrt(lambda2[i]));
        }
    }
    double thr = q_frac * d.q_peak;
    auto count_components = [&](double threshold, int& nvort, int& nsmall, int& nmid, int& nlarge) {
        nvort = nsmall = nmid = nlarge = 0;
        std::vector<char> seen(N,0);
        std::vector<int> stack;
        for (int s=0;s<N;++s){
            if (seen[s] || !active_cell(s, threshold)) continue;
            double area=0; stack.clear(); stack.push_back(s); seen[s]=1;
            while(!stack.empty()){ int ci=stack.back(); stack.pop_back(); area+=m.cell_volumes[ci];
                for(int nb : m.cell_neighbours[ci]) if(nb>=0&&!seen[nb]&&active_cell(nb, threshold)){
                    seen[nb]=1; stack.push_back(nb); } }
            ++nvort;
            if (area < ref_area) ++nsmall; else if (area < 8.0*ref_area) ++nmid; else ++nlarge;
        }
    };
    count_components(thr, d.n_vortices, d.n_small, d.n_mid, d.n_large);
    d.n_genuine = d.n_mid + d.n_large;
    int nv_l=0, ns_l=0, nm_l=0, nl_l=0, nv_h=0, ns_h=0, nm_h=0, nl_h=0;
    count_components(thr * std::max(0.0, 1.0 - TSPAN), nv_l, ns_l, nm_l, nl_l);
    count_components(thr * (1.0 + std::max(0.0, TSPAN)), nv_h, ns_h, nm_h, nl_h);
    d.n_stable_genuine = std::min(d.n_genuine, std::min(nm_l + nl_l, nm_h + nl_h));
    return d;
}

} // namespace cfd
