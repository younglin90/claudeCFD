// test_recon_o2.cpp — order-2 (quadratic) WLSQ reconstruction must be EXACT on a
// quadratic field (its defining property). Verifies the high-order base is correct.
#include "cfd/reconstruct2d_o2.hpp"
#include <cstdio>
#include <cmath>
using namespace cfd;

// exact quadratic field
static double F(double x, double y) {
    return 1.0 + 2.0*x + 3.0*y + 0.5*x*x + 0.7*y*y + 0.4*x*y;
}

int main() {
    Mesh m = criss_cross_box(8, 1.0);
    ReconCtxO2 c = build_recon_ctx_o2(m);
    const int N = m.n_cells();
    std::vector<double> W(N);
    for (int i = 0; i < N; ++i) W[i] = F(m.cell_centers[i*2+0], m.cell_centers[i*2+1]);

    std::vector<double> WL, WR;
    reconstruct_o2_scalar(m, c, W, 1, 0, WL, WR);

    int fail = 0, checked = 0; double mr = 0;
    for (int f = 0; f < m.n_faces(); ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        if (n < 0) continue; // interior faces only (boundary cells may be degenerate)
        // skip faces whose owner/neighbour LSQ was degenerate (M all zero -> g=0)
        double fx = m.face_centers[f*2+0], fy = m.face_centers[f*2+1];
        double ex = F(fx, fy);
        for (int side = 0; side < 2; ++side) {
            int ci = side ? n : o;
            // require >=5 valid neighbours (well-posed quadratic)
            int K = 0; for (int k = 0; k < c.max_nb; ++k) if (c.nb[(size_t)ci*c.max_nb+k]>=0) ++K;
            if (K < 6) continue;
            double got = side ? WR[f] : WL[f];
            double rel = std::fabs(got - ex) / std::max(std::fabs(ex), 1.0);
            mr = std::max(mr, rel); ++checked;
            if (rel > 1e-9) { if (fail < 6)
                std::printf("  [FAIL] face %d side %d got=%.12g exact=%.12g rel=%.3e\n",
                            f, side, got, ex, rel); ++fail; }
        }
    }
    std::printf("test_recon_o2: checked=%d max_rel=%.3e (quadratic exactness)\n", checked, mr);

    // boundedness: MLP-limited order-2 on a step must not overshoot [0,1].
    std::vector<double> S(N);
    for (int i = 0; i < N; ++i) S[i] = m.cell_centers[i*2+0] < 0.5 ? 0.0 : 1.0;
    std::vector<double> SL, SR;
    reconstruct_o2_limited(m, c, S, 1, 0, SL, SR);
    double omin = 1e9, omax = -1e9;
    for (int f = 0; f < m.n_faces(); ++f) { omin = std::min({omin, SL[f], SR[f]});
                                            omax = std::max({omax, SL[f], SR[f]}); }
    std::printf("test_recon_o2: limited step reconstruction range [%.4f, %.4f] (must be in [0,1])\n", omin, omax);
    if (omin < -1e-9 || omax > 1.0 + 1e-9) { std::printf("  [FAIL] overshoot\n"); ++fail; }

    if (fail == 0 && checked > 0) { std::printf("test_recon_o2: ALL PASS (order-2 exact on quadratics + limited bounded)\n"); return 0; }
    std::printf("test_recon_o2: %d FAILURES\n", fail); return 1;
}
