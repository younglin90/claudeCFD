// test_recon2d.cpp — validate cfd::reconstruct_bj_vertex (T-MLP-u-L core) vs
// Python MLPU1 (BJ+vertex+vertex_bounds) and MLPU1TMLPU (+ face LMP bound) on
// the criss-cross mesh.
#include "cfd/reconstruct2d.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
using namespace cfd;
#ifndef RECON_MLPU1_REF
#  define RECON_MLPU1_REF "recon_mlpu1_ref.txt"
#endif
#ifndef RECON_TMLPU_REF
#  define RECON_TMLPU_REF "recon_mlpu1tmlpu_ref.txt"
#endif

static int check(const char* tag, const Mesh& m, const ReconCtx& c,
                 const std::vector<double>& W, bool face_bound, const char* ref) {
    std::vector<double> WL, WR;
    reconstruct_bj_vertex(m, c, W, 1, WL, WR, face_bound);
    std::ifstream fin(ref);
    if (!fin) { std::printf("  [%s] cannot open %s\n", tag, ref); return 1; }
    int nf; fin >> nf;
    int fail = 0; double mr = 0;
    for (int f = 0; f < nf; ++f) {
        double rl, rr; fin >> rl >> rr;
        double dl = std::fabs(rl) > 1e-12 ? std::fabs(rl) : 1.0;
        double dr = std::fabs(rr) > 1e-12 ? std::fabs(rr) : 1.0;
        double el = std::fabs(WL[f]-rl)/dl, er = std::fabs(WR[f]-rr)/dr;
        mr = std::max(mr, std::max(el, er));
        if (el > 1e-12 || er > 1e-12) { if (fail < 6)
            std::printf("  [%s FAIL] face %d WL %.17g/%.17g WR %.17g/%.17g\n",
                        tag, f, WL[f], rl, WR[f], rr); ++fail; }
    }
    std::printf("  [%s] max_rel=%.3e %s\n", tag, mr, fail ? "FAIL" : "PASS");
    return fail;
}

int main() {
    Mesh m = criss_cross_box(4, 1.0);
    ReconCtx c = build_recon_ctx(m);
    const int N = m.n_cells();
    std::vector<double> W(N);
    for (int i = 0; i < N; ++i) {
        double cx = m.cell_centers[i*2+0], cy = m.cell_centers[i*2+1];
        W[i] = std::cos(2*M_PI*cx) * std::cos(2*M_PI*cy);
    }
    int fail = 0;
    fail += check("mlp_u1", m, c, W, false, RECON_MLPU1_REF);
    fail += check("mlp_u1_tmlpu", m, c, W, true, RECON_TMLPU_REF);
    if (fail == 0) { std::printf("test_recon2d: ALL PASS (T-MLP-u-L core matches Python)\n"); return 0; }
    std::printf("test_recon2d: %d FAILURES\n", fail); return 1;
}
