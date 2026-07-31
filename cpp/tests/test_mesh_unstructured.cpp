// test_mesh_unstructured.cpp — validate cfd::criss_cross_box vs Python mesh.py
// (N=2 Union-Jack triangulation: 16 cells, 28 faces).
#include "cfd/mesh.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
using namespace cfd;
#ifndef CC_REF
#  define CC_REF "mesh_cc_ref.txt"
#endif
static int g_fail = 0;
static void eqi(const char* w, int g, int r) {
    if (g != r) { std::printf("  [FAIL] %-8s got=%d ref=%d\n", w, g, r); ++g_fail; } }
static void eqd(const char* w, double g, double r) {
    double d = std::fabs(r) > 1e-300 ? std::fabs(r) : 1.0;
    if (std::fabs(g - r) / d > 1e-13) { std::printf("  [FAIL] %-8s got=%.17g ref=%.17g\n", w, g, r); ++g_fail; } }

int main() {
    Mesh m = criss_cross_box(2, 1.0);
    std::ifstream fin(CC_REF);
    if (!fin) { std::printf("test_mesh_unstructured: cannot open %s\n", CC_REF); return 2; }
    int nc, nf; fin >> nc >> nf;
    eqi("n_cells", m.n_cells(), nc);
    eqi("n_faces", m.n_faces(), nf);
    for (int i = 0; i < nc; ++i) {
        double cx, cy, vol; fin >> cx >> cy >> vol;
        eqd("cc.x", m.cell_centers[i*2+0], cx);
        eqd("cc.y", m.cell_centers[i*2+1], cy);
        eqd("vol",  m.cell_volumes[i], vol);
    }
    for (int f = 0; f < nf; ++f) {
        int o, n, bc; double nx, ny, fa;
        fin >> o >> n >> nx >> ny >> fa >> bc;
        eqi("fo", m.face_owner[f], o);
        eqi("fn", m.face_neighbour[f], n);
        eqd("fnx", m.face_normals[f*2+0], nx);
        eqd("fny", m.face_normals[f*2+1], ny);
        eqd("fa", m.face_areas[f], fa);
        eqi("bc", m.face_bc_tag[f], bc);
    }
    // sanity: total volume = 1
    double sv = 0; for (double v : m.cell_volumes) sv += v;
    eqd("sumvol", sv, 1.0);
    if (g_fail == 0) { std::printf("test_mesh_unstructured: ALL PASS (criss_cross N=2 matches Python)\n"); return 0; }
    std::printf("test_mesh_unstructured: %d FAILURES\n", g_fail); return 1;
}
