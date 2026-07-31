// test_mesh2d.cpp — validate cfd::build_structured_2d vs Python mesh.py
// (Nx=3, Ny=2, non-periodic).
#include "cfd/mesh.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace cfd;
static int g_fail = 0;
static void eqi(const char* w, int g, int r) {
    if (g != r) { std::printf("  [FAIL] %-8s got=%d ref=%d\n", w, g, r); ++g_fail; } }
static void eqd(const char* w, double g, double r) {
    double d = std::fabs(r) > 1e-300 ? std::fabs(r) : 1.0;
    if (std::fabs(g - r) / d > 1e-15) { std::printf("  [FAIL] %-8s got=%.17g ref=%.17g\n", w, g, r); ++g_fail; } }

int main() {
    Mesh m = build_structured_2d(3, 2, 1.0, 1.0, false, false);
    eqi("nc", m.n_cells(), 6);
    eqi("nf", m.n_faces(), 17);
    const double cc[6][2] = {{0.16666666666666666,0.25},{0.5,0.25},{0.8333333333333333,0.25},
                             {0.16666666666666666,0.75},{0.5,0.75},{0.8333333333333333,0.75}};
    for (int i = 0; i < 6; ++i) { eqd("cc.x", m.cell_centers[i*2+0], cc[i][0]);
                                  eqd("cc.y", m.cell_centers[i*2+1], cc[i][1]); }
    int fo[17] = {0,0,1,2,3,3,4,5,0,1,2,0,1,2,3,4,5};
    int fn[17] = {-1,1,2,-1,-1,4,5,-1,-1,-1,-1,3,4,5,-1,-1,-1};
    double fnx[17] = {-1,1,1,1,-1,1,1,1,0,0,0,0,0,0,0,0,0};
    double fny[17] = {0,0,0,0,0,0,0,0,-1,-1,-1,1,1,1,1,1,1};
    int bc[17] = {1,0,0,2,1,0,0,2,3,3,3,0,0,0,4,4,4};
    double dy = 0.5, dx = 1.0/3.0;
    for (int f = 0; f < 17; ++f) {
        eqi("fo", m.face_owner[f], fo[f]);
        eqi("fn", m.face_neighbour[f], fn[f]);
        eqd("fnx", m.face_normals[f*2+0], fnx[f]);
        eqd("fny", m.face_normals[f*2+1], fny[f]);
        eqi("bc", m.face_bc_tag[f], bc[f]);
        eqd("fa", m.face_areas[f], f < 8 ? dy : dx);
    }
    if (g_fail == 0) { std::printf("test_mesh2d: ALL PASS\n"); return 0; }
    std::printf("test_mesh2d: %d FAILURES\n", g_fail); return 1;
}
