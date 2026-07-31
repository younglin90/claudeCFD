// test_mesh.cpp — validate cfd::build_structured_1d against Python
// solver/solve_T-MLP-u/mesh.py (N=5, L=1.0; periodic and non-periodic).
#include "cfd/mesh.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace cfd;
static int g_fail = 0;

static void eqi(const char* w, int got, int ref) {
    if (got != ref) { std::printf("  [FAIL] %-10s got=%d ref=%d\n", w, got, ref); ++g_fail; }
}
static void eqd(const char* w, double got, double ref) {
    double d = std::fabs(ref) > 1e-300 ? std::fabs(ref) : 1.0;
    if (std::fabs(got - ref) / d > 1e-15) {
        std::printf("  [FAIL] %-10s got=%.17g ref=%.17g\n", w, got, ref); ++g_fail;
    }
}
template <class T>
static void eqveci(const char* w, const std::vector<T>& g, std::vector<int> r) {
    eqi(w, (int)g.size(), (int)r.size());
    for (size_t i = 0; i < r.size() && i < g.size(); ++i) eqi(w, (int)g[i], r[i]);
}
static void eqvecd(const char* w, const std::vector<double>& g, std::vector<double> r) {
    eqi(w, (int)g.size(), (int)r.size());
    for (size_t i = 0; i < r.size() && i < g.size(); ++i) eqd(w, g[i], r[i]);
}

int main() {
    // ── non-periodic ──
    Mesh m = build_structured_1d(5, 1.0, false);
    eqi("np.ncells", m.n_cells(), 5);
    eqi("np.nfaces", m.n_faces(), 6);
    eqvecd("np.cc", m.cell_centers, {0.1, 0.30000000000000004, 0.5, 0.7000000000000001, 0.9});
    eqvecd("np.cv", m.cell_volumes, {0.2, 0.2, 0.2, 0.2, 0.2});
    eqveci("np.fo", m.face_owner, {0, 0, 1, 2, 3, 4});
    eqveci("np.fn", m.face_neighbour, {-1, 1, 2, 3, 4, -1});
    eqvecd("np.fnorm", m.face_normals, {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    eqvecd("np.fc", m.face_centers, {0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0});
    eqveci("np.bc", m.face_bc_tag, {1, 0, 0, 0, 0, 2});
    int cnbr_np[5][2] = {{-1,1},{0,2},{1,3},{2,4},{3,-1}};
    for (int c = 0; c < 5; ++c) eqveci("np.cnbr", m.cell_neighbours[c],
                                       {cnbr_np[c][0], cnbr_np[c][1]});

    // ── periodic ──
    Mesh p = build_structured_1d(5, 1.0, true);
    eqi("p.nfaces", p.n_faces(), 5);
    eqveci("p.fo", p.face_owner, {4, 0, 1, 2, 3});
    eqveci("p.fn", p.face_neighbour, {0, 1, 2, 3, 4});
    eqvecd("p.fnorm", p.face_normals, {1.0, 1.0, 1.0, 1.0, 1.0});
    eqveci("p.bc", p.face_bc_tag, {0, 0, 0, 0, 0});
    int cnbr_p[5][2] = {{4,1},{0,2},{1,3},{2,4},{0,3}};
    for (int c = 0; c < 5; ++c) eqveci("p.cnbr", p.cell_neighbours[c],
                                       {cnbr_p[c][0], cnbr_p[c][1]});

    if (g_fail == 0) { std::printf("test_mesh: ALL PASS\n"); return 0; }
    std::printf("test_mesh: %d FAILURES\n", g_fail);
    return 1;
}
