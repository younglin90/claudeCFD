// test_region_gap.cpp — how far apart are the two solids?
//
// The fan decomposition tiles a cell with planar sub-tetrahedra hung off the cell centroid and
// the face centroids; the isoparametric element map describes a solid whose faces are bilinear.
// On a tetrahedron the two coincide. On a distorted hexahedron, prism or pyramid they do not,
// and the gap between them is charged to whichever rule is not the one the reference was built
// from. This measures that gap directly, as a relative volume difference, so it can be compared
// against the reconstruction errors being reported (1e-4 to 1e-3).
#include "cfd/quad_paper3d.hpp"
#include "cfd/mesh_unstructured3d.hpp"
#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>

using namespace cfd;

static const char* NAME[4] = {"tet", "pyramid", "prism", "hex"};
static const int NV[4] = {4, 5, 6, 8};

static void ref_shape(int which, double (*V)[3]) {
    static const double tet[4][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    static const double pyr[5][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0.5,0.5,1}};
    static const double pri[6][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}};
    static const double hex[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                     {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const double (*S)[3] = which==0?tet : which==1?pyr : which==2?pri : hex;
    for (int i = 0; i < NV[which]; ++i) for (int d = 0; d < 3; ++d) V[i][d] = S[i][d];
}

// volume of the solid the fan tiles: sum of |tet(cell centroid, face centroid, va, vb)|
static double fan_volume(int nv, const double (*V)[3]) {
    double c[3] = {0,0,0};
    for (int i = 0; i < nv; ++i) for (int d = 0; d < 3; ++d) c[d] += V[i][d] / nv;
    auto tmpl = u3_face_template(nv);
    double vol = 0;
    for (auto& f : tmpl) {
        double fc[3] = {0,0,0};
        for (int li : f) for (int d = 0; d < 3; ++d) fc[d] += V[li][d] / (double)f.size();
        for (size_t i = 0; i < f.size(); ++i) {
            const double* a = V[f[i]];
            const double* b = V[f[(i+1)%f.size()]];
            double A[3], B[3], C[3];
            for (int d = 0; d < 3; ++d) { A[d]=fc[d]-c[d]; B[d]=a[d]-c[d]; C[d]=b[d]-c[d]; }
            vol += std::fabs(jac_det(A, B, C)) / 6.0;
        }
    }
    return vol;
}

// volume of the isoparametric solid, by dense collapsed tensor quadrature
static double iso_volume(int nv, const double (*V)[3], int n) {
    std::vector<double> gx(n), gw(n);
    for (int i = 0; i < n; ++i) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), z1, pp;
        do { double p1 = 1, p2 = 0;
             for (int j = 0; j < n; ++j) { double p3 = p2; p2 = p1; p1 = ((2*j+1)*z*p2 - j*p3)/(j+1); }
             pp = n * (z*p1 - p2) / (z*z - 1.0); z1 = z; z = z1 - p1/pp;
        } while (std::fabs(z - z1) > 1e-15);
        gx[i] = 0.5*(1.0 - z); gw[i] = 1.0/((1.0 - z*z)*pp*pp);
    }
    double vol = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        double r = gx[i], s = gx[j], t = gx[k], duffy = 1.0;
        if (nv == 4) { r = gx[i]; s = gx[j]*(1-r); t = gx[k]*(1-r)*(1-gx[j]);
                       duffy = (1-gx[i])*(1-gx[i])*(1-gx[j]); }
        else if (nv == 6) { r = gx[i]; s = gx[j]*(1-r); duffy = (1-gx[i]); }
        double x[3], dr[3], ds[3], dt[3];
        map_point(nv, V, r, s, t, x, dr, ds, dt);
        vol += gw[i]*gw[j]*gw[k]*duffy*std::fabs(jac_det(dr, ds, dt));
    }
    return vol;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 2000;
    std::mt19937_64 rng(20260730);
    std::uniform_real_distribution<double> U(0.0, 1.0), S(-1.0, 1.0);

    std::printf("Relative volume gap between the fan solid and the isoparametric solid,\n");
    std::printf("%d randomly distorted cells per shape (same distortion law as the benchmark).\n\n", N);
    std::printf("%-9s %12s %12s %12s\n", "shape", "median", "p90", "max");

    for (int sh = 0; sh < 4; ++sh) {
        const int nv = NV[sh];
        std::vector<double> g;
        for (int c = 0; c < N; ++c) {
            double V[8][3];
            ref_shape(sh, V);
            const double amp = 0.05 + 0.40 * U(rng);
            for (int i = 0; i < nv; ++i) for (int d = 0; d < 3; ++d) V[i][d] += amp * S(rng);
            const double vf = fan_volume(nv, V), vi = iso_volume(nv, V, 24);
            if (vf > 1e-300) g.push_back(std::fabs(vf - vi) / vf);
        }
        std::sort(g.begin(), g.end());
        auto at = [&](double q) { return g[std::min(g.size()-1, (size_t)(q*g.size()))]; };
        std::printf("%-9s %12.3e %12.3e %12.3e\n", NAME[sh], at(0.5), at(0.9), g.back());
    }
    return 0;
}
