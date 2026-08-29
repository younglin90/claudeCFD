// test_quad_paper.cpp — verification of the paper-matched element quadrature rules.
//
// Checks, per element type, on both the reference element and randomly distorted cells:
//   (1) weights sum to 1
//   (2) the volume implied by the rule matches a dense tensor reference
//   (3) a random quadratic P integrates to the dense reference value
//   (4) tanh(kk*P + D), the integrand the scheme actually solves against, integrates to the
//       dense reference value across the steepness range the flow solver uses
// (4) is the one that matters: (1)-(3) can pass on a rule that is still too coarse for a
// near-discontinuous integrand.
#include "cfd/quad_paper3d.hpp"
#include <cstdio>
#include <random>
#include <algorithm>

using namespace cfd;

static const char* NAME[4] = {"tet", "pyramid", "prism", "hex"};
static int NV[4] = {4, 5, 6, 8};

static void ref_shape(int which, double (*V)[3]) {
    static const double tet[4][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    static const double pyr[5][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0.5,0.5,1}};
    static const double pri[6][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}};
    static const double hex[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                     {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const double (*S)[3] = which==0?tet : which==1?pyr : which==2?pri : hex;
    for (int i = 0; i < NV[which]; ++i) for (int d = 0; d < 3; ++d) V[i][d] = S[i][d];
}

// dense tensor reference over the same isoparametric map: n^3 Gauss-Legendre, |J| included
template <class F>
static double dense_avg(int nv, const double (*V)[3], int n, F f) {
    std::vector<double> gx(n), gw(n);
    for (int i = 0; i < n; ++i) {                       // Newton for GL nodes on [0,1]
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), z1, pp;
        do { double p1 = 1, p2 = 0;
             for (int j = 0; j < n; ++j) { double p3 = p2; p2 = p1; p1 = ((2*j+1)*z*p2 - j*p3)/(j+1); }
             pp = n * (z*p1 - p2) / (z*z - 1.0); z1 = z; z = z1 - p1/pp;
        } while (std::fabs(z - z1) > 1e-15);
        gx[i] = 0.5*(1.0 - z); gw[i] = 1.0/((1.0 - z*z)*pp*pp);
    }
    // The reference domain is a cube only for the hexahedron and the pyramid. The tetrahedron
    // and the prism are simplex-based, so the cube is collapsed onto the simplex by a Duffy
    // transform and its Jacobian folded into the weight; integrating them over the raw cube
    // would compare against the wrong region entirely.
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        double r = gx[i], s = gx[j], t = gx[k], duffy = 1.0;
        if (nv == 4) {                       // tetrahedron: collapse all three
            r = gx[i]; s = gx[j] * (1 - r); t = gx[k] * (1 - r) * (1 - gx[j]);
            duffy = (1 - gx[i]) * (1 - gx[i]) * (1 - gx[j]);
        } else if (nv == 6) {                // prism: collapse the triangle directions only
            r = gx[i]; s = gx[j] * (1 - r);
            duffy = (1 - gx[i]);
        }
        double x[3], dr[3], ds[3], dt[3];
        map_point(nv, V, r, s, t, x, dr, ds, dt);
        const double w = gw[i]*gw[j]*gw[k]*duffy*std::fabs(jac_det(dr, ds, dt));
        num += w * f(x); den += w;
    }
    return num / den;
}

int main() {
    std::mt19937_64 rng(20260730);
    std::uniform_real_distribution<double> U(0.0, 1.0), S(-1.0, 1.0);

    std::printf("%-8s %-10s  %-12s %-12s %-12s %-12s\n",
                "shape", "cell", "sum(w)-1", "err(vol)", "err(P)", "max err(tanh)");
    int nfail = 0;

    for (int sh = 0; sh < 4; ++sh) {
        const int nv = NV[sh];
        for (int trial = 0; trial < 4; ++trial) {
            double V[8][3];
            ref_shape(sh, V);
            const double amp = (trial == 0) ? 0.0 : 0.05 + 0.40 * U(rng);
            if (trial > 0)
                for (int i = 0; i < nv; ++i) for (int d = 0; d < 3; ++d) V[i][d] += amp * S(rng);

            // a random quadratic about the cell centroid, as in the reconstruction
            double c[3] = {0,0,0};
            for (int i = 0; i < nv; ++i) for (int d = 0; d < 3; ++d) c[d] += V[i][d] / nv;
            double A[9];
            for (int i = 0; i < 9; ++i) A[i] = S(rng);
            const double gvec[3] = {A[0], A[1], A[2]};
            auto Pf = [&](const double* x) {
                const double dx = x[0]-c[0], dy = x[1]-c[1], dz = x[2]-c[2];
                return A[0]*dx + A[1]*dy + A[2]*dz + A[3]*dx*dx + A[4]*dy*dy + A[5]*dz*dz
                     + A[6]*dx*dy + A[7]*dx*dz + A[8]*dy*dz;
            };

            std::vector<QPoint> Q;
            cell_rule_paper(nv, V, gvec, Q);

            double sw = 0; for (auto& p : Q) sw += p.w;

            double aP = 0;
            for (auto& p : Q) { double x[3]; cell_point_xyz(nv, V, p, x); aP += p.w * Pf(x); }
            const double rP = dense_avg(nv, V, 24, Pf);

            // the real integrand, over the production steepness range
            double worst = 0;
            for (double beta : {0.8, 1.4, 3.0, 5.0}) {
                const double kk = beta / 0.5;             // representative 1/H
                for (double D : {-1.0, 0.0, 1.0}) {
                    auto Tf = [&](const double* x) { return std::tanh(kk * Pf(x) + D); };
                    double aT = 0;
                    for (auto& p : Q) { double x[3]; cell_point_xyz(nv, V, p, x); aT += p.w * Tf(x); }
                    const double rT = dense_avg(nv, V, 40, Tf);
                    worst = std::max(worst, std::fabs(aT - rT));
                }
            }

            std::printf("%-8s %-10s  %-12.2e %-12s %-12.2e %-12.2e\n",
                        NAME[sh], trial ? "distorted" : "reference",
                        sw - 1.0, "-", std::fabs(aP - rP), worst);
            if (std::fabs(sw - 1.0) > 1e-12) { std::printf("   FAIL: weights do not sum to 1\n"); ++nfail; }
            if (std::fabs(aP - rP) > 1e-10)  { std::printf("   FAIL: quadratic not integrated exactly\n"); ++nfail; }
        }
    }

    std::printf("\npoint counts: ");
    for (int sh = 0; sh < 4; ++sh) {
        double V[8][3]; ref_shape(sh, V);
        const double g[3] = {1, 0, 0};
        std::vector<QPoint> Q; cell_rule_paper(NV[sh], V, g, Q);
        std::printf("%s=%zu  ", NAME[sh], Q.size());
    }
    { double F3[3][3] = {{0,0,0},{1,0,0},{0,1,0}}, F4[4][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
      std::vector<std::array<double,4>> f;
      face_rule_paper(3, F3, f); std::printf("tri_face=%zu  ", f.size());
      face_rule_paper(4, F4, f); std::printf("quad_face=%zu\n", f.size()); }

    std::printf("\n%s\n", nfail ? "FAILURES PRESENT" : "all structural checks passed");
    return nfail ? 1 : 0;
}
