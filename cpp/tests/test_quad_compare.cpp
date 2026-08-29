// test_quad_compare.cpp — the decisive comparison: the paper-matched element rules against the
// production fan rule now in use, on the SAME cells, the SAME quadratic fields, and the SAME
// integrand, with a dense evaluation as truth.
//
// The question this answers is not "is the 11-point rule a valid quadrature" (test_quad_paper
// settles that) but "does dropping from 96 points to 11 on a tetrahedron cost accuracy on
// tanh(kk*P + D), which is what the scheme actually integrates".
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

// The production fan: for every face, for every face edge, a sub-tetrahedron
// (cell centroid, face centroid, edge start, edge end) carrying NQ^3 Duffy-Gauss points.
// Reproduced here from u3_cell_pvals so the comparison needs no Mesh object.
static void fan_rule(int nv, const double (*V)[3], int NQ,
                     std::vector<std::array<double,4>>& out) {
    out.clear();
    const GL1D& gl = gl1d(NQ);
    std::vector<double> gx(NQ), gw(NQ);
    for (int i = 0; i < NQ; ++i) { gx[i] = gl.x[i]; gw[i] = gl.w[i]; }
    double c[3] = {0,0,0};
    for (int i = 0; i < nv; ++i) for (int d = 0; d < 3; ++d) c[d] += V[i][d] / nv;
    auto tmpl = u3_face_template(nv);
    double tot = 0;
    for (auto& f : tmpl) {
        double fc[3] = {0,0,0};
        for (int li : f) for (int d = 0; d < 3; ++d) fc[d] += V[li][d] / (double)f.size();
        for (size_t i = 0; i < f.size(); ++i) {
            const double* a = V[f[i]];
            const double* b = V[f[(i+1)%f.size()]];
            double A[3], B[3], C[3];
            for (int d = 0; d < 3; ++d) { A[d] = fc[d]-c[d]; B[d] = a[d]-c[d]; C[d] = b[d]-c[d]; }
            const double vol = std::fabs(jac_det(A, B, C)) / 6.0;
            for (int i1 = 0; i1 < NQ; ++i1) for (int j1 = 0; j1 < NQ; ++j1) for (int k1 = 0; k1 < NQ; ++k1) {
                const double xi = gx[i1], eta = gx[j1], ze = gx[k1];
                const double L1 = xi, L2 = (1-xi)*eta, L3 = (1-xi)*(1-eta)*ze;
                double x[3];
                for (int d = 0; d < 3; ++d) x[d] = c[d] + L1*A[d] + L2*B[d] + L3*C[d];
                const double w = gw[i1]*gw[j1]*gw[k1]*(1-xi)*(1-xi)*(1-eta)*6.0*vol;
                out.push_back({x[0], x[1], x[2], w});
                tot += w;
            }
        }
    }
    if (tot > 1e-300) for (auto& p : out) p[3] /= tot;
}

template <class F>
static double dense_avg(int nv, const double (*V)[3], int n, F f) {
    std::vector<double> gx(n), gw(n);
    for (int i = 0; i < n; ++i) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), z1, pp;
        do { double p1 = 1, p2 = 0;
             for (int j = 0; j < n; ++j) { double p3 = p2; p2 = p1; p1 = ((2*j+1)*z*p2 - j*p3)/(j+1); }
             pp = n * (z*p1 - p2) / (z*z - 1.0); z1 = z; z = z1 - p1/pp;
        } while (std::fabs(z - z1) > 1e-15);
        gx[i] = 0.5*(1.0 - z); gw[i] = 1.0/((1.0 - z*z)*pp*pp);
    }
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        double r = gx[i], s = gx[j], t = gx[k], duffy = 1.0;
        if (nv == 4) { r = gx[i]; s = gx[j]*(1-r); t = gx[k]*(1-r)*(1-gx[j]);
                       duffy = (1-gx[i])*(1-gx[i])*(1-gx[j]); }
        else if (nv == 6) { r = gx[i]; s = gx[j]*(1-r); duffy = (1-gx[i]); }
        double x[3], dr[3], ds[3], dt[3];
        map_point(nv, V, r, s, t, x, dr, ds, dt);
        const double w = gw[i]*gw[j]*gw[k]*duffy*std::fabs(jac_det(dr, ds, dt));
        num += w * f(x); den += w;
    }
    return num / den;
}

int main(int argc, char** argv) {
    const int NCELL = argc > 1 ? std::atoi(argv[1]) : 200;
    std::mt19937_64 rng(20260730);
    std::uniform_real_distribution<double> U(0.0, 1.0), S(-1.0, 1.0);
    const double BETAS[4] = {0.8, 1.4, 3.0, 5.0};

    std::printf("Cell-average error of tanh(kk*P + D) against a dense reference.\n");
    std::printf("paper = element rule at the published point count, fan = the production rule.\n");
    std::printf("%d random distorted cells per shape, D swept over {-1,0,1}.\n\n", NCELL);
    std::printf("%-8s %-6s %6s %6s   %-11s %-11s   %-11s %-11s  %s\n",
                "shape", "beta", "n_pap", "n_fan",
                "paper med", "paper p99", "fan med", "fan p99", "ratio med");

    for (int sh = 0; sh < 4; ++sh) {
        const int nv = NV[sh];
        for (int bi = 0; bi < 4; ++bi) {
            const double beta = BETAS[bi];
            std::vector<double> ep, ef;
            int npap = 0, nfan = 0;
            for (int ci = 0; ci < NCELL; ++ci) {
                double V[8][3];
                ref_shape(sh, V);
                const double amp = 0.05 + 0.40 * U(rng);
                for (int i = 0; i < nv; ++i) for (int d = 0; d < 3; ++d) V[i][d] += amp * S(rng);
                double c[3] = {0,0,0};
                for (int i = 0; i < nv; ++i) for (int d = 0; d < 3; ++d) c[d] += V[i][d] / nv;
                double H = 0;                                   // characteristic length
                for (int i = 0; i < nv; ++i) {
                    double d2 = 0; for (int d = 0; d < 3; ++d) { const double q = V[i][d]-c[d]; d2 += q*q; }
                    H = std::max(H, std::sqrt(d2));
                }
                double A[9]; for (int i = 0; i < 9; ++i) A[i] = S(rng);
                const double g[3] = {A[0], A[1], A[2]};
                auto Pf = [&](const double* x) {
                    const double dx = x[0]-c[0], dy = x[1]-c[1], dz = x[2]-c[2];
                    return A[0]*dx + A[1]*dy + A[2]*dz + A[3]*dx*dx + A[4]*dy*dy + A[5]*dz*dz
                         + A[6]*dx*dy + A[7]*dx*dz + A[8]*dy*dz;
                };
                const double kk = beta / H;

                std::vector<QPoint> QP; cell_rule_paper(nv, V, g, QP);
                std::vector<std::array<double,4>> QF; fan_rule(nv, V, 2, QF);
                npap = (int)QP.size(); nfan = (int)QF.size();

                for (double D : {-1.0, 0.0, 1.0}) {
                    auto Tf = [&](const double* x) { return std::tanh(kk * Pf(x) + D); };
                    double ap = 0;
                    for (auto& p : QP) { double x[3]; cell_point_xyz(nv, V, p, x); ap += p.w * Tf(x); }
                    double af = 0;
                    for (auto& p : QF) { const double x[3] = {p[0], p[1], p[2]}; af += p[3] * Tf(x); }
                    const double tr = dense_avg(nv, V, 40, Tf);
                    ep.push_back(std::fabs(ap - tr));
                    ef.push_back(std::fabs(af - tr));
                }
            }
            auto pick = [](std::vector<double>& v, double q) {
                std::sort(v.begin(), v.end());
                return v[std::min(v.size()-1, (size_t)(q * v.size()))];
            };
            const double pm = pick(ep, 0.50), pp = pick(ep, 0.99);
            const double fm = pick(ef, 0.50), fp = pick(ef, 0.99);
            std::printf("%-8s %-6.1f %6d %6d   %-11.3e %-11.3e   %-11.3e %-11.3e  %.2f\n",
                        NAME[sh], beta, npap, nfan, pm, pp, fm, fp,
                        fm > 0 ? pm / fm : 0.0);
        }
    }
    return 0;
}
