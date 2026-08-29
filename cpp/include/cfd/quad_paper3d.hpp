// quad_paper3d.hpp — element-type quadrature rules matching the point counts published for
// THINC/QQ (Xie & Xiao, JCP 2017, Appendix B), for use as the PRODUCTION rule.
//
// WHY THIS FILE EXISTS
//   The 3D production path built its cell rule as a fan over (face, face-edge) sub-tetrahedra
//   with NQ^3 Duffy-Gauss points each. That is valid for arbitrary polyhedra but, on the
//   standard element types, spends far more points than the published scheme: 96 points on a
//   tetrahedron against the 11 of the reference implementation, 192 on a hexahedron against 16.
//   The 2D path already had a simplex shortcut and already matched the paper (6 per triangle,
//   4 per edge); only 3D diverged.
//
//   Point counts implemented here, per Xie & Xiao Appendix B:
//     cell : tetrahedron 11, pyramid 12, prism 12, hexahedron 16
//     face : triangle 6, quadrilateral 9
//
// WHAT IS NOT CLAIMED
//   The paper states "standard" rules for the tetrahedron and pyramid without listing their
//   abscissae, and its hexahedral/prismatic tables give the direction-adaptive layout. The
//   rules below therefore reproduce the published POINT COUNTS and the published
//   direction-adaptive structure using standard published rules of those sizes:
//     tetrahedron 11 : Keast degree-4 (one negative weight, see the warning below)
//     triangle     6 : Dunavant degree-4 (identical to the table the 2D path already uses)
//     triangle     3 : degree-2 midpoint-of-edges rule
//     hexahedron  16 : 4-point Gauss-Legendre along the interface-principal reference axis,
//                      2-point along the other two, exactly as the paper prescribes
//     prism       12 : 6-point triangle x 2-point line, or 3-point triangle x 4-point line,
//                      selected by the principal direction, exactly as the paper prescribes
//     pyramid     12 : conical product, 2x2 Gauss-Legendre in the base x 3 points along the
//                      collapsing axis
//
// WARNING — the Keast 11-point rule carries a NEGATIVE centroid weight (-74/5625 before
//   normalisation). The cell-average constraint solved by Newton has derivative
//   sum_q w_q (1 - tanh^2), which a negative weight can drive non-positive once the positive
//   points saturate. This is a real failure mode in the steep regime and must be measured, not
//   assumed away; u3_tanh_cellD already guards on a non-positive derivative.
//
// Weights are normalised so that sum_q w_q == 1 exactly: each is w_ref * |J| divided by the
// summed w_ref * |J|. Normalising by the summed Jacobian rather than by the mesh's stored cell
// volume keeps the cell-average constraint self-consistent even where the isoparametric image
// of the element and the mesh's face-fan volume disagree, which they do on warped faces.
#pragma once
#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace cfd {

// ---------------------------------------------------------------------------------------
// 1D Gauss-Legendre on [0,1]
// ---------------------------------------------------------------------------------------
struct GL1D { int n; double x[4], w[4]; };

inline const GL1D& gl1d(int n) {
    static const GL1D g2 = {2, {0.5 - 0.5 / 1.7320508075688772, 0.5 + 0.5 / 1.7320508075688772, 0, 0},
                               {0.5, 0.5, 0, 0}};
    static const GL1D g3 = {3, {0.5 - 0.5 * 0.7745966692414834, 0.5, 0.5 + 0.5 * 0.7745966692414834, 0},
                               {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0, 0}};
    static const GL1D g4 = {4, {0.5 - 0.5 * 0.8611363115940526, 0.5 - 0.5 * 0.3399810435848563,
                                0.5 + 0.5 * 0.3399810435848563, 0.5 + 0.5 * 0.8611363115940526},
                               {0.3478548451374538 * 0.5, 0.6521451548625461 * 0.5,
                                0.6521451548625461 * 0.5, 0.3478548451374538 * 0.5}};
    return n <= 2 ? g2 : (n == 3 ? g3 : g4);
}

// ---------------------------------------------------------------------------------------
// Triangle rules in barycentric coordinates, weights summing to 1
// ---------------------------------------------------------------------------------------
struct TriRule { int n; double L1[6], L2[6], w[6]; };

inline const TriRule& tri_rule(int n) {
    // degree 2, 3 points
    static const TriRule t3 = {3,
        {2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 0, 0, 0},
        {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0, 0, 0},
        {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0, 0, 0}};
    // degree 4, 6 points (Dunavant) — same abscissae/weights as the 2D production table
    static const TriRule t6 = {6,
        {0.816847572980459, 0.091576213509771, 0.091576213509771,
         0.108103018168070, 0.445948490915965, 0.445948490915965},
        {0.091576213509771, 0.816847572980459, 0.091576213509771,
         0.445948490915965, 0.108103018168070, 0.445948490915965},
        {0.109951743655322, 0.109951743655322, 0.109951743655322,
         0.223381589678011, 0.223381589678011, 0.223381589678011}};
    return n <= 3 ? t3 : t6;
}

// ---------------------------------------------------------------------------------------
// Keast degree-4, 11 points, barycentric, weights normalised to sum 1
// ---------------------------------------------------------------------------------------
struct TetRule { int n; double L[11][4], w[11]; };

inline const TetRule& tet_rule11() {
    static const TetRule r = [] {
        TetRule t{}; t.n = 11;
        const double c = 0.25;
        const double p = 0.7857142857142857, q = 0.0714285714285714;   // 11/14, 1/14
        const double a = 0.3994035761667992, b = 0.1005964238332008;
        const double wc = -0.0789333333333333;   // -74/5625 * 6
        const double wp =  0.0457333333333333;   //  343/45000 * 6
        const double wb =  0.1493333333333333;   //   56/2250 * 6
        int k = 0;
        t.L[k][0] = c; t.L[k][1] = c; t.L[k][2] = c; t.L[k][3] = c; t.w[k] = wc; ++k;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) t.L[k][j] = q;
            t.L[k][i] = p; t.w[k] = wp; ++k;
        }
        static const int pr[6][4] = {{0,1,2,3},{0,2,1,3},{0,3,1,2},{1,2,0,3},{1,3,0,2},{2,3,0,1}};
        for (int i = 0; i < 6; ++i) {
            t.L[k][pr[i][0]] = a; t.L[k][pr[i][1]] = a;
            t.L[k][pr[i][2]] = b; t.L[k][pr[i][3]] = b;
            t.w[k] = wb; ++k;
        }
        return t;
    }();
    return r;
}

// ---------------------------------------------------------------------------------------
// Isoparametric maps. V holds the cell's vertices in the canonical order used by the mesh
// reader: tet (v0..v3); pyramid quad base v0..v3 then apex v4; prism bottom triangle v0..v2
// then top v3..v5; hexahedron bottom v0..v3 then top v4..v7.
// ---------------------------------------------------------------------------------------
inline void map_point(int nv, const double (*V)[3], double r, double s, double t,
                      double x[3], double dxr[3], double dxs[3], double dxt[3]) {
    for (int d = 0; d < 3; ++d) { x[d] = 0; dxr[d] = 0; dxs[d] = 0; dxt[d] = 0; }
    if (nv == 4) {                                    // tetrahedron, affine
        const double N[4] = {1 - r - s - t, r, s, t};
        const double dr[4] = {-1, 1, 0, 0}, ds[4] = {-1, 0, 1, 0}, dt[4] = {-1, 0, 0, 1};
        for (int i = 0; i < 4; ++i) for (int d = 0; d < 3; ++d) {
            x[d] += N[i] * V[i][d]; dxr[d] += dr[i] * V[i][d];
            dxs[d] += ds[i] * V[i][d]; dxt[d] += dt[i] * V[i][d]; }
    } else if (nv == 5) {                             // pyramid: bilinear base, collapsing to apex
        const double b[4]  = {(1 - r) * (1 - s), r * (1 - s), r * s, (1 - r) * s};
        const double br[4] = {-(1 - s), (1 - s), s, -s};
        const double bs[4] = {-(1 - r), -r, r, (1 - r)};
        for (int i = 0; i < 4; ++i) for (int d = 0; d < 3; ++d) {
            x[d]   += (1 - t) * b[i]  * V[i][d];
            dxr[d] += (1 - t) * br[i] * V[i][d];
            dxs[d] += (1 - t) * bs[i] * V[i][d];
            dxt[d] += -b[i] * V[i][d]; }
        for (int d = 0; d < 3; ++d) { x[d] += t * V[4][d]; dxt[d] += V[4][d]; }
    } else if (nv == 6) {                             // prism: triangle x line
        const double L[3]  = {1 - r - s, r, s};
        const double Lr[3] = {-1, 1, 0}, Ls[3] = {-1, 0, 1};
        for (int i = 0; i < 3; ++i) for (int d = 0; d < 3; ++d) {
            const double lo = V[i][d], hi = V[i + 3][d];
            x[d]   += L[i]  * ((1 - t) * lo + t * hi);
            dxr[d] += Lr[i] * ((1 - t) * lo + t * hi);
            dxs[d] += Ls[i] * ((1 - t) * lo + t * hi);
            dxt[d] += L[i]  * (hi - lo); }
    } else {                                          // hexahedron, trilinear
        const double N[8]  = {(1-r)*(1-s)*(1-t), r*(1-s)*(1-t), r*s*(1-t), (1-r)*s*(1-t),
                              (1-r)*(1-s)*t,     r*(1-s)*t,     r*s*t,     (1-r)*s*t};
        const double dr[8] = {-(1-s)*(1-t), (1-s)*(1-t), s*(1-t), -s*(1-t),
                              -(1-s)*t,     (1-s)*t,     s*t,     -s*t};
        const double ds[8] = {-(1-r)*(1-t), -r*(1-t), r*(1-t), (1-r)*(1-t),
                              -(1-r)*t,     -r*t,     r*t,     (1-r)*t};
        const double dt[8] = {-(1-r)*(1-s), -r*(1-s), -r*s, -(1-r)*s,
                               (1-r)*(1-s),  r*(1-s),  r*s,  (1-r)*s};
        for (int i = 0; i < 8; ++i) for (int d = 0; d < 3; ++d) {
            x[d] += N[i] * V[i][d]; dxr[d] += dr[i] * V[i][d];
            dxs[d] += ds[i] * V[i][d]; dxt[d] += dt[i] * V[i][d]; }
    }
}

inline double jac_det(const double a[3], const double b[3], const double c[3]) {
    return a[0] * (b[1] * c[2] - b[2] * c[1])
         - a[1] * (b[0] * c[2] - b[2] * c[0])
         + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

// Which reference axis the interface normal is most aligned with. The paper switches the
// hexahedral and prismatic layouts on the principal direction of the interface, expressed in
// the element's reference frame; g is the interface normal in physical space.
inline int principal_axis(int nv, const double (*V)[3], const double g[3]) {
    double x[3], t0[3], t1[3], t2[3];
    const double rc = (nv == 4 || nv == 6) ? 1.0 / 3.0 : 0.5;
    map_point(nv, V, rc, rc, 0.5, x, t0, t1, t2);
    const double* T[3] = {t0, t1, t2};
    int best = 0; double bv = -1;
    for (int i = 0; i < 3; ++i) {
        const double L = std::sqrt(T[i][0]*T[i][0] + T[i][1]*T[i][1] + T[i][2]*T[i][2]);
        if (!(L > 1e-300)) continue;
        const double v = std::fabs(g[0]*T[i][0] + g[1]*T[i][1] + g[2]*T[i][2]) / L;
        if (v > bv) { bv = v; best = i; }
    }
    return best;
}

// ---------------------------------------------------------------------------------------
// Cell rule. Emits reference-space points (r,s,t) and weights that already include |J| and
// are normalised to sum 1. `g` is the interface normal in physical space; it selects the
// direction-adaptive layout for hexahedra and prisms and is ignored for the other types.
// ---------------------------------------------------------------------------------------
struct QPoint { double r, s, t, w; };

// Dense tensor rule on the SAME isoparametric map, n points per axis, collapsed on the simplex
// directions. This is what the reference evaluator refines with: staying inside one geometric
// family is what makes the refinement a convergence test of the quadrature rather than a
// comparison between two different descriptions of the cell.
inline void cell_rule_dense(int nv, int n, std::vector<QPoint>& out) {
    out.clear();
    std::vector<double> gx(n), gw(n);
    for (int i = 0; i < n; ++i) {
        double z = std::cos(3.14159265358979323846 * (i + 0.75) / (n + 0.5)), z1, pp;
        do { double p1 = 1, p2 = 0;
             for (int j = 0; j < n; ++j) { double p3 = p2; p2 = p1; p1 = ((2*j+1)*z*p2 - j*p3)/(j+1); }
             pp = n * (z*p1 - p2) / (z*z - 1.0); z1 = z; z = z1 - p1/pp;
        } while (std::fabs(z - z1) > 1e-15);
        gx[i] = 0.5*(1.0 - z); gw[i] = 1.0/((1.0 - z*z)*pp*pp);
    }
    out.reserve((size_t)n*n*n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        double r = gx[i], s = gx[j], t = gx[k], duffy = 1.0;
        if (nv == 4) {                                  // collapse all three onto the simplex
            r = gx[i]; s = gx[j]*(1-r); t = gx[k]*(1-r)*(1-gx[j]);
            duffy = (1-gx[i])*(1-gx[i])*(1-gx[j]);
        } else if (nv == 6) {                           // collapse the triangle directions only
            r = gx[i]; s = gx[j]*(1-r); duffy = (1-gx[i]);
        }
        out.push_back({r, s, t, gw[i]*gw[j]*gw[k]*duffy});
    }
}

// nq is the quadrature order in play: the production order 2 selects the published point counts,
// any higher order selects the dense rule above so the reference converges within this family.
inline void cell_rule_paper(int nv, const double (*V)[3], const double g[3],
                            std::vector<QPoint>& out, int nq = 2) {
    out.clear();
    if (nq > 2) {
        cell_rule_dense(nv, nq, out);
    } else if (nv == 4) {                              // tetrahedron: 11
        const TetRule& R = tet_rule11();
        for (int q = 0; q < R.n; ++q)
            out.push_back({R.L[q][1], R.L[q][2], R.L[q][3], R.w[q]});
    } else if (nv == 5) {                              // pyramid: 2 x 2 x 3 conical product
        // The collapsing direction carries a (1-t)^2 factor in the Jacobian, so its three points
        // are Gauss-Jacobi (alpha=2, beta=0) rather than Gauss-Legendre: the factor is absorbed
        // into the weight instead of being left for the rule to resolve, which is exact through
        // degree 5 there. The (1-t)^2 is divided back out of |J| below so it is not counted twice.
        static const double GJ_T[3] = {0.072994024073149588, 0.34700376603835187,
                                       0.70500220988849849};
        static const double GJ_W[3] = {0.15713636106488660, 0.14624626925986600,
                                       0.029950703008580701};
        const GL1D& a = gl1d(2);
        for (int i = 0; i < a.n; ++i) for (int j = 0; j < a.n; ++j) for (int k = 0; k < 3; ++k)
            out.push_back({a.x[i], a.x[j], GJ_T[k], a.w[i] * a.w[j] * GJ_W[k]});
    } else if (nv == 6) {                              // prism: 6x2 or 3x4 by principal direction
        const int ax = principal_axis(nv, V, g);
        const bool along_axis = (ax == 2);              // principal perpendicular to the triangle
        const TriRule& T = tri_rule(along_axis ? 3 : 6);
        const GL1D&   L = gl1d(along_axis ? 4 : 2);
        for (int i = 0; i < T.n; ++i) for (int k = 0; k < L.n; ++k)
            out.push_back({T.L1[i], T.L2[i], L.x[k], T.w[i] * L.w[k]});
    } else {                                           // hexahedron: 4 x 2 x 2 by principal axis
        const int ax = principal_axis(nv, V, g);
        const GL1D& g4 = gl1d(4); const GL1D& g2 = gl1d(2);
        const GL1D* A[3] = {&g2, &g2, &g2}; A[ax] = &g4;
        for (int i = 0; i < A[0]->n; ++i) for (int j = 0; j < A[1]->n; ++j) for (int k = 0; k < A[2]->n; ++k)
            out.push_back({A[0]->x[i], A[1]->x[j], A[2]->x[k],
                           A[0]->w[i] * A[1]->w[j] * A[2]->w[k]});
    }
    // fold in |J| and renormalise so the weights are cell-average weights
    const bool gj_pyramid = (nv == 5 && nq <= 2);   // its weights already carry the (1-t)^2
    double sum = 0;
    for (auto& p : out) {
        double x[3], dr[3], ds[3], dt[3];
        map_point(nv, V, p.r, p.s, p.t, x, dr, ds, dt);
        double J = std::fabs(jac_det(dr, ds, dt));
        if (gj_pyramid) { const double u = 1.0 - p.t; J /= (u * u); }
        p.w *= J;
        sum += p.w;
    }
    if (sum > 1e-300) for (auto& p : out) p.w /= sum;
}

// Physical coordinates of a reference point, for evaluating the reconstruction polynomial.
inline void cell_point_xyz(int nv, const double (*V)[3], const QPoint& p, double x[3]) {
    double dr[3], ds[3], dt[3];
    map_point(nv, V, p.r, p.s, p.t, x, dr, ds, dt);
}

// ---------------------------------------------------------------------------------------
// Face rule: 6 points on a triangle, 9 on a quadrilateral (3x3 Gauss-Legendre on the bilinear
// map, so warped quadrilateral faces are integrated on the surface they actually describe).
// Emits physical points and weights summing to 1.
// ---------------------------------------------------------------------------------------
inline void face_rule_paper(int nfv, const double (*F)[3],
                            std::vector<std::array<double, 4>>& out, int nq = 2) {
    out.clear();
    if (nq > 2) {                       // dense refinement inside the same surface description
        std::vector<double> gx(nq), gw(nq);
        for (int i = 0; i < nq; ++i) {
            double z = std::cos(3.14159265358979323846 * (i + 0.75) / (nq + 0.5)), z1, pp;
            do { double p1 = 1, p2 = 0;
                 for (int j = 0; j < nq; ++j) { double p3 = p2; p2 = p1; p1 = ((2*j+1)*z*p2 - j*p3)/(j+1); }
                 pp = nq * (z*p1 - p2) / (z*z - 1.0); z1 = z; z = z1 - p1/pp;
            } while (std::fabs(z - z1) > 1e-15);
            gx[i] = 0.5*(1.0 - z); gw[i] = 1.0/((1.0 - z*z)*pp*pp);
        }
        double sum = 0;
        for (int i = 0; i < nq; ++i) for (int j = 0; j < nq; ++j) {
            double u = gx[i], v = gx[j], duffy = 1.0;
            if (nfv == 3) { v = gx[j] * (1 - u); duffy = (1 - gx[i]); }
            double x[3] = {0,0,0}, a[3] = {0,0,0}, b[3] = {0,0,0};
            if (nfv == 3) {
                const double L[3] = {1-u-v, u, v};
                const double dU[3] = {-1, 1, 0}, dV[3] = {-1, 0, 1};
                for (int k = 0; k < 3; ++k) for (int d = 0; d < 3; ++d) {
                    x[d] += L[k]*F[k][d]; a[d] += dU[k]*F[k][d]; b[d] += dV[k]*F[k][d]; }
            } else {
                const double N[4]  = {(1-u)*(1-v), u*(1-v), u*v, (1-u)*v};
                const double dU[4] = {-(1-v), (1-v), v, -v};
                const double dV[4] = {-(1-u), -u, u, (1-u)};
                for (int k = 0; k < 4; ++k) for (int d = 0; d < 3; ++d) {
                    x[d] += N[k]*F[k][d]; a[d] += dU[k]*F[k][d]; b[d] += dV[k]*F[k][d]; }
            }
            const double c0 = a[1]*b[2]-a[2]*b[1], c1 = a[2]*b[0]-a[0]*b[2], c2 = a[0]*b[1]-a[1]*b[0];
            const double w = gw[i]*gw[j]*duffy*std::sqrt(c0*c0+c1*c1+c2*c2);
            out.push_back({x[0], x[1], x[2], w}); sum += w;
        }
        if (sum > 1e-300) for (auto& p : out) p[3] /= sum;
        return;
    }
    if (nfv == 3) {
        const TriRule& T = tri_rule(6);
        for (int q = 0; q < T.n; ++q) {
            const double L0 = 1 - T.L1[q] - T.L2[q], L1 = T.L1[q], L2 = T.L2[q];
            out.push_back({L0 * F[0][0] + L1 * F[1][0] + L2 * F[2][0],
                           L0 * F[0][1] + L1 * F[1][1] + L2 * F[2][1],
                           L0 * F[0][2] + L1 * F[1][2] + L2 * F[2][2], T.w[q]});
        }
        return;
    }
    const GL1D& G = gl1d(3);
    double sum = 0;
    for (int i = 0; i < G.n; ++i) for (int j = 0; j < G.n; ++j) {
        const double u = G.x[i], v = G.x[j];
        const double N[4]  = {(1-u)*(1-v), u*(1-v), u*v, (1-u)*v};
        const double du[4] = {-(1-v), (1-v), v, -v};
        const double dv[4] = {-(1-u), -u, u, (1-u)};
        double x[3] = {0,0,0}, a[3] = {0,0,0}, b[3] = {0,0,0};
        for (int k = 0; k < 4; ++k) for (int d = 0; d < 3; ++d) {
            x[d] += N[k] * F[k][d]; a[d] += du[k] * F[k][d]; b[d] += dv[k] * F[k][d]; }
        const double c0 = a[1]*b[2] - a[2]*b[1], c1 = a[2]*b[0] - a[0]*b[2], c2 = a[0]*b[1] - a[1]*b[0];
        const double jw = G.w[i] * G.w[j] * std::sqrt(c0*c0 + c1*c1 + c2*c2);
        out.push_back({x[0], x[1], x[2], jw});
        sum += jw;
    }
    if (sum > 1e-300) for (auto& p : out) p[3] /= sum;
}

// Production rules are on by default. The caller must additionally gate on the quadrature order
// still being the production one: the reference evaluator raises THINCQQ_TANH_NQ to 8 or 16 and
// relies on the fan family converging under that refinement, which a fixed-size element rule
// cannot do. Gating on the order makes the reference safe by construction rather than by the
// runner remembering to set an environment variable.
inline bool paper_quad() {
    static const bool on = [] {
        const char* e = std::getenv("U3_PAPER_QUAD");
        return !(e && e[0] == '0');
    }();
    return on;
}
inline bool u3_paper_quad() { return paper_quad(); }   // 3D call sites

}  // namespace cfd
