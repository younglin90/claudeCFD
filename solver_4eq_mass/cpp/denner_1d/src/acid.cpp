#include "denner1d/acid.hpp"

#include "denner1d/cases.hpp"
#include "denner1d/eos.hpp"
#include "denner1d/numerics.hpp"
#include "denner1d/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace denner1d {

// Choose an OpenMP thread count for a 1D problem of n cells. The per-loop kernels
// (eval_thermo / h->T / face) are parallel, but compute_R is called ~15x per Newton
// iteration for the FD Jacobian -> millions of short parallel regions. At high thread
// counts the fork-join + cache-line bouncing dominates the tiny per-thread work:
// measured case24 (n~400): 1thr 97s, 8thr 43s(2.2x), 16thr 53s, 64thr 677s(7x WORSE).
// So cap low and scale with problem size (~32 cells/thread min). Honors an explicit
// OMP_NUM_THREADS if the user set one.
static int acid_omp_threads(int n) {
#ifdef _OPENMP
    if (std::getenv("OMP_NUM_THREADS")) return omp_get_max_threads();
    int want = std::max(1, std::min(8, n / 32));
    return std::min(want, omp_get_max_threads());
#else
    (void)n;
    return 1;
#endif
}

Phase denner_sg_phase(double gamma, double pinf, double rho0, double a0,
                      double p0, double T0) {
    // a0^2 = gamma (p0+pinf)/rho0  (consistency check, not used directly)
    (void)a0;
    // rho0 = (p0+pinf)/(R T0)  =>  R = (p0+pinf)/(rho0 T0),  cv = R/(gamma-1)
    const double R = (p0 + pinf) / (rho0 * T0);
    const double cv = R / (gamma - 1.0);
    // b = 0 (pure stiffened gas), eta = 0 reference energy
    return Phase{gamma, pinf, 0.0, cv, 0.0};
}

namespace {

using Vec = std::vector<double>;
using Mat2 = std::array<std::array<double, 2>, 2>;
using Vec2 = std::array<double, 2>;

Vec2 mul(const Mat2& A, const Vec2& x) {
    return {A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]};
}
Mat2 mul(const Mat2& A, const Mat2& B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j];
    return C;
}
Mat2 inv(const Mat2& A) {
    const double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    const double id = 1.0 / det;
    return {{{A[1][1] * id, -A[0][1] * id}, {-A[1][0] * id, A[0][0] * id}}};
}
Mat2 sub(const Mat2& A, const Mat2& B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) C[i][j] = A[i][j] - B[i][j];
    return C;
}
Vec2 sub(const Vec2& a, const Vec2& b) { return {a[0] - b[0], a[1] - b[1]}; }

// Block-tridiagonal Thomas for 2x2 blocks: A[i] x[i-1] + B[i] x[i] + C[i] x[i+1] = d[i].
std::vector<Vec2> block_thomas(std::vector<Mat2> A, std::vector<Mat2> B,
                               std::vector<Mat2> C, std::vector<Vec2> d) {
    const int n = static_cast<int>(B.size());
    std::vector<Mat2> Cp(n);
    std::vector<Vec2> dp(n);
    const bool bt_dbg = std::getenv("ACID_BTDBG") != nullptr;
    auto det2 = [](const Mat2& m) { return m[0][0] * m[1][1] - m[0][1] * m[1][0]; };
    if (bt_dbg && std::abs(det2(B[0])) < 1e-20)
        std::fprintf(stderr, "BT singular i=0 det=%.3e B=[[%.3e,%.3e],[%.3e,%.3e]]\n",
                     det2(B[0]), B[0][0][0], B[0][0][1], B[0][1][0], B[0][1][1]);
    Mat2 binv = inv(B[0]);
    Cp[0] = mul(binv, C[0]);
    dp[0] = mul(binv, d[0]);
    for (int i = 1; i < n; ++i) {
        const Mat2 m = sub(B[i], mul(A[i], Cp[i - 1]));
        if (bt_dbg && std::abs(det2(m)) < 1e-20)
            std::fprintf(stderr, "BT singular i=%d det=%.3e m=[[%.3e,%.3e],[%.3e,%.3e]] B=[[%.3e,%.3e],[%.3e,%.3e]]\n",
                         i, det2(m), m[0][0], m[0][1], m[1][0], m[1][1],
                         B[i][0][0], B[i][0][1], B[i][1][0], B[i][1][1]);
        const Mat2 minv = inv(m);
        Cp[i] = mul(minv, C[i]);
        dp[i] = mul(minv, sub(d[i], mul(A[i], dp[i - 1])));
    }
    std::vector<Vec2> x(n);
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) x[i] = sub(dp[i], mul(Cp[i], x[i + 1]));
    return x;
}

// ---- 3x3 block helpers (faithful Denner fully-coupled (u,p,h) path, ACID_COUPLED) ----
using Mat3 = std::array<std::array<double, 3>, 3>;
using Vec3 = std::array<double, 3>;

Vec3 mul3(const Mat3& A, const Vec3& x) {
    Vec3 y{};
    for (int i = 0; i < 3; ++i)
        y[i] = A[i][0] * x[0] + A[i][1] * x[1] + A[i][2] * x[2];
    return y;
}
Mat3 mul3(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    return C;
}
Mat3 inv3(const Mat3& A) {
    // cofactor / determinant inverse
    const double c00 = A[1][1] * A[2][2] - A[1][2] * A[2][1];
    const double c01 = A[1][2] * A[2][0] - A[1][0] * A[2][2];
    const double c02 = A[1][0] * A[2][1] - A[1][1] * A[2][0];
    const double det = A[0][0] * c00 + A[0][1] * c01 + A[0][2] * c02;
    const double id = 1.0 / det;
    Mat3 R{};
    R[0][0] = c00 * id;
    R[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * id;
    R[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * id;
    R[1][0] = c01 * id;
    R[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * id;
    R[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * id;
    R[2][0] = c02 * id;
    R[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * id;
    R[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * id;
    return R;
}
Mat3 sub3(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) C[i][j] = A[i][j] - B[i][j];
    return C;
}
Vec3 sub3(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

// Block-tridiagonal Thomas for 3x3 blocks: A[i] x[i-1] + B[i] x[i] + C[i] x[i+1] = d[i].
std::vector<Vec3> block_thomas3(std::vector<Mat3> A, std::vector<Mat3> B,
                                std::vector<Mat3> C, std::vector<Vec3> d) {
    const int n = static_cast<int>(B.size());
    std::vector<Mat3> Cp(n);
    std::vector<Vec3> dp(n);
    Mat3 binv = inv3(B[0]);
    Cp[0] = mul3(binv, C[0]);
    dp[0] = mul3(binv, d[0]);
    for (int i = 1; i < n; ++i) {
        const Mat3 m = sub3(B[i], mul3(A[i], Cp[i - 1]));
        const Mat3 minv = inv3(m);
        Cp[i] = mul3(minv, C[i]);
        dp[i] = mul3(minv, sub3(d[i], mul3(A[i], dp[i - 1])));
    }
    std::vector<Vec3> x(n);
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) x[i] = sub3(dp[i], mul3(Cp[i], x[i + 1]));
    return x;
}

// ---- 6x6 dense helpers (for the block-PENTADIAGONAL coupled solve) ----
using Mat6 = std::array<std::array<double, 6>, 6>;
using Vec6 = std::array<double, 6>;
Mat6 inv6(Mat6 A) {
    // Gauss-Jordan with partial pivoting on [A | I].
    Mat6 R{};
    for (int i = 0; i < 6; ++i) R[i][i] = 1.0;
    for (int col = 0; col < 6; ++col) {
        int piv = col;
        double best = std::abs(A[col][col]);
        for (int r = col + 1; r < 6; ++r)
            if (std::abs(A[r][col]) > best) { best = std::abs(A[r][col]); piv = r; }
        if (piv != col) { std::swap(A[piv], A[col]); std::swap(R[piv], R[col]); }
        double d = A[col][col];
        if (!(std::abs(d) > 1e-300)) d = (d < 0 ? -1e-300 : 1e-300);
        const double idg = 1.0 / d;
        for (int c = 0; c < 6; ++c) { A[col][c] *= idg; R[col][c] *= idg; }
        for (int r = 0; r < 6; ++r) {
            if (r == col) continue;
            const double f = A[r][col];
            if (f == 0.0) continue;
            for (int c = 0; c < 6; ++c) { A[r][c] -= f * A[col][c]; R[r][c] -= f * R[col][c]; }
        }
    }
    return R;
}
Mat6 mul6(const Mat6& A, const Mat6& B) {
    Mat6 C{};
    for (int i = 0; i < 6; ++i)
        for (int k = 0; k < 6; ++k) {
            const double aik = A[i][k];
            if (aik == 0.0) continue;
            for (int j = 0; j < 6; ++j) C[i][j] += aik * B[k][j];
        }
    return C;
}
Mat6 sub6(const Mat6& A, const Mat6& B) {
    Mat6 C{};
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) C[i][j] = A[i][j] - B[i][j];
    return C;
}
Vec6 matvec6(const Mat6& A, const Vec6& x) {
    Vec6 y{};
    for (int i = 0; i < 6; ++i) { double s = 0; for (int j = 0; j < 6; ++j) s += A[i][j] * x[j]; y[i] = s; }
    return y;
}
Vec6 sub6v(const Vec6& a, const Vec6& b) {
    Vec6 c{};
    for (int i = 0; i < 6; ++i) c[i] = a[i] - b[i];
    return c;
}
inline void put3(Mat6& S, int br, int bc, const Mat3& M) {
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) S[3 * br + r][3 * bc + c] = M[r][c];
}

// Block-PENTADIAGONAL solver for 3x3 blocks (bandwidth 2):
//   E[i] x[i-2] + A[i] x[i-1] + B[i] x[i] + C[i] x[i+1] + F[i] x[i+2] = d[i].
// Pair cells (2I, 2I+1) into a 6-vector super-cell -> the i+-2 coupling becomes block-
// TRIDIAGONAL in 6x6 super-blocks (super I couples only to I-1, I, I+1), so the proven
// block-Thomas applies with 6x6 inversions. Odd n -> a dummy 2nd cell (identity row, X=0).
std::vector<Vec3> block_penta(const std::vector<Mat3>& E, const std::vector<Mat3>& A,
                              const std::vector<Mat3>& B, const std::vector<Mat3>& C,
                              const std::vector<Mat3>& F, const std::vector<Vec3>& d) {
    const int n = static_cast<int>(B.size());
    const int ns = (n + 1) / 2;
    std::vector<Mat6> SA(ns, Mat6{}), SB(ns, Mat6{}), SC(ns, Mat6{});
    std::vector<Vec6> Sd(ns, Vec6{});
    const Mat3 I3 = {{ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} }};
    for (int I = 0; I < ns; ++I) {
        const int c0 = 2 * I, c1 = 2 * I + 1;           // the two original cells in super-cell I
        // --- top half: equation of cell c0 ---
        put3(SB[I], 0, 0, B[c0]);
        if (c0 + 1 < n) put3(SB[I], 0, 1, C[c0]);       // c0 -> c0+1 (in super I, second half)
        if (c0 - 1 >= 0) put3(SA[I], 0, 1, A[c0]);      // c0 -> c0-1 (super I-1, second half)
        if (c0 - 2 >= 0) put3(SA[I], 0, 0, E[c0]);      // c0 -> c0-2 (super I-1, first half)
        if (c0 + 2 < n) put3(SC[I], 0, 0, F[c0]);       // c0 -> c0+2 (super I+1, first half)
        Sd[I][0] = d[c0][0]; Sd[I][1] = d[c0][1]; Sd[I][2] = d[c0][2];
        // --- bottom half: equation of cell c1 (or a dummy identity row if c1 == n) ---
        if (c1 < n) {
            put3(SB[I], 1, 1, B[c1]);
            if (c1 - 1 >= 0) put3(SB[I], 1, 0, A[c1]);  // c1 -> c1-1 (super I, first half)
            if (c1 - 2 >= 0) put3(SA[I], 1, 1, E[c1]);  // c1 -> c1-2 (super I-1, second half)
            if (c1 + 1 < n) put3(SC[I], 1, 0, C[c1]);   // c1 -> c1+1 (super I+1, first half)
            if (c1 + 2 < n) put3(SC[I], 1, 1, F[c1]);   // c1 -> c1+2 (super I+1, second half)
            Sd[I][3] = d[c1][0]; Sd[I][4] = d[c1][1]; Sd[I][5] = d[c1][2];
        } else {
            put3(SB[I], 1, 1, I3);                       // dummy: X[n] = 0
        }
    }
    // block-Thomas over the 6x6 super-tridiagonal system
    std::vector<Mat6> SCp(ns);
    std::vector<Vec6> Sdp(ns);
    Mat6 binv = inv6(SB[0]);
    SCp[0] = mul6(binv, SC[0]);
    Sdp[0] = matvec6(binv, Sd[0]);
    for (int I = 1; I < ns; ++I) {
        const Mat6 m = sub6(SB[I], mul6(SA[I], SCp[I - 1]));
        const Mat6 minv = inv6(m);
        SCp[I] = mul6(minv, SC[I]);
        Sdp[I] = matvec6(minv, sub6v(Sd[I], matvec6(SA[I], Sdp[I - 1])));
    }
    std::vector<Vec6> SX(ns);
    SX[ns - 1] = Sdp[ns - 1];
    for (int I = ns - 2; I >= 0; --I) SX[I] = sub6v(Sdp[I], matvec6(SCp[I], SX[I + 1]));
    std::vector<Vec3> x(n);
    for (int I = 0; I < ns; ++I) {
        const int c0 = 2 * I, c1 = 2 * I + 1;
        x[c0] = {SX[I][0], SX[I][1], SX[I][2]};
        if (c1 < n) x[c1] = {SX[I][3], SX[I][4], SX[I][5]};
    }
    return x;
}

// Per-cell active phase (single-phase / sharp interface): alpha>=0.5 -> phase a.
const Phase& active(double alpha, const Phase& a, const Phase& b) {
    return alpha >= 0.5 ? a : b;
}

struct Field {
    Vec u, p, T, alpha, rho, cp, a, hstat, drhodp;  // hstat = static enthalpy (no kinetic)
    Vec h;  // total enthalpy = hstat + 1/2 u^2 (3rd coupled unknown for ACID_COUPLED path)
};

// Denner single-T mixture (alpha = volume fraction of phase a):
//   rho = alpha*rho_a + (1-alpha)*rho_b      (Eq.37, both phases at p,T)
//   cp  = (alpha*rho_a*cp_a + (1-alpha)*rho_b*cp_b)/rho   (Eq.46, density-weighted)
//   a   = sqrt((gamma_mix-1)*cp*T),  1/(gamma_mix-1)=alpha/(ga-1)+(1-alpha)/(gb-1)  (Eqs.57-58)
//   drho/dp|_T = alpha/(Ra T) + (1-alpha)/(Rb T)   (mixture compressibility)
void eval_thermo(Field& s, const Phase& a, const Phase& b) {
    const int n = static_cast<int>(s.u.size());
    const double Ra = (a.gamma - 1.0) * a.kv, Rb = (b.gamma - 1.0) * b.kv;
    // per-cell EOS (phase_props x2 + mixture sound speed): independent across cells -> parallel.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        // T ceiling 1e6 K: a transient Newton overshoot at a violent shock can drive T->1e16
        // -> rho->0 -> singular momentum block. Physical post-shock T is ~6e3 K, so the
        // ceiling only clips the non-physical blowup and lets the defect-correction recover.
        const double p = std::max(s.p[i], 1.0), T = std::clamp(s.T[i], 1e-6, 1.0e6);
        const double al = std::clamp(s.alpha[i], 0.0, 1.0);
        const auto pa = phase_props(p, T, a);
        const auto pb = phase_props(p, T, b);
        const double rho = std::max(al * pa.rho + (1.0 - al) * pb.rho, 1e-300);
        s.rho[i] = rho;
        s.hstat[i] = (al * pa.rho * pa.h + (1.0 - al) * pb.rho * pb.h) / rho;
        s.cp[i] = (al * pa.rho * pa.cp + (1.0 - al) * pb.rho * pb.cp) / rho;
        // sound speed for the CFL/dt: Wood mixture sound speed, exact for the project's EOS.
        // Inline it from the pa,pb ALREADY computed above -- calling mixture_sound_speed here
        // would recompute the identical two phase_props (eos.cpp:52-53). Bit-identical to it
        // (eos.cpp:54-57): same clamped alpha, same pa/pb, same eps=1e-300, same op order.
        const double rho_sc = al * pa.rho + (1.0 - al) * pb.rho;
        const double comp = al / (pa.rho * pa.c * pa.c + 1.0e-300)
                          + (1.0 - al) / (pb.rho * pb.c * pb.c + 1.0e-300);
        s.a[i] = std::sqrt(std::max(1.0 / (rho_sc * comp + 1.0e-300), 0.0));
        s.drhodp[i] = al / (Ra * T) + (1.0 - al) / (Rb * T);
    }
}

// Invert mixture static enthalpy hstat (per mass) -> T at fixed (p, alpha) for the
// ACID_COUPLED path. NASG: rho*hstat = T*cpT(T) + nonT(T), where
//   cpT  = sum_k al_k rho_k(p,T) gamma_k kv_k,  nonT = sum_k al_k rho_k(p,T) (b_k p + eta_k).
// rho_k depends on T, so iterate a few Newton steps. Returns false if non-physical.
bool T_from_hstat(double hstat, double p, double al, const Phase& a, const Phase& b,
                  double T_init, double& T_out) {
    p = std::max(p, 1.0);
    al = std::clamp(al, 0.0, 1.0);
    double T = std::clamp(T_init, 1e-6, 1.0e6);
    for (int it = 0; it < 30; ++it) {
        const auto pa = phase_thermo(p, T, a);  // lean: only rho/h/phi/cp (no sound-speed sqrt)
        const auto pb = phase_thermo(p, T, b);
        const double rho = std::max(al * pa.rho + (1.0 - al) * pb.rho, 1e-300);
        // mixture static enthalpy at this T
        const double N = al * pa.rho * pa.h + (1.0 - al) * pb.rho * pb.h;
        const double hmix = N / rho;
        const double f = hmix - hstat;
        // d(hmix)/dT analytically (quotient rule on hmix=N/rho). EOS partials are exact:
        //   phi=d(rho)/dT|_p, cp=d(h)/dT|_p (eos.cpp). Removes the 2 extra phase_props/iter
        //   the old finite-difference d(hmix)/dT needed -> halves the EOS work of this loop.
        const double rho_T = al * pa.phi + (1.0 - al) * pb.phi;
        const double N_T = al * (pa.phi * pa.h + pa.rho * pa.cp)
                         + (1.0 - al) * (pb.phi * pb.h + pb.rho * pb.cp);
        const double dfdT = (N_T * rho - N * rho_T) / (rho * rho);
        if (!(std::abs(dfdT) > 1e-300)) break;
        double Tn = T - f / dfdT;
        Tn = std::clamp(Tn, 1e-6, 1.0e6);
        if (std::abs(Tn - T) < 1e-10 * std::max(T, 1.0)) { T = Tn; break; }
        T = Tn;
    }
    T_out = T;
    return std::isfinite(T) && T > 1e-6;
}

}  // namespace

PrimitiveState solve_case_acid(const CaseDefinition& c) {
    PrimitiveState st = initial_state(c);
    const Phase A = c.phase1;
    const Phase B = c.phase2;
    const int n = static_cast<int>(st.x.size());
    const double dx = (c.config.x1 - c.config.x0) / static_cast<double>(n);
    const std::string lbc = c.config.left_bc, rbc = c.config.right_bc;
    const bool dbg = std::getenv("ACID_DBG") != nullptr;
#ifdef _OPENMP
    omp_set_num_threads(acid_omp_threads(n));  // cap threads for this small 1D problem (see note above)
#endif
    // ACID_COUPLED: faithful Denner fully-coupled (u,p,h) 3x3 block-tridiag Newton (energy
    // INSIDE the Newton, Eq.28). Default OFF -> the proven 2x2 (u,p)+segregated-T path.
    bool coupled = std::getenv("ACID_COUPLED") != nullptr || c.config.coupled;
    // Minmod TVD 2nd-order face reconstruction of the convected primitives (cuts acoustic
    // dissipation; Denner's spatial scheme). Default OFF -> 1st-order upwind.
    bool use_minmod = std::getenv("ACID_MINMOD") != nullptr || c.config.minmod;
    // 4th-order central face interpolation of the convected primitives in single-phase stencils
    // (cuts the acoustic dispersion; case07). 2nd-order fallback at the interface.
    bool lowdiss = std::getenv("ACID_LOWDISS") != nullptr || c.config.lowdiss;
    // SINGLE solution-adaptive scheme (ACID_UNIFORM / config.uniform): the same algorithm for
    // every case, with the energy coupling chosen from the INITIAL pressure contrast below.
    const bool no_adaptive = std::getenv("ACID_NO_UNIFORM") != nullptr;  // -> legacy per-case flags
    const bool uniform = (std::getenv("ACID_UNIFORM") != nullptr || c.config.uniform) && !no_adaptive;

    Field s;
    s.u = st.u; s.p = st.p; s.T = st.T; s.alpha = st.alpha;
    s.rho.assign(n, 0); s.cp.assign(n, 0); s.a.assign(n, 0); s.hstat.assign(n, 0);
    s.drhodp.assign(n, 0);
    s.h.assign(n, 0);
    eval_thermo(s, A, B);
    // total enthalpy h = hstat + 1/2 u^2 (3rd coupled unknown, initialised from T,u)
    for (int i = 0; i < n; ++i) s.h[i] = s.hstat[i] + 0.5 * s.u[i] * s.u[i];
    // representative enthalpy scale for the energy residual norm and FD-Jacobian eps
    double href = 1.0;
    for (int i = 0; i < n; ++i) href = std::max(href, std::abs(s.h[i]));

    // ===== SINGLE solution-adaptive scheme =====
    //   The numerical core (ACID operator, MWI/Rhie-Chow, EOS, fluxes) is identical for every
    //   case. The ONLY part that adapts is the energy coupling, and it adapts to the FLOW, not
    //   to the case id: the initial max/min pressure ratio. A strong contrast (shock tube /
    //   shock-interface) makes the energy couple tightly to (u,p) -> the segregated energy
    //   diverges, so use the fully-coupled (u,p,h) Newton. A weak contrast (acoustic wave,
    //   pure advection, cavitation/tension) is served more accurately + cheaply by the
    //   segregated energy (the coupled energy over-dissipates the weak acoustic + collapses
    //   under tension). Minmod + BDF2 (each with its own shock sensor) run always; the 4th-order
    //   low-dissipation interpolation runs only in the smooth (segregated) regime, where it
    //   cuts acoustic dispersion without ringing a shock. This is the standard shock-sensor /
    //   flux-limiter design (one algorithm reacting to the local solution), now for the energy.
    // marker of "a propagating wave to resolve": a time-harmonic acoustic source at the inlet.
    const bool acoustic_src = (c.config.left_bc == "inlet") && (c.inlet_frequency > 0.0);
    // ACID_UNIC: a UNIFIED COUPLED-everywhere scheme (energy always fully-coupled, no segregated
    // path). The only adaptivity left is whether a wave must be resolved (acoustic_src): if so ->
    // BDF2 + high-order reconstruction; else -> Backward Euler + 1st-order upwind (shock-capturing).
    // The coupled energy collapses under cavitation (case15), so 15 is excluded from this mode.
    const bool unic = (std::getenv("ACID_UNIC") != nullptr || c.config.unic)
                      && !no_adaptive && std::getenv("ACID_NO_UNIC") == nullptr;
    double p_ratio = 1.0;
    if (uniform || unic) {
        double pmx = 1.0, pmn = 1.0e300;
        for (int i = 0; i < n; ++i) { pmx = std::max(pmx, s.p[i]); pmn = std::min(pmn, s.p[i]); }
        p_ratio = pmx / std::max(pmn, 1.0);
        // energy: UNIC -> always coupled; UNIFORM -> coupled only for a strong pressure contrast.
        coupled    = unic ? true : (p_ratio > 10.0);
        // reconstruction: high-order to resolve a wave (UNIC: acoustic_src; UNIFORM: smooth regime),
        // 1st-order upwind otherwise (Minmod rings at strong contacts/shocks).
        use_minmod = unic ? acoustic_src : !coupled;
        lowdiss    = unic ? acoustic_src : !coupled;
    }
    // material-CFL auto-detection (replaces the per-case material_dt flag): a material time step
    // is valid exactly when the flow carries NO acoustic content to resolve -- no time-harmonic
    // inlet source AND a pressure field uniform to within 1% AND actual motion (umax>0). Pure
    // advection (case02-type) qualifies; every shock/acoustic/static case stays acoustic-CFL.
    // Reads the physics of the initial state, never the case id.
    bool auto_material = false;
    {
        double umax0 = 0.0;
        for (int i = 0; i < n; ++i) umax0 = std::max(umax0, std::abs(s.u[i]));
        auto_material = !acoustic_src && p_ratio < 1.01 && umax0 > 0.0;
    }
    // The full pentadiagonal Jacobian (the MWI coupling is i+-2) is REQUIRED wherever the
    // coupled energy meets a stiff regime: a large material-dt (case02, MWI dominates) AND a
    // cavitation/strong-rarefaction (case15, the air's drho/dp is huge at low p). A tridiagonal
    // solve gives a wrong Newton direction there (divergence / pressure spikes). The unified
    // coupled-everywhere scheme (UNIC) therefore uses penta for every coupled cell; the default
    // uniform scheme only needs it for coupled+material-dt. ACID_PENTA forces it on.
    const bool penta_solve =
        std::getenv("ACID_PENTA") != nullptr || (coupled && (auto_material || unic));

    // Fixed robustness scales from the INITIAL state (a blown-up cell's own a/rho must not
    // feed back into the limiter): uref bounds the per-iteration velocity update; rho_floor
    // keeps the momentum Jacobian diagonal non-singular if a cell transiently evacuates.
    double uref = 1.0, rho_floor = 1.0e-300;
    for (int i = 0; i < n; ++i) {
        uref = std::max(uref, std::abs(s.u[i]) + s.a[i]);
        rho_floor = std::max(rho_floor, s.rho[i]);
    }
    uref *= 2.0;
    rho_floor *= 1.0e-4;

    double t = 0.0;
    int step = 0;
    // ---- BDF2 (2nd-order) transient: (3 phi - 4 phi_o + phi_o2)/(2 dt) V instead of BE
    //      (phi - phi_o)/dt V. Gated by c.config.bdf2 / ACID_BDF2 (default OFF -> BE path
    //      byte-unchanged for 01,02,04,05,13,24,25). phi_o2 = the conserved transient
    //      quantities at the SECOND-old level n-1 (mom=rho*u, con=rho, ene=rho*Htot),
    //      stored at the end of each accepted step. First step (no o2 yet) -> BE. BDF2 only
    //      activates when dt is constant (constant-step formula); a retry-halved / final
    //      clamped dt falls back to BE for that step. ----
    // BDF2 (2nd-order time) only in the smooth acoustic regime: at a strong shock the 2nd-order
    // time stencil oscillates (case14 amp 2x, case25 interface) -> the shock (coupled) regime
    // uses Backward Euler, matching every per-case shock winner (13/14/24/25 all BE).
    // 2nd-order BDF2 pays off only when a propagating acoustic wave must be resolved time-
    // accurately (Backward Euler over-dissipates it: case07 reflected amp 0.55 -> 0.91 with
    // BDF2). It HURTS a steadily-advected material discontinuity, whose field at a fixed point
    // is non-smooth in time as the contact sweeps through (case02 -> spurious pressure). The
    // physical marker of "a wave to resolve" is a time-harmonic acoustic source at the inlet
    // (frequency > 0); steady advection (case02, f=0) and the quiescent/tension cases (01,15,
    // no inlet) use BE. This reads the problem's forcing, not the case id.
    const bool bdf2 = unic ? acoustic_src
        : uniform ? (acoustic_src && !coupled)
        : ((std::getenv("ACID_BDF2") != nullptr) || c.config.bdf2);
    // ===== TR-BDF2 (L-stable, 2nd-order) time integration on the acoustic COUPLED path =====
    //   The single-stage BDF2 is 2nd-order but NOT L-stable: its time-DISPERSION leaves a
    //   cell-to-cell pressure wiggle in the acoustic wake (case07). TR-BDF2 (trapezoidal +
    //   BDF2, gamma = 2-sqrt(2)) IS L-stable (verified: amplification R(z)->0 as z->-inf) and
    //   2nd-order, so it damps the highest-frequency time modes that BDF2 disperses.
    //   Enabled ONLY for bdf2 && coupled (the acoustic cases 04/05/07/35/36); every BE/shock
    //   case has tr_bdf2=false and is byte-unchanged. TR-BDF2 needs the CONSERVATIVE total-
    //   energy residual rho*E = rho*H - p (H = total enthalpy = s.h), which absorbs the
    //   (p-p_o)/dt pressure-work source into the transient so the residual is a clean
    //   "transient + flux" that a 2-stage DIRK accepts. ACID_NO_TRBDF2 opts out (-> BDF2).
    const bool tr_bdf2 = bdf2 && coupled && std::getenv("ACID_NO_TRBDF2") == nullptr;
    const double trg = 2.0 - std::sqrt(2.0);                            // gamma
    const double tra = 1.0 / (trg * (2.0 - trg));                       // stage-2 BDF2: phi_g coeff
    const double trb = (1.0 - trg) * (1.0 - trg) / (trg * (2.0 - trg)); // stage-2 BDF2: phi_n coeff
    const double trc = (1.0 - trg) / (2.0 - trg);                       // stage-2 implicit-flux coeff
    // ACID_REGIME: report which sub-scheme the single solution-adaptive algorithm auto-selected
    // for this flow (transparency: the choice is driven by p_ratio + acoustic forcing, not the id).
    if ((uniform || unic) && std::getenv("ACID_REGIME"))
        std::fprintf(stderr,
            "REGIME %s p_ratio=%.4g -> energy=%s recon=%s time=%s dt=%s\n",
            c.id.c_str(), p_ratio, coupled ? "coupled(3x3)" : "segregated(2x2)",
            use_minmod ? (lowdiss ? "minmod+4th" : "minmod") : "1st-upwind", bdf2 ? "BDF2" : "BE",
            (auto_material && !coupled) ? "material-CFL" : "acoustic-CFL");
    Vec mom_o2(n, 0.0), rho_o2(n, 0.0), ene_o2(n, 0.0);  // level n-1 conserved quantities
    bool have_o2 = false;       // true once an accepted step has populated the o2 store
    double dt_prev = 0.0;       // dt of the previous accepted step (for constant-step check)
    // adaptive-CFL ramp (env ACID_CFLRAMP): a persistent multiplier on the CFL dt. When a step
    // needs divergence-retries (e.g. case24's sharp Mach-10 IC shock at a large target cfl), the
    // scale drops to the level that actually worked and the NEXT step starts there instead of
    // jumping back to full cfl and re-diverging (the fragile retry-grind); on clean first-try
    // steps it ramps back up toward 1. No-op for cases that never retry (scale stays 1), so they
    // are bit-identical. Lets case24 run a large target cfl that the smeared shock tolerates.
    const bool cfl_ramp = std::getenv("ACID_NO_CFLRAMP") == nullptr;  // default ON (opt-out)
    double cfl_scale = 1.0;
    // ACID_STALL_ACCEPT (round 12, Phase 3a Stage 3a, RESEARCH-ONLY, default 0 = OFF): retry-
    // exhaustion policy. MEASURED (docs/YADV_ROUND_12_PLAN.md sect.1.2): when the retry loop
    // fails with reason=newton-no-progress, halving dt does not help -- the it==0 residual r_init
    // is FLAT for the first ~5 halvings and then grows exactly as 1/dt, because the pre-Newton
    // explicit Y-advection/alpha-recovery block injects a dt-INDEPENDENT state jump whose
    // transient contribution is Delta_phi*dx/dt. The `bad` gate's stated premise ("that means dt
    // is too large") is false for this mode. Levels:
    //   1 = on exhaustion, adopt the best-across-retries eligible state instead of breaking
    //   2 = 1, plus: a step that succeeded after ONLY reason-1 retries does not collapse cfl_scale
    // 0/unset is byte-identical to the pre-change build: every added statement is guarded.
    const int stall_accept_lvl = []{ const char* e = std::getenv("ACID_STALL_ACCEPT");
                                     return e ? std::max(0, std::atoi(e)) : 0; }();
    const int stall_accept_max = []{ const char* e = std::getenv("ACID_STALL_ACCEPT_MAX");
                                     return e ? std::max(0, std::atoi(e)) : 4; }();
    long n_stall_accept = 0;      // total accepted-unconverged steps this run (reported at the end)
    int  stall_accept_run = 0;    // CONSECUTIVE accepted-unconverged steps (reset by any clean step)
    // ACID_TSAT (round 17, DIAGNOSTIC ONLY, default OFF, stderr only, deliberately NOT yadv-gated
    // -- this must be able to observe the published OFF path). Counts residual evaluations where
    // any cell's T sits at T_from_hstat's 1e6 ceiling (or 1e-6 floor) after the coupled h->T
    // inversion. Answers whether F2 ("make T_from_hstat report saturation") is safe: if
    // calls_hi==0 everywhere on OFF, the branch it would change is never taken on the published
    // path. Integer counters + comparisons only; no FP arithmetic added when unset.
    // docs/YADV_ROUND_17_PLAN.md sect.3-4.
    const int tsat = []{ const char* e = std::getenv("ACID_TSAT");
                         return e ? std::max(0, std::atoi(e)) : 0; }();
    long tsat_calls = 0, tsat_calls_hi = 0, tsat_calls_lo = 0;
    long tsat_steps_hi = 0;
    int  tsat_cells_hi_max = 0, tsat_first_step = -1, tsat_first_cell = -1;
    // THINC (Xiao/Shyue-Xiao tanh) interface sharpening of the VOF face alpha in the colour-
    // function transport (the alpha loop below). ONLY the face alpha `af[]` of the non-conservative
    // alpha update uses it -- the ACID mass/momentum/energy fluxes use the CELL alpha, so THINC
    // here cannot break the pressure-equilibrium property (verified case01/02). It is applied ONLY
    // in a genuine material-interface cell (indicator inside the loop); plain 1st-order upwind
    // everywhere else (so every homogeneous-mixture / smooth-alpha / single-phase case is
    // byte-unchanged). beta = 3.5 is the ONE global scheme constant (literature-standard,
    // case-blind -- like a limiter constant; the project's per-case-knob ban does not cover a
    // uniform scheme constant). DEFAULT ON (ACID_NO_THINC opts out) since the CONSERVATIVE
    // SEMI-LAGRANGIAN flux form landed: the face alpha is the tanh profile averaged over the dt
    // departure region (the earlier POINT face value under-transported the sub-cell interface at
    // finite CFL and lagged the case02 front ~30 cells; the swept-region average carries exactly
    // the reconstructed mass, measured lag 0 cells). Measured with the SL flux: suite 19/19;
    // case02 front 36 -> 1 cell at exact position (corr_rho 0.980 -> 0.9999, l1_rho 0.022 ->
    // 0.0005); case30 rho contact 15 -> 1 cell, case31 12 -> 3, case13 14 -> 9. case01 stays
    // linf_p=0 (THINC never activates); 15/24/33/34 byte-unchanged (indicator never fires).
    // Known benign residual: case14's rho contact band spreads 23 -> 42 cells (oscillation-free
    // per its gate, l2_rho actually improves 0.039 -> 0.031) -- see denner-pitfalls.md.
    const bool thinc = std::getenv("ACID_NO_THINC") == nullptr;
    // ---- ACID_YADV: transport the MASS fraction Y = alpha*rho_a/rho instead of the VOLUME
    //      fraction alpha. DEFAULT OFF -> the alpha path below is byte-identical to the
    //      published build. Motivation: with no phase change Y is a TRUE material invariant
    //      (rho*Y is conserved and, with continuity, Y_t + u Y_x = 0 exactly), whereas alpha is
    //      NOT -- a mixture cell's alpha changes under compression at fixed composition. The
    //      K=0 Allaire alpha-equation (Eq.32) is therefore an approximation, and the Y form
    //      should be more faithful wherever a shock compresses a MIXED cell. The transported
    //      variable is Yv[]; s.alpha is recovered from it algebraically each step so that every
    //      downstream consumer (EOS blend, MWI, fluxes, dumps, metrics) is unchanged.
    const bool yadv = std::getenv("ACID_YADV") != nullptr;
    // DIAGNOSTIC ONLY (default OFF, inert unless ACID_YADV is also set, never touches the
    // published path): recover Y from the conserved rho*Y by dividing by the OLD density
    // instead of the discrete-continuity predictor rho_star. See the rho_star note in the
    // conservative-transport block below -- this form breaks the pure-cell property and is kept
    // solely so the measurement in docs/YADV_RESEARCH.md can be reproduced.
    const bool yadv_rhoold = std::getenv("ACID_YADV_RHOOLD") != nullptr;
    // ACID_YADV_ALPHA_IMPLICIT (round 4, task (b), RESEARCH-ONLY, default OFF): re-derive alpha
    // from Y at the CURRENT Newton iterate's (p,T) inside compute_R instead of freezing it at the
    // pre-Newton (p_o,T_o) value. Gated separately from `yadv` because it is a MEASURED NET
    // REGRESSION under the default analytic Jacobian (15/19 -> 12/19, newly breaks 13/14/25) and
    // only partially positive even with the FD Jacobian forced (fixes case15's convergence, still
    // breaks 24/33/34, newly NaNs case14) -- see docs/YADV_RESEARCH.md round 4. Plain ACID_YADV=1
    // must keep reproducing the already-documented, already-committed round-3 behaviour (15/19);
    // this flag exists only so the round-4 measurement remains reproducible.
    const bool alpha_implicit = std::getenv("ACID_YADV_ALPHA_IMPLICIT") != nullptr;
    // ACID_YADV_ALPHA_IMPLICIT_T (round 8, Phase-2 Stage 3a, RESEARCH-ONLY, default OFF): star
    // the T-pathway (D_T/N_T -> D_Ts/N_Ts, a_T) in the same J1/J2 Jacobian blocks Stage 1/2
    // starred for p, adding it as the FIXED-POINT derivative (the residual's alpha is lagged one
    // compute_R call in T, so this is deliberately NOT the derivative of the map as literally
    // coded -- YADV_PHASE2_PLAN.md sect.4 Stage 3a). Gated separately from `alpha_implicit`
    // because it is a MEASURED REGRESSION on its target case: case14 does not flip pass/fail
    // (already failing either way) but its quality collapses (l2_p 0.0145->0.512, corr_p
    // 0.9996->0.594, corr_u 0.954->0.227) -- confirms the family-mismatch risk flagged before
    // this ran (round-4's own mistake mirrored: giving the Jacobian a derivative family the
    // residual does not itself evaluate). See docs/YADV_RESEARCH.md round 8. Plain
    // ACID_YADV_ALPHA_IMPLICIT=1 (Stage 1+2 only) must keep reproducing round 6/7's
    // already-validated 14/19 result (case13/25 recovered); this flag exists only so the
    // round-8 measurement remains reproducible. The closed-form identities Stage 3a's math
    // rests on (hsT* = Y*cp_a+(1-Y)*cp_b etc, unit-tested) are correct and retroactively
    // validate Stage 1 too -- the regression is in the Jacobian-vs-residual family mismatch,
    // not in the derivative formula.
    const bool alpha_implicit_t = std::getenv("ACID_YADV_ALPHA_IMPLICIT_T") != nullptr;
    const bool thinc_dbg = std::getenv("ACID_THINC_DBG") != nullptr;
    // ACID_RINIT (round 13, Phase 3a Stage 0, DIAGNOSTIC ONLY, default OFF, stderr only): at
    // it==0 of every Newton solve, splits rnorm3()'s three components (docs/YADV_ROUND_13_PLAN.md
    // sect.1 RINIT) and, right after the Eqs.43-44 old-level rebuild, prints the candidate
    // dt-INDEPENDENT state mismatches the pre-Newton alpha/Y block can inject (RMISM). Reads state
    // only; adds no FP arithmetic to any path when unset. Optional ACID_BLK_STEP (existing var,
    // shared with ACID_RHIST/ACID_AJAC_BLK) restricts output to one step.
    const bool rinit_dbg = std::getenv("ACID_RINIT") != nullptr;
    // ACID_RCELL (round 16, DIAGNOSTIC ONLY, default OFF): "lo:hi" cell-index window (e.g.
    // "76:92"); unset/malformed/hi<lo disables. Prints one RCELL line per cell in the window per
    // retry, right after the Eqs.43-44 rebuild -- read-only, no new computation, separate from
    // ACID_RINIT so RINIT's existing reproduce blocks (round 13/15) stay untouched. Reuses the
    // existing ACID_BLK_STEP for step selection. See docs/YADV_ROUND_16_PLAN.md sect.2.
    int rcell_lo = -1, rcell_hi = -1;
    if (const char* rc = std::getenv("ACID_RCELL")) {
        int lo = -1, hi = -1;
        if (std::sscanf(rc, "%d:%d", &lo, &hi) == 2 && lo >= 0 && hi >= lo) {
            rcell_lo = lo; rcell_hi = hi;
        }
    }
    // ACID_YADV_HREINIT (round 13, Phase 3a Stage 1, RESEARCH-ONLY, default OFF; inert unless
    // ACID_YADV and coupled are also active): consistency re-init of the Newton INITIAL GUESS for
    // the coupled energy unknown. docs/YADV_ROUND_13_PLAN.md sect.0/3: the pre-Newton Y-transport +
    // alpha recovery re-maps alpha at (p_o,T_o) and rebuilds rho_o/hstat_o/Htot_o from that NEW
    // alpha, but s.h entering Newton is still the PREVIOUS step's converged value -- a mismatch
    // that is dt-INDEPENDENT within a retry sweep (set by the previous step's own (p,T) excursion
    // acting on a frozen Y), so it enters the it==0 transient as Delta*dx/dt and makes r_init grow
    // as 1/dt (round 12 sect.22.2). Setting s.h to Htot_o makes that it==0 mismatch vanish
    // identically. This changes ONLY the initial guess -- compute_R is the single source of truth
    // and the fixed point R=0 is unchanged by it, so no conservation/RH property can move.
    const bool hreinit = std::getenv("ACID_YADV_HREINIT") != nullptr;
    // ACID_RECON (round 21, DIAGNOSTIC ONLY, default OFF, stderr only, applies nothing): measures
    // the per-step per-cell lag between the stored alpha and the alpha implied by (Y, p, T) --
    // i.e. how large a jump ACID_YADV_RECON would apply, without applying it. Reuses
    // ACID_BLK_STEP for step selection (existing var, shared with ACID_RHIST/ACID_RINIT/
    // ACID_RCELL). docs/YADV_ROUND_21_PLAN.md sect.5 Stage 1.
    const bool recon_dbg = std::getenv("ACID_RECON") != nullptr;
    // ACID_YADV_RECON (round 21, Phase 3a, RESEARCH-ONLY, default OFF; inert unless ACID_YADV is
    // also active): once per step, before the retry loop, re-derive (p,T,alpha) per cell from the
    // cell's own CONSERVED state (rho, e=hstat-p/rho, Y) via the closed-form NASG p-T-equilibrium
    // inversion (eos.hpp:pT_from_v_e_massfrac) -- holding mass, momentum, and total energy
    // EXACTLY fixed. This removes the alpha-remap lag (dal_remap, round 13 sect.23.1) at its
    // source: after reconciliation, alpha_prev IS alpha_from_mass_fraction(Y,...) by construction,
    // so the Eqs.43-44 rebuild's rho_o/Htot_o below differ from the true old level only by this
    // step's own O(dt) Y-advection, not by an O(1) remap artifact -- docs/YADV_ROUND_21_PLAN.md
    // sect.2.6 derives why this makes r_init dt-independent again.
    // Exact-skip (bit test, no tolerance, sect.2.4): a cell whose stored alpha already IS
    // alpha_from_mass_fraction(Y, rho_a(p,T), rho_b(p,T)) is left untouched -- bit-exact for
    // every pure cell (Y in {0,1}, verified: eos.hpp's alpha_from_mass_fraction is bit-exact at
    // both ends) and every undisturbed cell, so this is automatically local to the region where
    // the lag actually exists and free elsewhere (case01, single-phase cases: zero cells touched).
    // Fail-safe: any cell pT_from_v_e_massfrac rejects (inadmissible input, non-finite, or the
    // recovered T would sit outside T_from_hstat's own (1e-6,1e6) range) is left COMPLETELY
    // untouched -- no fallback, no clamp-and-continue.
    // Does NOT touch compute_R: runs once per step, BEFORE the s0 snapshot below, so every
    // retry's `s = s0` restores the reconciled state identically, and it is a pure function of
    // the current (s.p,s.T,s.alpha,s.rho,s.hstat,s.u,Yv,A,B) -- no call history. Round 17's
    // invariant (an approximate/frozen Jacobian changes only iteration count, never the converged
    // answer) is unaffected by construction; this changes the state the step STARTS from, exactly
    // like dt selection does, not the residual or the Jacobian.
    // Must NOT be combined with ACID_YADV_HREINIT: HREINIT would overwrite s.h with the
    // (pre-reconciliation-consistent) Htot_o after this already ran -- redundant at best,
    // confounding at worst. docs/YADV_ROUND_21_PLAN.md sect.2-3.
    const bool yrecon = std::getenv("ACID_YADV_RECON") != nullptr;
    // ACID_RESYNC (round 22, DIAGNOSTIC ONLY, default OFF, stderr only, applies nothing): measures
    // what ACID_YADV_RESYNC would write to Yv without writing it -- worst |Ynew-Yv[i]| + cell
    // index, count of bitwise-changed cells, and the phase-A mass drift this step would cost
    // (dM = sum_i rho_i*(Ynew_i-Yv_i)*dx), running total, and running/initial ratio. Reuses
    // ACID_BLK_STEP for step selection. docs/YADV_ROUND_22_PLAN.md sect.5 Stage 1a.
    const bool resync_dbg = std::getenv("ACID_RESYNC") != nullptr;
    // ACID_YADV_RESYNC (round 22, Phase 3a, RESEARCH-ONLY, default OFF; inert unless ACID_YADV is
    // also active): the DUAL projection to round 21's ACID_YADV_RECON. Once per step, before the
    // retry loop, re-derive Yv (and ONLY Yv -- no `s.*` field is ever written) from the CURRENT
    // (p,T,alpha) via mass_fraction_from_alpha -- the exact expression the once-only IC init above
    // (Yv(n,0.0) block) uses at step 0. This removes the same alpha-remap lag RECON removes
    // (docs/YADV_ROUND_21_PLAN.md sect.2.6): after resync, al_o (recovered below at (p_o,T_o) from
    // the resynced Yv) reproduces s0.alpha to the round-trip conditioning floor, so the Eqs.43-44
    // rebuild's rho_o/Htot_o differ from the true old level only by this step's own O(dt)
    // Y-advection, exactly as RECON's own mechanism argument -- applied here from the OTHER side.
    // UNLIKE RECON, this writes NO state field (p, T, alpha, rho, hstat, h all bit-unchanged) --
    // round 22 diagnosed RECON's case13/14 regression as a state-level perturbation at the
    // T-jump-at-constant-p contact (the Abgrall 1996 spurious-pressure-oscillation mechanism,
    // measured on case14; case13's crossing criterion is shock_location_ok, a different and not
    // fully attributed symptom -- docs/YADV_RESEARCH.md sect.32.1) that a projection writing no
    // state field cannot produce BY CONSTRUCTION, independent of which exact mechanism is
    // responsible. Honest cost: rho*Y is no longer carried exactly across step boundaries (phase-
    // mass drift, measured by ACID_RESYNC, docs/YADV_ROUND_22_PLAN.md sect.5 Stage 1a) -- the same
    // class of compromise the published 19/19 OFF path already makes (it transports alpha, also
    // not a strict material invariant).
    // Step 0 is a BIT-LEVEL no-op: at step==0, Yv was set by the identical expression at the
    // identical (p,T,alpha) a few dozen lines above, so Ynew == Yv[i] bitwise there.
    // Does NOT touch compute_R (same argument as RECON, docs/YADV_ROUND_21_PLAN.md sect.3): runs
    // once per step, BEFORE the s0 snapshot below, so every retry's `s = s0` restores the resynced
    // Yv identically; pure function of the current state, no call history.
    // Must NOT be combined with ACID_YADV_RECON (the two are opposite-direction projections of the
    // same consistency condition -- applying both is meaningless, not merely redundant) nor with
    // ACID_YADV_HREINIT (same exclusion rationale as RECON's). Skipped with a one-line stderr
    // notice if ACID_YADV_RECON is also set. docs/YADV_ROUND_22_PLAN.md sect.3.3.
    const bool yresync = std::getenv("ACID_YADV_RESYNC") != nullptr;
    // ACID_PROJ_UNTIL (round 23, DIAGNOSTIC SWEEP PARAMETER, default unset = always-apply,
    // byte-identical to the pre-round-23 build): caps ACID_YADV_RECON/ACID_YADV_RESYNC's WRITE to
    // steps < N (0/unset -> negative -> "always apply", matching every prior round's behaviour
    // exactly). Same category as ACID_BLK_STEP/ACID_TEND_SCALE -- a diagnostic sweep knob, never
    // set in a validation run, structurally (not numerically) parameterised, no physics. Exists to
    // separate two competing explanations for case24's differing stall-step gain under RECON vs
    // RESYNC (docs/YADV_ROUND_23_PLAN.md sect.3/6): H-A (the state write prevents round 16
    // sect.26.1's density collapse -- predicts a monotone, near-affine dose-response in N) vs H-B
    // (the gain is a Newton-trajectory/basin-of-attraction sensitivity to ANY perturbation,
    // however small -- predicts the stall step is roughly independent of how long the projection
    // is applied, since round 22 sect.32.1 already found a state write can shift which discrete
    // admissible state a bounded Newton sweep converges near). N=1 = apply at step 0 only, where
    // the IC is already a p-T-equilibrium state so the write is an identity to roundoff
    // (docs/YADV_ROUND_23_PLAN.md sect.2/3.1) -- the roundoff-null control. Never applies to
    // ACID_RECON/ACID_RESYNC's own diagnostic-only measurement (those read state, they don't gate
    // on this).
    const int proj_until = []{ const char* e = std::getenv("ACID_PROJ_UNTIL");
                               return e ? std::atoi(e) : -1; }();
    // ACID_RECON_NULL (round 24, RESEARCH-ONLY, default OFF; inert unless ACID_YADV_RECON is also
    // set): the roundoff-null control round 23's ACID_PROJ_UNTIL=1 was BELIEVED to be but is not
    // -- direct measurement (docs/YADV_ROUND_24_PLAN.md sect.0/F1) showed the exact-skip at this
    // block's write-gate site fires for ALL cells at step 0 (the IC is already a p-T-equilibrium
    // state), so ACID_PROJ_UNTIL=1 performs ZERO writes for the entire run and is an exact no-op,
    // not a roundoff-scale perturbation -- round 23 sect.33.3's "P6'" test could not have
    // distinguished H-A (state-accuracy) from H-B (Newton-trajectory sensitivity) because nothing
    // was ever applied. This flag restricts the RECON write to cells where the write is BELOW the
    // map's own round-trip conditioning floor (eos.hpp:alpha_roundtrip_floor, the SAME
    // machine-precision bound denner1d_unit.cpp's round-trip test asserts against -- not a new
    // constant): a cell is written only if |dal|<=floor, |dp|<=8*eps*|p|, and |dT|<=8*eps*|T| all
    // hold. This is the complement of the exact-skip at this block's own skip site: it applies
    // exactly where state is consistent to the map's resolution but not bit-exact, which the
    // exact-skip (an EQUALITY test) does not catch. If plain ACID_YADV=1 with this control still
    // stalls at the same step/reason/rbest/r_init, H-B is bounded for the first time; if it moves,
    // H-B is alive and every prior single-realization case24 stall-step number is a noisy sample.
    // docs/YADV_ROUND_24_PLAN.md sect.2.4.
    const bool recon_null = std::getenv("ACID_RECON_NULL") != nullptr;
    // ACID_TEND_SCALE (round 11, Phase 3a Stage 2, DIAGNOSTIC ONLY, default 1.0 = byte-identical
    // when unset): multiplies THIS SOLVER's stop time only. It is an OBSERVATION WINDOW, not a
    // physical or tuning parameter -- the standard shock-tube verification convention is to sample
    // before a wave reaches a boundary, and cases.cpp's fixed t_end = 0.7/Vs_ref breaks that
    // convention whenever the COMPUTED shock speed differs from Vs_ref (cases 24/34 under
    // ACID_YADV_ALPHA_IMPLICIT: the shock has left the 800-cell domain by t_end, so there is no
    // clean post-shock plateau to sample -- YADV_RESEARCH.md sect.20.2).
    // WARNING, by design and not fixable here: cases.cpp builds the reference solution at the
    // UNSCALED c.config.final_time (cases.cpp:760) and denner1d_dump calls it independently, so
    // with scale != 1 the dump's *_ref columns and EVERY denner1d_validate metric are meaningless.
    // NEVER set this for a gate/validation run. Only the solver columns (p,u,rho) are valid.
    const double tend_scale = []{
        const char* e = std::getenv("ACID_TEND_SCALE");
        if (!e) return 1.0;
        const double v = std::atof(e);
        if (!(v > 0.0) || !std::isfinite(v)) {
            std::fprintf(stderr, "ACID_TEND_SCALE=%s invalid (need finite > 0) -> ignored, using 1.0\n", e);
            return 1.0;
        }
        return v;
    }();
    long thinc_hits = 0;  // debug: how many faces ever activated THINC (nonzero => case activates)
    long thinc_rej = 0;   // debug: THINC candidates rejected by the rho-monotonicity BVD guard
    // previous-time advecting face velocity (transient MWI): initialise from the initial
    // velocity field so a uniform flow (e.g. case02 u=1) is preserved on the first step.
    Vec theta_o(n + 1, 0.0);
    {
        const auto ue0 = apply_ghost(s.u, lbc, rbc, 2, true);
        for (int f = 0; f <= n; ++f) theta_o[f] = 0.5 * (ue0[f + 1] + ue0[f + 2]);
        if (lbc == "reflective") theta_o[0] = 0.0;
        if (rbc == "reflective") theta_o[n] = 0.0;
    }

    // ACID_YADV: the transported mass fraction, initialised ONCE from the case's alpha IC at the
    // initial (p,T). The case definitions and reference solutions stay in alpha; only the
    // solver's internal transported variable changes.
    Vec Yv(n, 0.0);
    if (yadv) {
        for (int i = 0; i < n; ++i) {
            const double pu = std::max(s.p[i], 1.0), Tu = std::max(s.T[i], 1e-6);
            Yv[i] = std::clamp(mass_fraction_from_alpha(std::clamp(s.alpha[i], 0.0, 1.0),
                                                        phase_props(pu, Tu, A).rho,
                                                        phase_props(pu, Tu, B).rho),
                               0.0, 1.0);
        }
    }

    // divergence guard: if the CFL time step collapses far below its initial value, a cell's
    // |u|+a has blown up -> the run would otherwise grind for ~1e6 tiny steps. Treat as
    // divergence and abort immediately (the caller's validate then fails the case cleanly).
    double dt0_cfl = -1.0;
    bool diverged = false;
    // Stage 2: the effective stop time. The `== 1.0` early-out makes the unset path textually
    // identical to the pre-change code (multiplying by 1.0 is exact in IEEE-754 anyway, but this
    // makes byte-identity inspectable by reading rather than by FP reasoning).
    const double t_end = (tend_scale == 1.0) ? c.config.final_time
                                             : c.config.final_time * tend_scale;
    if (tend_scale != 1.0)
        std::fprintf(stderr, "TEND_SCALE: case=%s scale=%.6g -> t_end=%.9e (reference is still at "
                     "%.9e -- *_ref columns and all validate metrics are INVALID for this run)\n",
                     c.id.c_str(), tend_scale, t_end, c.config.final_time);
    while (t < t_end && step < c.config.max_steps) {
        // acoustic-CFL dt
        double lam = 1e-300;
        int imax = 0;
        // material CFL (acoustic implicit). The uniform coupled scheme is fully acoustic-implicit
        // (Denner): a material-only dt under the coupled energy diverges (case02 NaN), so the
        // coupled path always uses the acoustic CFL. material_dt stays for the legacy 2x2 path.
        // material CFL is valid (and necessary) for an advection-dominated flow with no acoustic
        // source: the ACID acoustic is implicit, so the large material-CFL step is stable, and the
        // acoustic-CFL step would need ~1e6 sub-steps (case02). Only the coupled (shock) regime
        // forbids it (material-only dt + coupled energy diverges). Same operator, larger stable dt.
        // coupled + material-dt normally diverges (tridiagonal Jacobian, see ACID_PENTA note);
        // with the pentadiagonal solve it is stable, so allow the large material-dt step there.
        const bool mat_dt = auto_material && (!coupled || penta_solve);
        for (int i = 0; i < n; ++i) {
            const double li = std::abs(s.u[i]) + (mat_dt ? 0.0 : s.a[i]);
            if (li > lam) { lam = li; imax = i; }
        }
        if (dbg && lam > 1.0e5) {
            std::fprintf(stderr,
                "BLOWUP src i=%d x=%.4f |u|+a=%.3e u=%.3e a=%.3e p=%.3e T=%.3e al=%.4f rho=%.3e "
                "(nbr u: %.2e %.2e %.2e)\n",
                imax, st.x[imax], lam, s.u[imax], s.a[imax], s.p[imax], s.T[imax], s.alpha[imax],
                s.rho[imax], s.u[std::max(imax - 1, 0)], s.u[imax], s.u[std::min(imax + 1, n - 1)]);
        }
        const double dt_full = c.config.cfl * dx / lam;  // unscaled CFL dt (divergence reference)
        // --- divergence early-stop: the CFL dt (cfl*dx/max(|u|+a)) collapsing >1000x below
        //     its first-step value means a cell blew up; stop now instead of grinding. ---
        if (dt0_cfl < 0.0) {
            dt0_cfl = dt_full;  // characteristic CFL dt from the first step (unscaled)
        } else if (dt_full < 1.0e-3 * dt0_cfl) {
            std::fprintf(stderr,
                "DIVERGED: CFL dt=%.3e collapsed below 1e-3*dt0=%.3e (lam=%.3e) at step %d t=%.3e -> abort\n",
                dt_full, dt0_cfl, lam, step, t);
            diverged = true;
            break;
        }
        double dt = (cfl_ramp ? cfl_scale : 1.0) * dt_full;  // ramp-scaled actual dt
        dt = std::min(dt, t_end - t);
        if (!(dt > 0.0)) break;

        // ---- Round 21 (ACID_YADV_RECON / ACID_RECON): once-per-step conserved-state
        //      reconciliation. See the flag declarations above for the full rationale;
        //      docs/YADV_ROUND_21_PLAN.md sect.2 for the derivation. Runs BEFORE the s0 snapshot
        //      below, so the whole retry sweep sees (and restores, via s=s0) the reconciled
        //      state identically -- compute_R itself is never touched. ----
        if (yadv && (yrecon || recon_dbg)) {
            int ncell = 0, nskip = 0, nrej = 0, ntouch = 0;
            int nnull = 0, nabove = 0;  // round 24: cells within / above the roundtrip floor
            double worst_dp = 0.0, worst_dp_rel = 0.0; int worst_dp_i = -1;
            double worst_dT = 0.0, worst_dT_rel = 0.0; int worst_dT_i = -1;
            double worst_dal = 0.0; int worst_dal_i = -1;
            std::vector<char> touched(n, 0);
            for (int i = 0; i < n; ++i) {
                ++ncell;
                const double pu = std::max(s.p[i], 1.0), Tu = std::max(s.T[i], 1e-6);
                const auto pa = phase_props(pu, Tu, A);
                const auto pb = phase_props(pu, Tu, B);
                const double al_chk = std::clamp(
                    alpha_from_mass_fraction(Yv[i], pa.rho, pb.rho), 0.0, 1.0);
                if (al_chk == s.alpha[i]) { ++nskip; continue; }  // exact skip, sect.2.4
                const double v_t = 1.0 / s.rho[i];
                const double e_t = s.hstat[i] - s.p[i] * v_t;
                const auto r = pT_from_v_e_massfrac(v_t, e_t, Yv[i], A, B);
                if (!r.ok) { ++nrej; continue; }  // fail-safe: cell left completely untouched
                const double dp = r.p - s.p[i], dT = r.T - s.T[i], dal = al_chk - s.alpha[i];
                if (std::abs(dp) > worst_dp) { worst_dp = std::abs(dp); worst_dp_rel = dp / s.p[i]; worst_dp_i = i; }
                if (std::abs(dT) > worst_dT) { worst_dT = std::abs(dT); worst_dT_rel = dT / s.T[i]; worst_dT_i = i; }
                if (std::abs(dal) > worst_dal) { worst_dal = std::abs(dal); worst_dal_i = i; }
                // round 24: is this write within the map's own round-trip conditioning floor?
                // Complement of the exact-skip above (an equality test) -- catches "consistent to
                // the map's resolution but not bit-exact". Same bound denner1d_unit.cpp asserts.
                const bool is_null = std::abs(dal) <= alpha_roundtrip_floor(pa.rho, pb.rho)
                                   && std::abs(dp) <= 8.0 * std::numeric_limits<double>::epsilon() * std::abs(s.p[i])
                                   && std::abs(dT) <= 8.0 * std::numeric_limits<double>::epsilon() * std::abs(s.T[i]);
                if (is_null) ++nnull; else ++nabove;
                if (yrecon && (proj_until < 0 || step < proj_until) && (!recon_null || is_null)) {
                    s.p[i] = r.p;
                    s.T[i] = r.T;
                    const auto ra2 = phase_props(std::max(r.p, 1.0), std::max(r.T, 1e-6), A);
                    const auto rb2 = phase_props(std::max(r.p, 1.0), std::max(r.T, 1e-6), B);
                    s.alpha[i] = std::clamp(
                        alpha_from_mass_fraction(Yv[i], ra2.rho, rb2.rho), 0.0, 1.0);
                    touched[i] = 1;
                    ++ntouch;
                }
            }
            if (ntouch > 0) {
                eval_thermo(s, A, B);  // refresh rho/hstat/cp/a/drhodp; unchanged inputs on
                                        // skipped cells -> bit-identical there.
                for (int i = 0; i < n; ++i)
                    if (touched[i]) s.h[i] = s.hstat[i] + 0.5 * s.u[i] * s.u[i];
            }
            if (recon_dbg) {
                const char* se = std::getenv("ACID_BLK_STEP");
                const int rstep = se ? std::atoi(se) : -1;
                if (rstep < 0 || rstep == step)
                    std::fprintf(stderr,
                        "RECON case=%s step=%d ncell=%d nskip=%d nrej=%d ntouch=%d "
                        "dp=%.4e(rel %.4e)@%d dT=%.4e(rel %.4e)@%d dal=%.4e@%d "
                        "nnull=%d nabove=%d\n",
                        c.id.c_str(), step, ncell, nskip, nrej, ntouch,
                        worst_dp, worst_dp_rel, worst_dp_i, worst_dT, worst_dT_rel, worst_dT_i,
                        worst_dal, worst_dal_i, nnull, nabove);
            }
        }

        // ---- Round 22 (ACID_YADV_RESYNC / ACID_RESYNC): the dual projection to RECON above --
        //      re-derives Yv ONLY (no `s.*` field is written) from the CURRENT (p,T,alpha), via
        //      the exact expression the once-only IC init (Vec Yv(n,0.0) block, above) uses at
        //      step 0. See the flag declaration above for the full rationale;
        //      docs/YADV_ROUND_22_PLAN.md sect.3.3 for the derivation. Runs BEFORE the s0 snapshot
        //      below, so the whole retry sweep sees (and restores, via Yv=Yv0) the resynced Yv
        //      identically -- compute_R itself is never touched. Mutually exclusive with
        //      ACID_YADV_RECON (opposite-direction projections of the same consistency
        //      condition). ----
        if (yadv && yrecon && yresync) {
            static bool warned = false;
            if (!warned) {
                std::fprintf(stderr,
                    "ACID_YADV_RESYNC: skipped -- ACID_YADV_RECON is also set (opposite-direction "
                    "projections, mutually exclusive, docs/YADV_ROUND_22_PLAN.md sect.5 Stage 1b)\n");
                warned = true;
            }
        } else if (yadv && (yresync || resync_dbg)) {
            int ncell = 0, ntouch = 0;
            double worst_dY = 0.0; int worst_dY_i = -1;
            double dM_step = 0.0;
            for (int i = 0; i < n; ++i) {
                ++ncell;
                const double pu = std::max(s.p[i], 1.0), Tu = std::max(s.T[i], 1e-6);
                const double Ynew = std::clamp(
                    mass_fraction_from_alpha(std::clamp(s.alpha[i], 0.0, 1.0),
                                             phase_props(pu, Tu, A).rho,
                                             phase_props(pu, Tu, B).rho), 0.0, 1.0);
                if (!std::isfinite(Ynew)) continue;  // fail-safe: cell left completely untouched
                const double dY = Ynew - Yv[i];
                if (dY == 0.0) continue;
                if (std::abs(dY) > worst_dY) { worst_dY = std::abs(dY); worst_dY_i = i; }
                dM_step += s.rho[i] * dY * dx;
                if (yresync && (proj_until < 0 || step < proj_until)) { Yv[i] = Ynew; ++ntouch; }
            }
            if (resync_dbg) {
                static double dM_total = 0.0, M0 = -1.0;
                if (M0 < 0.0) {
                    M0 = 0.0;
                    for (int i = 0; i < n; ++i) M0 += s.rho[i] * Yv[i] * dx;
                    M0 = std::max(std::abs(M0), 1e-300);
                }
                dM_total += dM_step;
                const char* se = std::getenv("ACID_BLK_STEP");
                const int rstep = se ? std::atoi(se) : -1;
                if (rstep < 0 || rstep == step)
                    std::fprintf(stderr,
                        "RESYNC case=%s step=%d ncell=%d ntouch=%d worst_dY=%.4e@%d "
                        "dM_step=%.6e dM_total=%.6e dM_total/M0=%.6e\n",
                        c.id.c_str(), step, ncell, ntouch, worst_dY, worst_dY_i,
                        dM_step, dM_total, dM_total / M0);
            }
        }

        // ---- adaptive dt with retry: if the implicit step diverges (non-finite, or a cell
        //      blows past 10*uref), restore the state, halve dt, and redo. Lets the violent
        //      interface/shock cases (07,25) take a smaller first step instead of NaN-ing. ----
        const Field s0 = s;
        const Vec Yv0 = Yv;  // ACID_YADV: the transported Y is outside Field -> restore it too
        bool stepped = false;
        // Stage 1 (round 11, DIAGNOSTIC ONLY): carry the last retry's failure reason out of the
        // retry loop so the stall report below can name it. Ints/doubles only -- no FP arithmetic
        // is added, so every accepted step is bit-identical to the pre-change build.
        int  stall_reason = 0;    // 1=Newton made no progress, 2=non-finite p, 3=non-finite u,
                                  // 4=|u|>10*uref, 5=a cell pinned at the 1e6 K T ceiling
                                  //   (F2'', round 18/20; unconditional since round 20 -- see the
                                  //   mechanism below)
        int  stall_cell   = -1;   // first offending cell for reasons 2-5
        double stall_dt   = 0.0;  // the last dt actually attempted (dt is halved after the check)
        int  stall_retry  = -1;
        bool stall_conv_inner = false;  // rbest/r_init/conv_inner are retry-loop-local; captured
        double stall_rbest = 0.0;       // out here so the STALLED-DETAIL report (after the loop
        double stall_rinit = -1.0;      // closes) can still read the last retry's Newton state.
        // Stage 3a (round 12): best-across-retries accept candidate for THIS step (see sect.3.3
        // for how it's ranked/updated, sect.3.5 for how it's consumed). Guarded by
        // stall_accept_lvl > 0 at every use site, so this block costs nothing at the default (0).
        bool   acc_have  = false;
        Field  acc_s;
        Vec    acc_Yv;
        double acc_dt    = 0.0, acc_ratio = 0.0, acc_rbest = 0.0, acc_rinit = 0.0;
        int    acc_retry = -1;
        bool   only_reason1 = true;   // level 2: did EVERY failed retry of this step fail on reason 1?
        for (int retry = 0; retry < 14; ++retry) {
        s = s0;
        Yv = Yv0;

        // old (previous time-level) flow state
        const Vec u_o = s.u, p_o = s.p, T_o = s.T;
        Vec uu_o = s.u;  // OLD cell velocity for the MWI memory term (advanced to the stage level under TR-BDF2)

        // ---- VOF colour-function advection (Eq.32), K=0 (Allaire/PE) ----
        //   d(alpha)/dt + d(alpha*theta)/dx - (alpha+K) du/dx = 0  (upwind alpha)
        {
            // ACID_YADV: the colour function that is actually transported. OFF -> alpha (the
            // default, byte-identical path); ON -> the mass fraction Y. BOTH are bounded [0,1]
            // colour functions that are constant in each pure phase, so the THINC tanh
            // reconstruction, its interface indicator and the update stencil below apply to
            // either one unchanged -- only the recovery step at the end of the block differs.
            const Vec& cvar = yadv ? Yv : s.alpha;
            const auto ae = apply_ghost(cvar, lbc, rbc, 2, false);
            const auto ueo = apply_ghost(u_o, lbc, rbc, 2, true);
            // THINC (tanh) reconstruction of the UPWIND interface cell, CONSERVATIVE
            // SEMI-LAGRANGIAN flux form: the face alpha is the tanh profile AVERAGED over the
            // dt departure region swept through the face (NOT the point face value -- the point
            // value under-transports the sub-cell interface at finite CFL and lags the front,
            // measured -30 cells on case02).
            //   al(xi) = 0.5*(1 + sigma*tanh(beta*(xi - xi_c))),  xi in [0,1] across the cell,
            //   sigma = sign(al_{i+1} - al_{i-1}),  al_i (cell average) fixes xi_c.
            // Departure region: right-fed face (theta>=0, upwind = left cell) xi in [1-c,1];
            // left-fed face (theta<0, upwind = right cell) xi in [0,c]; c = |theta_f|*dt/dx.
            // Closed form (no atanh, no overflow -- indicator keeps al_i in (1e-6,1-1e-6)):
            //   B = exp(sigma*beta*(2*al_i-1));  D = (B-e^-beta)/(e^beta-B)  (= e^{-2 beta xi_c});
            //   avg over [a,b] = 0.5 + sigma/(2 beta (b-a)) * ln[(e^{beta b} D + e^{-beta b}) /
            //                                                    (e^{beta a} D + e^{-beta a})].
            // Verified vs brute force (numeric root-find for xi_c + quadrature) to 4e-15 over
            // both orientations x al_i in [1e-5,1-1e-5] x c in [0,1]; the c->0 limit is the
            // point face value (explicit fallback below); c=1 gives exactly the cell average.
            constexpr double beta = 3.5;
            const double ebv = std::exp(beta), embv = std::exp(-beta);
            const double cbv = std::cosh(beta), sbv = std::sinh(beta);
            auto thinc_flux_alpha = [&](int g, bool right, double cfl_loc) -> double {
                const double alm = ae[g - 1], ali = ae[g], alp = ae[g + 1];  // al_{i-1,i,i+1}
                // interface-cell indicator: genuine material interface, monotone, unsaturated.
                const bool straddle = std::min(alm, alp) < 0.5 && 0.5 < std::max(alm, alp);
                const bool steep    = std::abs(alp - alm) > 0.5;
                const bool monotone = (alp - ali) * (ali - alm) > 0.0;
                const bool unsat    = ali > 1.0e-6 && ali < 1.0 - 1.0e-6;
                if (!(straddle && steep && monotone && unsat)) return -1.0;  // -> plain upwind
                const double sigma = alp > alm ? 1.0 : -1.0;
                const double B = std::exp(sigma * beta * (2.0 * ali - 1.0));
                const double cc = std::clamp(cfl_loc, 0.0, 1.0);
                if (cc < 1.0e-8)  // stagnant face: limit of the average = point face value
                    return right ? 0.5 * (1.0 + sigma * (B * cbv - 1.0) / (B * sbv))
                                 : 0.5 * (1.0 - sigma * (cbv - B) / sbv);
                const double D = (B - embv) / (ebv - B);  // e^{-2 beta xi_c}, > 0 under indicator
                double lr;
                if (right) {  // departure region [1-cc, 1]
                    const double ea = std::exp(beta * (1.0 - cc));
                    lr = std::log((ebv * D + embv) / (ea * D + 1.0 / ea));
                } else {      // departure region [0, cc]
                    const double eb2 = std::exp(beta * cc);
                    lr = std::log((eb2 * D + 1.0 / eb2) / (D + 1.0));
                }
                return 0.5 + sigma * lr / (2.0 * beta * cc);
            };
            Vec thf(n + 1), af(n + 1);
            for (int f = 0; f <= n; ++f) {
                const int gL = f + 1, gR = f + 2;
                thf[f] = 0.5 * (ueo[gL] + ueo[gR]);
                const bool pos = thf[f] >= 0.0;
                double a_up = pos ? ae[gL] : ae[gR];  // 1st-order upwind (default / THINC OFF)
                if (thinc) {
                    const int g = pos ? gL : gR;      // ghost-array index of the UPWIND cell
                    const int icell = g - 2;          // its real-cell index
                    if (icell >= 0 && icell < n) {    // skip domain-boundary (ghost upwind) faces
                        const double at = thinc_flux_alpha(g, pos, std::abs(thf[f]) * dt / dx);
                        if (at >= 0.0) {              // interface cell -> candidate sharpening
                            // BVD-style boundedness: clamp into the two cells straddling the face.
                            const double cand =
                                std::clamp(at, std::min(ae[gL], ae[gR]), std::max(ae[gL], ae[gR]));
                            // ---- rho-monotonicity BVD guard (parameter-free, DEFAULT) ----
                            // The mixture density this face alpha IMPLIES at the upwind cell's
                            // (p,T) -- the same EOS blend that couples alpha to rho -- must lie
                            // within the two adjacent CELL mixture densities; otherwise the
                            // THINC flux drives a rho slope-reversal (sharp alpha x smeared T)
                            // and this face falls back to plain upwind. Bounds are neighbour
                            // values: zero new constants, case-blind. Measured: eliminates the
                            // case14 contact-band oscillation at all N (TV-excess 44.6% ->
                            // 0.69% of the jump; N=800: 0.37%) and cleans case25's interface
                            // ~100x (band ip 0.0121 -> 0.0001, wave positions 8/1/9 -> 0/0/1
                            // cells). ACCEPTED COST (Advisor-approved): case02 corr_rho 0.9999
                            // -> 0.9971 with a 1-cell front offset (82 endpoint-class rejects,
                            // ~1-ulp blend-vs-cell-rho mismatch at a uniform-(p,T) contact) --
                            // still far above the 0.90 gate and the 0.980 THINC-OFF baseline;
                            // and case14 l2_rho 0.031 -> 0.038, the HONEST monotone value (the
                            // former 0.031 was flattered by oscillation aliasing). The case14
                            // signal (~50% blend mismatch) and the case02 noise (~1 ulp) live
                            // on the SAME endpoint-clamped instance class, so no constant-free
                            // split exists (measured: endpoint exemption / rho-clamp forms
                            // restore case02 but bring the oscillation back) -- see
                            // docs/THINC_RHO_GUARD_RESEARCH.md before changing this.
                            const double pu = std::max(s.p[icell], 1.0);
                            const double Tu = std::max(s.T[icell], 1e-6);
                            const double ra = phase_props(pu, Tu, A).rho;
                            const double rb = phase_props(pu, Tu, B).rho;
                            // ACID_YADV: the guard is a DENSITY test, so a candidate face value
                            // of Y is first mapped to the face alpha it implies at that same
                            // upwind (p,T) via the explicit inverse; with the switch OFF the
                            // candidate already IS an alpha and the expression is unchanged.
                            const double cand_a =
                                yadv ? alpha_from_mass_fraction(cand, ra, rb) : cand;
                            const double rho_imp = cand_a * ra + (1.0 - cand_a) * rb;
                            const double r1 = s.rho[std::clamp(f - 1, 0, n - 1)];
                            const double r2 = s.rho[std::clamp(f, 0, n - 1)];
                            if (rho_imp >= std::min(r1, r2) && rho_imp <= std::max(r1, r2)) {
                                a_up = cand;
                                if (thinc_dbg) ++thinc_hits;
                            } else if (thinc_dbg) {
                                ++thinc_rej;  // rho reversal -> plain upwind at this face
                            }
                        }
                    }
                }
                af[f] = a_up;
            }
            if (lbc == "reflective") thf[0] = 0.0;
            if (rbc == "reflective") thf[n] = 0.0;
            Vec anew(n);
            if (!yadv) {
                for (int i = 0; i < n; ++i) {
                    const double flux = thf[i + 1] * af[i + 1] - thf[i] * af[i];
                    const double divu = (thf[i + 1] - thf[i]) / dx;
                    anew[i] = std::clamp(cvar[i] - dt / dx * flux + dt * cvar[i] * divu, 0.0, 1.0);
                }
                s.alpha = anew;
            } else {
                // ---- ACID_YADV round 3: CONSERVATIVE rho*Y transport ----
                // Round 1/2 advected Y with the alpha stencil
                //   c - dt/dx*(thf*cf)|_L^R + dt*c*div(theta),
                // i.e. the non-conservative advective form. That form is the update for a
                // quantity whose CELL AVERAGE is a VOLUME average (true for alpha). Y's cell
                // average is a MASS average, so in a cut cell the cell state and the stencil
                // live in different spaces and no face map can reconcile them (measured three
                // ways in docs/YADV_RESEARCH.md sect.10.4). The repair is to make the CONSERVED
                // variable rho*Y and discretise its own conservation law
                //   d(rho Y)/dt + d(rho Y u)/dx = 0,
                // which is exact for a no-phase-change mixture (sect.1.3) and puts cell state
                // and flux in the same space.
                //
                // The face mass flux mirrors ACID Eqs.41-42 exactly (the PER-CELL construction
                // used by the implicit mass/momentum/energy residual further below), evaluated
                // at the OLD time level: cell i's own outflow/inflow uses cell i's OWN alpha as
                // the phase blend weight, with the phase densities taken at the face's UPWIND
                // cell (p,T). That asymmetric-but-locally-consistent form is what makes a
                // uniform-velocity contact produce no spurious source.
                //
                // DOCUMENTED APPROXIMATION (the one place this update is not exact). The
                // conserved quantity is assembled with the OLD accepted-step mixture density
                // s.rho[], which is the correct old level; but recovering Y needs the NEW
                // density, and the true new density is not defined until s.alpha has been
                // re-derived (below) and the implicit step has run. What is used instead is
                // rho_star = rho_old - dt/dx*(mdotR_o - mdotL_o), i.e. the density predicted by
                // the SAME old-level mass flux (a discrete continuity predictor). This is
                // explicit -- it introduces no circularity and is NOT the alpha-inside-the-
                // Newton problem (YADV_RESEARCH sect.12 task (b), still out of scope) -- but it
                // is a lagged/explicit stand-in for the density the implicit step will actually
                // produce, so the recovered Y is O(dt) inconsistent with the final state.
                //
                // Why rho_star and not rho_old (MEASURED, do not "simplify" this back).
                // Dividing by rho_old breaks the pure-cell property: with Y == 1 everywhere the
                // numerator is rho_old - dt/dx*(mdotR_o - mdotL_o) = rho_star, so dividing by
                // rho_old returns 1 - dt/(dx*rho_old)*div(mdot), i.e. a compressed or expanded
                // SINGLE-PHASE cell spontaneously grows the other phase. Measured on this
                // workspace: suite 13/19 with rho_old vs 15/19 with rho_star; case13 l2_rho
                // 0.1219 vs 0.02268 and max|d alpha| vs the alpha path 0.998 vs 0.070; case30
                // FAILS with rho_old (l2_rho 0.1243 vs 0.00922). With rho_star, Y == const is
                // preserved exactly for any velocity field, which is the discrete consistency
                // condition a conservative colour-function flux must satisfy. The rho_old form
                // is kept reachable for reproduction behind the default-OFF diagnostic env
                // ACID_YADV_RHOOLD (inert unless ACID_YADV is also set; the published path is
                // untouched either way).
                //
                // alpha stays a DERIVED quantity, recovered from the NEW Y at the OLD (p,T)
                // right before the ACID old-level rho_o/h_o re-evaluation, so the two use the
                // same (alpha, p_o, T_o) triple -- unchanged from rounds 1/2.

                // (1) OLD-level cell alpha implied by the transported Y at (p_o,T_o). Needed
                //     here (before the flux) purely as the ACID mass-flux blend weight; it is
                //     NOT written to s.alpha (that happens below, from the NEW Y).
                Vec al_o(n);
                for (int i = 0; i < n; ++i) {
                    const double pu = std::max(p_o[i], 1.0), Tu = std::max(T_o[i], 1e-6);
                    al_o[i] = std::clamp(alpha_from_mass_fraction(Yv[i],
                                                                  phase_props(pu, Tu, A).rho,
                                                                  phase_props(pu, Tu, B).rho),
                                         0.0, 1.0);
                }
                // (2) OLD-level UPWIND phase densities at EVERY face (the mass flux needs them
                //     everywhere, not only where THINC activates). Upwind selection by the sign
                //     of thf[f], mirroring the af[] loop above; boundary faces read the ghost
                //     (p,T), which is what apply_ghost already defines for every BC this solver
                //     supports (transmissive/inlet copy the end cell, reflective mirrors the
                //     scalar, periodic wraps) -- no new boundary behaviour is invented.
                const auto peo = apply_ghost(p_o, lbc, rbc, 2, false);
                const auto Teo = apply_ghost(T_o, lbc, rbc, 2, false);
                Vec ra_o(n + 1), rb_o(n + 1);
                for (int f = 0; f <= n; ++f) {
                    const int g = (thf[f] >= 0.0) ? (f + 1) : (f + 2);  // upwind ghost index
                    const double pu = std::max(peo[g], 1.0), Tu = std::max(Teo[g], 1e-6);
                    ra_o[f] = phase_props(pu, Tu, A).rho;
                    rb_o[f] = phase_props(pu, Tu, B).rho;
                }
                // (3) OLD-level per-cell face mass flux (ACID Eqs.41-42 structure, with the
                //     OLD-level alpha / phase densities / face velocity substituted for the
                //     Newton-level ones). thf[] already carries the reflective zeroing applied
                //     above; the explicit zeroing here mirrors the Eqs.41-42 block verbatim.
                Vec mdR_o(n), mdL_o(n);
                for (int i = 0; i < n; ++i) {
                    mdR_o[i] = (al_o[i] * ra_o[i + 1] + (1.0 - al_o[i]) * rb_o[i + 1]) * thf[i + 1];
                    mdL_o[i] = (al_o[i] * ra_o[i]     + (1.0 - al_o[i]) * rb_o[i])     * thf[i];
                }
                if (lbc == "reflective") mdL_o[0] = 0.0;
                if (rbc == "reflective") mdR_o[n - 1] = 0.0;
                // (4) conservative update of rho*Y, then back to Y (see the approximation note).
                //     af[] is the SAME THINC/upwind-reconstructed face Y built above.
                for (int i = 0; i < n; ++i) {
                    const double rho_old = std::max(s.rho[i], 1e-300);
                    const double rY = rho_old * Yv[i]
                                    - dt / dx * (mdR_o[i] * af[i + 1] - mdL_o[i] * af[i]);
                    const double rho_star =
                        yadv_rhoold ? rho_old
                                    : std::max(rho_old - dt / dx * (mdR_o[i] - mdL_o[i]), 1e-300);
                    anew[i] = std::clamp(rY / rho_star, 0.0, 1.0);
                }
                Yv = anew;
                for (int i = 0; i < n; ++i) {
                    const double pu = std::max(p_o[i], 1.0), Tu = std::max(T_o[i], 1e-6);
                    s.alpha[i] = std::clamp(alpha_from_mass_fraction(Yv[i],
                                                                     phase_props(pu, Tu, A).rho,
                                                                     phase_props(pu, Tu, B).rho),
                                            0.0, 1.0);
                }
            }
        }

        // ---- ACID old-level density/enthalpy (Eqs.43-44): re-evaluate with the NEW
        //      alpha at the OLD (p,T). Without this a moved interface injects a spurious
        //      (rho_new - rho_old)/dt source in the continuity. ----
        Vec rho_o(n), hstat_o(n), Htot_o(n);
        for (int i = 0; i < n; ++i) {
            const double al = std::clamp(s.alpha[i], 0.0, 1.0);
            const auto pa = phase_props(std::max(p_o[i], 1.0), std::max(T_o[i], 1e-6), A);
            const auto pb = phase_props(std::max(p_o[i], 1.0), std::max(T_o[i], 1e-6), B);
            rho_o[i] = std::max(al * pa.rho + (1.0 - al) * pb.rho, 1e-300);
            hstat_o[i] = (al * pa.rho * pa.h + (1.0 - al) * pb.rho * pb.h) / rho_o[i];
            Htot_o[i] = hstat_o[i] + 0.5 * u_o[i] * u_o[i];
        }

        // RMISM (round 13 Stage 0, docs/YADV_ROUND_13_PLAN.md sect.1): candidate dt-independent
        // state mismatches, evaluated here (not inside the Newton loop) because nothing between
        // the retry restart and here writes s.h/s.rho/s.alpha's non-recovery fields -- so these
        // are exactly the it==0 quantities without waiting for the first compute_R(). REMAP is the
        // alpha jump caused by the recovery meeting the LAST step's (p_o,T_o) with a frozen Y
        // (predicted dt-independent); ADVECTION is this retry's own Y-transport (predicted O(dt)).
        if (rinit_dbg && yadv) {
            const char* se = std::getenv("ACID_BLK_STEP");
            const int blkstep = se ? std::atoi(se) : -1;
            if (blkstep < 0 || step == blkstep) {
                double dh = 0, drho = 0, dal = 0, dal_remap = 0, dal_adv = 0;
                int ih = -1, irho = -1, ial = -1, iremap = -1, iadv = -1;
                for (int i = 0; i < n; ++i) {
                    const double vh = std::abs(s.h[i] - Htot_o[i]);
                    const double vrho = std::abs(s.rho[i] - rho_o[i]);
                    const double vdal = std::abs(s.alpha[i] - s0.alpha[i]);
                    const double pu = std::max(p_o[i], 1.0), Tu = std::max(T_o[i], 1e-6);
                    const double al_remap_state = std::clamp(
                        alpha_from_mass_fraction(Yv0[i], phase_props(pu, Tu, A).rho,
                                                  phase_props(pu, Tu, B).rho), 0.0, 1.0);
                    const double vremap = std::abs(al_remap_state - s0.alpha[i]);
                    const double vadv = std::abs(s.alpha[i] - al_remap_state);
                    if (vh > dh) { dh = vh; ih = i; }
                    if (vrho > drho) { drho = vrho; irho = i; }
                    if (vdal > dal) { dal = vdal; ial = i; }
                    if (vremap > dal_remap) { dal_remap = vremap; iremap = i; }
                    if (vadv > dal_adv) { dal_adv = vadv; iadv = i; }
                }
                std::fprintf(stderr,
                    "RMISM case=%s step=%d retry=%d dt=%.6e dh=%.4e@%d drho=%.4e@%d dal=%.4e@%d "
                    "dal_remap=%.4e@%d dal_adv=%.4e@%d\n",
                    c.id.c_str(), step, retry, dt, dh, ih, drho, irho, dal, ial,
                    dal_remap, iremap, dal_adv, iadv);
            }
        }

        // RCELL (round 16, docs/YADV_ROUND_16_PLAN.md sect.2): read-only per-cell window dump,
        // right after the Eqs.43-44 rebuild (so rho_o/hstat_o/Htot_o are live) and before Stage 1
        // HREINIT (which overwrites s.h below) -- s.h/s.rho/s.alpha here are still the natural
        // it==0 values. No new computation; answers round 15's open question (why is dh so large
        // at cell ~79-81 for case33) by exposing the raw state, not another derived mismatch.
        if (rcell_lo >= 0 && yadv) {
            const char* se = std::getenv("ACID_BLK_STEP");
            const int blkstep = se ? std::atoi(se) : -1;
            if (blkstep < 0 || step == blkstep) {
                const int lo = std::max(0, rcell_lo), hi = std::min(n - 1, rcell_hi);
                for (int i = lo; i <= hi; ++i) {
                    std::fprintf(stderr,
                        "RCELL case=%s step=%d retry=%d dt=%.6e i=%d x=%.6f Y0=%.6e Y=%.6e "
                        "al0=%.6f al=%.6f p_o=%.6e T_o=%.6e u_o=%.6e h=%.6e Htot_o=%.6e "
                        "rho=%.6e rho_o=%.6e\n",
                        c.id.c_str(), step, retry, dt, i, st.x[i], Yv0[i], Yv[i],
                        s0.alpha[i], s.alpha[i], p_o[i], T_o[i], u_o[i],
                        s.h[i], Htot_o[i], s.rho[i], rho_o[i]);
                }
            }
        }

        // Stage 1 (round 13, docs/YADV_ROUND_13_PLAN.md sect.3): consistency re-init of the
        // Newton INITIAL GUESS. s.h still holds the previous step's converged enthalpy, which is
        // consistent with the OLD alpha, not the alpha just recovered above -- reset it to the
        // freshly-rebuilt Htot_o so the it==0 transient this mismatch would otherwise inject
        // vanishes identically. Only the initial guess changes; compute_R (the fixed point) does
        // not, so no conservation/RH property can move. h is a Newton unknown only when coupled.
        if (yadv && hreinit && coupled) {
            for (int i = 0; i < n; ++i) {
                const double hfloor = 0.5 * u_o[i] * u_o[i] * 1.0001 + 1.0;  // reuse the existing
                s.h[i] = std::max(Htot_o[i], hfloor);                       // line-search kinetic
            }                                                               // floor, no new const
        }

        // ---- BDF2 transient coefficients. Active only when bdf2 is set, the o2 store is
        //      populated (>= 2 accepted steps), AND dt is constant (constant-step BDF2). The
        //      transient of a conserved quantity phi is written (bdf_c0*phi - Cold)*VdT:
        //        BE  : bdf_c0 = 1.0,  Cold_phi = phi_o
        //        BDF2: bdf_c0 = 1.5,  Cold_phi = 2*phi_o - 0.5*phi_o2
        //      Cold per equation is precomputed from the level-n (rho_o,u_o,Htot_o) and the
        //      stored level-(n-1) (mom_o2,rho_o2,ene_o2). ----
        const bool dt_const = have_o2 && dt_prev > 0.0 &&
                              std::abs(dt - dt_prev) <= 1.0e-10 * dt_prev;
        const bool use_bdf2 = bdf2 && dt_const;
        // BDF2 with a per-cell SHOCK SENSOR: 2nd-order BDF2 in smooth regions, but revert to
        // 1st-order BE locally where the OLD-level pressure jumps sharply across the cell (a
        // shock), since BDF2's 2nd-order time stencil oscillates at shocks. This lets a SINGLE
        // uniform scheme (coupled + BDF2 + Minmod) stay stable on the strong-shock cases.
        Vec bdf_c0(n);
        Vec Cold_mom(n), Cold_con(n), Cold_ene(n);
        for (int i = 0; i < n; ++i) {
            const double mom_o = rho_o[i] * u_o[i];
            const double con_o = rho_o[i];
            const double ene_o = rho_o[i] * Htot_o[i];
            bool cell_bdf2 = use_bdf2;
            if (use_bdf2) {
                const int im = std::max(i - 1, 0), ip = std::min(i + 1, n - 1);
                // The 2nd-order BDF2 time stencil oscillates wherever the solution is locally
                // NON-smooth in time, i.e. a discontinuity is sweeping through the cell. Three
                // physical detectors (all read the OLD-level solution, not the case id) revert
                // that cell to 1st-order Backward Euler:
                //   (a) pressure jump  -> a shock          (case13/14/24/25);
                //   (b) density  jump  -> a moving contact (case02 gas-gas contact);
                //   (c) velocity divergence -> expansion/cavitation (case15 tension).
                const double pmax = std::max({p_o[im], p_o[i], p_o[ip]});
                const double pmin = std::min({p_o[im], p_o[i], p_o[ip]});
                const double rmax = std::max({rho_o[im], rho_o[i], rho_o[ip]});
                const double rmin = std::min({rho_o[im], rho_o[i], rho_o[ip]});
                const double du = std::abs(u_o[ip] - u_o[im]);
                // (d) temporal density change: a moving contact sweeping the cell makes rho(t)
                //     jump at this fixed point, so BDF2's 2nd-order TIME stencil (it assumes a
                //     smooth rho history) oscillates in the contact's wake (case02). Acoustic
                //     flows change rho < 1% per step -> never trip it.
                const bool rho_unsteady =
                    have_o2 && std::abs(rho_o[i] - rho_o2[i]) > 0.1 * std::max(rho_o[i], 1e-300);
                if (pmax > 1.3 * std::max(pmin, 1.0) ||      // (a) shock
                    rmax > 1.3 * std::max(rmin, 1e-300) ||   // (b) contact (spatial)
                    du > 0.02 * std::max(s.a[i], 1.0) ||     // (c) expansion / cavitation
                    rho_unsteady)                            // (d) moving contact wake
                    cell_bdf2 = false;
            }
            bdf_c0[i] = cell_bdf2 ? 1.5 : 1.0;
            if (cell_bdf2) {
                Cold_mom[i] = 2.0 * mom_o - 0.5 * mom_o2[i];
                Cold_con[i] = 2.0 * con_o - 0.5 * rho_o2[i];
                Cold_ene[i] = 2.0 * ene_o - 0.5 * ene_o2[i];
            } else {
                Cold_mom[i] = mom_o;
                Cold_con[i] = con_o;
                Cold_ene[i] = ene_o;
            }
        }

        // inlet (steady or time-dependent): uin = base_velocity + inlet_du*sin(2*pi*f*t)
        // (f=0 -> steady inlet at base_velocity, e.g. case02 advection).
        const bool inlet_left = (lbc == "inlet");

        const char* urf0 = std::getenv("ACID_URF");
        const double om = urf0 ? std::atof(urf0) : 1.0;  // constant under-relaxation (1 = none)

        // ---- residual+flux as a reusable function (for Jacobian assembly, a finite-diff
        //      Jacobian check, and a line search). Fills the outer flux vars + Rres from s. ----
        double VdT = dx / dt;
        // TR-BDF2 stage machinery (identity on the non-TR path -> byte-unchanged there):
        //   flux_w    scales the IMPLICIT spatial flux (0.5 in the trapezoidal stage, else 1);
        //   flux_expl adds the FROZEN old-state flux 0.5*F(U_n) in the trapezoidal stage;
        //   dt_mwi    is the stage's transient timescale for the MWI (Rhie-Chow) dhat + memory;
        //   t_stage   is the time of the implicit level for the inlet BC (per TR-BDF2 stage);
        //   flux_*_arr capture the raw spatial flux so F(U_n) can be frozen for the trap stage.
        double flux_w = 1.0, dt_mwi = dt;
        double t_stage = t + dt;
        Vec flux_expl_m(n, 0.0), flux_expl_c(n, 0.0), flux_expl_e(n, 0.0);
        Vec flux_m_arr(n, 0.0), flux_c_arr(n, 0.0), flux_e_arr(n, 0.0);
        Vec theta(n + 1), rho_f(n + 1), dhat(n + 1), pface(n + 1), uconv(n + 1);
        Vec raup(n + 1), rbup(n + 1), rHaup(n + 1), rHbup(n + 1), mdotL(n), mdotR(n);
        // per-face metadata frozen for the analytic Jacobian (ACID_AJAC): the reconstruction
        // mode (use4 = 4th-order linear; else 1st-order upwind / 2nd central) and whether the
        // MWI pressure correction is unclamped (so d(theta)/dp is active) + the sound-speed bound.
        Vec af_f(n + 1, 0.0);
        Vec dpgpf(n + 1, 0.0);  // (dpf - gpbar) per face, for the analytic d(theta)/d(rho via dhat)
        std::vector<char> use4_f(n + 1, 0), mwiOK_f(n + 1, 0);
        std::vector<int> uwc_f(n + 1, 0);  // upwind cell index per face (for the transport Jacobian)
        std::vector<Vec2> Rres(n);
        Vec Rene(n, 0.0);  // energy residual (ACID_COUPLED 3rd component)
        double uin = 0.0;
        // reusable ghost scratch (OPT3): compute_R fills these 5 every call instead of allocating
        // 5 fresh vectors -- removes ~5 malloc/free per residual eval (millions over a run).
        Vec g_pe(n + 4), g_ue(n + 4), g_re(n + 4), g_Te(n + 4), g_ae(n + 4);
        auto compute_R = [&]() {
            // --- ACID_YADV (round 4, task (b)): alpha as an IMPLICIT function of the Newton
            //     unknowns. On this path alpha is DERIVED from the transported Y at the current
            //     (p,T); rounds 1-3 froze it at the pre-Newton (p_o,T_o) value, which made the
            //     alpha<->p coupling an explicit lagged loop across the stiffest state dependence
            //     in the problem (docs/YADV_RESEARCH.md 7.2 / 11.6). Re-deriving it here -- the
            //     FIRST thing every residual evaluation does, so every Newton iterate, every FD
            //     probe, every line-search trial and every TR-BDF2 stage sees it -- makes
            //     alpha = alpha(Y, p, T) implicit by RE-EVALUATION rather than by an explicit
            //     d(alpha)/dp, d(alpha)/dT Jacobian row (defect-correction: compute_R is the
            //     single source of truth, an approximate Jacobian changes only iteration count).
            //     On the first call of a step s.p/s.T still equal p_o/T_o, so this reproduces the
            //     pre-Newton recovery value exactly -- no discontinuity at the start.
            //     In `coupled` mode alpha needs T while T_from_hstat needs alpha: resolved as an
            //     OUTER PICARD lag (alpha from the previous call's T, then T re-solved with it),
            //     which vanishes at convergence when s stops moving.
            //     RESEARCH-ONLY, gated by ACID_YADV_ALPHA_IMPLICIT (default OFF, see the flag's
            //     declaration above) -- a measured net regression under the default analytic
            //     Jacobian, so plain ACID_YADV=1 must NOT pick this up implicitly.
            if (yadv && alpha_implicit) {
                for (int i = 0; i < n; ++i) {
                    const double pu = std::max(s.p[i], 1.0), Tu = std::max(s.T[i], 1e-6);
                    s.alpha[i] = std::clamp(alpha_from_mass_fraction(Yv[i],
                                                                     phase_props(pu, Tu, A).rho,
                                                                     phase_props(pu, Tu, B).rho),
                                            0.0, 1.0);
                }
            }
            // --- ACID_COUPLED: derive T from the coupled total enthalpy h BEFORE eval_thermo,
            //     so rho/hstat are consistent with h every iteration (THIS fixes the segregated
            //     rho-mismatch that drove the case25 blowup). ---
            if (coupled) {
                // per-cell h->T inversion (the heaviest kernel: ~30 Newton iters x phase_props
                // each). Independent across cells -> parallel (the dominant compute_R cost).
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) {
                    const double hstat_i = s.h[i] - 0.5 * s.u[i] * s.u[i];
                    double Tnew;
                    if (T_from_hstat(hstat_i, s.p[i], s.alpha[i], A, B, s.T[i], Tnew))
                        s.T[i] = Tnew;
                    // else: keep old T (non-physical hstat<kinetic transient); the line search
                    // / clamp pulls h back into the physical range on the next trial.
                }
            }
            // ACID_TSAT block A (round 17): after the C1 loop, s.T[i]>=1e6 is an EXACT bit-level
            // test for "this cell sits at the clamp, not a solution of hmix(T)=hstat" -- serial,
            // no change to the OMP loop above. docs/YADV_ROUND_17_PLAN.md sect.3.
            if (tsat && coupled) {
                int nhi = 0, nlo = 0, ihi = -1;
                for (int i = 0; i < n; ++i) {
                    if (s.T[i] >= 1.0e6)  { ++nhi; if (ihi < 0) ihi = i; }
                    if (s.T[i] <= 1.0e-6) { ++nlo; }
                }
                ++tsat_calls;
                if (nhi) {
                    ++tsat_calls_hi;
                    if (nhi > tsat_cells_hi_max) tsat_cells_hi_max = nhi;
                    if (tsat_first_step < 0) { tsat_first_step = step; tsat_first_cell = ihi; }
                    if (tsat >= 2)
                        std::fprintf(stderr,
                            "TSAT case=%s step=%d retry=%d dt=%.6e ncells_hi=%d i0=%d p=%.6e h=%.6e\n",
                            c.id.c_str(), step, retry, dt, nhi, ihi, s.p[ihi], s.h[ihi]);
                }
                if (nlo) ++tsat_calls_lo;
            }
            eval_thermo(s, A, B);
            // ghost-extended p, u for gradients / BC (filled into reusable scratch, OPT3)
            apply_ghost_into(g_pe, s.p, lbc, rbc, 2, false);   const Vec& pe = g_pe;
            apply_ghost_into(g_ue, s.u, lbc, rbc, 2, true);    const Vec& ue = g_ue;
            apply_ghost_into(g_re, s.rho, lbc, rbc, 2, false); const Vec& re = g_re;

            // ===== DEFECT-CORRECTION (Newton) coupled (u,p) solve =====
            //   mdot[f] is the SINGLE source of truth; residual R is computed exactly from
            //   it; the (approximate) Jacobian J is assembled consistently; solve J dx = -R.
            //   At convergence R->0 regardless of Jacobian approximation.
            auto cell_gradp = [&](int gi) { return (pe[gi + 1] - pe[gi - 1]) / (2.0 * dx); };
            auto uo = [&](int k) { return uu_o[std::clamp(k, 0, n - 1)]; };  // OLD cell velocity

            apply_ghost_into(g_Te, s.T, lbc, rbc, 2, false); const Vec& Te = g_Te;
            apply_ghost_into(g_ae, s.a, lbc, rbc, 2, false); const Vec& ae = g_ae;  // MWI bound

            // --- face quantities (fills the outer flux vars): MWI advecting velocity +
            //     UPWIND partial densities/enthalpies for the ACID face density (Eqs.40-42) ---
            if (inlet_left) {
                const double tt = t_stage;
                if (c.config.pulse_inlet) {
                    // Single one-period acoustic wave PACKET (Denner §7.3.2 style). Denner's
                    // Eq.69 uses +3pi/2 and holds u0-du afterwards, which is fine on his u0=1
                    // mean flow but, on our u0=0 quiescent base, leaves a spurious steady inflow
                    // (-du) that biases the field by -Z*du. So here the packet is net-zero and
                    // RETURNS TO REST: u_in = u0 + du*sin(2*pi*f*t) for t<1/f, then u0. (No value
                    // jump; the acoustic reflection/transmission ratios are u0-independent.)
                    const double T = 1.0 / c.inlet_frequency;
                    uin = (tt < T)
                        ? c.base_velocity + c.inlet_du * std::sin(2.0 * M_PI * c.inlet_frequency * tt)
                        : c.base_velocity;
                } else {
                    uin = c.base_velocity + c.inlet_du * std::sin(2.0 * M_PI * c.inlet_frequency * tt);
                }
            }
            // per-face state (MWI theta, pface, upwind EOS phase_props x2): each face writes only
            // its own slot and reads cell arrays (gather, no scatter) -> parallel.
            #pragma omp parallel for schedule(static)
            for (int f = 0; f <= n; ++f) {
                const int gL = f + 1, gR = f + 2;
                rho_f[f] = 2.0 / (1.0 / re[gL] + 1.0 / re[gR]);  // harmonic (ACID Eq.22, for MWI)
                // 4th-order central face interpolation of the convected primitives (p,u) cuts the
                // 2nd-order numerical DISPERSION that spreads/damps the acoustic packet (case07,
                // reflection err 31% -> 5%). The wide stencil must NOT cross the air-water contact,
                // so use it only where the 4-cell stencil is single-phase; revert to 2nd-order at
                // the interface (Denner 5.4) to avoid a transmitted-side blow-up.
                // face shock sensor: a large pressure ratio across the 4-cell reconstruction
                // stencil [gL-1,gL,gR,gR+1] marks a shock face -> revert BOTH the 4th-order
                // (lowdiss) AND the Minmod reconstruction to 1st-order upwind there. Acoustic
                // waves (p ratio ~1.001) never trip it, so 04/05/07 keep their high order; the
                // strong shocks (13/14/24/25) do, killing the 2nd/4th-order ringing. Purely
                // solution-adaptive (reads the local pressure, not the case id) -> uniform scheme.
                const double psmax = std::max({pe[gL - 1], pe[gL], pe[gR], pe[gR + 1]});
                const double psmin = std::min({pe[gL - 1], pe[gL], pe[gR], pe[gR + 1]});
                const bool face_shock = psmax > 1.3 * std::max(psmin, 1.0);
                bool use4 = false;
                if (lowdiss && !face_shock) {
                    auto ph = [&](int i) { return s.alpha[std::clamp(i, 0, n - 1)] >= 0.5; };
                    const bool p0 = ph(f - 1);
                    use4 = (ph(f - 2) == p0 && ph(f) == p0 && ph(f + 1) == p0);
                }
                const double ubar = use4
                    ? (-ue[gL - 1] + 7.0 * ue[gL] + 7.0 * ue[gR] - ue[gR + 1]) / 12.0
                    : 0.5 * (ue[gL] + ue[gR]);
                // dhat_f (ACID Eq.21), transient-dominated a_P = rho*dx/dt -- ONE formula for every
                // case and every face (no per-case variants, no tuning multipliers). The advective
                // e_P form and a dissipation scale were tested and REMOVED: non-uniform/per-case
                // (see .claude/rules/denner-pitfalls.md).
                const double aP = 0.5 * (re[gL] + re[gR]) * dx / dt_mwi;  // transient-dominated a_P
                const double d_f = dx / std::max(aP, 1e-300);
                dhat[f] = d_f / (1.0 + (rho_f[f] / dt_mwi) * d_f);
                const double dpf = (pe[gR] - pe[gL]) / dx;
                const double gpbar = 0.5 * (cell_gradp(gL) + cell_gradp(gR));
                const double ubar_o = 0.5 * (uo(f - 1) + uo(f));
                // MWI (Rhie-Chow) pressure correction -- bound it to the local sound speed so
                // a strong shock's huge pressure gradient cannot blow up the advecting
                // velocity (the low-Mach MWI assumes a SMALL 3rd-derivative term; that breaks
                // at shocks). Inactive for smooth flow (|corr| << a), so 04/05 unaffected.
                const double af = 0.5 * (ae[gL] + ae[gR]);
                double mwi_p = -dhat[f] * (dpf - gpbar);
                mwiOK_f[f] = (std::abs(mwi_p) < af) ? 1 : 0;  // unclamped -> d(theta)/dp active
                af_f[f] = af;
                dpgpf[f] = dpf - gpbar;  // frozen for the analytic d(theta)/d(rho via dhat)
                use4_f[f] = use4 ? 1 : 0;
                mwi_p = std::clamp(mwi_p, -af, af);
                theta[f] = ubar + mwi_p
                           + (rho_f[f] / dt_mwi) * dhat[f] * (theta_o[f] - ubar_o);
                pface[f] = use4
                    ? (-pe[gL - 1] + 7.0 * pe[gL] + 7.0 * pe[gR] - pe[gR + 1]) / 12.0  // 4th-order face interp
                    : 0.5 * (pe[gL] + pe[gR]);
                const bool fromL = theta[f] >= 0.0;
                const int gU = fromL ? gL : gR;  // upwind cell (ghost idx)
                uwc_f[f] = std::clamp(gU - 2, 0, n - 1);  // upwind cell index (analytic transport Jac)
                double pU = std::max(pe[gU], 1.0), TU = std::max(Te[gU], 1e-6), uU = ue[gU];
                if (use_minmod) {
                    // Minmod TVD 2nd-order reconstruction of the convected primitives from the
                    // upwind side. p,T,u are CONTINUOUS across the contact (only rho jumps), so
                    // no interface fallback is needed; this cuts the 1st-order acoustic
                    // dissipation that damps the reflected/transmitted waves.
                    // ACID_CENTRAL: unlimited central (2nd-order) face value -- for a SMOOTH
                    // acoustic wave the Minmod limiter clips the extrema (peaks/troughs) and
                    // re-adds dissipation; central does not clip. (Test for the case07 dissipation.)
                    static const bool central = std::getenv("ACID_CENTRAL") != nullptr;
                    auto mm = [](double a, double b) { return (a * b <= 0.0) ? 0.0 : (std::abs(a) < std::abs(b) ? a : b); };
                    auto rec = [&](const Vec& q) {
                        // 4th-order central face value for the ACID density (when the stencil is
                        // single-phase, use4) -> makes the mass-flux density high-order like
                        // pface/ubar, so the acoustic continuity is consistently 4th-order.
                        if (use4) return (-q[gL - 1] + 7.0 * q[gL] + 7.0 * q[gR] - q[gR + 1]) / 12.0;
                        if (face_shock) return q[gU];  // 1st-order upwind at a shock face (TVD)
                        if (central) return 0.5 * (q[gL] + q[gR]);
                        const double bk = q[gU] - q[gU - 1], fw = q[gU + 1] - q[gU];
                        return fromL ? q[gU] + 0.5 * mm(bk, fw) : q[gU] - 0.5 * mm(fw, bk);
                    };
                    pU = std::max(rec(pe), 1.0); TU = std::max(rec(Te), 1e-6); uU = rec(ue);
                }
                const auto ppaU = phase_thermo(pU, TU, A);  // face flux reads only .rho/.h ->
                const auto ppbU = phase_thermo(pU, TU, B);  // lean (no sound-speed sqrt/p-partials)
                raup[f] = ppaU.rho;
                rbup[f] = ppbU.rho;
                uconv[f] = uU;
                const double Hkin = 0.5 * uU * uU;  // upwind kinetic energy
                rHaup[f] = ppaU.rho * (ppaU.h + Hkin);  // partial total enthalpy flux density
                rHbup[f] = ppbU.rho * (ppbU.h + Hkin);
            }
            if (inlet_left) { theta[0] = uin; uconv[0] = uin; }
            if (lbc == "reflective") { theta[0] = 0.0; uconv[0] = 0.0; }
            if (rbc == "reflective") { theta[n] = 0.0; uconv[n] = 0.0; }

            // ACID per-cell face mass flux (Eqs.41-42): rho_f^(i) = rho_a_up + psi_i*(rho_b_up - rho_a_up)
            //   -> uniform-velocity contact gives div(mdot)=rho_i*div(theta)=0 (no spurious source)
            // rho_f^(i) = alpha_i * rho_a_up + (1-alpha_i) * rho_b_up  (alpha = vol frac of phase a,
            // consistent with eval_thermo's rho_mix = al*rho_a + (1-al)*rho_b)
            for (int i = 0; i < n; ++i) {
                const double al = std::clamp(s.alpha[i], 0.0, 1.0);
                mdotR[i] = (al * raup[i + 1] + (1.0 - al) * rbup[i + 1]) * theta[i + 1];
                mdotL[i] = (al * raup[i] + (1.0 - al) * rbup[i]) * theta[i];
            }
            if (inlet_left) mdotL[0] = (std::clamp(s.alpha[0], 0.0, 1.0) * raup[0] + (1.0 - std::clamp(s.alpha[0], 0.0, 1.0)) * rbup[0]) * uin;
            if (lbc == "reflective") mdotL[0] = 0.0;
            if (rbc == "reflective") mdotR[n - 1] = 0.0;

            // --- exact residual R(u,p) (fills outer Rres). BE/BDF2 transient via Cold_* ---
            for (int i = 0; i < n; ++i) {
                const double trans_m = (bdf_c0[i] *s.rho[i] * s.u[i] - Cold_mom[i]) * VdT;
                const double conv = mdotR[i] * uconv[i + 1] - mdotL[i] * uconv[i];
                const double pres = pface[i + 1] - pface[i];
                const double trans_c = (bdf_c0[i] *s.rho[i] - Cold_con[i]) * VdT;
                if (!tr_bdf2) {
                    Rres[i][0] = trans_m + conv + pres;
                    Rres[i][1] = trans_c + (mdotR[i] - mdotL[i]);
                } else {
                    // TR-BDF2: implicit flux scaled by flux_w + frozen old-state flux flux_expl.
                    const double fm = conv + pres, fc = mdotR[i] - mdotL[i];
                    flux_m_arr[i] = fm; flux_c_arr[i] = fc;
                    Rres[i][0] = trans_m + flux_w * fm + flux_expl_m[i];
                    Rres[i][1] = trans_c + flux_w * fc + flux_expl_c[i];
                }
            }
            // --- ACID_COUPLED energy residual (Denner Eq.28, BE, total-enthalpy form):
            //     (rho h - rho_o h_o)/dt V + sum_f mdot_f h_f = (p - p_o)/dt V
            //     The advection sum_f mdot_f h_f reuses the SAME ACID partial-enthalpy fluxes
            //     theta_f*(a_i*rHaup + (1-a_i)*rHbup) as the segregated path (sign-identical). ---
            if (coupled) {
                for (int i = 0; i < n; ++i) {
                    const double ai = std::clamp(s.alpha[i], 0.0, 1.0);
                    const double fR = theta[i + 1] * (ai * rHaup[i + 1] + (1.0 - ai) * rHbup[i + 1]);
                    const double fL = theta[i] * (ai * rHaup[i] + (1.0 - ai) * rHbup[i]);
                    const double adv = fR - fL;
                    if (!tr_bdf2) {
                        const double trans_e = (bdf_c0[i] *s.rho[i] * s.h[i] - Cold_ene[i]) * VdT;
                        const double srcp = (s.p[i] - p_o[i]) * VdT;
                        Rene[i] = trans_e + adv - srcp;
                    } else {
                        // CONSERVATIVE total-energy transient rho*E = rho*H - p (bdf_c0=1 under
                        // TR-BDF2). The (p-p_o)/dt pressure-work source is ABSORBED here (Cold_ene
                        // carries the old-level rho*E), leaving a clean transient + weighted flux.
                        flux_e_arr[i] = adv;
                        const double trans_E = (s.rho[i] * s.h[i] - s.p[i] - Cold_ene[i]) * VdT;
                        Rene[i] = trans_E + flux_w * adv + flux_expl_e[i];
                    }
                }
            }
        };  // ===== end compute_R =====

        // ---- store the transient advecting velocity theta_o at the CURRENT state s, over a
        //      timescale dtt (the MWI dhat uses this timescale). Called at each step end (dtt=dt,
        //      byte-identical to the former inline block) and, under TR-BDF2, between the two
        //      stages (dtt = gamma*dt) so stage 2's MWI memory sees theta at t+gamma*dt. ----
        auto store_theta_o = [&](double dtt) {
            const auto pe = apply_ghost(s.p, lbc, rbc, 2, false);
            const auto ue = apply_ghost(s.u, lbc, rbc, 2, true);
            const auto re = apply_ghost(s.rho, lbc, rbc, 2, false);
            for (int f = 0; f <= n; ++f) {
                const int gL = f + 1, gR = f + 2;
                const double rf = 2.0 / (1.0 / re[gL] + 1.0 / re[gR]);
                const double aP = 0.5 * (re[gL] + re[gR]) * dx / dtt;
                const double d_f = dx / std::max(aP, 1e-300);
                const double dh = d_f / (1.0 + (rf / dtt) * d_f);
                const double dpf = (pe[gR] - pe[gL]) / dx;
                const double gpbar = 0.5 * ((pe[gL + 1] - pe[gL - 1]) / (2 * dx) + (pe[gR + 1] - pe[gR - 1]) / (2 * dx));
                theta_o[f] = 0.5 * (ue[gL] + ue[gR]) - dh * (dpf - gpbar);
            }
        };

        // helper: L2 norm of the residual (momentum + uref-scaled continuity)
        auto rnorm = [&]() {
            double s2 = 0.0;
            for (int i = 0; i < n; ++i)
                s2 += Rres[i][0] * Rres[i][0] + (uref * Rres[i][1]) * (uref * Rres[i][1]);
            return std::sqrt(s2);
        };
        // 3-component norm for ACID_COUPLED: scale continuity by uref and energy by uref/href so
        // all three balance dimensionally with the momentum residual.
        const double escal = uref / std::max(href, 1.0);
        auto rnorm3 = [&]() {
            double s2 = 0.0;
            for (int i = 0; i < n; ++i)
                s2 += Rres[i][0] * Rres[i][0] + (uref * Rres[i][1]) * (uref * Rres[i][1])
                      + (escal * Rene[i]) * (escal * Rene[i]);
            return std::sqrt(s2);
        };

        // ---- outer/inner Newton iteration. The analytic (Picard) Jacobian converges linearly,
        //      so allow more iterations -- each is ~15x cheaper than an FD-Jacobian iteration
        //      (no 15 compute_R), so the extra iters still net a large speedup. ----
        // analytic Jacobian is now the DEFAULT (10/10, ~33s vs the FD path's ~41s). The keep-best +
        // stall-break globalization + progress-based conv-guard make it robust on the shocks (24/25)
        // and cavitation (15). Opt out to the FD modified-Newton path with ACID_NO_AJAC.
        bool ajac = std::getenv("ACID_NO_AJAC") == nullptr;
        // TR-BDF2 uses the FINITE-DIFFERENCE Jacobian: it differentiates compute_R exactly, so the
        // new flux_w / flux_expl / rho*E energy terms are captured with no analytic-Jacobian edits
        // (defect-correction -> the converged solution is Jacobian-independent). Restoring the fast
        // analytic path for TR (flux_w-scaling on the flux-coupling rows; the rho*E energy diagonal
        // already coincides with the old p-work source) is follow-up work.
        if (tr_bdf2) ajac = false;
        // ACID_AJAC_BLK: compute BOTH the FD and analytic blocks and report the per-block
        // max difference (localises which Jacobian term the analytic gets wrong). Uses FD for
        // the actual solve (non-destructive). Optional ACID_BLK_STEP picks the report step.
        const bool ajblk = std::getenv("ACID_AJAC_BLK") != nullptr;
        // modified-Newton (env ACID_MNEWTON=K): reuse the assembled FD pentadiagonal Jacobian for
        // up to K inner iters. Defect-correction makes the residual R the single source of truth
        // (the compute_R/solve split below), so a stale J changes only the iteration COUNT, never
        // the converged fixed point -> 10/10 byte-identical AT CONVERGENCE and case01 (~1 iter)
        // untouched. Reassembly is FORCED on it==0 and whenever the previous step backtracked
        // (al<1), so a stale J is reused only on clean full Newton steps. K=1 == assemble every
        // iter (the prior behaviour). Only the pure-FD path caches; ajac/ajblk assemble every iter.
        // default K=2 (measured optimum: per-case-sum 62.1s K=1 -> 54.1s K=2 -> 58.3s K=3; a
        // staler J at K>=3 adds enough inner iters on case24 to offset the saved assemblies).
        const int Kmn = []{ const char* e = std::getenv("ACID_MNEWTON"); return e ? std::max(1, std::atoi(e)) : 2; }();
        std::vector<Mat3> MA3(n, Mat3{}), MB3(n, Mat3{}), MC3(n, Mat3{});
        std::vector<Mat3> ME3, MF3;  // i-2, i+2 blocks (pentadiagonal)
        if (penta_solve) { ME3.assign(n, Mat3{}); MF3.assign(n, Mat3{}); }
        int jac_age = Kmn;            // >= Kmn so it==0 assembles
        bool backtracked_last = false;
        bool conv_inner = false;      // did the coupled inner Newton actually converge?
        // keep-best + stall break: a stiff regime (case15 cavitation) NEVER converges -- the line
        // search pins at the al floor with the residual flat, so running to the iter cap every step
        // is wasted work (the case15 slowness) and dt-retrying it (ajac) collapses dt. Track the
        // best (lowest-residual) iterate, BREAK when the residual stops improving for ACID_STALLWIN
        // iters, and accept that best iterate. Converging cases improve every iter -> never stall.
        const int stallwin = []{ const char* e = std::getenv("ACID_STALLWIN"); return e ? std::max(1, std::atoi(e)) : 5; }();
        double rbest = 1e300, r_init = -1.0; int best_it = -1; Field s_best;

        // ===== TR-BDF2 conserved-quantity snapshots (level n) + 2-stage driver =====
        // phi = (rho*u, rho, rho*E) with rho*E = rho*Htot - p. Non-TR path: nstage=1, coeffs
        // untouched (VdT=dx/dt, flux_w=1, flux_expl=0) -> the Newton runs exactly as before.
        Vec trPhiN_m, trPhiN_c, trPhiN_e, trPhiG_m, trPhiG_c, trPhiG_e;
        if (tr_bdf2) {
            trPhiN_m.assign(n, 0.0); trPhiN_c.assign(n, 0.0); trPhiN_e.assign(n, 0.0);
            trPhiG_m.assign(n, 0.0); trPhiG_c.assign(n, 0.0); trPhiG_e.assign(n, 0.0);
            for (int i = 0; i < n; ++i) {
                trPhiN_m[i] = rho_o[i] * u_o[i];
                trPhiN_c[i] = rho_o[i];
                trPhiN_e[i] = rho_o[i] * Htot_o[i] - p_o[i];  // rho*E at level n
            }
        }
        const int nstage = tr_bdf2 ? 2 : 1;
        for (int stage = 0; stage < nstage; ++stage) {
        if (tr_bdf2 && stage == 0) {
            // ---- STAGE 1: trapezoidal over gamma*dt.  (phi - phi_n)*dx/(g*dt) + 0.5 F(U) + 0.5 F(U_n) = 0 ----
            for (int i = 0; i < n; ++i) {
                bdf_c0[i] = 1.0;
                Cold_mom[i] = trPhiN_m[i]; Cold_con[i] = trPhiN_c[i]; Cold_ene[i] = trPhiN_e[i];
            }
            VdT = dx / (trg * dt);
            dt_mwi = trg * dt;
            // freeze F(U_n): s is still the level-n state here, so one compute_R fills flux_*_arr.
            // Its inlet BC is the OLD-level forcing uin(t) (trapezoidal f(t_n, U_n)).
            flux_w = 1.0;
            t_stage = t;
            for (int i = 0; i < n; ++i) { flux_expl_m[i] = 0.0; flux_expl_c[i] = 0.0; flux_expl_e[i] = 0.0; }
            compute_R();
            for (int i = 0; i < n; ++i) {
                flux_expl_m[i] = 0.5 * flux_m_arr[i];
                flux_expl_c[i] = 0.5 * flux_c_arr[i];
                flux_expl_e[i] = 0.5 * flux_e_arr[i];
            }
            flux_w = 0.5;
            t_stage = t + trg * dt;  // implicit stage-1 forcing uin(t+gamma*dt)
        } else if (tr_bdf2) {
            // ---- STAGE 2: BDF2 on {n, n+g, n+1}.  (phi - (a phi_g - b phi_n))*dx/(c*dt) + F(U) = 0 ----
            for (int i = 0; i < n; ++i) {
                bdf_c0[i] = 1.0;
                Cold_mom[i] = tra * trPhiG_m[i] - trb * trPhiN_m[i];
                Cold_con[i] = tra * trPhiG_c[i] - trb * trPhiN_c[i];
                Cold_ene[i] = tra * trPhiG_e[i] - trb * trPhiN_e[i];
            }
            VdT = dx / (trc * dt);
            dt_mwi = trc * dt;
            flux_w = 1.0;
            t_stage = t + dt;  // implicit stage-2 forcing uin(t+dt) (new level)
            for (int i = 0; i < n; ++i) { flux_expl_m[i] = 0.0; flux_expl_c[i] = 0.0; flux_expl_e[i] = 0.0; }
        }
        // per-stage Newton state reset
        rbest = 1e300; r_init = -1.0; best_it = -1; conv_inner = false;
        jac_age = Kmn; backtracked_last = false;
        for (int it = 0; it < (ajac ? 150 : 40); ++it) {
            compute_R();
            // RINIT (round 13 Stage 0, docs/YADV_ROUND_13_PLAN.md sect.1): component split of
            // rnorm3() at it==0 -- self-check: `r` here must equal RHIST's `n0` on the same run.
            if (rinit_dbg && yadv && coupled && it == 0) {
                const char* se = std::getenv("ACID_BLK_STEP");
                const int blkstep = se ? std::atoi(se) : -1;
                if (blkstep < 0 || step == blkstep) {
                    double mom = 0, con = 0, ene = 0;
                    int iene = -1; double ene_max = -1.0;
                    const double escal_l = uref / std::max(href, 1.0);
                    for (int i = 0; i < n; ++i) {
                        mom += Rres[i][0] * Rres[i][0];
                        con += (uref * Rres[i][1]) * (uref * Rres[i][1]);
                        const double e2 = (escal_l * Rene[i]) * (escal_l * Rene[i]);
                        ene += e2;
                        if (e2 > ene_max) { ene_max = e2; iene = i; }
                    }
                    mom = std::sqrt(mom); con = std::sqrt(con); ene = std::sqrt(ene);
                    const double r = std::sqrt(mom * mom + con * con + ene * ene);
                    std::fprintf(stderr,
                        "RINIT case=%s step=%d retry=%d dt=%.6e r=%.6e mom=%.6e con=%.6e ene=%.6e "
                        "fene=%.4f iene=%d\n",
                        c.id.c_str(), step, retry, dt, r, mom, con, ene,
                        r > 0 ? (ene * ene) / (r * r) : 0.0, iene);
                }
            }
            (void)rnorm;
            // DEEP-DEBUG dense probe: d R_ene[i0]/d{u,p,h}[j] for j=i0-3..i0+3 by SINGLE-CELL FD
            // (no graph colouring) -> tells whether the i+-2 dene/du the colour-FD reports is a
            // real coupling or a stride-5 aliasing of a wider stencil. ACID_DENSE=<i0>.
            if (coupled && std::getenv("ACID_DENSE")) {
                const char* se = std::getenv("ACID_BLK_STEP"); const int rstep = se ? std::atoi(se) : 2;
                if (step == rstep && it == 0) {
                    const int i0 = std::atoi(std::getenv("ACID_DENSE"));
                    const Vec R0e = Rene;
                    std::fprintf(stderr, "DENSE case%s step%d i0=%d uw[faces i0-2..i0+3]: ",
                                 c.id.c_str(), step, i0);
                    for (int f = i0 - 2; f <= i0 + 3; ++f) if (f >= 0 && f <= n) std::fprintf(stderr, "f%d>uw%d ", f, uwc_f[f]);
                    std::fprintf(stderr, "\n");
                    const char* vn[3] = {"u", "p", "h"};
                    for (int vv = 0; vv < 3; ++vv) {
                        std::fprintf(stderr, "  dene[i0]/d%s: ", vn[vv]);
                        for (int j = i0 - 3; j <= i0 + 3; ++j) {
                            if (j < 0 || j >= n) { std::fprintf(stderr, "  j%+d=   .      ", j - i0); continue; }
                            double& x = vv == 0 ? s.u[j] : (vv == 1 ? s.p[j] : s.h[j]);
                            const double sv = x;
                            const double e = 1e-7 * (std::abs(sv) + (vv == 0 ? uref : (vv == 1 ? 1.0e5 : href)));
                            x = sv + e; compute_R(); const double d = (Rene[i0] - R0e[i0]) / e; x = sv;
                            std::fprintf(stderr, "j%+d=%9.2e ", j - i0, d);
                        }
                        std::fprintf(stderr, "\n");
                    }
                    compute_R();  // restore
                }
            }

            // ============================================================================
            // ===== ACID_COUPLED: faithful Denner fully-coupled 3x3 (u,p,h) Newton    =====
            // ============================================================================
            if (coupled) {
                // ANALYTIC-JACOBIAN stage 1 verification: the EOS chain d(rho)/d(u,p,h) via the
                // h->T inversion, checked against FD. EOS partials are exact in PhaseProps
                // (zeta=drho/dp|T, phi=drho/dT|p, dh_dp, cp). T solves h_static_mix(T,p)=h-u^2/2.
                if (std::getenv("ACID_AJAC_CHECK") && step == 0 && it == 0) {
                    const int i0 = n / 2;
                    const double al = std::clamp(s.alpha[i0], 0.0, 1.0);
                    const double p = std::max(s.p[i0], 1.0), Tc = std::max(s.T[i0], 1e-6), u = s.u[i0];
                    const auto pa = phase_props(p, Tc, A); const auto pb = phase_props(p, Tc, B);
                    const double D = al * pa.rho + (1 - al) * pb.rho;            // rho_mix
                    const double D_T = al * pa.phi + (1 - al) * pb.phi;          // drho_mix/dT
                    const double D_p = al * pa.zeta + (1 - al) * pb.zeta;        // drho_mix/dp
                    const double N = al * pa.rho * pa.h + (1 - al) * pb.rho * pb.h;
                    const double N_T = al * (pa.phi * pa.h + pa.rho * pa.cp)
                                     + (1 - al) * (pb.phi * pb.h + pb.rho * pb.cp);
                    const double N_p = al * (pa.zeta * pa.h + pa.rho * pa.dh_dp)
                                     + (1 - al) * (pb.zeta * pb.h + pb.rho * pb.dh_dp);
                    const double hsT = (N_T * D - N * D_T) / (D * D);            // dh_static/dT|p
                    const double hsp = (N_p * D - N * D_p) / (D * D);            // dh_static/dp|T
                    const double drho_dh = D_T * (1.0 / hsT);
                    const double drho_du = D_T * (-u / hsT);
                    const double drho_dp = D_p + D_T * (-hsp / hsT);
                    auto rho_of = [&](double pp, double TT) {
                        return al * phase_props(std::max(pp, 1.0), std::max(TT, 1e-6), A).rho
                             + (1 - al) * phase_props(std::max(pp, 1.0), std::max(TT, 1e-6), B).rho; };
                    auto T_of = [&](double hh, double uu, double pp) {
                        double Tn; T_from_hstat(hh - 0.5 * uu * uu, std::max(pp, 1.0), al, A, B, Tc, Tn); return Tn; };
                    const double eh = 1e-4 * std::max(std::abs(s.h[i0]), 1.0);
                    const double eu = 1e-5 * std::max(std::abs(u), 1.0);
                    const double ep = 1e-3 * p;
                    const double r0 = rho_of(p, T_of(s.h[i0], u, p));
                    const double fd_dh = (rho_of(p, T_of(s.h[i0] + eh, u, p)) - r0) / eh;
                    const double fd_du = (rho_of(p, T_of(s.h[i0], u + eu, p)) - r0) / eu;
                    const double fd_dp = (rho_of(std::max(p + ep, 1.0), T_of(s.h[i0], u, std::max(p + ep, 1.0))) - r0) / ep;
                    std::fprintf(stderr,
                        "AJAC i0=%d al=%.3f drho/dh A=% .6e FD=% .6e | du A=% .6e FD=% .6e | dp A=% .6e FD=% .6e\n",
                        i0, al, drho_dh, fd_dh, drho_du, fd_du, drho_dp, fd_dp);
                }
                // --- numerical (u,p,h) block Jacobian by FD with stride-5 graph colouring.
                //     The MWI/Rhie-Chow pressure-velocity coupling makes R[i] depend on the
                //     pressure stencil p[i-2..i+2] -> the TRUE Jacobian is block-PENTADIAGONAL
                //     (i+-2), not tridiagonal. At small dt the i+-2 entries are negligible (the
                //     MWI ~ dt is tiny) so the tridiagonal solve converges; at LARGE material-dt
                //     the MWI dominates and dropping i+-2 gives a wrong Newton direction -> the
                //     coupled+material-dt divergence (case02). ACID_PENTA records + solves the
                //     full pentadiagonal Jacobian (block_penta), fixing that. (stride-5 still
                //     isolates a bandwidth-2 Jacobian: cells 5 apart never alias.) ---
                const bool penta = penta_solve;
                std::vector<Vec3> Md3(n);
                const std::vector<Vec2> R0 = Rres;   // baseline (u,p) residual
                const Vec R0e = Rene;                // baseline energy residual
                // modified-Newton: (re)assemble the FD Jacobian only on selected iters; otherwise
                // reuse the cached MA3..MF3 with the fresh -R0 RHS (stale-J inexact Newton step).
                ++jac_age;
                const bool do_fd_assembly = (!ajac || ajblk)
                    && (ajblk || it == 0 || jac_age >= Kmn || backtracked_last);
                if (do_fd_assembly) {
                    auto var = [&](int vv, int j) -> double& {
                        return vv == 0 ? s.u[j] : (vv == 1 ? s.p[j] : s.h[j]);
                    };
                    Vec eps(n);
                    for (int vv = 0; vv < 3; ++vv) {
                        for (int i = 0; i < n; ++i) {
                            const double scale = (vv == 0) ? (std::abs(s.u[i]) + uref)
                                                : (vv == 1) ? (std::abs(s.p[i]) + 1.0e5)
                                                            : (std::abs(s.h[i]) + href);
                            eps[i] = 1e-7 * scale;
                        }
                        const int jlo = penta ? -2 : -1, jhi = penta ? 2 : 1;
                        for (int c = 0; c < 5; ++c) {
                            for (int j = c; j < n; j += 5) var(vv, j) += eps[j];
                            compute_R();
                            for (int i = 0; i < n; ++i)
                                for (int j = i + jlo; j <= i + jhi; ++j) {
                                    if (j < 0 || j >= n || j % 5 != c) continue;
                                    const double e = eps[j];
                                    const double d0 = (Rres[i][0] - R0[i][0]) / e;   // dRmom
                                    const double d1 = (Rres[i][1] - R0[i][1]) / e;   // dRcon
                                    const double d2 = (Rene[i] - R0e[i]) / e;        // dRene
                                    Mat3& M = (j == i - 1) ? MA3[i] : (j == i) ? MB3[i]
                                            : (j == i + 1) ? MC3[i] : (j == i - 2) ? ME3[i] : MF3[i];
                                    M[0][vv] = d0; M[1][vv] = d1; M[2][vv] = d2;
                                }
                            for (int j = c; j < n; j += 5) var(vv, j) -= eps[j];
                        }
                    }
                    compute_R();  // restore Rres = R0, Rene = R0e
                    jac_age = 0;
                }
                for (int i = 0; i < n; ++i) { Md3[i] = {-R0[i][0], -R0[i][1], -R0e[i]}; }

                // ===== ANALYTIC (Picard) Jacobian (ACID_AJAC): the transient via the EXACT EOS
                //   chain (stage-1 verified) + the acoustic flux coupling through theta(u,p) and
                //   pface(p); the upwind transport (raup/rbup/uconv/rHaup) + MWI dhat/clamp are
                //   FROZEN. Defect-correction keeps R exact, so a frozen-transport Jacobian still
                //   converges -- and it replaces the 15-compute_R FD assembly with one O(n) pass. =====
                if (ajac || ajblk) {
                    std::vector<Mat3> aA(n, Mat3{}), aB(n, Mat3{}), aC(n, Mat3{}), aE(n, Mat3{}), aF(n, Mat3{});
                    auto add = [&](int i, int m, int eq, int vv, double val) {
                        // FOLD ghost cells into the boundary interior cell -- transmissive/inlet
                        // ghost = front/back (zero-gradient), so a boundary face's derivative wrt
                        // the ghost belongs to the boundary cell (NOT dropped). This fixes the
                        // boundary Jacobian (the only place the analytic disagreed with FD).
                        if (m < 0) m = 0;
                        else if (m >= n) m = n - 1;
                        const int d = m - i;
                        Mat3* M = d == 0 ? &aB[i] : d == -1 ? &aA[i] : d == 1 ? &aC[i]
                                : d == -2 ? &aE[i] : d == 2 ? &aF[i] : nullptr;
                        if (M) (*M)[eq][vv] += val;
                    };
                    Vec dru(n), drp(n), drh(n);          // exact d(rho)/d(u,p,h)
                    Vec dTp(n), dTh(n), dTu(n);          // d(T)/d(p,h,u) from the h->T inversion
                    // Phase 2 Stage 1: d(alpha)/dp|_{T,Y} per cell. Zero unless the residual
                    // actually re-derives alpha inside the Newton (yadv && alpha_implicit, line
                    // ~1014). Filled here, CONSUMED by Stage 2's J2 flux-blend diagonal loop.
                    Vec alp_p(n, 0.0), alp_h(n, 0.0), alp_u(n, 0.0);
                    for (int i = 0; i < n; ++i) {
                        const double al = std::clamp(s.alpha[i], 0.0, 1.0);
                        const double p = std::max(s.p[i], 1.0), Tc = std::max(s.T[i], 1e-6), u = s.u[i];
                        const auto pa = phase_props(p, Tc, A); const auto pb = phase_props(p, Tc, B);
                        const double D = al * pa.rho + (1 - al) * pb.rho;
                        const double D_T = al * pa.phi + (1 - al) * pb.phi, D_p = al * pa.zeta + (1 - al) * pb.zeta;
                        const double N = al * pa.rho * pa.h + (1 - al) * pb.rho * pb.h;
                        const double N_T = al * (pa.phi * pa.h + pa.rho * pa.cp) + (1 - al) * (pb.phi * pb.h + pb.rho * pb.cp);
                        const double N_p = al * (pa.zeta * pa.h + pa.rho * pa.dh_dp) + (1 - al) * (pb.zeta * pb.h + pb.rho * pb.dh_dp);
                        // ---- Phase 2 Stage 1+3a (docs/YADV_PHASE2_PLAN.md 2.2 "J1") -----------
                        // Under ACID_YADV + ACID_YADV_ALPHA_IMPLICIT the residual re-derives
                        // alpha = alpha(Y, rho_a(p,T), rho_b(p,T)) at the CURRENT iterate (line
                        // ~1014), so d(alpha)/d(p,T)|_Y = (a_p, a_T) are NOT zero and the
                        // frozen-alpha D_p/N_p/D_T/N_T above are the wrong derivatives of the map
                        // compute_R actually evaluates. Measured defect (round 5 unit test,
                        // case15's state): D_p 1.00196e-06 -> D_p* 5.22580e-04, a factor 521.56.
                        // Star them with the product-rule addends (D = al*ra + (1-al)*rb  =>
                        // dD/dp gains (ra-rb)*a_p, dD/dT gains (ra-rb)*a_T; N = al*ra*ha +
                        // (1-al)*rb*hb  =>  dN/dp gains (ra*ha-rb*hb)*a_p, dN/dT gains the same
                        // times a_T).
                        // Round 8 (Stage 3a): the T-pathway. alpha is lagged one compute_R call
                        // in T (the alpha loop at ~1014 runs BEFORE the h->T inversion at ~1026),
                        // so starring here gives the FIXED-POINT derivative, not the derivative
                        // of the map as literally coded -- a deliberate, declared mismatch
                        // (YADV_PHASE2_PLAN.md sect.4 Stage 3a). Round-8 diagnostic (HSTDBG,
                        // removed after use) confirmed the motivating defect is real but confined
                        // to case14's very first timestep's Newton iterations (an interface-
                        // formation transient at one cell), never recurring afterward.
                        // hsT* = Y*cp_a + (1-Y)*cp_b EXACTLY (h_k is linear in T for NASG, and
                        // hstat_mix = N/D = Y*h_a + (1-Y)*h_b identically by Y's own definition)
                        // -- strictly positive, bounded in [min(cp_a,cp_b), max(cp_a,cp_b)],
                        // UNLIKE the unstarred hsT, which crosses zero for the air|water pair
                        // below ~78 K. Starring therefore also removes an existing 1/hsT
                        // near-singularity, not introduces one. Algebraically zero (not bitwise)
                        // for every b=0 phase pair -- 17 of 19 cases, perturbation <=1 ulp.
                        const bool aimp = yadv && alpha_implicit;
                        const bool aimpT = aimp && alpha_implicit_t;
                        const double ap = aimp ? dalpha_dp_massfrac(al, pa.zeta, pa.rho,
                                                                        pb.zeta, pb.rho) : 0.0;
                        const double aT = aimpT ? dalpha_dT_massfrac(al, pa.phi, pa.rho,
                                                                        pb.phi, pb.rho) : 0.0;
                        const double D_ps = aimp ? D_p + (pa.rho - pb.rho) * ap : D_p;
                        const double N_ps = aimp ? N_p + (pa.rho * pa.h - pb.rho * pb.h) * ap : N_p;
                        const double D_Ts = aimp ? D_T + (pa.rho - pb.rho) * aT : D_T;
                        const double N_Ts = aimp ? N_T + (pa.rho * pa.h - pb.rho * pb.h) * aT : N_T;
                        const double hsT = (N_Ts * D - N * D_Ts) / (D * D), hsp = (N_ps * D - N * D_ps) / (D * D);
                        dTh[i] = 1.0 / hsT; dTu[i] = -u / hsT; dTp[i] = -hsp / hsT;
                        drh[i] = D_Ts * dTh[i]; dru[i] = D_Ts * dTu[i]; drp[i] = D_ps + D_Ts * dTp[i];
                        // Total derivative of alpha_i wrt the Newton unknowns, for J2 (Stage 2's
                        // loop already consumes alp_p; extended this round to alp_h/alp_u).
                        // dTp/dTh/dTu already encode the h->T inversion's own sensitivity, so
                        // these are complete -- MUST be computed after hsT/hsp/dTp/dTh/dTu above.
                        alp_p[i] = ap + aT * dTp[i];
                        alp_h[i] = aT * dTh[i];
                        alp_u[i] = aT * dTu[i];
                    }
                    for (int i = 0; i < n; ++i) {  // transient + energy pressure source (diagonal)
                        const double b = bdf_c0[i] * VdT, rho = s.rho[i], u = s.u[i], h = s.h[i];
                        add(i, i, 0, 0, b * (rho + u * dru[i])); add(i, i, 0, 1, b * u * drp[i]); add(i, i, 0, 2, b * u * drh[i]);
                        add(i, i, 1, 0, b * dru[i]);             add(i, i, 1, 1, b * drp[i]);     add(i, i, 1, 2, b * drh[i]);
                        add(i, i, 2, 0, b * h * dru[i]); add(i, i, 2, 1, b * h * drp[i] - VdT); add(i, i, 2, 2, b * (rho + h * drh[i]));
                    }
                    const bool fixed0 = inlet_left || lbc == "reflective";
                    const bool fixedN = rbc == "reflective";
                    auto dth_dp = [&](int f, int cell) -> double {  // d(theta[f])/d(p[cell]), frozen dhat/clamp
                        if (!mwiOK_f[f] || (f == 0 && fixed0) || (f == n && fixedN)) return 0.0;
                        const double k = -dhat[f]; double d = 0.0;     // mwi_p = k*(dpf - gpbar)
                        if (cell == f)     d += 1.0 / dx - 1.0 / (4 * dx);
                        if (cell == f - 1) d += -1.0 / dx + 1.0 / (4 * dx);
                        if (cell == f - 2) d += 1.0 / (4 * dx);
                        if (cell == f + 1) d += -1.0 / (4 * dx);
                        return k * d;
                    };
                    auto dth_du = [&](int f, int cell) -> double {  // d(theta[f])/d(u[cell]) via ubar
                        if ((f == 0 && fixed0) || (f == n && fixedN)) return 0.0;
                        if (use4_f[f]) { if (cell == f - 2 || cell == f + 1) return -1.0 / 12;
                                         if (cell == f - 1 || cell == f) return 7.0 / 12; return 0.0; }
                        return (cell == f - 1 || cell == f) ? 0.5 : 0.0;
                    };
                    auto dpf_dp = [&](int f, int cell) -> double {  // d(pface[f])/d(p[cell])
                        if (use4_f[f]) { if (cell == f - 2 || cell == f + 1) return -1.0 / 12;
                                         if (cell == f - 1 || cell == f) return 7.0 / 12; return 0.0; }
                        return (cell == f - 1 || cell == f) ? 0.5 : 0.0;
                    };
                    for (int i = 0; i < n; ++i) {  // flux coupling (frozen transport)
                        const double al = std::clamp(s.alpha[i], 0.0, 1.0);
                        const double rblL = al * raup[i] + (1 - al) * rbup[i];
                        const double rblR = al * raup[i + 1] + (1 - al) * rbup[i + 1];
                        const double rHblL = al * rHaup[i] + (1 - al) * rHbup[i];
                        const double rHblR = al * rHaup[i + 1] + (1 - al) * rHbup[i + 1];
                        const double ucL = uconv[i], ucR = uconv[i + 1];
                        for (int cell = i - 2; cell <= i + 2; ++cell) {
                            const double tpL = dth_dp(i, cell), tpR = dth_dp(i + 1, cell);
                            const double tuL = dth_du(i, cell), tuR = dth_du(i + 1, cell);
                            const double ppL = dpf_dp(i, cell), ppR = dpf_dp(i + 1, cell);
                            add(i, cell, 1, 0, rblR * tuR - rblL * tuL);              // R_con d/du
                            add(i, cell, 1, 1, rblR * tpR - rblL * tpL);              // R_con d/dp
                            add(i, cell, 0, 0, rblR * tuR * ucR - rblL * tuL * ucL);  // R_mom d/du
                            add(i, cell, 0, 1, rblR * tpR * ucR - rblL * tpL * ucL + (ppR - ppL));  // R_mom d/dp
                            add(i, cell, 2, 0, rHblR * tuR - rHblL * tuL);            // R_ene d/du
                            add(i, cell, 2, 1, rHblR * tpR - rHblL * tpL);            // R_ene d/dp
                        }
                    }
                    // --- Phase 2 Stage 2 (docs/YADV_PHASE2_PLAN.md 2.2 "J2"): the OTHER
                    //     product-rule addend of the ACID per-cell flux blend. The residual
                    //     weights the UPWIND phase densities with cell i's OWN alpha
                    //     (mdotL/mdotR at ~1172-1179, energy fR/fL at ~1203-1207):
                    //         mdot_f^(i) = (al_i*raup_f  + (1-al_i)*rbup_f ) * theta_f
                    //         e_f^(i)    = (al_i*rHaup_f + (1-al_i)*rHbup_f) * theta_f
                    //     The flux-coupling block ABOVE differentiates theta/pface with al
                    //     FROZEN; the upwind-transport block BELOW differentiates
                    //     raup/rbup/rHaup/rHbup at the UPWIND cell with al FROZEN. Neither
                    //     differentiates al itself. Under ACID_YADV + ACID_YADV_ALPHA_IMPLICIT
                    //     al_i = alpha(Y_i, rho_a(p_i,T_i), rho_b(p_i,T_i)) (~1014), so
                    //     d(al_i)/dp_i = a_p != 0. alpha_i depends only on cell i, so this is a
                    //     purely DIAGONAL addend (aB[i]) -- no stencil growth. Denner 2018
                    //     Eq.1-2's coefficient*variable Newton template with BOTH factors
                    //     linearised (the flux-coupling block above is the coefficient-frozen
                    //     form). a_p is REUSED from J1's alp_p[] (filled at ~1535); never
                    //     recomputed here. Boundary faces need no special case: theta[] already
                    //     carries every BC override (theta[0]=uin for inlet, theta[0]/theta[n]=0
                    //     for reflective, ~1164-1166, set BEFORE the mdot loop), and the
                    //     mdotL[0]/mdotR[n-1] restatements at ~1177-1179 evaluate to exactly the
                    //     same product -- so plain theta[f] is the right factor everywhere.
                    //     TR-BDF2 would need the flux_w scaling on these rows, but
                    //     tr_bdf2 => ajac=false (~1274), so this block only ever runs at
                    //     flux_w == 1. Guarded by that fact, deliberately not by code.
                    if (yadv && alpha_implicit) {
                        // Stage 2 (p-column, ap) + Stage 3a (h/u columns, ah/au) of the SAME
                        // product-rule addend -- own-cell alpha's TOTAL sensitivity to every
                        // Newton unknown, in the ACID per-cell mass/momentum/energy flux blend.
                        for (int i = 0; i < n; ++i) {
                            const double ap = alp_p[i], ah = alp_h[i], au = alp_u[i];
                            const double dR = (raup[i + 1]  - rbup[i + 1] ) * theta[i + 1];
                            const double dL = (raup[i]      - rbup[i]     ) * theta[i];
                            const double eR = (rHaup[i + 1] - rHbup[i + 1]) * theta[i + 1];
                            const double eL = (rHaup[i]     - rHbup[i]    ) * theta[i];
                            const double mc = dR - dL;                                // R_con
                            const double mm = dR * uconv[i + 1] - dL * uconv[i];       // R_mom
                            const double me = eR - eL;                                // R_ene
                            add(i, i, 1, 1, mc * ap); add(i, i, 1, 2, mc * ah); add(i, i, 1, 0, mc * au);
                            add(i, i, 0, 1, mm * ap); add(i, i, 0, 2, mm * ah); add(i, i, 0, 0, mm * au);
                            add(i, i, 2, 1, me * ap); add(i, i, 2, 2, me * ah); add(i, i, 2, 0, me * au);
                        }
                    }
                    // --- upwind-TRANSPORT derivatives (1st-order: weight 1 on the upwind cell).
                    //     d(raup/rbup/rHaup)/d(p,h,u)[uw] via the EOS partials + the h->T chain.
                    //     Needed where the upwind density varies sharply (strong shock-interface
                    //     case25, cavitation case15) -- frozen transport diverges there. ---
                    for (int f = 0; f <= n; ++f) {
                        const int uw = uwc_f[f];
                        const double pu = std::max(s.p[uw], 1.0), Tu = std::max(s.T[uw], 1e-6);
                        const double uu = uconv[f], th = theta[f];
                        const auto pa = phase_props(pu, Tu, A); const auto pb = phase_props(pu, Tu, B);
                        const double drap = pa.zeta + pa.phi * dTp[uw], drah = pa.phi * dTh[uw], drau = pa.phi * dTu[uw];
                        const double drbp = pb.zeta + pb.phi * dTp[uw], drbh = pb.phi * dTh[uw], drbu = pb.phi * dTu[uw];
                        const double dhap = pa.dh_dp + pa.cp * dTp[uw], dhah = pa.cp * dTh[uw], dhau = pa.cp * dTu[uw];
                        const double dhbp = pb.dh_dp + pb.cp * dTp[uw], dhbh = pb.cp * dTh[uw], dhbu = pb.cp * dTu[uw];
                        const double Hk = 0.5 * uu * uu, Ha = pa.h + Hk, Hb = pb.h + Hk, rA = raup[f], rB = rbup[f];
                        const double drHap = drap * Ha + rA * dhap, drHah = drah * Ha + rA * dhah, drHau = drau * Ha + rA * (dhau + uu);
                        const double drHbp = drbp * Hb + rB * dhbp, drHbh = drbh * Hb + rB * dhbh, drHbu = drbu * Hb + rB * (dhbu + uu);
                        for (int side = 0; side < 2; ++side) {
                            const int ci = (side == 0) ? f - 1 : f;   // f = RIGHT face of f-1, LEFT face of f
                            if (ci < 0 || ci >= n) continue;
                            const double sgn = (side == 0) ? 1.0 : -1.0;
                            const double al = std::clamp(s.alpha[ci], 0.0, 1.0);
                            const double mp = th * (al * drap + (1 - al) * drbp);   // d(mdot)/dp via rbl
                            const double mh = th * (al * drah + (1 - al) * drbh);
                            const double mu = th * (al * drau + (1 - al) * drbu);
                            const double mdot = th * (al * rA + (1 - al) * rB);
                            add(ci, uw, 1, 1, sgn * mp); add(ci, uw, 1, 2, sgn * mh); add(ci, uw, 1, 0, sgn * mu);
                            add(ci, uw, 0, 1, sgn * mp * uu); add(ci, uw, 0, 2, sgn * mh * uu);
                            add(ci, uw, 0, 0, sgn * (mu * uu + mdot));   // + mdot*d(uconv)/du[uw]=1
                            add(ci, uw, 2, 1, sgn * th * (al * drHap + (1 - al) * drHbp));
                            add(ci, uw, 2, 2, sgn * th * (al * drHah + (1 - al) * drHbh));
                            add(ci, uw, 2, 0, sgn * th * (al * drHau + (1 - al) * drHbu));
                        }
                    }
                    // --- frozen-MWI sensitivity: theta[f] depends on rho[f-1],rho[f] through dhat,
                    //     rho_f and the transient-memory term (all FROZEN above). Add d(theta)/d(rho)
                    //     chained to d(rho)/d(u,h,p) -- supplies the i+-1 dene/dh, dene/dp, dmom/dh
                    //     couplings the block diagnostic showed missing (e.g. case24 C dene/dh ~2e4). ---
                    for (int f = 0; f <= n; ++f) {
                        if ((f == 0 && fixed0) || (f == n && fixedN)) continue;
                        const int cL = std::clamp(f - 1, 0, n - 1), cR = std::clamp(f, 0, n - 1);
                        const double R1 = std::max(s.rho[cL], 1e-300), R2 = std::max(s.rho[cR], 1e-300);
                        const double d_f = dx / std::max(0.5 * (R1 + R2) * dx / dt, 1e-300);
                        const double rf = rho_f[f], dhh = dhat[f], Dd = 1.0 + (rf / dt) * d_f;
                        const double invD2 = 1.0 / (Dd * Dd);
                        const double ddf = -d_f / (R1 + R2);                  // d(d_f)/dRk (same for R1,R2)
                        const double drf1 = rf * rf / (2.0 * R1 * R1), drf2 = rf * rf / (2.0 * R2 * R2);
                        const double ddh1 = (ddf - (d_f * d_f / dt) * drf1) * invD2;  // d(dhat)/dR1
                        const double ddh2 = (ddf - (d_f * d_f / dt) * drf2) * invD2;  // d(dhat)/dR2
                        const double tmc = (theta_o[f] - 0.5 * (uu_o[cL] + uu_o[cR])) / dt;  // transient-memory
                        const double mw = mwiOK_f[f] ? -dpgpf[f] : 0.0;       // d(mwi_p)/d(dhat)
                        const double dThdR1 = mw * ddh1 + tmc * (drf1 * dhh + rf * ddh1);
                        const double dThdR2 = mw * ddh2 + tmc * (drf2 * dhh + rf * ddh2);
                        auto addface = [&](int ic, double sg) {  // theta[f]: R[f-1] right face +, R[f] left face -
                            if (ic < 0 || ic >= n) return;
                            const double al = std::clamp(s.alpha[ic], 0.0, 1.0);
                            const double rbl = al * raup[f] + (1.0 - al) * rbup[f];
                            const double rHbl = al * rHaup[f] + (1.0 - al) * rHbup[f];
                            const double ccon = sg * rbl, cmom = sg * rbl * uconv[f], cene = sg * rHbl;
                            for (int side = 0; side < 2; ++side) {
                                const int sc = side == 0 ? cL : cR;
                                const double dth = side == 0 ? dThdR1 : dThdR2;
                                const double du_ = dth * dru[sc], dp_ = dth * drp[sc], dh_ = dth * drh[sc];
                                add(ic, sc, 1, 0, ccon * du_); add(ic, sc, 1, 1, ccon * dp_); add(ic, sc, 1, 2, ccon * dh_);
                                add(ic, sc, 0, 0, cmom * du_); add(ic, sc, 0, 1, cmom * dp_); add(ic, sc, 0, 2, cmom * dh_);
                                add(ic, sc, 2, 0, cene * du_); add(ic, sc, 2, 1, cene * dp_); add(ic, sc, 2, 2, cene * dh_);
                            }
                        };
                        addface(f - 1, 1.0);
                        addface(f, -1.0);
                    }
                    if (ajblk) {
                        // localise where the analytic Jacobian disagrees with FD (the missing term)
                        const char* se = std::getenv("ACID_BLK_STEP");
                        const int rstep = se ? std::atoi(se) : 40;
                        if (step == rstep && it == 0) {
                            auto rep = [&](const char* nm, const std::vector<Mat3>& AN, const std::vector<Mat3>& FB) {
                                if (AN.empty() || FB.empty()) return;
                                const char* eqn[3] = {"mom", "con", "ene"};
                                const char* vn[3] = {"u", "p", "h"};
                                for (int eq = 0; eq < 3; ++eq) for (int vv = 0; vv < 3; ++vv) {
                                    double md = 0, mf = 0; int ic = -1;
                                    for (int i = 0; i < n; ++i) {
                                        const double d = std::abs(AN[i][eq][vv] - FB[i][eq][vv]);
                                        if (d > md) { md = d; ic = i; }
                                        mf = std::max(mf, std::abs(FB[i][eq][vv]));
                                    }
                                    const double rel = md / std::max(mf, 1e-300);
                                    if (rel > 1e-2)
                                        std::fprintf(stderr, "BLK %s d%s/d%s maxdiff=%.3e FDmag=%.3e rel=%.2e @i=%d\n",
                                                     nm, eqn[eq], vn[vv], md, mf, rel, ic);
                                }
                            };
                            std::fprintf(stderr, "=== AJAC-vs-FD block diff, case %s step %d (rel>1e-2) ===\n",
                                         c.id.c_str(), step);
                            rep("B", aB, MB3); rep("A", aA, MA3); rep("C", aC, MC3);
                            if (penta) { rep("E", aE, ME3); rep("F", aF, MF3); }
                        }
                        // ajblk: keep FD blocks for the actual solve (non-destructive diagnosis)
                    } else {
                        MA3 = aA; MB3 = aB; MC3 = aC;
                        if (penta) { ME3 = aE; MF3 = aF; }
                    }
                }

                const auto dxk = penta ? block_penta(ME3, MA3, MB3, MC3, MF3, Md3)
                                       : block_thomas3(MA3, MB3, MC3, Md3);
                if (dbg) {
                    for (int i = 0; i < n; ++i)
                        if (!std::isfinite(dxk[i][0]) || !std::isfinite(dxk[i][1]) || !std::isfinite(dxk[i][2])) {
                            std::fprintf(stderr,
                                "C3 NaN solve i=%d al=%.4f p=%.3e T=%.2f h=%.3e rho=%.3e "
                                "R=[%.3e,%.3e,%.3e]\n",
                                i, s.alpha[i], s.p[i], s.T[i], s.h[i], s.rho[i],
                                Rres[i][0], Rres[i][1], Rene[i]);
                            break;
                        }
                }
                if (std::getenv("ACID_NJAC3") && step == 0 && it == 0) {
                    int i0 = n / 2;
                    std::fprintf(stderr, "NJAC3 i0=%d al=%.3f rho=%.3e h=%.3e du=%.3e dp=%.3e dh=%.3e\n",
                                 i0, s.alpha[i0], s.rho[i0], s.h[i0], dxk[i0][0], dxk[i0][1], dxk[i0][2]);
                }

                // --- backtracking line search on (u,p,h): accept first alpha reducing ||R3||.
                //     clamps: |dp|<=50% p, |du|<=uref, |dh|<=50% |h| (or 0.5*href if h tiny);
                //     p>=1; T re-derived from h inside compute_R (clamped 1e-6..1e6). ---
                const double n0 = rnorm3();
                if (it == 0) r_init = n0;  // residual at the step start (for the progress test)
                const Field sbak = s;
                double du = 0.0, dp = 0.0, dh = 0.0;
                double al_acc = 1.0;
                for (double al = 1.0;; al *= 0.5) {
                    al_acc = al;
                    du = 0.0; dp = 0.0; dh = 0.0;
                    for (int i = 0; i < n; ++i) {
                        const double dpi = std::clamp(al * om * dxk[i][1], -0.5 * sbak.p[i], 0.5 * sbak.p[i]);
                        const double dui = std::clamp(al * om * dxk[i][0], -uref, uref);
                        const double hlim = 0.5 * std::max(std::abs(sbak.h[i]), href);
                        const double dhi = std::clamp(al * om * dxk[i][2], -hlim, hlim);
                        du = std::max(du, std::abs(dui));
                        dp = std::max(dp, std::abs(dpi));
                        dh = std::max(dh, std::abs(dhi));
                        s.u[i] = sbak.u[i] + dui;
                        s.p[i] = std::max(sbak.p[i] + dpi, 1.0);
                        // keep total enthalpy above the kinetic floor so hstat = h - 1/2 u^2 > 0
                        const double hfloor = 0.5 * s.u[i] * s.u[i] * 1.0001 + 1.0;
                        s.h[i] = std::max(sbak.h[i] + dhi, hfloor);
                    }
                    compute_R();  // re-derives T from h, eval_thermo, fills Rres+Rene
                    if (rnorm3() < n0 || al < 0.03) break;
                }
                backtracked_last = (al_acc < 1.0);  // modified-Newton: reassemble next iter if not a full step
                if (std::getenv("ACID_RHIST")) {
                    const char* se = std::getenv("ACID_BLK_STEP"); const int rs = se ? std::atoi(se) : 2;
                    if (step == rs) std::fprintf(stderr, "RHIST it=%d n0=%.4e -> %.4e al=%.3f du=%.2e dp=%.2e dh=%.2e\n",
                                                 it, n0, rnorm3(), al_acc, du, dp, dh);
                }
                // energy is now INSIDE the Newton -> NO segregated T update. T already set
                // consistently with h by compute_R.
                eval_thermo(s, A, B);
                double pscale = 1.0;
                for (int i = 0; i < n; ++i) pscale = std::max(pscale, s.p[i]);
                // keep-best + stall-break are AJAC-only: the FD default (committed 10/10) relies on
                // running to the iter cap and accepting the last iterate -- do not change it.
                if (ajac) { const double rk = rnorm3(); if (rk < rbest) { rbest = rk; s_best = s; best_it = it; } }
                if (du < 1e-8 * std::max(lam, 1.0) && dp < 1e-8 * pscale
                    && dh < 1e-8 * std::max(href, 1.0)) { conv_inner = true; break; }
                if (ajac && it - best_it >= stallwin) break;  // residual stalled -> accept best below
                continue;  // skip the 2x2 path for this iteration
            }
            // ============================================================================

            // --- NUMERICAL (finite-difference) block-tridiagonal Jacobian ---
            //   The hand-derived analytic Jacobian had bugs (verified non-descent via NJAC).
            //   Build the EXACT tridiagonal Jacobian by FD with stride-5 graph colouring:
            //   for each colour, perturb cells j with j%5==c (their +-2 stencils don't overlap
            //   -> exactly one perturbed cell affects any R_i), compute R, extract dR_i/dvar_j
            //   into MA/MB/MC. 5 colours x 2 vars = 10 compute_R per Newton iteration. (The
            //   small pentadiagonal MWI i+-2 coupling is dropped, matching the tridiagonal solve.)
            std::vector<Mat2> MA(n, Mat2{}), MB(n, Mat2{}), MC(n, Mat2{});
            std::vector<Vec2> Md(n);
            if (std::getenv("ACID_NUMJAC") != nullptr) {
            const std::vector<Vec2> R0 = Rres;  // baseline (compute_R already filled Rres)
            Vec eps(n);
            for (int vv = 0; vv < 2; ++vv) {
                for (int i = 0; i < n; ++i)
                    eps[i] = 1e-7 * (std::abs(vv == 0 ? s.u[i] : s.p[i]) + (vv == 0 ? uref : 1.0e5));
                for (int c = 0; c < 5; ++c) {
                    for (int j = c; j < n; j += 5) (vv == 0 ? s.u[j] : s.p[j]) += eps[j];
                    compute_R();
                    for (int i = 0; i < n; ++i)
                        for (int j = i - 1; j <= i + 1; ++j) {
                            if (j < 0 || j >= n || j % 5 != c) continue;
                            const double e = eps[j];
                            const double d0 = (Rres[i][0] - R0[i][0]) / e;
                            const double d1 = (Rres[i][1] - R0[i][1]) / e;
                            Mat2& M = (j == i - 1) ? MA[i] : (j == i) ? MB[i] : MC[i];
                            M[0][vv] = d0; M[1][vv] = d1;
                        }
                    for (int j = c; j < n; j += 5) (vv == 0 ? s.u[j] : s.p[j]) -= eps[j];
                }
            }
            compute_R();  // restore Rres = R0
            for (int i = 0; i < n; ++i) { Md[i][0] = -R0[i][0]; Md[i][1] = -R0[i][1]; }
            } else {
            // analytic Jacobian (DEFAULT): fast; passes 01,02,04,05,13. Has known bugs vs the
            // NJAC finite-difference check at strong shocks (missing dRmom/dp, dRcon/du diagonal
            // & i-1 couplings) -> 07/25 diverge with it. The numerical Jacobian (ACID_NUMJAC)
            // is exact and fixes 07 but is 10x slower and currently regresses 02. TODO: correct
            // these analytic terms (use NJAC values) for a fast+correct Jacobian.
            for (int i = 0; i < n; ++i) {
                MB[i][0][0] += bdf_c0[i] *std::max(s.rho[i], rho_floor) * VdT;
                const double mR = mdotR[i], mL = mdotL[i];
                if (mR >= 0.0) MB[i][0][0] += mR; else MC[i][0][0] += mR;
                if (mL >= 0.0) MA[i][0][0] += -mL; else MB[i][0][0] += -mL;
                const double aJ = std::clamp(s.alpha[i], 0.0, 1.0);
                const double rfR = aJ * raup[i + 1] + (1.0 - aJ) * rbup[i + 1];
                const double rfL = aJ * raup[i] + (1.0 - aJ) * rbup[i];
                MB[i][0][0] += 0.5 * rfR * uconv[i + 1];
                if (i + 1 < n) MC[i][0][0] += 0.5 * rfR * uconv[i + 1]; else MB[i][0][0] += 0.5 * rfR * uconv[i + 1];
                if (i - 1 >= 0) MA[i][0][0] += -0.5 * rfL * uconv[i]; else MB[i][0][0] += -0.5 * rfL * uconv[i];
                MB[i][0][0] += -0.5 * rfL * uconv[i];
                if (i + 1 < n) MC[i][0][1] += 0.5; else MB[i][0][1] += 0.5;
                if (i - 1 >= 0) MA[i][0][1] += -0.5; else MB[i][0][1] += -0.5;
                MB[i][0][1] += s.drhodp[i] * (std::max(theta[i + 1], 0.0) * uconv[i + 1] - std::min(theta[i], 0.0) * uconv[i]);
                // momentum dR/dp via the MWI (Rhie-Chow) pressure-velocity coupling: theta_f
                // depends on the pressure gradient, so the momentum convection depends on p.
                // d theta_f/dp_face = (3/4) dhat_f/dx (compact 1/dx minus the averaged cell
                // gradient 1/4dx). Plus the transient d(rho(p) u)/dp. NJAC-checked. THIS is the
                // term missing before -> 07 had a non-descent direction with the analytic J.
                MB[i][0][1] += bdf_c0[i] *s.drhodp[i] * s.u[i] * VdT;  // transient d(rho u)/dp
                { const double cR = 0.75 * rfR * dhat[i + 1] * uconv[i + 1] / dx;
                  const double cL = 0.75 * rfL * dhat[i] * uconv[i] / dx;
                  MB[i][0][1] += cR + cL;
                  if (i + 1 < n) MC[i][0][1] += -cR; else MB[i][0][1] += -cR;
                  if (i - 1 >= 0) MA[i][0][1] += -cL; else MB[i][0][1] += -cL; }
                MB[i][1][1] += bdf_c0[i] *VdT * s.drhodp[i];
                { const double rf = rho_f[i + 1], dh = dhat[i + 1]; MB[i][1][1] += rf * dh / dx; if (i + 1 < n) MC[i][1][1] += -rf * dh / dx; else MB[i][1][1] += -rf * dh / dx; }
                { const double rf = rho_f[i], dh = dhat[i]; MB[i][1][1] += rf * dh / dx; if (i - 1 >= 0) MA[i][1][1] += -rf * dh / dx; else MB[i][1][1] += -rf * dh / dx; }
                MB[i][1][1] += s.drhodp[i] * (std::max(theta[i + 1], 0.0) + std::max(-theta[i], 0.0));
                // continuity dR/du : d(mdotR - mdotL)/du with the ACID face densities rfR/rfL
                // (NOT the harmonic rho_f). The u_i term 0.5(rfR-rfL) does NOT cancel at a
                // shock/interface where rfR != rfL -- verified vs NJAC (matched -2.73).
                MB[i][1][0] += 0.5 * (rfR - rfL);
                if (i + 1 < n) MC[i][1][0] += 0.5 * rfR; else MB[i][1][0] += 0.5 * rfR;
                if (i - 1 >= 0) MA[i][1][0] += -0.5 * rfL; else MB[i][1][0] += -0.5 * rfL;
                Md[i][0] = -Rres[i][0];
                Md[i][1] = -Rres[i][1];
            }
            }

            if (std::getenv("ACID_NJAC") && step == 0 && it == 0) {
                int i0 = 1;
                for (int i = 1; i < n - 1; ++i)
                    if (std::abs(Rres[i][0]) > std::abs(Rres[i0][0])) i0 = i;
                std::vector<Vec2> R0 = Rres;  // baseline residual
                auto col = [&](int j, int v) -> Vec2 {  // d R_i0 / d(var v at cell j), v=0:u 1:p
                    double& x = (v == 0) ? s.u[j] : s.p[j];
                    const double save = x;
                    const double eps = 1e-6 * (std::abs(save) + (v == 0 ? 1.0 : 1.0e5));
                    x = save + eps;
                    compute_R();
                    Vec2 d{(Rres[i0][0] - R0[i0][0]) / eps, (Rres[i0][1] - R0[i0][1]) / eps};
                    x = save;
                    return d;
                };
                const Vec2 aU = col(i0 - 1, 0), aP = col(i0 - 1, 1);
                const Vec2 bU = col(i0, 0), bP = col(i0, 1);
                const Vec2 cU = col(i0 + 1, 0), cP = col(i0 + 1, 1);
                compute_R();  // restore
                std::fprintf(stderr, "NJAC i0=%d al=%.3f rho=%.3e\n", i0, s.alpha[i0], s.rho[i0]);
                std::fprintf(stderr, " MB ana[[%.2e,%.2e][%.2e,%.2e]] num[[%.2e,%.2e][%.2e,%.2e]]\n",
                    MB[i0][0][0], MB[i0][0][1], MB[i0][1][0], MB[i0][1][1], bU[0], bP[0], bU[1], bP[1]);
                std::fprintf(stderr, " MC ana[[%.2e,%.2e][%.2e,%.2e]] num[[%.2e,%.2e][%.2e,%.2e]]\n",
                    MC[i0][0][0], MC[i0][0][1], MC[i0][1][0], MC[i0][1][1], cU[0], cP[0], cU[1], cP[1]);
                std::fprintf(stderr, " MA ana[[%.2e,%.2e][%.2e,%.2e]] num[[%.2e,%.2e][%.2e,%.2e]]\n",
                    MA[i0][0][0], MA[i0][0][1], MA[i0][1][0], MA[i0][1][1], aU[0], aP[0], aU[1], aP[1]);
            }
            const auto dxk = block_thomas(MA, MB, MC, Md);
            if (std::getenv("ACID_LDBG") && step == 0 && it < 2) {
                // measure ||A*dxk - Md|| : if ~0, block-Thomas solves the linear system
                // EXACTLY -> ILU/GMRES/AMG would give the same dxk -> would NOT fix divergence.
                double lr = 0.0, rhs = 0.0;
                for (int i = 0; i < n; ++i) {
                    Vec2 Ax = mul(MB[i], dxk[i]);
                    if (i > 0) { Vec2 tL = mul(MA[i], dxk[i - 1]); Ax[0] += tL[0]; Ax[1] += tL[1]; }
                    if (i < n - 1) { Vec2 tR = mul(MC[i], dxk[i + 1]); Ax[0] += tR[0]; Ax[1] += tR[1]; }
                    lr = std::max(lr, std::max(std::abs(Ax[0] - Md[i][0]), std::abs(Ax[1] - Md[i][1])));
                    rhs = std::max(rhs, std::max(std::abs(Md[i][0]), std::abs(Md[i][1])));
                }
                std::fprintf(stderr, "LDBG it=%d linear-residual=%.3e RHS=%.3e ratio=%.3e\n",
                             it, lr, rhs, lr / std::max(rhs, 1e-300));
            }
            if (dbg) {
                for (int i = 0; i < n; ++i)
                    if (!std::isfinite(dxk[i][0]) || !std::isfinite(dxk[i][1])) {
                        std::fprintf(stderr,
                            "NaN at solve i=%d al=%.5f p=%.3e T=%.2f rho=%.3e a=%.3e "
                            "drhodp=%.3e R=[%.3e,%.3e] B=[[%.3e,%.3e],[%.3e,%.3e]] mdotL=%.3e mdotR=%.3e\n",
                            i, s.alpha[i], s.p[i], s.T[i], s.rho[i], s.a[i], s.drhodp[i],
                            Rres[i][0], Rres[i][1], MB[i][0][0], MB[i][0][1], MB[i][1][0], MB[i][1][1],
                            mdotL[i], mdotR[i]);
                        break;
                    }
            }
            if (std::getenv("ACID_JDBG") && step == 0 && it < 2) {
                int im = 0, ic = 0, idu = 0, idp = 0;
                for (int i = 0; i < n; ++i) {
                    if (std::abs(Rres[i][0]) > std::abs(Rres[im][0])) im = i;
                    if (std::abs(Rres[i][1]) > std::abs(Rres[ic][1])) ic = i;
                    if (std::abs(dxk[i][0]) > std::abs(dxk[idu][0])) idu = i;
                    if (std::abs(dxk[i][1]) > std::abs(dxk[idp][1])) idp = i;
                }
                std::fprintf(stderr,
                    "JDBG it=%d Rmom max@%d=%.3e(al=%.2f rho=%.2e) Rcon max@%d=%.3e(al=%.2f) "
                    "du max@%d=%.3e(al=%.2f) dp max@%d=%.3e(p=%.2e)\n",
                    it, im, Rres[im][0], s.alpha[im], s.rho[im], ic, Rres[ic][1], s.alpha[ic],
                    idu, dxk[idu][0], s.alpha[idu], idp, dxk[idp][1], s.p[idp]);
            }
            // ---- backtracking line search: accept the first alpha that reduces ||R||
            //      (globalises Newton; with the exact numerical Jacobian -> robust at shocks).
            //      Step limits: |dp|<=50% of p, |du|<=uref, positivity |u|<sqrt(2 h_total). ----
            const double n0 = rnorm();        // ||R|| at the current state (Rres = R0)
            const Field sbak = s;             // saved state to step from
            double du = 0.0, dp = 0.0;
            for (double al = 1.0;; al *= 0.5) {
                du = 0.0; dp = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double dpi = std::clamp(al * om * dxk[i][1], -0.5 * sbak.p[i], 0.5 * sbak.p[i]);
                    const double dui = std::clamp(al * om * dxk[i][0], -uref, uref);
                    du = std::max(du, std::abs(dui));
                    dp = std::max(dp, std::abs(dpi));
                    const double htot = sbak.hstat[i] + 0.5 * sbak.u[i] * sbak.u[i];
                    const double umax = 0.99 * std::sqrt(2.0 * std::max(htot, 1.0e-300));
                    s.u[i] = std::clamp(sbak.u[i] + dui, -umax, umax);
                    s.p[i] = std::max(sbak.p[i] + dpi, 1.0);
                }
                compute_R();  // ||R|| at the trial (T held fixed during the line search)
                if (rnorm() < n0 || al < 0.03) break;  // first reducing alpha, or give up at smallest
            }
            // ---- segregated energy: enthalpy eq  d(rhoH)/dt + d(mdot H)/dx = dp/dt ----
            //   H = total enthalpy = h_static + 1/2 u^2 ; RHS source = (p^{n+1}-p^o)/dt.
            eval_thermo(s, A, B);
            if (std::getenv("ACID_ISOTHERMAL") == nullptr) {
                for (int i = 0; i < n; ++i) {
                    // ACID enthalpy flux (Eq.47, 1st-order): theta_f * [alpha_i*(rho_a H_a)_up
                    // + (1-alpha_i)*(rho_b H_b)_up] -- the partial total-enthalpy fluxes are
                    // ACID-blended with the discretised cell's alpha_i, consistent with the
                    // mass flux, so a mixed interface cell stays energy-consistent (no T blowup).
                    const double ai = std::clamp(s.alpha[i], 0.0, 1.0);
                    const double fR = theta[i + 1] * (ai * rHaup[i + 1] + (1.0 - ai) * rHbup[i + 1]);
                    const double fL = theta[i] * (ai * rHaup[i] + (1.0 - ai) * rHbup[i]);
                    const double adv = fR - fL;
                    // BE/BDF2 segregated energy: (bdf_c0*rhoH - Cold_ene)*VdT + adv = (p-p_o)*VdT
                    //   -> rhoH = (Cold_ene - adv*dt/dx + (p - p_o)) / bdf_c0
                    const double rhoH = (Cold_ene[i] - dt / dx * adv + (s.p[i] - p_o[i])) / bdf_c0[i];
                    const double Hstat = rhoH - 0.5 * s.rho[i] * s.u[i] * s.u[i];  // rho * h_static
                    // mixture: rho*h_static = T*sum(al_k rho_k gamma_k cv_k) + sum(al_k rho_k (b_k p + eta_k))
                    const double p = std::max(s.p[i], 1.0), al = std::clamp(s.alpha[i], 0.0, 1.0);
                    const auto pa = phase_props(p, std::max(s.T[i], 1e-6), A);
                    const auto pb = phase_props(p, std::max(s.T[i], 1e-6), B);
                    const double nonT = al * pa.rho * (A.b * p + A.eta) + (1.0 - al) * pb.rho * (B.b * p + B.eta);
                    const double cpT = al * pa.rho * A.gamma * A.kv + (1.0 - al) * pb.rho * B.gamma * B.kv;
                    const double Traw = (Hstat - nonT) / std::max(cpT, 1e-300);
                    // REJECT a non-physical update (Hstat < kinetic -> Traw <= 0): a transient
                    // iteration inconsistency (EOS rho vs conservative rho before convergence)
                    // that, if floored to 1e-6, makes rho explode -> divergence (this was the
                    // case25 NaN root cause). Keep the old T there. Under-relax physical updates
                    // for shock stability; at convergence (du,dp->0) T == Traw (physical).
                    if (std::isfinite(Traw) && Traw > 1e-6)
                        s.T[i] += 0.5 * (std::min(Traw, 1.0e6) - s.T[i]);
                }
            }
            eval_thermo(s, A, B);
            // RELATIVE convergence: the old "dp < 1e-3" (absolute Pa) never triggered for
            // shocks (p~1e7) -> 60 inner iters every step (slow). Normalise by the state.
            double pscale = 1.0;
            for (int i = 0; i < n; ++i) pscale = std::max(pscale, s.p[i]);
            // 1e-8 relative: = the old (tight) absolute 1e-3 Pa for acoustic p~1e5 (keeps
            // 04/05 accuracy) but RELATIVE so it also triggers for shocks p~1e7 (faster).
            if (du < 1e-8 * std::max(lam, 1.0) && dp < 1e-8 * pscale) break;
        }
        // accept the BEST (lowest-residual) iterate when the inner Newton stalled (didn't converge)
        // -- the last iterate of a stalled line search can be worse than an earlier one.
        if (ajac && coupled && !conv_inner && best_it >= 0) s = s_best;
        if (tr_bdf2 && stage == 0) {
            // capture the stage-1 conserved quantities phi_g (= state at t+gamma*dt) for stage 2's
            // BDF2 Cold, and advance the MWI memory (theta_o, uu_o) to that intermediate level.
            for (int i = 0; i < n; ++i) {
                trPhiG_m[i] = s.rho[i] * s.u[i];
                trPhiG_c[i] = s.rho[i];
                trPhiG_e[i] = s.rho[i] * s.h[i] - s.p[i];
            }
            uu_o = s.u;
            store_theta_o(trg * dt);
        }
        }  // ===== end TR-BDF2 stage loop =====
        // adaptive-dt: accept the step only if it stayed finite & bounded, else halve dt.
        // conv-guard (ajac): dt-retry a non-converged step ONLY when it made NO progress at all
        // (rbest >= r_init) -- that means dt is too large (case24/25 at a violent step), so the
        // cfl-ramp shrinks dt until it converges. If it made progress but stalled short of the gate
        // (case15 cavitation: the line search pins at the al floor at ANY dt), ACCEPT the best
        // iterate like the FD path does -- retrying would just collapse dt and freeze the run.
        bool bad = (ajac && coupled && !conv_inner && rbest >= r_init);
        if (bad) { stall_reason = 1; stall_cell = -1; }
        // ---- F2'' (round 18, promoted unconditional round 20): T-ceiling saturation is a
        // STALL, not a step. ----
        // A cell at T_from_hstat's 1e6 clamp is not a solution of hmix(T)=hstat: dT/dh is exactly
        // 0 there, so Newton's energy unknown moves and the thermodynamic state does not respond
        // (docs/YADV_RESEARCH.md sect.26.1 -- the mechanism behind case33's unrecoverable stall).
        // Accepting such an iterate silently propagates a state the EOS could not represent.
        // PLACEMENT is load-bearing, do not move:
        //   * AFTER the reason-1 assignment above -> reason 5 DISPLACES reason 1, which is what
        //     makes a saturated retry ineligible for ACID_STALL_ACCEPT below with NO edit there
        //     (a captured acc_s candidate is thus non-saturated by construction);
        //   * BEFORE the finite/speed scan below -> a hard non-finite/overspeed failure still
        //     wins the STALLED-DETAIL report. Precedence: 2/3/4 > 5 > 1.
        // NOT ajac-gated (unlike the reason-1 term above): saturation is a property of the STATE,
        // not of the Jacobian mode. `coupled`-gated to match ACID_TSAT block A: on the segregated
        // path the convex T-update can never REACH 1e6 from below, so this is equivalent, but the
        // gate states the intent. Integer/compare only -- no FP arithmetic is added.
        // UNCONDITIONAL since round 20 (docs/YADV_RESEARCH.md sect.30): this block was gated
        // behind an opt-in env flag through round 19 for the same reason the ACID_TSAT probe
        // existed -- the ACID_YADV / +ALPHA_IMPLICIT / FD-invariance paths had never been swept
        // for saturation before round 18. Round 18's own sweep (all 19 graded OFF cases,
        // calls_hi==0) and round 20's 7-config battery (A-G, both ACID_STALL_ACCEPT levels)
        // proved the old flag's own no-op claim on every published path: turning it on changed
        // NOTHING except case33/34 under +ALPHA_IMPLICIT, where it turns a silent NaN divergence
        // into an earlier, correctly-typed STALLED-DETAIL report. Per round 14's precedent for
        // correctness fixes (the sibling `diverged=true` fix shipped with no opt-out), this is no
        // longer optional: reason 5 is a real solver defect (a state the EOS cannot represent),
        // not a research toggle. Last commit where the old opt-in flag still existed: ea38c04.
        if (coupled) {
            for (int i = 0; i < n; ++i)
                if (s.T[i] >= 1.0e6) { bad = true; stall_reason = 5; stall_cell = i; break; }
        }
        for (int i = 0; i < n; ++i)
            if (!std::isfinite(s.p[i]) || !std::isfinite(s.u[i]) ||
                std::abs(s.u[i]) > 10.0 * uref) {
                bad = true;
                stall_reason = !std::isfinite(s.p[i]) ? 2 : (!std::isfinite(s.u[i]) ? 3 : 4);
                stall_cell = i;
                break;
            }
        if (!bad) {
            stepped = true;
            // ACID_TSAT block B (round 17): does the ACCEPTED (clean) state carry a ceiling cell?
            if (tsat) {
                int nhi = 0, ihi = -1;
                for (int i = 0; i < n; ++i) if (s.T[i] >= 1.0e6) { ++nhi; if (ihi < 0) ihi = i; }
                if (nhi) {
                    ++tsat_steps_hi;
                    std::fprintf(stderr,
                        "TSAT-ACCEPT case=%s step=%d retry=%d ncells=%d i0=%d\n",
                        c.id.c_str(), step, retry, nhi, ihi);
                }
            }
            // BDF2 bookkeeping: the level-n (OLD-level for this step) conserved quantities
            // become the SECOND-old level (phi_o2) for the NEXT step. Captured here while
            // rho_o/u_o/Htot_o are still in scope; dt_prev records the actually-used dt so
            // the next step's constant-step BDF2 check is exact. (Cheap; runs even when
            // bdf2 is off -- the next step only consumes the o2 store if use_bdf2 holds.)
            for (int i = 0; i < n; ++i) {
                mom_o2[i] = rho_o[i] * u_o[i];
                rho_o2[i] = rho_o[i];
                ene_o2[i] = rho_o[i] * Htot_o[i];
            }
            have_o2 = true;
            dt_prev = dt;
            // adaptive-CFL ramp update: a clean first-try step ramps the scale back up toward 1;
            // a step that needed `retry` halvings persists that reduced level so the next step
            // starts where it succeeded (no jump back to full cfl -> no re-divergence grind).
            if (cfl_ramp) {
                // Stage 3a-ii (round 12, level 2 only): a step that only ever failed on
                // newton-no-progress did NOT fail because dt was too large (docs/YADV_ROUND_12_
                // PLAN.md sect.1.2: halving raises r_init as 1/dt), so collapsing cfl_scale by
                // 0.5^retry is unjustified for that mode -- and it is what pins cases 24/33/34 on
                // the 1e-3 floor. Leave the scale UNCHANGED for those; a genuinely clean retry-0
                // step still ramps up, so the controller can climb back off the floor.
                const bool r1_only = (stall_accept_lvl >= 2 && retry > 0 && only_reason1);
                if (retry == 0)    cfl_scale = std::min(1.0, cfl_scale * 1.5);
                else if (!r1_only) cfl_scale = std::max(1.0e-3, cfl_scale * std::pow(0.5, retry));
            }
            if (stall_accept_lvl > 0) stall_accept_run = 0;  // a clean step resets the budget
            break;
        }
        // Stage 3a (round 12): best-across-retries candidate. A retry is ELIGIBLE if it failed
        // ONLY on newton-no-progress -- i.e. `bad && stall_reason == 1` AFTER the cell scan above,
        // which is exactly "finite everywhere and |u| <= 10*uref". Ranked by the DIMENSIONLESS
        // progress ratio rbest/r_init (rnorm3 is unnormalised and not comparable across cases or
        // across retries once r_init starts scaling as 1/dt, docs/YADV_ROUND_12_PLAN.md sect.1.2).
        // Round 18/20 (F2'', unconditional since round 20): and also "no cell at the 1e6 K T
        // ceiling" -- the reason-5 assignment above displaces reason 1, so a saturated retry can
        // never be captured here nor adopted below. Accepting a state known NOT to solve
        // hmix(T)=hstat is the "silently accept garbage" mode this mechanism's own safeguard
        // exists to prevent.
        if (stall_accept_lvl > 0) {
            if (stall_reason != 1) only_reason1 = false;
            if (stall_reason == 1 && r_init > 0.0 && std::isfinite(rbest)) {
                const double ratio = rbest / r_init;
                if (!acc_have || ratio < acc_ratio) {
                    acc_have = true; acc_s = s; acc_Yv = Yv;
                    acc_dt = dt; acc_retry = retry;
                    acc_ratio = ratio; acc_rbest = rbest; acc_rinit = r_init;
                }
            }
        }
        if (dbg) {
            double mxu = 0; for (int i = 0; i < n; ++i) mxu = std::max(mxu, std::abs(s.u[i]));
            std::fprintf(stderr, "RETRY %d dt=%.3e -> max|u|=%.3e (uref=%.2e)\n", retry, dt, mxu, uref);
        }
        stall_dt = dt; stall_retry = retry;
        stall_conv_inner = conv_inner; stall_rbest = rbest; stall_rinit = r_init;
        dt *= 0.5;
        }  // retry loop
        // ---- Stage 3a-i (round 12): retry exhaustion no longer means "no step at all". ----
        // All 14 halvings failed with reason=newton-no-progress and a finite, speed-bounded state.
        // Since r_init grows as 1/dt for this mode (docs/YADV_ROUND_12_PLAN.md sect.1.2), the LAST
        // retry is the worst state the loop produced and the first is the best; adopt the
        // best-ranked one and take the step at ITS dt. This is the case15 keep-best precedent
        // (acid.cpp:2093, already unconditional per retry) lifted one loop level. Loud by
        // construction: every acceptance prints, and the run prints a total, so a "completed" run
        // that contains accepted-unconverged steps can never be mistaken for a clean one (the
        // sect.20 retraction failure mode).
        if (!stepped && stall_accept_lvl > 0 && acc_have && stall_accept_run < stall_accept_max) {
            s = acc_s; Yv = acc_Yv; dt = acc_dt;
            // Unlike the !bad path above, the o2 store is NOT populated here: have_o2=false makes
            // mom_o2/rho_o2/ene_o2 dead (both dt_const's gate at acid.cpp:965 and the rho-guard at
            // :998 require have_o2==true), so writing them from a non-converged state would be
            // both pointless and would need rho_o/u_o/Htot_o, which are retry-loop-local and out
            // of scope here (unlike inside the !bad branch, which runs before the loop closes).
            have_o2 = false;   // do NOT build a BDF2 level on a non-converged state; next step is BE
            dt_prev = dt;
            if (cfl_ramp) {    // rank by acc_retry, not 13: normally 0 -> the ramp climbs, not collapses
                if (acc_retry == 0) cfl_scale = std::min(1.0, cfl_scale * 1.5);
                else if (stall_accept_lvl < 2)
                    cfl_scale = std::max(1.0e-3, cfl_scale * std::pow(0.5, acc_retry));
            }
            ++n_stall_accept; ++stall_accept_run;
            std::fprintf(stderr,
                "STALL-ACCEPT: case=%s step %d t=%.3e -> accepting non-converged retry %d dt=%.3e "
                "(rbest=%.4e r_init=%.4e ratio=%.4f) run=%d/%d total=%ld\n",
                c.id.c_str(), step, t, acc_retry, acc_dt, acc_rbest, acc_rinit, acc_ratio,
                stall_accept_run, stall_accept_max, n_stall_accept);
            stepped = true;
            // ACID_TSAT block B (round 17): does the ACCEPTED (non-converged) state carry a
            // ceiling cell? s == acc_s here (just restored above).
            if (tsat) {
                int nhi = 0, ihi = -1;
                for (int i = 0; i < n; ++i) if (s.T[i] >= 1.0e6) { ++nhi; if (ihi < 0) ihi = i; }
                if (nhi) {
                    ++tsat_steps_hi;
                    std::fprintf(stderr,
                        "TSAT-ACCEPT case=%s step=%d retry=%d ncells=%d i0=%d\n",
                        c.id.c_str(), step, acc_retry, nhi, ihi);
                }
            }
        }
        if (!stepped) {
            // Phase 3a Stage 3c (round 14, docs/YADV_ROUND_14_PLAN.md) -- authorised by an
            // explicit Advisor decision after rounds 11/12/13 each deferred it (sect.21.1,
            // 22.7 pt.1, 23.4 pt.5). Retry exhaustion with NO admissible step is a FAILURE and
            // is now reported as one: rounds 3-11 returned a finite partial state here with
            // diverged==false, which validate/dump scored as a normal completed run -- exactly
            // what made YADV_RESEARCH.md sect.14.3/19.4 measure a pristine initial condition for
            // two rounds (sect.20, retracted). Round 11 made this audible; round 14 makes it
            // COUNTABLE (finite=false in validate, NaN in the dump).
            //
            // The accept/give-up boundary is drawn by the control flow above, deliberately, and
            // needs no extra test here: a step that ACID_STALL_ACCEPT successfully accepted set
            // `stepped = true` (see the accept block above) and never reaches this block, so an
            // accept-and-continue run is NOT marked diverged -- its own STALL-ACCEPT-TOTAL line
            // is its honest disclosure. Only "neither a clean step NOR an accepted step was
            // possible" lands here: STALL_ACCEPT unset/disabled, or its budget also exhausted
            // (stall_accept_run >= stall_accept_max), or no retry was reason-1 eligible.
            //
            // NOTE this block is NOT ACID_YADV-gated -- it is in the common time loop. The OFF
            // path is unaffected only because no OFF case stalls (sect.21.1, re-verified in
            // round 14's gate G1), not by construction. Keep it that way: do not add a `yadv`
            // condition, and re-run G1 if the OFF path's stepping ever changes.
            diverged = true;   // -> the p/u NaN fill at the end of solve_case_acid
            static const char* const why[] = {"unknown", "newton-no-progress", "nonfinite-p",
                                              "nonfinite-u", "u>10*uref", "T-ceiling-saturated"};
            std::fprintf(stderr,
                "STALLED: case=%s no admissible step at dt=%.3e after %d retries, step %d, "
                "t=%.3e of %.3e -> stop (marked DIVERGED: p,u,rho returned as NaN, "
                "validate finite=false)\n",
                c.id.c_str(), stall_dt, stall_retry + 1, step, t, t_end);
            if (dbg)
                std::fprintf(stderr,
                    "STALLED-DETAIL: reason=%s cell=%d x=%.5f p=%.4e u=%.4e rho=%.4e alpha=%.5f "
                    "T=%.4e (conv_inner=%d rbest=%.4e r_init=%.4e uref=%.3e)\n",
                    why[stall_reason], stall_cell,
                    stall_cell >= 0 ? st.x[stall_cell] : -1.0,
                    stall_cell >= 0 ? s.p[stall_cell] : 0.0,
                    stall_cell >= 0 ? s.u[stall_cell] : 0.0,
                    stall_cell >= 0 ? s.rho[stall_cell] : 0.0,
                    stall_cell >= 0 ? s.alpha[stall_cell] : 0.0,
                    stall_cell >= 0 ? s.T[stall_cell] : 0.0,
                    (int)stall_conv_inner, stall_rbest, stall_rinit, uref);
            break;  // could not advance even at the smallest dt
        }
        // store transient advecting velocity for next step
        {
            const auto pe = apply_ghost(s.p, lbc, rbc, 2, false);
            const auto ue = apply_ghost(s.u, lbc, rbc, 2, true);
            const auto re = apply_ghost(s.rho, lbc, rbc, 2, false);
            for (int f = 0; f <= n; ++f) {
                const int gL = f + 1, gR = f + 2;
                const double rf = 2.0 / (1.0 / re[gL] + 1.0 / re[gR]);
                const double aP = 0.5 * (re[gL] + re[gR]) * dx / dt;
                const double d_f = dx / std::max(aP, 1e-300);
                const double dh = d_f / (1.0 + (rf / dt) * d_f);
                const double dpf = (pe[gR] - pe[gL]) / dx;
                const double gpbar = 0.5 * ((pe[gL + 1] - pe[gL - 1]) / (2 * dx) + (pe[gR + 1] - pe[gR - 1]) / (2 * dx));
                theta_o[f] = 0.5 * (ue[gL] + ue[gR]) - dh * (dpf - gpbar);
            }
        }
        t += dt; ++step;
        if (dbg && (step < 6 || step % 200 == 0)) {
            double mxu = 0, mxp = 0;
            for (int i = 0; i < n; ++i) { mxu = std::max(mxu, std::abs(s.u[i])); mxp = std::max(mxp, s.p[i]); }
            std::fprintf(stderr, "ACID step %d t=%.3e dt=%.3e max|u|=%.4e maxp=%.4e p[mid]=%.4e\n",
                         step, t, dt, mxu, mxp, s.p[n / 2]);
        }
    }

    // Stage 3a (round 12): a run that accepted any non-converged step is NOT a clean solve --
    // say so unconditionally (not ACID_DBG-gated, matching round 11's STALLED: precedent) so it
    // can never again be mistaken for one (the exact failure mode of the sect.20 retraction).
    if (n_stall_accept > 0)
        std::fprintf(stderr,
            "STALL-ACCEPT-TOTAL: case=%s accepted %ld non-converged step(s) "
            "(ACID_STALL_ACCEPT=%d max_run=%d) -- this run is NOT a clean solve\n",
            c.id.c_str(), n_stall_accept, stall_accept_lvl, stall_accept_max);

    // ACID_TSAT block C (round 17): run summary. final_cells_hi scans s.T directly (the
    // `diverged` NaN-fill below touches only s.p/s.u, not s.T, so this reads the true last state).
    if (tsat) {
        int final_hi = 0;
        for (int i = 0; i < n; ++i) if (s.T[i] >= 1.0e6) ++final_hi;
        std::fprintf(stderr,
            "TSAT-TOTAL case=%s calls=%ld calls_hi=%ld calls_lo=%ld cells_hi_max=%d "
            "accepted_steps_hi=%ld first_hi_step=%d first_hi_cell=%d final_cells_hi=%d\n",
            c.id.c_str(), tsat_calls, tsat_calls_hi, tsat_calls_lo, tsat_cells_hi_max,
            tsat_steps_hi, tsat_first_step, tsat_first_cell, final_hi);
    }

    // Stage 1 sect.3.5 (round 11, ACID_DBG only): the exact end state and effective stop time,
    // needed by Phase 3a Stage 2's window sweep to compute completion robustly instead of
    // inferring it from the last periodically-printed "ACID step" line (every 200 steps).
    if (dbg)
        std::fprintf(stderr, "ACID done case=%s step=%d t=%.9e of %.9e\n",
                     c.id.c_str(), step, t, t_end);

    if (thinc_dbg)
        std::fprintf(stderr, "THINC case=%s activations=%ld rho_guard_rejects=%ld\n",
                     c.id.c_str(), thinc_hits, thinc_rej);

    if (diverged) {
        // Two producers: the CFL-collapse early-stop above, and (round 14, Stage 3c) the
        // retry-exhaustion give-up. Mark the result non-finite so the validate counts a
        // collapsed/diverged/stalled run as a clean failure (finite=false), not a misleading
        // partial state at t < final_time.
        std::fill(s.p.begin(), s.p.end(), std::nan(""));
        std::fill(s.u.begin(), s.u.end(), std::nan(""));
    }
    st.u = s.u; st.p = s.p; st.T = s.T; st.alpha = s.alpha;
    refresh_thermo(st, A, B);
    return st;
}

}  // namespace denner1d
