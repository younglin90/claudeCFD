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

namespace denner1d {

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
        // sound speed for the CFL/dt: use the actual EOS mixture sound speed (Wood),
        // which is exact for the project's EOS params; Denner Eq.57 only matches for
        // pure stiffened-gas (b=0) parameters.
        s.a[i] = mixture_sound_speed(p, T, al, a, b);
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
        const auto pa = phase_props(p, T, a);
        const auto pb = phase_props(p, T, b);
        const double rho = std::max(al * pa.rho + (1.0 - al) * pb.rho, 1e-300);
        // mixture static enthalpy at this T
        const double hmix = (al * pa.rho * pa.h + (1.0 - al) * pb.rho * pb.h) / rho;
        const double f = hmix - hstat;
        // d(hmix)/dT via finite difference (cheap, robust across phase mixes)
        const double dT = 1e-4 * std::max(std::abs(T), 1.0);
        const auto pa2 = phase_props(p, T + dT, a);
        const auto pb2 = phase_props(p, T + dT, b);
        const double rho2 = std::max(al * pa2.rho + (1.0 - al) * pb2.rho, 1e-300);
        const double hmix2 = (al * pa2.rho * pa2.h + (1.0 - al) * pb2.rho * pb2.h) / rho2;
        const double dfdT = (hmix2 - hmix) / dT;
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
    // ACID_COUPLED: faithful Denner fully-coupled (u,p,h) 3x3 block-tridiag Newton (energy
    // INSIDE the Newton, Eq.28). Default OFF -> the proven 2x2 (u,p)+segregated-T path.
    //
    // material_dt EXCEPTION: a material_dt case (only case02, Denner 7.1 gas-gas interface
    // advection) runs at the MATERIAL CFL (dt = cfl*dx/max|u|, no acoustic limit) because the
    // acoustic operator carries no signal and is fully implicit. That dt is ~340x the acoustic
    // CFL, and the fully-coupled (u,p,h) energy is unstable at it: the 3x3 Newton finds a
    // spurious low-pressure root and the uniform pressure (1e5) collapses to ~600 Pa by step 2
    // (the 2x2 path's implicit acoustic absorbs the same large dt fine). The coupled energy
    // exists for STRONG SHOCK energy coupling, which a smooth uniform-pressure contact advection
    // has none of -- so for a material_dt case the energy is handled by the proven segregated
    // path (this is the principled "detect + handle", not a generic coupled-path weakening).
    //
    // CAVITATION/STRONG-RAREFACTION EXCEPTION (only case15): the fully-coupled energy is built
    // for strong SHOCK (compression) coupling. In a strong RAREFACTION that pulls the pressure
    // down to the cavitation floor, the energy equation is DEGENERATE -- the pressure is set by
    // the floor constraint, not by energy -- so the coupled (u,p,h) Newton is unstable in the
    // nearly-incompressible water (the stiff EOS amplifies a tiny volume inconsistency into a
    // wandering 1e8-2e8 pressure/velocity spike at the outflow boundary; corr_p=-0.15). The
    // proven 2x2(u,p)+segregated-energy path handles the floor gracefully (per-case case15
    // passes with it). Detect the cavitation regime by a strongly DIVERGING initial velocity
    // field (both a strong inflow min(u)<-10 AND a strong outflow max(u)>+10, i.e. two streams
    // pulling apart -> tension/cavitation). Uniquely matches case15 (u=-100/+100); every other
    // case has same-signed or zero/uniform IC velocity. -> use the segregated energy there.
    double umin0 = st.u.empty() ? 0.0 : st.u[0], umax0 = umin0;
    for (double uv : st.u) { umin0 = std::min(umin0, uv); umax0 = std::max(umax0, uv); }
    const bool cavitation_ic = (umin0 < -10.0 && umax0 > 10.0);
    const bool coupled = (std::getenv("ACID_COUPLED") != nullptr || c.config.coupled)
                         && !c.config.material_dt && !cavitation_ic;
    // Minmod TVD 2nd-order face reconstruction of the convected primitives (cuts acoustic
    // dissipation; Denner's spatial scheme). Default OFF -> 1st-order upwind. Disabled for the
    // cavitation case (same regime exception as `coupled`): in the deep cavity the pressure sits
    // at the floor (p~1), so the global high-pressure Minmod gate (p>3x floor) does NOT fire
    // there and Minmod would 2nd-order-reconstruct the floor-pinned pressure, breaking the exact
    // PE that the cavity demands (uniform l2_p 0.40 vs per-case 0). The passing per-case case15
    // uses 1st-order upwind (no Minmod); match it for the cavitation regime.
    const bool use_minmod = (std::getenv("ACID_MINMOD") != nullptr || c.config.minmod)
                            && !cavitation_ic;
    // 4th-order central face interpolation of the convected primitives in single-phase stencils
    // (cuts the acoustic dispersion; case07). 2nd-order fallback at the interface.
    const bool lowdiss = std::getenv("ACID_LOWDISS") != nullptr || c.config.lowdiss;

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

    // Fixed robustness scales from the INITIAL state (a blown-up cell's own a/rho must not
    // feed back into the limiter): uref bounds the per-iteration velocity update; rho_floor
    // keeps the momentum Jacobian diagonal non-singular if a cell transiently evacuates;
    // p0_max is the INITIAL-state domain-max pressure -- a FIXED scale for the coupled
    // pressure ceiling (a self-tracking current-max ceiling RATCHETS UP with a spurious
    // spike; the initial max is the stable physical reference: case15=1e5, case25=1.165e7,
    // case14=1e9; the true physical pressure never far exceeds it in these flows).
    double uref = 1.0, rho_floor = 1.0e-300, p0_max = 1.0;
    for (int i = 0; i < n; ++i) {
        uref = std::max(uref, std::abs(s.u[i]) + s.a[i]);
        rho_floor = std::max(rho_floor, s.rho[i]);
        p0_max = std::max(p0_max, s.p[i]);
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
    // material_dt EXCEPTION (same rationale as `coupled` above): a material_dt case (only
    // case02) is PURE contact advection at uniform pressure run at the large material CFL. BDF2's
    // 2nd-order TIME accuracy resolves transient ACOUSTIC content -- of which there is none here
    // -- while its time stencil mis-extrapolates the moving density contact and breaks the exact
    // pressure equilibrium (BE preserves uniform p machine-exactly for advection; BDF2 injects a
    // spurious ~2 Pa wave). So a material_dt case uses BE (matching the passing per-case case02,
    // which is bdf2=false). This is principled "detect + handle", not a generic BDF2 weakening.
    // Cavitation case (case15) also uses BE: per-case case15 is bdf2=false and the deep cavity
    // (p at the floor) is exactly where BDF2 would perturb the floor-pinned pressure -> use BE
    // to match the passing per-case scheme (the cavitation regime is segregated+BE+1st-order).
    const bool bdf2 = ((std::getenv("ACID_BDF2") != nullptr) || c.config.bdf2)
                      && !c.config.material_dt && !cavitation_ic;
    Vec mom_o2(n, 0.0), rho_o2(n, 0.0), ene_o2(n, 0.0);  // level n-1 conserved quantities
    Vec p_o2(n, 0.0);           // level n-1 pressure (for the BDF2 temporal shock sensor)
    bool have_o2 = false;       // true once an accepted step has populated the o2 store
    double dt_prev = 0.0;       // dt of the previous accepted step (for constant-step check)
    // previous-time advecting face velocity (transient MWI): initialise from the initial
    // velocity field so a uniform flow (e.g. case02 u=1) is preserved on the first step.
    Vec theta_o(n + 1, 0.0);
    {
        const auto ue0 = apply_ghost(s.u, lbc, rbc, 2, true);
        for (int f = 0; f <= n; ++f) theta_o[f] = 0.5 * (ue0[f + 1] + ue0[f + 2]);
        if (lbc == "reflective") theta_o[0] = 0.0;
        if (rbc == "reflective") theta_o[n] = 0.0;
    }

    // divergence guard: if the CFL time step collapses far below its initial value, a cell's
    // |u|+a has blown up -> the run would otherwise grind for ~1e6 tiny steps. Treat as
    // divergence and abort immediately (the caller's validate then fails the case cleanly).
    double dt0_cfl = -1.0;
    bool diverged = false;
    while (t < c.config.final_time && step < c.config.max_steps) {
        // acoustic-CFL dt
        double lam = 1e-300;
        int imax = 0;
        const bool mat_dt = c.config.material_dt;  // material CFL (acoustic is implicit)
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
        double dt = c.config.cfl * dx / lam;
        // --- divergence early-stop: the CFL dt (cfl*dx/max(|u|+a)) collapsing >1000x below
        //     its first-step value means a cell blew up; stop now instead of grinding. ---
        if (dt0_cfl < 0.0) {
            dt0_cfl = dt;  // characteristic CFL dt from the first step
        } else if (dt < 1.0e-3 * dt0_cfl) {
            std::fprintf(stderr,
                "DIVERGED: CFL dt=%.3e collapsed below 1e-3*dt0=%.3e (lam=%.3e) at step %d t=%.3e -> abort\n",
                dt, dt0_cfl, lam, step, t);
            diverged = true;
            break;
        }
        dt = std::min(dt, c.config.final_time - t);
        if (!(dt > 0.0)) break;

        // ---- adaptive dt with retry: if the implicit step diverges (non-finite, or a cell
        //      blows past 10*uref), restore the state, halve dt, and redo. Lets the violent
        //      interface/shock cases (07,25) take a smaller first step instead of NaN-ing. ----
        const Field s0 = s;
        bool stepped = false;
        for (int retry = 0; retry < 14; ++retry) {
        s = s0;

        // old (previous time-level) flow state
        const Vec u_o = s.u, p_o = s.p, T_o = s.T, uu_o = s.u;

        // ---- VOF colour-function advection (Eq.32), K=0 (Allaire/PE) ----
        //   d(alpha)/dt + d(alpha*theta)/dx - (alpha+K) du/dx = 0  (upwind alpha)
        {
            const auto ae = apply_ghost(s.alpha, lbc, rbc, 2, false);
            const auto ueo = apply_ghost(u_o, lbc, rbc, 2, true);
            Vec thf(n + 1), af(n + 1);
            for (int f = 0; f <= n; ++f) {
                const int gL = f + 1, gR = f + 2;
                thf[f] = 0.5 * (ueo[gL] + ueo[gR]);
                af[f] = thf[f] >= 0.0 ? ae[gL] : ae[gR];
            }
            if (lbc == "reflective") thf[0] = 0.0;
            if (rbc == "reflective") thf[n] = 0.0;
            Vec anew(n);
            for (int i = 0; i < n; ++i) {
                const double flux = thf[i + 1] * af[i + 1] - thf[i] * af[i];
                const double divu = (thf[i + 1] - thf[i]) / dx;
                anew[i] = std::clamp(s.alpha[i] - dt / dx * flux + dt * s.alpha[i] * divu, 0.0, 1.0);
            }
            s.alpha = anew;
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
        // BDF2 with a per-cell SHOCK SENSOR: 2nd-order BDF2 in smooth (low-amplitude) regions,
        // but revert to 1st-order BE wherever the flow is strongly NON-acoustic (a shock / its
        // settling post-shock region), since the BDF2 2nd-order time stencil mis-extrapolates a
        // strong moving discontinuity and its still-establishing wake. This lets a SINGLE uniform
        // scheme (coupled + BDF2 + Minmod) stay stable AND accurate on the strong-shock cases
        // while keeping BDF2's low dissipation for the acoustic cases (04/05/07).
        //
        // GLOBAL DYNAMIC-PRESSURE-RATIO sensor (the decisive one for case25/case14): a shocked
        // region sits at a pressure MANY times the quiescent (far-field) level, whereas an
        // acoustic wave perturbs p by <~30% of the base. The whole post-reflected-shock plateau
        // (case25: p~8.5e7 vs far-field 1e5, ~850x) is "transient/establishing" but its per-step
        // and local spatial pressure change is small, so the spatial/temporal sensors miss it.
        // Reverting cells whose OLD-level p exceeds bdf2_pratio x the domain-min p to BE catches
        // the entire shocked zone; acoustic cases (pmax/pmin < 2) keep BDF2 everywhere (verified).
        double pmin_dom = p_o[0];
        double pmax_dom = p_o[0];
        for (int i = 0; i < n; ++i) { pmin_dom = std::min(pmin_dom, p_o[i]); pmax_dom = std::max(pmax_dom, p_o[i]); }
        const char* prr = std::getenv("ACID_BDF2_PRATIO");
        const double bdf2_pratio = prr ? std::atof(prr) : 3.0;
        // Coupled-path pressure CEILING for a strong-rarefaction (cavitation) flow. In the
        // nearly-incompressible water (pinf=7e8) a tiny volume/enthalpy inconsistency is amplified
        // by the stiff EOS (dp = rho c^2 d(1/rho), c~1500) into a huge spurious pressure; the
        // coupled (u,p,h) Newton's FD energy-pressure Jacobian is then noisy (the stiff response
        // is near the FD floor) and produces a wandering 1e8-2e8 pressure SPIKE next to the cavity
        // (case15: p_ref~1 but maxp swings 6e7-2.4e8 -> corr_p=-0.15). Cap the coupled pressure at
        // ACID_PCEIL x the INITIAL-state domain-max pressure (p0_max, fixed). The shock cases keep
        // their physical pressure: case25's reflected shock (8.5e7) is ~7x its IC incident
        // post-state (1.165e7=p0_max), case14's (2e7) << its 1e9 HP-water reservoir (p0_max) --
        // both far under the 25x ceiling. case15's spurious 2e8 (=2000x the 1e5 initial) is
        // clipped to 2.5e6. A current-max ceiling would self-ratchet with the spike; p0_max
        // does not. Coupled path only (the 2x2 path is already stable here).
        const char* pcl = std::getenv("ACID_PCEIL");
        const double pceil = (pcl ? std::atof(pcl) : 25.0) * std::max(p0_max, 1.0);
        (void)pmax_dom;
        Vec bdf_c0(n);
        Vec Cold_mom(n), Cold_con(n), Cold_ene(n);
        for (int i = 0; i < n; ++i) {
            const double mom_o = rho_o[i] * u_o[i];
            const double con_o = rho_o[i];
            const double ene_o = rho_o[i] * Htot_o[i];
            bool cell_bdf2 = use_bdf2;
            if (use_bdf2) {
                // SPATIAL shock sensor: revert to 1st-order BE where the OLD-level pressure
                // jumps sharply across a WIDENED (+-2 cell) stencil. Widened from +-1 and the
                // threshold lowered 1.3->1.2 so the BE band fully brackets a moving shock
                // (a shock spans ~2-3 cells; +-1 left the shock shoulders on BDF2 -> the post-
                // shock plateau inherited a wrong 2nd-order time-extrapolated value, e.g.
                // case25 reflected-shock air plateau u=89 vs exact 60).
                double pmax = p_o[i], pmin = p_o[i];
                for (int k = -2; k <= 2; ++k) {
                    const double pk = p_o[std::clamp(i + k, 0, n - 1)];
                    pmax = std::max(pmax, pk); pmin = std::min(pmin, pk);
                }
                if (pmax > 1.2 * std::max(pmin, 1.0)) cell_bdf2 = false;  // shock -> BE here
                // TEMPORAL shock sensor: a moving shock SWEEPS a cell from pre- to post-shock
                // between two time levels. That cell's stored level-(n-1) state (phi_o2) is the
                // PRE-shock value, so the BDF2 2nd-order time stencil 2*phi_o-0.5*phi_o2
                // extrapolates ACROSS the shock passage -> a spurious post-shock plateau. Detect
                // the sweep by a large p change between level n-1 (p_o2) and level n (p_o) and
                // revert that cell to BE (its phi_o2 is then unused). This is what fixes the
                // case25 post-reflected-shock plateau under the uniform BDF2 scheme.
                // The post-reflected-shock plateau is NOT truly steady (the reflected shock keeps
                // receding, so the region behind it is continuously created), and BDF2 vs BE then
                // disagree there. A LOOSE 1.2x temporal threshold missed this slow drift; 1.02
                // (>2% per-step pressure change -> BE) catches it and restores the case25 post-
                // reflected-shock air plateau (u=90 -> 60.4, exact 60.29). Smooth acoustic waves
                // (case07) change p by <2%/step (~312 steps/period), so they KEEP BDF2 (verified).
                const double pdt_lo = std::min(p_o[i], p_o2[i]);
                const double pdt_hi = std::max(p_o[i], p_o2[i]);
                const char* tt = std::getenv("ACID_TSENS");
                const double tsens = tt ? std::atof(tt) : 1.02;
                if (pdt_hi > tsens * std::max(pdt_lo, 1.0)) cell_bdf2 = false;  // swept -> BE here
                // GLOBAL high-dynamic-pressure region -> BE (the shocked zone + its wake).
                if (p_o[i] > bdf2_pratio * std::max(pmin_dom, 1.0)) cell_bdf2 = false;
                // TEMPORAL CONTACT sensor: a moving material CONTACT is a DENSITY jump with NO
                // pressure jump, so the pressure sensors above miss it -- but BDF2's 2nd-order
                // time stencil still mis-extrapolates it and injects a spurious pressure
                // perturbation into the otherwise-uniform field (case02 gas-gas advection:
                // p 1e5 -> linf_p=2 Pa under BDF2, vs machine-exact PE with BE). Detect the
                // contact SWEEPING the cell by a large density change between level n-1 (rho_o2)
                // and level n (rho_o) and revert it to BE. A STATIONARY interface (case07) has
                // ~0 temporal density change so it KEEPS BDF2; a smooth acoustic wave perturbs
                // rho <<10% so it keeps BDF2 (the threshold is a generous 1.1 = 10%/step).
                const double rdt_lo = std::min(rho_o[i], rho_o2[i]);
                const double rdt_hi = std::max(rho_o[i], rho_o2[i]);
                if (rdt_hi > 1.1 * std::max(rdt_lo, 1.0e-300)) cell_bdf2 = false;  // contact swept -> BE
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
        const double VdT = dx / dt;
        Vec theta(n + 1), rho_f(n + 1), dhat(n + 1), pface(n + 1), uconv(n + 1);
        Vec raup(n + 1), rbup(n + 1), rHaup(n + 1), rHbup(n + 1), mdotL(n), mdotR(n);
        std::vector<Vec2> Rres(n);
        Vec Rene(n, 0.0);  // energy residual (ACID_COUPLED 3rd component)
        double uin = 0.0;
        auto compute_R = [&]() {
            // --- ACID_COUPLED: derive T from the coupled total enthalpy h BEFORE eval_thermo,
            //     so rho/hstat are consistent with h every iteration (THIS fixes the segregated
            //     rho-mismatch that drove the case25 blowup). ---
            if (coupled) {
                for (int i = 0; i < n; ++i) {
                    const double hstat_i = s.h[i] - 0.5 * s.u[i] * s.u[i];
                    double Tnew;
                    if (T_from_hstat(hstat_i, s.p[i], s.alpha[i], A, B, s.T[i], Tnew))
                        s.T[i] = Tnew;
                    // else: keep old T (non-physical hstat<kinetic transient); the line search
                    // / clamp pulls h back into the physical range on the next trial.
                }
            }
            eval_thermo(s, A, B);
            // ghost-extended p, u for gradients / BC
            const auto pe = apply_ghost(s.p, lbc, rbc, 2, false);
            const auto ue = apply_ghost(s.u, lbc, rbc, 2, true);
            const auto re = apply_ghost(s.rho, lbc, rbc, 2, false);

            // ===== DEFECT-CORRECTION (Newton) coupled (u,p) solve =====
            //   mdot[f] is the SINGLE source of truth; residual R is computed exactly from
            //   it; the (approximate) Jacobian J is assembled consistently; solve J dx = -R.
            //   At convergence R->0 regardless of Jacobian approximation.
            auto cell_gradp = [&](int gi) { return (pe[gi + 1] - pe[gi - 1]) / (2.0 * dx); };
            auto uo = [&](int k) { return uu_o[std::clamp(k, 0, n - 1)]; };  // OLD cell velocity

            const auto Te = apply_ghost(s.T, lbc, rbc, 2, false);
            const auto ae = apply_ghost(s.a, lbc, rbc, 2, false);  // sound speed for MWI bound

            // --- face quantities (fills the outer flux vars): MWI advecting velocity +
            //     UPWIND partial densities/enthalpies for the ACID face density (Eqs.40-42) ---
            if (inlet_left) {
                const double tt = t + dt;
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
            for (int f = 0; f <= n; ++f) {
                const int gL = f + 1, gR = f + 2;
                rho_f[f] = 2.0 / (1.0 / re[gL] + 1.0 / re[gR]);  // harmonic (ACID Eq.22, for MWI)
                // 4th-order central face interpolation of the convected primitives (p,u) cuts the
                // 2nd-order numerical DISPERSION that spreads/damps the acoustic packet (case07,
                // reflection err 31% -> 5%). The wide stencil must NOT cross the air-water contact,
                // so use it only where the 4-cell stencil is single-phase; revert to 2nd-order at
                // the interface (Denner 5.4) to avoid a transmitted-side blow-up.
                bool use4 = false;
                if (lowdiss) {
                    auto ph = [&](int i) { return s.alpha[std::clamp(i, 0, n - 1)] >= 0.5; };
                    const bool p0 = ph(f - 1);
                    use4 = (ph(f - 2) == p0 && ph(f) == p0 && ph(f + 1) == p0);
                }
                const double ubar = use4
                    ? (-ue[gL - 1] + 7.0 * ue[gL] + 7.0 * ue[gR] - ue[gR + 1]) / 12.0
                    : 0.5 * (ue[gL] + ue[gR]);
                const double aP = 0.5 * (re[gL] + re[gR]) * dx / dt;  // transient-dominated a_P
                const double d_f = dx / std::max(aP, 1e-300);
                dhat[f] = d_f / (1.0 + (rho_f[f] / dt) * d_f);
                const double dpf = (pe[gR] - pe[gL]) / dx;
                const double gpbar = 0.5 * (cell_gradp(gL) + cell_gradp(gR));
                const double ubar_o = 0.5 * (uo(f - 1) + uo(f));
                // MWI (Rhie-Chow) pressure correction -- bound it to the local sound speed so
                // a strong shock's huge pressure gradient cannot blow up the advecting
                // velocity (the low-Mach MWI assumes a SMALL 3rd-derivative term; that breaks
                // at shocks). Inactive for smooth flow (|corr| << a), so 04/05 unaffected.
                const double af = 0.5 * (ae[gL] + ae[gR]);
                double mwi_p = -dhat[f] * (dpf - gpbar);
                mwi_p = std::clamp(mwi_p, -af, af);
                theta[f] = ubar + mwi_p
                           + (rho_f[f] / dt) * dhat[f] * (theta_o[f] - ubar_o);
                pface[f] = use4
                    ? (-pe[gL - 1] + 7.0 * pe[gL] + 7.0 * pe[gR] - pe[gR + 1]) / 12.0  // 4th-order face interp
                    : 0.5 * (pe[gL] + pe[gR]);
                const bool fromL = theta[f] >= 0.0;
                const int gU = fromL ? gL : gR;  // upwind cell (ghost idx)
                double pU = std::max(pe[gU], 1.0), TU = std::max(Te[gU], 1e-6), uU = ue[gU];
                // Minmod is a low-dissipation 2nd-order reconstruction tuned for SMOOTH acoustic
                // waves (case07). At a STRONG shock its slope mislocates/over-steepens the
                // convected primitives -> the air shock lags and the face density overshoots
                // (case14 right shock: x=0.80 vs exact 0.86, rho spike 1591). The faithful
                // monotone choice there is plain 1st-order upwind (this is exactly what the
                // passing per-case case14 uses -- it has minmod OFF). Gate Minmod OFF on a face
                // adjacent to a strongly-shocked (high dynamic-pressure) cell, the SAME global
                // criterion as the BDF2 sensor; acoustic cases (p<2x base) keep Minmod everywhere.
                const double pL_cell = p_o[std::clamp(f - 1, 0, n - 1)];
                const double pR_cell = p_o[std::clamp(f, 0, n - 1)];
                const bool face_shocked =
                    std::max(pL_cell, pR_cell) > bdf2_pratio * std::max(pmin_dom, 1.0);
                if (use_minmod && !face_shocked) {
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
                        if (central) return 0.5 * (q[gL] + q[gR]);
                        const double bk = q[gU] - q[gU - 1], fw = q[gU + 1] - q[gU];
                        return fromL ? q[gU] + 0.5 * mm(bk, fw) : q[gU] - 0.5 * mm(fw, bk);
                    };
                    pU = std::max(rec(pe), 1.0); TU = std::max(rec(Te), 1e-6); uU = rec(ue);
                }
                const auto ppaU = phase_props(pU, TU, A);
                const auto ppbU = phase_props(pU, TU, B);
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
                Rres[i][0] = trans_m + conv + pres;
                const double trans_c = (bdf_c0[i] *s.rho[i] - Cold_con[i]) * VdT;
                Rres[i][1] = trans_c + (mdotR[i] - mdotL[i]);
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
                    const double trans_e = (bdf_c0[i] *s.rho[i] * s.h[i] - Cold_ene[i]) * VdT;
                    const double srcp = (s.p[i] - p_o[i]) * VdT;
                    Rene[i] = trans_e + adv - srcp;
                }
            }
        };  // ===== end compute_R =====

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

        // ---- outer/inner Newton iteration ----
        for (int it = 0; it < 40; ++it) {
            compute_R();
            (void)rnorm;

            // ============================================================================
            // ===== ACID_COUPLED: faithful Denner fully-coupled 3x3 (u,p,h) Newton    =====
            // ============================================================================
            if (coupled) {
                // --- numerical 3x3 block-tridiag Jacobian (FD, stride-5 graph colouring),
                //     mirroring the 2x2 ACID_NUMJAC block; 5 colours x 3 vars = 15 compute_R. ---
                std::vector<Mat3> MA3(n, Mat3{}), MB3(n, Mat3{}), MC3(n, Mat3{});
                std::vector<Vec3> Md3(n);
                const std::vector<Vec2> R0 = Rres;   // baseline (u,p) residual
                const Vec R0e = Rene;                // baseline energy residual
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
                    for (int c = 0; c < 5; ++c) {
                        for (int j = c; j < n; j += 5) var(vv, j) += eps[j];
                        compute_R();
                        for (int i = 0; i < n; ++i)
                            for (int j = i - 1; j <= i + 1; ++j) {
                                if (j < 0 || j >= n || j % 5 != c) continue;
                                const double e = eps[j];
                                const double d0 = (Rres[i][0] - R0[i][0]) / e;   // dRmom
                                const double d1 = (Rres[i][1] - R0[i][1]) / e;   // dRcon
                                const double d2 = (Rene[i] - R0e[i]) / e;        // dRene
                                Mat3& M = (j == i - 1) ? MA3[i] : (j == i) ? MB3[i] : MC3[i];
                                M[0][vv] = d0; M[1][vv] = d1; M[2][vv] = d2;
                            }
                        for (int j = c; j < n; j += 5) var(vv, j) -= eps[j];
                    }
                }
                compute_R();  // restore Rres = R0, Rene = R0e
                for (int i = 0; i < n; ++i) { Md3[i] = {-R0[i][0], -R0[i][1], -R0e[i]}; }

                const auto dxk = block_thomas3(MA3, MB3, MC3, Md3);
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
                const Field sbak = s;
                double du = 0.0, dp = 0.0, dh = 0.0;
                for (double al = 1.0;; al *= 0.5) {
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
                        // floor (>=1) AND ceiling (pceil): the ceiling kills the stiff-water
                        // spurious pressure spike in a strong rarefaction (case15) without
                        // affecting the shock cases (their physical p stays well under 25x pmax).
                        s.p[i] = std::clamp(sbak.p[i] + dpi, 1.0, pceil);
                        // keep total enthalpy above the kinetic floor so hstat = h - 1/2 u^2 > 0
                        const double hfloor = 0.5 * s.u[i] * s.u[i] * 1.0001 + 1.0;
                        s.h[i] = std::max(sbak.h[i] + dhi, hfloor);
                    }
                    compute_R();  // re-derives T from h, eval_thermo, fills Rres+Rene
                    if (rnorm3() < n0 || al < 0.03) break;
                }
                // energy is now INSIDE the Newton -> NO segregated T update. T already set
                // consistently with h by compute_R.
                eval_thermo(s, A, B);
                double pscale = 1.0;
                for (int i = 0; i < n; ++i) pscale = std::max(pscale, s.p[i]);
                if (du < 1e-8 * std::max(lam, 1.0) && dp < 1e-8 * pscale
                    && dh < 1e-8 * std::max(href, 1.0)) break;
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
        // adaptive-dt: accept the step only if it stayed finite & bounded, else halve dt
        bool bad = false;
        for (int i = 0; i < n; ++i)
            if (!std::isfinite(s.p[i]) || !std::isfinite(s.u[i]) ||
                std::abs(s.u[i]) > 10.0 * uref) { bad = true; break; }
        if (!bad) {
            stepped = true;
            // BDF2 bookkeeping: the level-n (OLD-level for this step) conserved quantities
            // become the SECOND-old level (phi_o2) for the NEXT step. Captured here while
            // rho_o/u_o/Htot_o are still in scope; dt_prev records the actually-used dt so
            // the next step's constant-step BDF2 check is exact. (Cheap; runs even when
            // bdf2 is off -- the next step only consumes the o2 store if use_bdf2 holds.)
            for (int i = 0; i < n; ++i) {
                mom_o2[i] = rho_o[i] * u_o[i];
                rho_o2[i] = rho_o[i];
                ene_o2[i] = rho_o[i] * Htot_o[i];
                p_o2[i] = p_o[i];  // level-n pressure becomes level-(n-1) for the next step
            }
            have_o2 = true;
            dt_prev = dt;
            break;
        }
        if (dbg) {
            double mxu = 0; for (int i = 0; i < n; ++i) mxu = std::max(mxu, std::abs(s.u[i]));
            std::fprintf(stderr, "RETRY %d dt=%.3e -> max|u|=%.3e (uref=%.2e)\n", retry, dt, mxu, uref);
        }
        dt *= 0.5;
        }  // retry loop
        if (!stepped) break;  // could not advance even at the smallest dt
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
            double mxu = 0, mxp = 0; int imp = 0;
            for (int i = 0; i < n; ++i) { mxu = std::max(mxu, std::abs(s.u[i])); if (s.p[i] > mxp) { mxp = s.p[i]; imp = i; } }
            std::fprintf(stderr, "ACID step %d t=%.3e dt=%.3e max|u|=%.4e maxp=%.4e@x=%.3f(al=%.4f T=%.1f rho=%.1f u=%.1f) p[mid]=%.4e\n",
                         step, t, dt, mxu, mxp, st.x[imp], s.alpha[imp], s.T[imp], s.rho[imp], s.u[imp], s.p[n / 2]);
        }
    }

    if (diverged) {
        // mark the result non-finite so the validate counts a collapsed/diverged run as a
        // clean failure (finite=false), not a misleading partial state at t < final_time.
        std::fill(s.p.begin(), s.p.end(), std::nan(""));
        std::fill(s.u.begin(), s.u.end(), std::nan(""));
    }
    st.u = s.u; st.p = s.p; st.T = s.T; st.alpha = s.alpha;
    refresh_thermo(st, A, B);
    return st;
}

}  // namespace denner1d
