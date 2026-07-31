// cfd/five_eq/material_update.hpp — M6 material update (production path).
//
// Header-only C++ twin of solver/five_eq_IMEX/imex_ad.py::_material_update
// (2170-2733) for the PRODUCTION 5-equation IMEX configuration used by 02_A /
// 07_B (regime_auto -> pressure_work_consistent / compressive_recovery):
//
//     time_integrator = imex_ad          material_flux   = slau2
//     primitive_scheme = tmlpu (vanleer) alpha_scheme    = adaptive_bvd
//     mixture_kind     = kapila          kapila_closure  = true
//     kapila_source_mode = mixed_path    material_energy_form = allaire
//
// One forward-Euler material substep: it builds the ghost-extended state, the
// SLAU2 material face velocity (Wood-Z acoustic base overwritten by SLAU2 where
// valid), the tmlpu primitive / density / mixture-hancock face reconstruction,
// the adaptive_bvd sharp-alpha face with the two-stage flux-corrected-transport
// limiter (face-local max-principle + Zalesak cell-update limiter), an optional
// primitive-FCT limiter, the non-conservative Kapila alpha source (mixed_path
// Simpson faces), and finally the conservative advective flux divergences.
//
// It REUSES the already-validated leaf kernels: cfd::EOS + primitive.hpp
// (prim_to_cons_W), sound_speed.hpp (phase_acoustic / phase_sound_speed_sq),
// slau2.hpp (slau2_face), reconstruct.hpp (limited_value / reconstruct_*),
// alpha_bvd.hpp (adaptive_bvd_alpha_face). Every clamp reproduces the Python
// guard EXACTLY (_EPS = 1e-30) so results are bit-comparable (rel <= 1e-12) vs
// tests/5eq_ref/material_update_ref.txt (02_A + a small 07_B-like state).
//
// The host-side branch gates (mixture-recon enable, sharp-alpha present,
// collocated pressure/material jump, passive pressure-equilibrium transport) and
// the data-dependent Zalesak loops make a full device offload impractical for
// this substep, so the array drivers run on the host and call the portable
// CFD_ROUTINE_SEQ scalar kernels from the reused headers.
#pragma once
#include "cfd/eos.hpp"
#include "cfd/primitive.hpp"
#include "cfd/five_eq/sound_speed.hpp"
#include "cfd/five_eq/slau2.hpp"
#include "cfd/five_eq/reconstruct.hpp"
#include "cfd/five_eq/alpha_bvd.hpp"
#include "cfd/five_eq/energy_flux.hpp"
#include "cfd/five_eq/regime_auto.hpp"
#include <cmath>
#include <optional>
#include <vector>

namespace cfd {

enum class BC5 { Periodic, Reflective, Transmissive, Inlet, Outlet, InletAcoustic, Dirichlet };
enum class MaterialFlux { Slau2, HllcContact };
enum class KapilaSourceMode { Path, Cell, Hybrid, Trapezoid, ImmiscibleTrapezoid, MixedTrapezoid, MixedPath };

struct MaterialConfig {
    double alpha_pure_tol;
    BC5    bc_l;
    BC5    bc_r;
    // Boundary values mirror boundary.extend_W.  Values not supplied retain
    // the zero-gradient ghost for that primitive component.
    std::optional<double> u_inlet_l;
    std::optional<double> p_inlet_l;
    std::optional<double> p_outlet_r;
    std::optional<double> alpha_inlet_l;
    std::optional<double> T1_inlet_l;
    std::optional<double> T2_inlet_l;
    // Optional Python material branches.  Defaults preserve the production
    // SLAU2/T-MLP-u path used by the acceptance cases.
    MaterialFlux material_flux = MaterialFlux::Slau2;
    bool characteristic_reconstruction = false;
    bool kapila_closure = true;
    KapilaSourceMode kapila_source_mode = KapilaSourceMode::MixedPath;
    // Python material_energy_form: Allaire is the production baseline;
    // Secant is the APEC path selected by pressure_closure=apec_pe.
    EnergyForm energy_form = EnergyForm::Allaire;
    MixtureSoundSpeedKind mixture_sound_speed_kind = MixtureSoundSpeedKind::Kapila;
    double energy_alpha_pure_tol = 1.e-12;
    TvdLimiter primitive_tvd_limiter = TvdLimiter::VanLeer;
};

struct MaterialResult {
    std::vector<double> q1_new, q2_new, m_adv, rhoE_new, rhoE_adv, alpha_new;
};

namespace detail {

constexpr double MU_EPS = 1e-30;
// np.finfo(float).eps**0.25 = 2^-13 exactly.
constexpr double EPS4 = 0.0001220703125;

inline double dmax(double a, double b) { return a > b ? a : b; }
inline double dmin(double a, double b) { return a < b ? a : b; }
inline double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// van Leer paired slope (imex_ad.py::_vanleer_pair 1274-1281).
inline double vanleer_pair(double a, double b) {
    bool same = (a * b) > 0.0;
    if (!same) return 0.0;
    double den = a + b;
    double den_safe = std::fabs(den) > MU_EPS ? den : 1.0;
    return 2.0 * a * b / den_safe;
}

// imex_ad.py::_tvd_pair. This slope form is used by the mixture Hancock and
// characteristic reconstructions, while reconstruct_lr_faces uses the
// equivalent ratio-form limiter from reconstruct.hpp.
inline double tvd_pair(double a, double b, TvdLimiter kind) {
    if (!(std::isfinite(a) && std::isfinite(b)) || a * b <= 0.0) return 0.0;
    const double sign = a > 0.0 ? 1.0 : -1.0;
    const double aa = std::fabs(a), ab = std::fabs(b);
    switch (kind) {
        case TvdLimiter::Minmod:
            return sign * dmin(aa, ab);
        case TvdLimiter::Superbee:
            return sign * dmax(dmin(2.0 * aa, ab), dmin(aa, 2.0 * ab));
        case TvdLimiter::MC:
            return sign * dmin(dmin(2.0 * aa, 2.0 * ab), 0.5 * std::fabs(a + b));
        case TvdLimiter::VanAlbada: {
            const double den = a * a + b * b;
            return den > MU_EPS ? a * b * (a + b) / den : 0.0;
        }
        case TvdLimiter::Umist: {
            const double r = b / a;
            double psi = dmin(dmin(2.0 * r, 0.25 + 0.75 * r),
                              dmin(0.75 + 0.25 * r, 2.0));
            return dmax(psi, 0.0) * a;
        }
        case TvdLimiter::VanLeer:
        default:
            return vanleer_pair(a, b);
    }
}

// imex_ad.py::_mixture_tvd_kind_for_state. Superbee is retained for shocks,
// but a homogeneous mixed double-rarefaction uses VanLeer.
inline TvdLimiter mixture_tvd_kind_for_state(
        const std::vector<double>& a_ext, const std::vector<double>& u_ext,
        TvdLimiter requested) {
    if (requested != TvdLimiter::Superbee || a_ext.size() < 2 || u_ext.size() < 2)
        return requested;
    double amin = a_ext.front(), amax = a_ext.front();
    for (double a : a_ext) {
        amin = dmin(amin, a);
        amax = dmax(amax, a);
    }
    if (amin <= EPS4 || amax >= 1.0 - EPS4) return requested;
    for (std::size_t i = 0; i + 1 < u_ext.size(); ++i)
        if (u_ext[i] < 0.0 && u_ext[i + 1] > 0.0) return TvdLimiter::VanLeer;
    return requested;
}

// ng=1 ghost extension of a cell-centred array (boundary.py::extend).
inline std::vector<double> extend1(const std::vector<double>& a,
                                   BC5 bc_l, BC5 bc_r, bool odd,
                                   std::optional<double> left_value = {},
                                   std::optional<double> right_value = {}) {
    int n = (int)a.size();
    std::vector<double> e(n + 2);
    for (int i = 0; i < n; ++i) e[i + 1] = a[i];
    switch (bc_l) {
        case BC5::Periodic:     e[0] = a[n - 1]; break;
        case BC5::Transmissive: e[0] = a[0];     break;
        case BC5::Reflective:   e[0] = odd ? -a[0] : a[0]; break;
        case BC5::Inlet:
        case BC5::InletAcoustic:
        case BC5::Dirichlet:    e[0] = left_value ? *left_value : a[0]; break;
        case BC5::Outlet:       e[0] = a[0]; break;
    }
    switch (bc_r) {
        case BC5::Periodic:     e[n + 1] = a[0];     break;
        case BC5::Transmissive: e[n + 1] = a[n - 1]; break;
        case BC5::Reflective:   e[n + 1] = odd ? -a[n - 1] : a[n - 1]; break;
        case BC5::Inlet:
        case BC5::InletAcoustic:
        case BC5::Dirichlet:
        case BC5::Outlet:       e[n + 1] = right_value ? *right_value : a[n - 1]; break;
    }
    return e;
}

// D_K_kapila (source_d1.py 21-35), per-cell scalar.
inline double D_K_kapila_cell(const EOS& eos1, const EOS& eos2,
                              double a1, double T1, double T2, double p) {
    double a2 = 1.0 - a1;
    double rho1 = eos1.density(p, T1);
    double rho2 = eos2.density(p, T2);
    double c1_sq = phase_sound_speed_sq(eos1, rho1, T1);
    double c2_sq = phase_sound_speed_sq(eos2, rho2, T2);
    double rho1c2 = dmax(rho1 * c1_sq, MU_EPS);
    double rho2c2 = dmax(rho2 * c2_sq, MU_EPS);
    double num = a1 * a2 * (rho2c2 - rho1c2);
    double den = dmax(a2 * rho1c2 + a1 * rho2c2, MU_EPS);
    double D = num / den;
    return (a1 * a2 > 1e-12) ? D : 0.0;
}

// ── host gate: _mixture_primitive_recon_enabled(tmlpu, W_ext) (1709-1763),
//    auto mode. ──────────────────────────────────────────────────────────────
inline bool mixture_recon_enabled(const std::vector<double>& a_ext,
                                  const std::vector<double>& u_ext,
                                  const std::vector<double>& p_ext) {
    int ne = (int)a_ext.size();
    if (ne == 0) return false;
    double pure_tol = EPS4;
    double amn = a_ext[0], amx = a_ext[0];
    for (double a : a_ext) { amn = dmin(amn, a); amx = dmax(amx, a); }
    bool has_pure = (amn <= pure_tol) || (amx >= 1.0 - pure_tol);
    if (has_pure) {
        double pmn = p_ext[0], pmx = p_ext[0], pabs = 0.0;
        for (double v : p_ext) { pmn = dmin(pmn, v); pmx = dmax(pmx, v); pabs = dmax(pabs, std::fabs(v)); }
        double umn = u_ext[0], umx = u_ext[0], uabs = 0.0;
        for (double v : u_ext) { umn = dmin(umn, v); umx = dmax(umx, v); uabs = dmax(uabs, std::fabs(v)); }
        double p_scale = dmax(pabs, 1.0), u_scale = dmax(uabs, 1.0);
        if ((pmx - pmn) <= pure_tol * p_scale && (umx - umn) <= pure_tol * u_scale)
            return false;   // passive pressure-equilibrium: reconstruct phase thermo
        return true;
    }
    // no pure material: enable on any pressure discontinuity.
    for (int f = 0; f + 1 < ne; ++f) {
        double pl = p_ext[f], pr = p_ext[f + 1];
        double rel = std::fabs(pr - pl) / dmax(dmax(std::fabs(pl), std::fabs(pr)), 1.0);
        if (rel > pure_tol) return true;
    }
    return false;
}

// _collocated_pressure_material_jump(W_n, pure_tol) (143-171).
inline bool collocated_pjump(const std::vector<double>& a,
                             const std::vector<double>& p, double pure_tol) {
    int n = (int)a.size();
    if (n < 2) return false;
    double jt = EPS4;
    pure_tol = dmax(pure_tol, jt);
    for (int f = 0; f + 1 < n; ++f) {
        double al = a[f], ar = a[f + 1];
        bool ajump = std::fabs(ar - al) > jt;
        bool pure = (al <= pure_tol) || (al >= 1.0 - pure_tol)
                 || (ar <= pure_tol) || (ar >= 1.0 - pure_tol);
        double pl = p[f], pr = p[f + 1];
        double relp = std::fabs(pr - pl) / dmax(dmax(std::fabs(pl), std::fabs(pr)), 1.0);
        if (ajump && pure && (relp > jt)) return true;
    }
    return false;
}

// _sharp_alpha_interface_present(W_n, alpha_pure_tol) (1465-1499).
inline bool sharp_alpha_present(const std::vector<double>& a, double alpha_pure_tol) {
    int n = (int)a.size();
    if (n < 2) return false;
    double pure_tol = dmax(alpha_pure_tol, EPS4);
    double pure_band = pure_tol * (1.0 + 1.0e-9) + 1.0e-15;
    int low = 0, high = 0, mixed = 0;
    for (double v : a) {
        if (v <= pure_band) ++low;
        if (v >= 1.0 - pure_band) ++high;
        if (v > pure_band && v < 1.0 - pure_band) ++mixed;
    }
    bool has_low = low > 0, has_high = high > 0;
    bool sharp_jump = false;
    for (int f = 0; f + 1 < n; ++f) {
        double al = a[f], ar = a[f + 1];
        if (((al <= pure_band) && (ar >= 1.0 - pure_band)) ||
            ((ar <= pure_band) && (al >= 1.0 - pure_band))) { sharp_jump = true; break; }
    }
    int lh = low + high; if (lh < 1) lh = 1;
    bool narrow = has_low && has_high && (mixed <= lh);
    return has_low && has_high && (sharp_jump || narrow);
}

// _passive_pressure_equilibrium_transport(W_n) (1515-1537).
inline bool passive_transport(const std::vector<double>& a,
                              const std::vector<double>& u,
                              const std::vector<double>& p) {
    int n = (int)a.size();
    if (n < 2) return false;
    double amn = a[0], amx = a[0];
    for (double v : a) { amn = dmin(amn, v); amx = dmax(amx, v); }
    if ((amx - amn) <= 1.0e-10) return false;
    double pabs = 0.0, uabs = 0.0, pjmp = 0.0, ujmp = 0.0;
    for (double v : p) pabs = dmax(pabs, std::fabs(v));
    for (double v : u) uabs = dmax(uabs, std::fabs(v));
    for (int i = 0; i + 1 < n; ++i) {
        pjmp = dmax(pjmp, std::fabs(p[i + 1] - p[i]));
        ujmp = dmax(ujmp, std::fabs(u[i + 1] - u[i]));
    }
    double p_scale = dmax(pabs, 1.0), u_scale = dmax(uabs, 1.0);
    return (pjmp / p_scale) <= 1.0e-10 && (ujmp / u_scale) <= 1.0e-10;
}

// _pure_material_cell_mask(alpha, pure_tol) (121-140).
inline std::vector<char> pure_material_cells(const std::vector<double>& a,
                                             double pure_tol) {
    int n = (int)a.size();
    std::vector<char> cells(n, 0);
    if (n < 2) return cells;
    double jt = EPS4;
    for (int f = 0; f + 1 < n; ++f) {
        double al = a[f], ar = a[f + 1];
        bool pf = (std::fabs(ar - al) > jt) &&
                  ((al <= pure_tol) || (al >= 1.0 - pure_tol)
                 || (ar <= pure_tol) || (ar >= 1.0 - pure_tol));
        if (pf) { cells[f] = 1; cells[f + 1] = 1; }
    }
    return cells;
}

// Mixture-primitive MUSCL-Hancock L/R states (imex_ad.py 1771-1835), vanleer.
struct HancockLR {
    std::vector<double> rho_L, rho_R, u_L, u_R, p_L, p_R, y_L, y_R, a_L, a_R;
};
inline HancockLR mixture_hancock_lr(const std::vector<double>& a_ext,
                                    const std::vector<double>& T1_ext,
                                    const std::vector<double>& T2_ext,
                                    const std::vector<double>& u_ext,
                                    const std::vector<double>& p_ext,
                                    const std::vector<double>& c_mix_sq_ext,
                                    const EOS& eos1, const EOS& eos2,
                                    double dt, double dx,
                                    TvdLimiter tvd_kind) {
    int ne = (int)a_ext.size();
    std::vector<double> rho_ext(ne), y1_ext(ne);
    for (int k = 0; k < ne; ++k) {
        double rho1 = dmax(eos1.density(p_ext[k], T1_ext[k]), MU_EPS);
        double rho2 = dmax(eos2.density(p_ext[k], T2_ext[k]), MU_EPS);
        double q1 = dmax(a_ext[k] * rho1, 0.0);
        double q2 = dmax((1.0 - a_ext[k]) * rho2, 0.0);
        double rho = dmax(q1 + q2, MU_EPS);
        rho_ext[k] = rho;
        y1_ext[k] = clip(q1 / rho, 0.0, 1.0);
    }
    auto slopes = [&](const std::vector<double>& phi) {
        std::vector<double> out(ne, 0.0);
        for (int k = 1; k + 1 < ne; ++k)
            out[k] = tvd_pair(phi[k] - phi[k - 1], phi[k + 1] - phi[k],
                              tvd_kind);
        return out;
    };
    std::vector<double> drho = slopes(rho_ext), dy1 = slopes(y1_ext),
                        du = slopes(u_ext), dp = slopes(p_ext), da = slopes(a_ext);
    double inv_dx = 1.0 / dx;
    std::vector<double> rho_t(ne), y_t(ne), u_t(ne), p_t(ne), a_t(ne);
    for (int k = 0; k < ne; ++k) {
        double c2 = dmax(c_mix_sq_ext[k], MU_EPS);
        double rho_safe = dmax(rho_ext[k], MU_EPS);
        double rho_x = drho[k] * inv_dx, y_x = dy1[k] * inv_dx,
               u_x = du[k] * inv_dx, p_x = dp[k] * inv_dx;
        rho_t[k] = -u_ext[k] * rho_x - rho_safe * u_x;
        y_t[k]   = -u_ext[k] * y_x;
        u_t[k]   = -u_ext[k] * u_x - p_x / rho_safe;
        p_t[k]   = -u_ext[k] * p_x - rho_safe * c2 * u_x;
        a_t[k]   = -u_ext[k] * da[k] * inv_dx;
    }
    int nf = ne - 1;
    HancockLR s;
    s.rho_L.resize(nf); s.rho_R.resize(nf); s.u_L.resize(nf); s.u_R.resize(nf);
    s.p_L.resize(nf); s.p_R.resize(nf); s.y_L.resize(nf); s.y_R.resize(nf);
    s.a_L.resize(nf); s.a_R.resize(nf);
    for (int f = 0; f < nf; ++f) {
        double rho_L = rho_ext[f]     + 0.5 * drho[f]     + 0.5 * dt * rho_t[f];
        double rho_R = rho_ext[f + 1] - 0.5 * drho[f + 1] + 0.5 * dt * rho_t[f + 1];
        double y_L   = y1_ext[f]      + 0.5 * dy1[f]      + 0.5 * dt * y_t[f];
        double y_R   = y1_ext[f + 1]  - 0.5 * dy1[f + 1]  + 0.5 * dt * y_t[f + 1];
        double u_L   = u_ext[f]       + 0.5 * du[f]       + 0.5 * dt * u_t[f];
        double u_R   = u_ext[f + 1]   - 0.5 * du[f + 1]   + 0.5 * dt * u_t[f + 1];
        double p_L   = p_ext[f]       + 0.5 * dp[f]       + 0.5 * dt * p_t[f];
        double p_R   = p_ext[f + 1]   - 0.5 * dp[f + 1]   + 0.5 * dt * p_t[f + 1];
        double a_L   = a_ext[f]       + 0.5 * da[f]       + 0.5 * dt * a_t[f];
        double a_R   = a_ext[f + 1]   - 0.5 * da[f + 1]   + 0.5 * dt * a_t[f + 1];
        s.rho_L[f] = dmax(rho_L, MU_EPS); s.rho_R[f] = dmax(rho_R, MU_EPS);
        s.y_L[f] = clip(y_L, 0.0, 1.0);   s.y_R[f] = clip(y_R, 0.0, 1.0);
        s.p_L[f] = dmax(p_L, 1.0e-12);    s.p_R[f] = dmax(p_R, 1.0e-12);
        s.a_L[f] = clip(a_L, 1.0e-12, 1.0 - 1.0e-12);
        s.a_R[f] = clip(a_R, 1.0e-12, 1.0 - 1.0e-12);
        s.u_L[f] = u_L; s.u_R[f] = u_R;
    }
    return s;
}

// Spatial-only mixture L/R states (imex_ad.py::_mixture_primitive_lr_states).
// HLLC deliberately uses this branch without the Hancock predictor.
inline HancockLR mixture_lr(const std::vector<double>& a_ext,
                            const std::vector<double>& T1_ext,
                            const std::vector<double>& T2_ext,
                            const std::vector<double>& u_ext,
                            const std::vector<double>& p_ext,
                            const EOS& eos1, const EOS& eos2,
                            TvdLimiter tvd_kind) {
    const int ne = (int)a_ext.size(), nf = ne - 1;
    std::vector<double> rho(ne), y(ne);
    for (int k = 0; k < ne; ++k) {
        const double r1 = dmax(eos1.density(p_ext[k], T1_ext[k]), MU_EPS);
        const double r2 = dmax(eos2.density(p_ext[k], T2_ext[k]), MU_EPS);
        const double q1 = dmax(a_ext[k] * r1, 0.0), q2 = dmax((1.0 - a_ext[k]) * r2, 0.0);
        rho[k] = dmax(q1 + q2, MU_EPS); y[k] = clip(q1 / rho[k], 0.0, 1.0);
    }
    HancockLR s;
    s.rho_L.resize(nf); s.rho_R.resize(nf); s.u_L.resize(nf); s.u_R.resize(nf);
    s.p_L.resize(nf); s.p_R.resize(nf); s.y_L.resize(nf); s.y_R.resize(nf);
    s.a_L.resize(nf); s.a_R.resize(nf);
    reconstruct_lr_faces(rho.data(), ne, tvd_kind, MU_EPS, s.rho_L.data(), s.rho_R.data());
    reconstruct_lr_faces(y.data(), ne, tvd_kind, -1.0, s.y_L.data(), s.y_R.data());
    reconstruct_lr_faces(u_ext.data(), ne, tvd_kind, -1.0, s.u_L.data(), s.u_R.data());
    reconstruct_lr_faces(p_ext.data(), ne, tvd_kind, 1.0e-12, s.p_L.data(), s.p_R.data());
    reconstruct_lr_faces(a_ext.data(), ne, tvd_kind, -1.0, s.a_L.data(), s.a_R.data());
    for (int f = 0; f < nf; ++f) {
        s.rho_L[f] = dmax(s.rho_L[f], MU_EPS); s.rho_R[f] = dmax(s.rho_R[f], MU_EPS);
        s.y_L[f] = clip(s.y_L[f], 0.0, 1.0); s.y_R[f] = clip(s.y_R[f], 0.0, 1.0);
        s.a_L[f] = clip(s.a_L[f], 1.0e-12, 1.0 - 1.0e-12);
        s.a_R[f] = clip(s.a_R[f], 1.0e-12, 1.0 - 1.0e-12);
    }
    return s;
}

inline double clip_stencil(double value, const std::vector<double>& cell, int i) {
    const int lo_i = i > 0 ? i - 1 : 0;
    const int hi_i = i + 2 < (int)cell.size() ? i + 2 : (int)cell.size();
    double lo = cell[lo_i], hi = cell[lo_i];
    for (int k = lo_i + 1; k < hi_i; ++k) { lo = dmin(lo, cell[k]); hi = dmax(hi, cell[k]); }
    return clip(value, lo, hi);
}

// Characteristic rho/u/p slopes and L/R states
// (imex_ad.py::_characteristic_{primitive_slopes,mixture_lr_states}).
inline HancockLR characteristic_lr(const std::vector<double>& a_ext,
                                   const std::vector<double>& T1_ext,
                                   const std::vector<double>& T2_ext,
                                   const std::vector<double>& u_ext,
                                   const std::vector<double>& p_ext,
                                   const std::vector<double>& c2_ext,
                                   const EOS& eos1, const EOS& eos2,
                                   TvdLimiter tvd_kind) {
    const int ne = (int)a_ext.size(), nf = ne - 1;
    std::vector<double> rho(ne), y(ne), drho(ne, 0.0), du(ne, 0.0), dp(ne, 0.0);
    for (int k = 0; k < ne; ++k) {
        const double r1 = dmax(eos1.density(p_ext[k], T1_ext[k]), MU_EPS);
        const double r2 = dmax(eos2.density(p_ext[k], T2_ext[k]), MU_EPS);
        const double q1 = dmax(a_ext[k] * r1, 0.0), q2 = dmax((1.0 - a_ext[k]) * r2, 0.0);
        rho[k] = dmax(q1 + q2, MU_EPS); y[k] = clip(q1 / rho[k], 0.0, 1.0);
    }
    for (int k = 1; k + 1 < ne; ++k) {
        const double r = dmax(rho[k], MU_EPS), c = std::sqrt(dmax(c2_ext[k], MU_EPS));
        const double ic2 = 1.0 / dmax(c * c, MU_EPS);
        const double dlr = rho[k] - rho[k - 1], dlu = u_ext[k] - u_ext[k - 1], dlp = p_ext[k] - p_ext[k - 1];
        const double drr = rho[k + 1] - rho[k], dru = u_ext[k + 1] - u_ext[k], drp = p_ext[k + 1] - p_ext[k];
        const double lm = .5 * (dlp * ic2 - r * dlu / c), lp = .5 * (dlp * ic2 + r * dlu / c), l0 = dlr - dlp * ic2;
        const double rm = .5 * (drp * ic2 - r * dru / c), rp = .5 * (drp * ic2 + r * dru / c), r0 = drr - drp * ic2;
        const double am = tvd_pair(lm, rm, tvd_kind);
        const double ap = tvd_pair(lp, rp, tvd_kind);
        const double a0 = tvd_pair(l0, r0, tvd_kind);
        drho[k] = a0 + am + ap; du[k] = c / r * (ap - am); dp[k] = c * c * (am + ap);
    }
    HancockLR s;
    s.rho_L.resize(nf); s.rho_R.resize(nf); s.u_L.resize(nf); s.u_R.resize(nf);
    s.p_L.resize(nf); s.p_R.resize(nf); s.y_L.resize(nf); s.y_R.resize(nf);
    s.a_L.resize(nf); s.a_R.resize(nf);
    reconstruct_lr_faces(y.data(), ne, tvd_kind, -1.0, s.y_L.data(), s.y_R.data());
    reconstruct_lr_faces(a_ext.data(), ne, tvd_kind, -1.0, s.a_L.data(), s.a_R.data());
    for (int f = 0; f < nf; ++f) {
        s.rho_L[f] = dmax(clip_stencil(rho[f] + .5 * drho[f], rho, f), MU_EPS);
        s.rho_R[f] = dmax(clip_stencil(rho[f + 1] - .5 * drho[f + 1], rho, f + 1), MU_EPS);
        s.u_L[f] = clip_stencil(u_ext[f] + .5 * du[f], u_ext, f);
        s.u_R[f] = clip_stencil(u_ext[f + 1] - .5 * du[f + 1], u_ext, f + 1);
        s.p_L[f] = dmax(clip_stencil(p_ext[f] + .5 * dp[f], p_ext, f), 1.0e-12);
        s.p_R[f] = dmax(clip_stencil(p_ext[f + 1] - .5 * dp[f + 1], p_ext, f + 1), 1.0e-12);
        s.y_L[f] = clip(s.y_L[f], 0.0, 1.0); s.y_R[f] = clip(s.y_R[f], 0.0, 1.0);
        s.a_L[f] = clip(s.a_L[f], 1.0e-12, 1.0 - 1.0e-12);
        s.a_R[f] = clip(s.a_R[f], 1.0e-12, 1.0 - 1.0e-12);
    }
    return s;
}

struct CharacteristicFaces { std::vector<double> rho, y, u, p; };

inline CharacteristicFaces characteristic_upwind_faces(
        const std::vector<double>& a_ext, const std::vector<double>& T1_ext,
        const std::vector<double>& T2_ext, const std::vector<double>& u_ext,
        const std::vector<double>& p_ext, const std::vector<double>& c2_ext,
        const std::vector<double>& u_face, const EOS& eos1, const EOS& eos2,
        double dt, double dx, TvdLimiter tvd_kind) {
    const int ne = (int)a_ext.size(), nf = ne - 1;
    std::vector<double> rho(ne), y(ne), drho(ne, 0.0), du(ne, 0.0), dp(ne, 0.0);
    for (int k = 0; k < ne; ++k) {
        const double r1 = dmax(eos1.density(p_ext[k], T1_ext[k]), MU_EPS);
        const double r2 = dmax(eos2.density(p_ext[k], T2_ext[k]), MU_EPS);
        const double q1 = dmax(a_ext[k] * r1, 0.0), q2 = dmax((1.0 - a_ext[k]) * r2, 0.0);
        rho[k] = dmax(q1 + q2, MU_EPS); y[k] = clip(q1 / rho[k], 0.0, 1.0);
    }
    for (int k = 1; k + 1 < ne; ++k) {
        const double r = dmax(rho[k], MU_EPS), c = std::sqrt(dmax(c2_ext[k], MU_EPS));
        const double ic2 = 1.0 / dmax(c * c, MU_EPS);
        const double dlr = rho[k] - rho[k - 1], dlu = u_ext[k] - u_ext[k - 1], dlp = p_ext[k] - p_ext[k - 1];
        const double drr = rho[k + 1] - rho[k], dru = u_ext[k + 1] - u_ext[k], drp = p_ext[k + 1] - p_ext[k];
        const double lm = .5 * (dlp * ic2 - r * dlu / c), lp = .5 * (dlp * ic2 + r * dlu / c), l0 = dlr - dlp * ic2;
        const double rm = .5 * (drp * ic2 - r * dru / c), rp = .5 * (drp * ic2 + r * dru / c), r0 = drr - drp * ic2;
        const double am = tvd_pair(lm, rm, tvd_kind);
        const double ap = tvd_pair(lp, rp, tvd_kind);
        const double a0 = tvd_pair(l0, r0, tvd_kind);
        drho[k] = a0 + am + ap; du[k] = c / r * (ap - am); dp[k] = c * c * (am + ap);
    }
    CharacteristicFaces out; out.rho.resize(nf); out.y.resize(nf); out.u.resize(nf); out.p.resize(nf);
    reconstruct_upwind_faces(y.data(), ne, u_face.data(), tvd_kind, dt, dx, true, 0.0, out.y.data());
    for (int f = 0; f < nf; ++f) {
        const bool left = u_face[f] >= 0.0; const int k = left ? f : f + 1;
        const double sign = (left ? .5 : -.5) * (1.0 - dmin(1.0, std::fabs(u_face[f]) * dt / dmax(dx, MU_EPS)));
        out.rho[f] = dmax(clip_stencil(rho[k] + sign * drho[k], rho, k), MU_EPS);
        out.u[f] = clip_stencil(u_ext[k] + sign * du[k], u_ext, k);
        out.p[f] = dmax(clip_stencil(p_ext[k] + sign * dp[k], p_ext, k), 1.0e-12);
        out.y[f] = clip(out.y[f], 0.0, 1.0);
    }
    return out;
}

// Zalesak cell-update FCT limiter shared by apply_update_lmp (2352-2423) and
// limit_high_order_flux (2527-2588). `periodic_ext` selects the low_update ghost
// duplication (wrap for periodic, edge-copy otherwise). Modifies theta in place.
inline void zalesak_update_limit(std::vector<double>& theta,
                                 const std::vector<double>& cell_now,
                                 const std::vector<double>& low_face,
                                 const std::vector<double>& anti_flux,
                                 const std::vector<double>& u_star,
                                 const std::vector<double>* extra_face_flux,
                                 double dt, double inv_dx, bool periodic_ext) {
    int n_cell = (int)cell_now.size();
    int n_face = (int)anti_flux.size();
    std::vector<double> low_flux(n_face);
    for (int f = 0; f < n_face; ++f) low_flux[f] = low_face[f] * u_star[f];
    std::vector<double> low_update(n_cell);
    for (int i = 0; i < n_cell; ++i) {
        low_update[i] = cell_now[i] - dt * inv_dx * (low_flux[i + 1] - low_flux[i]);
        if (extra_face_flux)
            low_update[i] -= dt * inv_dx * ((*extra_face_flux)[i + 1] - (*extra_face_flux)[i]);
    }
    std::vector<double> lo(n_cell), hi(n_cell);
    for (int i = 0; i < n_cell; ++i) {
        double lc = low_update[i];
        double ll = (i == 0) ? (periodic_ext ? low_update[n_cell - 1] : low_update[0]) : low_update[i - 1];
        double lr = (i == n_cell - 1) ? (periodic_ext ? low_update[0] : low_update[n_cell - 1]) : low_update[i + 1];
        bool monotone = (lc - ll) * (lr - lc) >= 0.0;
        double lo3 = dmin(dmin(ll, lc), lr), hi3 = dmax(dmax(ll, lc), lr);
        lo[i] = monotone ? dmin(ll, lr) : lo3;
        hi[i] = monotone ? dmax(ll, lr) : hi3;
    }
    std::vector<double> allow_pos(n_cell), allow_neg(n_cell),
                        sum_pos(n_cell, 0.0), sum_neg(n_cell, 0.0);
    for (int i = 0; i < n_cell; ++i) {
        allow_pos[i] = dmax(hi[i] - low_update[i], 0.0);
        allow_neg[i] = dmax(low_update[i] - lo[i], 0.0);
    }
    for (int f = 0; f < n_face; ++f) {
        double af = anti_flux[f];
        if (f > 0) {
            double c = -dt * inv_dx * af; int i = f - 1;
            if (c >= 0.0) sum_pos[i] += c; else sum_neg[i] += -c;
        }
        if (f < n_cell) {
            double c = dt * inv_dx * af; int i = f;
            if (c >= 0.0) sum_pos[i] += c; else sum_neg[i] += -c;
        }
    }
    std::vector<double> r_pos(n_cell, 1.0), r_neg(n_cell, 1.0);
    for (int i = 0; i < n_cell; ++i) {
        if (sum_pos[i] > MU_EPS) r_pos[i] = dmin(1.0, allow_pos[i] / sum_pos[i]);
        if (sum_neg[i] > MU_EPS) r_neg[i] = dmin(1.0, allow_neg[i] / sum_neg[i]);
    }
    for (int f = 0; f < n_face; ++f) {
        double af = anti_flux[f];
        double lim = 1.0;
        if (f > 0) {
            double c = -dt * inv_dx * af;
            lim = dmin(lim, c >= 0.0 ? r_pos[f - 1] : r_neg[f - 1]);
        }
        if (f < n_cell) {
            double c = dt * inv_dx * af;
            lim = dmin(lim, c >= 0.0 ? r_pos[f] : r_neg[f]);
        }
        theta[f] = dmin(theta[f], lim);
    }
    for (int f = 0; f < n_face; ++f) theta[f] = clip(theta[f], 0.0, 1.0);
}

} // namespace detail

// ── the material update driver ─────────────────────────────────────────────
inline MaterialResult material_update(const std::vector<double>& a1,
                                      const std::vector<double>& T1,
                                      const std::vector<double>& T2,
                                      const std::vector<double>& u,
                                      const std::vector<double>& p,
                                      double dt, double dx,
                                      const EOS& eos1, const EOS& eos2,
                                      const MaterialConfig& cfg) {
    using namespace detail;
    const int n = (int)a1.size();
    const int ne = n + 2;
    const int nf = n + 1;
    const double inv_dx = 1.0 / dx;
    const double apt = cfg.alpha_pure_tol;
    const double pure_tol_auto = dmax(apt, EPS4);

    // U_n conservative.
    std::vector<double> Un0(n), Un1(n), Un2(n), Un3(n), Un4(n);
    for (int i = 0; i < n; ++i) {
        ConsU U = prim_to_cons_W(PrimW{a1[i], T1[i], T2[i], u[i], p[i]}, eos1, eos2);
        Un0[i] = U.m1; Un1[i] = U.m2; Un2[i] = U.mom; Un3[i] = U.rhoE; Un4[i] = U.a1;
    }
    // Ghost-extended primitive state.
    std::optional<double> u_left = cfg.u_inlet_l;
    std::optional<double> p_left = cfg.p_inlet_l;
    if (cfg.bc_l == BC5::InletAcoustic) {
        // boundary.extend_W: prescribed incoming J+ and extrapolated outgoing
        // J- form a coupled primitive ghost at the left acoustic inlet.
        PhaseAcoustic pa0 = phase_acoustic(eos1, eos2, a1[0], T1[0], T2[0], p[0], apt,
                                            cfg.mixture_sound_speed_kind);
        double Z0 = dmax(pa0.Z, MU_EPS);
        double u_in = cfg.u_inlet_l ? *cfg.u_inlet_l : u[0];
        double p_in = cfg.p_inlet_l ? *cfg.p_inlet_l : p[0];
        double Jp = (u_in - u[0]) + (p_in - p[0]) / Z0;
        double Jm = 0.0;
        if (n >= 2) Jm = (u[1] - u[0]) - (p[1] - p[0]) / Z0;
        u_left = u[0] + 0.5 * (Jp + Jm);
        p_left = p[0] + 0.5 * Z0 * (Jp - Jm);
    }
    std::vector<double> a_ext = extend1(a1, cfg.bc_l, cfg.bc_r, false, cfg.alpha_inlet_l);
    std::vector<double> T1_ext = extend1(T1, cfg.bc_l, cfg.bc_r, false, cfg.T1_inlet_l);
    std::vector<double> T2_ext = extend1(T2, cfg.bc_l, cfg.bc_r, false, cfg.T2_inlet_l);
    std::vector<double> u_ext = extend1(u, cfg.bc_l, cfg.bc_r, true, u_left);
    std::vector<double> p_ext = extend1(p, cfg.bc_l, cfg.bc_r, false, p_left, cfg.p_outlet_r);
    // U_ext conservative.
    std::vector<double> Ue0(ne), Ue1(ne), Ue2(ne), Ue3(ne);
    for (int k = 0; k < ne; ++k) {
        ConsU U = prim_to_cons_W(PrimW{a_ext[k], T1_ext[k], T2_ext[k], u_ext[k], p_ext[k]}, eos1, eos2);
        Ue0[k] = U.m1; Ue1[k] = U.m2; Ue2[k] = U.mom; Ue3[k] = U.rhoE;
    }
    // Phase acoustic (c_mix_sq, Z) on the extended state.
    std::vector<double> c_ext(ne), Z_ext(ne);
    for (int k = 0; k < ne; ++k) {
        PhaseAcoustic pa = phase_acoustic(eos1, eos2, a_ext[k], T1_ext[k], T2_ext[k], p_ext[k], apt,
                                           cfg.mixture_sound_speed_kind);
        c_ext[k] = pa.c_mix_sq; Z_ext[k] = pa.Z;
    }

    // Wood-Z acoustic base p*/u*.
    std::vector<double> p_star(nf), u_star(nf);
    for (int f = 0; f < nf; ++f) {
        double ZL = Z_ext[f], ZR = Z_ext[f + 1];
        double pL = p_ext[f], pR = p_ext[f + 1], uL = u_ext[f], uR = u_ext[f + 1];
        double den = dmax(ZL + ZR, MU_EPS);
        p_star[f] = (ZR * pL + ZL * pR + ZL * ZR * (uL - uR)) / den;
        u_star[f] = (pL - pR + ZL * uL + ZR * uR) / den;
    }

    // Branch gate: mixture-primitive reconstruction on this extended state.
    bool mixture_recon = mixture_recon_enabled(a_ext, u_ext, p_ext);
    TvdLimiter primitive_tvd = mixture_tvd_kind_for_state(
        a_ext, u_ext, cfg.primitive_tvd_limiter);
    if (primitive_tvd == TvdLimiter::Superbee &&
        five_eq::pressure_jump_stiff_to_soft(
            a_ext, T1_ext, T2_ext, p_ext, eos1, eos2,
            apt, cfg.mixture_sound_speed_kind)) {
        primitive_tvd = TvdLimiter::VanLeer;
    }
    double a_min = a1[0], a_max = a1[0];
    for (double av : a1) { a_min = dmin(a_min, av); a_max = dmax(a_max, av); }
    const bool characteristic_recon = cfg.characteristic_reconstruction && (a_max - a_min <= EPS4);
    HancockLR hlr;   // computed once, shared by SLAU2 + material recon when active.
    bool have_hlr = false;
    auto ensure_hlr = [&]() {
        if (!have_hlr) {
            hlr = mixture_hancock_lr(a_ext, T1_ext, T2_ext, u_ext, p_ext,
                                     c_ext, eos1, eos2, dt, dx, primitive_tvd);
            have_hlr = true;
        }
    };

    // SLAU2/HLLC material face velocity.  The optional characteristic branch
    // is allowed only on composition-uniform states, exactly as Python.
    {
        std::vector<double> rho_L(nf), rho_R(nf), uL(nf), uR(nf), pL(nf), pR(nf);
        if (cfg.material_flux == MaterialFlux::HllcContact && mixture_recon) {
            HancockLR lr = mixture_lr(a_ext, T1_ext, T2_ext, u_ext, p_ext,
                                      eos1, eos2, primitive_tvd);
            for (int f = 0; f < nf; ++f) {
                rho_L[f] = lr.rho_L[f]; rho_R[f] = lr.rho_R[f];
                uL[f] = lr.u_L[f]; uR[f] = lr.u_R[f]; pL[f] = lr.p_L[f]; pR[f] = lr.p_R[f];
            }
        } else if (characteristic_recon) {
            HancockLR lr = characteristic_lr(a_ext, T1_ext, T2_ext, u_ext,
                                              p_ext, c_ext, eos1, eos2,
                                              primitive_tvd);
            for (int f = 0; f < nf; ++f) {
                rho_L[f] = lr.rho_L[f]; rho_R[f] = lr.rho_R[f];
                uL[f] = lr.u_L[f]; uR[f] = lr.u_R[f]; pL[f] = lr.p_L[f]; pR[f] = lr.p_R[f];
            }
        } else if (mixture_recon) {
            ensure_hlr();
            for (int f = 0; f < nf; ++f) {
                rho_L[f] = hlr.rho_L[f]; rho_R[f] = hlr.rho_R[f];
                uL[f] = hlr.u_L[f]; uR[f] = hlr.u_R[f];
                pL[f] = hlr.p_L[f]; pR[f] = hlr.p_R[f];
            }
        } else {
            // reconstruct_lr_faces on T1,T2,u,p then EOS mixture rho.
            std::vector<double> T1L(nf), T1R(nf), T2L(nf), T2R(nf);
            reconstruct_lr_faces(T1_ext.data(), ne, primitive_tvd, 1.0, T1L.data(), T1R.data());
            reconstruct_lr_faces(T2_ext.data(), ne, primitive_tvd, 1.0, T2L.data(), T2R.data());
            reconstruct_lr_faces(u_ext.data(), ne, primitive_tvd, -1.0, uL.data(), uR.data());
            reconstruct_lr_faces(p_ext.data(), ne, primitive_tvd, 1.0e-12, pL.data(), pR.data());
            for (int f = 0; f < nf; ++f) {
                double aL = clip(a_ext[f], 0.0, 1.0), aR = clip(a_ext[f + 1], 0.0, 1.0);
                double r1L = dmax(eos1.density(pL[f], T1L[f]), MU_EPS);
                double r1R = dmax(eos1.density(pR[f], T1R[f]), MU_EPS);
                double r2L = dmax(eos2.density(pL[f], T2L[f]), MU_EPS);
                double r2R = dmax(eos2.density(pR[f], T2R[f]), MU_EPS);
                rho_L[f] = dmax(aL * r1L + (1.0 - aL) * r2L, MU_EPS);
                rho_R[f] = dmax(aR * r1R + (1.0 - aR) * r2R, MU_EPS);
            }
        }
        for (int f = 0; f < nf; ++f) {
            if (cfg.material_flux == MaterialFlux::HllcContact) {
                const double cL = std::sqrt(dmax(c_ext[f], MU_EPS));
                const double cR = std::sqrt(dmax(c_ext[f + 1], MU_EPS));
                const double sL = dmin(uL[f] - cL, uR[f] - cR);
                const double sR = dmax(uL[f] + cL, uR[f] + cR);
                const double den = rho_L[f] * (sL - uL[f]) - rho_R[f] * (sR - uR[f]);
                const double den_safe = std::fabs(den) > MU_EPS ? den : (den + 1.0e-300 >= 0.0 ? MU_EPS : -MU_EPS);
                const double sM = (pR[f] - pL[f] + rho_L[f] * uL[f] * (sL - uL[f])
                    - rho_R[f] * uR[f] * (sR - uR[f])) / den_safe;
                const double pmL = pL[f] + rho_L[f] * (sL - uL[f]) * (sM - uL[f]);
                const double pmR = pR[f] + rho_R[f] * (sR - uR[f]) * (sM - uR[f]);
                const double pm = .5 * (pmL + pmR);
                if (std::isfinite(sM) && std::isfinite(pm) && (rho_L[f] > MU_EPS) &&
                    (rho_R[f] > MU_EPS) && (pm > 0.0) && (sR > sL)) {
                    p_star[f] = pm; u_star[f] = sM;
                }
            } else {
                Slau2Face fc = slau2_face(rho_L[f], rho_R[f], uL[f], uR[f], pL[f], pR[f],
                                          c_ext[f], c_ext[f + 1]);
                if (std::isfinite(fc.u_face) && std::isfinite(fc.p_face) &&
                    (rho_L[f] > MU_EPS) && (rho_R[f] > MU_EPS)) {
                    p_star[f] = fc.p_face; u_star[f] = fc.u_face;
                }
            }
        }
        if (cfg.bc_l == BC5::Reflective) { p_star[0] = p_ext[1]; u_star[0] = 0.0; }
        if (cfg.bc_r == BC5::Reflective) { p_star[nf - 1] = p_ext[ne - 2]; u_star[nf - 1] = 0.0; }
    }
    // material_update reflective u* override (2218-2221).
    if (cfg.bc_l == BC5::Reflective) u_star[0] = 0.0;
    if (cfg.bc_r == BC5::Reflective) u_star[nf - 1] = 0.0;

    // Upwind alpha + Python-configured T-MLP-u primitive reconstruction.
    std::vector<char> upwind_left(nf);
    std::vector<double> alpha_upwind(nf);
    for (int f = 0; f < nf; ++f) {
        upwind_left[f] = (u_star[f] >= 0.0) ? 1 : 0;
        alpha_upwind[f] = upwind_left[f] ? a_ext[f] : a_ext[f + 1];
    }
    std::vector<double> T1_f(nf), T2_f(nf), u_adv_f(nf), p_adv_f(nf);
    reconstruct_upwind_faces(T1_ext.data(), ne, u_star.data(), primitive_tvd, dt, dx, true, 1.0, T1_f.data());
    reconstruct_upwind_faces(T2_ext.data(), ne, u_star.data(), primitive_tvd, dt, dx, true, 1.0, T2_f.data());
    reconstruct_upwind_faces(u_ext.data(), ne, u_star.data(), primitive_tvd, dt, dx, true, -1.0, u_adv_f.data());
    reconstruct_upwind_faces(p_ext.data(), ne, u_star.data(), primitive_tvd, dt, dx, true, 1.0e-12, p_adv_f.data());

    std::vector<double> rho1_f(nf), rho2_f(nf);
    for (int f = 0; f < nf; ++f) {
        rho1_f[f] = dmax(eos1.density(p_adv_f[f], T1_f[f]), MU_EPS);
        rho2_f[f] = dmax(eos2.density(p_adv_f[f], T2_f[f]), MU_EPS);
    }
    // mixture / density reconstruction branch.
    std::vector<double> mix_rho_f, mix_y1_f;   // empty => not on mixture path
    if (characteristic_recon) {
        CharacteristicFaces cf = characteristic_upwind_faces(
            a_ext, T1_ext, T2_ext, u_ext, p_ext, c_ext, u_star, eos1, eos2,
            dt, dx, primitive_tvd);
        mix_rho_f = cf.rho; mix_y1_f = cf.y;
        for (int f = 0; f < nf; ++f) {
            const double a_mix = dmax(alpha_upwind[f], 1.0e-12);
            const double a2_mix = dmax(1.0 - alpha_upwind[f], 1.0e-12);
            u_adv_f[f] = cf.u[f]; p_adv_f[f] = cf.p[f];
            rho1_f[f] = dmax(cf.y[f] * cf.rho[f] / a_mix, MU_EPS);
            rho2_f[f] = dmax((1.0 - cf.y[f]) * cf.rho[f] / a2_mix, MU_EPS);
        }
    } else if (mixture_recon) {
        ensure_hlr();
        mix_rho_f.resize(nf); mix_y1_f.resize(nf);
        for (int f = 0; f < nf; ++f) {
            bool L = upwind_left[f];
            double a_mix = L ? hlr.a_L[f] : hlr.a_R[f];
            double rho   = L ? hlr.rho_L[f] : hlr.rho_R[f];
            double y1    = L ? hlr.y_L[f] : hlr.y_R[f];
            a_mix = clip(a_mix, 1.0e-12, 1.0 - 1.0e-12);
            y1 = clip(y1, 0.0, 1.0);
            u_adv_f[f] = L ? hlr.u_L[f] : hlr.u_R[f];
            p_adv_f[f] = L ? hlr.p_L[f] : hlr.p_R[f];
            double q1_mix = y1 * rho, q2_mix = (1.0 - y1) * rho;
            rho1_f[f] = dmax(q1_mix / a_mix, MU_EPS);
            rho2_f[f] = dmax(q2_mix / (1.0 - a_mix), MU_EPS);
            mix_rho_f[f] = rho; mix_y1_f[f] = y1;
        }
    } else {
        // FIVE_EQ_IMEX_DENSITY_RECON == "1": reconstruct phase densities (minmod).
        std::vector<double> rho1_ext(ne), rho2_ext(ne);
        for (int k = 0; k < ne; ++k) {
            rho1_ext[k] = dmax(eos1.density(p_ext[k], T1_ext[k]), MU_EPS);
            rho2_ext[k] = dmax(eos2.density(p_ext[k], T2_ext[k]), MU_EPS);
        }
        reconstruct_upwind_faces(rho1_ext.data(), ne, u_star.data(), TvdLimiter::Minmod, dt, dx, true, MU_EPS, rho1_f.data());
        reconstruct_upwind_faces(rho2_ext.data(), ne, u_star.data(), TvdLimiter::Minmod, dt, dx, true, MU_EPS, rho2_f.data());
    }

    std::vector<double> e1_f(nf), e2_f(nf), E1_f(nf), E2_f(nf),
                        q1_cons(nf), q2_cons(nf), m_cons(nf), rE_cons(nf);
    for (int f = 0; f < nf; ++f) {
        e1_f[f] = eos1.energy(rho1_f[f], p_adv_f[f]);
        e2_f[f] = eos2.energy(rho2_f[f], p_adv_f[f]);
        double ke = 0.5 * u_adv_f[f] * u_adv_f[f];
        E1_f[f] = e1_f[f] + ke; E2_f[f] = e2_f[f] + ke;
        q1_cons[f] = alpha_upwind[f] * rho1_f[f];
        q2_cons[f] = (1.0 - alpha_upwind[f]) * rho2_f[f];
        m_cons[f] = (q1_cons[f] + q2_cons[f]) * u_adv_f[f];
        rE_cons[f] = q1_cons[f] * E1_f[f] + q2_cons[f] * E2_f[f];
    }

    // ── adaptive_bvd sharp-alpha FCT block ──────────────────────────────────
    bool collocated = collocated_pjump(a1, p, pure_tol_auto);
    AlphaTvd alpha_tvd = collocated ? AlphaTvd::Superbee : AlphaTvd::Umist;
    std::vector<double> alpha_sharp(nf);
    adaptive_bvd_alpha_face(a_ext.data(), ne, u_star.data(), nf, dt, dx,
                            alpha_tvd, apt, alpha_sharp.data());
    for (int f = 0; f < nf; ++f) alpha_sharp[f] = clip(alpha_sharp[f], 1.0e-12, 1.0 - 1.0e-12);
    std::vector<double> delta_alpha(nf);
    for (int f = 0; f < nf; ++f) delta_alpha[f] = alpha_sharp[f] - alpha_upwind[f];
    std::vector<double> theta(nf, 1.0);

    bool alpha_fct = sharp_alpha_present(a1, pure_tol_auto) && !collocated;
    bool periodic = (cfg.bc_l == BC5::Periodic && cfg.bc_r == BC5::Periodic);

    auto apply_lmp = [&](const std::vector<double>& base, const std::vector<double>& coeff,
                         int uidx) {
        // uidx selects U_ext block for the left/right stencil (0..3) or -1 for rho.
        for (int f = 0; f < nf; ++f) {
            double left, right;
            if (uidx == 0) { left = Ue0[f]; right = Ue0[f + 1]; }
            else if (uidx == 1) { left = Ue1[f]; right = Ue1[f + 1]; }
            else if (uidx == 2) { left = Ue2[f]; right = Ue2[f + 1]; }
            else if (uidx == 3) { left = Ue3[f]; right = Ue3[f + 1]; }
            else { left = Ue0[f] + Ue1[f]; right = Ue0[f + 1] + Ue1[f + 1]; }
            double delta = coeff[f] * delta_alpha[f];
            double lo = dmin(dmin(left, right), base[f]);
            double hi = dmax(dmax(left, right), base[f]);
            if (delta > MU_EPS) theta[f] = dmin(theta[f], (hi - base[f]) / delta);
            else if (delta < -MU_EPS) theta[f] = dmin(theta[f], (lo - base[f]) / delta);
            theta[f] = clip(theta[f], 0.0, 1.0);
        }
    };
    if (alpha_fct) {
        std::vector<double> coeff(nf), base(nf);
        // q1: coeff=rho1_f, base=q1_cons, block 0
        apply_lmp(q1_cons, rho1_f, 0);
        // q2: coeff=-rho2_f, base=q2_cons, block 1
        for (int f = 0; f < nf; ++f) coeff[f] = -rho2_f[f];
        apply_lmp(q2_cons, coeff, 1);
        // rho: coeff=rho1_f-rho2_f, base=q1_cons+q2_cons, block rho(-1)
        for (int f = 0; f < nf; ++f) { coeff[f] = rho1_f[f] - rho2_f[f]; base[f] = q1_cons[f] + q2_cons[f]; }
        apply_lmp(base, coeff, -1);
        // m: coeff=(rho1_f-rho2_f)*u_adv_f, base=m_cons, block 2
        for (int f = 0; f < nf; ++f) coeff[f] = (rho1_f[f] - rho2_f[f]) * u_adv_f[f];
        apply_lmp(m_cons, coeff, 2);
        // rE: coeff=rho1_f*E1_f-rho2_f*E2_f, base=rE_cons, block 3
        for (int f = 0; f < nf; ++f) coeff[f] = rho1_f[f] * E1_f[f] - rho2_f[f] * E2_f[f];
        apply_lmp(rE_cons, coeff, 3);

        // Zalesak cell-update limiter.
        auto anti = [&](const std::vector<double>& coeffv) {
            std::vector<double> af(nf);
            for (int f = 0; f < nf; ++f) af[f] = coeffv[f] * delta_alpha[f] * u_star[f];
            return af;
        };
        std::vector<double> cf(nf);
        // q1
        zalesak_update_limit(theta, Un0, q1_cons, anti(rho1_f), u_star, nullptr, dt, inv_dx, periodic);
        // q2
        for (int f = 0; f < nf; ++f) cf[f] = -rho2_f[f];
        zalesak_update_limit(theta, Un1, q2_cons, anti(cf), u_star, nullptr, dt, inv_dx, periodic);
        // rho
        std::vector<double> Unrho(n), q12(nf);
        for (int i = 0; i < n; ++i) Unrho[i] = Un0[i] + Un1[i];
        for (int f = 0; f < nf; ++f) { cf[f] = rho1_f[f] - rho2_f[f]; q12[f] = q1_cons[f] + q2_cons[f]; }
        zalesak_update_limit(theta, Unrho, q12, anti(cf), u_star, nullptr, dt, inv_dx, periodic);
        // m
        for (int f = 0; f < nf; ++f) cf[f] = (rho1_f[f] - rho2_f[f]) * u_adv_f[f];
        zalesak_update_limit(theta, Un2, m_cons, anti(cf), u_star, nullptr, dt, inv_dx, periodic);
        // rE (with extra pressure-work face flux p_star*u_star)
        std::vector<double> pu(nf);
        for (int f = 0; f < nf; ++f) { cf[f] = rho1_f[f] * E1_f[f] - rho2_f[f] * E2_f[f]; pu[f] = p_star[f] * u_star[f]; }
        zalesak_update_limit(theta, Un3, rE_cons, anti(cf), u_star, &pu, dt, inv_dx, periodic);
    }

    std::vector<double> alpha_f(nf), q1_f(nf), q2_f(nf), m_f(nf), rE_f(nf);
    for (int f = 0; f < nf; ++f) {
        alpha_f[f] = clip(alpha_upwind[f] + theta[f] * delta_alpha[f], 1.0e-12, 1.0 - 1.0e-12);
        double da = alpha_f[f] - alpha_upwind[f];
        q1_f[f] = q1_cons[f] + rho1_f[f] * da;
        q2_f[f] = q2_cons[f] - rho2_f[f] * da;
        m_f[f]  = m_cons[f] + (rho1_f[f] - rho2_f[f]) * u_adv_f[f] * da;
        rE_f[f] = rE_cons[f] + (rho1_f[f] * E1_f[f] - rho2_f[f] * E2_f[f]) * da;
    }
    // preserve scalar-TVD mixture rho/Y at the sharp alpha face (2447-2490).
    if (!mix_rho_f.empty()) {
        for (int f = 0; f < nf; ++f) {
            double al = a_ext[f], ar = a_ext[f + 1];
            bool true_mixture = (dmin(al, ar) > pure_tol_auto) && (dmax(al, ar) < 1.0 - pure_tol_auto);
            bool preserve = !true_mixture;   // mix_preserve_mask is None on this path
            double y1m = clip(mix_y1_f[f], 0.0, 1.0);
            double q1_pres = y1m * mix_rho_f[f];
            double q2_pres = (1.0 - y1m) * mix_rho_f[f];
            if (preserve) { q1_f[f] = q1_pres; q2_f[f] = q2_pres; }
            double rho1_c = dmax(q1_f[f] / dmax(alpha_f[f], 1.0e-12), MU_EPS);
            double rho2_c = dmax(q2_f[f] / dmax(1.0 - alpha_f[f], 1.0e-12), MU_EPS);
            double ke = 0.5 * u_adv_f[f] * u_adv_f[f];
            double E1c = eos1.energy(rho1_c, p_adv_f[f]) + ke;
            double E2c = eos2.energy(rho2_c, p_adv_f[f]) + ke;
            m_f[f] = mix_rho_f[f] * u_adv_f[f];
            rE_f[f] = q1_f[f] * E1c + q2_f[f] * E2c;
        }
    }

    // ── primitive-FCT limiter (2500-2601) ──────────────────────────────────
    bool primitive_fct = !passive_transport(a1, u, p);   // tmlpu != upwind, env default on
    if (primitive_fct) {
        std::vector<double> q1_lo(nf), q2_lo(nf), m_lo(nf), rE_lo(nf);
        for (int f = 0; f < nf; ++f) {
            bool L = upwind_left[f];
            double T1u = L ? T1_ext[f] : T1_ext[f + 1];
            double T2u = L ? T2_ext[f] : T2_ext[f + 1];
            double uu  = L ? u_ext[f] : u_ext[f + 1];
            double pu  = L ? p_ext[f] : p_ext[f + 1];
            double r1 = dmax(eos1.density(pu, T1u), MU_EPS);
            double r2 = dmax(eos2.density(pu, T2u), MU_EPS);
            double e1 = eos1.energy(r1, pu), e2 = eos2.energy(r2, pu);
            double ke = 0.5 * uu * uu;
            double E1 = e1 + ke, E2 = e2 + ke;
            double q1 = alpha_f[f] * r1, q2 = (1.0 - alpha_f[f]) * r2;
            q1_lo[f] = q1; q2_lo[f] = q2;
            m_lo[f] = (q1 + q2) * uu;
            rE_lo[f] = q1 * E1 + q2 * E2;
        }
        std::vector<double> theta_ho(nf, 1.0);
        auto anti_ho = [&](const std::vector<double>& lo_v, const std::vector<double>& hi_v) {
            std::vector<double> af(nf);
            for (int f = 0; f < nf; ++f) af[f] = (hi_v[f] - lo_v[f]) * u_star[f];
            return af;
        };
        zalesak_update_limit(theta_ho, Un0, q1_lo, anti_ho(q1_lo, q1_f), u_star, nullptr, dt, inv_dx, periodic);
        zalesak_update_limit(theta_ho, Un1, q2_lo, anti_ho(q2_lo, q2_f), u_star, nullptr, dt, inv_dx, periodic);
        std::vector<double> Unrho(n), rho_lo(nf), rho_hi(nf);
        for (int i = 0; i < n; ++i) Unrho[i] = Un0[i] + Un1[i];
        for (int f = 0; f < nf; ++f) { rho_lo[f] = q1_lo[f] + q2_lo[f]; rho_hi[f] = q1_f[f] + q2_f[f]; }
        zalesak_update_limit(theta_ho, Unrho, rho_lo, anti_ho(rho_lo, rho_hi), u_star, nullptr, dt, inv_dx, periodic);
        zalesak_update_limit(theta_ho, Un2, m_lo, anti_ho(m_lo, m_f), u_star, nullptr, dt, inv_dx, periodic);
        zalesak_update_limit(theta_ho, Un3, rE_lo, anti_ho(rE_lo, rE_f), u_star, nullptr, dt, inv_dx, periodic);
        for (int f = 0; f < nf; ++f) {
            q1_f[f] = q1_lo[f] + theta_ho[f] * (q1_f[f] - q1_lo[f]);
            q2_f[f] = q2_lo[f] + theta_ho[f] * (q2_f[f] - q2_lo[f]);
            m_f[f]  = m_lo[f]  + theta_ho[f] * (m_f[f]  - m_lo[f]);
            rE_f[f] = rE_lo[f] + theta_ho[f] * (rE_f[f] - rE_lo[f]);
        }
    }

    // reflective-wall zero fluxes.
    if (cfg.bc_l == BC5::Reflective) { q1_f[0] = q2_f[0] = m_f[0] = rE_f[0] = alpha_f[0] = 0.0; }
    if (cfg.bc_r == BC5::Reflective) { q1_f[nf - 1] = q2_f[nf - 1] = m_f[nf - 1] = rE_f[nf - 1] = alpha_f[nf - 1] = 0.0; }

    // ── conservative advective fluxes + Kapila source + update ──────────────
    std::vector<double> F_q1(nf), F_q2(nf), F_m(nf), F_alpha(nf), F_rE(nf), F_pu(nf);
    for (int f = 0; f < nf; ++f) {
        F_q1[f] = q1_f[f] * u_star[f];
        F_q2[f] = q2_f[f] * u_star[f];
        F_m[f]  = m_f[f]  * u_star[f];
        F_alpha[f] = alpha_f[f] * u_star[f];
        F_rE[f] = rE_f[f] * u_star[f];        // allaire advective energy flux
        F_pu[f] = p_star[f] * u_star[f];
    }
    if (cfg.energy_form == EnergyForm::Secant || cfg.energy_form == EnergyForm::Differential) {
        std::vector<double> F_rho(nf);
        for (int f = 0; f < nf; ++f) F_rho[f] = F_q1[f] + F_q2[f];
        FaceEnergy face;
        face.alpha = alpha_f;
        face.p = p_star;
        face.u = u_star;
        face.a_L.resize(nf); face.a_R.resize(nf);
        face.rho1 = rho1_f; face.rho2 = rho2_f;
        face.rho1_L.resize(nf); face.rho1_R.resize(nf);
        face.rho2_L.resize(nf); face.rho2_R.resize(nf);
        face.T1.resize(nf); face.T2.resize(nf); face.e1.resize(nf); face.e2.resize(nf);
        for (int f = 0; f < nf; ++f) {
            face.a_L[f] = a_ext[f]; face.a_R[f] = a_ext[f + 1];
            face.rho1_L[f] = dmax(eos1.density(p_star[f], T1_ext[f]), MU_EPS);
            face.rho1_R[f] = dmax(eos1.density(p_star[f], T1_ext[f + 1]), MU_EPS);
            face.rho2_L[f] = dmax(eos2.density(p_star[f], T2_ext[f]), MU_EPS);
            face.rho2_R[f] = dmax(eos2.density(p_star[f], T2_ext[f + 1]), MU_EPS);
            const bool left = upwind_left[f];
            face.T1[f] = left ? T1_ext[f] : T1_ext[f + 1];
            face.T2[f] = left ? T2_ext[f] : T2_ext[f + 1];
            face.e1[f] = eos1.energy(rho1_f[f], p_star[f]);
            face.e2[f] = eos2.energy(rho2_f[f], p_star[f]);
        }
        total_energy_flux(face, eos1, eos2, F_q1, F_q2, F_alpha,
                          F_rho, cfg.energy_form,
                          std::fmax(cfg.energy_alpha_pure_tol, 0.0), F_rE);
    }

    // Kapila alpha source.  All seven Python kapila_source_mode branches use
    // the same face/cell candidates; the selector below is branch-exact.
    std::vector<double> B_ext(ne), B_f(nf);
    for (int k = 0; k < ne; ++k)
        B_ext[k] = a_ext[k] + D_K_kapila_cell(eos1, eos2, a_ext[k], T1_ext[k], T2_ext[k], p_ext[k]);
    for (int f = 0; f < nf; ++f) {
        double am = 0.5 * (a_ext[f] + a_ext[f + 1]);
        double T1m = 0.5 * (T1_ext[f] + T1_ext[f + 1]);
        double T2m = 0.5 * (T2_ext[f] + T2_ext[f + 1]);
        double pm = 0.5 * (p_ext[f] + p_ext[f + 1]);
        double B_mid = am + D_K_kapila_cell(eos1, eos2, am, T1m, T2m, pm);
        B_f[f] = (B_ext[f] + 4.0 * B_mid + B_ext[f + 1]) / 6.0;
    }
    std::vector<char> mat_cells = pure_material_cells(a1, pure_tol_auto);
    std::vector<double> source_alpha(n);
    for (int i = 0; i < n; ++i) {
        double div_u = (u_star[i + 1] - u_star[i]) * inv_dx;
        double src_face = (B_f[i + 1] * (u_star[i + 1] - u[i])
                         + B_f[i] * (u[i] - u_star[i])) * inv_dx;
        double src_cell = (a1[i] + D_K_kapila_cell(eos1, eos2, a1[i], T1[i], T2[i], p[i])) * div_u;
        if (!cfg.kapila_closure) {
            source_alpha[i] = a1[i] * div_u;
            continue;
        }
        double src_hybrid = mat_cells[i] ? src_cell : src_face;
        // 3-neighbour stencil on a_ext: a_ext[i], a_ext[i+1], a_ext[i+2].
        double a_lo = dmin(dmin(a_ext[i], a_ext[i + 1]), a_ext[i + 2]);
        double a_hi = dmax(dmax(a_ext[i], a_ext[i + 1]), a_ext[i + 2]);
        bool true_mix = (a_lo > pure_tol_auto) && (a_hi < 1.0 - pure_tol_auto);
        double src_trap = .5 * (src_face + src_cell);
        bool immiscible = (a_lo <= pure_tol_auto) && (a_hi >= 1.0 - pure_tol_auto);
        switch (cfg.kapila_source_mode) {
            case KapilaSourceMode::Path:                 source_alpha[i] = src_face; break;
            case KapilaSourceMode::Cell:                 source_alpha[i] = src_cell; break;
            case KapilaSourceMode::Hybrid:               source_alpha[i] = src_hybrid; break;
            case KapilaSourceMode::Trapezoid:            source_alpha[i] = src_trap; break;
            case KapilaSourceMode::ImmiscibleTrapezoid:  source_alpha[i] = immiscible ? src_hybrid : src_trap; break;
            case KapilaSourceMode::MixedTrapezoid:       source_alpha[i] = true_mix ? src_trap : src_hybrid; break;
            case KapilaSourceMode::MixedPath:            source_alpha[i] = true_mix ? src_face : src_hybrid; break;
        }
    }

    MaterialResult out;
    out.q1_new.resize(n); out.q2_new.resize(n); out.m_adv.resize(n);
    out.rhoE_new.resize(n); out.rhoE_adv.resize(n); out.alpha_new.resize(n);
    for (int i = 0; i < n; ++i) {
        double L_q1 = (F_q1[i + 1] - F_q1[i]) * inv_dx;
        double L_q2 = (F_q2[i + 1] - F_q2[i]) * inv_dx;
        double L_m  = (F_m[i + 1] - F_m[i]) * inv_dx;
        double L_rE_adv = (F_rE[i + 1] - F_rE[i]) * inv_dx;
        double L_pu = (F_pu[i + 1] - F_pu[i]) * inv_dx;
        double L_alpha = (F_alpha[i + 1] - F_alpha[i]) * inv_dx - source_alpha[i];
        out.q1_new[i]  = Un0[i] - dt * L_q1;
        out.q2_new[i]  = Un1[i] - dt * L_q2;
        out.m_adv[i]   = Un2[i] - dt * L_m;
        out.rhoE_new[i] = Un3[i] - dt * (L_rE_adv + L_pu);
        out.rhoE_adv[i] = Un3[i] - dt * L_rE_adv;
        out.alpha_new[i] = Un4[i] - dt * L_alpha;
    }
    return out;
}

} // namespace cfd
