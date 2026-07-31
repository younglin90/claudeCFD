// cfd/five_eq/explicit.hpp -- first-order pressure-based explicit five-equation step.
// Port of five_eq_IMEX/explicit.py::explicit_rusanov_step for the portable
// upwind-alpha branch.  The production IMEX material operator remains in
// material_update.hpp; this header covers main.py time_integrator='explicit'.
#pragma once

#include <cmath>
#include <optional>
#include <vector>

#include "cfd/eos.hpp"
#include "cfd/primitive.hpp"
#include "cfd/five_eq/material_update.hpp"
#include "cfd/five_eq/sound_speed.hpp"
#include "cfd/five_eq/step.hpp"

namespace cfd::five_eq {

inline std::vector<double> explicit_alpha_faces(const std::vector<double>& alpha_ext,
                                                 const std::vector<double>& u_face,
                                                 double dt, double dx,
                                                 const StepConfig& cfg) {
    const int n_face = static_cast<int>(u_face.size());
    std::vector<double> alpha_face(n_face);
    bool clip_high_order = false;
    switch (cfg.explicit_alpha_scheme) {
        case AlphaFaceScheme::Cicsam:
            cicsam_alpha_face(alpha_ext.data(), static_cast<int>(alpha_ext.size()), u_face.data(), n_face, dt, dx, alpha_face.data());
            clip_high_order = true;
            break;
        case AlphaFaceScheme::Stacs:
            stacs_alpha_face(alpha_ext.data(), static_cast<int>(alpha_ext.size()), u_face.data(), n_face, alpha_face.data());
            clip_high_order = true;
            break;
        case AlphaFaceScheme::Mstacs:
            mstacs_alpha_face(alpha_ext.data(), static_cast<int>(alpha_ext.size()), u_face.data(), n_face, dt, dx, alpha_face.data());
            break;
        case AlphaFaceScheme::VanLeer:
            vanleer_alpha_face(alpha_ext.data(), static_cast<int>(alpha_ext.size()), u_face.data(), n_face, alpha_face.data());
            clip_high_order = true;
            break;
        case AlphaFaceScheme::AdaptiveBvd:
            adaptive_bvd_alpha_face(alpha_ext.data(), static_cast<int>(alpha_ext.size()), u_face.data(), n_face, dt, dx,
                                    cfg.explicit_alpha_limiter, cfg.alpha_pure_tol, alpha_face.data());
            clip_high_order = true;
            break;
        case AlphaFaceScheme::Thinc:
            thinc_alpha_face(alpha_ext.data(), static_cast<int>(alpha_ext.size()), u_face.data(), n_face, alpha_face.data());
            break;
        case AlphaFaceScheme::ThincBvd:
            thinc_bvd_alpha_face(alpha_ext.data(), static_cast<int>(alpha_ext.size()), u_face.data(), n_face, dt, dx,
                                 cfg.explicit_alpha_limiter, alpha_face.data());
            break;
        case AlphaFaceScheme::Upwind:
        case AlphaFaceScheme::Muscl:
        case AlphaFaceScheme::Central:
        default:
            for (int f = 0; f < n_face; ++f)
                alpha_face[f] = u_face[f] >= 0.0 ? alpha_ext[f] : alpha_ext[f + 1];
            break;
    }
    if (clip_high_order) for (double& value : alpha_face)
        value = std::clamp(value, 1.e-12, 1.0 - 1.e-12);
    return alpha_face;
}

inline StepResult explicit_rusanov_step(const std::vector<double>& a,
                                        const std::vector<double>& T1,
                                        const std::vector<double>& T2,
                                        const std::vector<double>& u,
                                        const std::vector<double>& p,
                                        double dt, double dx,
                                        const EOS& eos1, const EOS& eos2,
                                        const StepConfig& cfg) {
    const int n = static_cast<int>(a.size());
    constexpr double eps = 1.e-30;
    std::vector<double> al = a, tl = T1, tr = T2, ul = u, pl = p;
    std::optional<double> u_left = cfg.u_inlet_l, p_left = cfg.p_inlet_l;
    if (cfg.bc_l == BC5::InletAcoustic) {
        const PhaseAcoustic pa = phase_acoustic(eos1, eos2, a[0], T1[0], T2[0], p[0], cfg.alpha_pure_tol,
                                                 cfg.mixture_sound_speed_kind);
        const double Z = std::fmax(pa.Z, eps);
        const double ui = cfg.u_inlet_l ? *cfg.u_inlet_l : u[0];
        const double pi = cfg.p_inlet_l ? *cfg.p_inlet_l : p[0];
        const double jp = (ui - u[0]) + (pi - p[0]) / Z;
        const double jm = n >= 2 ? (u[1] - u[0]) - (p[1] - p[0]) / Z : 0.0;
        u_left = u[0] + .5 * (jp + jm);
        p_left = p[0] + .5 * Z * (jp - jm);
    }
    const auto ae = cfd::detail::extend1(al, cfg.bc_l, cfg.bc_r, false, cfg.alpha_inlet_l);
    const auto t1e = cfd::detail::extend1(tl, cfg.bc_l, cfg.bc_r, false, cfg.T1_inlet_l);
    const auto t2e = cfd::detail::extend1(tr, cfg.bc_l, cfg.bc_r, false, cfg.T2_inlet_l);
    const auto ue = cfd::detail::extend1(ul, cfg.bc_l, cfg.bc_r, true, u_left);
    const auto pe = cfd::detail::extend1(pl, cfg.bc_l, cfg.bc_r, false, p_left, cfg.p_outlet_r);

    std::vector<ConsU> Ue(n + 2), Un(n);
    std::vector<double> Z(n + 2);
    for (int k = 0; k < n + 2; ++k) {
        Ue[k] = prim_to_cons_W(PrimW{ae[k], t1e[k], t2e[k], ue[k], pe[k]}, eos1, eos2);
        Z[k] = phase_acoustic(eos1, eos2, ae[k], t1e[k], t2e[k], pe[k], cfg.alpha_pure_tol,
                              cfg.mixture_sound_speed_kind).Z;
    }
    for (int i = 0; i < n; ++i) Un[i] = prim_to_cons_W(PrimW{a[i], T1[i], T2[i], u[i], p[i]}, eos1, eos2);

    std::vector<double> f1(n + 1), f2(n + 1), fm(n + 1), fe(n + 1), fa(n + 1), uf(n + 1), ps(n + 1);
    for (int f = 0; f <= n; ++f) {
        const double den = std::fmax(Z[f] + Z[f + 1], eps);
        ps[f] = (Z[f + 1] * pe[f] + Z[f] * pe[f + 1] + Z[f] * Z[f + 1] * (ue[f] - ue[f + 1])) / den;
        uf[f] = (pe[f] - pe[f + 1] + Z[f] * ue[f] + Z[f + 1] * ue[f + 1]) / den;
    }
    const std::vector<double> alpha_face = explicit_alpha_faces(ae, uf, dt, dx, cfg);
    for (int f = 0; f <= n; ++f) {
        const bool left = uf[f] >= 0.0;
        const ConsU& up = Ue[left ? f : f + 1];
        f1[f] = up.m1 * uf[f]; f2[f] = up.m2 * uf[f];
        fm[f] = up.mom * uf[f] + ps[f]; fe[f] = up.rhoE * uf[f] + ps[f] * uf[f];
        fa[f] = alpha_face[f] * uf[f];
    }
    if (cfg.bc_l == BC5::Reflective) { f1[0] = f2[0] = fe[0] = fa[0] = uf[0] = 0.0; fm[0] = p[0]; }
    if (cfg.bc_r == BC5::Reflective) { f1[n] = f2[n] = fe[n] = fa[n] = uf[n] = 0.0; fm[n] = p[n - 1]; }

    StepResult out;
    out.alpha.resize(n); out.T1.resize(n); out.T2.resize(n); out.u.resize(n); out.p.resize(n);
    for (int i = 0; i < n; ++i) {
        const double div1 = (f1[i + 1] - f1[i]) / dx;
        const double div2 = (f2[i + 1] - f2[i]) / dx;
        const double divm = (fm[i + 1] - fm[i]) / dx;
        const double diva = (fa[i + 1] - fa[i]) / dx;
        const double divu = (uf[i + 1] - uf[i]) / dx;
        double B = a[i];
        if (cfg.kapila_closure) B += cfd::detail::D_K_kapila_cell(eos1, eos2, a[i], T1[i], T2[i], p[i]);
        const PhaseAcoustic pa = phase_acoustic(eos1, eos2, a[i], T1[i], T2[i], p[i], cfg.alpha_pure_tol,
                                                 cfg.mixture_sound_speed_kind);
        const double dpb = (pe[i + 1] - pe[i]) / dx;
        const double dpf = (pe[i + 2] - pe[i + 1]) / dx;
        const double Lp = u[i] * (u[i] >= 0.0 ? dpb : dpf) + pa.rho * pa.c_mix_sq * divu;
        const double q1 = Un[i].m1 - dt * div1;
        const double q2 = Un[i].m2 - dt * div2;
        const double mom = Un[i].mom - dt * divm;
        const double anew = Un[i].a1 - dt * (diva - B * divu);
        const double rho = std::fmax(q1 + q2, eps);
        const double pnew = p[i] - dt * Lp;
        const double r1 = q1 / std::fmax(anew, 1.e-12);
        const double r2 = q2 / std::fmax(1.0 - anew, 1.e-12);
        out.alpha[i] = anew; out.u[i] = mom / rho; out.p[i] = pnew;
        out.T1[i] = eos1.temperature(r1, eos1.energy(r1, pnew));
        out.T2[i] = eos2.temperature(r2, eos2.energy(r2, pnew));
    }
    return out;
}

} // namespace cfd::five_eq
