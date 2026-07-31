#include "denner1d/solver.hpp"

#include "denner1d/acid.hpp"
#include "denner1d/cases.hpp"
#include "denner1d/eos.hpp"
#include "denner1d/numerics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace denner1d {
namespace {

struct AcousticFaces {
    std::vector<double> u;
    std::vector<double> p;
};

struct FaceStates {
    std::vector<double> left;
    std::vector<double> right;
};

struct LocalRiemannState {
    double rho;
    double u;
    double p;
    Phase phase;
};

const Phase& active_phase(double alpha, const Phase& a, const Phase& b) {
    return alpha >= 0.5 ? a : b;
}

double local_nasg_sound_speed(const LocalRiemannState& s, double rho = -1.0, double p = -1.0) {
    const double rr = rho > 0.0 ? rho : s.rho;
    const double pp = p > 0.0 ? p : s.p;
    const double denom = std::max(rr * (1.0 - s.phase.b * rr), 1.0e-300);
    return std::sqrt(std::max(s.phase.gamma * (pp + s.phase.pinf) / denom, 1.0e-300));
}

double local_nasg_shock_volume(double p, const LocalRiemannState& s) {
    const double v0 = 1.0 / std::max(s.rho, 1.0e-300);
    const double A = p + s.phase.gamma * s.phase.pinf;
    const double A0 = s.p + s.phase.gamma * s.phase.pinf;
    const double psum = p + s.p;
    const double coeff = A / (s.phase.gamma - 1.0) + 0.5 * psum;
    const double rhs = A * s.phase.b / (s.phase.gamma - 1.0) +
                       A0 * (v0 - s.phase.b) / (s.phase.gamma - 1.0) +
                       0.5 * psum * v0;
    return std::max(rhs / std::max(coeff, 1.0e-300), s.phase.b + 1.0e-14);
}

double local_nasg_prefun(double p_in, const LocalRiemannState& s) {
    const double p = std::max(p_in, 1.0e-12);
    if (p > s.p) {
        const double v0 = 1.0 / std::max(s.rho, 1.0e-300);
        const double vs = local_nasg_shock_volume(p, s);
        return std::sqrt(std::max((p - s.p) * (v0 - vs), 0.0));
    }
    const double c0 = local_nasg_sound_speed(s);
    const double theta = std::max((p + s.phase.pinf) / std::max(s.p + s.phase.pinf, 1.0e-300), 1.0e-300);
    return 2.0 * c0 * (1.0 - s.phase.b * s.rho) / (s.phase.gamma - 1.0) *
           (std::pow(theta, (s.phase.gamma - 1.0) / (2.0 * s.phase.gamma)) - 1.0);
}

bool local_nasg_star(const LocalRiemannState& left,
                     const LocalRiemannState& right,
                     double& p_star,
                     double& u_star) {
    auto phi = [&](double p) {
        return local_nasg_prefun(p, left) + local_nasg_prefun(p, right) + right.u - left.u;
    };
    double p_low = std::max(1.0e-12, -std::min(left.phase.pinf, right.phase.pinf) + 1.0e-12);
    double p_high = std::max({left.p, right.p, 1.0});
    for (int k = 0; k < 160 && phi(p_high) < 0.0; ++k) p_high = 2.0 * p_high + 1.0;
    if (!std::isfinite(phi(p_high))) return false;
    if (phi(p_low) > 0.0) {
        p_star = p_low;
    } else {
        double lo = p_low;
        double hi = p_high;
        for (int k = 0; k < 120; ++k) {
            const double mid = 0.5 * (lo + hi);
            if (phi(mid) > 0.0) hi = mid;
            else lo = mid;
        }
        p_star = 0.5 * (lo + hi);
    }
    const double fL = local_nasg_prefun(p_star, left);
    const double fR = local_nasg_prefun(p_star, right);
    u_star = 0.5 * (left.u + right.u + fR - fL);
    return std::isfinite(p_star) && std::isfinite(u_star);
}

FaceStates limited_face_states(const std::vector<double>& q,
                               const std::string& left_bc,
                               const std::string& right_bc,
                               bool velocity = false) {
    const int n = static_cast<int>(q.size());
    const int ghosts = 2;
    const auto ext = apply_ghost(q, left_bc, right_bc, ghosts, velocity);
    std::vector<double> slope(n + 2 * ghosts, 0.0);
    for (int i = 1; i + 1 < n + 2 * ghosts; ++i) {
        const double dl = ext[i] - ext[i - 1];
        const double dr = ext[i + 1] - ext[i];
        const double mm = minmod(dl, dr);
        const double mc = minmod(0.5 * (dl + dr), minmod(2.0 * dl, 2.0 * dr));
        const double rough = std::abs(dr - dl) / (std::abs(dr) + std::abs(dl) + 1.0e-300);
        const double base = std::clamp(1.0 - rough, 0.0, 1.0);
        const double smooth = base * base * base * base;
        slope[i] = mm + smooth * (mc - mm);
    }
    FaceStates fs;
    fs.left.assign(n + 1, 0.0);
    fs.right.assign(n + 1, 0.0);
    for (int face = 0; face <= n; ++face) {
        const int iL = ghosts + face - 1;
        const int iR = ghosts + face;
        fs.left[face] = ext[iL] + 0.5 * slope[iL];
        fs.right[face] = ext[iR] - 0.5 * slope[iR];
    }
    return fs;
}

FaceStates mc_face_states(const std::vector<double>& q,
                          const std::string& left_bc,
                          const std::string& right_bc,
                          bool velocity = false) {
    const int n = static_cast<int>(q.size());
    const int ghosts = 2;
    const auto ext = apply_ghost(q, left_bc, right_bc, ghosts, velocity);
    std::vector<double> slope(n + 2 * ghosts, 0.0);
    for (int i = 1; i + 1 < n + 2 * ghosts; ++i) {
        const double dl = ext[i] - ext[i - 1];
        const double dr = ext[i + 1] - ext[i];
        slope[i] = minmod(0.5 * (dl + dr), minmod(2.0 * dl, 2.0 * dr));
    }
    FaceStates fs;
    fs.left.assign(n + 1, 0.0);
    fs.right.assign(n + 1, 0.0);
    for (int face = 0; face <= n; ++face) {
        const int iL = ghosts + face - 1;
        const int iR = ghosts + face;
        fs.left[face] = ext[iL] + 0.5 * slope[iL];
        fs.right[face] = ext[iR] - 0.5 * slope[iR];
    }
    return fs;
}

FaceStates minmod_face_states(const std::vector<double>& q,
                              const std::string& left_bc,
                              const std::string& right_bc,
                              bool velocity = false) {
    const int n = static_cast<int>(q.size());
    const int ghosts = 2;
    const auto ext = apply_ghost(q, left_bc, right_bc, ghosts, velocity);
    std::vector<double> slope(n + 2 * ghosts, 0.0);
    for (int i = 1; i + 1 < n + 2 * ghosts; ++i) {
        slope[i] = minmod(ext[i] - ext[i - 1], ext[i + 1] - ext[i]);
    }
    FaceStates fs;
    fs.left.assign(n + 1, 0.0);
    fs.right.assign(n + 1, 0.0);
    for (int face = 0; face <= n; ++face) {
        const int iL = ghosts + face - 1;
        const int iR = ghosts + face;
        fs.left[face] = ext[iL] + 0.5 * slope[iL];
        fs.right[face] = ext[iR] - 0.5 * slope[iR];
    }
    return fs;
}

// Limited kappa=1/3 MUSCL (3rd-order upwind-biased in smooth regions, TVD-limited).
// Better phase accuracy than the symmetric MC slope -> less acoustic dispersion.
FaceStates kappa13_face_states(const std::vector<double>& q,
                               const std::string& left_bc,
                               const std::string& right_bc,
                               bool velocity = false) {
    const int n = static_cast<int>(q.size());
    const int ghosts = 2;
    const auto ext = apply_ghost(q, left_bc, right_bc, ghosts, velocity);
    const double k = 1.0 / 3.0;
    const double bcomp = (3.0 - k) / (1.0 - k);  // = 4 for k=1/3
    FaceStates fs;
    fs.left.assign(n + 1, 0.0);
    fs.right.assign(n + 1, 0.0);
    for (int face = 0; face <= n; ++face) {
        const int iL = ghosts + face - 1;  // left cell of this face
        const int iR = ghosts + face;      // right cell of this face
        // left state from cell iL: extrapolate to its right face
        {
            const double dm = ext[iL] - ext[iL - 1];
            const double dp = ext[iL + 1] - ext[iL];
            const double sl = minmod(dm, bcomp * dp);
            const double sr = minmod(dp, bcomp * dm);
            fs.left[face] = ext[iL] + 0.25 * ((1.0 - k) * sl + (1.0 + k) * sr);
        }
        // right state from cell iR: extrapolate to its left face
        {
            const double dm = ext[iR] - ext[iR - 1];
            const double dp = ext[iR + 1] - ext[iR];
            const double sl = minmod(dm, bcomp * dp);
            const double sr = minmod(dp, bcomp * dm);
            fs.right[face] = ext[iR] - 0.25 * ((1.0 + k) * sl + (1.0 - k) * sr);
        }
    }
    return fs;
}

FaceStates first_order_face_states(const std::vector<double>& q,
                                   const std::string& left_bc,
                                   const std::string& right_bc,
                                   bool velocity = false) {
    const int n = static_cast<int>(q.size());
    const int ghosts = 1;
    const auto ext = apply_ghost(q, left_bc, right_bc, ghosts, velocity);
    FaceStates fs;
    fs.left.assign(n + 1, 0.0);
    fs.right.assign(n + 1, 0.0);
    for (int face = 0; face <= n; ++face) {
        fs.left[face] = ext[ghosts + face - 1];
        fs.right[face] = ext[ghosts + face];
    }
    return fs;
}

std::vector<double> sound_speed(const PrimitiveState& s, const Phase& a, const Phase& b) {
    const int n = static_cast<int>(s.x.size());
    std::vector<double> c(n);
    for (int i = 0; i < n; ++i) {
        c[i] = mixture_sound_speed(s.p[i], s.T[i], s.alpha[i], a, b);
    }
    for (int i = 0; i < n; ++i) {
        const int im = std::max(0, i - 1);
        const int ip = std::min(n - 1, i + 1);
        const double dp_l = s.p[i] - s.p[im];
        const double dp_r = s.p[ip] - s.p[i];
        const double pressure_strength = (std::abs(dp_l) + std::abs(dp_r)) /
                                         (std::abs(s.p[im]) + std::abs(s.p[i]) +
                                          std::abs(s.p[ip]) + 1.0e-300);
        const double z_l = std::max(s.rho[im] * c[im], 1.0e-300);
        const double z_r = std::max(s.rho[ip] * c[ip], 1.0e-300);
        const double imp = std::abs(z_r - z_l) / (z_r + z_l + 1.0e-300);
        const double acoustic_impedance_weight = std::clamp((1.0 - pressure_strength) * imp, 0.0, 1.0);
        c[i] *= (1.0 + acoustic_impedance_weight);
    }
    return c;
}

double enthalpy_temperature(double p,
                            double alpha,
                            double h_target,
                            const Phase& a,
                            const Phase& b,
                            double guess) {
    double T = std::max(guess, 1.0e-6);
    for (int it = 0; it < 12; ++it) {
        const double h = mixture_enthalpy(p, T, alpha, a, b);
        const double err = h - h_target;
        const double dT = std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(std::abs(T), 1.0);
        const double Tp = T + dT;
        const double Tm = std::max(T - dT, 1.0e-6);
        const double hp = mixture_enthalpy(p, Tp, alpha, a, b);
        const double hm = mixture_enthalpy(p, Tm, alpha, a, b);
        const double dhdT = (hp - hm) / std::max(Tp - Tm, 1.0e-300);
        if (!std::isfinite(dhdT) || std::abs(dhdT) <= 1.0e-300) break;
        const double dT_newton = err / dhdT;
        double next = T - dT_newton;
        if (!std::isfinite(next) || next <= 0.0) next = 0.5 * T;
        T = std::max(next, 1.0e-6);
        if (std::abs(dT_newton) <=
            10.0 * std::numeric_limits<double>::epsilon() * std::max(T, 1.0)) {
            break;
        }
    }
    return T;
}

AcousticFaces acoustic_faces(const PrimitiveState& s,
                             const std::vector<double>& c,
                             const CaseDefinition& def,
                             const SolverConfig& cfg,
                             double dt,
                             double dx,
                             double time) {
    const int n = static_cast<int>(s.x.size());
    AcousticFaces f;
    f.u.assign(n + 1, 0.0);
    f.p.assign(n + 1, 0.0);
    const auto u_lr = limited_face_states(s.u, cfg.left_bc, cfg.right_bc, true);
    const auto p_lr = limited_face_states(s.p, cfg.left_bc, cfg.right_bc, false);
    const auto rho_lr = limited_face_states(s.rho, cfg.left_bc, cfg.right_bc, false);
    const auto c_lr = limited_face_states(c, cfg.left_bc, cfg.right_bc, false);
    const auto u_mc = mc_face_states(s.u, cfg.left_bc, cfg.right_bc, true);
    const auto p_mc = mc_face_states(s.p, cfg.left_bc, cfg.right_bc, false);
    const auto rho_mc = mc_face_states(s.rho, cfg.left_bc, cfg.right_bc, false);
    const auto c_mc = mc_face_states(c, cfg.left_bc, cfg.right_bc, false);
    for (int face = 0; face <= n; ++face) {
        const int l = std::max(0, face - 1);
        const int r = std::min(n - 1, face);
        if (face == 0 && cfg.left_bc == "wall") {
            f.u[face] = 0.0;
            f.p[face] = s.p.front();
            continue;
        }
        if (face == 0 && cfg.left_bc == "inlet" && def.inlet_frequency > 0.0) {
            const Phase& active = def.id == "05" ? def.phase2 : def.phase1;
            const double T0 = ((def.base_pressure + active.pinf) / std::max(def.reference_density, 1.0e-300)
                               - active.b * (def.base_pressure + active.pinf)) /
                              (active.kv * (active.gamma - 1.0) + 1.0e-300);
            const double c0 = phase_props(def.base_pressure, T0, active).c;
            const double dp_amp = def.reference_density * c0 * def.inlet_du;
            const double phase = std::sin(2.0 * M_PI * def.inlet_frequency * time);
            f.u[face] = def.base_velocity + def.inlet_du * phase;
            f.p[face] = def.base_pressure + dp_amp * phase;
            continue;
        }
        if (face == n && cfg.right_bc == "wall") {
            f.u[face] = 0.0;
            f.p[face] = s.p.back();
            continue;
        }
        const double mach_l = std::abs(s.u[l]) / (std::max(c[l], 1.0e-300));
        const double mach_r = std::abs(s.u[r]) / (std::max(c[r], 1.0e-300));
        const double smooth = 1.0 / std::sqrt(1.0 + mach_l * mach_l + mach_r * mach_r);
        const double pressure_strength = std::abs(s.p[r] - s.p[l]) /
                                         (std::abs(s.p[r]) + std::abs(s.p[l]) + 1.0e-300);
        const double z_cell_l = std::max(s.rho[l] * c[l], 1.0e-300);
        const double z_cell_r = std::max(s.rho[r] * c[r], 1.0e-300);
        const double impedance_jump = std::abs(z_cell_r - z_cell_l) / (z_cell_r + z_cell_l + 1.0e-300);
        const double recon_blend = std::clamp((1.0 - pressure_strength) * (1.0 - impedance_jump), 0.0, 1.0);
        const double u_left_state = u_lr.left[face] + recon_blend * (u_mc.left[face] - u_lr.left[face]);
        const double u_right_state = u_lr.right[face] + recon_blend * (u_mc.right[face] - u_lr.right[face]);
        const double p_left_state = p_lr.left[face] + recon_blend * (p_mc.left[face] - p_lr.left[face]);
        const double p_right_state = p_lr.right[face] + recon_blend * (p_mc.right[face] - p_lr.right[face]);
        const double rho_left_state = rho_lr.left[face] + recon_blend * (rho_mc.left[face] - rho_lr.left[face]);
        const double rho_right_state = rho_lr.right[face] + recon_blend * (rho_mc.right[face] - rho_lr.right[face]);
        const double c_left_state = c_lr.left[face] + recon_blend * (c_mc.left[face] - c_lr.left[face]);
        const double c_right_state = c_lr.right[face] + recon_blend * (c_mc.right[face] - c_lr.right[face]);
        const double u_l = s.u[l] + smooth * (u_left_state - s.u[l]);
        const double u_r = s.u[r] + smooth * (u_right_state - s.u[r]);
        const double p_l = s.p[l] + smooth * (p_left_state - s.p[l]);
        const double p_r = s.p[r] + smooth * (p_right_state - s.p[r]);
        const double rho_l = std::max(s.rho[l] + smooth * (rho_left_state - s.rho[l]), 1.0e-300);
        const double rho_r = std::max(s.rho[r] + smooth * (rho_right_state - s.rho[r]), 1.0e-300);
        const double c_l = std::max(c[l] + smooth * (c_left_state - c[l]), 1.0e-300);
        const double c_r = std::max(c[r] + smooth * (c_right_state - c[r]), 1.0e-300);
        const double z_l = rho_l * c_l;
        const double z_r = rho_r * c_r;
        const double denom = z_l + z_r + 1.0e-300;

        double u_star = (z_l * u_l + z_r * u_r + (p_l - p_r)) / denom;
        double p_star = (z_r * p_l + z_l * p_r + z_l * z_r * (u_l - u_r)) / denom;

        const double alpha_l = std::clamp(s.alpha[l], 0.0, 1.0);
        const double alpha_r = std::clamp(s.alpha[r], 0.0, 1.0);
        const bool same_pure_phase =
            (alpha_l > 1.0 - 1.0e-4 && alpha_r > 1.0 - 1.0e-4) ||
            (alpha_l < 1.0e-4 && alpha_r < 1.0e-4);
        const double material_weight = pressure_strength * std::sqrt(impedance_jump);
        const double pure_shock_weight = same_pure_phase ? pressure_strength : 0.0;
        const double nonlinear_weight = std::clamp(std::max(material_weight, pure_shock_weight), 0.0, 1.0);
        if (nonlinear_weight > std::numeric_limits<double>::epsilon()) {
            const LocalRiemannState left{rho_l, u_l, std::max(p_l, 1.0), active_phase(alpha_l, def.phase1, def.phase2)};
            const LocalRiemannState right{rho_r, u_r, std::max(p_r, 1.0), active_phase(alpha_r, def.phase1, def.phase2)};
            double p_nonlinear = p_star;
            double u_nonlinear = u_star;
            if (local_nasg_star(left, right, p_nonlinear, u_nonlinear)) {
                p_star += nonlinear_weight * (p_nonlinear - p_star);
                u_star += nonlinear_weight * (u_nonlinear - u_star);
            }
        }

        const double u_bar = (z_l * u_l + z_r * u_r) / denom;
        const double d_hat = dt / std::sqrt(std::max(rho_l * rho_r, 1.0e-300));
        const double dpdx = (p_r - p_l) / dx;
        const double mwi = bounded_mwi_delta(-d_hat * dpdx, d_hat, dx, u_bar, u_l, u_r,
                                             rho_l, rho_r, c_l, c_r);
        f.u[face] = u_star - nonlinear_weight * mwi;
        f.p[face] = p_star;
    }
    return f;
}

void advance_scalar(std::vector<double>& q,
                    const std::vector<double>& u_face,
                    const SolverConfig& cfg,
                    double dt,
                    double dx) {
    const auto qf = reconstruct_faces(q, u_face, cfg.left_bc, cfg.right_bc, "mc");
    std::vector<double> rhs(q.size(), 0.0);
    for (std::size_t i = 0; i < q.size(); ++i) {
        const double flux_l = u_face[i] * qf[i];
        const double flux_r = u_face[i + 1] * qf[i + 1];
        rhs[i] = -(flux_r - flux_l) / dx;
    }
    for (std::size_t i = 0; i < q.size(); ++i) q[i] += dt * rhs[i];
}

void advance_acoustic(PrimitiveState& s,
                      const Phase& a,
                      const Phase& b,
                      const CaseDefinition& def,
                      const SolverConfig& cfg,
                      double dt,
                      double dx,
                      double time) {
    const int n = static_cast<int>(s.x.size());
    const auto c = sound_speed(s, a, b);
    const auto f = acoustic_faces(s, c, def, cfg, dt, dx, time);
    std::vector<double> p_new = s.p;
    std::vector<double> u_new = s.u;
    std::vector<double> h_new = s.h;
    advance_scalar(h_new, f.u, cfg, dt, dx);
    for (int i = 0; i < n; ++i) {
        const double rho = std::max(s.rho[i], 1.0e-300);
        const double bulk = std::max(rho * c[i] * c[i], 1.0);
        const double dudx = (f.u[i + 1] - f.u[i]) / dx;
        const double dpdx = (f.p[i + 1] - f.p[i]) / dx;
        const double adv_p = -s.u[i] * (i == 0 ? (s.p[std::min(1, n - 1)] - s.p[i]) / dx
                                               : i == n - 1 ? (s.p[i] - s.p[i - 1]) / dx
                                                            : (s.p[i + 1] - s.p[i - 1]) / (2.0 * dx));
        const double adv_u = -s.u[i] * (i == 0 ? (s.u[std::min(1, n - 1)] - s.u[i]) / dx
                                               : i == n - 1 ? (s.u[i] - s.u[i - 1]) / dx
                                                            : (s.u[i + 1] - s.u[i - 1]) / (2.0 * dx));
        const int im = std::max(0, i - 1);
        const int ip = std::min(n - 1, i + 1);
        const double dp_l = s.p[i] - s.p[im];
        const double dp_r = s.p[ip] - s.p[i];
        p_new[i] = s.p[i] + dt * (adv_p - bulk * dudx);
        u_new[i] = s.u[i] + dt * (adv_u - dpdx / rho);
        const double rough_p = std::abs(dp_r - dp_l) /
                               (std::abs(dp_r) + std::abs(dp_l) + 1.0e-300);
        const double z_l = std::max(s.rho[im] * c[im], 1.0e-300);
        const double z_r = std::max(s.rho[ip] * c[ip], 1.0e-300);
        const double imp = std::abs(z_r - z_l) / (z_r + z_l + 1.0e-300);
        const double sensor = std::sqrt(rough_p * imp);
        const double relax = sensor / (1.0 + sensor);
        const double u_face_mean = 0.5 * (f.u[i] + f.u[i + 1]);
        u_new[i] += relax * (u_face_mean - u_new[i]);
        h_new[i] += dt * (-c[i] * c[i] * dudx);
    }
    const std::vector<double> u_relaxed = u_new;
    for (int i = 0; i < n; ++i) {
        const int im = std::max(0, i - 1);
        const int ip = std::min(n - 1, i + 1);
        const double dp_l = s.p[i] - s.p[im];
        const double dp_r = s.p[ip] - s.p[i];
        const double rough_p = std::abs(dp_r - dp_l) /
                               (std::abs(dp_r) + std::abs(dp_l) + 1.0e-300);
        const double pressure_strength = (std::abs(dp_l) + std::abs(dp_r)) /
                                         (std::abs(s.p[im]) + std::abs(s.p[i]) +
                                          std::abs(s.p[ip]) + 1.0e-300);
        const double z_l = std::max(s.rho[im] * c[im], 1.0e-300);
        const double z_r = std::max(s.rho[ip] * c[ip], 1.0e-300);
        const double imp = std::abs(z_r - z_l) / (z_r + z_l + 1.0e-300);
        const double sensor = std::sqrt(rough_p * imp);
        const double filter = pressure_strength * sensor / (1.0 + sensor);
        u_new[i] += filter * (u_relaxed[ip] - 2.0 * u_relaxed[i] + u_relaxed[im]);
    }
    for (int i = 0; i < n; ++i) {
        const int im = std::max(0, i - 1);
        const int ip = std::min(n - 1, i + 1);
        const double dp_l = s.p[i] - s.p[im];
        const double dp_r = s.p[ip] - s.p[i];
        const double rough_p = std::abs(dp_r - dp_l) /
                               (std::abs(dp_r) + std::abs(dp_l) + 1.0e-300);
        const double pressure_strength = (std::abs(dp_l) + std::abs(dp_r)) /
                                         (std::abs(s.p[im]) + std::abs(s.p[i]) +
                                          std::abs(s.p[ip]) + 1.0e-300);
        const double z_l = std::max(s.rho[im] * c[im], 1.0e-300);
        const double z_r = std::max(s.rho[ip] * c[ip], 1.0e-300);
        const double imp = std::abs(z_r - z_l) / (z_r + z_l + 1.0e-300);
        const double bound_weight = std::clamp(pressure_strength * rough_p * (1.0 - imp), 0.0, 1.0);
        const double lo = std::min({s.p[im], s.p[i], s.p[ip]});
        const double hi = std::max({s.p[im], s.p[i], s.p[ip]});
        const double bounded = std::clamp(p_new[i], lo, hi);
        p_new[i] += bound_weight * (bounded - p_new[i]);
    }
    const std::vector<double> p_limited = p_new;
    for (int i = 0; i < n; ++i) {
        const int im = std::max(0, i - 1);
        const int ip = std::min(n - 1, i + 1);
        const double d_l = p_limited[i] - p_limited[im];
        const double d_r = p_limited[ip] - p_limited[i];
        const double curvature = std::abs(d_r - d_l) /
                                 (std::abs(d_r) + std::abs(d_l) + 1.0e-300);
        const double pressure_strength = (std::abs(d_l) + std::abs(d_r)) /
                                         (std::abs(p_limited[im]) + std::abs(p_limited[i]) +
                                          std::abs(p_limited[ip]) + 1.0e-300);
        const double z_l = std::max(s.rho[im] * c[im], 1.0e-300);
        const double z_r = std::max(s.rho[ip] * c[ip], 1.0e-300);
        const double imp = std::abs(z_r - z_l) / (z_r + z_l + 1.0e-300);
        const double median = std::max(std::min(p_limited[im], p_limited[ip]),
                                       std::min(std::max(p_limited[im], p_limited[ip]), p_limited[i]));
        const double median_weight = std::clamp(pressure_strength * curvature * (1.0 - imp), 0.0, 1.0);
        p_new[i] += median_weight * (median - p_new[i]);
    }
    const std::vector<double> u_limited = u_new;
    for (int i = 0; i < n; ++i) {
        const int im = std::max(0, i - 1);
        const int ip = std::min(n - 1, i + 1);
        const double du_l = u_limited[i] - u_limited[im];
        const double du_r = u_limited[ip] - u_limited[i];
        const double u_curvature = std::abs(du_r - du_l) /
                                   (std::abs(du_r) + std::abs(du_l) + 1.0e-300);
        const double dp_l = p_limited[i] - p_limited[im];
        const double dp_r = p_limited[ip] - p_limited[i];
        const double pressure_strength = (std::abs(dp_l) + std::abs(dp_r)) /
                                         (std::abs(p_limited[im]) + std::abs(p_limited[i]) +
                                          std::abs(p_limited[ip]) + 1.0e-300);
        const double z_l = std::max(s.rho[im] * c[im], 1.0e-300);
        const double z_r = std::max(s.rho[ip] * c[ip], 1.0e-300);
        const double imp = std::abs(z_r - z_l) / (z_r + z_l + 1.0e-300);
        const double median = std::max(std::min(u_limited[im], u_limited[ip]),
                                       std::min(std::max(u_limited[im], u_limited[ip]), u_limited[i]));
        const double median_weight = std::clamp(pressure_strength * u_curvature * (1.0 - imp), 0.0, 1.0);
        u_new[i] += median_weight * (median - u_new[i]);
    }
    const std::vector<double> p_filtered = p_new;
    const std::vector<double> u_filtered = u_new;
    for (int i = 0; i < n; ++i) {
        const int im = std::max(0, i - 1);
        const int ip = std::min(n - 1, i + 1);
        const double dp_l = p_filtered[i] - p_filtered[im];
        const double dp_r = p_filtered[ip] - p_filtered[i];
        const double du_l = u_filtered[i] - u_filtered[im];
        const double du_r = u_filtered[ip] - u_filtered[i];
        const double p_curvature = std::abs(dp_r - dp_l) /
                                   (std::abs(dp_r) + std::abs(dp_l) + 1.0e-300);
        const double u_curvature = std::abs(du_r - du_l) /
                                   (std::abs(du_r) + std::abs(du_l) + 1.0e-300);
        const double pressure_strength = (std::abs(dp_l) + std::abs(dp_r)) /
                                         (std::abs(p_filtered[im]) + std::abs(p_filtered[i]) +
                                          std::abs(p_filtered[ip]) + 1.0e-300);
        const double expansion_fraction = (std::max(du_l, 0.0) + std::max(du_r, 0.0)) /
                                          (std::abs(du_l) + std::abs(du_r) + 1.0e-300);
        const double pressure_drop_fraction = (std::max(-dp_l, 0.0) + std::max(-dp_r, 0.0)) /
                                              (std::abs(dp_l) + std::abs(dp_r) + 1.0e-300);
        const double z_l = std::max(s.rho[im] * c[im], 1.0e-300);
        const double z_r = std::max(s.rho[ip] * c[ip], 1.0e-300);
        const double imp = std::abs(z_r - z_l) / (z_r + z_l + 1.0e-300);
        const double smoothness = std::clamp((1.0 - p_curvature) * (1.0 - u_curvature), 0.0, 1.0);
        const double rarefaction_base = std::sqrt(pressure_strength * smoothness) *
                                        expansion_fraction * std::sqrt(pressure_drop_fraction);
        const double p_weight = std::clamp(rarefaction_base * (1.0 - imp), 0.0, 1.0);
        const double u_weight = std::clamp(rarefaction_base * std::sqrt(std::clamp(1.0 - imp, 0.0, 1.0)),
                                           0.0, 1.0);
        const double p_laplacian = p_filtered[ip] - 2.0 * p_filtered[i] + p_filtered[im];
        const double u_laplacian = u_filtered[ip] - 2.0 * u_filtered[i] + u_filtered[im];
        p_new[i] += p_weight * p_laplacian;
        u_new[i] += u_weight * u_laplacian;
    }
    s.p.swap(p_new);
    s.u.swap(u_new);
    advance_scalar(s.alpha, f.u, cfg, dt, dx);
    s.h.swap(h_new);
    for (int i = 0; i < n; ++i) {
        s.T[i] = enthalpy_temperature(s.p[i], s.alpha[i], s.h[i], a, b, s.T[i]);
    }
}

void regularize_density(PrimitiveState& s, const Phase& a, const Phase& b) {
    const int n = static_cast<int>(s.x.size());
    const auto c = sound_speed(s, a, b);
    const std::vector<double> rho_old = s.rho;
    for (int i = 0; i < n; ++i) {
        const int im = std::max(0, i - 1);
        const int ip = std::min(n - 1, i + 1);
        const double dp_l = s.p[i] - s.p[im];
        const double dp_r = s.p[ip] - s.p[i];
        const double p_curvature = std::abs(dp_r - dp_l) /
                                   (std::abs(dp_r) + std::abs(dp_l) + 1.0e-300);
        const double pressure_strength = (std::abs(dp_l) + std::abs(dp_r)) /
                                         (std::abs(s.p[im]) + std::abs(s.p[i]) +
                                          std::abs(s.p[ip]) + 1.0e-300);
        const double median = std::max(std::min(rho_old[im], rho_old[ip]),
                                       std::min(std::max(rho_old[im], rho_old[ip]), rho_old[i]));
        const double alpha_jump = std::abs(s.alpha[ip] - s.alpha[im]);
        const double same_material = std::clamp(1.0 - alpha_jump, 0.0, 1.0);
        const double median_weight = std::clamp(pressure_strength * p_curvature * same_material, 0.0, 1.0);
        s.rho[i] += median_weight * (median - s.rho[i]);
    }
}

// ============================================================================
// Conservative 4-equation operator (PEP/IEC oriented).
// Conserved U = [q1 = alpha*rho_a, q2 = (1-alpha)*rho_b, m = rho u, E = rho e + 0.5 rho u^2].
// Single face state + single mass flux drive composition, momentum, and energy
// (HLLC), so pressure-equilibrium compatibility is built in rather than patched.
// Primitive reconstruction basis W = [p, u, T, alpha] (Collis IEC): density/energy
// are recovered from the EOS at the face, never reconstructed directly.
// ============================================================================

struct ConsFull {
    double p, u, T, alpha, rho, c;
    double q1, q2, m, E;
};

ConsFull cons_full_from_prim(double p, double u, double T, double alpha,
                             const Phase& a, const Phase& b) {
    alpha = std::clamp(alpha, 0.0, 1.0);
    p = std::max(p, 1.0);
    T = std::max(T, 1.0e-6);
    const auto pa = phase_props(p, T, a);
    const auto pb = phase_props(p, T, b);
    ConsFull s;
    s.p = p;
    s.u = u;
    s.T = T;
    s.alpha = alpha;
    s.q1 = alpha * pa.rho;
    s.q2 = (1.0 - alpha) * pb.rho;
    s.rho = std::max(s.q1 + s.q2, 1.0e-300);
    s.c = mixture_sound_speed(p, T, alpha, a, b);
    const double e_int = alpha * pa.e_vol + (1.0 - alpha) * pb.e_vol;  // rho * e
    s.m = s.rho * u;
    s.E = e_int + 0.5 * s.rho * u * u;
    return s;
}

// Recover (p,T,alpha) from conserved (q1,q2, e_int = rho*e) via analytic 2D Newton.
//   F1 = q1/rho_a + q2/rho_b - 1            (volume closure)
//   F2 = q1*h_a + q2*h_b - p - e_int        (internal energy: rho*e = q1 h_a + q2 h_b - p)
bool recover_from_conserved(double q1, double q2, double e_int,
                            const Phase& a, const Phase& b,
                            double& p, double& T, double& alpha) {
    q1 = std::max(q1, 0.0);
    q2 = std::max(q2, 0.0);
    p = std::max(p, 1.0);
    T = std::max(T, 1.0e-6);
    bool ok = false;
    const double e_scale = std::max(std::abs(e_int), 1.0);
    for (int it = 0; it < 100; ++it) {
        const auto pa = phase_props(p, T, a);
        const auto pb = phase_props(p, T, b);
        const double ra = std::max(pa.rho, 1.0e-300);
        const double rb = std::max(pb.rho, 1.0e-300);
        const double F1 = q1 / ra + q2 / rb - 1.0;
        const double F2 = q1 * pa.h + q2 * pb.h - p - e_int;
        // Residual-based convergence (do NOT use the damped step: a shrinking
        // line-search step would otherwise declare false convergence and corrupt
        // conservation at strong-shock cells).
        if (std::abs(F1) <= 1.0e-13 && std::abs(F2) <= 1.0e-12 * e_scale) {
            ok = true;
            break;
        }
        const double dF1dp = -q1 * pa.zeta / (ra * ra) - q2 * pb.zeta / (rb * rb);
        const double dF1dT = -q1 * pa.phi / (ra * ra) - q2 * pb.phi / (rb * rb);
        const double dF2dp = q1 * pa.dh_dp + q2 * pb.dh_dp - 1.0;
        const double dF2dT = q1 * pa.cp + q2 * pb.cp;
        const double det = dF1dp * dF2dT - dF1dT * dF2dp;
        if (!std::isfinite(det) || std::abs(det) < 1.0e-300) break;
        const double dp = (-F1 * dF2dT + dF1dT * F2) / det;
        const double dT = (F1 * dF2dp - F2 * dF1dp) / det;
        // backtracking line search on the residual norm with positivity
        const double f0 = F1 * F1 + (F2 / e_scale) * (F2 / e_scale);
        double damp = 1.0;
        bool moved = false;
        for (int ls = 0; ls < 50; ++ls) {
            const double pt = p + damp * dp;
            const double Tt = T + damp * dT;
            if (std::isfinite(pt) && std::isfinite(Tt) && pt > 0.0 && Tt > 0.0) {
                const auto qa = phase_props(pt, Tt, a);
                const auto qb = phase_props(pt, Tt, b);
                const double rqa = std::max(qa.rho, 1.0e-300);
                const double rqb = std::max(qb.rho, 1.0e-300);
                const double g1 = q1 / rqa + q2 / rqb - 1.0;
                const double g2 = q1 * qa.h + q2 * qb.h - pt - e_int;
                const double f1 = g1 * g1 + (g2 / e_scale) * (g2 / e_scale);
                if (f1 < (1.0 - 1.0e-4 * damp) * f0 || damp < 1.0e-10) {
                    p = pt;
                    T = Tt;
                    moved = true;
                    break;
                }
            }
            damp *= 0.5;
        }
        if (!moved) break;
    }
    const double ra = std::max(phase_props(p, T, a).rho, 1.0e-300);
    alpha = std::clamp(q1 / ra, 0.0, 1.0);
    return ok && std::isfinite(p) && std::isfinite(T);
}

std::array<double, 4> cons_vec(const ConsFull& s) {
    return {s.q1, s.q2, s.m, s.E};
}

std::array<double, 4> cons_phys_flux(const ConsFull& s) {
    return {s.q1 * s.u, s.q2 * s.u, s.m * s.u + s.p, (s.E + s.p) * s.u};
}

std::array<double, 4> hll_flux(const ConsFull& L, const ConsFull& R) {
    const double SL = std::min(L.u - L.c, R.u - R.c);
    const double SR = std::max(L.u + L.c, R.u + R.c);
    if (SL >= 0.0) return cons_phys_flux(L);
    if (SR <= 0.0) return cons_phys_flux(R);
    auto FL = cons_phys_flux(L);
    auto FR = cons_phys_flux(R);
    auto UL = cons_vec(L);
    auto UR = cons_vec(R);
    std::array<double, 4> F{};
    for (int k = 0; k < 4; ++k)
        F[k] = (SR * FL[k] - SL * FR[k] + SL * SR * (UR[k] - UL[k])) / (SR - SL);
    return F;
}

// Two-species HLLC for the mixture EOS.
std::array<double, 4> hllc_flux(const ConsFull& L, const ConsFull& R) {
    const double SL = std::min(L.u - L.c, R.u - R.c);
    const double SR = std::max(L.u + L.c, R.u + R.c);
    if (SL >= 0.0) return cons_phys_flux(L);
    if (SR <= 0.0) return cons_phys_flux(R);
    const double mL = L.rho * (SL - L.u);
    const double mR = R.rho * (SR - R.u);
    const double denom = mL - mR;
    if (!std::isfinite(denom) || std::abs(denom) < 1.0e-300) {
        // degenerate: fall back to HLL average
        auto FL = cons_phys_flux(L);
        auto FR = cons_phys_flux(R);
        auto UL = cons_vec(L);
        auto UR = cons_vec(R);
        std::array<double, 4> F{};
        for (int k = 0; k < 4; ++k)
            F[k] = (SR * FL[k] - SL * FR[k] + SL * SR * (UR[k] - UL[k])) / (SR - SL);
        return F;
    }
    const double Sstar = (R.p - L.p + L.m * (SL - L.u) - R.m * (SR - R.u)) / denom;
    auto star_state = [&](const ConsFull& s, double S) {
        const double fac = s.rho * (S - s.u) / (S - Sstar);
        std::array<double, 4> U;
        U[0] = (s.q1 / s.rho) * fac;
        U[1] = (s.q2 / s.rho) * fac;
        U[2] = fac * Sstar;
        U[3] = fac * (s.E / s.rho +
                      (Sstar - s.u) * (Sstar + s.p / (s.rho * (S - s.u))));
        return U;
    };
    std::array<double, 4> F{};
    if (Sstar >= 0.0) {
        auto FL = cons_phys_flux(L);
        auto UL = cons_vec(L);
        auto US = star_state(L, SL);
        for (int k = 0; k < 4; ++k) F[k] = FL[k] + SL * (US[k] - UL[k]);
    } else {
        auto FR = cons_phys_flux(R);
        auto UR = cons_vec(R);
        auto US = star_state(R, SR);
        for (int k = 0; k < 4; ++k) F[k] = FR[k] + SR * (US[k] - UR[k]);
    }
    return F;
}

double nasg_T_from_p_rho(double p, double rho, const Phase& ph) {
    return ((p + ph.pinf) / std::max(rho, 1.0e-300) - ph.b * (p + ph.pinf)) /
           std::max(ph.kv * (ph.gamma - 1.0), 1.0e-300);
}

// time-dependent inlet primitive state for cases 04/05 (linear isentropic acoustic forcing)
ConsFull inlet_state(const CaseDefinition& def, const Phase& a, const Phase& b, double time) {
    const Phase& active = def.id == "05" ? def.phase2 : def.phase1;
    const double T0 = nasg_T_from_p_rho(def.base_pressure, def.reference_density, active);
    const double c0 = phase_props(def.base_pressure, T0, active).c;
    const double dp_amp = def.reference_density * c0 * def.inlet_du;
    const double ph = std::sin(2.0 * M_PI * def.inlet_frequency * time);
    const double p = def.base_pressure + dp_amp * ph;
    const double u = def.base_velocity + def.inlet_du * ph;
    const double rho = def.reference_density + (dp_amp * ph) / (c0 * c0);
    const double T = nasg_T_from_p_rho(p, rho, active);
    return cons_full_from_prim(p, u, T, def.alpha_value, a, b);
}

void advance_conservative(PrimitiveState& s,
                          const Phase& a,
                          const Phase& b,
                          const CaseDefinition& def,
                          const SolverConfig& cfg,
                          double dt,
                          double dx,
                          double time) {
    const int n = static_cast<int>(s.x.size());
    const bool first_order = std::getenv("DENNER_FIRST_ORDER") != nullptr;
    const bool use_mc = std::getenv("DENNER_MC") != nullptr;
    auto recon = [&](const std::vector<double>& q, bool vel) {
        if (first_order) return first_order_face_states(q, cfg.left_bc, cfg.right_bc, vel);
        if (use_mc) return mc_face_states(q, cfg.left_bc, cfg.right_bc, vel);
        return kappa13_face_states(q, cfg.left_bc, cfg.right_bc, vel);
    };
    const auto pf = recon(s.p, false);
    const auto uf = recon(s.u, true);
    const auto Tf = recon(s.T, false);
    const auto af = recon(s.alpha, false);

    const bool use_hll = std::getenv("DENNER_HLL") != nullptr;
    const bool inlet_left = (cfg.left_bc == "inlet" && def.inlet_frequency > 0.0);
    const ConsFull inlet = inlet_left ? inlet_state(def, a, b, time) : ConsFull{};

    std::vector<std::array<double, 4>> F(n + 1);
    for (int face = 0; face <= n; ++face) {
        ConsFull L = cons_full_from_prim(pf.left[face], uf.left[face], Tf.left[face],
                                         af.left[face], a, b);
        ConsFull R = cons_full_from_prim(pf.right[face], uf.right[face], Tf.right[face],
                                         af.right[face], a, b);
        if (face == 0 && inlet_left) L = inlet;
        F[face] = use_hll ? hll_flux(L, R) : hllc_flux(L, R);
    }

    for (int i = 0; i < n; ++i) {
        const ConsFull cell = cons_full_from_prim(s.p[i], s.u[i], s.T[i], s.alpha[i], a, b);
        const double q1 = cell.q1 - dt / dx * (F[i + 1][0] - F[i][0]);
        const double q2 = cell.q2 - dt / dx * (F[i + 1][1] - F[i][1]);
        const double m = cell.m - dt / dx * (F[i + 1][2] - F[i][2]);
        const double E = cell.E - dt / dx * (F[i + 1][3] - F[i][3]);
        const double rho = std::max(q1 + q2, 1.0e-300);
        const double e_int = E - 0.5 * m * m / rho;
        double p = s.p[i];
        double T = s.T[i];
        double alpha = s.alpha[i];
        recover_from_conserved(q1, q2, e_int, a, b, p, T, alpha);
        s.p[i] = p;
        s.T[i] = T;
        s.alpha[i] = alpha;
        s.u[i] = m / rho;
    }
    refresh_thermo(s, a, b);
}

// ---------------------------------------------------------------------------
// IMEX pressure-based operator (Denner-style, gated by DENNER_PB).
//   explicit: separate-u material advection (no Riemann -> no Abgrall spike)
//   implicit: acoustic pressure-velocity coupling via a tridiagonal Helmholtz
//             solve (provides all-speed stability at the Mach-10 reflected shock)
// ---------------------------------------------------------------------------

// Thomas algorithm: solves A[i] x[i-1] + B[i] x[i] + C[i] x[i+1] = D[i].
std::vector<double> thomas_solve(const std::vector<double>& A,
                                 std::vector<double> B,
                                 const std::vector<double>& C,
                                 std::vector<double> D) {
    const int n = static_cast<int>(B.size());
    for (int i = 1; i < n; ++i) {
        const double w = A[i] / B[i - 1];
        B[i] -= w * C[i - 1];
        D[i] -= w * D[i - 1];
    }
    std::vector<double> x(n);
    x[n - 1] = D[n - 1] / B[n - 1];
    for (int i = n - 2; i >= 0; --i) x[i] = (D[i] - C[i] * x[i + 1]) / B[i];
    return x;
}

void advance_pressure_imex(PrimitiveState& s,
                           const Phase& a,
                           const Phase& b,
                           const CaseDefinition& def,
                           const SolverConfig& cfg,
                           double dt,
                           double dx,
                           double time) {
    const int n = static_cast<int>(s.x.size());
    const bool wallL = (cfg.left_bc == "reflective");
    const bool wallR = (cfg.right_bc == "reflective");

    // --- high-order face reconstruction (velocity reconstructed SEPARATELY) ---
    const bool fo = std::getenv("DENNER_PB_FO") != nullptr;
    const bool mm = std::getenv("DENNER_PB_MINMOD") != nullptr;
    auto rc = [&](const std::vector<double>& q, bool vel) {
        if (fo) return first_order_face_states(q, cfg.left_bc, cfg.right_bc, vel);
        if (mm) return minmod_face_states(q, cfg.left_bc, cfg.right_bc, vel);
        return kappa13_face_states(q, cfg.left_bc, cfg.right_bc, vel);
    };
    const auto uf = rc(s.u, true);
    const auto pf = rc(s.p, false);
    const auto Tf = rc(s.T, false);
    const auto af = rc(s.alpha, false);

    std::vector<ConsFull> cell(n);
    std::vector<double> rho_c(n);
    for (int i = 0; i < n; ++i) {
        cell[i] = cons_full_from_prim(s.p[i], s.u[i], s.T[i], s.alpha[i], a, b);
        rho_c[i] = cell[i].rho;
    }
    // ghost-extended cell fields for the Rhie-Chow advecting velocity
    const auto pe = apply_ghost(s.p, cfg.left_bc, cfg.right_bc, 2, false);
    const auto ue = apply_ghost(s.u, cfg.left_bc, cfg.right_bc, 2, true);
    const auto re = apply_ghost(rho_c, cfg.left_bc, cfg.right_bc, 2, false);

    // --- explicit material advection fluxes (conservative Rusanov, NO pressure) ---
    //   Physical advection flux F_side = u * U (mass q1,q2 / momentum rho*u^2 /
    //   energy E), Rusanov-stabilised: Fadv = 1/2(F_L+F_R) - 1/2|lambda|(U_R-U_L)
    //   with lambda the material wave speed.  This is conservative (correct
    //   post-shock state), captures the velocity shock, and -- crucially -- lets
    //   NO water mass cross the air/water contact (the central+dissipation pair
    //   cancels there), avoiding the Abgrall spike.  Pressure is implicit below.
    (void)pe; (void)ue; (void)re;
    std::vector<std::array<double, 4>> Fadv(n + 1);
    for (int face = 0; face <= n; ++face) {
        const ConsFull L = cons_full_from_prim(pf.left[face], uf.left[face],
                                               Tf.left[face], af.left[face], a, b);
        const ConsFull R = cons_full_from_prim(pf.right[face], uf.right[face],
                                               Tf.right[face], af.right[face], a, b);
        const std::array<double, 4> UL{L.q1, L.q2, L.m, L.E};
        const std::array<double, 4> UR{R.q1, R.q2, R.m, R.E};
        const std::array<double, 4> FL{L.u * L.q1, L.u * L.q2, L.u * L.m, L.u * L.E};
        const std::array<double, 4> FR{R.u * R.q1, R.u * R.q2, R.u * R.m, R.u * R.E};
        const double lam = std::max(std::abs(L.u), std::abs(R.u));
        for (int k = 0; k < 4; ++k)
            Fadv[face][k] = 0.5 * (FL[k] + FR[k]) - 0.5 * lam * (UR[k] - UL[k]);
    }
    if (wallL) Fadv[0] = {0.0, 0.0, 0.0, 0.0};          // no flux through rigid wall
    if (wallR) Fadv[n] = {0.0, 0.0, 0.0, 0.0};

    // --- advected conserved state + thermodynamics ---
    std::vector<double> q1(n), q2(n), m_adv(n), E_adv(n);
    std::vector<double> rho(n), u_adv(n), p_adv(n), T_adv(n), al_adv(n), c2(n);
    for (int i = 0; i < n; ++i) {
        q1[i] = cell[i].q1 - dt / dx * (Fadv[i + 1][0] - Fadv[i][0]);
        q2[i] = cell[i].q2 - dt / dx * (Fadv[i + 1][1] - Fadv[i][1]);
        m_adv[i] = cell[i].m - dt / dx * (Fadv[i + 1][2] - Fadv[i][2]);
        E_adv[i] = cell[i].E - dt / dx * (Fadv[i + 1][3] - Fadv[i][3]);
        rho[i] = std::max(q1[i] + q2[i], 1.0e-300);
        u_adv[i] = m_adv[i] / rho[i];
        const double e_int = E_adv[i] - 0.5 * m_adv[i] * m_adv[i] / rho[i];
        p_adv[i] = s.p[i];
        T_adv[i] = s.T[i];
        al_adv[i] = s.alpha[i];
        recover_from_conserved(q1[i], q2[i], e_int, a, b, p_adv[i], T_adv[i], al_adv[i]);
        const double cc = mixture_sound_speed(p_adv[i], T_adv[i], al_adv[i], a, b);
        c2[i] = cc * cc;
    }

    // --- face densities + advected face velocity (for the acoustic operator) ---
    std::vector<double> rf(n + 1), uadv_f(n + 1);
    for (int face = 0; face <= n; ++face) {
        const double rl = face > 0 ? rho[face - 1] : rho[0];
        const double rr = face < n ? rho[face] : rho[n - 1];
        rf[face] = 0.5 * (rl + rr);
        const double ul = face > 0 ? u_adv[face - 1] : u_adv[0];
        const double ur = face < n ? u_adv[face] : u_adv[n - 1];
        uadv_f[face] = 0.5 * (ul + ur);
    }
    if (wallL) uadv_f[0] = 0.0;
    if (wallR) uadv_f[n] = 0.0;

    // --- implicit acoustic Helmholtz for p^{n+1} (tridiagonal) ---
    //   p_i - dt^2 rho_i c2_i / dx^2 * [ (p_{i+1}-p_i)/rf_{i+1/2}
    //                                  - (p_i-p_{i-1})/rf_{i-1/2} ]
    //       = p_adv_i - dt rho_i c2_i ( uadv_{i+1/2} - uadv_{i-1/2} ) / dx
    //   transmissive & rigid-wall both -> Neumann dp/dx = 0 (drop end coupling).
    std::vector<double> A(n, 0.0), B(n, 0.0), C(n, 0.0), D(n, 0.0);
    for (int i = 0; i < n; ++i) {
        const double k = dt * dt * rho[i] * c2[i] / (dx * dx);
        const double wL = (i > 0) ? k / rf[i] : 0.0;
        const double wR = (i < n - 1) ? k / rf[i + 1] : 0.0;
        A[i] = -wL;
        C[i] = -wR;
        B[i] = 1.0 + wL + wR;
        D[i] = p_adv[i] - dt * rho[i] * c2[i] * (uadv_f[i + 1] - uadv_f[i]) / dx;
    }
    const std::vector<double> p_new = thomas_solve(A, B, C, D);

    // --- corrected face velocity from the new pressure gradient ---
    std::vector<double> uf_new(n + 1);
    for (int face = 0; face <= n; ++face) {
        if (face == 0 || face == n) {
            uf_new[face] = uadv_f[face];
        } else {
            uf_new[face] = uadv_f[face] - (dt / rf[face]) * (p_new[face] - p_new[face - 1]) / dx;
        }
    }
    if (wallL) uf_new[0] = 0.0;
    if (wallR) uf_new[n] = 0.0;

    // --- conservative momentum + energy update with the implicit pressure ---
    const bool diag = std::getenv("DENNER_PB_DIAG") != nullptr;
    for (int i = 0; i < n; ++i) {
        const double pL = (i > 0) ? 0.5 * (p_new[i - 1] + p_new[i]) : p_new[i];
        const double pR = (i < n - 1) ? 0.5 * (p_new[i] + p_new[i + 1]) : p_new[i];
        const double m_new = m_adv[i] - dt / dx * (pR - pL);
        const double E_new = E_adv[i] - dt / dx * (pR * uf_new[i + 1] - pL * uf_new[i]);
        const double e_int = E_new - 0.5 * m_new * m_new / rho[i];
        double p = p_new[i], T = T_adv[i], al = al_adv[i];
        recover_from_conserved(q1[i], q2[i], e_int, a, b, p, T, al);
        if (diag && std::abs(s.x[i] - 0.505) < 0.012) {
            std::fprintf(stderr,
                "DIAG x=%.4f u_adv=%.2f m_adv=%.1f p_adv=%.3e p_new=%.3e p_rec=%.3e "
                "uf=[%.2f,%.2f] m_new=%.1f\n",
                s.x[i], u_adv[i], m_adv[i], p_adv[i], p_new[i], p, uf_new[i],
                uf_new[i + 1], m_new);
        }
        s.p[i] = p;
        s.T[i] = T;
        s.alpha[i] = al;
        s.u[i] = m_new / rho[i];
    }
    refresh_thermo(s, a, b);
}

}  // namespace

void refresh_thermo(PrimitiveState& s, const Phase& a, const Phase& b) {
    for (std::size_t i = 0; i < s.x.size(); ++i) {
        s.alpha[i] = std::clamp(s.alpha[i], 0.0, 1.0);
        s.p[i] = std::max(s.p[i], 1.0);
        s.T[i] = std::max(s.T[i], 1.0e-6);
        s.rho[i] = mixture_density(s.p[i], s.T[i], s.alpha[i], a, b);
        s.h[i] = mixture_enthalpy(s.p[i], s.T[i], s.alpha[i], a, b);
    }
}

PrimitiveState solve_case(const CaseDefinition& c) {
    if (std::getenv("DENNER_ACID") != nullptr) return solve_case_acid(c);
    PrimitiveState s = initial_state(c);
    const int n = static_cast<int>(s.x.size());
    const double dx = (c.config.x1 - c.config.x0) / static_cast<double>(n);
    double t = 0.0;
    int step = 0;
    while (t < c.config.final_time && step < c.config.max_steps) {
        std::vector<double> sound(n);
        for (int i = 0; i < n; ++i) {
            sound[i] = mixture_sound_speed(s.p[i], s.T[i], s.alpha[i], c.phase1, c.phase2);
        }
        double dt = stable_dt(s.u, sound, dx, c.config.cfl);
        dt = std::min(dt, c.config.final_time - t);
        if (!(dt > 0.0)) break;

        const bool pb = std::getenv("DENNER_PB") != nullptr;
        if (pb) {
            // IMEX pressure-based operator with SSP-RK2 (Heun) to cut the
            // advection/acoustic Lie-splitting error from O(dt) to O(dt^2).
            const bool pb_rk1 = std::getenv("DENNER_PB_RK1") != nullptr;
            const PrimitiveState s0 = s;
            PrimitiveState s1 = s0;
            advance_pressure_imex(s1, c.phase1, c.phase2, c, c.config, dt, dx, t);
            if (pb_rk1) {
                s = s1;
            } else {
                PrimitiveState s2 = s1;
                advance_pressure_imex(s2, c.phase1, c.phase2, c, c.config, dt, dx, t + dt);
                for (int i = 0; i < n; ++i) {
                    const ConsFull U0 = cons_full_from_prim(s0.p[i], s0.u[i], s0.T[i],
                                                            s0.alpha[i], c.phase1, c.phase2);
                    const ConsFull U2 = cons_full_from_prim(s2.p[i], s2.u[i], s2.T[i],
                                                            s2.alpha[i], c.phase1, c.phase2);
                    const double q1 = 0.5 * (U0.q1 + U2.q1);
                    const double q2 = 0.5 * (U0.q2 + U2.q2);
                    const double mm2 = 0.5 * (U0.m + U2.m);
                    const double E = 0.5 * (U0.E + U2.E);
                    const double rho = std::max(q1 + q2, 1.0e-300);
                    const double e_int = E - 0.5 * mm2 * mm2 / rho;
                    double p = s2.p[i], T = s2.T[i], al = s2.alpha[i];
                    recover_from_conserved(q1, q2, e_int, c.phase1, c.phase2, p, T, al);
                    s.p[i] = p;
                    s.T[i] = T;
                    s.alpha[i] = al;
                    s.u[i] = mm2 / rho;
                }
                refresh_thermo(s, c.phase1, c.phase2);
            }
            t += dt;
            ++step;
            continue;
        }
        const bool rk1 = std::getenv("DENNER_RK1") != nullptr;
        // SSP-RK2 (Heun) averaged in conserved variables -> conservative & TVD.
        const PrimitiveState s0 = s;
        PrimitiveState s1 = s0;
        advance_conservative(s1, c.phase1, c.phase2, c, c.config, dt, dx, t);
        if (c.config.rk_order >= 2 && !rk1) {
            PrimitiveState s2 = s1;
            advance_conservative(s2, c.phase1, c.phase2, c, c.config, dt, dx, t + dt);
            for (int i = 0; i < n; ++i) {
                const ConsFull U0 =
                    cons_full_from_prim(s0.p[i], s0.u[i], s0.T[i], s0.alpha[i], c.phase1, c.phase2);
                const ConsFull U2 =
                    cons_full_from_prim(s2.p[i], s2.u[i], s2.T[i], s2.alpha[i], c.phase1, c.phase2);
                const double q1 = 0.5 * (U0.q1 + U2.q1);
                const double q2 = 0.5 * (U0.q2 + U2.q2);
                const double m = 0.5 * (U0.m + U2.m);
                const double E = 0.5 * (U0.E + U2.E);
                const double rho = std::max(q1 + q2, 1.0e-300);
                const double e_int = E - 0.5 * m * m / rho;
                double p = s2.p[i];
                double T = s2.T[i];
                double alpha = s2.alpha[i];
                recover_from_conserved(q1, q2, e_int, c.phase1, c.phase2, p, T, alpha);
                s.p[i] = p;
                s.T[i] = T;
                s.alpha[i] = alpha;
                s.u[i] = m / rho;
            }
            refresh_thermo(s, c.phase1, c.phase2);
        } else {
            s = s1;
        }
        t += dt;
        ++step;
    }
    return s;
}

}  // namespace denner1d
