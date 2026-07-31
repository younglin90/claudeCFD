#include "denner1d/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace denner1d {
namespace {
constexpr double eps = 1.0e-300;
}

void apply_ghost_into(std::vector<double>& out,
                      const std::vector<double>& a,
                      const std::string& left_bc,
                      const std::string& right_bc,
                      int ghosts,
                      bool velocity) {
    const int n = static_cast<int>(a.size());
    const int m = n + 2 * ghosts;
    if (static_cast<int>(out.size()) != m) out.resize(m);
    for (int i = 0; i < n; ++i) out[ghosts + i] = a[i];
    for (int g = 0; g < ghosts; ++g) {
        if (left_bc == "periodic") out[ghosts - 1 - g] = a[n - 1 - g];
        else if (left_bc == "wall" || left_bc == "reflective") out[ghosts - 1 - g] = velocity ? -a[g] : a[g];
        else if (left_bc == "transmissive" || left_bc == "inlet") out[ghosts - 1 - g] = a.front();
        else throw std::runtime_error("unknown left boundary: " + left_bc);

        if (right_bc == "periodic") out[ghosts + n + g] = a[g];
        else if (right_bc == "wall" || right_bc == "reflective") out[ghosts + n + g] = velocity ? -a[n - 1 - g] : a[n - 1 - g];
        else if (right_bc == "transmissive" || right_bc == "inlet") out[ghosts + n + g] = a.back();
        else throw std::runtime_error("unknown right boundary: " + right_bc);
    }
}

std::vector<double> apply_ghost(const std::vector<double>& a,
                                const std::string& left_bc,
                                const std::string& right_bc,
                                int ghosts,
                                bool velocity) {
    std::vector<double> out;
    apply_ghost_into(out, a, left_bc, right_bc, ghosts, velocity);
    return out;
}

double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return std::copysign(std::min(std::abs(a), std::abs(b)), a);
}

double van_leer_phi(double r) {
    return (r + std::abs(r)) / (1.0 + std::abs(r) + eps);
}

double mc_phi(double r) {
    return std::max(0.0, std::min({2.0 * r, 0.5 * (1.0 + r), 2.0}));
}

std::vector<double> reconstruct_faces(const std::vector<double>& q,
                                      const std::vector<double>& face_velocity,
                                      const std::string& left_bc,
                                      const std::string& right_bc,
                                      const std::string& limiter,
                                      int ghosts) {
    const int n = static_cast<int>(q.size());
    auto ext = apply_ghost(q, left_bc, right_bc, ghosts, false);
    std::vector<double> f(n + 1);
    for (int face = 0; face <= n; ++face) {
        const int iL = ghosts + face - 1;
        const int iR = ghosts + face;
        const bool pos = face_velocity[face] >= 0.0;
        const double c = pos ? ext[iL] : ext[iR];
        const double d = pos ? ext[iL - 1] : ext[iR + 1];
        const double a = pos ? ext[iR] : ext[iL];
        const double diff = c - d;
        const double scale = std::max({std::abs(c), std::abs(d), std::abs(a), 1.0e-30});
        if (std::abs(diff) < 1.0e-12 * scale) {
            f[face] = c;
            continue;
        }
        const double r = (a - c) / diff;
        double phi = 0.0;
        if (limiter == "vanleer") phi = van_leer_phi(r);
        else if (limiter == "mc") phi = mc_phi(r);
        else phi = std::max(0.0, std::min(1.0, r));
        f[face] = c + 0.5 * phi * diff;
    }
    return f;
}

double bounded_mwi_delta(double delta_unbounded,
                         double d_hat,
                         double dx,
                         double u_bar,
                         double u_left,
                         double u_right,
                         double rho_left,
                         double rho_right,
                         double c_left,
                         double c_right) {
    const double u_jump = std::abs(u_right - u_left);
    const double p_delta_eff = (d_hat != 0.0 && std::isfinite(d_hat))
        ? std::abs(delta_unbounded) * std::max(dx, eps) / std::abs(d_hat)
        : 0.0;
    const double z_sum = std::abs(rho_left * c_left) + std::abs(rho_right * c_right);
    const double u_ac = p_delta_eff / (z_sum + eps);
    const double u_ref = std::sqrt(u_bar * u_bar + u_jump * u_jump + u_ac * u_ac) + eps;
    const double r = std::abs(delta_unbounded) / u_ref;
    const double phi = std::pow(1.0 + std::pow(r, 4.0), -0.25);
    return delta_unbounded * phi;
}

double stable_dt(const std::vector<double>& u,
                 const std::vector<double>& c,
                 double dx,
                 double cfl) {
    double lam = 1.0e-300;
    for (std::size_t i = 0; i < u.size(); ++i) {
        lam = std::max(lam, std::abs(u[i]) + c[i]);
    }
    return cfl * dx / lam;
}

}  // namespace denner1d
