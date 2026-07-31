#pragma once

#include <string>
#include <vector>

namespace denner1d {

std::vector<double> apply_ghost(
    const std::vector<double>& a,
    const std::string& left_bc,
    const std::string& right_bc,
    int ghosts,
    bool velocity);

// In-place variant: fill a caller-owned buffer instead of allocating a fresh vector. The
// hot coupled-Newton path calls apply_ghost ~5x per residual eval, millions of times -- a
// reusable scratch buffer removes that allocator traffic. Produces byte-identical contents
// to apply_ghost. `out` is resized to a.size()+2*ghosts (no-op once it has the capacity).
void apply_ghost_into(
    std::vector<double>& out,
    const std::vector<double>& a,
    const std::string& left_bc,
    const std::string& right_bc,
    int ghosts,
    bool velocity);

double minmod(double a, double b);
double van_leer_phi(double r);
double mc_phi(double r);
std::vector<double> reconstruct_faces(
    const std::vector<double>& q,
    const std::vector<double>& face_velocity,
    const std::string& left_bc,
    const std::string& right_bc,
    const std::string& limiter,
    int ghosts = 2);

double bounded_mwi_delta(
    double delta_unbounded,
    double d_hat,
    double dx,
    double u_bar,
    double u_left,
    double u_right,
    double rho_left,
    double rho_right,
    double c_left,
    double c_right);

double stable_dt(const std::vector<double>& u,
                 const std::vector<double>& c,
                 double dx,
                 double cfl);

}  // namespace denner1d
