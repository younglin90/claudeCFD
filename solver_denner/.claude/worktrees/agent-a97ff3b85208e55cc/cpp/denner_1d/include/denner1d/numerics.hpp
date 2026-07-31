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
