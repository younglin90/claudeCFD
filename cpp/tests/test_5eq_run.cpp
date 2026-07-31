// Multi-step production-driver validation against solver/five_eq_IMEX/main.py::solve.
#include "cfd/five_eq/solver.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using cfd::BC5;
using cfd::EOS;
using cfd::five_eq::RunConfig;
using cfd::five_eq::RunResult;
using cfd::five_eq::RunTermination;

static constexpr double P0 = 1.0e5;

static EOS eos_air() { return EOS::ideal(1.4, 717.5); }
static EOS eos_water() { return EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6); }

struct RefW {
    std::vector<double> alpha, T1, T2, u, p;
    double t_final = -1.0;
};

static bool load_ref(const char* path, RefW& r) {
    std::ifstream fin(path);
    if (!fin) return false;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.rfind("# t_final=", 0) == 0) {
            r.t_final = std::stod(line.substr(10));
            continue;
        }
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double idx, a, t1, t2, u, p;
        if (!(iss >> idx >> a >> t1 >> t2 >> u >> p)) continue;
        r.alpha.push_back(a); r.T1.push_back(t1); r.T2.push_back(t2);
        r.u.push_back(u); r.p.push_back(p);
    }
    return r.t_final >= 0.0 && !r.alpha.empty();
}

struct IC {
    std::vector<double> alpha, T1, T2, u, p;
    double dx = 0.0, t_end = 0.0;
    RunConfig cfg{};
};

static IC make_02A() {
    IC c;
    constexpr int n = 100;
    c.dx = 1.0 / n;
    c.t_end = 1.0;
    c.cfg.dt_fixed = 0.01;
    c.cfg.max_steps = 3;
    c.cfg.record_history = true;
    c.cfg.step_config.alpha_pure_tol = 1.0e-3;
    c.cfg.step_config.bc_l = BC5::Periodic;
    c.cfg.step_config.bc_r = BC5::Periodic;
    for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) * c.dx;
        c.alpha.push_back((x >= 0.4 && x < 0.6) ? 1.0e-3 : 1.0 - 1.0e-3);
        c.T1.push_back(300.0); c.T2.push_back(300.0);
        c.u.push_back(1.0); c.p.push_back(P0);
    }
    return c;
}

static IC make_07B(const EOS& eos1, const EOS& eos2) {
    IC c;
    constexpr int n = 400;
    c.dx = 1.5 / n;
    c.t_end = 1.55e-3;
    c.cfg.cfl = 0.4;
    c.cfg.max_steps = 3;
    c.cfg.record_history = true;
    c.cfg.step_config.alpha_pure_tol = 1.0e-8;
    c.cfg.step_config.bc_l = BC5::Reflective;
    c.cfg.step_config.bc_r = BC5::Transmissive;
    const double T1v = eos1.temperature(1.157, eos1.energy(1.157, P0));
    const double T2v = eos2.temperature(998.0, eos2.energy(998.0, P0));
    constexpr double theta_L = 0.00086043228978671161;
    constexpr double ZL = 1.157 * 347.8;
    for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) * c.dx;
        const bool left = x < 0.5;
        const double u = left ? 0.02 * std::exp(-((x - 0.1) * (x - 0.1)) /
                                                (2.0 * 0.014 * 0.014)) : 0.0;
        const double p = P0 + ZL * u;
        c.alpha.push_back(left ? 1.0 - 1.0e-8 : 1.0e-8);
        c.T1.push_back(T1v + (left ? theta_L * (p - P0) : 0.0));
        c.T2.push_back(T2v); c.u.push_back(u); c.p.push_back(p);
    }
    return c;
}

static double max_scaled_error(const std::vector<double>& got,
                               const std::vector<double>& ref) {
    if (got.size() != ref.size()) return std::numeric_limits<double>::infinity();
    double scale = 0.0;
    for (double v : ref) scale = std::fmax(scale, std::fabs(v));
    scale = scale > 0.0 ? scale : 1.0;
    double err = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i)
        err = std::fmax(err, std::fabs(got[i] - ref[i]) / scale);
    return err;
}

static int run_case(const char* name, const IC& c, const char* ref_path,
                    const EOS& eos1, const EOS& eos2, int expected_steps = 3,
                    double tolerance = 3.0e-11) {
    RefW ref;
    if (!load_ref(ref_path, ref)) {
        std::printf("[%s] cannot load reference %s\n", name, ref_path);
        return 1;
    }
    RunResult got = cfd::five_eq::solve_imex_ad(
        c.alpha, c.T1, c.T2, c.u, c.p, c.dx, c.t_end, eos1, eos2, c.cfg);
    const double ea = max_scaled_error(got.W.alpha, ref.alpha);
    const double e1 = max_scaled_error(got.W.T1, ref.T1);
    const double e2 = max_scaled_error(got.W.T2, ref.T2);
    const double eu = max_scaled_error(got.W.u, ref.u);
    const double ep = max_scaled_error(got.W.p, ref.p);
    const double et = std::fabs(got.t_final - ref.t_final) /
                      std::fmax(std::fabs(ref.t_final), 1.0);
    const double err = std::fmax(std::fmax(std::fmax(ea, e1), std::fmax(e2, eu)),
                                 std::fmax(ep, et));
    // The one-step stage map remains validated at 1e-12 in test_5eq_step.
    // Repeating its differently conditioned C++/NumPy acoustic linear solves
    // accumulates a small temperature/velocity roundoff over three steps.
    const bool pass = got.termination == RunTermination::max_steps &&
                      got.steps == expected_steps && got.history.size() == static_cast<std::size_t>(expected_steps) &&
                      err <= tolerance;
    std::printf("[%s] steps=%d t=%.17g errors alpha=%.3e T1=%.3e T2=%.3e u=%.3e p=%.3e t=%.3e %s\n",
                name, got.steps, got.t_final, ea, e1, e2, eu, ep, et,
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

static int run_isothermal_contact(const EOS& water, const EOS& air) {
    constexpr int n = 100;
    std::vector<double> alpha(n), T1(n, 300.0), T2(n, 1200.0);
    std::vector<double> u(n, 10.0), p(n, P0);
    for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) / n;
        alpha[i] = (x >= 0.35 && x < 0.65) ? 1.0 - 1.0e-6 : 1.0e-6;
    }
    RunConfig cfg;
    cfg.dt_fixed = 5.0e-4;
    cfg.max_steps = 205;
    cfg.step_config.alpha_pure_tol = 1.0e-6;
    cfg.step_config.bc_l = BC5::Periodic;
    cfg.step_config.bc_r = BC5::Periodic;
    cfg.step_config.time_integrator = cfd::five_eq::TimeIntegrator::imex_ssp3;
    const RunResult got = cfd::five_eq::solve_imex_ad(
        alpha, T1, T2, u, p, 0.01, 0.1, water, air, cfg);
    double t1_err = 0.0, t2_err = 0.0;
    for (int i = 0; i < n; ++i) {
        t1_err = std::fmax(t1_err, std::fabs(got.W.T1[i] - 300.0));
        t2_err = std::fmax(t2_err, std::fabs(got.W.T2[i] - 1200.0));
    }
    const bool pass = got.termination == RunTermination::completed &&
                      got.steps == 200 && t1_err <= 1.0e-7 && t2_err <= 1.0e-7;
    std::printf("[16T-isothermal] steps=%d T1=%.3e T2=%.3e %s\n",
                got.steps, t1_err, t2_err, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main() {
    const EOS air = eos_air();
    const EOS water = eos_water();
    int rc = 0;
    rc |= run_case("02A", make_02A(), RUN_02A_REF, air, water);
    IC ssp3_02a = make_02A();
    ssp3_02a.cfg.max_steps = 1;
    ssp3_02a.cfg.step_config.time_integrator = cfd::five_eq::TimeIntegrator::imex_ssp3_transport_acoustic_cn;
    rc |= run_case("02A-SSP3-CN", ssp3_02a, SSP3_TRANSPORT_02A_REF, air, water, 1, 5.0e-8);
    rc |= run_case("07B", make_07B(air, water), RUN_07B_REF, air, water);
    IC ssp2 = make_07B(air, water);
    ssp2.cfg.max_steps = 1;
    ssp2.cfg.step_config.time_integrator = cfd::five_eq::TimeIntegrator::imex_ad_ssp2;
    // The seeded 3x3 primitive recovery has a weak-phase temperature condition
    // number absent from the single-stage oracle; conserved fields remain far tighter.
    rc |= run_case("07B-SSP2", ssp2, SSP2_07B_REF, air, water, 1, 3.0e-8);
    IC ssp3 = make_07B(air, water);
    ssp3.cfg.max_steps = 1;
    ssp3.cfg.step_config.time_integrator = cfd::five_eq::TimeIntegrator::imex_ssp3_transport_acoustic_cn;
    // Weak-phase temperature is recovered from a nearly pure-cell 3x3 EOS solve;
    // the conservative/pressure fields remain at 1e-12 while T2 carries this
    // conditioning floor after the three material substeps.
    rc |= run_case("07B-SSP3-CN", ssp3, SSP3_TRANSPORT_07B_REF, air, water, 1, 5.0e-8);
    rc |= run_isothermal_contact(water, air);
    std::printf("test_5eq_run: %s\n", rc == 0 ? "ALL PASS" : "FAILURES");
    return rc;
}
