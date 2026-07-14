// test_5eq_step.cpp — end-to-end validation of cfd/five_eq/step.hpp (M10) against
// the Python oracle step refs tests/5eq_ref/step_02A_ref.txt / step_07B_ref.txt
// (gen_5eq_oracle.py::gen_step_02A / gen_step_07B).  Each ref is the full
// primitive state W=(alpha1,T1,T2,u,p) after ONE production imex_ad_step from the
// exact 02_A / 07_B initial condition (see .codex-loop/verify_02_07_acceptance.py).
//
// The refs carry only the final W, so the ICs are reconstructed here from the
// documented setup; the one-step dt is the dt_used documented in each ref header.
//
// Criterion (mirrors test_5eq_acoustic_solve): FIELD-SCALED max error
//   |got - ref| / max_j|ref_j|  <=  1e-12   per field.
// The implicit acoustic u-solve is badly scaled (u~1, p~1e5, beta~1e9), so the
// recovered u carries that linear-solve conditioning as an absolute floor; p and
// the EOS-recovered T/alpha stay tight.  Strict per-cell rel is also reported for
// the well-determined cells (|ref| >= 1e-6 * field scale).
#include "cfd/five_eq/step.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::EOS;
using cfd::BC5;
using cfd::five_eq::StepConfig;
using cfd::five_eq::StepResult;
using cfd::five_eq::PressureClosure;

static const double P0 = 1.0e5;

static EOS eos_air()   { return EOS::ideal(1.4, 717.5); }
static EOS eos_water() { return EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6); }

struct RefW {
    std::vector<double> alpha, T1, T2, u, p;
};

static bool load_ref(const char* path, RefW& r) {
    std::ifstream fin(path);
    if (!fin) { std::printf("cannot open ref %s\n", path); return false; }
    std::string line;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double idx, a, t1, t2, u, p;
        if (!(iss >> idx >> a >> t1 >> t2 >> u >> p)) continue;
        r.alpha.push_back(a); r.T1.push_back(t1); r.T2.push_back(t2);
        r.u.push_back(u); r.p.push_back(p);
    }
    return !r.alpha.empty();
}

// ── initial conditions (exact, from verify_02_07_acceptance.py) ──────────────
struct IC { std::vector<double> alpha, T1, T2, u, p; double dt, dx; StepConfig cfg; };

static IC make_02A() {
    IC c;
    int n = 100; c.dx = 1.0 / n; c.dt = 0.01;   // dt_fixed=0.01
    c.cfg.alpha_pure_tol = 1.0e-3;
    c.cfg.bc_l = BC5::Periodic; c.cfg.bc_r = BC5::Periodic;
    double af = 1.0e-3;
    for (int i = 0; i < n; ++i) {
        double x = (i + 0.5) * c.dx;
        c.alpha.push_back((x >= 0.4 && x < 0.6) ? af : (1.0 - af));
        c.T1.push_back(300.0); c.T2.push_back(300.0);
        c.u.push_back(1.0); c.p.push_back(P0);
    }
    return c;
}

static IC make_07B(const EOS& eos1, const EOS& eos2) {
    IC c;
    int n = 400; double L = 1.5; c.dx = L / n;
    c.dt = 9.5712228747485977e-07;                 // dt_used from ref header
    c.cfg.alpha_pure_tol = 1.0e-8;
    c.cfg.bc_l = BC5::Reflective; c.cfg.bc_r = BC5::Transmissive;
    double af = 1.0e-8, x_intf = 0.5, x_src = 0.1, sigma = 0.014, UP = 0.02;
    double ZL = 1.157 * 347.8;
    double T1v = eos1.temperature(1.157, eos1.energy(1.157, P0));
    double T2v = eos2.temperature(998.0, eos2.energy(998.0, P0));
    double theta_L = 0.00086043228978671161;       // theta_L from ref header
    for (int i = 0; i < n; ++i) {
        double x = (i + 0.5) * c.dx;
        bool mL = x < x_intf;
        c.alpha.push_back(mL ? (1.0 - af) : af);
        double u = mL ? UP * std::exp(-((x - x_src) * (x - x_src)) / (2.0 * sigma * sigma)) : 0.0;
        double p = P0 + ZL * u;
        c.u.push_back(u); c.p.push_back(p);
        c.T1.push_back(T1v + (mL ? theta_L * (p - P0) : 0.0));
        c.T2.push_back(T2v);
    }
    return c;
}

// ── field-scaled comparison ──────────────────────────────────────────────────
struct FieldStat { double scaled_max = 0.0; double strict_big = 0.0; int worst = -1; };

static FieldStat compare(const std::vector<double>& got, const std::vector<double>& ref,
                         const char* fld, bool verbose) {
    FieldStat st;
    double scale = 0.0;
    for (double v : ref) scale = std::fmax(scale, std::fabs(v));
    double denom = (scale > 0.0) ? scale : 1.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        double err = std::fabs(got[i] - ref[i]) / denom;
        if (err > st.scaled_max) { st.scaled_max = err; st.worst = (int)i; }
        if (std::fabs(ref[i]) >= 1e-6 * scale) {
            double rel = std::fabs(got[i] - ref[i]) /
                         (std::fabs(ref[i]) > 1e-300 ? std::fabs(ref[i]) : 1.0);
            st.strict_big = std::fmax(st.strict_big, rel);
        }
        if (verbose && err > 1e-12) {
            std::printf("    [FAIL] %-6s cell %zu got=%.17g ref=%.17g scaled=%.3e\n",
                        fld, i, got[i], ref[i], err);
        }
    }
    std::printf("    %-6s field-scale=%.3e  field-scaled max=%.3e (cell %d)  strict-rel(big)=%.3e\n",
                fld, scale, st.scaled_max, st.worst, st.strict_big);
    return st;
}

static int run_case(const char* name, const IC& c, const char* refpath,
                    const EOS& eos1, const EOS& eos2) {
    RefW ref;
    if (!load_ref(refpath, ref)) return 1;
    if ((int)ref.alpha.size() != (int)c.alpha.size()) {
        std::printf("[%s] size mismatch: ref=%zu ic=%zu\n", name, ref.alpha.size(), c.alpha.size());
        return 1;
    }
    StepResult R = cfd::five_eq::imex_ad_step(
        c.alpha, c.T1, c.T2, c.u, c.p, c.dt, c.dx, eos1, eos2, c.cfg);

    const char* cl = (R.closure == PressureClosure::pressure_work_consistent)
        ? "pressure_work_consistent"
        : (R.closure == PressureClosure::compressive_recovery ? "compressive_recovery"
                                                              : "implicit_energy");
    std::printf("[%s] closure=%s  vacuum_velocity_cells=%d\n", name, cl, R.vacuum_velocity_cells);

    FieldStat sa = compare(R.alpha, ref.alpha, "alpha", true);
    FieldStat s1 = compare(R.T1,    ref.T1,    "T1",    true);
    FieldStat s2 = compare(R.T2,    ref.T2,    "T2",    true);
    FieldStat su = compare(R.u,     ref.u,     "u",     true);
    FieldStat sp = compare(R.p,     ref.p,     "p",     true);

    double maxfs = std::fmax(std::fmax(std::fmax(sa.scaled_max, s1.scaled_max),
                             std::fmax(s2.scaled_max, su.scaled_max)), sp.scaled_max);
    bool pass = maxfs <= 1e-12;
    std::printf("[%s] %s  max field-scaled=%.3e  (p strict-rel=%.3e, alpha strict-rel=%.3e)\n",
                name, pass ? "PASS" : "FAIL", maxfs, sp.strict_big, sa.strict_big);
    return pass ? 0 : 1;
}

int main() {
    EOS eos1 = eos_air();
    EOS eos2 = eos_water();
    int rc = 0;
    rc |= run_case("02A", make_02A(), STEP_02A_REF, eos1, eos2);
    std::printf("\n");
    rc |= run_case("07B", make_07B(eos1, eos2), STEP_07B_REF, eos1, eos2);
    std::printf("\ntest_5eq_step: %s\n", rc == 0 ? "ALL PASS" : "FAILURES");
    return rc;
}
