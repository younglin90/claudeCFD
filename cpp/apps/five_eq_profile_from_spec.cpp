// Generic paper-case profile runner. Inputs are exported primitive states from
// the Python validation definitions; time advancement is always C++.
#include "cfd/five_eq/solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using cfd::BC5;
using cfd::EOS;
using cfd::five_eq::RunConfig;
using cfd::five_eq::TimeIntegrator;

namespace {

using Values = std::unordered_map<std::string, std::string>;

Values read_values(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read spec: " + path.string());
    Values values;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos || line.empty() || line[0] == '#') continue;
        values.emplace(line.substr(0, eq), line.substr(eq + 1));
    }
    return values;
}

const std::string& required(const Values& values, const char* key) {
    const auto it = values.find(key);
    if (it == values.end()) throw std::runtime_error(std::string("missing spec key: ") + key);
    return it->second;
}

double number(const Values& values, const char* key) { return std::stod(required(values, key)); }

double optional_number(const Values& values, const char* key, double fallback) {
    const auto it = values.find(key);
    return it == values.end() ? fallback : std::stod(it->second);
}

bool optional_bool(const Values& values, const char* key, bool fallback = false) {
    const auto it = values.find(key);
    if (it == values.end()) return fallback;
    return it->second == "1" || it->second == "true" || it->second == "on";
}

EOS eos_from_spec(const Values& values, const char* prefix) {
    const std::string stem{prefix};
    const std::string kind = required(values, (stem + ".kind").c_str());
    const double gamma = number(values, (stem + ".gamma").c_str());
    const double kv = number(values, (stem + ".kv").c_str());
    if (kind == "ideal") return EOS::ideal(gamma, kv);
    const double pinf = number(values, (stem + ".pinf").c_str());
    if (kind == "sg") return EOS::sg(gamma, pinf, kv);
    if (kind == "nasg") return EOS::nasg(gamma, pinf, kv,
                                            number(values, (stem + ".b").c_str()),
                                            number(values, (stem + ".eta").c_str()));
    throw std::runtime_error("unsupported EOS kind: " + kind);
}

BC5 boundary(const std::string& name) {
    if (name == "periodic") return BC5::Periodic;
    if (name == "reflective") return BC5::Reflective;
    if (name == "transmissive") return BC5::Transmissive;
    if (name == "inlet") return BC5::Inlet;
    if (name == "outlet") return BC5::Outlet;
    throw std::runtime_error("unsupported boundary: " + name);
}

TimeIntegrator integrator(const std::string& name) {
    if (name == "imex_ad") return TimeIntegrator::imex_ad;
    if (name == "imex_ssp3") return TimeIntegrator::imex_ssp3;
    if (name == "ars222") return TimeIntegrator::ars222;
    throw std::runtime_error("unsupported integrator: " + name);
}

cfd::five_eq::PureTvdLimiter pure_tvd_limiter(const std::string& name) {
    using cfd::five_eq::PureTvdLimiter;
    if (name == "minmod") return PureTvdLimiter::Minmod;
    if (name == "mc") return PureTvdLimiter::Mc;
    if (name == "superbee") return PureTvdLimiter::Superbee;
    if (name == "van_albada") return PureTvdLimiter::VanAlbada;
    if (name == "umist") return PureTvdLimiter::Umist;
    if (name == "vanleer") return PureTvdLimiter::VanLeer;
    throw std::runtime_error("unsupported pure TVD limiter: " + name);
}

cfd::TvdLimiter material_tvd_limiter(const std::string& name) {
    using cfd::TvdLimiter;
    if (name == "minmod") return TvdLimiter::Minmod;
    if (name == "mc") return TvdLimiter::MC;
    if (name == "superbee") return TvdLimiter::Superbee;
    if (name == "van_albada") return TvdLimiter::VanAlbada;
    if (name == "umist") return TvdLimiter::Umist;
    if (name == "vanleer") return TvdLimiter::VanLeer;
    throw std::runtime_error("unsupported material TVD limiter: " + name);
}

cfd::five_eq::PrimitiveFilter primitive_filter(const std::string& name) {
    using cfd::five_eq::PrimitiveFilter;
    if (name == "none" || name == "off") return PrimitiveFilter::Off;
    if (name == "auto") return PrimitiveFilter::Auto;
    if (name == "led") return PrimitiveFilter::Led;
    if (name == "led_pressure") return PrimitiveFilter::LedPressure;
    if (name == "led_velocity") return PrimitiveFilter::LedVelocity;
    throw std::runtime_error("unsupported primitive filter: " + name);
}

cfd::KapilaSourceMode kapila_source_mode(const std::string& name) {
    using cfd::KapilaSourceMode;
    if (name == "path") return KapilaSourceMode::Path;
    if (name == "cell") return KapilaSourceMode::Cell;
    if (name == "hybrid") return KapilaSourceMode::Hybrid;
    if (name == "trapezoid") return KapilaSourceMode::Trapezoid;
    if (name == "immiscible_trapezoid") return KapilaSourceMode::ImmiscibleTrapezoid;
    if (name == "mixed_trapezoid") return KapilaSourceMode::MixedTrapezoid;
    if (name == "mixed_path") return KapilaSourceMode::MixedPath;
    throw std::runtime_error("unsupported Kapila source mode: " + name);
}

void read_w0(const std::filesystem::path& path, cfd::five_eq::StepResult& W) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read initial profile: " + path.string());
    std::string line;
    std::getline(in, line);
    const bool has_x = line.rfind("x,", 0) == 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        double x, alpha, T1, T2, u, p;
        if (has_x) {
            if (!(row >> x >> alpha >> T1 >> T2 >> u >> p)) throw std::runtime_error("invalid W0 row");
        } else if (!(row >> alpha >> T1 >> T2 >> u >> p)) {
            throw std::runtime_error("invalid W0 row");
        }
        W.alpha.push_back(alpha); W.T1.push_back(T1); W.T2.push_back(T2);
        W.u.push_back(u); W.p.push_back(p);
    }
    if (W.alpha.empty()) throw std::runtime_error("initial profile has no cells");
}

void write_profile(const std::filesystem::path& path, const cfd::five_eq::RunResult& result,
                   double dx, const EOS& eos1, const EOS& eos2) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write profile: " + path.string());
    out << std::setprecision(17) << "x,alpha,T1,T2,u,p,rho\n";
    for (std::size_t i = 0; i < result.W.alpha.size(); ++i) {
        const double alpha = result.W.alpha[i];
        const double p = result.W.p[i];
        const double rho = alpha * eos1.density(p, result.W.T1[i])
                         + (1.0 - alpha) * eos2.density(p, result.W.T2[i]);
        out << (static_cast<double>(i) + 0.5) * dx << ',' << result.W.alpha[i] << ','
            << result.W.T1[i] << ',' << result.W.T2[i] << ',' << result.W.u[i] << ','
            << result.W.p[i] << ',' << rho << '\n';
    }
}

void report_finiteness(const cfd::five_eq::StepResult& W) {
    const auto report = [](const char* name, const std::vector<double>& values) {
        std::size_t nonfinite = 0;
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (double value : values) {
            if (!std::isfinite(value)) { ++nonfinite; continue; }
            lo = std::min(lo, value);
            hi = std::max(hi, value);
        }
        std::cerr << "  " << name << " nonfinite=" << nonfinite
                  << " min=" << lo << " max=" << hi << '\n';
    };
    report("alpha", W.alpha); report("T1", W.T1); report("T2", W.T2);
    report("u", W.u); report("p", W.p);
}

void report_step(int step, double t, const cfd::five_eq::StepResult& W) {
    const auto bounds = [](const std::vector<double>& values) {
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (double value : values) {
            if (!std::isfinite(value)) return std::pair<double, double>{value, value};
            lo = std::min(lo, value); hi = std::max(hi, value);
        }
        return std::pair<double, double>{lo, hi};
    };
    const auto alpha = bounds(W.alpha), T1 = bounds(W.T1), T2 = bounds(W.T2);
    const auto u = bounds(W.u), p = bounds(W.p);
    std::cerr << std::setprecision(17) << "TRACE step=" << step << " t=" << t
              << " alpha=[" << alpha.first << ',' << alpha.second << ']'
              << " T1=[" << T1.first << ',' << T1.second << ']'
              << " T2=[" << T2.first << ',' << T2.second << ']'
              << " u=[" << u.first << ',' << u.second << ']'
              << " p=[" << p.first << ',' << p.second << "]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: five_eq_profile_from_spec CASE.spec OUTPUT.csv\n";
        return 2;
    }
    try {
        const std::filesystem::path spec_path{argv[1]};
        const Values values = read_values(spec_path);
        const EOS eos1 = eos_from_spec(values, "eos1");
        const EOS eos2 = eos_from_spec(values, "eos2");
        cfd::five_eq::StepResult W0;
        read_w0(spec_path.parent_path() / required(values, "w0"), W0);
        const double dx = number(values, "dx"), t_end = number(values, "t_end");
        RunConfig cfg;
        cfg.cfl = optional_number(values, "cfl", 0.4);
        cfg.max_steps = static_cast<int>(optional_number(values, "max_steps", 100000));
        cfg.step_config.bc_l = boundary(required(values, "bc_l"));
        cfg.step_config.bc_r = boundary(required(values, "bc_r"));
        cfg.step_config.alpha_pure_tol = optional_number(values, "alpha_pure_tol", 1.e-8);
        cfg.step_config.time_integrator = integrator(required(values, "integrator"));
        cfg.step_config.kapila_closure = optional_bool(values, "kapila_closure", true);
        cfg.step_config.pure_branch = optional_bool(values, "pure_branch", true);
        const auto source_it = values.find("kapila_source_mode");
        if (source_it != values.end())
            cfg.step_config.kapila_source_mode = kapila_source_mode(source_it->second);
        const bool characteristic = optional_bool(values, "characteristic_reconstruction", true);
        cfg.step_config.material_characteristic_reconstruction = characteristic;
        const auto material_limiter_it = values.find("material_tvd_limiter");
        cfg.step_config.material_tvd_limiter = material_tvd_limiter(
            material_limiter_it == values.end() ? "superbee"
                                                : material_limiter_it->second);
        cfg.step_config.pure_euler_characteristic_reconstruction =
            optional_bool(values, "pure_characteristic_reconstruction", false);
        const auto limiter_it = values.find("pure_tvd_limiter");
        cfg.step_config.pure_tvd_limiter = pure_tvd_limiter(
            limiter_it == values.end() ? "superbee" : limiter_it->second);
        cfg.step_config.pure_euler_rusanov_fallback = optional_bool(values, "rusanov_fallback", false);
        const auto filter_it = values.find("primitive_filter");
        if (filter_it != values.end())
            cfg.step_config.primitive_filter = primitive_filter(filter_it->second);
        cfg.uniform_periodic_remap = optional_bool(values, "uniform_periodic_remap", false);
        if (values.count("inlet.frequency")) {
            const double frequency = number(values, "inlet.frequency");
            const double u_base = number(values, "inlet.u_base");
            const double u_amplitude = number(values, "inlet.u_amplitude");
            const double p_base = number(values, "inlet.p_base");
            const double p_amplitude = number(values, "inlet.p_amplitude");
            cfg.u_inlet_at = [=](double t) {
                return u_base + u_amplitude * std::sin(2.0 * std::acos(-1.0) * frequency * t);
            };
            cfg.p_inlet_at = [=](double t) {
                return p_base + p_amplitude * std::sin(2.0 * std::acos(-1.0) * frequency * t);
            };
        }
        if (values.count("dt_fixed")) cfg.dt_fixed = number(values, "dt_fixed");
        if (values.count("dt_min")) cfg.dt_min = number(values, "dt_min");
        if (optional_bool(values, "trace")) {
            cfg.step_callback = [](const cfd::five_eq::StepRecord& record,
                                   const cfd::five_eq::StepResult& W) {
                report_step(record.step, record.t, W);
                return true;
            };
        }
        const auto result = cfd::five_eq::solve_imex_ad(W0.alpha, W0.T1, W0.T2, W0.u, W0.p,
                                                         dx, t_end, eos1, eos2, cfg);
        write_profile(argv[2], result, dx, eos1, eos2);
        std::cout << "CXX case=" << required(values, "case") << " steps=" << result.steps
                  << " t=" << std::setprecision(17) << result.t_final
                  << " termination=" << static_cast<int>(result.termination) << '\n';
        if (result.termination != cfd::five_eq::RunTermination::completed) report_finiteness(result.W);
        return result.termination == cfd::five_eq::RunTermination::completed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "five_eq_profile_from_spec: " << error.what() << '\n';
        return 2;
    }
}
