// Core 07-B linear-acoustic reflection/transmission validation for C++ imex_ad.
#include "cfd/five_eq/solver.hpp"
#include "cfd/validation/oscillation_guards.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using cfd::BC5;
using cfd::EOS;
using cfd::five_eq::RunConfig;
using cfd::five_eq::RunTermination;

namespace {

constexpr double P0 = 1.0e5;
constexpr double U_PEAK = 0.02;
constexpr int N = 400;
constexpr int PEAK_CELL_TOL = 3;

struct Material {
    const char* name;
    double gamma, kv, rho, c;
    bool water = false;
};

struct Case {
    const char* name;
    Material left, right;
    double x_interface, x_source, sigma, t_end;
};

const Material AIR{"Air", 1.4, 717.5, 1.157, 347.8};
const Material HELIUM{"Helium", 1.667, 3120.0, 0.164, 1008.2};
const Material ARGON{"Argon", 1.660, 312.0, 1.748, 308.2};
const Material WATER{"Water", 1.187, 3610.0, 998.0, 1567.3350584385664, true};
const Case CASES[] = {
    {"Air-Water", AIR, WATER, 0.5, 0.1, 0.014, 1.55e-3},
    {"Helium-Air", HELIUM, AIR, 1.0, 0.2, 0.049, 1.513e-3},
    {"Argon-Air", ARGON, AIR, 0.5, 0.1, 0.038, 2.02e-3},
};

EOS make_eos(const Material& m) {
    return m.water ? EOS::nasg(m.gamma, 7.028e8, m.kv, 6.61e-4, -1.177788e6)
                   : EOS::ideal(m.gamma, m.kv);
}

double temperature_at(const EOS& eos, double rho) {
    return eos.temperature(rho, eos.energy(rho, P0));
}

double theta_from_eos(const EOS& eos, double T) {
    const double rho = eos.density(P0, T);
    const double pr2 = P0 / std::fmax(rho * rho, 1.0e-300);
    const double den = eos.dedT_p(rho, T) - pr2 * eos.drhodT_p(rho, T);
    return (pr2 * eos.drhodp_T(rho, T) - eos.dedp_T(rho, T)) /
           (std::fabs(den) > 1.0e-300 ? den : 1.0e-300);
}

double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0.0, mb = 0.0;
    for (int i = 0; i < N; ++i) { ma += a[i]; mb += b[i]; }
    ma /= N; mb /= N;
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (int i = 0; i < N; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        dot += da * db; aa += da * da; bb += db * db;
    }
    return aa * bb > 1.0e-300 ? dot / std::sqrt(aa * bb) : 1.0;
}

struct Metrics {
    double L2p, Lip, L2u, Liu, frac_p, frac_u, L1p, L1u, corr_p, corr_u;
};

struct Checkerboard { double alt_ratio = 0.0, amplitude = 0.0; };

Checkerboard checkerboard_metric(const std::vector<double>& x, const std::vector<double>& residual,
                                 double center, double dx) {
    std::vector<double> values;
    const double width = std::fmax(10.0 * dx, 0.06);
    for (int i = 0; i < N; ++i) if (std::fabs(x[i] - center) <= width) values.push_back(residual[i]);
    if (values.size() < 6) return {};
    double mean = 0.0;
    for (double v : values) mean += v;
    mean /= values.size();
    double amplitude = 0.0, alternating = 0.0, l1 = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double v = values[i] - mean;
        amplitude = std::fmax(amplitude, std::fabs(v));
        alternating += (i % 2 ? -v : v);
        l1 += std::fabs(v);
    }
    return {amplitude > 1.0e-14 ? std::fabs(alternating) / std::fmax(l1, 1.0e-300) : 0.0, amplitude};
}

struct PeakMetrics {
    bool location_ok = true;
    bool amplitude_ok = true;
    int p_index = -1, p_exact_index = -1;
    int u_index = -1, u_exact_index = -1;
    double p_amplitude_ratio = 1.0, u_amplitude_ratio = 1.0;
};

struct PacketAmplitude { double min_ratio = 1.0, max_ratio = 1.0; bool present = false; };

std::string case_slug(const char* name) {
    std::string slug{name};
    for (char& ch : slug) {
        if (ch == '-') ch = '_';
        else if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return slug;
}

void write_profile_csv(const Case& c, const std::vector<double>& x,
                       const cfd::five_eq::StepResult& W,
                       const std::vector<double>& pe, const std::vector<double>& ue) {
    const std::filesystem::path out_dir = "solver_5eq/results/1D/cpp_validation";
    std::filesystem::create_directories(out_dir);
    std::ofstream out(out_dir / (case_slug(c.name) + ".csv"));
    out << std::setprecision(17) << "x,p_num,p_exact,u_num,u_exact\n";
    for (int i = 0; i < N; ++i)
        out << x[i] << ',' << W.p[i] << ',' << pe[i] << ',' << W.u[i] << ',' << ue[i] << '\n';
}

int abs_peak_index(const std::vector<double>& values) {
    return static_cast<int>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end(),
        [](double a, double b) { return std::fabs(a) < std::fabs(b); })));
}

PacketAmplitude packet_amplitude_ratios(const std::vector<double>& num,
                                        const std::vector<double>& exact,
                                        double exact_abs_amplitude) {
    PacketAmplitude out;
    if (exact_abs_amplitude <= 1.0e-30) return out;
    bool in_packet = false;
    for (int i = 0; i <= N; ++i) {
        const bool active = i < N && std::fabs(exact[i]) >= 0.10 * exact_abs_amplitude;
        if (active && !in_packet) {
            in_packet = true;
            int first = i;
            int last = i;
            while (last + 1 < N && std::fabs(exact[last + 1]) >= 0.10 * exact_abs_amplitude) ++last;
            const int lo = std::max(0, first - PEAK_CELL_TOL);
            const int hi = std::min(N - 1, last + PEAK_CELL_TOL);
            double num_amp = 0.0, exact_amp = 0.0;
            for (int j = lo; j <= hi; ++j) num_amp = std::fmax(num_amp, std::fabs(num[j]));
            for (int j = first; j <= last; ++j) exact_amp = std::fmax(exact_amp, std::fabs(exact[j]));
            if (exact_amp > 1.0e-30) {
                const double ratio = num_amp / exact_amp;
                out.min_ratio = out.present ? std::fmin(out.min_ratio, ratio) : ratio;
                out.max_ratio = out.present ? std::fmax(out.max_ratio, ratio) : ratio;
                out.present = true;
            }
        }
        if (!active) in_packet = false;
    }
    return out;
}

bool signed_extrema_location_ok(const std::vector<double>& num, const std::vector<double>& exact) {
    const double abs_amplitude = std::fabs(exact[abs_peak_index(exact)]);
    if (abs_amplitude <= 1.0e-30) return true;
    for (bool maximum : {true, false}) {
        const int exact_index = maximum
            ? static_cast<int>(std::distance(exact.begin(), std::max_element(exact.begin(), exact.end())))
            : static_cast<int>(std::distance(exact.begin(), std::min_element(exact.begin(), exact.end())));
        if (std::fabs(exact[exact_index]) < 0.10 * abs_amplitude) continue;
        const int index = maximum
            ? static_cast<int>(std::distance(num.begin(), std::max_element(num.begin(), num.end())))
            : static_cast<int>(std::distance(num.begin(), std::min_element(num.begin(), num.end())));
        if (std::abs(index - exact_index) > PEAK_CELL_TOL) return false;
    }
    return true;
}

PeakMetrics peak_metrics(const std::vector<double>& p, const std::vector<double>& u,
                         const std::vector<double>& pe, const std::vector<double>& ue,
                         bool air_water) {
    std::vector<double> ps(N), pes(N);
    for (int i = 0; i < N; ++i) { ps[i] = p[i] - P0; pes[i] = pe[i] - P0; }
    PeakMetrics out;
    out.p_index = abs_peak_index(ps);
    out.p_exact_index = abs_peak_index(pes);
    out.u_index = abs_peak_index(u);
    out.u_exact_index = abs_peak_index(ue);
    const double p_exact_amp = std::fabs(pes[out.p_exact_index]);
    const double u_exact_amp = std::fabs(ue[out.u_exact_index]);
    out.p_amplitude_ratio = p_exact_amp > 1.0e-30 ? std::fabs(ps[out.p_index]) / p_exact_amp : 1.0;
    out.u_amplitude_ratio = u_exact_amp > 1.0e-30 ? std::fabs(u[out.u_index]) / u_exact_amp : 1.0;
    const bool abs_location_ok = std::abs(out.p_index - out.p_exact_index) <= PEAK_CELL_TOL &&
                                 std::abs(out.u_index - out.u_exact_index) <= PEAK_CELL_TOL;
    out.location_ok = signed_extrema_location_ok(ps, pes) && signed_extrema_location_ok(u, ue) &&
                      (!air_water || abs_location_ok);
    const double min_ratio = air_water ? 0.85 : 0.80;
    const double max_ratio = air_water ? 1.10 : 1.13;
    const PacketAmplitude p_packet = packet_amplitude_ratios(ps, pes, p_exact_amp);
    const PacketAmplitude u_packet = packet_amplitude_ratios(u, ue, u_exact_amp);
    out.amplitude_ok = out.p_amplitude_ratio >= min_ratio && out.p_amplitude_ratio <= max_ratio &&
                       out.u_amplitude_ratio >= min_ratio && out.u_amplitude_ratio <= max_ratio &&
                       p_packet.min_ratio >= min_ratio && p_packet.max_ratio <= max_ratio &&
                       u_packet.min_ratio >= min_ratio && u_packet.max_ratio <= max_ratio;
    return out;
}

double symmetry_error(const std::vector<double>& num, const std::vector<double>& exact) {
    double worst = 0.0;
    const double global = *std::max_element(exact.begin(), exact.end(),
        [](double a, double b) { return std::fabs(a) < std::fabs(b); });
    if (std::fabs(global) <= 1.0e-30) return 0.0;
    for (int center = 1; center + 1 < N; ++center) {
        const bool extremum = (exact[center] >= exact[center - 1] && exact[center] >= exact[center + 1]) ||
                              (exact[center] <= exact[center - 1] && exact[center] <= exact[center + 1]);
        const double amplitude = std::fabs(exact[center]);
        if (!extremum || amplitude < 0.15 * std::fabs(global)) continue;
        const double sign = exact[center] >= 0.0 ? 1.0 : -1.0;
        int left = center, right = center;
        while (left > 0 && sign * exact[left - 1] >= 0.10 * amplitude) --left;
        while (right + 1 < N && sign * exact[right + 1] >= 0.10 * amplitude) ++right;
        int peak_idx = left;
        for (int i = left + 1; i <= right; ++i)
            if (sign * num[i] > sign * num[peak_idx]) peak_idx = i;
        const int radius = std::min(peak_idx - left, right - peak_idx);
        if (radius < 3) continue;
        double sum = 0.0;
        for (int k = 1; k <= radius; ++k)
            sum += std::fabs(num[peak_idx - k] - num[peak_idx + k]);
        worst = std::fmax(worst, sum / radius / std::fmax(std::fabs(num[peak_idx]), amplitude));
    }
    return worst;
}

Metrics profile_metrics(const std::vector<double>& p, const std::vector<double>& u,
                        const std::vector<double>& pe, const std::vector<double>& ue,
                        double dp_wave) {
    double e2p = 0.0, e2u = 0.0, eip = 0.0, eiu = 0.0;
    double in_p = 0.0, in_u = 0.0, den_p = 0.0, den_u = 0.0;
    int good_p = 0, good_u = 0;
    for (int i = 0; i < N; ++i) {
        const double ep = p[i] - pe[i], eu = u[i] - ue[i];
        e2p += ep * ep; e2u += eu * eu;
        eip = std::fmax(eip, std::fabs(ep)); eiu = std::fmax(eiu, std::fabs(eu));
        in_p += std::fabs(ep); in_u += std::fabs(eu);
        den_p += std::fabs(pe[i] - P0); den_u += std::fabs(ue[i]);
        good_p += std::fabs(ep) < 0.30 * dp_wave;
        good_u += std::fabs(eu) < 0.30 * U_PEAK;
    }
    return Metrics{
        std::sqrt(e2p / N) / dp_wave, eip / dp_wave,
        std::sqrt(e2u / N) / U_PEAK, eiu / U_PEAK,
        double(good_p) / N, double(good_u) / N,
        in_p / std::fmax(den_p, 1.0e-300), in_u / std::fmax(den_u, 1.0e-300),
        pearson(p, pe), pearson(u, ue),
    };
}

int run_case(const Case& c) {
    const EOS eos1 = make_eos(c.left), eos2 = make_eos(c.right);
    constexpr double length = 1.5;
    constexpr double dx = length / N;
    const double T1 = temperature_at(eos1, c.left.rho);
    const double T2 = temperature_at(eos2, c.right.rho);
    const double theta = theta_from_eos(eos1, T1);
    const double ZL = c.left.rho * c.left.c;
    std::vector<double> a, t1, t2, u, p, x(N), pe(N), ue(N);
    a.reserve(N); t1.reserve(N); t2.reserve(N); u.reserve(N); p.reserve(N);
    const double ZR = c.right.rho * c.right.c;
    const double R = (ZR - ZL) / (ZR + ZL);
    const double Tu = 2.0 * ZL / (ZL + ZR);
    const double Tp = 2.0 * ZR / (ZL + ZR);
    const double t_hit = (c.x_interface - c.x_source) / c.left.c;
    for (int i = 0; i < N; ++i) {
        const double xi = (i + 0.5) * dx;
        const bool left = xi < c.x_interface;
        const double g0 = std::exp(-((xi - c.x_source) * (xi - c.x_source)) /
                                   (2.0 * c.sigma * c.sigma));
        const double ui = left ? U_PEAK * g0 : 0.0;
        const double pi = P0 + ZL * ui;
        x[i] = xi; a.push_back(left ? 1.0 - 1.0e-8 : 1.0e-8);
        t1.push_back(T1 + (left ? theta * (pi - P0) : 0.0));
        t2.push_back(T2); u.push_back(ui); p.push_back(pi);

        const double inc = U_PEAK * std::exp(-((xi - (c.x_source + c.left.c * c.t_end)) *
                                                (xi - (c.x_source + c.left.c * c.t_end))) /
                                               (2.0 * c.sigma * c.sigma)) * left;
        const double ref = U_PEAK * std::exp(-((xi - (2.0 * c.x_interface - c.x_source - c.left.c * c.t_end)) *
                                                (xi - (2.0 * c.x_interface - c.x_source - c.left.c * c.t_end))) /
                                               (2.0 * c.sigma * c.sigma)) * left;
        double trans = 0.0;
        if (c.t_end > t_hit && !left) {
            const double sigma_r = c.sigma * c.right.c / c.left.c;
            const double center = c.x_interface + c.right.c * (c.t_end - t_hit);
            trans = U_PEAK * std::exp(-((xi - center) * (xi - center)) /
                                       (2.0 * sigma_r * sigma_r));
        }
        ue[i] = inc - R * ref + Tu * trans;
        pe[i] = P0 + ZL * inc + R * ZL * ref + Tp * ZL * trans;
    }

    RunConfig cfg;
    cfg.cfl = 0.4;
    cfg.max_steps = 100000;
    cfg.step_config.alpha_pure_tol = 1.0e-8;
    cfg.step_config.bc_l = BC5::Reflective;
    cfg.step_config.bc_r = BC5::Transmissive;
    std::printf("CXX 07B %-11s start\n", c.name);
    std::fflush(stdout);
    const auto out = cfd::five_eq::solve_imex_ad(a, t1, t2, u, p, dx, c.t_end, eos1, eos2, cfg);
    write_profile_csv(c, x, out.W, pe, ue);
    const Metrics m = profile_metrics(out.W.p, out.W.u, pe, ue, ZL * U_PEAK);
    const bool air_water = std::string(c.name) == "Air-Water";
    const PeakMetrics peak = peak_metrics(out.W.p, out.W.u, pe, ue, air_water);
    std::vector<double> ps(N), pes(N);
    for (int i = 0; i < N; ++i) { ps[i] = out.W.p[i] - P0; pes[i] = pe[i] - P0; }
    const double p_symmetry = symmetry_error(ps, pes);
    const double u_symmetry = symmetry_error(out.W.u, ue);
    std::vector<double> rho(N), rho_exact(N);
    for (int i = 0; i < N; ++i) {
        rho[i] = out.W.alpha[i] * eos1.density(out.W.p[i], out.W.T1[i]) +
                 (1.0 - out.W.alpha[i]) * eos2.density(out.W.p[i], out.W.T2[i]);
        const bool left = x[i] < c.x_interface;
        const double c0 = left ? c.left.c : c.right.c;
        rho_exact[i] = (left ? c.left.rho : c.right.rho) + (pe[i] - P0) / (c0 * c0);
    }
    const auto hf = cfd::validation::high_frequency_guard(
        x,
        {{"rho", rho, rho_exact, 1.0}, {"u", out.W.u, ue, U_PEAK}, {"p", out.W.p, pe, ZL * U_PEAK}},
        {c.x_interface}, 0.10, 0.80, 6, 0.18, 1.10, 4);
    const auto& hf_u = hf.fields[1];
    const auto& hf_p = hf.fields[2];
    const bool air_water_wiggle = !air_water ||
        (hf_p.local_tv_excess <= 0.30 && hf_u.local_tv_excess <= 0.20 && hf_p.local_hf <= 0.04);
    std::vector<double> rp(N), ru(N);
    for (int i = 0; i < N; ++i) {
        rp[i] = (out.W.p[i] - pe[i]) / (ZL * U_PEAK);
        ru[i] = (out.W.u[i] - ue[i]) / U_PEAK;
    }
    const Checkerboard p_checker = checkerboard_metric(x, rp, c.x_interface, dx);
    const Checkerboard u_checker = checkerboard_metric(x, ru, c.x_interface, dx);
    const bool checkerboard_ok = !((p_checker.alt_ratio > 0.60 && p_checker.amplitude > 0.20) ||
                                   p_checker.amplitude > 0.30 ||
                                   (u_checker.alt_ratio > 0.60 && u_checker.amplitude > 0.20) ||
                                   u_checker.amplitude > 0.45);
    const double lip_limit = std::string(c.name) == "Air-Water" ? 0.756 : 0.81;
    const bool pass = out.termination == RunTermination::completed &&
                      m.L2p < 0.216 && m.Lip < lip_limit &&
                      m.L2u < 0.216 && m.Liu < lip_limit &&
                      m.frac_p >= 0.76 && m.frac_u >= 0.76 &&
                      m.L1p < 0.648 && m.L1u < 0.648 &&
                      m.corr_p > 0.88 && m.corr_u > 0.88 &&
                      peak.location_ok && peak.amplitude_ok &&
                      p_symmetry <= 0.38 && u_symmetry <= 0.38 &&
                      hf.ok && air_water_wiggle && checkerboard_ok;
    std::printf("CXX 07B %-11s steps=%d t=%.9e L2p=%.3f Lip=%.3f L2u=%.3f Liu=%.3f "
                "frac=(%.3f,%.3f) corr=(%.3f,%.3f) peak=(%d/%d,%d/%d) "
                "amp=(%.3f,%.3f) sym=(%.3f,%.3f) hf=%d cb=%d aw=%d %s\n",
                c.name, out.steps, out.t_final, m.L2p, m.Lip, m.L2u, m.Liu,
                m.frac_p, m.frac_u, m.corr_p, m.corr_u,
                peak.p_index, peak.p_exact_index, peak.u_index, peak.u_exact_index,
                peak.p_amplitude_ratio, peak.u_amplitude_ratio, p_symmetry, u_symmetry,
                hf.ok ? 1 : 0, checkerboard_ok ? 1 : 0, air_water_wiggle ? 1 : 0, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    const std::string selected = argc > 1 ? argv[1] : "all";
    int rc = 0;
    for (const Case& c : CASES) {
        if (selected == "all" || selected == c.name) rc |= run_case(c);
    }
    if (selected != "all" && selected != "Air-Water" && selected != "Helium-Air" &&
        selected != "Argon-Air") {
        std::fprintf(stderr, "unknown 07B case: %s\n", selected.c_str());
        return 2;
    }
    return rc;
}
