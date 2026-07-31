// Shared 1-D validation guards. Port of solver_5eq/.codex-loop/oscillation_guards.py.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cfd::validation {

struct HFField { std::string name; const std::vector<double>& num; const std::vector<double>& ref; double floor; };
struct HFFieldMetrics { double smooth_hf = 0.0, local_hf = 0.0, local_tv_excess = 0.0, overshoot = 0.0, tv_excess = 0.0; int local_turns = 0, sharp_turns = 0; bool ok = true; };
struct HFMetrics { bool ok = true; int sharp_cells = 0, smooth_cells = 0; std::vector<HFFieldMetrics> fields; };

inline std::vector<std::pair<int, int>> mask_segments(const std::vector<bool>& mask) {
    std::vector<std::pair<int, int>> out;
    for (int i = 0, n = static_cast<int>(mask.size()); i < n;) {
        if (!mask[i]) { ++i; continue; }
        const int first = i;
        while (i + 1 < n && mask[i + 1]) ++i;
        out.emplace_back(first, i++);
    }
    return out;
}

inline double field_scale(const std::vector<double>& ref, double floor) {
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (double v : ref) if (std::isfinite(v)) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
    return std::isfinite(lo) ? std::max({hi - lo, floor, 1.0e-300}) : std::fmax(floor, 1.0);
}

inline void grow_sharp_mask(std::vector<bool>& mask, const std::vector<double>& x,
                            const std::vector<double>& values, double dx, int cells = 24) {
    const int n = static_cast<int>(x.size());
    if (values.size() != x.size() || n < 3) return;
    const auto [lo_it, hi_it] = std::minmax_element(values.begin(), values.end());
    const double amp = *hi_it - *lo_it;
    if (!std::isfinite(amp) || amp <= 1.0e-300) return;
    std::vector<double> edge(n - 1);
    for (int i = 0; i + 1 < n; ++i) edge[i] = std::fabs(values[i + 1] - values[i]);
    auto middle = edge.begin() + edge.size() / 2;
    std::nth_element(edge.begin(), middle, edge.end());
    const double threshold = std::max({0.15 * amp, 8.0 * *middle, 1.0e-14});
    for (int i = 0; i + 1 < n; ++i) if (std::fabs(values[i + 1] - values[i]) > threshold)
        for (int j = std::max(0, i - cells); j <= std::min(n - 1, i + cells + 1); ++j) mask[j] = true;
}

inline int count_turns(const std::vector<double>& y, double tolerance) {
    int turns = 0, previous = 0;
    for (int i = 1; i < static_cast<int>(y.size()); ++i) {
        const double d = y[i] - y[i - 1];
        if (std::fabs(d) <= tolerance) continue;
        const int sign = d > 0.0 ? 1 : -1;
        if (previous && sign != previous) ++turns;
        previous = sign;
    }
    return turns;
}

inline std::pair<double, double> smooth_hf(const HFField& f, const std::vector<bool>& smooth, double scale) {
    double peak = 0.0, sum_sq = 0.0; int count = 0;
    for (int i = 1; i + 1 < static_cast<int>(f.num.size()); ++i) {
        if (!(smooth[i - 1] && smooth[i] && smooth[i + 1])) continue;
        const double r0 = f.num[i - 1] - f.ref[i - 1], r1 = f.num[i] - f.ref[i], r2 = f.num[i + 1] - f.ref[i + 1];
        const double d2 = std::fabs(r1 - 0.5 * (r0 + r2));
        peak = std::fmax(peak, d2); sum_sq += d2 * d2; ++count;
    }
    return {peak / scale, count ? std::sqrt(sum_sq / count) / scale : 0.0};
}

inline void smooth_local(const HFField& f, const std::vector<bool>& smooth, double& local_hf,
                         double& tv_excess, int& turns_out) {
    constexpr int width_target = 21;
    for (const auto [lo, hi] : mask_segments(smooth)) {
        const int width = std::min(width_target, hi - lo + 1);
        if (width < 5) continue;
        for (int start = lo; start + width - 1 <= hi; ++start) {
            const int end = start + width - 1;
            double tv_ref = 0.0, tv_num = 0.0, mn_ref = f.ref[start], mx_ref = f.ref[start];
            double mag = std::fmax(std::fabs(f.ref[start]), std::fabs(f.num[start]));
            std::vector<double> nb; nb.reserve(width);
            for (int i = start; i <= end; ++i) {
                mn_ref = std::fmin(mn_ref, f.ref[i]); mx_ref = std::fmax(mx_ref, f.ref[i]);
                mag = std::fmax(mag, std::fmax(std::fabs(f.ref[i]), std::fabs(f.num[i]))); nb.push_back(f.num[i]);
                if (i > start) { tv_ref += std::fabs(f.ref[i] - f.ref[i - 1]); tv_num += std::fabs(f.num[i] - f.num[i - 1]); }
            }
            const double span = std::fabs(f.ref[end] - f.ref[start]);
            const double scale = std::max({tv_ref, mx_ref - mn_ref, f.floor, 1.0e-300});
            if (tv_ref - span > 1.0e-10 * scale) continue;
            tv_excess = std::fmax(tv_excess, std::fmax(0.0, tv_num - std::fmax(tv_ref, span)) / scale);
            for (int i = 1; i + 1 < width; ++i) {
                const double d2 = std::fabs((nb[i] - f.ref[start + i]) - 0.5 * ((nb[i - 1] - f.ref[start + i - 1]) + (nb[i + 1] - f.ref[start + i + 1])));
                local_hf = std::fmax(local_hf, d2 / scale);
            }
            const double slope_tol = std::max({1.0e-10 * std::fmax(mag, 1.0), 1.0e-8 * scale, 0.05 * tv_num / (width - 1)});
            turns_out = std::max(turns_out, count_turns(nb, slope_tol));
        }
    }
}

inline HFMetrics high_frequency_guard(const std::vector<double>& x, const std::vector<HFField>& fields,
                                      const std::vector<double>& centers = {}, double smooth_limit = 0.08,
                                      double local_tv_limit = 0.50, int local_turn_limit = 4,
                                      double sharp_overshoot_limit = 0.12, double sharp_tv_limit = 0.75,
                                      int sharp_turn_limit = 2) {
    HFMetrics out; const int n = static_cast<int>(x.size());
    if (n < 4) return out;
    std::vector<bool> sharp(n, false);
    for (const auto& f : fields) { grow_sharp_mask(sharp, x, f.ref, 1.0); grow_sharp_mask(sharp, x, f.num, 1.0); }
    const double dx = n > 1 ? std::fabs(x[1] - x[0]) : 1.0;
    for (double center : centers) for (int i = 0; i < n; ++i) if (std::fabs(x[i] - center) <= 24.0 * dx) sharp[i] = true;
    std::vector<bool> smooth(n, true);
    for (int i = 0; i < n; ++i) smooth[i] = !sharp[i];
    if (n > 4) { smooth[0] = smooth[1] = smooth[n - 2] = smooth[n - 1] = false; }
    out.sharp_cells = static_cast<int>(std::count(sharp.begin(), sharp.end(), true));
    out.smooth_cells = static_cast<int>(std::count(smooth.begin(), smooth.end(), true));
    for (const auto& f : fields) {
        HFFieldMetrics m; const double scale = field_scale(f.ref, f.floor);
        m.smooth_hf = smooth_hf(f, smooth, scale).first;
        smooth_local(f, smooth, m.local_hf, m.local_tv_excess, m.local_turns);
        for (const auto [lo, hi] : mask_segments(sharp)) {
            double mn = f.ref[lo], mx = f.ref[lo], tv_ref = 0.0, tv_num = 0.0;
            for (int i = lo; i <= hi; ++i) { mn = std::fmin(mn, f.ref[i]); mx = std::fmax(mx, f.ref[i]); if (i > lo) { tv_ref += std::fabs(f.ref[i] - f.ref[i - 1]); tv_num += std::fabs(f.num[i] - f.num[i - 1]); } }
            const double physical_tv = std::max({tv_ref, mx - mn, 1.0e-300});
            const double jump = std::fmax(mx - mn, scale);
            double nmn = f.num[lo], nmx = f.num[lo]; std::vector<double> nb;
            for (int i = lo; i <= hi; ++i) { nmn = std::fmin(nmn, f.num[i]); nmx = std::fmax(nmx, f.num[i]); nb.push_back(f.num[i]); }
            m.overshoot = std::fmax(m.overshoot, std::fmax(0.0, std::fmax(nmx - mx, mn - nmn)) / jump);
            m.tv_excess = std::fmax(m.tv_excess, std::fmax(0.0, tv_num - physical_tv) / std::fmax(physical_tv, scale));
            m.sharp_turns = std::max(m.sharp_turns, count_turns(nb, 0.01 * jump));
        }
        m.ok = m.smooth_hf <= smooth_limit && (m.local_turns <= local_turn_limit || (m.local_hf <= smooth_limit && m.local_tv_excess <= local_tv_limit)) && m.overshoot <= sharp_overshoot_limit && m.tv_excess <= sharp_tv_limit && m.sharp_turns <= sharp_turn_limit;
        out.ok = out.ok && m.ok; out.fields.push_back(m);
    }
    return out;
}

} // namespace cfd::validation
