// Multidimensional primitive reconstruction and boundary sampling.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "cfd/five_eq/primitive_nd.hpp"

namespace cfd::five_eq {

enum class NDBoundary { Periodic, Transmissive, Reflective };
enum class NDLimiter { FirstOrder, Minmod, VanLeer, Superbee };

template <int D> struct NDReconstruction {
    NDLimiter limiter = NDLimiter::Superbee;
    bool bounded_primitive = true; // Python "tmlpu" behavior.
    bool alpha_superbee = true;    // CICSAM/STACS/MSTACS fallback behavior.
};

template <int D>
inline int nd_row_major_index(const std::array<int, D>& coordinate,
                              const std::array<int, D>& shape) {
    int index = 0;
    for (int d = 0; d < D; ++d) index = index * shape[d] + coordinate[d];
    return index;
}

template <int D>
inline std::array<int, D> nd_row_major_coordinate(int index,
                                                   const std::array<int, D>& shape) {
    std::array<int, D> coordinate{};
    for (int d = D - 1; d >= 0; --d) {
        coordinate[d] = index % shape[d];
        index /= shape[d];
    }
    return coordinate;
}

template <int D>
inline int nd_component_count() { return D + 4; }

template <int D>
inline double nd_primitive_component(const PrimND<D>& W, int component) {
    if (component == 0) return W.alpha;
    if (component == 1) return W.T1;
    if (component == 2) return W.T2;
    if (component < D + 3) return W.velocity[component - 3];
    return W.p;
}

template <int D>
inline void set_nd_primitive_component(PrimND<D>& W, int component, double value) {
    if (component == 0) W.alpha = value;
    else if (component == 1) W.T1 = value;
    else if (component == 2) W.T2 = value;
    else if (component < D + 3) W.velocity[component - 3] = value;
    else W.p = value;
}

template <int D>
inline PrimND<D> nd_sample_primitive(const std::vector<PrimND<D>>& W,
                                     const std::array<int, D>& shape,
                                     int cell, int axis, int offset,
                                     NDBoundary boundary) {
    auto coordinate = nd_row_major_coordinate<D>(cell, shape);
    int target = coordinate[axis] + offset;
    bool reflected = false;
    const int n = shape[axis];

    if (boundary == NDBoundary::Periodic) {
        target %= n;
        if (target < 0) target += n;
    } else if (boundary == NDBoundary::Transmissive) {
        target = std::clamp(target, 0, n - 1);
    } else {
        while (target < 0 || target >= n) {
            if (target < 0) target = -target - 1;
            else target = 2 * n - target - 1;
            reflected = !reflected;
        }
    }

    coordinate[axis] = target;
    PrimND<D> value = W[nd_row_major_index<D>(coordinate, shape)];
    if (reflected) value.velocity[axis] = -value.velocity[axis];
    return value;
}

inline double nd_limited_slope(double dm, double dp, NDLimiter limiter) {
    if (limiter == NDLimiter::FirstOrder || dm * dp <= 0.0) return 0.0;
    const double sign = dm > 0.0 ? 1.0 : -1.0;
    const double adm = std::fabs(dm);
    const double adp = std::fabs(dp);
    if (limiter == NDLimiter::Minmod) return sign * std::min(adm, adp);
    if (limiter == NDLimiter::VanLeer) {
        const double denominator = dm + dp;
        return std::fabs(denominator) > 1.e-12 ? 2.0 * dm * dp / denominator : 0.0;
    }
    return sign * std::max(std::min(2.0 * adm, adp), std::min(adm, 2.0 * adp));
}

template <int D>
inline std::array<PrimND<D>, 2> nd_reconstruct_face(
    const std::vector<PrimND<D>>& W, const std::array<int, D>& shape,
    int left_cell, int axis, int left_offset, NDBoundary boundary,
    const NDReconstruction<D>& reconstruction = {}) {
    PrimND<D> left{};
    PrimND<D> right{};
    for (int component = 0; component < nd_component_count<D>(); ++component) {
        const NDLimiter limiter = component == 0 && reconstruction.alpha_superbee
            ? NDLimiter::Superbee : reconstruction.limiter;
        const auto value = [&](int offset) {
            return nd_primitive_component<D>(
                nd_sample_primitive<D>(W, shape, left_cell, axis, offset, boundary), component);
        };
        const double q0 = value(left_offset);
        const double q1 = value(left_offset + 1);
        const double slope0 = nd_limited_slope(value(left_offset) - value(left_offset - 1),
                                               value(left_offset + 1) - value(left_offset), limiter);
        const double slope1 = nd_limited_slope(value(left_offset + 1) - value(left_offset),
                                               value(left_offset + 2) - value(left_offset + 1), limiter);
        double q_left = q0 + 0.5 * slope0;
        double q_right = q1 - 0.5 * slope1;
        if (component == 0 || reconstruction.bounded_primitive) {
            const double lo = std::min(q0, q1);
            const double hi = std::max(q0, q1);
            q_left = std::clamp(q_left, lo, hi);
            q_right = std::clamp(q_right, lo, hi);
        }
        set_nd_primitive_component<D>(left, component, q_left);
        set_nd_primitive_component<D>(right, component, q_right);
    }
    left.alpha = std::clamp(left.alpha, 0.0, 1.0);
    right.alpha = std::clamp(right.alpha, 0.0, 1.0);
    return {left, right};
}

} // namespace cfd::five_eq
