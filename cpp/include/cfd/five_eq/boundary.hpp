// Primitive ghost construction, port of five_eq_IMEX/boundary.py::extend.
#pragma once

#include <optional>
#include <vector>

#include "cfd/five_eq/material_update.hpp"

namespace cfd::five_eq {

inline std::vector<double> extend_component(const std::vector<double>& values, BC5 left, BC5 right,
                                            bool odd = false, std::optional<double> left_value = {},
                                            std::optional<double> right_value = {}) {
    const int n = static_cast<int>(values.size());
    std::vector<double> out(n + 2);
    for (int i = 0; i < n; ++i) out[i + 1] = values[i];
    if (left == BC5::Periodic && right == BC5::Periodic) { out[0] = values[n-1]; out[n+1] = values[0]; return out; }
    const bool left_dir = left == BC5::Inlet || left == BC5::InletAcoustic || left == BC5::Dirichlet;
    const bool right_dir = right == BC5::Inlet || right == BC5::Outlet || right == BC5::Dirichlet;
    out[0] = left_dir && left_value ? *left_value : (left == BC5::Reflective && odd ? -values[0] : values[0]);
    out[n + 1] = right_dir && right_value ? *right_value : (right == BC5::Reflective && odd ? -values[n-1] : values[n-1]);
    return out;
}

} // namespace cfd::five_eq
