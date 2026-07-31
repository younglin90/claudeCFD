// Array helpers matching primitive.py's public pack/unpack/uniform utilities.
#pragma once

#include <stdexcept>
#include <vector>

#include "cfd/five_eq/step.hpp"

namespace cfd::five_eq {

inline std::vector<double> pack_W(const StepResult& W) {
    const std::size_t n = W.alpha.size();
    if (W.T1.size() != n || W.T2.size() != n || W.u.size() != n || W.p.size() != n)
        throw std::invalid_argument("pack_W requires equally sized primitive fields");
    std::vector<double> out;
    out.reserve(5 * n);
    out.insert(out.end(), W.alpha.begin(), W.alpha.end());
    out.insert(out.end(), W.T1.begin(), W.T1.end());
    out.insert(out.end(), W.T2.begin(), W.T2.end());
    out.insert(out.end(), W.u.begin(), W.u.end());
    out.insert(out.end(), W.p.begin(), W.p.end());
    return out;
}

inline StepResult unpack_W(const std::vector<double>& flat, std::size_t n) {
    if (flat.size() != 5 * n)
        throw std::invalid_argument("unpack_W requires exactly 5*N values");
    StepResult out;
    out.alpha.assign(flat.begin(), flat.begin() + n);
    out.T1.assign(flat.begin() + n, flat.begin() + 2 * n);
    out.T2.assign(flat.begin() + 2 * n, flat.begin() + 3 * n);
    out.u.assign(flat.begin() + 3 * n, flat.begin() + 4 * n);
    out.p.assign(flat.begin() + 4 * n, flat.end());
    return out;
}

inline StepResult uniform_W(std::size_t n, double alpha, double T1, double T2,
                            double u, double p) {
    StepResult out;
    out.alpha.assign(n, alpha);
    out.T1.assign(n, T1);
    out.T2.assign(n, T2);
    out.u.assign(n, u);
    out.p.assign(n, p);
    return out;
}

} // namespace cfd::five_eq
