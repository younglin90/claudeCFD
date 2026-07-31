// Typed C++ counterpart of eos_facade.py.
#pragma once

#include <stdexcept>
#include <string_view>
#include <utility>

#include "cfd/eos.hpp"

namespace cfd::five_eq {

struct EOSConfig {
    double gamma = 1.4;
    double pinf = 0.0;
    double kv = 717.5;
    double b = 0.0;
    double eta = 0.0;
};

inline EOS make_eos(std::string_view kind, const EOSConfig& config = {}) {
    if (kind == "ideal" || kind == "gas")
        return EOS::ideal(config.gamma, config.kv);
    if (kind == "sg" || kind == "stiffened")
        return EOS::sg(config.gamma, config.pinf, config.kv);
    if (kind == "nasg")
        return EOS::nasg(config.gamma, config.pinf, config.kv, config.b, config.eta);
    throw std::invalid_argument("five_eq make_eos supports ideal, sg, and nasg");
}

inline bool eos_is_admissible(const EOS& eos, double rho) {
    return eos.is_admissible(rho) &&
           (eos.kind != EOS::NASG || eos.b * rho < 0.95);
}

struct EOSPair {
    EOS eos1;
    EOS eos2;

    std::pair<std::string_view, std::string_view> names() const {
        const auto name = [](const EOS& eos) -> std::string_view {
            return eos.kind == EOS::Ideal ? "ideal" :
                   eos.kind == EOS::SG ? "sg" : "nasg";
        };
        return {name(eos1), name(eos2)};
    }

    bool admissible(double rho1, double rho2) const {
        return eos_is_admissible(eos1, rho1) && eos_is_admissible(eos2, rho2);
    }

    void assert_admissible(double rho1, double rho2) const {
        if (!admissible(rho1, rho2))
            throw std::domain_error("inadmissible five-equation EOS state");
    }
};

} // namespace cfd::five_eq
