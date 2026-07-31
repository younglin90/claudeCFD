#include "denner1d/cases.hpp"
#include "denner1d/solver.hpp"
#include "denner1d/validation.hpp"

#include <iostream>

int main(int argc, char** argv) {
    const std::string id = argc > 1 ? argv[1] : "01";
    try {
        const auto c = denner1d::find_case(id);
        const auto s = denner1d::solve_case(c);
        const auto r = denner1d::reference_state(c);
        const auto m = denner1d::compare(s, r);
        std::cout << denner1d::metrics_json(c.id, m, c.config.cells) << "\n";
        return m.finite ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "denner1d_run: " << e.what() << "\n";
        return 2;
    }
}
