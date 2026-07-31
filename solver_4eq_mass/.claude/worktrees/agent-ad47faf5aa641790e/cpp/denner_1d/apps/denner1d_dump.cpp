#include "denner1d/cases.hpp"
#include "denner1d/solver.hpp"

#include <iostream>

int main(int argc, char** argv) {
    const std::string id = argc > 1 ? argv[1] : "13";
    try {
        const auto c = denner1d::find_case(id);
        const auto s = denner1d::solve_case(c);
        const auto r = denner1d::reference_state(c);
        std::cout << "x,alpha,p,u,rho,p_ref,u_ref,rho_ref\n";
        for (std::size_t i = 0; i < s.x.size(); ++i) {
            std::cout << s.x[i] << "," << s.alpha[i] << ","
                      << s.p[i] << "," << s.u[i] << "," << s.rho[i] << ","
                      << r.p[i] << "," << r.u[i] << "," << r.rho[i] << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "denner1d_dump: " << e.what() << "\n";
        return 2;
    }
}
