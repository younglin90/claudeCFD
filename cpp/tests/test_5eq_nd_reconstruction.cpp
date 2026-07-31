#include "cfd/five_eq/nd_reconstruction.hpp"

#include <cmath>
#include <vector>

using namespace cfd::five_eq;

int main() {
    const std::array<int, 2> shape{3, 2};
    std::vector<PrimND<2>> W(6);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            auto& w = W[i * 2 + j];
            w.alpha = 0.2 + 0.2 * i;
            w.T1 = 300.0 + i;
            w.T2 = 310.0 + j;
            w.velocity = {1.0 + i, -2.0 - j};
            w.p = 1.e5 + 100.0 * i + j;
        }
    }

    const auto periodic = nd_reconstruct_face<2>(W, shape, 4, 0, 0, NDBoundary::Periodic);
    if (std::fabs(periodic[0].alpha - periodic[1].alpha) < 1.e-12) return 1;
    const auto reflected = nd_sample_primitive<2>(W, shape, 0, 0, -1, NDBoundary::Reflective);
    if (std::fabs(reflected.velocity[0] + W[0].velocity[0]) > 1.e-12) return 2;
    if (std::fabs(reflected.velocity[1] - W[0].velocity[1]) > 1.e-12) return 3;
    const auto transmissive = nd_sample_primitive<2>(W, shape, 0, 0, -2, NDBoundary::Transmissive);
    if (std::fabs(transmissive.p - W[0].p) > 1.e-12) return 4;
    const auto bounded = nd_reconstruct_face<2>(W, shape, 2, 0, 0, NDBoundary::Periodic);
    if (bounded[0].alpha < 0.0 || bounded[0].alpha > 1.0 ||
        bounded[1].alpha < 0.0 || bounded[1].alpha > 1.0) return 5;
    return 0;
}
