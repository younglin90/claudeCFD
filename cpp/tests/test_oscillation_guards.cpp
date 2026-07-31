#include "cfd/validation/oscillation_guards.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr int n = 80;
    std::vector<double> x(n), ref(n), smooth(n), ringing(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 0.01 * i;
        ref[i] = std::sin(2.0 * 3.141592653589793 * x[i]);
        smooth[i] = ref[i] + 1.0e-5;
        ringing[i] = ref[i] + (i % 2 ? 0.25 : -0.25);
    }
    const auto good = cfd::validation::high_frequency_guard(
        x, {{"u", smooth, ref, 1.0}}, {}, 0.08, 0.50, 4, 0.12, 0.75, 2);
    const auto bad = cfd::validation::high_frequency_guard(
        x, {{"u", ringing, ref, 1.0}}, {}, 0.08, 0.50, 4, 0.12, 0.75, 2);
    if (!good.ok || bad.ok) {
        std::fprintf(stderr, "oscillation guard mismatch: good=%d bad=%d\n", good.ok, bad.ok);
        return 1;
    }
    std::printf("oscillation guards passed\n");
    return 0;
}
