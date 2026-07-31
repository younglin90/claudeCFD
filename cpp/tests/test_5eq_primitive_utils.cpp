#include "cfd/five_eq/primitive_utils.hpp"

#include <stdexcept>

using namespace cfd::five_eq;

int main() {
    const StepResult input = uniform_W(3, .4, 300., 310., .2, 1.e5);
    const auto packed = pack_W(input);
    if (packed.size() != 15 || packed[0] != .4 || packed[3] != 300. || packed[12] != 1.e5) return 1;
    const StepResult restored = unpack_W(packed, 3);
    if (restored.alpha != input.alpha || restored.T1 != input.T1 || restored.T2 != input.T2 ||
        restored.u != input.u || restored.p != input.p) return 2;
    try {
        unpack_W(packed, 2);
        return 3;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
