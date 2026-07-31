#include "denner1d/validation.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> split_ids(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}
}

int main(int argc, char** argv) {
    std::vector<std::string> only;
    std::string out = "results_cpp/1D";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--only" && i + 1 < argc) only = split_ids(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out = argv[++i];
    }
    return denner1d::validate_cases(only, out);
}
