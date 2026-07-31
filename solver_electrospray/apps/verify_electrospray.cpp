#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string runCommandCapture(const std::string& command, int& status) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen((command + " 2>&1").c_str(), "r");
  if (!pipe) throw std::runtime_error("failed to launch command");
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  status = pclose(pipe);
  return output;
}

int firstMatchInt(const std::string& text, const std::regex& pattern) {
  std::smatch match;
  if (std::regex_search(text, match, pattern) && match.size() > 1) {
    return std::stoi(match[1].str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string buildDir = "build";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--build-dir") {
      if (i + 1 >= argc) {
        std::cerr << "usage: verify_electrospray [--build-dir path]\n";
        return 2;
      }
      buildDir = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cerr << "usage: verify_electrospray [--build-dir path]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  try {
    int status = 0;
    std::string output =
        runCommandCapture("ctest --test-dir " + buildDir + " --output-on-failure", status);
    std::cout << output;

    int passed = firstMatchInt(output, std::regex("([0-9]+)% tests passed"));
    int total = firstMatchInt(output, std::regex("out of ([0-9]+)"));
    int failed = firstMatchInt(output, std::regex("([0-9]+) tests failed"));
    if (passed == 100 && total > 0 && failed == 0) {
      passed = total;
    } else if (total > 0 && failed >= 0) {
      passed = std::max(0, total - failed);
    }
    int failures = status == 0 ? 0 : std::max(1, failed);
    int primaryMetric = failures * 1000 - passed;
    std::cout << "{\"primary_metric\":" << primaryMetric << ",\"failures\":" << failures
              << ",\"pass_count\":" << passed << ",\"total\":" << total << "}\n";
    return status == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "verify_electrospray error: " << e.what() << "\n";
    std::cout << "{\"primary_metric\":1000,\"failures\":1,\"pass_count\":0,\"total\":0}\n";
    return 1;
  }
}
