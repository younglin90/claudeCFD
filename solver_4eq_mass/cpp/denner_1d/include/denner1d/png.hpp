#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace denner1d {

struct Rgb {
    std::uint8_t r = 255;
    std::uint8_t g = 255;
    std::uint8_t b = 255;
};

void write_png_rgb(const std::string& path,
                   int width,
                   int height,
                   const std::vector<Rgb>& pixels);

}  // namespace denner1d
