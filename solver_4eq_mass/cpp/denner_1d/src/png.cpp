#include "denner1d/png.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace denner1d {
namespace {

std::uint32_t crc32(const std::uint8_t* data, std::size_t n) {
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < n; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return crc ^ 0xffffffffu;
}

std::uint32_t adler32(const std::vector<std::uint8_t>& data) {
    constexpr std::uint32_t mod = 65521u;
    std::uint32_t a = 1u;
    std::uint32_t b = 0u;
    for (auto v : data) {
        a = (a + v) % mod;
        b = (b + a) % mod;
    }
    return (b << 16u) | a;
}

void u32be(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>((v >> 24u) & 255u));
    out.push_back(static_cast<std::uint8_t>((v >> 16u) & 255u));
    out.push_back(static_cast<std::uint8_t>((v >> 8u) & 255u));
    out.push_back(static_cast<std::uint8_t>(v & 255u));
}

void chunk(std::vector<std::uint8_t>& png, const char type[4], const std::vector<std::uint8_t>& data) {
    u32be(png, static_cast<std::uint32_t>(data.size()));
    const std::size_t start = png.size();
    for (int i = 0; i < 4; ++i) png.push_back(static_cast<std::uint8_t>(type[i]));
    png.insert(png.end(), data.begin(), data.end());
    const std::uint32_t crc = crc32(png.data() + start, png.size() - start);
    u32be(png, crc);
}

}  // namespace

void write_png_rgb(const std::string& path,
                   int width,
                   int height,
                   const std::vector<Rgb>& pixels) {
    if (width <= 0 || height <= 0 || static_cast<int>(pixels.size()) != width * height) {
        throw std::runtime_error("invalid PNG dimensions");
    }

    std::vector<std::uint8_t> raw;
    raw.reserve(height * (1 + 3 * width));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        for (int x = 0; x < width; ++x) {
            const auto& p = pixels[y * width + x];
            raw.push_back(p.r);
            raw.push_back(p.g);
            raw.push_back(p.b);
        }
    }

    std::vector<std::uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    std::size_t pos = 0;
    while (pos < raw.size()) {
        const std::size_t block = std::min<std::size_t>(65535, raw.size() - pos);
        const bool final = pos + block == raw.size();
        z.push_back(final ? 1 : 0);
        const auto len = static_cast<std::uint16_t>(block);
        const auto nlen = static_cast<std::uint16_t>(~len);
        z.push_back(static_cast<std::uint8_t>(len & 255u));
        z.push_back(static_cast<std::uint8_t>((len >> 8u) & 255u));
        z.push_back(static_cast<std::uint8_t>(nlen & 255u));
        z.push_back(static_cast<std::uint8_t>((nlen >> 8u) & 255u));
        z.insert(z.end(), raw.begin() + static_cast<long>(pos), raw.begin() + static_cast<long>(pos + block));
        pos += block;
    }
    u32be(z, adler32(raw));

    std::vector<std::uint8_t> png = {137, 80, 78, 71, 13, 10, 26, 10};
    std::vector<std::uint8_t> ihdr;
    u32be(ihdr, static_cast<std::uint32_t>(width));
    u32be(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8);
    ihdr.push_back(2);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    chunk(png, "IHDR", ihdr);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", {});

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open PNG output: " + path);
    f.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
}

}  // namespace denner1d
