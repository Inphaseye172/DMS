#include "image_io.h"

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

namespace depthcli {
namespace {
inline unsigned char to8(float v) {
    if (!std::isfinite(v)) v = 0.f;
    v = std::clamp(v, 0.f, 1.f);
    return static_cast<unsigned char>(std::lround(v * 255.f));
}
struct FileCloser {
    void operator()(std::FILE* f) const noexcept {
        if (f) std::fclose(f);
    }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;
}  // namespace

bool load_image(const std::string& path, Image& out, std::string& err) {
    int w = 0, h = 0, n = 0;
    // 16-bit load keeps precision for 16-bit PNGs; 8-bit sources are scaled up by stb.
    std::unique_ptr<stbi_us, void (*)(void*)> px(stbi_load_16(path.c_str(), &w, &h, &n, 3), stbi_image_free);
    if (!px) {
        err = stbi_failure_reason() ? stbi_failure_reason() : "unknown stb error";
        return false;
    }
    if (w <= 0 || h <= 0) {
        err = "zero-size image";
        return false;
    }
    out.width = w;
    out.height = h;
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
    out.rgb.resize(count);
    for (size_t i = 0; i < count; ++i) out.rgb[i] = static_cast<float>(px.get()[i]) / 65535.f;
    return true;
}

bool write_png_gray8(const std::string& path, const float* v, int w, int h) {
    std::vector<unsigned char> b(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (size_t i = 0; i < b.size(); ++i) b[i] = to8(v[i]);
    return stbi_write_png(path.c_str(), w, h, 1, b.data(), w) != 0;
}

bool write_png_rgb8(const std::string& path, const float* rgb, int w, int h) {
    std::vector<unsigned char> b(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
    for (size_t i = 0; i < b.size(); ++i) b[i] = to8(rgb[i]);
    return stbi_write_png(path.c_str(), w, h, 3, b.data(), w * 3) != 0;
}

bool write_png_rgba8(const std::string& path, const float* rgb, const float* alpha, int w, int h) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<unsigned char> b(n * 4u);
    for (size_t i = 0; i < n; ++i) {
        b[i * 4] = to8(rgb[i * 3]);
        b[i * 4 + 1] = to8(rgb[i * 3 + 1]);
        b[i * 4 + 2] = to8(rgb[i * 3 + 2]);
        b[i * 4 + 3] = to8(alpha[i]);
    }
    return stbi_write_png(path.c_str(), w, h, 4, b.data(), w * 4) != 0;
}

bool write_pgm16(const std::string& path, const float* v, int w, int h) {
    FilePtr f(std::fopen(path.c_str(), "wb"));
    if (!f) return false;
    std::fprintf(f.get(), "P5\n%d %d\n65535\n", w, h);
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<unsigned char> b(n * 2u);
    for (size_t i = 0; i < n; ++i) {
        float x = std::isfinite(v[i]) ? std::clamp(v[i], 0.f, 1.f) : 0.f;
        const auto q = static_cast<uint16_t>(std::lround(x * 65535.f));
        b[i * 2] = static_cast<unsigned char>(q >> 8);
        b[i * 2 + 1] = static_cast<unsigned char>(q & 0xff);
    }
    return std::fwrite(b.data(), 1, b.size(), f.get()) == b.size();
}

bool write_pfm(const std::string& path, const float* v, int w, int h) {
    FilePtr f(std::fopen(path.c_str(), "wb"));
    if (!f) return false;
    std::fprintf(f.get(), "Pf\n%d %d\n-1.0\n", w, h);  // negative scale = little-endian
    for (int y = h - 1; y >= 0; --y) {  // PFM rows are bottom-to-top
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        if (std::fwrite(v + row, sizeof(float), static_cast<size_t>(w), f.get()) != static_cast<size_t>(w)) return false;
    }
    return true;
}

bool read_pfm(const std::string& path, std::vector<float>& v, int& w, int& h) {
    FilePtr f(std::fopen(path.c_str(), "rb"));
    if (!f) return false;
    char magic[3] = {0};
    float scale = 0.f;
    if (std::fscanf(f.get(), "%2s %d %d %f", magic, &w, &h, &scale) != 4 || std::strcmp(magic, "Pf") != 0 || w <= 0 || h <= 0) return false;
    std::fgetc(f.get());
    v.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = h - 1; y >= 0; --y) {
        if (std::fread(v.data() + static_cast<size_t>(y) * static_cast<size_t>(w), sizeof(float), static_cast<size_t>(w), f.get()) != static_cast<size_t>(w))
            return false;
    }
    return true;
}

}  // namespace depthcli
