#include "pixels.h"

#include <cmath>
#include <cstring>

namespace dms_plugin {
namespace {
inline float sanitize01(float v) noexcept {
    if (!std::isfinite(v)) return 0.f;
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}
template <typename P>
inline void read_px(const P& p, float scale, float& a, float& r, float& g, float& b) noexcept {
    a = sanitize01(static_cast<float>(p.a) * scale);
    r = sanitize01(static_cast<float>(p.r) * scale);
    g = sanitize01(static_cast<float>(p.g) * scale);
    b = sanitize01(static_cast<float>(p.b) * scale);
}
template <typename P>
void convert(const WorldView& w, float scale, float* rgb, float* alpha) noexcept {
    for (int32_t y = 0; y < w.height; ++y) {
        const P* row = reinterpret_cast<const P*>(w.row(y));
        float* o = rgb + static_cast<size_t>(y) * static_cast<size_t>(w.width) * 3u;
        for (int32_t x = 0; x < w.width; ++x) {
            float a, r, g, b;
            read_px(row[x], scale, a, r, g, b);
            const float k = 0.5f * (1.f - a);
            o[x * 3 + 0] = r * a + k;
            o[x * 3 + 1] = g * a + k;
            o[x * 3 + 2] = b * a + k;
            if (alpha) alpha[static_cast<size_t>(y) * static_cast<size_t>(w.width) + static_cast<size_t>(x)] = a;
        }
    }
}
inline uint64_t mix(uint64_t h, uint64_t v) noexcept {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h *= 0xff51afd7ed558ccdull;
    return h ^ (h >> 33);
}
}  // namespace

bool world_to_rgb(const WorldView& w, std::vector<float>& rgb, std::vector<float>* alpha) {
    if (!w.valid()) return false;
    const size_t n = static_cast<size_t>(w.width) * static_cast<size_t>(w.height);
    rgb.resize(n * 3u);
    if (alpha) alpha->resize(n);
    float* ap = alpha ? alpha->data() : nullptr;
    switch (w.depth) {
        case PixelDepth::U8: convert<Pixel8>(w, 1.f / kMax8, rgb.data(), ap); break;
        case PixelDepth::U16: convert<Pixel16>(w, 1.f / kMax16, rgb.data(), ap); break;
        case PixelDepth::F32: convert<PixelF>(w, 1.f, rgb.data(), ap); break;
        default: return false;
    }
    return true;
}

void resize_rgb(const std::vector<float>& src, int sw, int sh, std::vector<float>& dst, int dw, int dh) {
    dst.resize(static_cast<size_t>(dw) * static_cast<size_t>(dh) * 3u);
    const float sx = static_cast<float>(sw) / static_cast<float>(dw);
    const float sy = static_cast<float>(sh) / static_cast<float>(dh);
    for (int y = 0; y < dh; ++y) {
        float fy = (static_cast<float>(y) + 0.5f) * sy - 0.5f;
        fy = fy < 0.f ? 0.f : (fy > static_cast<float>(sh - 1) ? static_cast<float>(sh - 1) : fy);
        const int y0 = static_cast<int>(fy), y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
        const float wy = fy - static_cast<float>(y0);
        for (int x = 0; x < dw; ++x) {
            float fx = (static_cast<float>(x) + 0.5f) * sx - 0.5f;
            fx = fx < 0.f ? 0.f : (fx > static_cast<float>(sw - 1) ? static_cast<float>(sw - 1) : fx);
            const int x0 = static_cast<int>(fx), x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
            const float wx = fx - static_cast<float>(x0);
            for (int c = 0; c < 3; ++c) {
                auto at = [&](int xx, int yy) { return src[(static_cast<size_t>(yy) * static_cast<size_t>(sw) + static_cast<size_t>(xx)) * 3u + static_cast<size_t>(c)]; };
                const float top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * wx;
                const float bot = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * wx;
                dst[(static_cast<size_t>(y) * static_cast<size_t>(dw) + static_cast<size_t>(x)) * 3u + static_cast<size_t>(c)] = top + (bot - top) * wy;
            }
        }
    }
}

uint64_t hash_world(const WorldView& w) noexcept {
    if (!w.valid()) return 0;
    const size_t px = w.depth == PixelDepth::U8 ? 4 : w.depth == PixelDepth::U16 ? 8 : 16;
    const size_t row_bytes = px * static_cast<size_t>(w.width);
    uint64_t h = mix(0x84222325cbf29ce4ull, (static_cast<uint64_t>(w.width) << 32) | static_cast<uint32_t>(w.height));
    h = mix(h, static_cast<uint64_t>(w.depth));
    for (int32_t y = 0; y < w.height; ++y) {
        const uint8_t* r = w.row(y);
        size_t i = 0;
        uint64_t acc = 0;
        for (; i + 8 <= row_bytes; i += 8) {
            uint64_t v;
            std::memcpy(&v, r + i, 8);
            acc = (acc ^ v) * 0x100000001b3ull + (acc >> 29);
        }
        for (; i < row_bytes; ++i) acc = (acc ^ r[i]) * 0x100000001b3ull;
        h = mix(h, acc);
    }
    return h == 0 ? 1 : h;
}

uint64_t hash_combine(uint64_t a, uint64_t b) noexcept { return mix(a, b); }

void fill_world(const WorldView& out, float value, float alpha) noexcept {
    if (!out.valid()) return;
    value = sanitize01(value);
    alpha = sanitize01(alpha);
    for (int32_t y = 0; y < out.height; ++y) {
        uint8_t* row = out.row(y);
        for (int32_t x = 0; x < out.width; ++x) {
            switch (out.depth) {
                case PixelDepth::U8: {
                    const auto v = static_cast<uint8_t>(std::lround(value * kMax8));
                    reinterpret_cast<Pixel8*>(row)[x] = {static_cast<uint8_t>(std::lround(alpha * kMax8)), v, v, v};
                    break;
                }
                case PixelDepth::U16: {
                    const auto v = static_cast<uint16_t>(std::lround(value * kMax16));
                    reinterpret_cast<Pixel16*>(row)[x] = {static_cast<uint16_t>(std::lround(alpha * kMax16)), v, v, v};
                    break;
                }
                case PixelDepth::F32: reinterpret_cast<PixelF*>(row)[x] = {alpha, value, value, value}; break;
            }
        }
    }
}

}  // namespace dms_plugin
