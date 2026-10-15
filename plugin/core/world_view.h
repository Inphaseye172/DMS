// Host-independent view of an After Effects pixel buffer. Pixel layouts mirror the AE SDK:
//   PF_Pixel8     { A_u_char  alpha, red, green, blue }   max 255
//   PF_Pixel16    { A_u_short alpha, red, green, blue }   max 32768 (PF_MAX_CHAN16)
//   PF_PixelFloat { float     alpha, red, green, blue }   nominal 0..1 (may exceed in 32 bpc)
// AE pixels are straight (un-premultiplied) alpha.
#pragma once

#include <cstddef>
#include <cstdint>

namespace dms_plugin {

enum class PixelDepth : int { U8 = 8, U16 = 16, F32 = 32 };

struct Pixel8 { uint8_t a, r, g, b; };
struct Pixel16 { uint16_t a, r, g, b; };
struct PixelF { float a, r, g, b; };
static_assert(sizeof(Pixel8) == 4 && sizeof(Pixel16) == 8 && sizeof(PixelF) == 16, "AE pixel layouts");

constexpr float kMax8 = 255.f;
constexpr float kMax16 = 32768.f;

struct WorldView {
    void* data = nullptr;
    int32_t width = 0;
    int32_t height = 0;
    ptrdiff_t rowbytes = 0;
    PixelDepth depth = PixelDepth::U8;

    bool valid() const noexcept {
        if (!data || width <= 0 || height <= 0) return false;
        const ptrdiff_t px = depth == PixelDepth::U8 ? 4 : depth == PixelDepth::U16 ? 8 : 16;
        return rowbytes >= px * width;
    }
    uint8_t* row(int32_t y) const noexcept { return static_cast<uint8_t*>(data) + static_cast<ptrdiff_t>(y) * rowbytes; }
};

struct Rect {
    int32_t left = 0, top = 0, right = 0, bottom = 0;
    int32_t width() const noexcept { return right > left ? right - left : 0; }
    int32_t height() const noexcept { return bottom > top ? bottom - top : 0; }
    bool empty() const noexcept { return width() == 0 || height() == 0; }
};

inline Rect intersect(const Rect& a, const Rect& b) noexcept {
    Rect r{a.left > b.left ? a.left : b.left, a.top > b.top ? a.top : b.top, a.right < b.right ? a.right : b.right,
           a.bottom < b.bottom ? a.bottom : b.bottom};
    if (r.right < r.left) r.right = r.left;
    if (r.bottom < r.top) r.bottom = r.top;
    return r;
}

}  // namespace dms_plugin
