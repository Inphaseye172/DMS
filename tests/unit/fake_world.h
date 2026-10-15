#pragma once
#include "world_view.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace dms_test {

// Owns a buffer laid out like an AE PF_EffectWorld (with row padding to catch stride bugs).
struct FakeWorld {
    std::vector<uint8_t> buf;
    dms_plugin::WorldView view;
    FakeWorld(int w, int h, dms_plugin::PixelDepth d, int pad_bytes = 12) {
        const int px = d == dms_plugin::PixelDepth::U8 ? 4 : d == dms_plugin::PixelDepth::U16 ? 8 : 16;
        view.width = w;
        view.height = h;
        view.depth = d;
        view.rowbytes = static_cast<ptrdiff_t>(w * px + pad_bytes);
        buf.assign(static_cast<size_t>(view.rowbytes) * static_cast<size_t>(h > 0 ? h : 1) + 64, 0xCD);
        view.data = buf.data();
    }
    void set(int x, int y, float a, float r, float g, float b) {
        uint8_t* row = view.row(y);
        using namespace dms_plugin;
        switch (view.depth) {
            case PixelDepth::U8:
                reinterpret_cast<Pixel8*>(row)[x] = {static_cast<uint8_t>(a * 255.f + .5f), static_cast<uint8_t>(r * 255.f + .5f),
                                                     static_cast<uint8_t>(g * 255.f + .5f), static_cast<uint8_t>(b * 255.f + .5f)};
                break;
            case PixelDepth::U16:
                reinterpret_cast<Pixel16*>(row)[x] = {static_cast<uint16_t>(a * 32768.f + .5f), static_cast<uint16_t>(r * 32768.f + .5f),
                                                      static_cast<uint16_t>(g * 32768.f + .5f), static_cast<uint16_t>(b * 32768.f + .5f)};
                break;
            case PixelDepth::F32: reinterpret_cast<PixelF*>(row)[x] = {a, r, g, b}; break;
        }
    }
    void get(int x, int y, float& a, float& r, float& g, float& b) const {
        const uint8_t* row = view.row(y);
        using namespace dms_plugin;
        switch (view.depth) {
            case PixelDepth::U8: { auto p = reinterpret_cast<const Pixel8*>(row)[x]; a = p.a / 255.f; r = p.r / 255.f; g = p.g / 255.f; b = p.b / 255.f; break; }
            case PixelDepth::U16: { auto p = reinterpret_cast<const Pixel16*>(row)[x]; a = p.a / 32768.f; r = p.r / 32768.f; g = p.g / 32768.f; b = p.b / 32768.f; break; }
            case PixelDepth::F32: { auto p = reinterpret_cast<const PixelF*>(row)[x]; a = p.a; r = p.r; g = p.g; b = p.b; break; }
        }
    }
    // True if the row padding bytes are untouched (0xCD).
    bool padding_intact() const {
        const int px = view.depth == dms_plugin::PixelDepth::U8 ? 4 : view.depth == dms_plugin::PixelDepth::U16 ? 8 : 16;
        for (int y = 0; y < view.height; ++y)
            for (ptrdiff_t i = px * view.width; i < view.rowbytes; ++i)
                if (buf[static_cast<size_t>(y) * static_cast<size_t>(view.rowbytes) + static_cast<size_t>(i)] != 0xCD) return false;
        return true;
    }
};

}  // namespace dms_test
