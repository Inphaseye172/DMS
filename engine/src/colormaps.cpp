#include "colormaps.h"

#include "common.h"

#include <cmath>

namespace dms {
namespace {
#include "colormaps_data.inc"
}  // namespace

int32_t colormap_count() noexcept { return kColormapCount + 1; }  // + "None"

const char* colormap_name(int32_t index) noexcept {
    if (index == 0) return "None";
    if (index < 0 || index > kColormapCount) return "";
    return kColormapNames[index - 1];
}

void apply_colormap(const float* v, size_t n, int32_t index, float* rgb) noexcept {
    for (size_t i = 0; i < n; ++i) {
        float x = v[i];
        if (!std::isfinite(x)) x = 0.f;
        x = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
        if (index <= 0 || index > kColormapCount) {
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = x;  // grayscale, white = near
            continue;
        }
        const auto& lut = kColormapLut[index - 1];
        const int k = static_cast<int>(std::lround(x * 255.f));
        rgb[i * 3] = static_cast<float>(lut[k][0]) / 255.f;
        rgb[i * 3 + 1] = static_cast<float>(lut[k][1]) / 255.f;
        rgb[i * 3 + 2] = static_cast<float>(lut[k][2]) / 255.f;
    }
}

}  // namespace dms
