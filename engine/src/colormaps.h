#pragma once
#include <cstddef>
#include <cstdint>

namespace dms {
int32_t colormap_count() noexcept;  // including index 0 = "None"
const char* colormap_name(int32_t index) noexcept;
// v: n mapped values in [0,1]; rgb: 3n floats. index 0 (or out of range) = grayscale, white = near.
void apply_colormap(const float* v, size_t n, int32_t index, float* rgb) noexcept;
}  // namespace dms
