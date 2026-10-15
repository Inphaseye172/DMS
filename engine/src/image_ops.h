// Pure CPU image helpers (no allocation unless documented).
#pragma once
#include <cstdint>
#include <vector>

namespace dms {

// Bilinear resize, single channel, pixel-center aligned (align_corners = false).
void resize_bilinear(const float* src, int sw, int sh, float* dst, int dw, int dh) noexcept;
// Bilinear resize of an interleaved RGB image with row stride (floats) into a planar NCHW tensor slice,
// applying ImageNet normalization. dst holds 3*dw*dh floats.
void resize_rgb_to_nchw_normalized(const float* rgb, int sw, int sh, int src_stride, float* dst, int dw, int dh,
                                   bool flip_x, float shift_x, float shift_y) noexcept;
// Rounds v to the nearest multiple of m (at least m).
int round_to_multiple(int v, int m) noexcept;
// Computes inference dims so the long side == base (rounded to multiple), aspect preserved.
void inference_dims(int w, int h, int base, int multiple, int& out_w, int& out_h) noexcept;
// Min/max over finite values; returns false if none finite.
bool finite_min_max(const float* p, size_t n, float& mn, float& mx) noexcept;
// Replaces NaN/Inf with `value`.
void sanitize(float* p, size_t n, float value) noexcept;
// Horizontal flip in place, single channel.
void flip_x_inplace(float* p, int w, int h) noexcept;
// Box blur (separable, radius r) using an integral approach; tmp must have w*h floats.
void box_blur(const float* src, float* dst, float* tmp, int w, int h, int r) noexcept;

}  // namespace dms
