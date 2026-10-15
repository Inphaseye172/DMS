#include "pipeline.h"

#include "image_ops.h"

#include <vector>

namespace dms {

Augment augment_for_sample(int k, int n) noexcept {
    Augment a;
    if (n <= 1 || k == 0) return a;
    a.flip = (k % 2) == 1;
    // Sub-pixel jitter pattern (inference pixels), cycling through quarter-pixel offsets.
    static constexpr float kOffsets[8][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.25f, 0.25f}, {0.25f, 0.25f}, {-0.25f, 0.25f}, {-0.25f, 0.25f}, {0.25f, -0.25f}, {0.25f, -0.25f}};
    const int i = k % 8;
    a.shift_x = kOffsets[i][0];
    a.shift_y = kOffsets[i][1];
    if (k >= 8) {  // samples 8..15: half-pixel pattern
        a.shift_x *= 2.f;
        a.shift_y *= 2.f;
    }
    return a;
}

void preprocess_crop(const FrameView& f, const Crop& c, int iw, int ih, const Augment& a, float* dst) {
    const float* origin = f.rgb + static_cast<size_t>(c.y0) * static_cast<size_t>(f.stride) + static_cast<size_t>(c.x0) * 3u;
    // Shift is expressed in inference pixels; convert to source pixels for sampling.
    const float sx = a.shift_x * static_cast<float>(c.width()) / static_cast<float>(iw);
    const float sy = a.shift_y * static_cast<float>(c.height()) / static_cast<float>(ih);
    resize_rgb_to_nchw_normalized(origin, c.width(), c.height(), f.stride, dst, iw, ih, a.flip, sx, sy);
}

void deaugment(float* depth, int w, int h, const Augment& a) {
    if (a.flip) flip_x_inplace(depth, w, h);
    if (a.shift_x == 0.f && a.shift_y == 0.f) return;
    // The model saw the image sampled at +shift; resample its output at -shift to realign.
    std::vector<float> tmp(depth, depth + static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        float fy = static_cast<float>(y) - a.shift_y;
        fy = fy < 0.f ? 0.f : (fy > static_cast<float>(h - 1) ? static_cast<float>(h - 1) : fy);
        const int y0 = static_cast<int>(fy);
        const int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
        const float wy = fy - static_cast<float>(y0);
        for (int x = 0; x < w; ++x) {
            float fx = static_cast<float>(x) - a.shift_x;  // flip already undone above
            fx = fx < 0.f ? 0.f : (fx > static_cast<float>(w - 1) ? static_cast<float>(w - 1) : fx);
            const int x0 = static_cast<int>(fx);
            const int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
            const float wx = fx - static_cast<float>(x0);
            const float* r0 = tmp.data() + static_cast<size_t>(y0) * static_cast<size_t>(w);
            const float* r1 = tmp.data() + static_cast<size_t>(y1) * static_cast<size_t>(w);
            const float top = r0[x0] + (r0[x1] - r0[x0]) * wx;
            const float bot = r1[x0] + (r1[x1] - r1[x0]) * wx;
            depth[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = top + (bot - top) * wy;
        }
    }
}

}  // namespace dms
