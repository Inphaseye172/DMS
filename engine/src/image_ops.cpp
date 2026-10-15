#include "image_ops.h"

#include <algorithm>
#include <cmath>

namespace dms {

namespace {
struct Tap {
    int i0, i1;
    float w1;
};
inline Tap tap(int d, int dn, int sn, float shift) noexcept {
    const float scale = static_cast<float>(sn) / static_cast<float>(dn);
    float s = (static_cast<float>(d) + 0.5f) * scale - 0.5f + shift;
    if (s < 0.f) s = 0.f;
    const float maxs = static_cast<float>(sn - 1);
    if (s > maxs) s = maxs;
    const int i0 = static_cast<int>(s);
    const int i1 = i0 + 1 < sn ? i0 + 1 : sn - 1;
    return {i0, i1, s - static_cast<float>(i0)};
}
}  // namespace

void resize_bilinear(const float* src, int sw, int sh, float* dst, int dw, int dh) noexcept {
    if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    if (sw == dw && sh == dh) {
        std::copy(src, src + static_cast<size_t>(sw) * static_cast<size_t>(sh), dst);
        return;
    }
    std::vector<Tap> xs(static_cast<size_t>(dw));
    for (int x = 0; x < dw; ++x) xs[static_cast<size_t>(x)] = tap(x, dw, sw, 0.f);
    for (int y = 0; y < dh; ++y) {
        const Tap ty = tap(y, dh, sh, 0.f);
        const float* r0 = src + static_cast<size_t>(ty.i0) * static_cast<size_t>(sw);
        const float* r1 = src + static_cast<size_t>(ty.i1) * static_cast<size_t>(sw);
        float* out = dst + static_cast<size_t>(y) * static_cast<size_t>(dw);
        for (int x = 0; x < dw; ++x) {
            const Tap& tx = xs[static_cast<size_t>(x)];
            const float a = r0[tx.i0] + (r0[tx.i1] - r0[tx.i0]) * tx.w1;
            const float b = r1[tx.i0] + (r1[tx.i1] - r1[tx.i0]) * tx.w1;
            out[x] = a + (b - a) * ty.w1;
        }
    }
}

void resize_rgb_to_nchw_normalized(const float* rgb, int sw, int sh, int stride, float* dst, int dw, int dh, bool flip_x,
                                   float shift_x, float shift_y) noexcept {
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kInvStd[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f};
    if (!rgb || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    const size_t plane = static_cast<size_t>(dw) * static_cast<size_t>(dh);
    std::vector<Tap> xs(static_cast<size_t>(dw));
    for (int x = 0; x < dw; ++x) xs[static_cast<size_t>(x)] = tap(x, dw, sw, shift_x);
    for (int y = 0; y < dh; ++y) {
        const Tap ty = tap(y, dh, sh, shift_y);
        const float* r0 = rgb + static_cast<size_t>(ty.i0) * static_cast<size_t>(stride);
        const float* r1 = rgb + static_cast<size_t>(ty.i1) * static_cast<size_t>(stride);
        for (int x = 0; x < dw; ++x) {
            const Tap& tx = xs[static_cast<size_t>(x)];
            const int ox = flip_x ? (dw - 1 - x) : x;
            const size_t o = static_cast<size_t>(y) * static_cast<size_t>(dw) + static_cast<size_t>(ox);
            for (int c = 0; c < 3; ++c) {
                const float p00 = r0[tx.i0 * 3 + c], p01 = r0[tx.i1 * 3 + c];
                const float p10 = r1[tx.i0 * 3 + c], p11 = r1[tx.i1 * 3 + c];
                const float a = p00 + (p01 - p00) * tx.w1;
                const float b = p10 + (p11 - p10) * tx.w1;
                float v = a + (b - a) * ty.w1;
                if (!std::isfinite(v)) v = 0.f;
                v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                dst[static_cast<size_t>(c) * plane + o] = (v - kMean[c]) * kInvStd[c];
            }
        }
    }
}

int round_to_multiple(int v, int m) noexcept {
    if (m <= 1) return v < 1 ? 1 : v;
    int r = ((v + m / 2) / m) * m;
    return r < m ? m : r;
}

void inference_dims(int w, int h, int base, int multiple, int& ow, int& oh) noexcept {
    if (w <= 0 || h <= 0) {
        ow = oh = round_to_multiple(base, multiple);
        return;
    }
    const double s = static_cast<double>(base) / static_cast<double>(std::max(w, h));
    ow = round_to_multiple(static_cast<int>(std::lround(w * s)), multiple);
    oh = round_to_multiple(static_cast<int>(std::lround(h * s)), multiple);
}

bool finite_min_max(const float* p, size_t n, float& mn, float& mx) noexcept {
    bool any = false;
    mn = 0.f;
    mx = 0.f;
    for (size_t i = 0; i < n; ++i) {
        const float v = p[i];
        if (!std::isfinite(v)) continue;
        if (!any) {
            mn = mx = v;
            any = true;
        } else {
            mn = v < mn ? v : mn;
            mx = v > mx ? v : mx;
        }
    }
    return any;
}

void sanitize(float* p, size_t n, float value) noexcept {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) p[i] = value;
}

void flip_x_inplace(float* p, int w, int h) noexcept {
    for (int y = 0; y < h; ++y) std::reverse(p + static_cast<size_t>(y) * static_cast<size_t>(w), p + static_cast<size_t>(y + 1) * static_cast<size_t>(w));
}

void box_blur(const float* src, float* dst, float* tmp, int w, int h, int r) noexcept {
    if (r <= 0) {
        if (dst != src) std::copy(src, src + static_cast<size_t>(w) * static_cast<size_t>(h), dst);
        return;
    }
    // Horizontal pass: src -> tmp (running sum, edge-clamped).
    for (int y = 0; y < h; ++y) {
        const float* row = src + static_cast<size_t>(y) * static_cast<size_t>(w);
        float* out = tmp + static_cast<size_t>(y) * static_cast<size_t>(w);
        double acc = 0.0;
        for (int k = -r; k <= r; ++k) acc += static_cast<double>(row[std::clamp(k, 0, w - 1)]);
        const double inv = 1.0 / (2 * r + 1);
        for (int x = 0; x < w; ++x) {
            out[x] = static_cast<float>(acc * inv);
            acc += static_cast<double>(row[std::min(x + r + 1, w - 1)]) - static_cast<double>(row[std::max(x - r, 0)]);
        }
    }
    // Vertical pass: tmp -> dst.
    const double inv = 1.0 / (2 * r + 1);
    std::vector<double> acc(static_cast<size_t>(w), 0.0);
    for (int k = -r; k <= r; ++k) {
        const float* row = tmp + static_cast<size_t>(std::clamp(k, 0, h - 1)) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) acc[static_cast<size_t>(x)] += static_cast<double>(row[x]);
    }
    for (int y = 0; y < h; ++y) {
        float* out = dst + static_cast<size_t>(y) * static_cast<size_t>(w);
        const float* add = tmp + static_cast<size_t>(std::min(y + r + 1, h - 1)) * static_cast<size_t>(w);
        const float* sub = tmp + static_cast<size_t>(std::max(y - r, 0)) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            out[x] = static_cast<float>(acc[static_cast<size_t>(x)] * inv);
            acc[static_cast<size_t>(x)] += static_cast<double>(add[x]) - static_cast<double>(sub[x]);
        }
    }
}

}  // namespace dms
