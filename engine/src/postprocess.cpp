#include "postprocess.h"

#include "common.h"
#include "image_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dms {

namespace {
inline float clamp01(float v) noexcept { return std::isfinite(v) ? (v < 0.f ? 0.f : (v > 1.f ? 1.f : v)) : 0.f; }
inline float smoothstep(float e0, float e1, float x) noexcept {
    if (e1 <= e0) return x >= e1 ? 1.f : 0.f;
    const float t = clamp01((x - e0) / (e1 - e0));
    return t * t * (3.f - 2.f * t);
}

// Guided filter core on single-channel guide I and input p (both normalized), writes q.
void guided_filter(const std::vector<float>& I, const std::vector<float>& p, int w, int h, int r, float eps, std::vector<float>& q) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<float> tmp(n), mI(n), mp(n), II(n), Ip(n), mII(n), mIp(n);
    for (size_t i = 0; i < n; ++i) {
        II[i] = I[i] * I[i];
        Ip[i] = I[i] * p[i];
    }
    box_blur(I.data(), mI.data(), tmp.data(), w, h, r);
    box_blur(p.data(), mp.data(), tmp.data(), w, h, r);
    box_blur(II.data(), mII.data(), tmp.data(), w, h, r);
    box_blur(Ip.data(), mIp.data(), tmp.data(), w, h, r);
    // Reuse buffers: a -> II, b -> Ip
    for (size_t i = 0; i < n; ++i) {
        const float var = mII[i] - mI[i] * mI[i];
        const float cov = mIp[i] - mI[i] * mp[i];
        const float a = cov / (var + eps);
        II[i] = a;
        Ip[i] = mp[i] - a * mI[i];
    }
    box_blur(II.data(), mII.data(), tmp.data(), w, h, r);  // mean a
    box_blur(Ip.data(), mIp.data(), tmp.data(), w, h, r);  // mean b
    q.resize(n);
    for (size_t i = 0; i < n; ++i) q[i] = mII[i] * I[i] + mIp[i];
}

// Normalizes depth to [0,1]; returns false if degenerate. Non-finite -> min.
bool normalized(const float* d, size_t n, std::vector<float>& out, float& mn, float& range) {
    float mx = 0.f;
    if (!finite_min_max(d, n, mn, mx) || !(mx > mn)) return false;
    range = mx - mn;
    out.resize(n);
    const float inv = 1.f / range;
    for (size_t i = 0; i < n; ++i) out[i] = std::isfinite(d[i]) ? (d[i] - mn) * inv : 0.f;
    return true;
}

// Depth maps are up-sampled from the model's working resolution (~518 px long side), so their edges are
// blurred over roughly (frame size / 518) pixels. Filter radii scale with that factor.
int radius_for(int w, int h, float s01, float base, float gain) {
    const float factor = std::max(1.f, static_cast<float>(std::max(w, h)) / 518.f);
    const float r = factor * (base + gain * s01);
    return clampi(static_cast<int>(std::lround(r)), 1, 64);
}
}  // namespace

float sample_depth(const float* depth, int w, int h, float x, float y) noexcept {
    if (!std::isfinite(x)) x = static_cast<float>(w) * 0.5f;
    if (!std::isfinite(y)) y = static_cast<float>(h) * 0.5f;
    const int cx = clampi(static_cast<int>(std::floor(x)), 0, w - 1);
    const int cy = clampi(static_cast<int>(std::floor(y)), 0, h - 1);
    double sum = 0.0;
    int count = 0;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            const int xx = clampi(cx + dx, 0, w - 1), yy = clampi(cy + dy, 0, h - 1);
            const float v = depth[static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)];
            if (std::isfinite(v)) {
                sum += static_cast<double>(v);
                ++count;
            }
        }
    return count ? static_cast<float>(sum / count) : 0.f;
}

void map_depth(const float* depth, int w, int h, const dms_mapping_params& p, float* out) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    switch (p.mode) {
        case DMS_MAP_RAW:
            for (size_t i = 0; i < n; ++i) out[i] = std::isfinite(depth[i]) ? depth[i] : 0.f;
            return;
        case DMS_MAP_GUIDED: {
            const float dn = sample_depth(depth, w, h, p.near_x, p.near_y);
            const float df = sample_depth(depth, w, h, p.far_x, p.far_y);
            const float vn = clamp01(p.near_value), vf = clamp01(p.far_value);
            float mn = 0.f, mx = 0.f;
            const bool have_range = finite_min_max(depth, n, mn, mx) && mx > mn;
            if (std::fabs(df - dn) > (have_range ? (mx - mn) * 1e-4f : 1e-12f)) {
                // Linear map through the two guide points, clamped to [0,1].
                const float k = (vf - vn) / (df - dn);
                for (size_t i = 0; i < n; ++i) out[i] = std::isfinite(depth[i]) ? clamp01(vn + (depth[i] - dn) * k) : vf;
            } else {
                // Degenerate (both points at the same depth): normalize, then span far..near values.
                for (size_t i = 0; i < n; ++i) {
                    const float t = (have_range && std::isfinite(depth[i])) ? (depth[i] - mn) / (mx - mn) : 0.5f;
                    out[i] = vf + (vn - vf) * clamp01(t);
                }
            }
            return;
        }
        case DMS_MAP_NORMALIZE:
        default: {
            float mn = p.raw_min, mx = p.raw_max;
            if (!(std::isfinite(mn) && std::isfinite(mx) && mx > mn)) {
                if (!finite_min_max(depth, n, mn, mx)) mn = mx = 0.f;
            }
            if (!(mx > mn)) {
                for (size_t i = 0; i < n; ++i) out[i] = 0.5f;
                return;
            }
            const float inv = 1.f / (mx - mn);
            for (size_t i = 0; i < n; ++i) out[i] = std::isfinite(depth[i]) ? clamp01((depth[i] - mn) * inv) : 0.f;
            return;
        }
    }
}

void slice_mask(const float* v, int w, int h, float mn, float mx, float feather, float* out) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    mn = clamp01(mn);
    mx = clamp01(mx);
    if (mx < mn) std::swap(mn, mx);
    feather = clamp01(feather);
    const float width = std::max(feather * std::max(mx - mn, 0.05f), 0.f);
    for (size_t i = 0; i < n; ++i) {
        const float x = std::isfinite(v[i]) ? v[i] : -1.f;
        if (width <= 0.f) {
            out[i] = (x >= mn && x <= mx) ? 1.f : 0.f;
        } else {
            out[i] = smoothstep(mn - width, mn, x) * (1.f - smoothstep(mx, mx + width, x));
        }
    }
}

void refine_edges(const float* depth, const float* rgb, int stride, int w, int h, float strength, float* out) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    const float s = clampf(strength, 0.f, 100.f) / 100.f;
    std::vector<float> p;
    float mn = 0.f, range = 1.f;
    if (s <= 0.f || !normalized(depth, n, p, mn, range) || w < 3 || h < 3) {
        if (out != depth) std::copy(depth, depth + n, out);
        return;
    }
    std::vector<float> I(n);
    for (int y = 0; y < h; ++y) {
        const float* row = rgb + static_cast<size_t>(y) * static_cast<size_t>(stride);
        for (int x = 0; x < w; ++x) {
            const float l = 0.2126f * row[x * 3] + 0.7152f * row[x * 3 + 1] + 0.0722f * row[x * 3 + 2];
            I[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = std::isfinite(l) ? l : 0.f;
        }
    }
    const int r = radius_for(w, h, s, 2.f, 4.f);
    const float eps = 1e-2f * (1.f - 0.95f * s) * (1.f - 0.95f * s);  // stronger -> follow image edges more tightly
    std::vector<float> q;
    guided_filter(I, p, w, h, r, eps, q);
    for (size_t i = 0; i < n; ++i) {
        const float refined = mn + std::clamp(q[i], -0.25f, 1.25f) * range;
        const float orig = std::isfinite(depth[i]) ? depth[i] : mn;
        out[i] = orig + (refined - orig) * s;
    }
}

void smooth_depth(const float* depth, int w, int h, float strength, float* out) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    const float s = clampf(strength, 0.f, 100.f) / 100.f;
    std::vector<float> p;
    float mn = 0.f, range = 1.f;
    if (s <= 0.f || !normalized(depth, n, p, mn, range) || w < 3 || h < 3) {
        if (out != depth) std::copy(depth, depth + n, out);
        return;
    }
    const int r = radius_for(w, h, s, 1.f, 3.f);
    const float e = 0.004f + 0.03f * s;  // edges larger than ~e (normalized depth) are preserved
    std::vector<float> q;
    guided_filter(p, p, w, h, r, e * e, q);
    for (size_t i = 0; i < n; ++i) {
        const float sm = mn + q[i] * range;
        const float orig = std::isfinite(depth[i]) ? depth[i] : mn;
        out[i] = orig + (sm - orig) * s;
    }
}

}  // namespace dms
