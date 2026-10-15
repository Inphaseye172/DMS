#include "temporal.h"

#include <cmath>
#include <cstring>

namespace dms {

int32_t effective_window(int32_t model_window, bool memopt) noexcept {
    int32_t w = model_window < 1 ? 1 : (model_window > 64 ? 64 : model_window);
    if (memopt) w = w / 2 > 0 ? w / 2 : 1;
    return w;
}

WindowSpan temporal_window(int64_t t, int32_t model_window, bool memopt) noexcept {
    const int32_t w = effective_window(model_window, memopt);
    WindowSpan s;
    if (w <= 1) {
        s.start = t;
        s.count = 1;
        return s;
    }
    const int32_t overlap = w / 4 > 0 ? w / 4 : 1;
    const int64_t stride = w - overlap;
    // floor division (works for negative frame indices too)
    const int64_t k = t >= 0 ? t / stride : -((-t + stride - 1) / stride);
    s.start = k * stride;
    s.count = w;
    return s;
}

void PlaneStore::put(const std::string& key, Plane p) {
    if (!p.data) return;
    const uint64_t sz = p.data->size() * sizeof(float) + key.size() + 64;
    std::lock_guard<std::mutex> lk(mu_);
    if (sz > budget_) return;
    auto it = index_.find(key);
    if (it != index_.end()) {
        used_ -= it->second->second.data->size() * sizeof(float) + key.size() + 64;
        lru_.erase(it->second);
        index_.erase(it);
    }
    while (used_ + sz > budget_ && !lru_.empty()) {
        const auto& b = lru_.back();
        used_ -= b.second.data->size() * sizeof(float) + b.first.size() + 64;
        index_.erase(b.first);
        lru_.pop_back();
    }
    lru_.emplace_front(key, std::move(p));
    index_[key] = lru_.begin();
    used_ += sz;
}

bool PlaneStore::get(const std::string& key, Plane& out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(key);
    if (it == index_.end()) return false;
    lru_.splice(lru_.begin(), lru_, it->second);
    out = it->second->second;
    return true;
}

void PlaneStore::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    lru_.clear();
    index_.clear();
    used_ = 0;
}

size_t PlaneStore::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return lru_.size();
}

void AffineAccumulator::add(const float* src, const float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(src[i]), y = static_cast<double>(dst[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        sx_ += x;
        sy_ += y;
        sxx_ += x * x;
        sxy_ += x * y;
        ++n_;
    }
}

bool AffineAccumulator::solve(Affine& out) const {
    if (n_ < 16) return false;
    const double n = static_cast<double>(n_);
    const double mx = sx_ / n, my = sy_ / n;
    const double var = sxx_ / n - mx * mx;
    const double cov = sxy_ / n - mx * my;
    if (!(var > 1e-12)) return false;
    const double s = cov / var;
    if (!(s > 0.05 && s < 20.0)) return false;  // reject implausible fits (scene cut, garbage)
    out.s = s;
    out.b = my - s * mx;
    return std::isfinite(out.b);
}

uint64_t frame_fingerprint(const float* rgb, int w, int h, int stride) noexcept {
    uint64_t hsh = 0xcbf29ce484222325ull ^ (static_cast<uint64_t>(static_cast<uint32_t>(w)) << 32) ^ static_cast<uint64_t>(static_cast<uint32_t>(h));
    const int step_y = h > 64 ? h / 64 : 1;
    const int step_x = w > 64 ? w / 64 : 1;
    for (int y = 0; y < h; y += step_y) {
        const float* row = rgb + static_cast<size_t>(y) * static_cast<size_t>(stride);
        for (int x = 0; x < w; x += step_x) {
            uint32_t bits[3];
            std::memcpy(bits, row + x * 3, sizeof(bits));
            for (uint32_t b : bits) {
                hsh ^= b;
                hsh *= 0x100000001b3ull;
            }
        }
    }
    return hsh;
}

}  // namespace dms
