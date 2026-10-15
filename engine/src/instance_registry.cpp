#include "instance_registry.h"

#include "image_ops.h"

#include <algorithm>
#include <cmath>

namespace dms {

void InstanceRegistry::open(const Guid& g) {
    std::lock_guard<std::mutex> lk(mu_);
    if (map_.size() >= kMaxInstances && !map_.count(g)) map_.erase(map_.begin());
    map_.try_emplace(g);
}

void InstanceRegistry::close(const Guid& g) {
    std::lock_guard<std::mutex> lk(mu_);
    map_.erase(g);
}

void InstanceRegistry::store(const Guid& g, const float* depth, int w, int h) {
    if (!depth || w <= 0 || h <= 0) return;
    int sw = w, sh = h;
    if (std::max(w, h) > kMaxStoredSide) {
        const double s = static_cast<double>(kMaxStoredSide) / std::max(w, h);
        sw = std::max(1, static_cast<int>(std::lround(w * s)));
        sh = std::max(1, static_cast<int>(std::lround(h * s)));
    }
    auto v = std::make_shared<std::vector<float>>(static_cast<size_t>(sw) * static_cast<size_t>(sh));
    resize_bilinear(depth, w, h, v->data(), sw, sh);
    std::lock_guard<std::mutex> lk(mu_);
    if (map_.size() >= kMaxInstances && !map_.count(g)) map_.erase(map_.begin());
    Instance& in = map_[g];
    in.depth = std::move(v);
    in.w = sw;
    in.h = sh;
    in.reported.clear();
}

bool InstanceRegistry::last(const Guid& g, float* out, int w, int h) const {
    std::shared_ptr<const std::vector<float>> d;
    int sw = 0, sh = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(g);
        if (it == map_.end() || !it->second.depth) return false;
        d = it->second.depth;
        sw = it->second.w;
        sh = it->second.h;
    }
    resize_bilinear(d->data(), sw, sh, out, w, h);
    return true;
}

bool InstanceRegistry::should_report(const Guid& g, int code) {
    std::lock_guard<std::mutex> lk(mu_);
    if (map_.size() >= kMaxInstances && !map_.count(g)) map_.erase(map_.begin());
    return map_[g].reported.insert(code).second;
}

size_t InstanceRegistry::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return map_.size();
}

}  // namespace dms
