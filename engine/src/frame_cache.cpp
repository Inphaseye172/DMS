#include "frame_cache.h"

#include "logging.h"

namespace dms {

size_t CacheKeyHash::operator()(const CacheKey& k) const noexcept {
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&h](uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix(k.layer_hash);
    mix(static_cast<uint64_t>(k.time_num));
    mix(static_cast<uint64_t>(k.time_den));
    mix(std::hash<std::string>{}(k.model_id));
    mix(static_cast<uint64_t>(k.base_res));
    mix(static_cast<uint64_t>(k.samples) << 8 | static_cast<uint64_t>(k.tiles));
    mix(static_cast<uint64_t>(static_cast<uint32_t>(k.edges_q)) << 32 | static_cast<uint32_t>(k.smoothing_q));
    mix(static_cast<uint64_t>(k.fp16));
    mix(static_cast<uint64_t>(k.width) << 32 | static_cast<uint32_t>(k.height));
    return static_cast<size_t>(h);
}

FrameCache::FrameCache(uint64_t budget, RamProbe ram, uint64_t low_ram) : budget_(budget), ram_(std::move(ram)), low_ram_(low_ram) {}

bool FrameCache::get(const CacheKey& k, CacheEntry& out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(k);
    if (it == index_.end()) {
        ++misses_;
        return false;
    }
    lru_.splice(lru_.begin(), lru_, it->second);
    out = it->second->second;
    ++hits_;
    return true;
}

void FrameCache::evict_to_locked(uint64_t target) {
    while (used_ > target && !lru_.empty()) {
        auto& back = lru_.back();
        used_ -= back.second.bytes();
        index_.erase(back.first);
        lru_.pop_back();
        ++evictions_;
    }
}

void FrameCache::put(const CacheKey& k, CacheEntry e) {
    if (!e.depth) return;
    const uint64_t sz = e.bytes();
    std::lock_guard<std::mutex> lk(mu_);
    if (sz > budget_) return;  // never let one frame blow the budget
    auto it = index_.find(k);
    if (it != index_.end()) {
        used_ -= it->second->second.bytes();
        lru_.erase(it->second);
        index_.erase(it);
    }
    uint64_t target = budget_ - sz;
    // Memory pressure: when the machine is low on RAM, shrink to half the budget.
    if (ram_) {
        const uint64_t avail = ram_();
        if (avail > 0 && avail < low_ram_) {
            target = budget_ / 2 > sz ? budget_ / 2 - sz : 0;
            logf(DMS_LOG_DEBUG, "frame cache: low system memory (%llu MB available), trimming", static_cast<unsigned long long>(avail >> 20));
        }
    }
    evict_to_locked(target);
    lru_.emplace_front(k, std::move(e));
    index_[k] = lru_.begin();
    used_ += sz;
}

void FrameCache::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    lru_.clear();
    index_.clear();
    used_ = 0;
}

void FrameCache::set_budget(uint64_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    budget_ = bytes;
    evict_to_locked(budget_);
}

CacheStats FrameCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    CacheStats s;
    s.bytes_used = used_;
    s.bytes_budget = budget_;
    s.entries = lru_.size();
    s.hits = hits_;
    s.misses = misses_;
    s.evictions = evictions_;
    return s;
}

}  // namespace dms
