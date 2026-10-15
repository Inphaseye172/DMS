// LRU cache of computed depth maps, bounded in bytes. Keys combine the layer content checksum and
// frame time with every parameter that changes the depth result. Thread-safe.
#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dms {

struct CacheKey {
    uint64_t layer_hash = 0;
    int64_t time_num = 0;
    int64_t time_den = 1;
    std::string model_id;
    int base_res = 0;
    int samples = 1;
    int tiles = 1;
    int edges_q = 0;      // edges * 10, rounded
    int smoothing_q = 0;  // smoothing * 10, rounded
    bool fp16 = false;
    int width = 0;
    int height = 0;

    bool operator==(const CacheKey& o) const {
        return layer_hash == o.layer_hash && time_num == o.time_num && time_den == o.time_den && model_id == o.model_id && base_res == o.base_res &&
               samples == o.samples && tiles == o.tiles && edges_q == o.edges_q && smoothing_q == o.smoothing_q && fp16 == o.fp16 && width == o.width &&
               height == o.height;
    }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const noexcept;
};

struct CacheEntry {
    std::shared_ptr<const std::vector<float>> depth;
    float raw_min = 0.f;
    float raw_max = 0.f;
    size_t bytes() const noexcept { return depth ? depth->size() * sizeof(float) + 256 : 256; }
};

struct CacheStats {
    uint64_t bytes_used = 0, bytes_budget = 0, entries = 0, hits = 0, misses = 0, evictions = 0;
};

class FrameCache {
public:
    // available_ram: returns currently available system RAM (injectable for tests).
    using RamProbe = std::function<uint64_t()>;
    explicit FrameCache(uint64_t budget_bytes, RamProbe available_ram = nullptr, uint64_t low_ram_threshold = 1ull << 30);

    bool get(const CacheKey& k, CacheEntry& out);
    void put(const CacheKey& k, CacheEntry e);
    void clear();
    void set_budget(uint64_t bytes);
    CacheStats stats() const;

private:
    void evict_to_locked(uint64_t target);
    using Item = std::pair<CacheKey, CacheEntry>;
    mutable std::mutex mu_;
    std::list<Item> lru_;  // front = most recent
    std::unordered_map<CacheKey, std::list<Item>::iterator, CacheKeyHash> index_;
    uint64_t budget_;
    uint64_t used_ = 0;
    uint64_t hits_ = 0, misses_ = 0, evictions_ = 0;
    RamProbe ram_;
    uint64_t low_ram_;
};

}  // namespace dms
