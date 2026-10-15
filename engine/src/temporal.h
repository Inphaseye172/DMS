// Temporal (video) depth support: window scheduling, a store of per-frame window results, and
// cross-window scale/shift alignment so consecutive windows produce flicker-free depth.
#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dms {

struct WindowSpan {
    int64_t start = 0;
    int32_t count = 1;
};

// Effective window length (memory optimization halves it), at least 1.
int32_t effective_window(int32_t model_window, bool memory_optimization) noexcept;
// Overlapping chunk schedule: stride = W - W/4.
WindowSpan temporal_window(int64_t frame_index, int32_t model_window, bool memory_optimization) noexcept;

// Small per-frame depth planes (pre post-processing) keyed by a string; LRU bounded in bytes.
class PlaneStore {
public:
    struct Plane {
        std::shared_ptr<const std::vector<float>> data;
        int w = 0, h = 0;
    };
    explicit PlaneStore(uint64_t budget_bytes) : budget_(budget_bytes) {}
    void put(const std::string& key, Plane p);
    bool get(const std::string& key, Plane& out);
    void clear();
    size_t size() const;

private:
    mutable std::mutex mu_;
    std::list<std::pair<std::string, Plane>> lru_;
    std::unordered_map<std::string, std::list<std::pair<std::string, Plane>>::iterator> index_;
    uint64_t budget_;
    uint64_t used_ = 0;
};

// Least-squares affine fit new ~ s*x + b mapping `src` onto `dst` (same size). Returns false if degenerate.
struct Affine {
    double s = 1.0, b = 0.0;
};
class AffineAccumulator {
public:
    void add(const float* src, const float* dst, size_t n);
    bool solve(Affine& out) const;
    size_t count() const { return n_; }

private:
    double sx_ = 0, sy_ = 0, sxx_ = 0, sxy_ = 0;
    size_t n_ = 0;
};

// Sub-sampled content hash of an interleaved RGB float frame (cheap anchor identity).
uint64_t frame_fingerprint(const float* rgb, int w, int h, int stride) noexcept;

}  // namespace dms
