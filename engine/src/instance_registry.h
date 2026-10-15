// Per effect-instance state owned by the engine (the plugin keeps only a GUID in its flat sequence data):
// the last good depth map (fallback when inference fails) and de-duplication of user-facing messages.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace dms {

class InstanceRegistry {
public:
    using Guid = std::array<uint8_t, 16>;
    static constexpr int kMaxStoredSide = 1024;  // last-good maps are stored downsampled to bound memory
    static constexpr size_t kMaxInstances = 4096;

    void open(const Guid& g);
    void close(const Guid& g);
    // Stores a copy (downsampled if large). Resets message de-duplication for the instance.
    void store(const Guid& g, const float* depth, int w, int h);
    // Resizes the stored map into out (w*h). False if none.
    bool last(const Guid& g, float* out, int w, int h) const;
    // True the first time `code` is reported since the last successful store.
    bool should_report(const Guid& g, int code);
    size_t size() const;

private:
    struct Instance {
        std::shared_ptr<const std::vector<float>> depth;
        int w = 0, h = 0;
        std::set<int> reported;
    };
    mutable std::mutex mu_;
    std::map<Guid, Instance> map_;
};

}  // namespace dms
