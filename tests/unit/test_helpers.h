#pragma once
#include "dms/engine.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <cstdio>
#include <string>
#include <vector>

namespace dms_test {

inline std::filesystem::path tmp_dir(const std::string& name) {
    auto p = std::filesystem::path(DMS_TEST_TMP_DIR) / name;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p;
}

inline std::filesystem::path fixture(const std::string& name) { return std::filesystem::path(DMS_FIXTURES_DIR) / name; }

// RAII engine handle for tests.
struct EngineHandle {
    dms_engine* e = nullptr;
    std::string models_dir, log_dir;
    explicit EngineHandle(const std::string& tag, uint64_t cache_bytes = 0) {
        models_dir = tmp_dir(tag + "_models").string();
        log_dir = (std::filesystem::path(DMS_TEST_TMP_DIR) / "logs").string();
        dms_engine_config c{};
        c.struct_size = sizeof(c);
        c.models_dir = models_dir.c_str();
        c.log_dir = log_dir.c_str();
        c.log_level = DMS_LOG_DEBUG;
        c.cache_budget_bytes = cache_bytes;
        EXPECT_EQ(dms_engine_acquire(&c, &e), DMS_OK) << dms_last_error();
    }
    ~EngineHandle() {
        if (e) dms_engine_release(e);
    }
    EngineHandle(const EngineHandle&) = delete;
    EngineHandle& operator=(const EngineHandle&) = delete;
};

}  // namespace dms_test

namespace dms_test {

// Installs a synthetic model into `models_dir` under the given id via a models.json override.
inline void install_synthetic_model(const std::string& models_dir, const std::string& id, const std::string& fixture_file,
                                    const std::string& kind = "single", int window = 1) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::copy_file(fixture("onnx/" + fixture_file), fs::path(models_dir) / fixture_file, fs::copy_options::overwrite_existing, ec);
    const auto mf = fs::path(models_dir) / "models.json";
    std::string body = "{\"models\":[";
    // Preserve previously installed overrides (simple append by re-reading is unnecessary for tests: one call per id).
    std::FILE* f = std::fopen(mf.string().c_str(), "rb");
    std::string prev;
    if (f) {
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) prev.append(buf, n);
        std::fclose(f);
    }
    std::string entry = "{\"id\":\"" + id + "\",\"display_name\":\"" + id + "\",\"kind\":\"" + kind + "\",\"window\":" + std::to_string(window) +
                        ",\"patch_multiple\":14,\"file\":\"" + fixture_file + "\"" + (kind == "temporal" ? ",\"input_layout\":\"BTCHW\"" : "") + "}";
    if (!prev.empty()) {
        const size_t pos = prev.rfind("]}");
        body = prev.substr(0, pos) + "," + entry + "]}";
    } else {
        body += entry + "]}";
    }
    f = std::fopen(mf.string().c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fwrite(body.data(), 1, body.size(), f);
    std::fclose(f);
}

// Simple RGB test frame: horizontal brightness gradient (bright = right).
struct TestFrame {
    int w, h;
    std::vector<float> rgb;
    TestFrame(int w_, int h_) : w(w_), h(h_), rgb(static_cast<size_t>(w_) * static_cast<size_t>(h_) * 3u) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                    rgb[(static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 3u + static_cast<size_t>(c)] =
                        w > 1 ? static_cast<float>(x) / static_cast<float>(w - 1) : 0.5f;
    }
    dms_frame frame(int64_t t = 0) const {
        dms_frame f{};
        f.struct_size = sizeof(f);
        f.rgb = rgb.data();
        f.width = w;
        f.height = h;
        f.row_stride_floats = w * 3;
        f.time_num = t;
        f.time_den = 24;
        return f;
    }
};

inline dms_depth_request make_request(const char* model, const dms_frame* frames, int count, int center) {
    dms_depth_request rq{};
    rq.struct_size = sizeof(rq);
    rq.model_id = model;
    rq.frames = frames;
    rq.frame_count = count;
    rq.center_index = center;
    rq.base_resolution = 518;
    rq.backend.struct_size = sizeof(rq.backend);
    rq.backend.hardware_acceleration = 0;
    rq.backend.samples = 1;
    rq.backend.tiles = 1;
    rq.cache_key.struct_size = sizeof(rq.cache_key);
    return rq;
}

inline dms_depth_result make_result(std::vector<float>& buf, int w, int h) {
    buf.assign(static_cast<size_t>(w) * static_cast<size_t>(h), -1.f);
    dms_depth_result r{};
    r.struct_size = sizeof(r);
    r.depth = buf.data();
    r.width = w;
    r.height = h;
    return r;
}

}  // namespace dms_test
