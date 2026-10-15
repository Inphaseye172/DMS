#pragma once
#include "dms/engine.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

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
