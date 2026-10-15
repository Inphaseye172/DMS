#include "test_helpers.h"

#include <thread>
#include <vector>

TEST(EngineLifecycle, AcquireReleaseRefcount) {
    dms_engine* a = nullptr;
    dms_engine* b = nullptr;
    dms_engine_config c{};
    c.struct_size = sizeof(c);
    const auto logs = dms_test::tmp_dir("lifecycle_logs").string();
    c.log_dir = logs.c_str();
    ASSERT_EQ(dms_engine_acquire(&c, &a), DMS_OK);
    ASSERT_EQ(dms_engine_acquire(&c, &b), DMS_OK);
    EXPECT_EQ(a, b);  // process-wide singleton
    EXPECT_EQ(dms_engine_release(a), DMS_OK);
    EXPECT_EQ(dms_engine_set_log_level(b, DMS_LOG_DEBUG), DMS_OK);  // still alive
    EXPECT_EQ(dms_engine_release(b), DMS_OK);
    // After the last release the handle is dead and rejected without crashing.
    EXPECT_EQ(dms_engine_set_log_level(b, DMS_LOG_DEBUG), DMS_ERR_NOT_INITIALIZED);
    EXPECT_EQ(dms_engine_release(b), DMS_ERR_NOT_INITIALIZED);
}

TEST(EngineLifecycle, NullAndGarbageArgumentsAreRejected) {
    EXPECT_EQ(dms_engine_acquire(nullptr, nullptr), DMS_ERR_INVALID_ARG);
    dms_engine_config bad{};  // struct_size = 0
    dms_engine* e = nullptr;
    EXPECT_EQ(dms_engine_acquire(&bad, &e), DMS_ERR_INVALID_ARG);
    EXPECT_EQ(e, nullptr);
    int not_an_engine = 0;
    EXPECT_EQ(dms_engine_release(reinterpret_cast<dms_engine*>(&not_an_engine)), DMS_ERR_NOT_INITIALIZED);
    EXPECT_EQ(dms_depth_compute(nullptr, nullptr, nullptr), DMS_ERR_INVALID_ARG);
    EXPECT_STRNE(dms_status_message(DMS_ERR_DEGRADED), nullptr);
}

TEST(EngineLifecycle, ConcurrentAcquireRelease) {
    const auto logs = dms_test::tmp_dir("lifecycle_conc_logs").string();
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&] {
            for (int i = 0; i < 50; ++i) {
                dms_engine_config c{};
                c.struct_size = sizeof(c);
                c.log_dir = logs.c_str();
                dms_engine* e = nullptr;
                if (dms_engine_acquire(&c, &e) == DMS_OK) {
                    dms_log(e, DMS_LOG_DEBUG, "concurrent");
                    dms_engine_release(e);
                }
            }
        });
    }
    for (auto& t : ts) t.join();
    SUCCEED();
}
