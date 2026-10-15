#include "engine_impl.h"
#include "test_helpers.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

using namespace dms_test;

TEST(EngineInference, SyntheticModelProducesExpectedDepth) {
    EngineHandle eh("inf_synth");
    install_synthetic_model(eh.models_dir, "synthetic", "synthetic_depth.onnx");
    ASSERT_EQ(dms_model_reload_manifest(eh.e), DMS_OK);
    dms_model_info mi{};
    mi.struct_size = sizeof(mi);
    ASSERT_EQ(dms_model_info_by_id(eh.e, "synthetic", &mi), DMS_OK);
    EXPECT_EQ(mi.installed, 1);

    TestFrame tf(97, 53);  // odd size, not a multiple of 14
    const dms_frame f = tf.frame();
    auto rq = make_request("synthetic", &f, 1, 0);
    std::vector<float> depth;
    auto rs = make_result(depth, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_OK) << dms_last_error();
    EXPECT_EQ(rs.backend_used, DMS_BACKEND_CPU);
    // Synthetic model = mean of normalized channels: monotonic in brightness => left < right.
    for (float v : depth) ASSERT_TRUE(std::isfinite(v));
    const float left = depth[static_cast<size_t>(26 * tf.w + 2)];
    const float right = depth[static_cast<size_t>(26 * tf.w + tf.w - 3)];
    EXPECT_LT(left, right);
    EXPECT_LT(rs.raw_min, rs.raw_max);
}

TEST(EngineInference, ExplicitModelPathWorks) {
    EngineHandle eh("inf_path");
    TestFrame tf(64, 64);
    const dms_frame f = tf.frame();
    const std::string path = fixture("onnx/synthetic_depth.onnx").string();
    auto rq = make_request(path.c_str(), &f, 1, 0);
    std::vector<float> depth;
    auto rs = make_result(depth, tf.w, tf.h);
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_OK) << dms_last_error();
}

TEST(EngineInference, MissingModelIsReportedNotCrashed) {
    EngineHandle eh("inf_missing");
    TestFrame tf(32, 32);
    const dms_frame f = tf.frame();
    auto rq = make_request("da2-small", &f, 1, 0);
    std::vector<float> depth;
    auto rs = make_result(depth, tf.w, tf.h);
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_MODEL_NOT_INSTALLED);
    EXPECT_NE(std::string(dms_last_error()).find("not installed"), std::string::npos);
    rq.model_id = "no-such-model";
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_NOT_FOUND);
    for (float v : depth) EXPECT_EQ(v, -1.f);  // output untouched on error
}

TEST(EngineInference, CorruptOrIncompatibleModelFailsCleanly) {
    EngineHandle eh("inf_bad");
    install_synthetic_model(eh.models_dir, "badrank", "bad_rank.onnx");
    dms_model_reload_manifest(eh.e);
    TestFrame tf(32, 32);
    const dms_frame f = tf.frame();
    auto rq = make_request("badrank", &f, 1, 0);
    std::vector<float> depth;
    auto rs = make_result(depth, tf.w, tf.h);
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_MODEL_LOAD_FAILED);
    // Garbage bytes as a model.
    const auto junk = std::filesystem::path(eh.models_dir) / "junk.onnx";
    { std::FILE* fp = std::fopen(junk.string().c_str(), "wb"); std::fputs("this is not an onnx model", fp); std::fclose(fp); }
    const std::string jp = junk.string();
    rq.model_id = jp.c_str();
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_MODEL_LOAD_FAILED);
}

TEST(EngineInference, InvalidRequestsAreRejected) {
    EngineHandle eh("inf_invalid");
    install_synthetic_model(eh.models_dir, "synthetic", "synthetic_depth.onnx");
    dms_model_reload_manifest(eh.e);
    TestFrame tf(32, 16);
    dms_frame f = tf.frame();
    std::vector<float> depth;
    auto rs = make_result(depth, tf.w, tf.h);
    auto rq = make_request("synthetic", &f, 1, 0);
    rq.center_index = 3;
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_INVALID_ARG);
    rq.center_index = 0;
    f.row_stride_floats = 5;  // too small
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_INVALID_ARG);
    f = tf.frame();
    f.rgb = nullptr;
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_INVALID_ARG);
    f = tf.frame();
    rs.width = 31;  // result buffer mismatch
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_INVALID_ARG);
    rs.width = tf.w;
    f.width = 0;
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_INVALID_ARG);
    f = tf.frame();
    rq.struct_size = 0;
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_ERR_INVALID_ARG);
}

TEST(EngineInference, AbortCallbackStopsWaiting) {
    EngineHandle eh("inf_abort");
    install_synthetic_model(eh.models_dir, "synthetic", "synthetic_depth.onnx");
    dms_model_reload_manifest(eh.e);
    TestFrame tf(64, 64);
    const dms_frame f = tf.frame();
    auto rq = make_request("synthetic", &f, 1, 0);
    rq.abort_fn = [](void*) -> int { return 1; };
    std::vector<float> depth;
    auto rs = make_result(depth, tf.w, tf.h);
    const dms_status st = dms_depth_compute(eh.e, &rq, &rs);
    // Either the job was abandoned before running (ABORTED) or it finished first (OK) — never a crash/hang.
    EXPECT_TRUE(st == DMS_ERR_ABORTED || st == DMS_OK) << dms_status_message(st);
}

TEST(EngineInference, ConcurrentRendersAreSerializedSafely) {
    EngineHandle eh("inf_concurrent");
    install_synthetic_model(eh.models_dir, "synthetic", "synthetic_depth.onnx");
    dms_model_reload_manifest(eh.e);
    std::atomic<int> ok{0}, bad{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&, t] {
            TestFrame tf(40 + t * 7, 30 + t * 3);
            const dms_frame f = tf.frame();
            for (int i = 0; i < 10; ++i) {
                auto rq = make_request("synthetic", &f, 1, 0);
                rq.base_resolution = (i % 2) ? 384 : 518;
                std::vector<float> depth;
                auto rs = make_result(depth, tf.w, tf.h);
                if (dms_depth_compute(eh.e, &rq, &rs) == DMS_OK) ++ok;
                else ++bad;
            }
        });
    }
    for (auto& t : ts) t.join();
    EXPECT_EQ(bad.load(), 0);
    EXPECT_EQ(ok.load(), 80);
}
