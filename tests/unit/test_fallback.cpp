#include "backend_plan.h"
#include "engine_impl.h"
#include "test_helpers.h"

#include <atomic>
#include <mutex>
#include <set>

using namespace dms;
using namespace dms_test;

namespace {
GpuSnapshot nvidia(int driver_major = 576, uint64_t free = 20ull << 30) {
    GpuSnapshot g;
    g.vendor = DMS_GPU_NVIDIA;
    g.name = "Fake RTX";
    g.driver.major = driver_major;
    g.driver.minor = 2;
    g.vram_total = 24ull << 30;
    g.vram_free = free;
    g.has_dxgi_adapter = true;
    return g;
}
PlanInputs base_inputs() {
    PlanInputs p;
    p.gpu = nvidia();
    p.runtime_has_cuda = true;
    p.base_res = 1024;
    p.tiles = 1;
    return p;
}
}  // namespace

TEST(BackendPlan, NvidiaChainOrder) {
    PlanInputs p = base_inputs();
    p.fp16 = true;
    const auto plan = build_plan(p);
    ASSERT_GE(plan.size(), 6u);
    EXPECT_EQ(plan[0].backend, Backend::CUDA);
    EXPECT_TRUE(plan[0].fp16);
    EXPECT_EQ(plan[1].backend, Backend::CUDA);  // FP16 -> FP32
    EXPECT_FALSE(plan[1].fp16);
    EXPECT_EQ(plan[1].res, 1024);
    EXPECT_EQ(plan[2].res, 768);  // lower resolution ladder
    EXPECT_EQ(plan[3].res, 518);
    EXPECT_EQ(plan[4].res, 384);
    EXPECT_GT(plan[5].tiles, 1);  // more tiles
    EXPECT_LT(plan[5].res, 384);
    EXPECT_EQ(plan.back().backend, Backend::CPU);  // last inference rung
    EXPECT_EQ(plan.back().res, 1024);
}

TEST(BackendPlan, OldDriverSkipsCudaAndUsesDirectMLIfAvailable) {
    PlanInputs p = base_inputs();
    p.gpu = nvidia(470);
    auto plan = build_plan(p);
    for (const auto& a : plan) EXPECT_NE(a.backend, Backend::CUDA);
    p.runtime_has_dml = true;
    plan = build_plan(p);
    EXPECT_EQ(plan[0].backend, Backend::DirectML);
}

TEST(BackendPlan, HardwareOffOrStickyMeansCpuOnly) {
    PlanInputs p = base_inputs();
    p.hardware = false;
    auto plan = build_plan(p);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].backend, Backend::CPU);
    p.hardware = true;
    p.cuda_sticky_unavailable = true;
    plan = build_plan(p);
    ASSERT_EQ(plan.size(), 1u);
}

TEST(BackendPlan, ParallelSamplesFallBackToSequential) {
    PlanInputs p = base_inputs();
    p.parallel = true;
    const auto plan = build_plan(p);
    EXPECT_TRUE(plan[0].parallel);
    EXPECT_FALSE(plan[1].parallel);
    EXPECT_EQ(plan[1].res, plan[0].res);
}

TEST(BackendPlan, AutoTiles) {
    EXPECT_EQ(resolve_tiles(0, 1280, 720, 518), 1);
    EXPECT_EQ(resolve_tiles(0, 1920, 1080, 518), 4);
    EXPECT_EQ(resolve_tiles(0, 3840, 2160, 518), 16);
    EXPECT_EQ(resolve_tiles(0, 3840, 2160, 1024), 4);
    EXPECT_EQ(resolve_tiles(9, 10, 10, 518), 9);
    EXPECT_EQ(resolve_tiles(7, 100, 100, 518), 1);  // invalid -> auto
}

TEST(BackendPlan, VramEstimateGrowsWithResolutionAndBatch) {
    ModelSpec s;
    s.params_millions = 335;
    s.patch_multiple = 14;
    const uint64_t a = estimate_vram(s, 1'300'000'000, 518, 518, 1, false, false);
    const uint64_t b = estimate_vram(s, 1'300'000'000, 1022, 1022, 1, false, false);
    const uint64_t c = estimate_vram(s, 1'300'000'000, 518, 518, 4, false, false);
    const uint64_t d = estimate_vram(s, 1'300'000'000, 518, 518, 1, false, true);
    EXPECT_GT(b, a);
    EXPECT_GT(c, a);
    EXPECT_LT(d, a);  // resident weights are not counted again
}

namespace {
// Fault-injecting session factory: pretends GPU sessions exist (backed by CPU ORT).
struct FaultInjector {
    std::mutex mu;
    std::multiset<std::string> created;  // backend names of attempted creations
    dms_status cuda_failure = DMS_OK;    // thrown when a CUDA session is created
    bool fp16_nan = false;               // FP16 sessions produce NaN
    std::string nan_model;
    InferenceService::SessionFactory factory() {
        return [this](OrtRuntime& rt, const SessionConfig& cfg) {
            {
                std::lock_guard<std::mutex> lk(mu);
                created.insert(std::string(backend_name(cfg.backend)) + (cfg.fp16 ? "-fp16" : ""));
            }
            if (cfg.backend == Backend::CUDA && cuda_failure != DMS_OK) fail(cuda_failure, "injected CUDA failure (CUDA failure 2: out of memory)");
            SessionConfig c = cfg;
            c.backend = Backend::CPU;
            if (cfg.fp16 && fp16_nan) c.model_path = nan_model;
            return std::make_unique<ModelSession>(rt, c);
        };
    }
    size_t count(const std::string& k) {
        std::lock_guard<std::mutex> lk(mu);
        return created.count(k);
    }
};

struct FallbackFixture : public ::testing::Test {
    std::unique_ptr<EngineHandle> eh;
    std::shared_ptr<Engine> engine;
    FakeGpuMonitor* gpu = nullptr;
    FaultInjector inj;
    void SetUp() override {
        eh = std::make_unique<EngineHandle>("fallback");
        install_synthetic_model(eh->models_dir, "synthetic", "synthetic_depth.onnx");
        dms_model_reload_manifest(eh->e);
        engine = lock_engine(eh->e);
        auto g = std::make_unique<FakeGpuMonitor>(nvidia());
        gpu = g.get();
        engine->set_gpu_monitor(std::move(g));
        engine->inference().set_provider_override({"CUDAExecutionProvider", "CPUExecutionProvider"});
        engine->inference().release_sessions();
        inj.nan_model = fixture("onnx/nan_depth.onnx").string();
        engine->inference().set_session_factory(inj.factory());
    }
    void TearDown() override {
        engine.reset();
        eh.reset();
    }
    dms_status run(dms_depth_result& rs, std::vector<float>& depth, bool fp16 = false) {
        static TestFrame tf(48, 32);
        static const dms_frame f = tf.frame();
        auto rq = make_request("synthetic", &f, 1, 0);
        rq.backend.hardware_acceleration = 1;
        rq.backend.fp16 = fp16;
        rs = make_result(depth, tf.w, tf.h);
        return dms_depth_compute(eh->e, &rq, &rs);
    }
};
}  // namespace

TEST_F(FallbackFixture, CudaWorksWhenHealthy) {
    dms_depth_result rs;
    std::vector<float> d;
    ASSERT_EQ(run(rs, d), DMS_OK) << dms_last_error();
    EXPECT_EQ(rs.backend_used, DMS_BACKEND_CUDA);
    EXPECT_EQ(rs.fallback_steps, 0);
}

TEST_F(FallbackFixture, CudaOomFallsBackToCpuWithMessage) {
    inj.cuda_failure = DMS_ERR_OUT_OF_MEMORY;
    dms_depth_result rs;
    std::vector<float> d;
    ASSERT_EQ(run(rs, d), DMS_ERR_DEGRADED) << dms_last_error();
    EXPECT_EQ(rs.backend_used, DMS_BACKEND_CPU);
    EXPECT_GT(rs.fallback_steps, 0);
    EXPECT_NE(std::string(rs.message).find("out of memory"), std::string::npos);
    for (float v : d) ASSERT_TRUE(std::isfinite(v));  // never garbage
    // OOM memo: the next frame skips the failed CUDA rungs instead of retrying them every frame.
    const size_t before = inj.count("CUDA");
    ASSERT_EQ(run(rs, d), DMS_ERR_DEGRADED);
    EXPECT_EQ(inj.count("CUDA"), before);
}

TEST_F(FallbackFixture, LowFreeVramSkipsGpuWithoutCreatingSessions) {
    gpu->set_free(512ull << 20);  // below the 1 GB reserve
    dms_depth_result rs;
    std::vector<float> d;
    ASSERT_EQ(run(rs, d), DMS_ERR_DEGRADED);
    EXPECT_EQ(rs.backend_used, DMS_BACKEND_CPU);
    EXPECT_EQ(inj.count("CUDA"), 0u);
    EXPECT_NE(std::string(rs.message).find("out of memory"), std::string::npos);
}

TEST_F(FallbackFixture, InsufficientDriverIsStickyAndSilentAfterFirstFrame) {
    inj.cuda_failure = DMS_ERR_BACKEND_UNAVAILABLE;  // e.g. cudaErrorInsufficientDriver at session creation
    dms_depth_result rs;
    std::vector<float> d;
    ASSERT_EQ(run(rs, d), DMS_ERR_DEGRADED);
    EXPECT_EQ(inj.count("CUDA"), 1u);
    ASSERT_EQ(run(rs, d), DMS_OK);  // CUDA no longer planned: plain CPU result
    EXPECT_EQ(inj.count("CUDA"), 1u);
    EXPECT_EQ(rs.backend_used, DMS_BACKEND_CPU);
}

TEST_F(FallbackFixture, Fp16NonFiniteOutputFallsBackToFp32) {
    inj.fp16_nan = true;
    dms_depth_result rs;
    std::vector<float> d;
    ASSERT_EQ(run(rs, d, /*fp16=*/true), DMS_ERR_DEGRADED) << dms_last_error();
    EXPECT_EQ(rs.backend_used, DMS_BACKEND_CUDA);
    EXPECT_EQ(rs.fallback_steps, 1);
    EXPECT_EQ(inj.count("CUDA-fp16"), 1u);
    EXPECT_EQ(inj.count("CUDA"), 1u);
    for (float v : d) ASSERT_TRUE(std::isfinite(v));
}

TEST_F(FallbackFixture, OldDriverNeverTriesCudaAndMemShareIsDisabled) {
    engine->set_gpu_monitor(std::make_unique<FakeGpuMonitor>(nvidia(470)));
    static TestFrame tf(48, 32);
    static const dms_frame f = tf.frame();
    auto rq = make_request("synthetic", &f, 1, 0);
    rq.backend.hardware_acceleration = 1;
    rq.backend.cuda_memory_sharing = 1;
    std::vector<float> d;
    auto rs = make_result(d, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh->e, &rq, &rs), DMS_OK);
    EXPECT_EQ(rs.backend_used, DMS_BACKEND_CPU);
    EXPECT_EQ(inj.count("CUDA"), 0u);
}

TEST_F(FallbackFixture, SamplesParallelAndSequentialAgree) {
    static TestFrame tf(70, 42);
    static const dms_frame f = tf.frame();
    auto rq = make_request("synthetic", &f, 1, 0);
    rq.backend.hardware_acceleration = 0;
    rq.backend.samples = 4;
    rq.backend.parallel_samples = 0;
    std::vector<float> seq, par;
    auto r1 = make_result(seq, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh->e, &rq, &r1), DMS_OK) << dms_last_error();
    rq.backend.parallel_samples = 1;
    auto r2 = make_result(par, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh->e, &rq, &r2), DMS_OK) << dms_last_error();
    for (size_t i = 0; i < seq.size(); ++i) ASSERT_NEAR(seq[i], par[i], 1e-4f);
}
