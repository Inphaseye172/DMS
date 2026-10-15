#include "engine_impl.h"
#include "fake_world.h"
#include "render_core.h"
#include "temporal.h"
#include "test_helpers.h"

#include <cmath>
#include <set>

using namespace dms;
using namespace dms_test;

TEST(TemporalWindow, OverlappingChunksCoverEveryFrame) {
    for (int W : {32, 16, 8, 5}) {
        for (bool memopt : {false, true}) {
            const int eff = effective_window(W, memopt);
            EXPECT_EQ(eff, memopt ? W / 2 : W);
            for (int64_t t = 0; t < 200; ++t) {
                const WindowSpan s = temporal_window(t, W, memopt);
                EXPECT_EQ(s.count, eff);
                EXPECT_LE(s.start, t);
                EXPECT_LT(t, s.start + s.count);  // the frame is inside its window
                // Consecutive windows overlap so they can be aligned: the window that renders this window's
                // last frame starts later but before this window ends.
                const WindowSpan n = temporal_window(s.start + s.count - 1, W, memopt);
                if (eff > 1) {
                    EXPECT_GT(n.start, s.start);
                    EXPECT_LT(n.start, s.start + s.count);
                }
            }
        }
    }
    const WindowSpan neg = temporal_window(-3, 8, false);
    EXPECT_LE(neg.start, -3);
    EXPECT_EQ(temporal_window(17, 1, false).count, 1);
}

TEST(TemporalWindow, AffineFitRecoversScaleAndShift) {
    std::vector<float> a(400), b(400);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i % 37) * 0.1f;
        b[i] = 2.5f * a[i] + 0.75f;
    }
    AffineAccumulator acc;
    acc.add(a.data(), b.data(), a.size());
    Affine af;
    ASSERT_TRUE(acc.solve(af));
    EXPECT_NEAR(af.s, 2.5, 1e-6);
    EXPECT_NEAR(af.b, 0.75, 1e-6);
    AffineAccumulator flat;
    std::vector<float> c(400, 1.f);
    flat.add(c.data(), b.data(), c.size());
    EXPECT_FALSE(flat.solve(af));  // degenerate
}

namespace {
// Sequence whose brightness rises over time (so the synthetic temporal model's window mean shifts).
struct Seq {
    int w = 42, h = 28, n = 24;
    std::vector<std::vector<float>> rgb;
    Seq() {
        for (int t = 0; t < n; ++t) {
            std::vector<float> f(static_cast<size_t>(w * h * 3));
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    for (int c = 0; c < 3; ++c)
                        f[static_cast<size_t>((y * w + x) * 3 + c)] = 0.1f + 0.02f * static_cast<float>(t) + 0.3f * static_cast<float>(x) / static_cast<float>(w);
            rgb.push_back(std::move(f));
        }
    }
    dms_frame frame(int t) const {
        dms_frame f{};
        f.struct_size = sizeof(f);
        f.rgb = rgb[static_cast<size_t>(t)].data();
        f.width = w;
        f.height = h;
        f.row_stride_floats = w * 3;
        f.time_num = t;
        f.time_den = 24;
        return f;
    }
};
}  // namespace

TEST(TemporalEngine, WindowSharesInferenceAndAlignsAcrossWindows) {
    EngineHandle eh("temporal_engine");
    install_synthetic_model(eh.models_dir, "video-test", "synthetic_temporal.onnx", "temporal", 8);
    ASSERT_EQ(dms_model_reload_manifest(eh.e), DMS_OK);
    Seq seq;
    std::vector<float> center_values;
    int inferences = 0;
    for (int t = 0; t < seq.n; ++t) {
        int64_t start = 0;
        int32_t count = 0;
        ASSERT_EQ(dms_temporal_window(eh.e, "video-test", t, 0, &start, &count), DMS_OK);
        EXPECT_EQ(count, 8);
        const int lo = static_cast<int>(std::max<int64_t>(0, start));
        const int hi = static_cast<int>(std::min<int64_t>(seq.n, start + count));
        std::vector<dms_frame> frames;
        uint64_t window_hash = uint64_t{1469598103934665603u};
        for (int k = lo; k < hi; ++k) {
            frames.push_back(seq.frame(k));
            window_hash = (window_hash ^ static_cast<uint64_t>(k + 1)) * uint64_t{1099511628211u};  // caller-defined window identity
        }
        auto rq = make_request("video-test", frames.data(), static_cast<int>(frames.size()), t - lo);
        rq.cache_key.layer_hash = window_hash;
        rq.cache_key.time_num = t;
        rq.cache_key.time_den = 24;
        std::vector<float> d;
        auto rs = make_result(d, seq.w, seq.h);
        ASSERT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_OK) << dms_last_error();
        if (!rs.from_cache) ++inferences;
        center_values.push_back(d[static_cast<size_t>(seq.h / 2 * seq.w + seq.w / 2)]);
    }
    // stride = 8 - 2 = 6 => frames 0..23 need windows starting at 0, 6, 12, 18 => 4 inferences, not 24.
    EXPECT_EQ(inferences, 4);
    // Flicker check: the frame-to-frame change across window boundaries stays close to the in-window change.
    double inside = 0, worst_boundary = 0;
    int n_inside = 0;
    for (int t = 1; t < seq.n; ++t) {
        const double step = std::fabs(static_cast<double>(center_values[static_cast<size_t>(t)] - center_values[static_cast<size_t>(t - 1)]));
        if (t % 6 == 0) worst_boundary = std::max(worst_boundary, step);
        else { inside += step; ++n_inside; }
    }
    inside /= n_inside;
    EXPECT_LT(worst_boundary, 2.0 * inside) << "boundary jump " << worst_boundary << " vs typical " << inside;
}

TEST(TemporalEngine, MemoryOptimizationHalvesTheWindow) {
    EngineHandle eh("temporal_memopt");
    install_synthetic_model(eh.models_dir, "video-test", "synthetic_temporal.onnx", "temporal", 8);
    dms_model_reload_manifest(eh.e);
    int64_t s = 0;
    int32_t c = 0;
    ASSERT_EQ(dms_temporal_window(eh.e, "video-test", 10, 1, &s, &c), DMS_OK);
    EXPECT_EQ(c, 4);
    ASSERT_EQ(dms_temporal_window(eh.e, "synthetic-not-there", 10, 1, &s, &c), DMS_ERR_NOT_FOUND);
}

TEST(TemporalEngine, GuideFrameDrivesOutputSize) {
    EngineHandle eh("temporal_guide");
    install_synthetic_model(eh.models_dir, "video-test", "synthetic_temporal.onnx", "temporal", 4);
    dms_model_reload_manifest(eh.e);
    Seq small;  // 42x28 model frames
    TestFrame big(168, 112);
    std::vector<dms_frame> frames = {small.frame(0), small.frame(1), small.frame(2)};
    const dms_frame guide = big.frame(1);
    auto rq = make_request("video-test", frames.data(), 3, 1);
    rq.guide = &guide;
    rq.edges_enhancement = 50.f;
    std::vector<float> d;
    auto rs = make_result(d, big.w, big.h);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &rs), DMS_OK) << dms_last_error();
    for (float v : d) ASSERT_TRUE(std::isfinite(v));
    auto bad = make_result(d, small.w, small.h);  // result must match the guide, not the frames
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &bad), DMS_ERR_INVALID_ARG);
}

TEST(TemporalPlugin, RenderCoreDownscalesWindowAndUsesGuide) {
    EngineHandle eh("temporal_plugin");
    install_synthetic_model(eh.models_dir, "video-da-small", "synthetic_temporal.onnx", "temporal", 4);  // stand-in
    dms_model_reload_manifest(eh.e);
    using namespace dms_plugin;
    std::vector<std::unique_ptr<FakeWorld>> worlds;
    RenderJob job;
    for (int k = 0; k < 4; ++k) {
        worlds.push_back(std::make_unique<FakeWorld>(1300, 30, PixelDepth::U8));  // wider than 2*518 => downscaled
        for (int y = 0; y < 30; ++y)
            for (int x = 0; x < 1300; ++x) worlds.back()->set(x, y, 1.f, static_cast<float>(x) / 1300.f, 0.3f, 0.2f + 0.1f * static_cast<float>(k));
        job.frames.push_back(worlds.back()->view);
    }
    job.frames.push_back(WorldView{});  // a missing neighbour (outside the layer) is dropped safely
    FakeWorld out(1300, 30, PixelDepth::U8);
    job.params.model_index = 5;  // "video-da-small"
    job.center_index = 2;
    job.input_rect = {0, 0, 1300, 30};
    job.output_rect = {0, 0, 1300, 30};
    job.output = out.view;
    job.time_num = 2002;
    job.time_den = 24000;
    job.time_step = 1001;
    const auto o = render(eh.e, job);
    EXPECT_EQ(o.status, DMS_OK) << o.message;
    EXPECT_TRUE(out.padding_intact());
}
