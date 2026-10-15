#include "postprocess.h"
#include "test_helpers.h"

#include <cmath>
#include <limits>
#include <numeric>

using namespace dms;

namespace {
std::vector<float> ramp(int w, int h, float lo, float hi) {
    std::vector<float> d(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) d[static_cast<size_t>(y * w + x)] = lo + (hi - lo) * static_cast<float>(x) / static_cast<float>(w - 1);
    return d;
}
dms_mapping_params params(int mode) {
    dms_mapping_params p{};
    p.struct_size = sizeof(p);
    p.mode = mode;
    p.near_value = 1.f;
    p.far_value = 0.f;
    return p;
}
}  // namespace

TEST(Mapping, RawCopiesAndScrubsNonFinite) {
    auto d = ramp(16, 4, 2.f, 9.f);
    d[3] = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> o(d.size());
    map_depth(d.data(), 16, 4, params(DMS_MAP_RAW), o.data());
    EXPECT_EQ(o[0], 2.f);
    EXPECT_EQ(o[15], 9.f);
    EXPECT_EQ(o[3], 0.f);
}

TEST(Mapping, NormalizeSpansZeroToOneWhiteIsNear) {
    auto d = ramp(32, 2, 3.f, 7.f);  // larger raw = nearer (relative inverse depth)
    std::vector<float> o(d.size());
    map_depth(d.data(), 32, 2, params(DMS_MAP_NORMALIZE), o.data());
    EXPECT_FLOAT_EQ(o[0], 0.f);
    EXPECT_FLOAT_EQ(o[31], 1.f);
    // Precomputed range is honoured.
    auto p = params(DMS_MAP_NORMALIZE);
    p.raw_min = 3.f;
    p.raw_max = 11.f;
    map_depth(d.data(), 32, 2, p, o.data());
    EXPECT_NEAR(o[31], 0.5f, 1e-6f);
    // Constant map -> mid gray, not NaN.
    std::vector<float> c(64, 4.f), oc(64);
    map_depth(c.data(), 8, 8, params(DMS_MAP_NORMALIZE), oc.data());
    for (float v : oc) EXPECT_EQ(v, 0.5f);
    // In-place.
    map_depth(d.data(), 32, 2, params(DMS_MAP_NORMALIZE), d.data());
    EXPECT_FLOAT_EQ(d[31], 1.f);
}

TEST(Mapping, GuidedMapsGuidePointsToTheirValues) {
    const int w = 101, h = 11;
    auto d = ramp(w, h, 0.f, 10.f);
    auto p = params(DMS_MAP_GUIDED);
    p.near_x = 80;
    p.near_y = 5;
    p.far_x = 20;
    p.far_y = 5;
    p.near_value = 0.9f;
    p.far_value = 0.1f;
    std::vector<float> o(d.size());
    map_depth(d.data(), w, h, p, o.data());
    EXPECT_NEAR(o[static_cast<size_t>(5 * w + 80)], 0.9f, 1e-3f);
    EXPECT_NEAR(o[static_cast<size_t>(5 * w + 20)], 0.1f, 1e-3f);
    EXPECT_NEAR(o[static_cast<size_t>(5 * w + 50)], 0.5f, 1e-3f);
    EXPECT_EQ(o[static_cast<size_t>(5 * w + 100)], 1.f);  // clamped
    EXPECT_EQ(o[static_cast<size_t>(5 * w + 0)], 0.f);
}

TEST(Mapping, GuidedDegenerateAndHostileInputs) {
    const int w = 50, h = 10;
    auto d = ramp(w, h, 1.f, 2.f);
    auto p = params(DMS_MAP_GUIDED);
    p.near_x = p.far_x = 25;  // same point -> degenerate
    p.near_y = p.far_y = 5;
    std::vector<float> o(d.size());
    map_depth(d.data(), w, h, p, o.data());
    for (float v : o) ASSERT_TRUE(std::isfinite(v) && v >= 0.f && v <= 1.f);
    EXPECT_GT(o[49], o[0]);  // still near=bright
    p.near_x = std::numeric_limits<float>::quiet_NaN();
    p.far_x = 1e30f;  // far outside -> clamped to edge
    p.far_y = -1e30f;
    map_depth(d.data(), w, h, p, o.data());
    for (float v : o) ASSERT_TRUE(std::isfinite(v));
    EXPECT_EQ(dms_map_depth(d.data(), w, h, nullptr, o.data()), DMS_ERR_INVALID_ARG);
    p.mode = 7;
    EXPECT_EQ(dms_map_depth(d.data(), w, h, &p, o.data()), DMS_ERR_INVALID_ARG);
}

TEST(Slicing, HardAndFeatheredMask) {
    auto v = ramp(101, 1, 0.f, 1.f);
    std::vector<float> m(v.size());
    slice_mask(v.data(), 101, 1, 0.3f, 0.6f, 0.f, m.data());
    EXPECT_EQ(m[29], 0.f);
    EXPECT_EQ(m[30], 1.f);
    EXPECT_EQ(m[60], 1.f);
    EXPECT_EQ(m[61], 0.f);
    slice_mask(v.data(), 101, 1, 0.3f, 0.6f, 0.5f, m.data());
    EXPECT_EQ(m[45], 1.f);
    EXPECT_GT(m[25], 0.f);
    EXPECT_LT(m[25], 1.f);  // soft edge
    EXPECT_EQ(m[0], 0.f);
    slice_mask(v.data(), 101, 1, 0.6f, 0.3f, 0.f, m.data());  // swapped bounds are tolerated
    EXPECT_EQ(m[45], 1.f);
}

TEST(EdgesAndSmoothing, ZeroStrengthIsIdentityAndOutputStaysFinite) {
    const int w = 64, h = 48;
    auto d = ramp(w, h, 0.f, 5.f);
    std::vector<float> rgb(static_cast<size_t>(w * h * 3), 0.5f), o(d.size());
    refine_edges(d.data(), rgb.data(), w * 3, w, h, 0.f, o.data());
    EXPECT_EQ(o, d);
    smooth_depth(d.data(), w, h, 0.f, o.data());
    EXPECT_EQ(o, d);
    d[100] = std::numeric_limits<float>::infinity();
    refine_edges(d.data(), rgb.data(), w * 3, w, h, 100.f, o.data());
    for (float v : o) ASSERT_TRUE(std::isfinite(v));
    smooth_depth(d.data(), w, h, 100.f, o.data());
    for (float v : o) ASSERT_TRUE(std::isfinite(v));
}

TEST(EdgesAndSmoothing, EdgeEnhancementSnapsDepthEdgeToImageEdge) {
    // Image has a sharp vertical edge at x=40; the depth edge is blurred and offset (like an up-sampled map).
    const int w = 96, h = 32;
    std::vector<float> rgb(static_cast<size_t>(w * h * 3)), d(static_cast<size_t>(w * h));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float img = x < 40 ? 0.1f : 0.9f;
            for (int c = 0; c < 3; ++c) rgb[static_cast<size_t>((y * w + x) * 3 + c)] = img;
            d[static_cast<size_t>(y * w + x)] = 1.f / (1.f + std::exp(-(static_cast<float>(x) - 44.f) / 4.f));  // soft, shifted
        }
    std::vector<float> o(d.size());
    refine_edges(d.data(), rgb.data(), w * 3, w, h, 100.f, o.data());
    const int y = 16;
    const float before = std::fabs(d[static_cast<size_t>(y * w + 41)] - d[static_cast<size_t>(y * w + 38)]);
    const float after = std::fabs(o[static_cast<size_t>(y * w + 41)] - o[static_cast<size_t>(y * w + 38)]);
    EXPECT_GT(after, before * 1.5f);  // edge became sharper at the image edge
}

TEST(EdgesAndSmoothing, SmoothingRemovesStairStepsButKeepsLargeEdges) {
    const int w = 120, h = 20;
    std::vector<float> d(static_cast<size_t>(w * h));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float v = std::floor(static_cast<float>(x) / 6.f) * 0.01f;  // small stair steps
            if (x >= 60) v += 1.f;                                     // large real depth edge
            d[static_cast<size_t>(y * w + x)] = v;
        }
    std::vector<float> o(d.size());
    smooth_depth(d.data(), w, h, 100.f, o.data());
    const int y = 10;
    // Stair step at x=30 (6-px boundary) gets smoother.
    const float step_before = d[static_cast<size_t>(y * w + 30)] - d[static_cast<size_t>(y * w + 29)];
    const float step_after = o[static_cast<size_t>(y * w + 30)] - o[static_cast<size_t>(y * w + 29)];
    EXPECT_LT(step_after, step_before);
    // Large edge preserved (most of the 1.0 jump within a few pixels).
    EXPECT_GT(o[static_cast<size_t>(y * w + 63)] - o[static_cast<size_t>(y * w + 56)], 0.8f);
}

TEST(EngineMapping, EdgesAndSmoothingAreAppliedAndCachedSeparately) {
    dms_test::EngineHandle eh("engine_post");
    dms_test::install_synthetic_model(eh.models_dir, "synthetic", "synthetic_depth.onnx");
    dms_model_reload_manifest(eh.e);
    dms_test::TestFrame tf(64, 40);
    const dms_frame f = tf.frame();
    auto rq = dms_test::make_request("synthetic", &f, 1, 0);
    rq.cache_key.layer_hash = 77;
    std::vector<float> a, b;
    auto r1 = dms_test::make_result(a, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &r1), DMS_OK);
    rq.subpixel_smoothing = 80.f;
    rq.edges_enhancement = 50.f;
    auto r2 = dms_test::make_result(b, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &r2), DMS_OK);
    EXPECT_EQ(r2.from_cache, 0);
    for (float v : b) ASSERT_TRUE(std::isfinite(v));
    rq.subpixel_smoothing = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(dms_depth_compute(eh.e, &rq, &r2), DMS_ERR_INVALID_ARG);
}
