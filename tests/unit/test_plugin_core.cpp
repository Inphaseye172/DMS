#include "fake_world.h"
#include "params_model.h"
#include "pixels.h"
#include "render_core.h"
#include "sequence_data.h"
#include "test_helpers.h"

#include <cmath>
#include <cstring>
#include <limits>

using namespace dms_plugin;
using dms_test::FakeWorld;

TEST(ParamsModel, SanitizeClampsAndRejectsNonFinite) {
    RawParamValues r;
    r.model_popup = 99;
    r.base_res_popup = 0;
    r.edges = std::numeric_limits<double>::quiet_NaN();
    r.smoothing = 1e9;
    r.mapping_popup = -3;
    r.near_x = std::numeric_limits<double>::infinity();
    r.near_value = 7;
    r.far_value = -2;
    r.slice_min = 0.9;
    r.slice_max = 0.1;
    r.samples = 1000;
    r.parallel = 1;
    r.tiles_popup = 42;
    r.cuda_memshare = 1;
    r.memopt = 1;
    const auto s = sanitize(r, 21, /*is_nvidia=*/false);
    EXPECT_EQ(s.model_index, 0);
    EXPECT_EQ(s.base_res, 518);
    EXPECT_EQ(s.edges, 0.f);
    EXPECT_EQ(s.smoothing, 100.f);
    EXPECT_EQ(s.mapping, Mapping::Normalize);
    EXPECT_EQ(s.near_x, 0.f);
    EXPECT_EQ(s.near_value, 1.f);
    EXPECT_EQ(s.far_value, 0.f);
    EXPECT_LE(s.slice_min, s.slice_max);
    EXPECT_EQ(s.samples, 16);
    EXPECT_EQ(s.parallel, 1);
    EXPECT_EQ(s.tiles, 0);
    EXPECT_EQ(s.cuda_memshare, 0);  // not NVIDIA
    EXPECT_EQ(s.memopt, 0);         // not a temporal model
}

TEST(ParamsModel, ParallelRequiresMoreThanTwoSamplesAndMemoptRequiresVideo) {
    RawParamValues r;
    r.samples = 2;
    r.parallel = 1;
    r.model_popup = 6;  // video
    r.memopt = 1;
    auto s = sanitize(r, 1, true);
    EXPECT_EQ(s.parallel, 0);
    EXPECT_EQ(s.memopt, 1);
    EXPECT_TRUE(model_is_temporal(s.model_index));
    const auto ui = compute_ui_state(r, true);
    EXPECT_FALSE(ui.enabled[P_PARALLEL]);
    EXPECT_TRUE(ui.enabled[P_MEMOPT]);
}

TEST(ParamsModel, UiStateShowsDependentParams) {
    RawParamValues r;
    r.mapping_popup = 3;
    r.mode_popup = 2;
    auto ui = compute_ui_state(r, false);
    EXPECT_TRUE(ui.visible[P_NEAR_POINT]);
    EXPECT_TRUE(ui.visible[P_FAR_VALUE]);
    EXPECT_FALSE(ui.visible[P_COLORMAP]);
    EXPECT_TRUE(ui.visible[P_SLICE_MIN]);
    EXPECT_FALSE(ui.visible[P_CUDA_MEMSHARE]);
    r.mapping_popup = 1;
    r.mode_popup = 1;
    ui = compute_ui_state(r, true);
    EXPECT_FALSE(ui.visible[P_NEAR_POINT]);
    EXPECT_TRUE(ui.visible[P_COLORMAP]);
    EXPECT_FALSE(ui.visible[P_SLICE_MAX]);
    EXPECT_TRUE(ui.visible[P_CUDA_MEMSHARE]);
}

TEST(ParamsModel, ParamOrderIsFrozen) {
    // Changing these values breaks saved projects. This test exists to make that a conscious decision.
    EXPECT_EQ(P_MODEL, 1);
    EXPECT_EQ(P_MAPPING, 5);
    EXPECT_EQ(P_MODE, 10);
    EXPECT_EQ(P_BACKEND_TOPIC, 14);
    EXPECT_EQ(P_MEMOPT, 22);
    EXPECT_EQ(P_NUM_PARAMS, 24);
    EXPECT_EQ(ID_BACKEND_TOPIC_END, 23);
}

TEST(SequenceData, RoundTripAndForeignVersions) {
    SequenceData a;
    init_sequence_data(a);
    a.flags = 0xABCD;
    SequenceData b;
    ASSERT_TRUE(upgrade_sequence_data(&a, sizeof(a), b));
    EXPECT_EQ(b.flags, 0xABCDu);
    EXPECT_NE(std::memcmp(a.instance_guid, b.instance_guid, 16), 0);  // copies get a fresh identity

    // Garbage, truncated, null.
    uint8_t junk[200];
    for (size_t i = 0; i < sizeof(junk); ++i) junk[i] = static_cast<uint8_t>(i * 13);
    EXPECT_FALSE(upgrade_sequence_data(junk, sizeof(junk), b));
    EXPECT_EQ(b.magic, kSeqMagic);
    EXPECT_FALSE(upgrade_sequence_data(nullptr, 0, b));
    EXPECT_FALSE(upgrade_sequence_data(&a, 5, b));

    // Older writer (smaller struct: header only) and newer writer (bigger struct).
    SequenceData older = a;
    older.size = 16;
    EXPECT_TRUE(upgrade_sequence_data(&older, 16, b));
    EXPECT_EQ(b.flags, 0u);
    std::vector<uint8_t> newer(400, 0);
    SequenceData nv = a;
    nv.size = 400;
    nv.version = 7;
    std::memcpy(newer.data(), &nv, sizeof(nv));
    EXPECT_TRUE(upgrade_sequence_data(newer.data(), newer.size(), b));
    EXPECT_EQ(b.flags, 0xABCDu);
    EXPECT_EQ(b.version, kSeqVersion);
    // Declared size larger than the actual buffer must not over-read.
    nv.size = 100000;
    EXPECT_TRUE(upgrade_sequence_data(&nv, sizeof(nv), b));
}

class PixelDepthTest : public ::testing::TestWithParam<PixelDepth> {};

TEST_P(PixelDepthTest, WorldToRgbCompositesStraightAlphaOverGray) {
    FakeWorld w(3, 2, GetParam());
    w.set(0, 0, 1.f, 1.f, 0.f, 0.f);
    w.set(1, 0, 0.f, 1.f, 1.f, 1.f);  // fully transparent: hidden RGB must not leak
    w.set(2, 0, 0.5f, 1.f, 1.f, 1.f);
    std::vector<float> rgb, alpha;
    ASSERT_TRUE(world_to_rgb(w.view, rgb, &alpha));
    EXPECT_NEAR(rgb[0], 1.f, 1e-3f);
    EXPECT_NEAR(rgb[1], 0.f, 1e-3f);
    EXPECT_NEAR(rgb[3], 0.5f, 1e-3f);
    EXPECT_NEAR(rgb[6], 0.75f, 5e-3f);
    EXPECT_NEAR(alpha[2], 0.5f, 5e-3f);
}

TEST_P(PixelDepthTest, RenderWritesExactlyTheOutputWorld) {
    dms_test::EngineHandle eh("render_core");
    FakeWorld in(37, 21, GetParam());  // odd sizes
    for (int y = 0; y < 21; ++y)
        for (int x = 0; x < 37; ++x) in.set(x, y, 1.f, 0.3f, 0.5f, 0.7f);
    // Sub-rect request: output covers layer rect [5,3)-(25,19).
    FakeWorld out(20, 16, GetParam());
    RenderJob job;
    job.params.hw_accel = 0;  // hermetic: CPU only, whatever GPU the test machine has
    job.frames = {in.view};
    job.center_index = 0;
    job.input_rect = {0, 0, 37, 21};
    job.output_rect = {5, 3, 25, 19};
    job.output = out.view;
    job.params.mapping = Mapping::Raw;
    const auto o = render(eh.e, job);
    ASSERT_TRUE(o.output_written) << o.message;
    EXPECT_TRUE(o.status == DMS_OK || o.status == DMS_ERR_DEGRADED) << o.message;
    EXPECT_TRUE(out.padding_intact());
    float a = 0, r = 0, g = 0, b = 0;
    out.get(0, 0, a, r, g, b);
    EXPECT_NEAR(a, 1.f, 1e-3f);
    EXPECT_NEAR(r, g, 1e-6f);  // grayscale
}

TEST_P(PixelDepthTest, ZeroSizeInputYieldsTransparentOutput) {
    dms_test::EngineHandle eh("render_core_zero");
    FakeWorld out(8, 4, GetParam());
    RenderJob job;
    job.params.hw_accel = 0;  // hermetic: CPU only, whatever GPU the test machine has
    WorldView empty;
    job.frames = {empty};
    job.output = out.view;
    const auto o = render(eh.e, job);
    EXPECT_TRUE(o.output_written);
    float a = 0, r = 0, g = 0, b = 0;
    out.get(3, 2, a, r, g, b);
    EXPECT_EQ(a, 0.f);
    EXPECT_TRUE(out.padding_intact());
}

INSTANTIATE_TEST_SUITE_P(AllDepths, PixelDepthTest, ::testing::Values(PixelDepth::U8, PixelDepth::U16, PixelDepth::F32, PixelDepth::F32_BGRA));

TEST(PluginCore, OutputOutsideInputIsTransparent) {
    dms_test::EngineHandle eh("render_core_outside");
    FakeWorld in(10, 10, PixelDepth::F32);
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 10; ++x) in.set(x, y, 1.f, 0.2f, 0.2f, 0.2f);
    FakeWorld out(10, 10, PixelDepth::F32);
    RenderJob job;
    job.params.hw_accel = 0;  // hermetic: CPU only, whatever GPU the test machine has
    job.frames = {in.view};
    job.input_rect = {0, 0, 10, 10};
    job.output_rect = {5, 5, 15, 15};  // half outside
    job.output = out.view;
    render(eh.e, job);
    float a = 0, r = 0, g = 0, b = 0;
    out.get(9, 9, a, r, g, b);
    EXPECT_EQ(a, 0.f);
    out.get(0, 0, a, r, g, b);
    EXPECT_GT(a, 0.f);
}

TEST(PluginCore, HashChangesWithContent) {
    FakeWorld a(16, 16, PixelDepth::U8), b(16, 16, PixelDepth::U8);
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            a.set(x, y, 1, 0.5f, 0.5f, 0.5f);
            b.set(x, y, 1, 0.5f, 0.5f, 0.5f);
        }
    EXPECT_EQ(hash_world(a.view), hash_world(b.view));  // padding bytes must not matter
    b.set(3, 7, 1, 0.5f, 0.6f, 0.5f);
    EXPECT_NE(hash_world(a.view), hash_world(b.view));
}

namespace {
void fill_gradient(FakeWorld& w) {
    for (int y = 0; y < w.view.height; ++y)
        for (int x = 0; x < w.view.width; ++x) {
            const float v = static_cast<float>(x) / static_cast<float>(w.view.width - 1);
            w.set(x, y, 1.f, v, v, v);
        }
}
}  // namespace

TEST(PluginCore, MissingModelRendersGrayAndReportsOnce) {
    dms_test::EngineHandle eh("render_missing");
    FakeWorld in(40, 20, PixelDepth::F32), out(40, 20, PixelDepth::F32);
    fill_gradient(in);
    RenderJob job;
    job.params.hw_accel = 0;  // hermetic: CPU only, whatever GPU the test machine has
    job.frames = {in.view};
    job.input_rect = {0, 0, 40, 20};
    job.output_rect = {0, 0, 40, 20};
    job.output = out.view;
    job.instance_guid[0] = 0x42;
    dms_instance_open(eh.e, job.instance_guid);
    auto o = render(eh.e, job);
    EXPECT_EQ(o.status, DMS_ERR_DEGRADED);
    EXPECT_TRUE(o.report_to_user);
    EXPECT_NE(o.message.find("not installed"), std::string::npos) << o.message;
    float a = 0, r = 0, g = 0, b = 0;
    out.get(10, 10, a, r, g, b);
    EXPECT_NEAR(r, 0.5f, 1e-3f);  // flat mid-gray, never garbage
    o = render(eh.e, job);
    EXPECT_FALSE(o.report_to_user);  // no dialog per frame
}

TEST(PluginCore, FailureAfterSuccessUsesLastGoodDepth) {
    dms_test::EngineHandle eh("render_lastgood");
    dms_test::install_synthetic_model(eh.models_dir, "da2-small", "synthetic_depth.onnx");  // stand-in for the real model
    dms_model_reload_manifest(eh.e);
    FakeWorld in(40, 20, PixelDepth::U16), out1(40, 20, PixelDepth::U16), out2(40, 20, PixelDepth::U16);
    fill_gradient(in);
    RenderJob job;
    job.params.hw_accel = 0;  // hermetic: CPU only, whatever GPU the test machine has
    job.frames = {in.view};
    job.input_rect = {0, 0, 40, 20};
    job.output_rect = {0, 0, 40, 20};
    job.output = out1.view;
    job.instance_guid[0] = 0x43;
    job.params.mapping = Mapping::Normalize;
    auto o = render(eh.e, job);
    ASSERT_EQ(o.status, DMS_OK) << o.message;
    // Model disappears (deleted / corrupted) and the cache is cleared: the instance's last good map is used.
    std::filesystem::remove(std::filesystem::path(eh.models_dir) / "synthetic_depth.onnx");
    dms_cache_clear(eh.e);
    job.output = out2.view;
    o = render(eh.e, job);
    EXPECT_EQ(o.status, DMS_ERR_DEGRADED);
    EXPECT_NE(o.message.find("last good"), std::string::npos) << o.message;
    float a1 = 0, r1 = 0, g1 = 0, b1 = 0, a2 = 0, r2 = 0, g2 = 0, b2 = 0;
    out1.get(30, 10, a1, r1, g1, b1);
    out2.get(30, 10, a2, r2, g2, b2);
    EXPECT_NEAR(r1, r2, 0.02f);
    EXPECT_TRUE(out2.padding_intact());
}
