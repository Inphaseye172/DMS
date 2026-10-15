// Opt-in tests against the real Depth Anything v2 Small model.
// Enable with DMS_REAL_MODEL_DIR=<dir containing depth_anything_v2_vits_dynamic.onnx>.
#include "test_helpers.h"

#include <stb/stb_image.h>

#include <cstdlib>
#include <memory>
#include <numeric>

namespace {
std::vector<float> load_rgb(const std::string& path, int& w, int& h) {
    int n = 0;
    std::unique_ptr<unsigned char, void (*)(void*)> px(stbi_load(path.c_str(), &w, &h, &n, 3), stbi_image_free);
    std::vector<float> rgb;
    if (!px) return rgb;
    rgb.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<float>(px.get()[i]) / 255.f;
    return rgb;
}
}  // namespace

TEST(RealModel, DepthAnythingV2SmallOrdersFloorScene) {
    const char* dir = std::getenv("DMS_REAL_MODEL_DIR");
    if (!dir || !*dir) GTEST_SKIP() << "set DMS_REAL_MODEL_DIR to run";
    dms_engine_config c{};
    c.struct_size = sizeof(c);
    c.models_dir = dir;
    const auto logs = dms_test::tmp_dir("real_logs").string();
    c.log_dir = logs.c_str();
    dms_engine* e = nullptr;
    ASSERT_EQ(dms_engine_acquire(&c, &e), DMS_OK);
    struct R { dms_engine* e; ~R() { dms_engine_release(e); } } r{e};
    dms_model_info mi{};
    mi.struct_size = sizeof(mi);
    ASSERT_EQ(dms_model_info_by_id(e, "da2-small", &mi), DMS_OK);
    if (!mi.installed) GTEST_SKIP() << "da2-small not installed in " << dir;

    int w = 0, h = 0;
    auto rgb = load_rgb(dms_test::fixture("scene_floor.png").string(), w, h);
    ASSERT_FALSE(rgb.empty());
    dms_frame f{};
    f.struct_size = sizeof(f);
    f.rgb = rgb.data();
    f.width = w;
    f.height = h;
    f.row_stride_floats = w * 3;
    auto rq = dms_test::make_request("da2-small", &f, 1, 0);
    std::vector<float> depth;
    auto rs = dms_test::make_result(depth, w, h);
    ASSERT_EQ(dms_depth_compute(e, &rq, &rs), DMS_OK) << dms_last_error();
    auto band_mean = [&](double y0, double y1) {
        double s = 0;
        size_t n = 0;
        for (int y = static_cast<int>(h * y0); y < static_cast<int>(h * y1); ++y)
            for (int x = 0; x < w; ++x, ++n) s += static_cast<double>(depth[static_cast<size_t>(y * w + x)]);
        return s / static_cast<double>(n);
    };
    // Larger value = nearer: the floor at the bottom must be much nearer than the horizon.
    EXPECT_GT(band_mean(0.9, 1.0), 3.0 * band_mean(0.42, 0.5));
}
