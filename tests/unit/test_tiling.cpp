#include "test_helpers.h"

#include <atomic>
#include <cmath>

using namespace dms_test;

namespace {
// Smooth structured image: diagonal gradient plus low-frequency waves (every tile has variance).
struct Pattern {
    int w, h;
    std::vector<float> rgb;
    Pattern(int w_, int h_) : w(w_), h(h_), rgb(static_cast<size_t>(w_ * h_ * 3)) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float fx = static_cast<float>(x) / static_cast<float>(w), fy = static_cast<float>(y) / static_cast<float>(h);
                const float v = 0.15f + 0.5f * (0.6f * fx + 0.4f * fy) + 0.1f * std::sin(fx * 9.f) * std::cos(fy * 7.f);
                for (int c = 0; c < 3; ++c) rgb[static_cast<size_t>((y * w + x) * 3 + c)] = v;
            }
    }
    dms_frame frame() const {
        dms_frame f{};
        f.struct_size = sizeof(f);
        f.rgb = rgb.data();
        f.width = w;
        f.height = h;
        f.row_stride_floats = w * 3;
        return f;
    }
};

std::vector<float> run(dms_engine* e, const Pattern& p, int tiles, dms_status* st_out = nullptr, dms_abort_fn abort_fn = nullptr,
                       dms_progress_fn progress_fn = nullptr, void* user = nullptr) {
    const dms_frame f = p.frame();
    auto rq = make_request("norm", &f, 1, 0);
    rq.base_resolution = 140;
    rq.backend.tiles = tiles;
    rq.abort_fn = abort_fn;
    rq.progress_fn = progress_fn;
    rq.user = user;
    std::vector<float> d;
    auto rs = make_result(d, p.w, p.h);
    const dms_status st = dms_depth_compute(e, &rq, &rs);
    if (st_out) *st_out = st;
    return d;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
}  // namespace

class TilingTest : public ::testing::TestWithParam<int> {};

TEST_P(TilingTest, TiledMatchesSinglePassWithoutSeams) {
    EngineHandle eh("tiling");
    install_synthetic_model(eh.models_dir, "norm", "synthetic_normalizing.onnx");
    dms_model_reload_manifest(eh.e);
    const Pattern p(480, 360);
    dms_status st = DMS_OK;
    const auto single = run(eh.e, p, 1, &st);
    ASSERT_EQ(st, DMS_OK) << dms_last_error();
    const auto tiled = run(eh.e, p, GetParam(), &st);
    ASSERT_EQ(st, DMS_OK) << dms_last_error();
    float mn = 1e9f, mx = -1e9f;
    for (float v : single) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    const float range = mx - mn;
    // Each tile is standardized separately by the model; only alignment brings it back onto the global frame.
    EXPECT_LT(max_abs_diff(single, tiled), 0.05f * range);
    // Seam check: horizontal second differences are uniformly small (no step anywhere).
    float worst = 0.f;
    for (int y = 2; y < p.h - 2; y += 3)
        for (int x = 1; x < p.w - 1; ++x) {
            const size_t i = static_cast<size_t>(y * p.w + x);
            worst = std::max(worst, std::fabs(tiled[i + 1] - 2.f * tiled[i] + tiled[i - 1]));
        }
    EXPECT_LT(worst, 0.02f * range);
}

INSTANTIATE_TEST_SUITE_P(Grids, TilingTest, ::testing::Values(4, 9, 16));

TEST(Tiling, TinyFrameFallsBackToSinglePass) {
    EngineHandle eh("tiling_tiny");
    install_synthetic_model(eh.models_dir, "norm", "synthetic_normalizing.onnx");
    dms_model_reload_manifest(eh.e);
    const Pattern p(60, 40);
    dms_status st = DMS_OK;
    const auto a = run(eh.e, p, 16, &st);
    EXPECT_EQ(st, DMS_OK) << dms_last_error();
    const auto b = run(eh.e, p, 1, &st);
    EXPECT_LT(max_abs_diff(a, b), 1e-5f);
}

TEST(Tiling, AbortBetweenTilesAndProgressReporting) {
    EngineHandle eh("tiling_abort");
    install_synthetic_model(eh.models_dir, "norm", "synthetic_normalizing.onnx");
    dms_model_reload_manifest(eh.e);
    const Pattern p(480, 360);
    struct State {
        std::atomic<int> calls{0};
        int abort_after = 1 << 30;
        float last_progress = 0.f;
        bool monotonic = true;
    } s;
    auto abort_fn = [](void* u) -> int {
        auto* st = static_cast<State*>(u);
        return ++st->calls > st->abort_after ? 1 : 0;
    };
    auto progress_fn = [](void* u, float f) -> int {
        auto* st = static_cast<State*>(u);
        if (f + 1e-6f < st->last_progress) st->monotonic = false;
        st->last_progress = f;
        return 0;
    };
    dms_status st = DMS_OK;
    run(eh.e, p, 9, &st, abort_fn, progress_fn, &s);
    ASSERT_EQ(st, DMS_OK);
    EXPECT_TRUE(s.monotonic);
    EXPECT_NEAR(s.last_progress, 1.f, 1e-5f);
    const int total_calls = s.calls.load();
    s.calls = 0;
    s.abort_after = total_calls / 2;  // abort mid-way through the tiles
    s.last_progress = 0.f;
    dms_cache_clear(eh.e);
    run(eh.e, p, 9, &st, abort_fn, progress_fn, &s);
    EXPECT_EQ(st, DMS_ERR_ABORTED);
}
