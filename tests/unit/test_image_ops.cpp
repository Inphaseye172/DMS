#include "image_ops.h"
#include "test_helpers.h"

#include <cmath>
#include <limits>
#include <vector>

TEST(ImageOps, ResizeIdentityAndConstant) {
    std::vector<float> src(7 * 5);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i);
    std::vector<float> dst(src.size());
    dms::resize_bilinear(src.data(), 7, 5, dst.data(), 7, 5);
    EXPECT_EQ(src, dst);

    std::vector<float> c(9 * 3, 0.25f), up(31 * 17);
    dms::resize_bilinear(c.data(), 9, 3, up.data(), 31, 17);
    for (float v : up) EXPECT_FLOAT_EQ(v, 0.25f);
}

TEST(ImageOps, ResizePreservesLinearGradientInterior) {
    const int sw = 16, sh = 4, dw = 64, dh = 8;
    std::vector<float> src(sw * sh), dst(dw * dh);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x) src[static_cast<size_t>(y * sw + x)] = static_cast<float>(x);
    dms::resize_bilinear(src.data(), sw, sh, dst.data(), dw, dh);
    // Interior samples follow x_src = (x+0.5)*sw/dw - 0.5.
    for (int x = 4; x < dw - 4; ++x) {
        const float expect = (static_cast<float>(x) + 0.5f) * sw / dw - 0.5f;
        EXPECT_NEAR(dst[static_cast<size_t>(3 * dw + x)], expect, 1e-4f);
    }
}

TEST(ImageOps, InferenceDimsAreMultiplesAndKeepAspect) {
    int w = 0, h = 0;
    dms::inference_dims(1920, 1080, 518, 14, w, h);
    EXPECT_EQ(w, 518);
    EXPECT_EQ(h % 14, 0);
    EXPECT_NEAR(static_cast<double>(w) / h, 1920.0 / 1080.0, 0.05);
    dms::inference_dims(1, 1, 518, 14, w, h);  // tiny input still yields valid dims
    EXPECT_EQ(w, 518);
    EXPECT_EQ(h, 518);
    dms::inference_dims(3, 4001, 384, 14, w, h);  // extreme aspect: short side never below one patch
    EXPECT_GE(w, 14);
    EXPECT_EQ(h % 14, 0);
}

TEST(ImageOps, NormalizationAndNaNHandling) {
    // 2x1 RGB image: one pixel NaN, one pixel white.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> rgb = {nan, nan, nan, 1.f, 1.f, 1.f};
    std::vector<float> out(3 * 2);
    dms::resize_rgb_to_nchw_normalized(rgb.data(), 2, 1, 6, out.data(), 2, 1, false, 0.f, 0.f);
    for (float v : out) EXPECT_TRUE(std::isfinite(v));
    EXPECT_NEAR(out[1], (1.f - 0.485f) / 0.229f, 1e-4f);  // R plane, pixel 1
    EXPECT_NEAR(out[0], (0.f - 0.485f) / 0.229f, 1e-4f);  // NaN treated as 0
}

TEST(ImageOps, BoxBlurPreservesConstantAndMean) {
    const int w = 23, h = 11;
    std::vector<float> a(w * h, 3.f), b(a.size()), tmp(a.size());
    dms::box_blur(a.data(), b.data(), tmp.data(), w, h, 4);
    for (float v : b) EXPECT_NEAR(v, 3.f, 1e-5f);
}

TEST(ImageOps, FiniteMinMax) {
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> v = {inf, -2.f, 5.f, std::numeric_limits<float>::quiet_NaN()};
    float mn = 0, mx = 0;
    ASSERT_TRUE(dms::finite_min_max(v.data(), v.size(), mn, mx));
    EXPECT_EQ(mn, -2.f);
    EXPECT_EQ(mx, 5.f);
    std::vector<float> none = {inf, -inf};
    EXPECT_FALSE(dms::finite_min_max(none.data(), none.size(), mn, mx));
}
