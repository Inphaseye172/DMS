#include "test_helpers.h"

#include <cmath>
#include <limits>
#include <set>
#include <string>

TEST(Colormaps, CountNamesAndBounds) {
    const int n = dms_colormap_count();
    EXPECT_EQ(n, 21);  // None + 20 LUTs
    EXPECT_STREQ(dms_colormap_name(0), "None");
    std::set<std::string> names;
    for (int i = 0; i < n; ++i) {
        const char* nm = dms_colormap_name(i);
        ASSERT_NE(nm, nullptr);
        EXPECT_GT(std::string(nm).size(), 0u);
        names.insert(nm);
    }
    EXPECT_EQ(names.size(), static_cast<size_t>(n));  // unique (they label a popup)
    EXPECT_STREQ(dms_colormap_name(-1), "");
    EXPECT_STREQ(dms_colormap_name(n), "");
}

TEST(Colormaps, GrayscaleIsWhiteNearAndLutsStayInRange) {
    std::vector<float> v = {0.f, 0.5f, 1.f, -3.f, 7.f, std::numeric_limits<float>::quiet_NaN()};
    std::vector<float> rgb(v.size() * 3);
    ASSERT_EQ(dms_apply_colormap(v.data(), static_cast<int>(v.size()), 1, 0, rgb.data()), DMS_OK);
    EXPECT_EQ(rgb[6], 1.f);   // near = white
    EXPECT_EQ(rgb[0], 0.f);   // far = black
    EXPECT_EQ(rgb[9], 0.f);   // clamped
    EXPECT_EQ(rgb[12], 1.f);  // clamped
    EXPECT_EQ(rgb[15], 0.f);  // NaN -> far
    for (int m = 1; m < dms_colormap_count(); ++m) {
        ASSERT_EQ(dms_apply_colormap(v.data(), static_cast<int>(v.size()), 1, m, rgb.data()), DMS_OK);
        for (float c : rgb) ASSERT_TRUE(c >= 0.f && c <= 1.f) << "map " << m;
        // Near and far must differ (a colormap that maps everything to one color is broken).
        EXPECT_GT(std::fabs(rgb[0] - rgb[6]) + std::fabs(rgb[1] - rgb[7]) + std::fabs(rgb[2] - rgb[8]), 0.1f) << dms_colormap_name(m);
    }
    EXPECT_EQ(dms_apply_colormap(v.data(), 6, 1, 99, rgb.data()), DMS_ERR_INVALID_ARG);
    EXPECT_EQ(dms_apply_colormap(nullptr, 6, 1, 1, rgb.data()), DMS_ERR_INVALID_ARG);
}

TEST(Colormaps, TurboEndpointsMatchReference) {
    // matplotlib turbo: t=0 -> (48,18,59), t=1 -> (122,4,3)
    std::vector<float> v = {0.f, 1.f}, rgb(6);
    ASSERT_EQ(dms_apply_colormap(v.data(), 2, 1, 1, rgb.data()), DMS_OK);
    EXPECT_NEAR(rgb[0] * 255.f, 48.f, 1.f);
    EXPECT_NEAR(rgb[2] * 255.f, 59.f, 1.f);
    EXPECT_NEAR(rgb[3] * 255.f, 122.f, 1.f);
    EXPECT_NEAR(rgb[5] * 255.f, 3.f, 1.f);
}
