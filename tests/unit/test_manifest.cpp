#include "common.h"
#include "manifest.h"
#include "test_helpers.h"

#include <fstream>

using dms::Manifest;

TEST(Manifest, BuiltinParsesAndHasPinnedModels) {
    const auto dir = dms_test::tmp_dir("manifest_builtin");
    Manifest m(dir);
    ASSERT_GE(m.count(), 6u);
    const auto da2 = m.find("da2-small");
    ASSERT_TRUE(da2.has_value());
    EXPECT_EQ(da2->sha256, "46c4e8eeda3a27f34701831b6a2ec7753d7b38779b215acb5633424703deed8f");
    EXPECT_EQ(da2->size, 99092268u);
    EXPECT_EQ(da2->patch_multiple, 14);
    const auto video = m.find("video-da-small");
    ASSERT_TRUE(video.has_value());
    EXPECT_EQ(video->kind, DMS_MODEL_KIND_TEMPORAL);
    EXPECT_EQ(video->input_layout, "BTCHW");
    EXPECT_TRUE(video->url.empty());  // to be supplied by the user
    EXPECT_FALSE(m.find("does-not-exist").has_value());
}

TEST(Manifest, RejectsUnsafeOrInconsistentEntries) {
    EXPECT_THROW(Manifest::parse("not json"), dms::Error);
    EXPECT_THROW(Manifest::parse("{\"models\":[{\"id\":\"x\",\"file\":\"../evil.onnx\"}]}"), dms::Error);
    EXPECT_THROW(Manifest::parse("{\"models\":[{\"id\":\"x\",\"file\":\"C:evil.onnx\"}]}"), dms::Error);
    EXPECT_THROW(Manifest::parse("{\"models\":[{\"id\":\"x\",\"file\":\"a.onnx\",\"url\":\"https://x\"}]}"), dms::Error);  // url w/o sha
    EXPECT_THROW(Manifest::parse("{\"models\":[{\"id\":\"x\",\"file\":\"a.onnx\",\"sha256\":\"zz\"}]}"), dms::Error);
    EXPECT_THROW(Manifest::parse("{\"models\":[{\"id\":\"x\",\"file\":\"a.onnx\",\"input_layout\":\"HWC\"}]}"), dms::Error);
    EXPECT_NO_THROW(Manifest::parse("{\"models\":[{\"id\":\"x\",\"file\":\"a.onnx\"}]}"));
}

TEST(Manifest, UserOverrideReplacesAndAdds_InvalidOverrideIgnored) {
    const auto dir = dms_test::tmp_dir("manifest_override");
    {
        std::ofstream f(dir / "models.json");
        f << R"({"models":[{"id":"da3-small","display_name":"DA3 S custom","file":"da3s.onnx",
                 "url":"https://example.com/da3s.onnx","sha256":"0000000000000000000000000000000000000000000000000000000000000000","size":5},
                {"id":"extra","file":"extra.onnx"}]})";
    }
    Manifest m(dir);
    const auto d = m.find("da3-small");
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->display_name, "DA3 S custom");
    EXPECT_EQ(d->url, "https://example.com/da3s.onnx");
    EXPECT_TRUE(m.find("extra").has_value());
    {
        std::ofstream f(dir / "models.json");
        f << "{ broken";
    }
    m.reload();  // must not throw; falls back to built-ins
    EXPECT_TRUE(m.find("da2-small").has_value());
    EXPECT_FALSE(m.find("extra").has_value());
}

TEST(Manifest, ExplicitOnnxPathIsACustomModel) {
    const auto dir = dms_test::tmp_dir("manifest_path");
    Manifest m(dir);
    const auto p = dms_test::fixture("onnx/synthetic_depth.onnx").string();
    const auto spec = m.find(p);
    ASSERT_TRUE(spec.has_value());
    EXPECT_TRUE(spec->custom_path);
    EXPECT_FALSE(m.find(p + ".missing.onnx").has_value());
}
