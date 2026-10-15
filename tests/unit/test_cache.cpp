#include "engine_impl.h"
#include "frame_cache.h"
#include "instance_registry.h"
#include "test_helpers.h"

#include <atomic>
#include <thread>

using namespace dms;
using namespace dms_test;

namespace {
CacheKey key(uint64_t layer, int64_t t) {
    CacheKey k;
    k.layer_hash = layer;
    k.time_num = t;
    k.time_den = 24;
    k.model_id = "m";
    k.base_res = 518;
    k.width = 10;
    k.height = 10;
    return k;
}
CacheEntry entry(size_t floats, float v = 1.f) {
    CacheEntry e;
    e.depth = std::make_shared<const std::vector<float>>(floats, v);
    return e;
}
}  // namespace

TEST(FrameCache, HitMissAndLruOrder) {
    const uint64_t one = entry(1000).bytes();
    FrameCache c(one * 3);
    c.put(key(1, 0), entry(1000, 1));
    c.put(key(1, 1), entry(1000, 2));
    c.put(key(1, 2), entry(1000, 3));
    CacheEntry out;
    ASSERT_TRUE(c.get(key(1, 0), out));  // touch 0 -> most recent
    c.put(key(1, 3), entry(1000, 4));    // evicts LRU = key 1
    EXPECT_FALSE(c.get(key(1, 1), out));
    EXPECT_TRUE(c.get(key(1, 0), out));
    EXPECT_EQ((*out.depth)[0], 1.f);
    EXPECT_TRUE(c.get(key(1, 3), out));
    const auto s = c.stats();
    EXPECT_EQ(s.entries, 3u);
    EXPECT_EQ(s.evictions, 1u);
    EXPECT_LE(s.bytes_used, s.bytes_budget);
    EXPECT_EQ(s.hits, 3u);
    EXPECT_EQ(s.misses, 1u);
}

TEST(FrameCache, KeyIncludesEveryResultAffectingParameter) {
    FrameCache c(1ull << 30);
    CacheKey a = key(7, 3);
    c.put(a, entry(10));
    CacheEntry out;
    for (int variant = 0; variant < 9; ++variant) {
        CacheKey b = a;
        switch (variant) {
            case 0: b.layer_hash++; break;
            case 1: b.time_num++; break;
            case 2: b.model_id = "other"; break;
            case 3: b.base_res = 768; break;
            case 4: b.samples = 4; break;
            case 5: b.tiles = 4; break;
            case 6: b.edges_q = 5; break;
            case 7: b.smoothing_q = 5; break;
            case 8: b.fp16 = true; break;
        }
        EXPECT_FALSE(c.get(b, out)) << "variant " << variant;
    }
    EXPECT_TRUE(c.get(a, out));
}

TEST(FrameCache, BudgetShrinkAndOversizedEntries) {
    const uint64_t one = entry(1000).bytes();
    FrameCache c(one * 10);
    for (int i = 0; i < 10; ++i) c.put(key(2, i), entry(1000));
    EXPECT_EQ(c.stats().entries, 10u);
    c.set_budget(one * 4);
    EXPECT_EQ(c.stats().entries, 4u);
    c.put(key(3, 0), entry(100000));  // larger than the whole budget: ignored, nothing evicted
    EXPECT_EQ(c.stats().entries, 4u);
    c.clear();
    EXPECT_EQ(c.stats().bytes_used, 0u);
}

TEST(FrameCache, EvictsHarderUnderMemoryPressure) {
    std::atomic<uint64_t> avail{8ull << 30};
    const uint64_t one = entry(1000).bytes();
    FrameCache c(one * 10, [&] { return avail.load(); }, 1ull << 30);
    for (int i = 0; i < 10; ++i) c.put(key(4, i), entry(1000));
    EXPECT_EQ(c.stats().entries, 10u);
    avail = 256ull << 20;  // system under pressure
    c.put(key(4, 100), entry(1000));
    EXPECT_LE(c.stats().bytes_used, one * 5);
}

TEST(FrameCache, ConcurrentAccessIsSafe) {
    const uint64_t one = entry(256).bytes();
    FrameCache c(one * 32);
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&, t] {
            CacheEntry out;
            for (int i = 0; i < 2000; ++i) {
                c.put(key(static_cast<uint64_t>(t), i % 50), entry(256));
                c.get(key(static_cast<uint64_t>((t + 1) % 8), i % 50), out);
            }
        });
    for (auto& t : ts) t.join();
    EXPECT_LE(c.stats().bytes_used, one * 32);
}

TEST(InstanceRegistry, LastGoodDepthAndMessageDedup) {
    InstanceRegistry r;
    InstanceRegistry::Guid g{};
    g[0] = 1;
    r.open(g);
    std::vector<float> out(16);
    EXPECT_FALSE(r.last(g, out.data(), 4, 4));
    EXPECT_TRUE(r.should_report(g, DMS_ERR_MODEL_NOT_INSTALLED));
    EXPECT_FALSE(r.should_report(g, DMS_ERR_MODEL_NOT_INSTALLED));  // no dialog per frame
    EXPECT_TRUE(r.should_report(g, DMS_ERR_OUT_OF_MEMORY));
    std::vector<float> d(2000 * 1000, 0.25f);  // large: stored downsampled
    r.store(g, d.data(), 2000, 1000);
    EXPECT_TRUE(r.last(g, out.data(), 4, 4));
    for (float v : out) EXPECT_NEAR(v, 0.25f, 1e-6f);
    EXPECT_TRUE(r.should_report(g, DMS_ERR_MODEL_NOT_INSTALLED));  // reset after success
    r.close(g);
    EXPECT_FALSE(r.last(g, out.data(), 4, 4));
}

TEST(EngineCache, SecondComputeIsACacheHitAndParamsInvalidate) {
    EngineHandle eh("cache_engine");
    install_synthetic_model(eh.models_dir, "synthetic", "synthetic_depth.onnx");
    dms_model_reload_manifest(eh.e);
    TestFrame tf(64, 40);
    const dms_frame f = tf.frame();
    auto rq = make_request("synthetic", &f, 1, 0);
    rq.cache_key.layer_hash = 0xABCDEF;
    rq.cache_key.time_num = 5;
    rq.cache_key.time_den = 24;
    std::vector<float> a, b, c;
    auto r1 = make_result(a, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &r1), DMS_OK) << dms_last_error();
    EXPECT_EQ(r1.from_cache, 0);
    auto r2 = make_result(b, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &r2), DMS_OK);
    EXPECT_EQ(r2.from_cache, 1);
    EXPECT_EQ(a, b);
    EXPECT_EQ(r1.raw_min, r2.raw_min);
    rq.base_resolution = 384;  // changes the result => miss
    auto r3 = make_result(c, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &r3), DMS_OK);
    EXPECT_EQ(r3.from_cache, 0);
    // Cache-only lookup API.
    std::vector<float> d;
    auto r4 = make_result(d, tf.w, tf.h);
    EXPECT_EQ(dms_depth_cache_lookup(eh.e, &rq.cache_key, "synthetic", 384, &rq.backend, 0, 0, &r4), DMS_OK);
    EXPECT_EQ(d, c);
    rq.cache_key.time_num = 6;
    EXPECT_EQ(dms_depth_cache_lookup(eh.e, &rq.cache_key, "synthetic", 384, &rq.backend, 0, 0, &r4), DMS_ERR_NOT_FOUND);
    dms_cache_stats st{};
    st.struct_size = sizeof(st);
    ASSERT_EQ(dms_cache_stats_query(eh.e, &st), DMS_OK);
    EXPECT_GE(st.entries, 2u);
    EXPECT_EQ(dms_cache_clear(eh.e), DMS_OK);
    ASSERT_EQ(dms_cache_stats_query(eh.e, &st), DMS_OK);
    EXPECT_EQ(st.entries, 0u);
}

TEST(EngineCache, LayerHashZeroDisablesCaching) {
    EngineHandle eh("cache_disabled");
    install_synthetic_model(eh.models_dir, "synthetic", "synthetic_depth.onnx");
    dms_model_reload_manifest(eh.e);
    TestFrame tf(32, 32);
    const dms_frame f = tf.frame();
    auto rq = make_request("synthetic", &f, 1, 0);
    std::vector<float> a;
    auto r = make_result(a, tf.w, tf.h);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &r), DMS_OK);
    ASSERT_EQ(dms_depth_compute(eh.e, &rq, &r), DMS_OK);
    EXPECT_EQ(r.from_cache, 0);
}

TEST(EngineInstances, AbiRoundTrip) {
    EngineHandle eh("instances_abi");
    uint8_t g[16] = {9, 9, 9};
    ASSERT_EQ(dms_instance_open(eh.e, g), DMS_OK);
    std::vector<float> d(8 * 8, 0.75f), out(4 * 4, 0.f);
    EXPECT_EQ(dms_instance_last_depth(eh.e, g, out.data(), 4, 4), DMS_ERR_NOT_FOUND);
    ASSERT_EQ(dms_instance_store_depth(eh.e, g, d.data(), 8, 8), DMS_OK);
    ASSERT_EQ(dms_instance_last_depth(eh.e, g, out.data(), 4, 4), DMS_OK);
    EXPECT_NEAR(out[5], 0.75f, 1e-6f);
    int32_t first = 0;
    ASSERT_EQ(dms_instance_should_report(eh.e, g, 3, &first), DMS_OK);
    EXPECT_EQ(first, 1);
    ASSERT_EQ(dms_instance_should_report(eh.e, g, 3, &first), DMS_OK);
    EXPECT_EQ(first, 0);
    EXPECT_EQ(dms_instance_close(eh.e, g), DMS_OK);
    EXPECT_EQ(dms_instance_open(eh.e, nullptr), DMS_ERR_INVALID_ARG);
}
