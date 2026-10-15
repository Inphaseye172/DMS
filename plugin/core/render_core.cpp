#include "render_core.h"

#include "pixels.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <new>

namespace dms_plugin {
namespace {

inline float clamp01(float v) noexcept {
    if (!std::isfinite(v)) return 0.f;
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}

template <typename P, typename T>
inline T quantize(float v, float maxv) noexcept {
    return static_cast<T>(std::lround(clamp01(v) * maxv));
}

// Writes one output pixel (straight alpha) at depth `d`.
inline void put(uint8_t* row, int32_t x, PixelDepth depth, float a, float r, float g, float b, bool raw_float) noexcept {
    switch (depth) {
        case PixelDepth::U8:
            reinterpret_cast<Pixel8*>(row)[x] = {quantize<Pixel8, uint8_t>(a, kMax8), quantize<Pixel8, uint8_t>(r, kMax8),
                                                 quantize<Pixel8, uint8_t>(g, kMax8), quantize<Pixel8, uint8_t>(b, kMax8)};
            break;
        case PixelDepth::U16:
            reinterpret_cast<Pixel16*>(row)[x] = {quantize<Pixel16, uint16_t>(a, kMax16), quantize<Pixel16, uint16_t>(r, kMax16),
                                                  quantize<Pixel16, uint16_t>(g, kMax16), quantize<Pixel16, uint16_t>(b, kMax16)};
            break;
        case PixelDepth::F32: {
            // 32 bpc keeps unclamped values for Raw mapping (still finite).
            auto f = [raw_float](float v) { return std::isfinite(v) ? (raw_float ? v : clamp01(v)) : 0.f; };
            reinterpret_cast<PixelF*>(row)[x] = {clamp01(a), f(r), f(g), f(b)};
            break;
        }
    }
}

template <typename P>
inline void read_src(const uint8_t* row, int32_t x, float scale, float& a, float& r, float& g, float& b) noexcept {
    const P& p = reinterpret_cast<const P*>(row)[x];
    a = clamp01(static_cast<float>(p.a) * scale);
    r = static_cast<float>(p.r) * scale;
    g = static_cast<float>(p.g) * scale;
    b = static_cast<float>(p.b) * scale;
}

// Concatenates message parts (no runtime format strings). Never throws.
void set_msg(RenderOutcome& o, const char* a, const char* b = nullptr, const char* c = nullptr, const char* d = nullptr,
             const char* e = nullptr) noexcept {
    try {
        std::string m;
        for (const char* p : {a, b, c, d, e})
            if (p) m += p;
        if (m.size() > 500) m.resize(500);
        o.message = std::move(m);
    } catch (...) {
    }
}

}  // namespace

void write_output(const RenderJob& job, const std::vector<float>& mapped, const std::vector<float>* rgb_colormap,
                  const std::vector<float>* slice_mask, const std::vector<float>& input_alpha) noexcept {
    const WorldView& out = job.output;
    const WorldView& src = job.frames[static_cast<size_t>(job.center_index)];
    const int32_t dw = src.width, dh = src.height;
    const bool raw_float = job.params.mapping == Mapping::Raw;
    const float src_scale = src.depth == PixelDepth::U8 ? 1.f / kMax8 : src.depth == PixelDepth::U16 ? 1.f / kMax16 : 1.f;
    for (int32_t oy = 0; oy < out.height; ++oy) {
        uint8_t* orow = out.row(oy);
        const int32_t dy = oy + job.output_rect.top - job.input_rect.top;
        for (int32_t ox = 0; ox < out.width; ++ox) {
            const int32_t dx = ox + job.output_rect.left - job.input_rect.left;
            if (dx < 0 || dy < 0 || dx >= dw || dy >= dh) {
                put(orow, ox, out.depth, 0.f, 0.f, 0.f, 0.f, false);  // outside the source layer: transparent
                continue;
            }
            const size_t i = static_cast<size_t>(dy) * static_cast<size_t>(dw) + static_cast<size_t>(dx);
            const float a = input_alpha.empty() ? 1.f : input_alpha[i];
            if (slice_mask) {
                float sa, r, g, b;
                const uint8_t* srow = src.row(dy);
                switch (src.depth) {
                    case PixelDepth::U8: read_src<Pixel8>(srow, dx, src_scale, sa, r, g, b); break;
                    case PixelDepth::U16: read_src<Pixel16>(srow, dx, src_scale, sa, r, g, b); break;
                    default: read_src<PixelF>(srow, dx, src_scale, sa, r, g, b); break;
                }
                put(orow, ox, out.depth, sa * (*slice_mask)[i], r, g, b, true);
            } else if (rgb_colormap) {
                put(orow, ox, out.depth, a, (*rgb_colormap)[i * 3], (*rgb_colormap)[i * 3 + 1], (*rgb_colormap)[i * 3 + 2], false);
            } else {
                const float v = mapped[i];
                put(orow, ox, out.depth, a, v, v, v, raw_float);
            }
        }
    }
}

RenderOutcome render(dms_engine* engine, const RenderJob& job) noexcept {
    RenderOutcome o;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        if (!job.output.valid()) {
            o.status = DMS_ERR_INVALID_ARG;
            set_msg(o, "DMS: invalid output buffer");
            return o;
        }
        if (job.frames.empty() || job.center_index < 0 || job.center_index >= static_cast<int32_t>(job.frames.size()) ||
            !job.frames[static_cast<size_t>(job.center_index)].valid()) {
            // Zero-size / missing input: write transparent output, never touch invalid memory.
            fill_world(job.output, 0.f, 0.f);
            o.output_written = true;
            o.status = DMS_OK;
            return o;
        }
        const WorldView& src = job.frames[static_cast<size_t>(job.center_index)];
        const int32_t w = src.width, h = src.height;
        const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);

        // 1) Convert the window to RGB float. Neighbor frames with mismatched size are dropped.
        std::vector<std::vector<float>> rgbs;
        std::vector<dms_frame> frames;
        std::vector<float> alpha;
        int32_t center = 0;
        uint64_t content_hash = 0;
        rgbs.reserve(job.frames.size());
        for (size_t k = 0; k < job.frames.size(); ++k) {
            const WorldView& f = job.frames[k];
            const bool is_center = static_cast<int32_t>(k) == job.center_index;
            if (!f.valid() || f.width != w || f.height != h) continue;
            if (is_center) center = static_cast<int32_t>(rgbs.size());
            rgbs.emplace_back();
            if (!world_to_rgb(f, rgbs.back(), is_center ? &alpha : nullptr)) {
                rgbs.pop_back();
                continue;
            }
            content_hash = hash_combine(content_hash, hash_world(f));
        }
        for (size_t k = 0; k < rgbs.size(); ++k) {
            dms_frame f{};
            f.struct_size = sizeof(f);
            f.rgb = rgbs[k].data();
            f.width = w;
            f.height = h;
            f.row_stride_floats = w * 3;
            f.time_num = job.time_num + static_cast<int64_t>(k) - center;
            f.time_den = job.time_den;
            frames.push_back(f);
        }

        // 2) Inference (or cache hit) via the engine.
        std::vector<float> depth(n, 0.5f);
        dms_depth_request rq{};
        rq.struct_size = sizeof(rq);
        rq.model_id = model_id(job.params.model_index);
        rq.frames = frames.data();
        rq.frame_count = static_cast<int32_t>(frames.size());
        rq.center_index = center;
        rq.base_resolution = job.params.base_res;
        rq.edges_enhancement = job.params.edges;
        rq.subpixel_smoothing = job.params.smoothing;
        rq.backend.struct_size = sizeof(rq.backend);
        rq.backend.hardware_acceleration = job.params.hw_accel;
        rq.backend.fp16 = job.params.fp16;
        rq.backend.cuda_memory_sharing = job.params.cuda_memshare;
        rq.backend.offload_mode = job.params.offload;
        rq.backend.samples = job.params.samples;
        rq.backend.parallel_samples = job.params.parallel;
        rq.backend.tiles = job.params.tiles;
        rq.backend.memory_optimization = job.params.memopt;
        rq.cache_key.struct_size = sizeof(rq.cache_key);
        rq.cache_key.layer_hash = content_hash;
        rq.cache_key.time_num = job.time_num;
        rq.cache_key.time_den = job.time_den;
        rq.abort_fn = job.abort_fn;
        rq.progress_fn = job.progress_fn;
        rq.user = job.callback_user;
        rq.timeout_ms = job.timeout_ms;

        dms_depth_result rs{};
        rs.struct_size = sizeof(rs);
        rs.depth = depth.data();
        rs.width = w;
        rs.height = h;
        dms_status st = frames.empty() ? DMS_ERR_INVALID_ARG : dms_depth_compute(engine, &rq, &rs);

        if (st == DMS_ERR_ABORTED) {
            o.status = DMS_ERR_ABORTED;
            return o;
        }
        bool degraded = st == DMS_ERR_DEGRADED;
        if (st == DMS_OK || st == DMS_ERR_DEGRADED) {
            if (rs.message[0]) set_msg(o, "DMS: ", rs.message);
            if (dms_instance_store_depth(engine, job.instance_guid, depth.data(), w, h) != DMS_OK) { /* non-fatal */ }
        } else {
            // Fallback: last good depth of this instance, else flat mid-gray.
            const char* why = dms_status_message(st);
            const char* detail = dms_last_error();
            if (dms_instance_last_depth(engine, job.instance_guid, depth.data(), w, h) == DMS_OK) {
                set_msg(o, "DMS: ", why, " (", detail, ") - showing last good depth map");
                rs.raw_min = 1.f;
                rs.raw_max = 0.f;  // force recompute of the range below
            } else {
                std::fill(depth.begin(), depth.end(), 0.5f);
                set_msg(o, "DMS: ", why, " (", detail, ") - showing flat gray");
                rs.raw_min = 0.f;
                rs.raw_max = 1.f;
            }
            degraded = true;
            int32_t first = 0;
            if (dms_instance_should_report(engine, job.instance_guid, static_cast<int32_t>(st), &first) == DMS_OK) o.report_to_user = first != 0;
        }

        // 3) Mapping over the full map (needs global statistics), then mode-specific output for the sub-rect.
        std::vector<float> mapped(n);
        dms_mapping_params mp{};
        mp.struct_size = sizeof(mp);
        mp.mode = static_cast<int32_t>(job.params.mapping);
        mp.near_x = job.params.near_x - static_cast<float>(job.input_rect.left);
        mp.near_y = job.params.near_y - static_cast<float>(job.input_rect.top);
        mp.far_x = job.params.far_x - static_cast<float>(job.input_rect.left);
        mp.far_y = job.params.far_y - static_cast<float>(job.input_rect.top);
        mp.near_value = job.params.near_value;
        mp.far_value = job.params.far_value;
        mp.raw_min = rs.raw_min;
        mp.raw_max = rs.raw_max;
        st = dms_map_depth(depth.data(), w, h, &mp, mapped.data());
        if (st == DMS_ERR_UNSUPPORTED) {
            mapped = depth;
        } else if (st != DMS_OK) {
            std::fill(mapped.begin(), mapped.end(), 0.5f);
            degraded = true;
            set_msg(o, "DMS: depth mapping failed (", dms_last_error(), ")");
        }

        if (job.params.mode == Mode::Slicing) {
            std::vector<float> mask(n);
            if (dms_slice_mask(mapped.data(), w, h, job.params.slice_min, job.params.slice_max, job.params.smoothing / 100.f, mask.data()) != DMS_OK)
                std::fill(mask.begin(), mask.end(), 1.f);
            write_output(job, mapped, nullptr, &mask, alpha);
        } else if (job.params.colormap > 0) {
            std::vector<float> rgb(n * 3u);
            if (dms_apply_colormap(mapped.data(), w, h, job.params.colormap, rgb.data()) == DMS_OK)
                write_output(job, mapped, &rgb, nullptr, alpha);
            else
                write_output(job, mapped, nullptr, nullptr, alpha);
        } else {
            write_output(job, mapped, nullptr, nullptr, alpha);
        }
        o.output_written = true;
        o.status = degraded ? DMS_ERR_DEGRADED : DMS_OK;
    } catch (const std::bad_alloc&) {
        fill_world(job.output, 0.5f, 1.f);
        o.output_written = true;
        o.status = DMS_ERR_DEGRADED;
        set_msg(o, "DMS: out of memory - showing flat gray");
        o.report_to_user = true;
    } catch (...) {
        fill_world(job.output, 0.5f, 1.f);
        o.output_written = true;
        o.status = DMS_ERR_DEGRADED;
        set_msg(o, "DMS: internal error - showing flat gray");
        o.report_to_user = true;
    }
    o.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return o;
}

}  // namespace dms_plugin
