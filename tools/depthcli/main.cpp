// depthcli — runs the DMS depth engine on an image or image sequence without After Effects.
#include "image_io.h"

#include "dms/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using depthcli::Image;

namespace {

struct Options {
    std::vector<std::string> inputs;
    std::string output_dir = "depth_out";
    std::string model = "da2-small";
    int base_res = 518;
    std::string mapping = "normalize";
    float near_x = -1, near_y = -1, far_x = -1, far_y = -1;
    float near_value = 1.f, far_value = 0.f;
    std::string mode = "depth";
    std::string colormap = "None";
    float slice_min = 0.f, slice_max = 1.f;
    float edges = 0.f, smoothing = 0.f;
    int samples = 1;
    bool parallel = false;
    int tiles = 0;
    std::string ep = "auto";
    bool fp16 = false, memshare = false, trt = false, memopt = false;
    int offload = 0;
    int window = 0;  // temporal window override (0 = model default)
    std::string models_dir, deps_dir, log_dir;
    int log_level = DMS_LOG_INFO;
    std::string download;
    bool list_models = false, device = false, bench = false, quiet = false, no_cache = false;
    std::string format = "png";
    int repeat = 1;
    uint64_t cache_mb = 0;
    uint32_t timeout_ms = 0;
};

void usage() {
    std::puts(
        "depthcli - DMS depth engine command line\n"
        "\n"
        "  depthcli --input <file|dir|pattern%04d.png> [--input ...] --output <dir> [options]\n"
        "  depthcli --list-models | --device | --download <model-id>\n"
        "\n"
        "Model / quality:\n"
        "  --model <id|file.onnx>     da2-small (default), da2-base, da2-large, da3-*, video-*, or a path\n"
        "  --res 384|518|768|1024     base resolution (long side), default 518\n"
        "  --edges 0..100             edges enhancement\n"
        "  --smoothing 0..100         subpixel smoothing\n"
        "Mapping:\n"
        "  --mapping raw|normalize|guided\n"
        "  --near x,y --far x,y       guide points (pixels); --near-value v --far-value v\n"
        "Output:\n"
        "  --mode depth|slice         --colormap <name|index> (depth)   --slice-min v --slice-max v (slice)\n"
        "  --format png|pgm16|pfm     (slice mode always writes RGBA png)\n"
        "Backend:\n"
        "  --ep auto|cpu  --fp16  --memshare  --trt  --offload keep|ram|unload\n"
        "  --samples 1..16  --parallel  --tiles auto|1|4|9|16  --memopt  --window N\n"
        "Misc:\n"
        "  --models-dir D --deps-dir D --log-dir D --log-level 0..3 --cache-mb N --no-cache\n"
        "  --timeout-ms N --repeat N --bench --quiet");
}

bool parse_pair(const char* s, float& a, float& b) { return std::sscanf(s, "%f,%f", &a, &b) == 2; }

int parse_tiles(const std::string& s) {
    if (s == "auto") return 0;
    const int t = std::atoi(s.c_str());
    return (t == 1 || t == 4 || t == 9 || t == 16) ? t : -1;
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else if (a == "--input" || a == "-i") o.inputs.emplace_back(next("--input"));
        else if (a == "--output" || a == "-o") o.output_dir = next("--output");
        else if (a == "--model") o.model = next("--model");
        else if (a == "--res") o.base_res = std::atoi(next("--res"));
        else if (a == "--mapping") o.mapping = next("--mapping");
        else if (a == "--near") { if (!parse_pair(next("--near"), o.near_x, o.near_y)) return false; }
        else if (a == "--far") { if (!parse_pair(next("--far"), o.far_x, o.far_y)) return false; }
        else if (a == "--near-value") o.near_value = static_cast<float>(std::atof(next("--near-value")));
        else if (a == "--far-value") o.far_value = static_cast<float>(std::atof(next("--far-value")));
        else if (a == "--mode") o.mode = next("--mode");
        else if (a == "--colormap") o.colormap = next("--colormap");
        else if (a == "--slice-min") o.slice_min = static_cast<float>(std::atof(next("--slice-min")));
        else if (a == "--slice-max") o.slice_max = static_cast<float>(std::atof(next("--slice-max")));
        else if (a == "--edges") o.edges = static_cast<float>(std::atof(next("--edges")));
        else if (a == "--smoothing") o.smoothing = static_cast<float>(std::atof(next("--smoothing")));
        else if (a == "--samples") o.samples = std::atoi(next("--samples"));
        else if (a == "--parallel") o.parallel = true;
        else if (a == "--tiles") { o.tiles = parse_tiles(next("--tiles")); if (o.tiles < 0) return false; }
        else if (a == "--ep") o.ep = next("--ep");
        else if (a == "--fp16") o.fp16 = true;
        else if (a == "--memshare") o.memshare = true;
        else if (a == "--trt") o.trt = true;
        else if (a == "--memopt") o.memopt = true;
        else if (a == "--window") o.window = std::atoi(next("--window"));
        else if (a == "--offload") {
            const std::string v = next("--offload");
            o.offload = v == "ram" ? DMS_OFFLOAD_TO_RAM : v == "unload" ? DMS_OFFLOAD_FULL_UNLOAD : DMS_OFFLOAD_KEEP_VRAM;
        }
        else if (a == "--models-dir") o.models_dir = next("--models-dir");
        else if (a == "--deps-dir") o.deps_dir = next("--deps-dir");
        else if (a == "--log-dir") o.log_dir = next("--log-dir");
        else if (a == "--log-level") o.log_level = std::atoi(next("--log-level"));
        else if (a == "--download") o.download = next("--download");
        else if (a == "--list-models") o.list_models = true;
        else if (a == "--device") o.device = true;
        else if (a == "--bench") o.bench = true;
        else if (a == "--quiet" || a == "-q") o.quiet = true;
        else if (a == "--format") o.format = next("--format");
        else if (a == "--repeat") o.repeat = std::max(1, std::atoi(next("--repeat")));
        else if (a == "--cache-mb") o.cache_mb = static_cast<uint64_t>(std::atoll(next("--cache-mb")));
        else if (a == "--no-cache") o.no_cache = true;
        else if (a == "--timeout-ms") o.timeout_ms = static_cast<uint32_t>(std::atol(next("--timeout-ms")));
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return false;
        }
    }
    return true;
}

bool is_image(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp" || e == ".tga";
}

// Substitutes the first %d / %0Nd in a pattern with i (no printf with a runtime format string).
std::string format_frame(const std::string& pat, int i) {
    const size_t p = pat.find('%');
    if (p == std::string::npos) return pat;
    size_t q = p + 1;
    int width = 0;
    while (q < pat.size() && pat[q] >= '0' && pat[q] <= '9') width = width * 10 + (pat[q++] - '0');
    if (q >= pat.size() || pat[q] != 'd') return pat;
    std::string num = std::to_string(i);
    if (static_cast<int>(num.size()) < width) num.insert(0, static_cast<size_t>(width) - num.size(), '0');
    return pat.substr(0, p) + num + pat.substr(q + 1);
}

std::vector<std::string> expand_inputs(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    for (const auto& s : in) {
        std::error_code ec;
        if (fs::is_directory(s, ec)) {
            std::vector<std::string> d;
            for (const auto& e : fs::directory_iterator(s, ec))
                if (e.is_regular_file() && is_image(e.path())) d.push_back(e.path().string());
            std::sort(d.begin(), d.end());
            out.insert(out.end(), d.begin(), d.end());
        } else if (s.find('%') != std::string::npos) {
            for (int i = 0; i < 100000; ++i) {
                const std::string name = format_frame(s, i);
                if (fs::exists(name, ec)) out.push_back(name);
                else if (i > 1) break;  // allow sequences starting at 0 or 1
            }
        } else {
            out.push_back(s);
        }
    }
    return out;
}

int find_colormap(const std::string& name) {
    const int n = dms_colormap_count();
    for (int i = 0; i < n; ++i) {
        const char* cn = dms_colormap_name(i);
        if (cn && name == cn) return i;
    }
    char* end = nullptr;
    const long v = std::strtol(name.c_str(), &end, 10);
    if (end && *end == '\0' && v >= 0 && v < n) return static_cast<int>(v);
    return -1;
}

void print_status(const char* what, dms_status s) {
    std::fprintf(stderr, "%s: %s (%d) %s\n", what, dms_status_message(s), static_cast<int>(s), dms_last_error());
}

int download_model(dms_engine* eng, const std::string& id, bool quiet) {
    dms_status s = dms_download_start(eng, id.c_str());
    if (s != DMS_OK) { print_status("download_start", s); return 1; }
    for (;;) {
        dms_download_info di{};
        di.struct_size = sizeof(di);
        s = dms_download_query(eng, id.c_str(), &di);
        if (s != DMS_OK) { print_status("download_query", s); return 1; }
        if (!quiet) {
            const double pct = di.bytes_total ? 100.0 * static_cast<double>(di.bytes_done) / static_cast<double>(di.bytes_total) : 0.0;
            std::fprintf(stderr, "\r%-10s %6.1f%%  %llu / %llu bytes  %.1f MB/s   ",
                         di.state == DMS_DL_VERIFYING ? "verifying" : "download", pct,
                         static_cast<unsigned long long>(di.bytes_done), static_cast<unsigned long long>(di.bytes_total),
                         static_cast<double>(di.bytes_per_second) / 1e6);
        }
        if (di.state == DMS_DL_DONE) { if (!quiet) std::fputs("\ndone\n", stderr); return 0; }
        if (di.state == DMS_DL_FAILED || di.state == DMS_DL_CANCELLED) {
            std::fprintf(stderr, "\ndownload failed: %s\n", di.message);
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) { usage(); return 2; }

    dms_engine_config cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.models_dir = o.models_dir.empty() ? nullptr : o.models_dir.c_str();
    cfg.deps_dir = o.deps_dir.empty() ? nullptr : o.deps_dir.c_str();
    cfg.log_dir = o.log_dir.empty() ? nullptr : o.log_dir.c_str();
    cfg.log_level = o.log_level;
    cfg.cache_budget_bytes = o.cache_mb * 1024ull * 1024ull;
    cfg.inference_timeout_ms = o.timeout_ms;

    dms_engine* eng = nullptr;
    dms_status s = dms_engine_acquire(&cfg, &eng);
    if (s != DMS_OK) { print_status("engine_acquire", s); return 1; }
    struct Releaser { dms_engine* e; ~Releaser() { dms_engine_release(e); } } releaser{eng};

    if (o.device) {
        dms_device_info di{};
        di.struct_size = sizeof(di);
        s = dms_device_query(eng, &di);
        if (s != DMS_OK) { print_status("device_query", s); return 1; }
        std::printf("GPU vendor=%d name=%s driver=%s VRAM free/total=%llu/%llu MB\nRAM free/total=%llu/%llu MB\n"
                    "CUDA=%d DirectML=%d TensorRT=%d ORT=%s\nnotes: %s\n",
                    di.gpu_vendor, di.gpu_name, di.driver_version, static_cast<unsigned long long>(di.vram_free_bytes >> 20),
                    static_cast<unsigned long long>(di.vram_total_bytes >> 20), static_cast<unsigned long long>(di.ram_free_bytes >> 20),
                    static_cast<unsigned long long>(di.ram_total_bytes >> 20), di.cuda_available, di.directml_available,
                    di.tensorrt_available, di.ort_version, di.notes);
        if (o.inputs.empty() && !o.list_models && o.download.empty()) return 0;
    }
    if (o.list_models) {
        int32_t n = 0;
        s = dms_model_count(eng, &n);
        if (s != DMS_OK) { print_status("model_count", s); return 1; }
        for (int32_t i = 0; i < n; ++i) {
            dms_model_info mi{};
            mi.struct_size = sizeof(mi);
            if (dms_model_info_at(eng, i, &mi) != DMS_OK) continue;
            std::printf("%-16s %-34s %-8s window=%d installed=%d url=%d size=%llu MB\n    %s\n", mi.id, mi.display_name,
                        mi.kind == DMS_MODEL_KIND_TEMPORAL ? "temporal" : "single", mi.window_size, mi.installed, mi.has_url,
                        static_cast<unsigned long long>(mi.expected_bytes >> 20), mi.path);
        }
        if (o.inputs.empty() && o.download.empty()) return 0;
    }
    if (!o.download.empty()) {
        const int rc = download_model(eng, o.download, o.quiet);
        if (rc != 0 || o.inputs.empty()) return rc;
    }

    const auto files = expand_inputs(o.inputs);
    if (files.empty()) { std::fprintf(stderr, "no input images\n"); usage(); return 2; }
    std::error_code ec;
    fs::create_directories(o.output_dir, ec);

    int colormap = find_colormap(o.colormap);
    if (colormap < 0) { std::fprintf(stderr, "unknown colormap '%s'\n", o.colormap.c_str()); return 2; }

    // Temporal window: number of frames the model consumes (from the model info), clamped by the sequence length.
    int window = 1;
    {
        dms_model_info mi{};
        mi.struct_size = sizeof(mi);
        if (dms_model_info_by_id(eng, o.model.c_str(), &mi) == DMS_OK && mi.kind == DMS_MODEL_KIND_TEMPORAL)
            window = o.window > 0 ? o.window : mi.window_size;
    }
    window = std::clamp(window, 1, static_cast<int>(files.size()));

    std::map<size_t, Image> loaded;  // small LRU-ish store for the sliding window
    auto get_image = [&](size_t idx) -> const Image* {
        auto it = loaded.find(idx);
        if (it != loaded.end()) return &it->second;
        Image im;
        std::string err;
        if (!depthcli::load_image(files[idx], im, err)) {
            std::fprintf(stderr, "failed to load %s: %s\n", files[idx].c_str(), err.c_str());
            return nullptr;
        }
        // Drop images that fell out of the window.
        for (auto e = loaded.begin(); e != loaded.end();) {
            if (e->first + static_cast<size_t>(window) < idx) e = loaded.erase(e);
            else ++e;
        }
        return &loaded.emplace(idx, std::move(im)).first->second;
    };

    int failures = 0;
    double total_ms = 0.0;
    int processed = 0;
    for (int rep = 0; rep < o.repeat; ++rep) {
        for (size_t fi = 0; fi < files.size(); ++fi) {
            // Build the frame window centered on fi (clamped to the sequence).
            const int half = (window - 1) / 2;
            int start = static_cast<int>(fi) - half;
            start = std::clamp(start, 0, static_cast<int>(files.size()) - window);
            std::vector<dms_frame> frames;
            std::vector<const Image*> imgs;
            bool ok = true;
            for (int k = 0; k < window; ++k) {
                const Image* im = get_image(static_cast<size_t>(start + k));
                if (!im) { ok = false; break; }
                imgs.push_back(im);
            }
            if (!ok) { ++failures; continue; }
            const Image& center = *imgs[fi - static_cast<size_t>(start)];
            for (int k = 0; k < window; ++k) {
                if (imgs[static_cast<size_t>(k)]->width != center.width || imgs[static_cast<size_t>(k)]->height != center.height) {
                    std::fprintf(stderr, "frame size mismatch in window at %s\n", files[fi].c_str());
                    ok = false;
                    break;
                }
                dms_frame f{};
                f.struct_size = sizeof(f);
                f.rgb = imgs[static_cast<size_t>(k)]->rgb.data();
                f.width = center.width;
                f.height = center.height;
                f.row_stride_floats = center.width * 3;
                f.time_num = start + k;
                f.time_den = 24;
                frames.push_back(f);
            }
            if (!ok) { ++failures; continue; }

            const size_t n = static_cast<size_t>(center.width) * static_cast<size_t>(center.height);
            std::vector<float> depth(n), mapped(n);

            dms_depth_request rq{};
            rq.struct_size = sizeof(rq);
            rq.model_id = o.model.c_str();
            rq.frames = frames.data();
            rq.frame_count = window;
            rq.center_index = static_cast<int32_t>(fi) - start;
            rq.base_resolution = o.base_res;
            rq.edges_enhancement = o.edges;
            rq.subpixel_smoothing = o.smoothing;
            rq.backend.struct_size = sizeof(rq.backend);
            rq.backend.hardware_acceleration = o.ep != "cpu";
            rq.backend.fp16 = o.fp16;
            rq.backend.cuda_memory_sharing = o.memshare;
            rq.backend.offload_mode = o.offload;
            rq.backend.samples = o.samples;
            rq.backend.parallel_samples = o.parallel;
            rq.backend.tiles = o.tiles;
            rq.backend.memory_optimization = o.memopt;
            rq.backend.use_tensorrt = o.trt;
            rq.cache_key.struct_size = sizeof(rq.cache_key);
            rq.cache_key.layer_hash = o.no_cache ? 0 : std::hash<std::string>{}(files[fi]) | 1u;
            rq.cache_key.time_num = static_cast<int64_t>(fi);
            rq.cache_key.time_den = 24;
            rq.timeout_ms = o.timeout_ms;

            dms_depth_result rs{};
            rs.struct_size = sizeof(rs);
            rs.depth = depth.data();
            rs.width = center.width;
            rs.height = center.height;

            const auto t0 = std::chrono::steady_clock::now();
            s = dms_depth_compute(eng, &rq, &rs);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (s != DMS_OK && s != DMS_ERR_DEGRADED) {
                print_status(files[fi].c_str(), s);
                ++failures;
                continue;
            }
            total_ms += ms;
            ++processed;

            dms_mapping_params mp{};
            mp.struct_size = sizeof(mp);
            mp.mode = o.mapping == "raw" ? DMS_MAP_RAW : o.mapping == "guided" ? DMS_MAP_GUIDED : DMS_MAP_NORMALIZE;
            mp.near_x = o.near_x >= 0 ? o.near_x : static_cast<float>(center.width) * 0.5f;
            mp.near_y = o.near_y >= 0 ? o.near_y : static_cast<float>(center.height) * 0.9f;
            mp.far_x = o.far_x >= 0 ? o.far_x : static_cast<float>(center.width) * 0.5f;
            mp.far_y = o.far_y >= 0 ? o.far_y : static_cast<float>(center.height) * 0.1f;
            mp.near_value = o.near_value;
            mp.far_value = o.far_value;
            mp.raw_min = rs.raw_min;
            mp.raw_max = rs.raw_max;
            s = dms_map_depth(depth.data(), center.width, center.height, &mp, mapped.data());
            if (s == DMS_ERR_UNSUPPORTED) {
                mapped = depth;  // mapping not available yet (pre-milestone e)
            } else if (s != DMS_OK) {
                print_status("map_depth", s);
                ++failures;
                continue;
            }

            if (rep == o.repeat - 1) {
                const std::string stem = (fs::path(o.output_dir) / fs::path(files[fi]).stem()).string();
                bool wrote = false;
                if (o.mode == "slice") {
                    std::vector<float> mask(n);
                    s = dms_slice_mask(mapped.data(), center.width, center.height, o.slice_min, o.slice_max, o.smoothing / 100.f, mask.data());
                    if (s != DMS_OK) { print_status("slice_mask", s); ++failures; continue; }
                    wrote = depthcli::write_png_rgba8(stem + "_slice.png", center.rgb.data(), mask.data(), center.width, center.height);
                } else if (o.format == "pfm") {
                    wrote = depthcli::write_pfm(stem + "_depth.pfm", mapped.data(), center.width, center.height);
                } else if (o.format == "pgm16") {
                    wrote = depthcli::write_pgm16(stem + "_depth.pgm", mapped.data(), center.width, center.height);
                } else if (colormap > 0) {
                    std::vector<float> rgb(n * 3u);
                    s = dms_apply_colormap(mapped.data(), center.width, center.height, colormap, rgb.data());
                    if (s != DMS_OK) { print_status("apply_colormap", s); ++failures; continue; }
                    wrote = depthcli::write_png_rgb8(stem + "_depth.png", rgb.data(), center.width, center.height);
                } else {
                    wrote = depthcli::write_png_gray8(stem + "_depth.png", mapped.data(), center.width, center.height);
                }
                if (!wrote) { std::fprintf(stderr, "failed to write output for %s\n", files[fi].c_str()); ++failures; continue; }
            }
            if (!o.quiet)
                std::printf("%s  %dx%d  %.1f ms  backend=%d cache=%d fallbacks=%d %s%s\n", fs::path(files[fi]).filename().string().c_str(),
                            center.width, center.height, ms, rs.backend_used, rs.from_cache, rs.fallback_steps,
                            s == DMS_ERR_DEGRADED ? "[degraded] " : "", rs.message);
        }
    }
    if (o.bench || !o.quiet) {
        dms_cache_stats cs{};
        cs.struct_size = sizeof(cs);
        if (dms_cache_stats_query(eng, &cs) == DMS_OK)
            std::printf("cache: %llu entries, %.1f MB, hits=%llu misses=%llu evictions=%llu\n", static_cast<unsigned long long>(cs.entries),
                        static_cast<double>(cs.bytes_used) / 1e6, static_cast<unsigned long long>(cs.hits),
                        static_cast<unsigned long long>(cs.misses), static_cast<unsigned long long>(cs.evictions));
        std::printf("processed=%d failures=%d avg=%.1f ms\n", processed, failures, processed ? total_ms / processed : 0.0);
    }
    return failures == 0 ? 0 : 1;
}
