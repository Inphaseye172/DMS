/*
 * DMS Depth Engine — C ABI.
 *
 * This is the only header the After Effects plugin (and the CLI) include. Rules:
 *   - Every function returns a dms_status; no exception ever crosses this boundary.
 *   - All structs carry struct_size for forward compatibility; zero-initialize them.
 *   - The engine is a process-wide, reference-counted singleton (dms_engine_acquire/release).
 *   - All functions are thread-safe unless noted. Inference is serialized internally.
 *   - No function blocks on network I/O; downloads run on an engine-owned worker thread.
 */
#ifndef DMS_ENGINE_H
#define DMS_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define DMS_API __cdecl
#else
#  define DMS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DMS_ENGINE_API_VERSION 1

typedef enum dms_status {
    DMS_OK = 0,
    DMS_ERR_INVALID_ARG = 1,
    DMS_ERR_NOT_INITIALIZED = 2,
    DMS_ERR_MODEL_NOT_INSTALLED = 3,
    DMS_ERR_MODEL_LOAD_FAILED = 4,
    DMS_ERR_BACKEND_UNAVAILABLE = 5,
    DMS_ERR_OUT_OF_MEMORY = 6,
    DMS_ERR_INFERENCE_FAILED = 7,
    DMS_ERR_TIMEOUT = 8,
    DMS_ERR_ABORTED = 9,
    DMS_ERR_IO = 10,
    DMS_ERR_NETWORK = 11,
    DMS_ERR_CHECKSUM = 12,
    DMS_ERR_INTERNAL = 13,
    DMS_ERR_BUSY = 14,
    DMS_ERR_UNSUPPORTED = 15,
    DMS_ERR_NOT_FOUND = 16,
    DMS_ERR_DEGRADED = 17 /* result produced through a fallback; message explains */
} dms_status;

typedef enum dms_log_level { DMS_LOG_ERROR = 0, DMS_LOG_WARN = 1, DMS_LOG_INFO = 2, DMS_LOG_DEBUG = 3 } dms_log_level;

typedef enum dms_backend_kind { DMS_BACKEND_NONE = 0, DMS_BACKEND_CPU = 1, DMS_BACKEND_CUDA = 2, DMS_BACKEND_DIRECTML = 3, DMS_BACKEND_TENSORRT = 4 } dms_backend_kind;

typedef enum dms_gpu_vendor { DMS_GPU_UNKNOWN = 0, DMS_GPU_NONE = 1, DMS_GPU_NVIDIA = 2, DMS_GPU_AMD = 3, DMS_GPU_INTEL = 4 } dms_gpu_vendor;

typedef enum dms_offload_mode { DMS_OFFLOAD_KEEP_VRAM = 0, DMS_OFFLOAD_TO_RAM = 1, DMS_OFFLOAD_FULL_UNLOAD = 2 } dms_offload_mode;

typedef enum dms_mapping_mode { DMS_MAP_RAW = 0, DMS_MAP_NORMALIZE = 1, DMS_MAP_GUIDED = 2 } dms_mapping_mode;

typedef enum dms_model_kind { DMS_MODEL_KIND_SINGLE = 0, DMS_MODEL_KIND_TEMPORAL = 1 } dms_model_kind;

typedef enum dms_download_state {
    DMS_DL_IDLE = 0, DMS_DL_RUNNING = 1, DMS_DL_VERIFYING = 2, DMS_DL_DONE = 3, DMS_DL_FAILED = 4, DMS_DL_CANCELLED = 5
} dms_download_state;

typedef struct dms_engine dms_engine; /* opaque */

/* Returns non-zero to request abort. Called from the engine on the caller's thread or the worker thread. */
typedef int (DMS_API *dms_abort_fn)(void* user);
/* fraction in [0,1]. Return non-zero to abort. */
typedef int (DMS_API *dms_progress_fn)(void* user, float fraction);

typedef struct dms_engine_config {
    uint32_t struct_size;
    const char* models_dir;         /* UTF-8. NULL = platform default (%LOCALAPPDATA%\DMS\models) */
    const char* log_dir;            /* UTF-8. NULL = platform default (%LOCALAPPDATA%\DMS\logs) */
    const char* deps_dir;           /* UTF-8. Directory containing ort-cuda/, ort-dml/, nvidia/ (NULL = next to the engine host binary) */
    int32_t log_level;              /* dms_log_level */
    uint64_t vram_reserve_bytes;    /* 0 = default 1 GiB */
    uint64_t cache_budget_bytes;    /* 0 = default 2 GiB (clamped to 1/4 of physical RAM) */
    uint32_t inference_timeout_ms;  /* 0 = default 120000 */
    uint32_t max_neighbor_frames;   /* 0 = default 32; hard cap on temporal window */
    uint32_t offload_idle_ms;       /* 0 = default 10000; idle time before offload policies kick in */
} dms_engine_config;

/* Acquire the process-wide engine (creates it on first call). Pair with dms_engine_release. */
dms_status DMS_API dms_engine_acquire(const dms_engine_config* config, dms_engine** out_engine);
dms_status DMS_API dms_engine_release(dms_engine* engine);
dms_status DMS_API dms_engine_set_log_level(dms_engine* engine, int32_t level);
/* Human readable message for a status code (static string, never NULL). */
const char* DMS_API dms_status_message(dms_status status);
/* Last detailed error message for the calling thread (static-lifetime buffer, never NULL). */
const char* DMS_API dms_last_error(void);
/* Version string of the engine. */
const char* DMS_API dms_engine_version(void);
/* Write a line to the engine log (used by the plugin layer). */
void DMS_API dms_log(dms_engine* engine, int32_t level, const char* message);

/* ----- Models --------------------------------------------------------------------------------- */

#define DMS_MAX_NAME 128
#define DMS_MAX_PATH 1024
#define DMS_MAX_MSG 512

typedef struct dms_model_info {
    uint32_t struct_size;
    char id[DMS_MAX_NAME];          /* stable id, e.g. "da2-small" */
    char display_name[DMS_MAX_NAME];
    int32_t kind;                   /* dms_model_kind */
    int32_t window_size;            /* temporal models: frames per window (1 for single-frame) */
    int32_t patch_multiple;         /* input dims must be multiples of this (14 for DINOv2) */
    uint64_t expected_bytes;        /* 0 if unknown */
    int32_t has_url;                /* 0 if the manifest has no download URL (user must install manually) */
    int32_t installed;              /* file exists in models dir */
    int32_t verified;               /* checksum verified at least once this process */
    char path[DMS_MAX_PATH];        /* UTF-8 path where the model is/would be stored */
} dms_model_info;

dms_status DMS_API dms_model_count(dms_engine* engine, int32_t* out_count);
dms_status DMS_API dms_model_info_at(dms_engine* engine, int32_t index, dms_model_info* out_info);
dms_status DMS_API dms_model_info_by_id(dms_engine* engine, const char* model_id, dms_model_info* out_info);
/* Verify the on-disk checksum synchronously (can take seconds for large models; do not call from render). */
dms_status DMS_API dms_model_verify(dms_engine* engine, const char* model_id);
/* Reload models.json overrides from the models directory. */
dms_status DMS_API dms_model_reload_manifest(dms_engine* engine);

typedef struct dms_download_info {
    uint32_t struct_size;
    int32_t state;                  /* dms_download_state */
    uint64_t bytes_done;
    uint64_t bytes_total;           /* 0 if unknown */
    float bytes_per_second;
    char message[DMS_MAX_MSG];      /* error text when FAILED */
} dms_download_info;

/* Start (or resume) a download on the engine worker thread. Returns DMS_ERR_BUSY if a download is running. */
dms_status DMS_API dms_download_start(dms_engine* engine, const char* model_id);
dms_status DMS_API dms_download_query(dms_engine* engine, const char* model_id, dms_download_info* out_info);
dms_status DMS_API dms_download_cancel(dms_engine* engine, const char* model_id);

/* ----- Device --------------------------------------------------------------------------------- */

typedef struct dms_device_info {
    uint32_t struct_size;
    int32_t gpu_vendor;             /* dms_gpu_vendor */
    char gpu_name[DMS_MAX_NAME];
    char driver_version[32];        /* e.g. "576.02" */
    uint64_t vram_total_bytes;
    uint64_t vram_free_bytes;
    uint64_t ram_total_bytes;
    uint64_t ram_free_bytes;
    int32_t cuda_available;         /* CUDA EP loadable (driver + DLLs) */
    int32_t directml_available;
    int32_t tensorrt_available;
    int32_t active_backend;         /* dms_backend_kind of the most recent session, NONE if none */
    char ort_version[32];
    char notes[DMS_MAX_MSG];        /* why something is unavailable */
} dms_device_info;

/* Full device report. Loads ONNX Runtime on first use (can take ~1 s): call from Options / CLI, not per frame. */
dms_status DMS_API dms_device_query(dms_engine* engine, dms_device_info* out_info);
/* Cheap GPU vendor query (GPU probe only, never loads ONNX Runtime). Safe for UI commands and per-frame use. */
dms_status DMS_API dms_gpu_vendor_query(dms_engine* engine, int32_t* out_vendor);

/* ----- Inference ------------------------------------------------------------------------------ */

typedef struct dms_backend_settings {
    uint32_t struct_size;
    int32_t hardware_acceleration;  /* 0 = force CPU */
    int32_t fp16;                   /* lower precision */
    int32_t cuda_memory_sharing;    /* NVIDIA only; keeps tensors device-resident */
    int32_t offload_mode;           /* dms_offload_mode */
    int32_t samples;                /* 1..16 */
    int32_t parallel_samples;       /* batch samples in one run (only meaningful when samples > 2) */
    int32_t tiles;                  /* 0 = auto, else 1/4/9/16 */
    int32_t memory_optimization;    /* temporal model only */
    int32_t use_tensorrt;           /* only if compiled with DMS_ENABLE_TENSORRT */
} dms_backend_settings;

/* Interleaved RGB float32 frame in [0,1] (alpha already handled by the caller). */
typedef struct dms_frame {
    uint32_t struct_size;
    const float* rgb;               /* width*3 floats per row at least */
    int32_t width;
    int32_t height;
    int32_t row_stride_floats;      /* >= width*3 */
    int64_t time_num;               /* frame time (for cache keys / temporal ordering) */
    int64_t time_den;
} dms_frame;

/* Cache key components. layer_hash = caller-defined hash of the source layer identity. */
typedef struct dms_cache_key {
    uint32_t struct_size;
    uint64_t layer_hash;
    int64_t time_num;
    int64_t time_den;
} dms_cache_key;

typedef struct dms_depth_request {
    uint32_t struct_size;
    const char* model_id;
    const dms_frame* frames;        /* frames[center_index] is the frame to produce depth for */
    int32_t frame_count;            /* 1 for single-frame models; up to window for temporal */
    int32_t center_index;
    int32_t base_resolution;        /* long side: 384 / 518 / 768 / 1024 */
    float edges_enhancement;        /* 0..100 */
    float subpixel_smoothing;       /* 0..100 */
    dms_backend_settings backend;
    dms_cache_key cache_key;        /* layer_hash == 0 disables caching */
    dms_abort_fn abort_fn;          /* may be NULL */
    dms_progress_fn progress_fn;    /* may be NULL */
    void* user;
    uint32_t timeout_ms;            /* 0 = engine default */
    /* Optional full-resolution center frame. When set, the output (and edge guide) uses its size and
     * frames[] may be downscaled copies (recommended for temporal windows to bound memory). */
    const dms_frame* guide;
} dms_depth_request;

typedef struct dms_depth_result {
    uint32_t struct_size;
    float* depth;                   /* caller-provided buffer of width*height floats (relative inverse depth, larger = nearer) */
    int32_t width;                  /* must equal frames[center].width */
    int32_t height;
    /* outputs */
    int32_t from_cache;
    int32_t backend_used;           /* dms_backend_kind */
    int32_t fallback_steps;         /* number of fallback transitions taken */
    float raw_min;
    float raw_max;
    float ms_total;
    char message[DMS_MAX_MSG];      /* fallback / degradation explanation (empty if none) */
} dms_depth_result;

/* Compute (or fetch from cache) a depth map. Thread-safe; inference serialized internally.
 * Returns DMS_OK, DMS_ERR_DEGRADED (output valid but produced by a fallback), or an error (output untouched). */
dms_status DMS_API dms_depth_compute(dms_engine* engine, const dms_depth_request* request, dms_depth_result* result);

/* Temporal models: the window of frame indices [*out_start, *out_start + *out_count) to supply for
 * `frame_index` (indices are in frames, relative to any fixed origin; negative results are possible
 * and simply mean "not available"). Windows overlap by 1/4 so consecutive windows can be aligned.
 * Single-frame models return start = frame_index, count = 1. */
dms_status DMS_API dms_temporal_window(dms_engine* engine, const char* model_id, int64_t frame_index, int32_t memory_optimization,
                                       int64_t* out_start, int32_t* out_count);

/* Cache-only lookup; DMS_ERR_NOT_FOUND if absent. Does not run inference. */
dms_status DMS_API dms_depth_cache_lookup(dms_engine* engine, const dms_cache_key* key, const char* model_id,
                                          int32_t base_resolution, const dms_backend_settings* backend,
                                          float edges, float smoothing, dms_depth_result* result);

typedef struct dms_cache_stats {
    uint32_t struct_size;
    uint64_t bytes_used;
    uint64_t bytes_budget;
    uint64_t entries;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
} dms_cache_stats;
dms_status DMS_API dms_cache_stats_query(dms_engine* engine, dms_cache_stats* out_stats);
dms_status DMS_API dms_cache_clear(dms_engine* engine);
dms_status DMS_API dms_cache_set_budget(dms_engine* engine, uint64_t bytes);

/* ----- Per-instance state (effect instances) -------------------------------------------------- */
/* Registers an effect instance (by 16-byte GUID). Instances keep a "last good depth" for fallbacks. */
dms_status DMS_API dms_instance_open(dms_engine* engine, const uint8_t guid[16]);
dms_status DMS_API dms_instance_close(dms_engine* engine, const uint8_t guid[16]);
/* Copies the last successfully computed depth of this instance, resized to width*height. DMS_ERR_NOT_FOUND if none. */
dms_status DMS_API dms_instance_last_depth(dms_engine* engine, const uint8_t guid[16], float* out, int32_t width, int32_t height);
dms_status DMS_API dms_instance_store_depth(dms_engine* engine, const uint8_t guid[16], const float* depth, int32_t width, int32_t height);
/* De-duplicates user-facing messages: *out_first = 1 the first time `code` is reported for this instance
 * since the last success (dms_instance_store_depth resets it), else 0. Prevents an error dialog per frame. */
dms_status DMS_API dms_instance_should_report(dms_engine* engine, const uint8_t guid[16], int32_t code, int32_t* out_first);

/* ----- Post-processing (pure CPU, re-entrant, no engine state) -------------------------------- */

typedef struct dms_mapping_params {
    uint32_t struct_size;
    int32_t mode;                   /* dms_mapping_mode */
    float near_x, near_y;           /* guided: pixel coordinates in the depth map */
    float far_x, far_y;
    float near_value;               /* guided: output value at the near point (0..1) */
    float far_value;
    float raw_min, raw_max;         /* normalize: optional precomputed range; if raw_min>=raw_max it is computed */
} dms_mapping_params;

/* Maps relative depth to display values. For Raw the values are copied; Normalize → [0,1] (1 = near);
 * Guided → linear fit through the two guide points, clamped to [0,1]. in/out may alias. */
dms_status DMS_API dms_map_depth(const float* depth, int32_t width, int32_t height, const dms_mapping_params* params, float* out);

/* Slicing: mask = 1 where mapped in [min,max], feathered by `feather` (0..1 fraction of range). */
dms_status DMS_API dms_slice_mask(const float* mapped, int32_t width, int32_t height, float min_value, float max_value, float feather, float* out_mask);

int32_t DMS_API dms_colormap_count(void);
const char* DMS_API dms_colormap_name(int32_t index); /* index 0 = "None" (grayscale) */
/* Applies a colormap: out_rgb has width*height*3 floats in [0,1]. index 0 → grayscale. Values are clamped to [0,1]. */
dms_status DMS_API dms_apply_colormap(const float* mapped, int32_t width, int32_t height, int32_t colormap_index, float* out_rgb);

/* Edge enhancement: guided filter of `depth` using `rgb` guide (interleaved, stride floats). strength 0..100. */
dms_status DMS_API dms_refine_edges(const float* depth, const float* rgb, int32_t rgb_stride_floats, int32_t width, int32_t height, float strength, float* out);
/* Subpixel smoothing: edge-aware smoothing of depth. strength 0..100. */
dms_status DMS_API dms_smooth_depth(const float* depth, int32_t width, int32_t height, float strength, float* out);

/* Bilinear resize of a single-channel float image. */
dms_status DMS_API dms_resize_f32(const float* src, int32_t src_w, int32_t src_h, float* dst, int32_t dst_w, int32_t dst_h);

/* SHA-256 of a buffer as lowercase hex (65 bytes incl. NUL). */
dms_status DMS_API dms_sha256_hex(const void* data, size_t size, char out_hex[65]);

#ifdef __cplusplus
}
#endif
#endif /* DMS_ENGINE_H */
