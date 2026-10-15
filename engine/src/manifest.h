// Model manifest: built-in entries (models_manifest.json compiled in) merged with an optional
// user override file <models_dir>/models.json (same schema; entries matched by id).
#pragma once

#include "dms/engine.h"

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace dms {

struct ModelSpec {
    std::string id;
    std::string display_name;
    int kind = DMS_MODEL_KIND_SINGLE;
    int window = 1;           // temporal window (frames per inference)
    int patch_multiple = 14;
    std::string input_layout = "BCHW";  // or BTCHW for temporal models
    std::string file;
    std::string url;
    std::string sha256;       // lowercase hex; empty = unknown (custom models)
    uint64_t size = 0;
    double params_millions = 0.0;
    std::string fp16_file;    // optional FP16 variant
    std::string fp16_url;
    std::string fp16_sha256;
    uint64_t fp16_size = 0;
    std::string license;
    bool custom_path = false; // id was an explicit .onnx path
};

class Manifest {
public:
    explicit Manifest(std::filesystem::path models_dir);
    // (Re)loads built-in + override. Malformed overrides are logged and ignored (never fatal).
    void reload();
    size_t count() const;
    std::optional<ModelSpec> at(size_t index) const;
    // Resolves an id, or an explicit path to an .onnx file (treated as a single-frame DA-style model).
    std::optional<ModelSpec> find(const std::string& id_or_path) const;
    std::filesystem::path path_for(const ModelSpec& m, bool fp16 = false) const;
    const std::filesystem::path& models_dir() const { return dir_; }

    // Parses a manifest JSON text into specs (exposed for tests). Throws dms::Error on malformed input.
    static std::vector<ModelSpec> parse(const std::string& json_text);

private:
    std::filesystem::path dir_;
    mutable std::mutex mu_;
    std::vector<ModelSpec> models_;
};

}  // namespace dms
