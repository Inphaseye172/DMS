#include "manifest.h"

#include "common.h"
#include "logging.h"
#include "platform.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace dms {

namespace {
#include "models_manifest.inc"  // defines kBuiltinManifest (generated from models_manifest.json)

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool safe_filename(const std::string& f) {
    // Model files must stay inside the models dir: no separators, no "..", not empty.
    if (f.empty() || f.size() > 200) return false;
    if (f.find('/') != std::string::npos || f.find('\\') != std::string::npos || f.find("..") != std::string::npos) return false;
    if (f.find(':') != std::string::npos) return false;
    return true;
}

bool valid_sha(const std::string& s) {
    if (s.empty()) return true;
    if (s.size() != 64) return false;
    return std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

ModelSpec from_json(const nlohmann::json& j) {
    ModelSpec m;
    m.id = j.at("id").get<std::string>();
    if (m.id.empty() || m.id.size() >= DMS_MAX_NAME) fail(DMS_ERR_INVALID_ARG, "manifest: invalid id");
    m.display_name = j.value("display_name", m.id);
    if (m.display_name.size() >= DMS_MAX_NAME) m.display_name.resize(DMS_MAX_NAME - 1);
    const std::string kind = j.value("kind", "single");
    m.kind = kind == "temporal" ? DMS_MODEL_KIND_TEMPORAL : DMS_MODEL_KIND_SINGLE;
    m.window = clampi(j.value("window", m.kind == DMS_MODEL_KIND_TEMPORAL ? 32 : 1), 1, 64);
    m.patch_multiple = clampi(j.value("patch_multiple", 14), 1, 64);
    m.input_layout = j.value("input_layout", m.kind == DMS_MODEL_KIND_TEMPORAL ? "BTCHW" : "BCHW");
    if (m.input_layout != "BCHW" && m.input_layout != "BTCHW") fail(DMS_ERR_INVALID_ARG, "manifest: invalid input_layout for " + m.id);
    m.file = j.value("file", "");
    if (!safe_filename(m.file)) fail(DMS_ERR_INVALID_ARG, "manifest: unsafe file name for " + m.id);
    m.url = j.value("url", "");
    m.sha256 = lower(j.value("sha256", ""));
    if (!valid_sha(m.sha256)) fail(DMS_ERR_INVALID_ARG, "manifest: invalid sha256 for " + m.id);
    if (!m.url.empty() && m.sha256.empty()) fail(DMS_ERR_INVALID_ARG, "manifest: url without sha256 for " + m.id);
    m.size = j.value("size", static_cast<uint64_t>(0));
    m.params_millions = j.value("params_millions", 0.0);
    if (j.contains("fp16") && j["fp16"].is_object()) {
        const auto& f = j["fp16"];
        m.fp16_file = f.value("file", "");
        m.fp16_url = f.value("url", "");
        m.fp16_sha256 = lower(f.value("sha256", ""));
        m.fp16_size = f.value("size", static_cast<uint64_t>(0));
        if (!m.fp16_file.empty() && !safe_filename(m.fp16_file)) fail(DMS_ERR_INVALID_ARG, "manifest: unsafe fp16 file for " + m.id);
        if (!valid_sha(m.fp16_sha256)) fail(DMS_ERR_INVALID_ARG, "manifest: invalid fp16 sha256 for " + m.id);
    }
    m.license = j.value("license", "");
    return m;
}
}  // namespace

std::vector<ModelSpec> Manifest::parse(const std::string& text) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const std::exception& e) {
        fail(DMS_ERR_INVALID_ARG, std::string("manifest: JSON parse error: ") + e.what());
    }
    if (!j.is_object() || !j.contains("models") || !j["models"].is_array()) fail(DMS_ERR_INVALID_ARG, "manifest: missing 'models' array");
    std::vector<ModelSpec> out;
    for (const auto& e : j["models"]) {
        try {
            out.push_back(from_json(e));
        } catch (const nlohmann::json::exception& ex) {
            fail(DMS_ERR_INVALID_ARG, std::string("manifest: bad entry: ") + ex.what());
        }
    }
    return out;
}

Manifest::Manifest(std::filesystem::path models_dir) : dir_(std::move(models_dir)) { reload(); }

void Manifest::reload() {
    std::vector<ModelSpec> models = parse(kBuiltinManifest);
    const auto override_path = dir_ / "models.json";
    std::error_code ec;
    if (std::filesystem::exists(override_path, ec)) {
        try {
            std::ifstream f(override_path, std::ios::binary);
            std::stringstream ss;
            ss << f.rdbuf();
            for (auto& o : parse(ss.str())) {
                auto it = std::find_if(models.begin(), models.end(), [&](const ModelSpec& m) { return m.id == o.id; });
                if (it != models.end()) *it = std::move(o);
                else models.push_back(std::move(o));
            }
            logf(DMS_LOG_INFO, "loaded model manifest override %s", platform::path_to_utf8(override_path).c_str());
        } catch (const std::exception& e) {
            logf(DMS_LOG_WARN, "ignoring invalid model manifest override %s: %s", platform::path_to_utf8(override_path).c_str(), e.what());
        }
    }
    std::lock_guard<std::mutex> lk(mu_);
    models_ = std::move(models);
}

size_t Manifest::count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return models_.size();
}

std::optional<ModelSpec> Manifest::at(size_t index) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (index >= models_.size()) return std::nullopt;
    return models_[index];
}

std::optional<ModelSpec> Manifest::find(const std::string& id) const {
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& m : models_)
            if (m.id == id) return m;
    }
    // Explicit model path (CLI / tests): must exist and end in .onnx.
    if (id.size() > 5 && lower(id.substr(id.size() - 5)) == ".onnx") {
        const auto p = platform::path_from_utf8(id);
        std::error_code ec;
        if (std::filesystem::is_regular_file(p, ec)) {
            ModelSpec m;
            m.id = id.size() < DMS_MAX_NAME ? id : id.substr(id.size() - (DMS_MAX_NAME - 1));
            m.display_name = platform::path_to_utf8(p.filename());
            m.file = platform::path_to_utf8(std::filesystem::absolute(p, ec));
            m.custom_path = true;
            return m;
        }
    }
    return std::nullopt;
}

std::filesystem::path Manifest::path_for(const ModelSpec& m, bool fp16) const {
    if (m.custom_path) return platform::path_from_utf8(m.file);
    return dir_ / platform::path_from_utf8(fp16 && !m.fp16_file.empty() ? m.fp16_file : m.file);
}

}  // namespace dms
