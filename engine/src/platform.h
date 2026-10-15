// Thin OS abstraction: paths, memory info, dynamic libraries.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace dms::platform {

std::filesystem::path default_data_dir();          // %LOCALAPPDATA%\DMS  |  $XDG_STATE_HOME/dms or ~/.local/state/dms
std::filesystem::path module_dir();                // directory of the binary containing the engine (the .aex / exe)
std::filesystem::path path_from_utf8(const std::string& s);
std::string path_to_utf8(const std::filesystem::path& p);

struct MemoryInfo {
    uint64_t total = 0;
    uint64_t available = 0;
};
MemoryInfo system_memory() noexcept;
uint64_t process_rss_bytes() noexcept;

// Dynamic library handle (RAII). Never throws on load failure: check ok().
class DynLib {
public:
    DynLib() = default;
    explicit DynLib(const std::filesystem::path& path, const std::filesystem::path& search_dir = {});
    ~DynLib();
    DynLib(const DynLib&) = delete;
    DynLib& operator=(const DynLib&) = delete;
    DynLib(DynLib&& o) noexcept : h_(o.h_), err_(std::move(o.err_)), pinned_(o.pinned_) { o.h_ = nullptr; }
    DynLib& operator=(DynLib&& o) noexcept;

    bool ok() const noexcept { return h_ != nullptr; }
    // Keeps the library loaded for the rest of the process (no FreeLibrary/dlclose on destruction).
    void pin() noexcept { pinned_ = true; }
    const std::string& error() const noexcept { return err_; }
    void* symbol(const char* name) const noexcept;
    template <typename F>
    F fn(const char* name) const noexcept { return reinterpret_cast<F>(symbol(name)); }

private:
    void* h_ = nullptr;
    std::string err_;
    bool pinned_ = false;
};

// Adds a directory to the DLL search path for subsequently loaded dependencies (Windows only; no-op elsewhere).
void add_dll_directory(const std::filesystem::path& dir) noexcept;

}  // namespace dms::platform
