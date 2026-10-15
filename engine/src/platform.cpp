#include "platform.h"

#include <cstdlib>
#include <system_error>

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#  include <shlobj.h>
#else
#  include <dlfcn.h>
#  include <sys/sysinfo.h>
#  include <unistd.h>
#  include <cstdio>
#endif

namespace dms::platform {

std::filesystem::path path_from_utf8(const std::string& s) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
#else
    return std::filesystem::u8path(s);
#endif
}

std::string path_to_utf8(const std::filesystem::path& p) {
    const auto u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}

std::filesystem::path default_data_dir() {
#if defined(_WIN32)
    PWSTR raw = nullptr;
    std::filesystem::path out;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &raw)) && raw) {
        out = std::filesystem::path(raw) / L"DMS";
    } else {
        out = std::filesystem::temp_directory_path() / L"DMS";
    }
    if (raw) CoTaskMemFree(raw);
    return out;
#else
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read-only, the engine never calls setenv
    if (const char* x = std::getenv("XDG_STATE_HOME"); x && *x) return std::filesystem::path(x) / "dms";
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read-only, the engine never calls setenv
    if (const char* h = std::getenv("HOME"); h && *h) return std::filesystem::path(h) / ".local" / "state" / "dms";
    return std::filesystem::temp_directory_path() / "dms";
#endif
}

std::filesystem::path module_dir() {
#if defined(_WIN32)
    HMODULE hm = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&module_dir), &hm))
        return {};
    wchar_t buf[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(hm, buf, static_cast<DWORD>(sizeof(buf) / sizeof(buf[0])));
    if (n == 0) return {};
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&module_dir), &info) && info.dli_fname) {
        std::error_code ec;
        auto p = std::filesystem::weakly_canonical(info.dli_fname, ec);
        return (ec ? std::filesystem::path(info.dli_fname) : p).parent_path();
    }
    return {};
#endif
}

MemoryInfo system_memory() noexcept {
    MemoryInfo m;
#if defined(_WIN32)
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        m.total = st.ullTotalPhys;
        m.available = st.ullAvailPhys;
    }
#else
    // Prefer MemAvailable from /proc/meminfo; fall back to sysinfo.
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long v = 0;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::sscanf(line, "MemTotal: %llu kB", &v) == 1) m.total = v * 1024ull;
            else if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) m.available = v * 1024ull;
        }
        std::fclose(f);
    }
    if (m.total == 0) {
        struct sysinfo si{};
        if (sysinfo(&si) == 0) {
            m.total = static_cast<uint64_t>(si.totalram) * si.mem_unit;
            m.available = static_cast<uint64_t>(si.freeram) * si.mem_unit;
        }
    }
#endif
    return m;
}

uint64_t process_rss_bytes() noexcept {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.WorkingSetSize;
    return 0;
#else
    long pages = 0, rss = 0;
    if (std::FILE* f = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
        std::fclose(f);
    }
    return static_cast<uint64_t>(rss) * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
#endif
}

DynLib::DynLib(const std::filesystem::path& path, const std::filesystem::path& search_dir) {
#if defined(_WIN32)
    (void)search_dir;
    // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: dependencies are resolved from the DLL's own folder first,
    // never from the AE install folder or PATH (avoids picking up a conflicting onnxruntime.dll).
    const DWORD flags = path.is_absolute() ? (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) : 0;
    const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
    HMODULE m = LoadLibraryExW(path.c_str(), nullptr, flags);
    SetErrorMode(old_mode);
    h_ = reinterpret_cast<void*>(m);
    if (!h_) err_ = "LoadLibraryExW failed for " + path_to_utf8(path) + " (error " + std::to_string(GetLastError()) + ")";
#else
    (void)search_dir;
    h_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h_) {
        const char* e = dlerror();  // NOLINT(concurrency-mt-unsafe): glibc keeps dlerror state per thread
        err_ = e ? e : ("dlopen failed for " + path.string());
    }
#endif
}

DynLib& DynLib::operator=(DynLib&& o) noexcept {
    if (this != &o) {
        this->~DynLib();
        h_ = o.h_;
        err_ = std::move(o.err_);
        pinned_ = o.pinned_;
        o.h_ = nullptr;
    }
    return *this;
}

DynLib::~DynLib() {
    if (!h_ || pinned_) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(h_));
#else
    dlclose(h_);
#endif
    h_ = nullptr;
}

void* DynLib::symbol(const char* name) const noexcept {
    if (!h_ || !name) return nullptr;
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(h_), name));
#else
    return dlsym(h_, name);
#endif
}

void add_dll_directory(const std::filesystem::path& dir) noexcept {
#if defined(_WIN32)
    std::error_code ec;
    if (std::filesystem::is_directory(dir, ec)) AddDllDirectory(dir.c_str());
#else
    (void)dir;
#endif
}

}  // namespace dms::platform
