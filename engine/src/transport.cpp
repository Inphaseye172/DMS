#include "transport.h"

#include "platform.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <winhttp.h>
#else
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace dms {
namespace {

std::string url_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const std::string hex = s.substr(i + 1, 2);
            o.push_back(static_cast<char>(std::strtol(hex.c_str(), nullptr, 16)));
            i += 2;
        } else {
            o.push_back(s[i]);
        }
    }
    return o;
}

class FileTransport : public Transport {
public:
    FetchResult fetch(const std::string& url, uint64_t offset, const std::function<bool(const void*, size_t)>& sink) override {
        FetchResult r;
        std::string path = url.rfind("file://", 0) == 0 ? url_decode(url.substr(7)) : url;
#if defined(_WIN32)
        if (path.size() > 2 && path[0] == '/' && path[2] == ':') path = path.substr(1);  // file:///C:/x
#endif
        const auto p = platform::path_from_utf8(path);
        std::ifstream f(p, std::ios::binary);
        if (!f) {
            r.error = "cannot open " + path;
            return r;
        }
        f.seekg(0, std::ios::end);
        r.total_size = static_cast<uint64_t>(f.tellg());
        if (offset > r.total_size) offset = 0;
        f.seekg(static_cast<std::streamoff>(offset));
        r.range_honored = true;
        std::vector<char> buf(1u << 18);
        while (f) {
            f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
            const auto n = f.gcount();
            if (n > 0 && !sink(buf.data(), static_cast<size_t>(n))) {
                r.error = "cancelled";
                return r;
            }
        }
        r.ok = !f.bad();
        if (!r.ok) r.error = "read error";
        return r;
    }
};

#if defined(_WIN32)
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

struct WinHttpHandle {
    HINTERNET h = nullptr;
    explicit WinHttpHandle(HINTERNET x) : h(x) {}
    ~WinHttpHandle() {
        if (h) WinHttpCloseHandle(h);
    }
    WinHttpHandle(const WinHttpHandle&) = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;
};

// WinHTTP with system proxy settings, TLS verification on, redirects followed (GitHub releases redirect
// to a CDN), explicit timeouts so a stalled connection fails instead of hanging the worker forever.
class WinHttpTransport : public Transport {
public:
    FetchResult fetch(const std::string& url, uint64_t offset, const std::function<bool(const void*, size_t)>& sink) override {
        FetchResult r;
        const std::wstring wurl = widen(url);
        URL_COMPONENTS uc{};
        uc.dwStructSize = sizeof(uc);
        wchar_t host[256] = {0};
        wchar_t path[4096] = {0};
        uc.lpszHostName = host;
        uc.dwHostNameLength = 255;
        uc.lpszUrlPath = path;
        uc.dwUrlPathLength = 4095;
        if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) {
            r.error = "invalid URL";
            return r;
        }
        if (uc.nScheme != INTERNET_SCHEME_HTTPS) {
            r.error = "only https:// downloads are allowed";
            return r;
        }
        WinHttpHandle session(WinHttpOpen(L"DMS-Depth/1.0", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
        if (!session.h) {
            r.error = "WinHttpOpen failed (" + std::to_string(GetLastError()) + ")";
            return r;
        }
        WinHttpSetTimeouts(session.h, 15000, 15000, 30000, 60000);
        WinHttpHandle conn(WinHttpConnect(session.h, host, uc.nPort, 0));
        if (!conn.h) {
            r.error = "WinHttpConnect failed (" + std::to_string(GetLastError()) + ")";
            return r;
        }
        WinHttpHandle req(WinHttpOpenRequest(conn.h, L"GET", path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE));
        if (!req.h) {
            r.error = "WinHttpOpenRequest failed (" + std::to_string(GetLastError()) + ")";
            return r;
        }
        DWORD redirect = WINHTTP_OPTION_REDIRECT_POLICY_DISALLOW_HTTPS_TO_HTTP;
        WinHttpSetOption(req.h, WINHTTP_OPTION_REDIRECT_POLICY, &redirect, sizeof(redirect));
        std::wstring headers;
        if (offset > 0) headers = L"Range: bytes=" + std::to_wstring(offset) + L"-\r\n";
        if (!WinHttpSendRequest(req.h, headers.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : headers.c_str(), headers.empty() ? 0 : static_cast<DWORD>(-1L),
                                WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
            !WinHttpReceiveResponse(req.h, nullptr)) {
            r.error = "HTTP request failed (" + std::to_string(GetLastError()) + ")";
            return r;
        }
        DWORD status = 0, sz = sizeof(status);
        WinHttpQueryHeaders(req.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz, WINHTTP_NO_HEADER_INDEX);
        if (status != 200 && status != 206) {
            r.error = "HTTP status " + std::to_string(status);
            return r;
        }
        r.range_honored = status == 206 || offset == 0;
        wchar_t len_buf[64] = {0};
        DWORD len_sz = sizeof(len_buf);
        if (WinHttpQueryHeaders(req.h, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, len_buf, &len_sz, WINHTTP_NO_HEADER_INDEX))
            r.total_size = static_cast<uint64_t>(_wcstoui64(len_buf, nullptr, 10)) + (status == 206 ? offset : 0);
        std::vector<char> buf(1u << 18);
        for (;;) {
            DWORD avail = 0;
            if (!WinHttpQueryDataAvailable(req.h, &avail)) {
                r.error = "connection lost (" + std::to_string(GetLastError()) + ")";
                return r;
            }
            if (avail == 0) break;
            DWORD got = 0;
            const DWORD want = avail < buf.size() ? avail : static_cast<DWORD>(buf.size());
            if (!WinHttpReadData(req.h, buf.data(), want, &got)) {
                r.error = "read failed (" + std::to_string(GetLastError()) + ")";
                return r;
            }
            if (got == 0) break;
            if (!sink(buf.data(), got)) {
                r.error = "cancelled";
                return r;
            }
        }
        r.ok = true;
        return r;
    }
};
#else
// POSIX developer convenience: stream through the `curl` executable (not shipped; Windows uses WinHTTP).
class CurlTransport : public Transport {
public:
    FetchResult fetch(const std::string& url, uint64_t offset, const std::function<bool(const void*, size_t)>& sink) override {
        FetchResult r;
        if (url.rfind("https://", 0) != 0) {
            r.error = "only https:// downloads are allowed";
            return r;
        }
        int fds[2];
        if (pipe(fds) != 0) {
            r.error = "pipe failed";
            return r;
        }
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&fa, fds[0]);
        const std::string range = std::to_string(offset) + "-";
        std::vector<std::string> args = {"curl", "-sSfL", "--connect-timeout", "15", "--speed-time", "60", "--speed-limit", "1024"};
        if (offset > 0) {
            args.push_back("-r");
            args.push_back(range);
        }
        args.push_back(url);
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        argv.push_back(nullptr);
        pid_t pid = 0;
        const int rc = posix_spawnp(&pid, "curl", &fa, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        close(fds[1]);
        if (rc != 0) {
            close(fds[0]);
            r.error = "curl not available";
            return r;
        }
        std::vector<char> buf(1u << 18);
        bool cancelled = false;
        for (;;) {
            const ssize_t n = read(fds[0], buf.data(), buf.size());
            if (n <= 0) break;
            if (!sink(buf.data(), static_cast<size_t>(n))) {
                cancelled = true;
                kill(pid, SIGTERM);
                break;
            }
        }
        close(fds[0]);
        int st = 0;
        waitpid(pid, &st, 0);
        if (cancelled) {
            r.error = "cancelled";
            return r;
        }
        r.ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
        r.range_honored = true;  // curl -r fails (-f) or the server sends the requested range
        if (!r.ok) r.error = "curl exited with status " + std::to_string(WIFEXITED(st) ? WEXITSTATUS(st) : -1);
        return r;
    }
};
#endif

}  // namespace

std::unique_ptr<Transport> make_file_transport() { return std::make_unique<FileTransport>(); }

std::unique_ptr<Transport> make_http_transport() {
#if defined(_WIN32)
    return std::make_unique<WinHttpTransport>();
#else
    return std::make_unique<CurlTransport>();
#endif
}

std::unique_ptr<Transport> make_transport_for(const std::string& url) {
    if (url.rfind("file://", 0) == 0) return make_file_transport();
    return make_http_transport();
}

}  // namespace dms
