// Byte-range fetch abstraction for model downloads. Runs only on the engine's download worker thread.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace dms {

struct FetchResult {
    bool ok = false;
    bool range_honored = false;  // false = server restarted from byte 0
    uint64_t total_size = 0;     // full resource size if known (0 = unknown)
    std::string error;
};

class Transport {
public:
    virtual ~Transport() = default;
    // Streams the resource starting at `offset` into `sink`. `sink` returns false to cancel.
    virtual FetchResult fetch(const std::string& url, uint64_t offset, const std::function<bool(const void*, size_t)>& sink) = 0;
};

// file:// URLs (tests, offline installs, network shares).
std::unique_ptr<Transport> make_file_transport();
// Platform HTTP transport: WinHTTP on Windows; on POSIX dev machines, the `curl` executable if present.
std::unique_ptr<Transport> make_http_transport();
// Chooses by URL scheme.
std::unique_ptr<Transport> make_transport_for(const std::string& url);

}  // namespace dms
