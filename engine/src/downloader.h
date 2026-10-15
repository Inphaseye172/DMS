// Resumable, checksum-verified model downloader. One download at a time on a dedicated worker thread.
//   <models_dir>/<file>.part        partial data
//   <models_dir>/<file>.part.json   sidecar: url, expected sha256/size, bytes, SHA-256 mid-state
// On completion the SHA-256 is compared with the manifest; only then is the file renamed into place.
#pragma once

#include "dms/engine.h"
#include "manifest.h"
#include "transport.h"

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace dms {

struct DownloadStatus {
    int state = DMS_DL_IDLE;
    uint64_t done = 0;
    uint64_t total = 0;
    float bytes_per_second = 0.f;
    dms_status error = DMS_OK;
    std::string message;
};

class Downloader {
public:
    using TransportFactory = std::function<std::unique_ptr<Transport>(const std::string& url)>;
    explicit Downloader(TransportFactory factory = make_transport_for);
    ~Downloader();  // cancels and joins
    Downloader(const Downloader&) = delete;
    Downloader& operator=(const Downloader&) = delete;

    // Starts downloading `spec` to `dest`. DMS_ERR_BUSY if another download is running.
    dms_status start(const ModelSpec& spec, const std::filesystem::path& dest);
    DownloadStatus status(const std::string& model_id) const;
    void cancel(const std::string& model_id);
    // Blocks until the current download (if any) finishes. For tests/CLI.
    void wait();

    // Synchronous worker body (exposed for tests).
    static DownloadStatus run(const ModelSpec& spec, const std::filesystem::path& dest, Transport& transport, const std::atomic<bool>& cancel,
                              const std::function<void(const DownloadStatus&)>& progress);

private:
    void join_finished_locked();
    TransportFactory factory_;
    mutable std::mutex mu_;
    std::map<std::string, DownloadStatus> status_;
    std::string active_id_;
    std::thread worker_;
    std::atomic<bool> cancel_{false};
    std::atomic<bool> running_{false};
};

}  // namespace dms
