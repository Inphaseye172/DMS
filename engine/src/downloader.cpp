#include "downloader.h"

#include "common.h"
#include "logging.h"
#include "platform.h"
#include "sha256.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace dms {

namespace {

constexpr uint64_t kSidecarEvery = 8ull << 20;  // persist resume state every 8 MiB
constexpr uint64_t kDiskSlack = 64ull << 20;

std::string state_to_json(const Sha256::State& s) {
    nlohmann::json j;
    j["h"] = std::vector<uint32_t>(s.h.begin(), s.h.end());
    j["buf"] = to_hex(s.buf.data(), s.buf.size());
    j["total"] = s.total;
    j["buf_len"] = s.buf_len;
    return j.dump();
}

bool state_from_json(const nlohmann::json& j, Sha256::State& s) {
    try {
        const auto h = j.at("h").get<std::vector<uint32_t>>();
        const auto buf = j.at("buf").get<std::string>();
        if (h.size() != 8 || buf.size() != 128) return false;
        for (size_t i = 0; i < 8; ++i) s.h[i] = h[i];
        for (size_t i = 0; i < 64; ++i) s.buf[i] = static_cast<uint8_t>(std::stoul(buf.substr(i * 2, 2), nullptr, 16));
        s.total = j.at("total").get<uint64_t>();
        s.buf_len = j.at("buf_len").get<uint32_t>();
        return s.buf_len < 64;
    } catch (...) {
        return false;
    }
}

void write_sidecar(const std::filesystem::path& p, const ModelSpec& spec, uint64_t bytes, const Sha256& h) {
    nlohmann::json j;
    j["url"] = spec.url;
    j["sha256"] = spec.sha256;
    j["size"] = spec.size;
    j["bytes"] = bytes;
    j["sha_state"] = nlohmann::json::parse(state_to_json(h.state()));
    const auto tmp = p.string() + ".tmp";
    {
        std::ofstream f(platform::path_from_utf8(tmp), std::ios::binary | std::ios::trunc);
        f << j.dump();
    }
    std::error_code ec;
    std::filesystem::rename(platform::path_from_utf8(tmp), p, ec);
}

}  // namespace

Downloader::Downloader(TransportFactory factory) : factory_(std::move(factory)) {}

Downloader::~Downloader() {
    cancel_ = true;
    std::thread t;
    {
        std::lock_guard<std::mutex> lk(mu_);
        t = std::move(worker_);
    }
    if (t.joinable()) t.join();
}

void Downloader::join_finished_locked() {
    if (!running_ && worker_.joinable()) worker_.join();
}

dms_status Downloader::start(const ModelSpec& spec, const std::filesystem::path& dest) {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_) return active_id_ == spec.id ? DMS_OK : DMS_ERR_BUSY;
    join_finished_locked();
    if (spec.url.empty()) fail(DMS_ERR_NOT_FOUND, "no download URL configured for " + spec.id);
    if (spec.sha256.empty()) fail(DMS_ERR_CHECKSUM, "no checksum configured for " + spec.id + " (refusing unverified download)");
    cancel_ = false;
    running_ = true;
    active_id_ = spec.id;
    DownloadStatus st;
    st.state = DMS_DL_RUNNING;
    st.total = spec.size;
    status_[spec.id] = st;
    std::unique_ptr<Transport> transport = factory_(spec.url);
    worker_ = std::thread([this, spec, dest, tr = std::shared_ptr<Transport>(std::move(transport))]() {
        DownloadStatus final_status;
        try {
            final_status = run(spec, dest, *tr, cancel_, [this, &spec](const DownloadStatus& s) {
                std::lock_guard<std::mutex> lk2(mu_);
                status_[spec.id] = s;
            });
        } catch (const std::exception& e) {
            final_status.state = DMS_DL_FAILED;
            final_status.error = DMS_ERR_INTERNAL;
            final_status.message = e.what();
        } catch (...) {
            final_status.state = DMS_DL_FAILED;
            final_status.error = DMS_ERR_INTERNAL;
            final_status.message = "unknown error";
        }
        logf(final_status.state == DMS_DL_DONE ? DMS_LOG_INFO : DMS_LOG_WARN, "download %s finished: state=%d %s", spec.id.c_str(),
             final_status.state, final_status.message.c_str());
        std::lock_guard<std::mutex> lk2(mu_);
        status_[spec.id] = final_status;
        running_ = false;
    });
    return DMS_OK;
}

DownloadStatus Downloader::status(const std::string& id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = status_.find(id);
    return it == status_.end() ? DownloadStatus{} : it->second;
}

void Downloader::cancel(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_ && active_id_ == id) cancel_ = true;
}

void Downloader::wait() {
    std::thread t;
    {
        std::lock_guard<std::mutex> lk(mu_);
        t = std::move(worker_);
    }
    if (t.joinable()) t.join();
}

DownloadStatus Downloader::run(const ModelSpec& spec, const std::filesystem::path& dest, Transport& transport, const std::atomic<bool>& cancel,
                               const std::function<void(const DownloadStatus&)>& progress) {
    DownloadStatus st;
    st.state = DMS_DL_RUNNING;
    st.total = spec.size;
    auto failed = [&](dms_status e, const std::string& msg) {
        st.state = DMS_DL_FAILED;
        st.error = e;
        st.message = msg;
        return st;
    };
    std::error_code ec;
    std::filesystem::create_directories(dest.parent_path(), ec);
    const auto part = platform::path_from_utf8(platform::path_to_utf8(dest) + ".part");
    const auto sidecar = platform::path_from_utf8(platform::path_to_utf8(dest) + ".part.json");

    // Resume state: only if the sidecar describes the same artifact and matches the .part size.
    Sha256 hasher;
    uint64_t have = 0;
    if (std::filesystem::exists(part, ec) && std::filesystem::exists(sidecar, ec)) {
        try {
            std::ifstream f(sidecar, std::ios::binary);
            std::stringstream ss;
            ss << f.rdbuf();
            const auto j = nlohmann::json::parse(ss.str());
            const uint64_t part_size = std::filesystem::file_size(part, ec);
            Sha256::State s{};
            if (!ec && j.value("url", "") == spec.url && j.value("sha256", "") == spec.sha256 && j.value("bytes", 0ull) <= part_size &&
                state_from_json(j.at("sha_state"), s) && s.total == j.value("bytes", 0ull)) {
                have = j.value("bytes", 0ull);
                hasher.restore(s);
                // Trim anything written after the last persisted state.
                if (part_size != have) std::filesystem::resize_file(part, have, ec);
                if (ec) {
                    have = 0;
                    hasher.reset();
                }
            }
        } catch (...) {
            have = 0;
            hasher.reset();
        }
    }
    if (have == 0) {
        std::filesystem::remove(part, ec);
        std::filesystem::remove(sidecar, ec);
    }
    if (spec.size && have > spec.size) {
        have = 0;
        hasher.reset();
        std::filesystem::remove(part, ec);
    }

    // Disk space check before writing anything.
    const auto space = std::filesystem::space(dest.parent_path(), ec);
    if (!ec && spec.size && space.available < (spec.size - have) + kDiskSlack)
        return failed(DMS_ERR_IO, "not enough disk space in " + platform::path_to_utf8(dest.parent_path()));

    std::ofstream out(part, std::ios::binary | (have ? std::ios::app : std::ios::trunc));
    if (!out) return failed(DMS_ERR_IO, "cannot write " + platform::path_to_utf8(part));
    if (have) logf(DMS_LOG_INFO, "resuming download of %s at %llu bytes", spec.id.c_str(), static_cast<unsigned long long>(have));

    st.done = have;
    uint64_t since_sidecar = 0;
    bool overflow = false;
    auto t_start = std::chrono::steady_clock::now();
    const uint64_t start_bytes = have;
    auto last_report = t_start;
    FetchResult fr = transport.fetch(spec.url, have, [&](const void* data, size_t n) -> bool {
        if (cancel) return false;
        if (spec.size && st.done + n > spec.size) {
            overflow = true;
            return false;
        }
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
        if (!out) return false;
        hasher.update(data, n);
        st.done += n;
        since_sidecar += n;
        if (since_sidecar >= kSidecarEvery) {
            out.flush();
            write_sidecar(sidecar, spec, st.done, hasher);
            since_sidecar = 0;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report > std::chrono::milliseconds(150)) {
            const double secs = std::chrono::duration<double>(now - t_start).count();
            st.bytes_per_second = secs > 0 ? static_cast<float>(static_cast<double>(st.done - start_bytes) / secs) : 0.f;
            last_report = now;
            if (progress) progress(st);
        }
        return true;
    });
    out.flush();
    const bool write_ok = static_cast<bool>(out);
    out.close();

    if (cancel) {
        write_sidecar(sidecar, spec, st.done, hasher);  // keep partial data for resume
        st.state = DMS_DL_CANCELLED;
        st.message = "cancelled (partial download kept for resume)";
        return st;
    }
    if (overflow || (fr.ok && !fr.range_honored && have > 0)) {
        std::filesystem::remove(part, ec);
        std::filesystem::remove(sidecar, ec);
        return failed(DMS_ERR_NETWORK, overflow ? "server sent more data than expected; partial file discarded, retry" : "server ignored resume request; retry");
    }
    if (!write_ok) return failed(DMS_ERR_IO, "write error (disk full?)");
    if (!fr.ok) {
        write_sidecar(sidecar, spec, st.done, hasher);
        return failed(DMS_ERR_NETWORK, "network error: " + fr.error + " (will resume)");
    }

    st.state = DMS_DL_VERIFYING;
    if (progress) progress(st);
    const std::string got = hasher.finish_hex();
    if (spec.size && st.done != spec.size) {
        std::filesystem::remove(part, ec);
        std::filesystem::remove(sidecar, ec);
        return failed(DMS_ERR_CHECKSUM, "size mismatch: got " + std::to_string(st.done) + " expected " + std::to_string(spec.size));
    }
    if (got != spec.sha256) {
        std::filesystem::remove(part, ec);
        std::filesystem::remove(sidecar, ec);
        return failed(DMS_ERR_CHECKSUM, "checksum mismatch (got " + got + ")");
    }
    std::filesystem::remove(dest, ec);
    std::filesystem::rename(part, dest, ec);
    if (ec) return failed(DMS_ERR_IO, "cannot move download into place: " + ec.message());
    std::filesystem::remove(sidecar, ec);
    st.state = DMS_DL_DONE;
    st.message.clear();
    return st;
}

}  // namespace dms
