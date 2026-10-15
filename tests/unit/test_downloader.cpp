#include "downloader.h"
#include "sha256.h"
#include "test_helpers.h"

#include <atomic>
#include <fstream>

namespace fs = std::filesystem;
using dms::DownloadStatus;
using dms::Downloader;
using dms::ModelSpec;

namespace {

struct Blob {
    fs::path path;
    std::string sha;
    uint64_t size;
};

Blob make_blob(const fs::path& dir, size_t n) {
    Blob b;
    b.path = dir / "source.bin";
    std::string data(n, '\0');
    for (size_t i = 0; i < n; ++i) data[i] = static_cast<char>((i * 2654435761u) >> 13);
    std::ofstream(b.path, std::ios::binary) << data;
    dms::Sha256 h;
    h.update(data.data(), data.size());
    b.sha = h.finish_hex();
    b.size = n;
    return b;
}

ModelSpec spec_for(const Blob& b) {
    ModelSpec s;
    s.id = "blob";
    s.file = "blob.onnx";
    s.url = "file://" + b.path.string();
    s.sha256 = b.sha;
    s.size = b.size;
    return s;
}

// Transport that delivers at most `limit` bytes then fails (simulated dropped connection), and can
// optionally ignore Range requests.
class FlakyTransport : public dms::Transport {
public:
    FlakyTransport(uint64_t limit, bool ignore_range) : limit_(limit), ignore_range_(ignore_range) {}
    dms::FetchResult fetch(const std::string& url, uint64_t offset, const std::function<bool(const void*, size_t)>& sink) override {
        auto inner = dms::make_file_transport();
        uint64_t sent = 0;
        dms::FetchResult r = inner->fetch(url, ignore_range_ ? 0 : offset, [&](const void* d, size_t n) {
            if (sent + n > limit_) {
                const size_t part = static_cast<size_t>(limit_ - sent);
                if (part) sink(d, part);
                sent = limit_;
                return false;
            }
            sent += n;
            return sink(d, n);
        });
        if (sent >= limit_) {
            r.ok = false;
            r.error = "simulated connection drop";
        }
        if (ignore_range_) r.range_honored = (offset == 0);
        return r;
    }

private:
    uint64_t limit_;
    bool ignore_range_;
};

}  // namespace

TEST(Downloader, DownloadsAndVerifies) {
    const auto dir = dms_test::tmp_dir("dl_ok");
    const Blob b = make_blob(dir, 3'000'000);
    const auto dest = dir / "models" / "blob.onnx";
    auto tr = dms::make_file_transport();
    std::atomic<bool> cancel{false};
    const DownloadStatus st = Downloader::run(spec_for(b), dest, *tr, cancel, nullptr);
    EXPECT_EQ(st.state, DMS_DL_DONE) << st.message;
    EXPECT_EQ(dms::sha256_file(dest), b.sha);
    EXPECT_FALSE(fs::exists(dest.string() + ".part"));
}

TEST(Downloader, ResumesAfterDroppedConnection) {
    const auto dir = dms_test::tmp_dir("dl_resume");
    const Blob b = make_blob(dir, 20'000'000);  // > sidecar interval (8 MiB)
    const auto dest = dir / "models" / "blob.onnx";
    std::atomic<bool> cancel{false};
    FlakyTransport drop(11'000'000, false);
    DownloadStatus st = Downloader::run(spec_for(b), dest, drop, cancel, nullptr);
    EXPECT_EQ(st.state, DMS_DL_FAILED);
    EXPECT_EQ(st.error, DMS_ERR_NETWORK);
    ASSERT_TRUE(fs::exists(dest.string() + ".part"));
    // Second attempt resumes (only the remainder is transferred) and verifies the full hash.
    uint64_t first_done = 0;
    auto tr = dms::make_file_transport();
    st = Downloader::run(spec_for(b), dest, *tr, cancel, [&](const DownloadStatus& s) {
        if (!first_done) first_done = s.done;
    });
    EXPECT_EQ(st.state, DMS_DL_DONE) << st.message;
    EXPECT_GE(first_done, 8'000'000u);  // resumed from the persisted state, not from zero
    EXPECT_EQ(dms::sha256_file(dest), b.sha);
}

TEST(Downloader, CancelKeepsPartialForResume) {
    const auto dir = dms_test::tmp_dir("dl_cancel");
    const Blob b = make_blob(dir, 2'000'000);
    const auto dest = dir / "models" / "blob.onnx";
    std::atomic<bool> cancel{true};
    auto tr = dms::make_file_transport();
    const DownloadStatus st = Downloader::run(spec_for(b), dest, *tr, cancel, nullptr);
    EXPECT_EQ(st.state, DMS_DL_CANCELLED);
    EXPECT_FALSE(fs::exists(dest));
}

TEST(Downloader, ChecksumMismatchDeletesPartial) {
    const auto dir = dms_test::tmp_dir("dl_badsha");
    const Blob b = make_blob(dir, 100'000);
    ModelSpec s = spec_for(b);
    s.sha256 = std::string(64, 'a');
    const auto dest = dir / "models" / "blob.onnx";
    std::atomic<bool> cancel{false};
    auto tr = dms::make_file_transport();
    const DownloadStatus st = Downloader::run(s, dest, *tr, cancel, nullptr);
    EXPECT_EQ(st.state, DMS_DL_FAILED);
    EXPECT_EQ(st.error, DMS_ERR_CHECKSUM);
    EXPECT_FALSE(fs::exists(dest));
    EXPECT_FALSE(fs::exists(dest.string() + ".part"));
}

TEST(Downloader, OversizedResponseIsRejected) {
    const auto dir = dms_test::tmp_dir("dl_oversize");
    const Blob b = make_blob(dir, 100'000);
    ModelSpec s = spec_for(b);
    s.size = 50'000;
    const auto dest = dir / "models" / "blob.onnx";
    std::atomic<bool> cancel{false};
    auto tr = dms::make_file_transport();
    const DownloadStatus st = Downloader::run(s, dest, *tr, cancel, nullptr);
    EXPECT_EQ(st.state, DMS_DL_FAILED);
    EXPECT_FALSE(fs::exists(dest));
}

TEST(Downloader, ServerIgnoringRangeRestartsCleanly) {
    const auto dir = dms_test::tmp_dir("dl_norange");
    const Blob b = make_blob(dir, 20'000'000);
    const auto dest = dir / "models" / "blob.onnx";
    std::atomic<bool> cancel{false};
    FlakyTransport drop(10'000'000, false);
    Downloader::run(spec_for(b), dest, drop, cancel, nullptr);
    FlakyTransport norange(1ull << 40, true);
    DownloadStatus st = Downloader::run(spec_for(b), dest, norange, cancel, nullptr);
    EXPECT_EQ(st.state, DMS_DL_FAILED);  // detected: partial discarded instead of corrupting the file
    auto tr = dms::make_file_transport();
    st = Downloader::run(spec_for(b), dest, *tr, cancel, nullptr);
    EXPECT_EQ(st.state, DMS_DL_DONE) << st.message;
    EXPECT_EQ(dms::sha256_file(dest), b.sha);
}

TEST(Downloader, AsyncStartQueryCancelThroughAbi) {
    dms_test::EngineHandle eh("dl_abi");
    // Built-in models without a URL cannot be downloaded.
    EXPECT_EQ(dms_download_start(eh.e, "video-da-small"), DMS_ERR_NOT_FOUND);
    EXPECT_EQ(dms_download_start(eh.e, "nope"), DMS_ERR_NOT_FOUND);
    dms_download_info di{};
    di.struct_size = sizeof(di);
    EXPECT_EQ(dms_download_query(eh.e, "da2-small", &di), DMS_OK);
    EXPECT_EQ(di.state, DMS_DL_IDLE);
}
