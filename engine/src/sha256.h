#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace dms {

// Streaming SHA-256 (FIPS 180-4). Used to verify downloaded model files.
class Sha256 {
public:
    Sha256() noexcept { reset(); }
    void reset() noexcept;
    void update(const void* data, size_t len) noexcept;
    std::array<uint8_t, 32> finish() noexcept;
    std::string finish_hex();

    // Serializable mid-stream state so resumable downloads need not re-hash the partial file.
    struct State {
        std::array<uint32_t, 8> h;
        std::array<uint8_t, 64> buf;
        uint64_t total;
        uint32_t buf_len;
    };
    State state() const noexcept { return {h_, buf_, total_, buf_len_}; }
    void restore(const State& s) noexcept { h_ = s.h; buf_ = s.buf; total_ = s.total; buf_len_ = s.buf_len; }

private:
    void block(const uint8_t* p) noexcept;
    std::array<uint32_t, 8> h_{};
    std::array<uint8_t, 64> buf_{};
    uint64_t total_ = 0;
    uint32_t buf_len_ = 0;
};

std::string to_hex(const uint8_t* p, size_t n);
// Hashes a file; returns empty string on I/O error. abort_flag (optional) is polled between chunks.
std::string sha256_file(const std::filesystem::path& p, const volatile bool* abort_flag = nullptr);

}  // namespace dms
