#include "sequence_data.h"

#include "params_model.h"

#include <chrono>
#include <cstring>
#include <random>
#include <thread>

namespace dms_plugin {

void init_sequence_data(SequenceData& s) {
    std::memset(&s, 0, sizeof(s));
    s.magic = kSeqMagic;
    s.version = kSeqVersion;
    s.size = sizeof(SequenceData);
    s.params_version = DMS_PARAMS_VERSION;
    // random_device may be deterministic on some platforms; mix in time + thread id.
    std::random_device rd;
    const uint64_t t = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const uint64_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::seed_seq seq{rd(), rd(), static_cast<uint32_t>(t), static_cast<uint32_t>(t >> 32), static_cast<uint32_t>(tid)};
    std::mt19937 gen(seq);
    for (size_t i = 0; i < 16; i += 4) {
        const auto v = static_cast<uint32_t>(gen());
        std::memcpy(s.instance_guid + i, &v, 4);
    }
    s.instance_guid[6] = static_cast<uint8_t>((s.instance_guid[6] & 0x0F) | 0x40);  // RFC 4122 v4
    s.instance_guid[8] = static_cast<uint8_t>((s.instance_guid[8] & 0x3F) | 0x80);
}

bool upgrade_sequence_data(const void* data, size_t size, SequenceData& out) {
    init_sequence_data(out);
    if (!data || size < 12) return false;
    uint32_t hdr[3];
    std::memcpy(hdr, data, sizeof(hdr));
    if (hdr[0] != kSeqMagic) return false;
    // Copy the overlapping prefix: fields we know are taken from the source, the rest stay default.
    const size_t declared = hdr[2];
    size_t n = size < declared ? size : declared;
    if (n > sizeof(SequenceData)) n = sizeof(SequenceData);
    SequenceData src;
    std::memset(&src, 0, sizeof(src));
    std::memcpy(&src, data, n);
    // Only the flags field carries persistent meaning in v1. GUID is regenerated (copies must not alias).
    if (n >= offsetof(SequenceData, flags) + sizeof(uint32_t)) out.flags = src.flags;
    return true;
}

}  // namespace dms_plugin
