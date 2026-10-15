// Flat, pointer-free per-instance sequence data. The in-memory and flattened forms are identical,
// so FLATTEN / GET_FLATTENED_SEQUENCE_DATA are plain copies and no host pointer is ever persisted.
#pragma once

#include <cstddef>
#include <cstdint>

namespace dms_plugin {

constexpr uint32_t kSeqMagic = 0x51534D44u;  // 'DMSQ' little-endian
constexpr uint32_t kSeqVersion = 1;

#pragma pack(push, 4)
struct SequenceData {
    uint32_t magic;
    uint32_t version;        // version of the writer
    uint32_t size;           // sizeof(SequenceData) of the writer
    uint32_t params_version; // DMS_PARAMS_VERSION of the writer
    uint8_t instance_guid[16];  // runtime identity; regenerated on every SETUP/RESETUP
    uint32_t flags;
    uint8_t reserved[60];    // zero; future fields go here (never shrink this struct)
};
#pragma pack(pop)
static_assert(sizeof(SequenceData) == 96, "SequenceData layout is frozen");

// Fills a fresh SequenceData with a new random GUID.
void init_sequence_data(SequenceData& s);

// Reads sequence data written by ANY plugin version (older, newer, truncated, or garbage).
// Returns true if the source was recognized (magic matched); `out` is always valid afterwards.
// Unknown/garbage input yields defaults. A new GUID is always generated.
bool upgrade_sequence_data(const void* data, size_t size, SequenceData& out);

}  // namespace dms_plugin
