// Temporary stubs for ABI functions whose implementation lands in later milestones.
// Each returns DMS_ERR_UNSUPPORTED. This file shrinks to nothing by milestone (h).
#include "engine_impl.h"

extern "C" {
#define DMS_STUB(name, ...) \
    dms_status DMS_API name(__VA_ARGS__) { return DMS_ERR_UNSUPPORTED; }

DMS_STUB(dms_map_depth, const float*, int32_t, int32_t, const dms_mapping_params*, float*)
DMS_STUB(dms_slice_mask, const float*, int32_t, int32_t, float, float, float, float*)
DMS_STUB(dms_apply_colormap, const float*, int32_t, int32_t, int32_t, float*)
DMS_STUB(dms_refine_edges, const float*, const float*, int32_t, int32_t, int32_t, float, float*)
DMS_STUB(dms_smooth_depth, const float*, int32_t, int32_t, float, float*)

int32_t DMS_API dms_colormap_count(void) { return 1; }
const char* DMS_API dms_colormap_name(int32_t index) { return index == 0 ? "None" : ""; }
}
