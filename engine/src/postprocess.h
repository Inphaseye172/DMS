// Depth post-processing: mapping modes, slicing mask, edge refinement and edge-preserving smoothing.
// Pure functions on caller-provided buffers; NaN/Inf-safe; no engine state.
#pragma once

#include "dms/engine.h"

#include <cstdint>

namespace dms {

// Mapping: see dms_mapping_params in engine.h. out may alias depth.
void map_depth(const float* depth, int w, int h, const dms_mapping_params& p, float* out);
// Robust depth sample at (x,y): mean of finite values in a 5x5 window, coordinates clamped to the image.
float sample_depth(const float* depth, int w, int h, float x, float y) noexcept;
// Slicing mask in [0,1]: 1 inside [mn,mx], smooth falloff of width feather*(mx-mn) outside.
void slice_mask(const float* mapped, int w, int h, float mn, float mx, float feather, float* out);
// Guided filter (He et al. 2010) of depth with the image luminance as guide; strength 0..100 blends
// toward the refined result and controls radius/eps. rgb is interleaved with `stride` floats per row.
void refine_edges(const float* depth, const float* rgb, int stride, int w, int h, float strength, float* out);
// Edge-preserving smoothing (self-guided filter) that removes up-sampling stair steps; strength 0..100.
void smooth_depth(const float* depth, int w, int h, float strength, float* out);

}  // namespace dms
